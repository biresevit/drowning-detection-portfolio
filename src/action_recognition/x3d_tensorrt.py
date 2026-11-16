import os
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
from torchvision.transforms._transforms_video import NormalizeVideo
from action_recognition.custom_transforms import ResizeAndPadVideo
import tensorrt as trt
import pycuda.driver as cuda
import threading
from action_recognition.model_base import ModelBase
import time
import logging
# Debug: dump the crop frames handed to the model. Off in normal runs - it writes
# a PNG per frame per cluster and slows inference.
SAVE_X3D_CROPS = os.getenv("SAVE_X3D_CROPS", "False").lower() == "true"

logger = logging.getLogger()


class X3dTensorRT(ModelBase):
    def __init__(self, model_name, engine_path, classes, shared_cuda_context=None, engine_bytes=None):
        """
        Initialize the X3dTensorRT model with a shared CUDA context if provided.

        Args:
            model_name (str): Name of the X3D model (e.g., "x3d_s").
            engine_path (str): Path to the TensorRT engine file.
            classes (list): List of class names (e.g., ["normal", "drowning"]).
            shared_cuda_context (pycuda.driver.Context, optional): Shared CUDA context from the app.
        """
        self.model_name = model_name
        self.device = "cpu"
        self.engine_path = engine_path
        self.classes = classes
        self.class_dict = {i: cls for i, cls in enumerate(classes)}

        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.model_transform_params = self.get_model_configs()
        self.transform_params = self.model_transform_params[self.model_name]
        self.frames_per_second = 30
        self.num_frames = self.transform_params["num_frames"]
        self.crop_size = self.transform_params["crop_size"]
        self.sampling_rate = self.transform_params["sampling_rate"]
        self.clip_duration = (
            self.num_frames * self.sampling_rate
        ) / self.frames_per_second
        self.val_transform = self.get_validation_transform()

        # Use shared context if provided, otherwise create one
        self.device_cuda = cuda.Device(0)  # Define device_cuda for logging
        self.cuda_context = (
            shared_cuda_context
            if shared_cuda_context
            else self.device_cuda.make_context()
        )
        self.owns_context = shared_cuda_context is None  # Track ownership for cleanup

        try:
            # Load engine and create context
            self.engine = self.load_model()
            self.execution_context = self.engine.create_execution_context()

            # Determine IO tensor names
            self.input_name = None
            self.output_name = None
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(tensor_name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = tensor_name
                elif mode == trt.TensorIOMode.OUTPUT:
                    self.output_name = tensor_name
            if self.input_name is None or self.output_name is None:
                raise RuntimeError("Failed to determine input/output tensor names.")

            # Allocate buffers for max batch size
            self.max_batch_size = 12  # Matches your trtexec --maxShapes
            self.bindings = []
            self.inputs = []
            self.outputs = []
            self.stream = cuda.Stream()
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(tensor_name)
                if shape[0] == -1:  # Dynamic shape
                    shape = (self.max_batch_size,) + shape[1:]
                size = trt.volume(shape)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                host_mem = np.zeros(size, dtype=dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    self.inputs.append((host_mem, device_mem))
                else:
                    self.outputs.append((host_mem, device_mem))

            self.lock = threading.Lock()
            logger.info(
                f"TensorRT is using GPU: {self.device_cuda.name()}, "
                f"Compute Capability: {self.device_cuda.compute_capability()[0]}.{self.device_cuda.compute_capability()[1]}"
            )
            self.warm_up()
        finally:
            if self.owns_context and self.cuda_context:
                self.cuda_context.pop()

    def __del__(self):
        """Clean up CUDA context only if we own it."""
        if self.owns_context and hasattr(self, "cuda_context") and self.cuda_context:
            try:
                self.cuda_context.pop()
                self.cuda_context = None
            except Exception as e:
                logger.error(f"Error in X3dTensorRT __del__: {e}")

    def cleanup(self):
        """Explicitly clean up CUDA context if owned."""
        if self.owns_context and hasattr(self, "cuda_context") and self.cuda_context:
            try:
                self.cuda_context.pop()
                self.cuda_context = None
                logger.info("CUDA context cleaned up for X3dTensorRT.")
            except Exception as e:
                logger.error(f"Error during explicit CUDA context cleanup: {e}")

    def get_validation_transform(self):
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    # UniformTemporalSubsample(self.transform_params["num_frames"]),
                    Lambda(lambda x: x / 255.0),
                    ResizeAndPadVideo(size=self.transform_params["crop_size"]),
                    NormalizeVideo(self.mean, self.std),
                ]
            ),
        )

    def load_model(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
    
        if hasattr(self, "engine_bytes") and self.engine_bytes is not None:
            # deserialize directly from bytes
            engine = runtime.deserialize_cuda_engine(self.engine_bytes)
        else:
            # fallback na fajl
            if not os.path.exists(self.engine_path):
                raise FileNotFoundError(f"Engine file not found at {self.engine_path}")
            with open(self.engine_path, "rb") as f:
                engine_data = f.read()
            engine = runtime.deserialize_cuda_engine(engine_data)
    
        if engine is None:
            raise RuntimeError(
                f"Failed to load TensorRT engine from bytes or {self.engine_path}"
            )
        return engine


    def preprocess_video(self, frames):
        frames_tensor = torch.tensor(
            np.stack(frames), dtype=torch.float16
        ).permute(3, 0, 1, 2)

        video_data = {"video": frames_tensor}
        transformed_data = self.val_transform(video_data)

        return transformed_data["video"].to(dtype=torch.float16)


    def infer(self, input_tensor):
        current_batch = input_tensor.shape[0]
        if current_batch > self.max_batch_size:
            raise ValueError(
                f"Batch size {current_batch} exceeds engine max batch size {self.max_batch_size}. "
                f"Rebuild engine with --maxShapes=input:{current_batch}x3x{self.num_frames}x{self.crop_size}x{self.crop_size}"
            )

        with self.lock:
            self.cuda_context.push()
            try:
                self.execution_context.set_input_shape(
                    self.input_name, input_tensor.shape
                )
                input_size = trt.volume(input_tensor.shape)
                np.copyto(self.inputs[0][0][:input_size], input_tensor.ravel())
                cuda.memcpy_htod_async(
                    self.inputs[0][1], self.inputs[0][0], self.stream
                )
                logger.debug(
                    f"Input tensor shape: {input_tensor.shape}, min: {input_tensor.min()}, max: {input_tensor.max()}"
                )
                self.execution_context.execute_v2(self.bindings)
                self.stream.synchronize()
                cuda.memcpy_dtoh_async(
                    self.outputs[0][0], self.outputs[0][1], self.stream
                )
                self.stream.synchronize()
                output_shape = self.execution_context.get_tensor_shape(self.output_name)
                output_size = trt.volume(output_shape)
                output = self.outputs[0][0][:output_size].reshape(output_shape)
                logger.debug(
                    f"Output shape: {output.shape}, min: {output.min()}, max: {output.max()}"
                )
                return output
            except Exception as e:
                logger.error(f"Exception in infer: {e}")
                raise
            finally:
                self.cuda_context.pop()

    def predict_class(self, frames_batches: list):
        transformed_batches = []
        start_infer = time.perf_counter()
        for frames in frames_batches:
            if SAVE_X3D_CROPS:
                self.save_video_frames(frames)
            frames_tensor = torch.tensor(
                np.stack(frames), dtype=torch.float16
            ).permute(3, 0, 1, 2)
            video_data = {"video": frames_tensor}
            transformed_data = self.val_transform(video_data)
            #self.save_video_tensors(transformed_data["video"])
            transformed_batches.append(transformed_data["video"])

        inputs_batch = torch.stack(transformed_batches).to(self.device)
        inputs_np = inputs_batch.cpu().numpy().astype(np.float16)
        logger.info(
            f"Elapsed time for preprocessing: {time.perf_counter() - start_infer:.3f}s"
        )
        output = self.infer(inputs_np)

        num_classes = len(self.classes)
        results = []
        for i in range(output.shape[0]):
            batch_pred = output[i]
            if num_classes == 2:
                # Clip batch_pred to avoid overflow in exp
                batch_pred = np.clip(
                    batch_pred, -60, 60
                )  # Reasonable range for stability
                pos_prob = 1 / (1 + np.exp(-batch_pred))  # Sigmoid
                neg_prob = 1 - pos_prob
                final_pred = np.concatenate([neg_prob, pos_prob], axis=0)
            else:
                exp_pred = np.exp(batch_pred - np.max(batch_pred))
                final_pred = exp_pred / exp_pred.sum()  # Softmax
            top_index = np.argmax(final_pred)
            top_confidence = float(final_pred[top_index])
            top_class_name = self.class_dict[top_index]
            results.append((top_class_name, top_confidence))
        return results

    def _generate_dummy_frames(self):
        frames = []
        for _ in range(self.num_frames):
            frame = np.random.randint(
                0, 256,
                (self.crop_size, self.crop_size, 3),
                dtype=np.uint8
            )
            frames.append(frame)
        return frames

    def warm_up(self):
        try:
            batch_size = self.max_batch_size
            warmup_iters = 5

            frames_batches = [
                self._generate_dummy_frames()
                for _ in range(batch_size)
            ]

            logger.info(
                f"Starting X3D warm-up: "
                f"{warmup_iters} iterations, batch size {batch_size}"
            )

            # The warm-up feeds random pixels: only the tensor shape matters for
            # allocating buffers and letting TensorRT pick kernels. Flagged so the crop
            # dump skips them - otherwise every run starts with iterations x batch_size
            # folders of noise (60 by default) before the first real frame is read.
            self._warming_up = True
            try:
                for i in range(warmup_iters):
                    start = time.perf_counter()
                    _ = self.predict_class(frames_batches)
                    elapsed = time.perf_counter() - start
                    logger.info(
                        f"X3D warm-up iteration {i+1}/{warmup_iters}: "
                        f"{elapsed:.3f}s"
                    )
            finally:
                self._warming_up = False

            logger.info("X3D warm-up complete.")

        except Exception as e:
            logger.error(f"Exception during X3D warm-up: {e}")


    def save_video_frames(self, frames: list):
        """Dump the raw crop frames fed to the model, one folder per cluster batch.

        Debug only, enabled with SAVE_X3D_CROPS=True. It writes one PNG per frame per
        cluster from inside the timed preprocessing block, so a full evaluation run
        would produce tens of thousands of files and report inflated preprocessing times.

        Debugging aid: shows what the action model actually sees, before resizing and
        letterboxing, so a cluster that crops off a drowning person's arms is visible.
        Each batch gets its own folder named with the crop size, since every cluster is
        framed differently - writing them all to one folder would overwrite the same
        16 files on every inference and leave only the last cluster of the last window.

        The crops arrive as RGB, not BGR: processing_thread converts the frame once
        before buffering it, because the action model wants RGB. cv2.imwrite writes BGR,
        so the channels are swapped back here - without that the pool comes out orange.
        The dump is the only place that needs this; the model itself gets RGB as intended.

        Written with cv2 rather than PIL because PIL silently turns an unexpected dtype
        into noise instead of raising, hiding the problem. The line logged per batch
        states dtype and range, so a non-uint8 input is visible rather than mysterious.
        """
        if not frames or getattr(self, "_warming_up", False):
            return
        first = frames[0]
        h, w = first.shape[:2]
        self._frame_dump_id = getattr(self, "_frame_dump_id", 0) + 1
        out_dir = os.path.join("raw_frames", f"{self._frame_dump_id:04d}_{w}x{h}")
        os.makedirs(out_dir, exist_ok=True)
        logger.info(
            "Crop dump %s: %d frames shape=%s dtype=%s range=[%s..%s]",
            out_dir, len(frames), first.shape, first.dtype,
            np.min(first), np.max(first),
        )
        for idx, frame in enumerate(frames):
            img = np.asarray(frame)
            if img.ndim == 3 and img.shape[2] == 4:      # drop alpha if one is present
                img = img[:, :, :3]
            if img.dtype != np.uint8:
                img = np.nan_to_num(img.astype(np.float32))
                if img.max() <= 1.5:                     # normalised 0-1 rather than 0-255
                    img = img * 255.0
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[2] == 3:
                img = img[:, :, ::-1]                    # RGB in, BGR out for cv2
            cv2.imwrite(os.path.join(out_dir, f"frame_{idx:02d}.png"),
                        np.ascontiguousarray(img))

    def save_video_tensors(self, transformed_video: torch.Tensor):
        os.makedirs("transformed_frames", exist_ok=True)
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        for idx in range(transformed_video.shape[1]):
            frame = transformed_video[:, idx, :, :]
            frame = frame * std + mean
            frame = frame.permute(1, 2, 0)
            frame_np = (frame * 255).clip(0, 255).byte().cpu().numpy()
            frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            frame_path = os.path.join(
                "transformed_frames", f"transformed_frame_{idx:02d}.png"
            )
            Image.fromarray(frame_rgb).save(frame_path)

    def get_num_frames(self):
        return self.transform_params["num_frames"]

    def get_sampling_rate(self):
        return self.transform_params["sampling_rate"]
