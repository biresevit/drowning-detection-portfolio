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
import time
import logging

from action_recognition.model_base import ModelBase

logger = logging.getLogger()

INDEX_BATCH = 0


class SwinTensorRT(ModelBase):
    DEBUG_SAVE_FRAMES = False

    def __init__(self, model_name, engine_path, classes, shared_cuda_context=None, engine_bytes=None):
        self.model_name = model_name
        self.device = "cpu"
        self.engine_path = engine_path
        self.classes = classes
        self.class_dict = {i: cls for i, cls in enumerate(classes)}

        self.model_transform_params = self.get_model_configs()
        self.transform_params = self.model_transform_params[self.model_name]

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.frames_per_second = 30
        self.num_frames = self.transform_params["num_frames"]
        self.crop_size = self.transform_params["crop_size"]
        self.sampling_rate = self.transform_params["sampling_rate"]
        self.clip_duration = (self.num_frames * self.sampling_rate) / self.frames_per_second

        self.val_transform = self.get_validation_transform()

        # CUDA context handling
        self.device_cuda = cuda.Device(0)
        self.cuda_context = shared_cuda_context or self.device_cuda.make_context()
        self.owns_context = shared_cuda_context is None

        try:
            # Load engine
            self.engine = self.load_engine(engine_bytes)
            self.execution_context = self.engine.create_execution_context()

            # Identify input/output tensor names
            self.input_name, self.output_name = None, None
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                elif mode == trt.TensorIOMode.OUTPUT:
                    self.output_name = name
            if self.input_name is None or self.output_name is None:
                raise RuntimeError("Could not determine input/output tensor names")

            # Allocate max batch buffers
            self.max_batch_size = 12
            self.bindings, self.inputs, self.outputs = [], [], []
            self.stream = cuda.Stream()

            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                if shape[0] == -1:
                    shape = (self.max_batch_size,) + shape[1:]
                size = trt.volume(shape)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                host_mem = np.zeros(size, dtype=dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.inputs.append((host_mem, device_mem))
                else:
                    self.outputs.append((host_mem, device_mem))

            self.lock = threading.Lock()
            self.warm_up()
        finally:
            if self.owns_context:
                self.cuda_context.pop()

    def __del__(self):
        if self.owns_context and hasattr(self, "cuda_context") and self.cuda_context:
            try:
                self.cuda_context.pop()
            except Exception as e:
                logger.error(f"SwinTensorRT __del__ error: {e}")

    def get_validation_transform(self):
        return ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(self.num_frames),
                Lambda(lambda x: x / 255.0),
                ResizeAndPadVideo(size=self.crop_size),
                NormalizeVideo(self.mean, self.std),
            ]),
        )

    def load_engine(self, engine_bytes=None):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        if engine_bytes is not None:
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        else:
            if not os.path.exists(self.engine_path):
                raise FileNotFoundError(f"Engine not found: {self.engine_path}")
            with open(self.engine_path, "rb") as f:
                engine_data = f.read()
            engine = runtime.deserialize_cuda_engine(engine_data)

        if engine is None:
            raise RuntimeError(f"Failed to load engine from {self.engine_path}")
        return engine

    def preprocess_video(self, frames):
        # Normalize in FP32 to preserve precision, then cast to FP16 for the engine
        frames_tensor = torch.tensor(np.stack(frames), dtype=torch.float32).permute(3, 0, 1, 2)
        video_data = {"video": frames_tensor}
        transformed = self.val_transform(video_data)
        return transformed["video"].to(dtype=torch.float16)

    def infer(self, input_tensor):
        batch_size = input_tensor.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} > max {self.max_batch_size}")

        with self.lock:
            self.cuda_context.push()
            try:
                self.execution_context.set_input_shape(self.input_name, input_tensor.shape)
                np.copyto(self.inputs[0][0][:input_tensor.size], input_tensor.ravel())
                cuda.memcpy_htod_async(self.inputs[0][1], self.inputs[0][0], self.stream)
                self.execution_context.execute_v2(self.bindings)
                self.stream.synchronize()
                cuda.memcpy_dtoh_async(self.outputs[0][0], self.outputs[0][1], self.stream)
                self.stream.synchronize()
                out_shape = self.execution_context.get_tensor_shape(self.output_name)
                return self.outputs[0][0][:trt.volume(out_shape)].reshape(out_shape)
            finally:
                self.cuda_context.pop()

    def save_video_frames(self, frames: list, batch_index: int, clip_index: int):
        folder = os.path.join("raw_frames", f"batch_{batch_index}", f"clip_{clip_index}")
        os.makedirs(folder, exist_ok=True)
        for idx, frame in enumerate(frames):
            # no need to convert to RGB as frames are already RGB as video swin requires frames in rgb
            Image.fromarray(frame).save(os.path.join(folder, f"frame_{idx:02d}.png"))

    def save_video_tensors(self, transformed_video: torch.Tensor, batch_index: int, clip_index: int):
        folder = os.path.join("transformed_frames", f"batch_{batch_index}", f"clip_{clip_index}")
        os.makedirs(folder, exist_ok=True)
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        for idx in range(transformed_video.shape[1]):
            frame = transformed_video[:, idx, :, :].float() * std + mean
            frame_np = (frame.permute(1, 2, 0) * 255).clip(0, 255).byte().cpu().numpy()
            Image.fromarray(frame_np).save(os.path.join(folder, f"transformed_frame_{idx:02d}.png"))

    def predict_class(self, frames_batches: list, is_warm_up=False):
        global INDEX_BATCH
        start_time = time.perf_counter()

        transformed_batches = []
        for frames in frames_batches:
            video_tensor = self.preprocess_video(frames)
            transformed_batches.append(video_tensor)

        if self.DEBUG_SAVE_FRAMES and not is_warm_up:
            for clip_idx, (frames, tensor) in enumerate(zip(frames_batches, transformed_batches)):
                self.save_video_frames(frames, INDEX_BATCH, clip_idx)
                self.save_video_tensors(tensor, INDEX_BATCH, clip_idx)
            INDEX_BATCH += 1

        inputs = torch.stack(transformed_batches).cpu().numpy().astype(np.float16)
        elapsed = time.perf_counter() - start_time
        logger.info(f"Video preprocessing time: {elapsed:.3f}s")
        output = self.infer(inputs)

        results = []
        num_classes = len(self.classes)
        for i in range(output.shape[0]):
            batch_pred = output[i]
            if num_classes == 2:
                batch_pred = np.clip(batch_pred, -60, 60)
                pos_prob = 1 / (1 + np.exp(-batch_pred))
                neg_prob = 1 - pos_prob
                final_pred = np.concatenate([neg_prob, pos_prob])
            else:
                exp_pred = np.exp(batch_pred - np.max(batch_pred))
                final_pred = exp_pred / exp_pred.sum()
            idx = np.argmax(final_pred)
            results.append((self.class_dict[idx], float(final_pred[idx])))
        return results

    def warm_up(self, iterations: int = 5):
        """
        Full pipeline warm-up using predict_class().
        Simulates real OpenCV frames and batch processing.
        """

        batch_size = self.max_batch_size
        num_frames = self.num_frames

        logger.info(
            f"Starting FULL pipeline warm-up "
            f"(batch={batch_size}, frames={num_frames}, iterations={iterations})"
        )

        # --- Create dummy OpenCV frames ---
        dummy_batch = []

        for _ in range(batch_size):
            clip_frames = []

            for _ in range(num_frames):
                # Simulate random BGR OpenCV frame
                frame = np.random.randint(
                    0,
                    256,
                    (self.crop_size, self.crop_size, 3),
                    dtype=np.uint8,
                )
                clip_frames.append(frame)

            dummy_batch.append(clip_frames)

        # --- Warm-up loop ---
        times = []

        for i in range(iterations):
            start = time.perf_counter()

            _ = self.predict_class(dummy_batch, is_warm_up=True)

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            logger.info(
                f"Warm-up iteration {i+1}/{iterations}: {elapsed:.4f}s"
            )

        avg_time = sum(times) / len(times)
        logger.info(f"Warm-up average pipeline time: {avg_time:.4f}s")



    def get_num_frames(self):
        return self.transform_params["num_frames"]

    def get_sampling_rate(self):
        return self.transform_params["sampling_rate"]