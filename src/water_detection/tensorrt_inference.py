import tensorrt as trt
import cv2
import numpy as np
import os
import time
from enum import Enum
import pycuda.driver as cuda
import threading
import logging

logger = logging.getLogger()


class SegmentationModelType(Enum):
    DEEPLABV3PLUS = 0
    UNET = 1


class SegmentationTrtInfer:
    def __init__(
        self,
        model_path,
        model_type=SegmentationModelType.DEEPLABV3PLUS,
        input_size=640,
        shared_cuda_context=None,
    ):
        """
        Initialize the SegmentationTrtInfer model with a shared CUDA context if provided.

        Args:
            model_path (str): Path to the TensorRT engine file.
            model_type (SegmentationModelType): Type of segmentation model (DEEPLABV3PLUS or UNET).
            input_size (int): Input size for the model (default: 640).
            shared_cuda_context (pycuda.driver.Context, optional): Shared CUDA context from the app.
        """
        self.model_path = model_path
        self.model_type = model_type
        self.input_size = input_size
        self.batch_size = 1

        # Use shared context if provided, otherwise create one
        self.device = cuda.Device(0)  # Define device for logging
        self.cuda_context = (
            shared_cuda_context if shared_cuda_context else self.device.make_context()
        )
        self.owns_context = shared_cuda_context is None  # Track ownership for cleanup

        try:
            # Load the engine
            self.engine = self.load_model()
            self.execution_context = self.engine.create_execution_context()

            # Determine input/output tensor names
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

            # Allocate buffers once
            self.bindings = []
            self.inputs = []
            self.outputs = []
            self.stream = cuda.Stream()
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(tensor_name)
                size = trt.volume(shape) * self.batch_size
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                host_mem = np.zeros(size, dtype=dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))
                mode = self.engine.get_tensor_mode(tensor_name)
                if mode == trt.TensorIOMode.INPUT:
                    self.inputs.append((host_mem, device_mem))
                elif mode == trt.TensorIOMode.OUTPUT:
                    self.outputs.append((host_mem, device_mem))

            # Thread lock for safe multi-threaded access
            self.lock = threading.Lock()

            # Log GPU info
            device_properties = self.device.get_attributes()
            logger.info(
                f"TensorRT is using GPU: {self.device.name()}, Compute Capability: "
                f"{device_properties[cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR]}."
                f"{device_properties[cuda.device_attribute.COMPUTE_CAPABILITY_MINOR]}"
            )

            # Warm-up
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
                logger.error(f"Error in SegmentationTrtInfer __del__: {e}")

    def cleanup(self):
        """Explicitly clean up CUDA context if owned."""
        if self.owns_context and hasattr(self, "cuda_context") and self.cuda_context:
            try:
                self.cuda_context.pop()
                self.cuda_context = None
                logger.info("CUDA context cleaned up for SegmentationTrtInfer.")
            except Exception as e:
                logger.error(f"Error during explicit CUDA context cleanup: {e}")

    def load_model(self):
        with open(self.model_path, "rb") as f:
            engine_data = f.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {self.model_path}")
        return engine

    def preprocess_input(self, img):
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        if self.model_type == SegmentationModelType.DEEPLABV3PLUS:
            imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
            img = (
                cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            img = (img - imagenet_stats[0]) / imagenet_stats[1]
            return np.expand_dims(img.transpose(2, 0, 1), axis=0)
        else:
            img = (
                cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            return np.expand_dims(img.transpose(2, 0, 1), axis=0)

    def infer(self, input_tensor):
        with self.lock:
            self.cuda_context.push()
            try:
                np.copyto(self.inputs[0][0], input_tensor.ravel())
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
                output_shape = self.engine.get_tensor_shape(self.output_name)
                if output_shape[0] == -1:
                    output_shape = (self.batch_size, *output_shape[1:])
                output = self.outputs[0][0].reshape(output_shape)
                logger.debug(
                    f"Output shape: {output.shape}, min: {output.min()}, max: {output.max()}"
                )
                return output
            except Exception as e:
                logger.error(f"Exception in infer: {e}")
                raise
            finally:
                self.cuda_context.pop()

    def get_pred(self, img):
        try:
            start = time.perf_counter()
            height, width, _ = img.shape
            input_tensor = self.preprocess_input(img)
            output = self.infer(input_tensor)

            if self.model_type == SegmentationModelType.DEEPLABV3PLUS:
                mask = output[0].argmax(0)
            else:
                mask = np.squeeze(output)
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0

            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(np.uint8)
            logger.debug(
                f"Elapsed for {self.model_type} model: {(time.perf_counter() - start):.03f}"
            )
            return mask
        except Exception as ex:
            logger.error(f"Exception in get_pred: {ex}")
            return None

    def get_prediction_mask(self, original, input_size=None):
        if input_size is None:
            input_size = self.input_size
        height, width, _ = original.shape
        frame = cv2.resize(original, (input_size, input_size))

        start_time = time.time()
        mask = self.get_pred(frame)
        if mask is None:
            return None
        end_time = time.time()
        logger.debug(f"Inference time: {end_time - start_time:.03f} seconds")
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    def warm_up(self):
        try:
            random_input = np.random.rand(self.input_size, self.input_size, 3).astype(
                np.float32
            )
            input_tensor = self.preprocess_input(random_input)
            _ = self.infer(input_tensor)
            logger.info("Warm-up inference completed.")
        except Exception as e:
            logger.error(f"Exception during warm-up: {e}")


def main(
    input_dir,
    output_dir,
    model_path,
    model_type=SegmentationModelType.DEEPLABV3PLUS,
    input_size=640,
):
    os.makedirs(output_dir, exist_ok=True)

    try:
        seg_model = SegmentationTrtInfer(
            model_path, model_type=model_type, input_size=input_size
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(valid_extensions):
            continue

        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to load {filename}")
            continue

        logger.info(f"Processing {filename}")
        mask = seg_model.get_prediction_mask(img)
        if mask is None:
            logger.warning(f"Failed to generate mask for {filename}")
            continue

        mask_output = (
            (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        )
        output_filename = os.path.splitext(filename)[0] + "_mask.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, mask_output)
        logger.info(f"Saved mask to {output_path}")


if __name__ == "__main__":
    logger.info("Starting segmentation process")
    INPUT_DIR = r"path/to/data"
    OUTPUT_DIR = r"path/to/data"
    MODEL_PATH = "models/water_640.engine"

    main(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        model_path=MODEL_PATH,
        model_type=SegmentationModelType.DEEPLABV3PLUS,
        input_size=640,
    )
