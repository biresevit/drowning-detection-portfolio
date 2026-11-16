"""
inference_classification_tensorrt.py

TensorRT inference for the head-visibility image classifier.

Input:  list of BGR numpy crops (any size — they are resized internally)
Output: list of class-name strings  OR  (N, num_classes) probability array

Class order matches alphabetical ImageFolder sorting (same as training).
See head_visibility_classes.INFERENCE_CLASS_NAMES for the authoritative list.

Build the engine on Jetson first:
    /usr/src/tensorrt/bin/trtexec \
        --onnx=models/head_visibility.onnx \
        --saveEngine=models/head_visibility.engine \
        --fp16 \
        --minShapes=input:1x3x224x224 \
        --maxShapes=input:32x3x224x224
"""

import logging
import threading

import cv2
import numpy as np

try:
    from human_tracking.head_visibility_classes import INFERENCE_CLASS_NAMES
except ImportError:
    from head_visibility_classes import INFERENCE_CLASS_NAMES

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES: list[str] = INFERENCE_CLASS_NAMES


def _preprocess(crops_bgr: list[np.ndarray], img_size: int) -> np.ndarray:
    """Return (N, 3, img_size, img_size) float32 array, ImageNet-normalised."""
    batch = []
    for crop in crops_bgr:
        img = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        batch.append(img.transpose(2, 0, 1))       # HWC → CHW
    return np.stack(batch, axis=0).astype(np.float32)


class HeadVisibilityTRT:
    """Runs the head-visibility classifier via TensorRT."""

    def __init__(self, engine_path: str, max_batch_size: int = 32, img_size: int = 224, shared_cuda_context=None):
        import tensorrt as trt
        import pycuda.driver as cuda

        self._trt  = trt
        self._cuda = cuda

        self.engine_path    = engine_path
        self.max_batch_size = max_batch_size
        self.img_size       = img_size
        self.lock           = threading.Lock()

        logger.info("HeadVisibilityTRT init: engine=%s  max_batch=%d  img_size=%d",
                    engine_path, max_batch_size, img_size)

        # cuda.init() must be called once per process before any other CUDA call.
        # It is idempotent — safe to call even if a shared context already exists.
        try:
            cuda.init()
            logger.info("CUDA initialised.")
        except Exception as e:
            logger.exception("cuda.init() failed: %s", e)
            raise

        try:
            self.device_cuda  = cuda.Device(0)
            self.cuda_context = shared_cuda_context or self.device_cuda.make_context()
            self.owns_context = shared_cuda_context is None
            logger.info("CUDA context ready (owns=%s).", self.owns_context)
        except Exception as e:
            logger.exception("CUDA context creation failed: %s", e)
            raise

        try:
            self._engine = self._load_engine()
            logger.info("TRT engine deserialised.")
            self._ctx = self._engine.create_execution_context()

            self.input_name = self.output_name = None
            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                mode = self._engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                elif mode == trt.TensorIOMode.OUTPUT:
                    self.output_name = name

            if self.input_name is None or self.output_name is None:
                raise RuntimeError("Could not determine input/output tensor names from engine")

            logger.info("Tensors: input=%s  output=%s", self.input_name, self.output_name)
            self._stream = cuda.Stream()
            self._alloc_buffers()
            logger.info("Buffers allocated.")
            self._warm_up()
            logger.info("Warm-up done.")
        except Exception as e:
            logger.exception("Engine setup failed: %s", e)
            raise
        finally:
            if self.owns_context:
                self.cuda_context.pop()

        logger.info("HeadVisibilityTRT ready: %s  max_batch=%d  img_size=%d",
                    engine_path, max_batch_size, img_size)

    # ---------------------------------------------------------------- setup

    def _load_engine(self):
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime    = trt.Runtime(TRT_LOGGER)
        with open(self.engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialise TRT engine: {self.engine_path}")
        return engine

    def _alloc_buffers(self):
        import tensorrt as trt
        import pycuda.driver as cuda

        self._bindings = []
        self._h_inputs, self._d_inputs   = [], []
        self._h_outputs, self._d_outputs = [], []

        for i in range(self._engine.num_io_tensors):
            name  = self._engine.get_tensor_name(i)
            shape = self._engine.get_tensor_shape(name)
            if shape[0] == -1:
                shape = (self.max_batch_size,) + tuple(shape[1:])
            size  = trt.volume(shape)
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            h_mem = np.zeros(size, dtype=dtype)
            d_mem = cuda.mem_alloc(h_mem.nbytes)
            self._bindings.append(int(d_mem))
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._h_inputs.append(h_mem)
                self._d_inputs.append(d_mem)
            else:
                self._h_outputs.append(h_mem)
                self._d_outputs.append(d_mem)

    def _warm_up(self):
        dummy = np.zeros((1, 3, self.img_size, self.img_size), dtype=np.float32)
        self._run(dummy)

    # ---------------------------------------------------------------- inference

    def _run(self, batch_np: np.ndarray) -> np.ndarray:
        """Run TRT inference; returns raw logits (N, num_classes)."""
        import pycuda.driver as cuda
        import tensorrt as trt

        n = batch_np.shape[0]
        with self.lock:
            self.cuda_context.push()
            try:
                self._ctx.set_input_shape(self.input_name, batch_np.shape)
                np.copyto(self._h_inputs[0][:batch_np.size], batch_np.ravel())
                cuda.memcpy_htod_async(self._d_inputs[0], self._h_inputs[0], self._stream)
                self._ctx.execute_v2(self._bindings)
                self._stream.synchronize()
                cuda.memcpy_dtoh_async(self._h_outputs[0], self._d_outputs[0], self._stream)
                self._stream.synchronize()
                out_shape = self._ctx.get_tensor_shape(self.output_name)
                return self._h_outputs[0][: trt.volume(out_shape)].reshape(out_shape)
            finally:
                self.cuda_context.pop()

    def predict_probs(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Return (N, num_classes) softmax probabilities."""
        if not crops_bgr:
            return np.empty((0, len(CLASS_NAMES)), dtype=np.float32)
        batch = _preprocess(crops_bgr, self.img_size)
        n = batch.shape[0]
        if n <= self.max_batch_size:
            logits = self._run(batch).astype(np.float32)
        else:
            chunks = [
                self._run(batch[i : i + self.max_batch_size]).astype(np.float32)
                for i in range(0, n, self.max_batch_size)
            ]
            logits = np.concatenate(chunks, axis=0)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, crops_bgr: list[np.ndarray]) -> list[str]:
        """Return list of predicted class names."""
        probs   = self.predict_probs(crops_bgr)
        indices = probs.argmax(axis=1)
        return [CLASS_NAMES[i] for i in indices]

    def __del__(self):
        if self.owns_context and hasattr(self, "cuda_context") and self.cuda_context:
            try:
                self.cuda_context.pop()
            except Exception:
                pass
