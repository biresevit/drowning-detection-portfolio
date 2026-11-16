import os


def _ensure_cuda_dlls_on_path():
    """onnxruntime-gpu's CUDA provider needs cuDNN/cuBLAS DLLs on PATH (classic
    LoadLibrary search — os.add_dll_directory is not honored by onnxruntime's
    native loader). Reuses PyTorch's own bundled cuDNN/cuBLAS in torch/lib —
    NOT a separately pip-installed cudnn package, which would be a different
    version and cause DLL export mismatches (WinError 127) when torch loads.
    """
    if os.name != "nt":
        return
    try:
        import torch
    except ImportError:
        return
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib) and torch_lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")


_ensure_cuda_dlls_on_path()

import onnxruntime as ort
import cv2
import numpy as np
import time
from enum import Enum
import time


class SegmentationModelType(Enum):
    DEEPLABV3PLUS = 0
    UNET = 1


class SegmentationOnnxInfer:
    def __init__(
        self, model_path, num_cores=None, model_type=SegmentationModelType.DEEPLABV3PLUS
    ):
        self.model_path = model_path
        self.num_cores = num_cores
        self.model_type = model_type
        self.ort_session = self.load_model()
        self.input_size = self.read_input_size()
        # Get the list of execution providers
        execution_providers = self.ort_session.get_providers()

        # Check if CUDA is being used
        if "CUDAExecutionProvider" in execution_providers:
            print("ONNX Runtime is using CUDA for inference.")
        else:
            print(
                "ONNX Runtime is not using CUDA. Current providers:",
                execution_providers,
            )

        # Perform warm-up
        self.warm_up()

    def read_input_size(self):
        input_shape = self.ort_session.get_inputs()[
            0
        ].shape  # Get the shape of the input
        input_size = input_shape[2]
        print(f"read input_size of model {input_size}, shape: {input_shape}")
        return input_shape[2]

    def remove_transparency(self, img):
        if img.shape[2] == 4:
            # Set 0 transparency to black
            img[np.where(img[:, :, 3] == 0)] = [0, 0, 0, 255]

            # Remove alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def cleanup(self):
        pass

    def load_model(self):
        # List of providers in priority order
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        if not os.path.exists(self.model_path):
            raise FileExistsError(f"{self.model_path}")

        if self.num_cores is None:
            return ort.InferenceSession(self.model_path, providers=providers)
        else:
            # Set the number of threads to use
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.num_cores
            return ort.InferenceSession(
                self.model_path, sess_options=sess_options, providers=providers
            )

    # unet has different normalization than deeplabv3
    def get_input_tensor_unet(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # Convert to float
        img = np.expand_dims(img, axis=0)
        return img

    # Define the standard transforms that need to be done at inference time
    def get_input_tensor_deeplabv3(self, img):
        imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # Convert to float
        img = (img - imagenet_stats[0]) / imagenet_stats[1]  # Normalize
        img = np.transpose(img, (2, 0, 1))  # Change data from HWC to CHW format
        input_tensor = np.expand_dims(img, axis=0)  # Add batch dimension
        input_tensor = input_tensor.astype(np.float32)
        return input_tensor

    def get_pred_unet(self, input_tensor):
        # Prepare the input to the model
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor}

        # Run the model
        ort_outs = self.ort_session.run(None, ort_inputs)

        # Extract the output and return it
        output = ort_outs[0]
        output = np.squeeze(output)
        return output

    def get_pred(self, img):
        start = time.perf_counter()
        height, width, _ = img.shape
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        input_tensor = (
            self.get_input_tensor_deeplabv3(img_resized)
            if self.model_type == SegmentationModelType.DEEPLABV3PLUS
            else self.get_input_tensor_unet(img_resized)
        )
        mask = (
            self.get_pred_deeplabv3(input_tensor)
            if self.model_type == SegmentationModelType.DEEPLABV3PLUS
            else self.get_pred_unet(input_tensor)
        )
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.uint8)
        # print(f"Elapsed for {self.model_type} model: {(time.perf_counter() - start):.03f}")
        return mask

    def get_pred_deeplabv3(self, input_tensor):
        # Prepare the input to the model
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor}

        # Run the model
        ort_outs = self.ort_session.run(None, ort_inputs)

        # Extract the output and return it
        output = ort_outs[0][0].argmax(0)

        return output

    def get_binary_mask_deeplabv3(self, mask, label):
        return mask == label

    def get_binary_mask_unet(self, mask):
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return mask

    # takes input image and predicts mask using deeplabv3, resizes mask to orginal image size
    def get_prediction_mask(self, original, input_size):
        height, width, channels = original.shape
        frame = cv2.resize(original, (input_size, input_size))

        start_time = time.time()
        mask = self.get_pred(frame)
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time} seconds")

        # Resize the labels to match the frame dimensions
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        # display_bool_img(mask, "mask")
        return mask

    def warm_up(self):
        """
        Perform a warm-up inference with random input data.
        """
        try:
            # Create random input data
            random_input = np.random.rand(self.input_size, self.input_size, 3).astype(
                np.float32
            )
            random_input = cv2.cvtColor(random_input, cv2.COLOR_BGR2RGB)
            input_tensor = (
                self.get_input_tensor_deeplabv3(random_input)
                if self.model_type == SegmentationModelType.DEEPLABV3PLUS
                else self.get_input_tensor_unet(random_input)
            )

            # Perform inference
            _ = (
                self.get_pred_deeplabv3(input_tensor)
                if self.model_type == SegmentationModelType.DEEPLABV3PLUS
                else self.get_pred_unet(input_tensor)
            )
            print("Warm-up inference completed.")
        except Exception as e:
            print(f"Exception during warm-up: {e}")
