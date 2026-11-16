import io
import os

import torch


def _ensure_cuda_dlls_on_path():
    """onnxruntime-gpu's CUDA provider needs cuDNN/cuBLAS DLLs on PATH (classic
    LoadLibrary search — os.add_dll_directory is not honored by onnxruntime's
    native loader). Reuses PyTorch's own bundled cuDNN/cuBLAS in torch/lib —
    NOT a separately pip-installed cudnn package, which would be a different
    version and cause DLL export mismatches (WinError 127) when torch loads.
    """
    if os.name != "nt":
        return
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib) and torch_lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")


_ensure_cuda_dlls_on_path()

from action_recognition.base_model_loader import BaseModelLoader
from action_recognition.x3d import X3d
from action_recognition.mvit_v2 import MViTv2
from action_recognition.video_swin import VideoSwin
from human_tracking.tracking_humans_onnx import TrackingHumansOnnx
import logging
import onnxruntime as ort

logger = logging.getLogger()


class GenericModelLoader(BaseModelLoader):
    def load_action_model(self, model_variant, classes, model_extension=".onnx", enc=False):
        """
        Load an X3d action model. Supports encrypted models in memory.
        """
        if enc:
            enc_key = bytes.fromhex("fb7a601ec8a96d0fe0d9dfd617b10f2ccb8b81103ae6c89b44875a147834ed4b")

            enc_file_path = self.get_model_path(f"drowning_{model_variant}", model_extension + ".enc")
            logger.info(f"Decrypting X3d model from {enc_file_path}...")
            decrypted_bytes = self.decrypt_model_file(enc_file_path, enc_key)
            
            # Load model from bytes directly
            buffer = io.BytesIO(decrypted_bytes)
            checkpoint = torch.load(buffer, map_location="cuda")
            logger.info("Decrypted checkpoint loaded into memory.")
        else:
            action_model_path = self.get_model_path(f"drowning_{model_variant}", model_extension)
            logger.info(f"Loading action recognition model from {action_model_path}...")

        if "x3d" in model_variant:
            action_model = X3d(
                model_name=model_variant,
                device="cuda",
                checkpoint_path=None if enc else action_model_path,  # X3d mora podržati checkpoint dict
                classes=classes,
                checkpoint_dict=checkpoint if enc else None
            )
        elif "swin_t" in model_variant:
            action_model = VideoSwin(
                model_name=model_variant,
                device="cuda",
                checkpoint_path=None if enc else action_model_path,
                classes=classes,
                checkpoint_dict=checkpoint if enc else None
            )
        elif "mvit_v2" in model_variant:
            action_model = MViTv2(
                model_name=model_variant,
                device="cuda",
                checkpoint_path=None if enc else action_model_path,
                classes=classes,
                checkpoint_dict=checkpoint if enc else None
            )
        logger.info(f"Action model {model_variant} loaded successfully.")
        return action_model

    def load_human_tracker(
        self,
        imgsz,
        max_clusters,
        skip_water_process_frames,
        max_water_deque_size,
        deque_length=12,
        detect_water=True,
        head_classifier_path=None,
        use_behavior_analysis: bool | None = None,
    ):
        """
        Load the human tracker model.
        The model extension can be .onnx, .pth, or any supported format.

        Args:
            imgsz (int): Image size for detection.
            max_clusters (int): Maximum number of clusters.
            skip_water_process_frames (int): Number of frames to skip for water mask updates.
            max_water_deque_size (int): Maximum size of the water mask deque.
            deque_length (int): Length of the tracking deque for ObjectTracker (default: 12).
            detect_water (bool): Whether to enable water detection (default: True).
            head_classifier_path (str | None): Path to HeadVisibility model (ignored for ONNX loader).

        Returns:
            TrackingHumansOnnx: Initialized human tracker.
        """
        try:
            model_extension = ".onnx"
            human_model_path = self.get_model_path("humans", model_extension)
            water_model_path = self.get_model_path("water_640", model_extension)

            human_tracker = TrackingHumansOnnx(
                human_model_path=human_model_path,
                water_model_path=water_model_path,
                threshold_frames=1,
                tracking_deque_length=deque_length,
                num_total_crops=max_clusters,
                imgsz=imgsz,
                skip_water_process_frames=skip_water_process_frames,
                max_water_deque_size=max_water_deque_size,
                detect_water=detect_water,
                use_behavior_analysis=use_behavior_analysis,
            )

            logger.info("Human tracker loaded successfully.")
            return human_tracker
        except Exception as e:
            logger.exception(f"Failed to load human tracker: {e}")
            return None
