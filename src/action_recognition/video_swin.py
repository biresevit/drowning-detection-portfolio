from multiprocessing import dummy
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import cv2
from PIL import Image

from torchvision.models.video import (
    swin3d_t,
    Swin3D_T_Weights,
)
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
)
from torchvision.transforms._transforms_video import NormalizeVideo

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from action_recognition.custom_transforms import ResizeAndPadVideo
from action_recognition.model_base import ModelBase

logger = logging.getLogger()


class VideoSwin(ModelBase):
    """
    Video Swin Transformer (aka video twin transformer)
    """

    def __init__(self, model_name, device, checkpoint_path, classes, checkpoint_dict=None):
        self.checkpoint_dict = checkpoint_dict
        super().__init__(model_name, device, checkpoint_path, classes)

        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

        self.model_transform_params = self.get_model_configs()
        self.transform_params = self.model_transform_params.get(
            self.model_name
        )

        self.val_transform = self.get_validation_transform()

        self.clip_duration = (
            self.transform_params["num_frames"]
            * self.transform_params["sampling_rate"]
        ) / self.frames_per_second

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self):
        """
        Loads pretrained Video Swin Transformer and adapts the classification head.
        """
        weights = Swin3D_T_Weights.KINETICS400_V1
        self.model = swin3d_t(weights=weights)

        num_classes = len(self.class_dict)
        num_outputs = 1 if num_classes <= 2 else num_classes

        # Replace classifier
        self.model.head = nn.Linear(
            self.model.head.in_features,
            num_outputs,
            bias=True,
        )

        if num_classes == 2:
            self.activation_fn = nn.Sigmoid()
            logger.info("Binary classification: 1 output + sigmoid")
        else:
            self.activation_fn = nn.Softmax(dim=1)
            logger.info(f"Multi-class classification: {num_classes} outputs + softmax")

        self.model = self.model.to(self.device)
        self.model.eval()

    def load_model_disk(self, checkpoint_path, use_fp16=False):
        """
        Loads Video Swin Transformer from a local checkpoint.
        """
        self.model = swin3d_t(weights=None)

        num_classes = len(self.class_dict)
        num_outputs = 1 if num_classes <= 2 else num_classes

        self.model.head = nn.Linear(
            self.model.head.in_features,
            num_outputs,
            bias=True,
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(state_dict, strict=False)

        self.activation_fn = (
            nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1)
        )

        self.model = self.model.to(self.device)

        if use_fp16:
            self.model.half()

        self.model.eval()

    # ------------------------------------------------------------------
    # Validation transforms
    # ------------------------------------------------------------------
    def get_validation_transform(self):
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.transform_params["num_frames"]),
                    Lambda(lambda x: x / 255.0),
                    ResizeAndPadVideo(size=self.transform_params["crop_size"]),
                    NormalizeVideo(self.mean, self.std),
                ]
            ),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_class(self, frames_batches: list):
        results = []
        try:
            transformed_batches = []

            for frames in frames_batches:
                frames_np = np.stack(frames).astype(np.float32)  # [T, H, W, C]
                frames_tensor = torch.from_numpy(frames_np)
                frames_tensor = frames_tensor.permute(3, 0, 1, 2).contiguous()

                video_data = {"video": frames_tensor}
                transformed = self.val_transform(video_data)["video"]
                transformed_batches.append(transformed.to(torch.float32))

            if not transformed_batches:
                return []

            inputs = torch.stack(transformed_batches)
            inputs = inputs.to(self.device)
            inputs = inputs.to(dtype=next(self.model.parameters()).dtype)

            with torch.no_grad():
                preds_gpu = self.model(inputs)

            preds = preds_gpu.detach().cpu()
            del preds_gpu, inputs
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            num_classes = len(self.class_dict)
            idx_to_class = {k: v for k, v in self.class_dict.items()}

            for p in preds:
                if num_classes == 2:
                    pos = torch.sigmoid(p)
                    final = torch.cat([1 - pos, pos], dim=0)
                else:
                    final = torch.softmax(p, dim=0)

                score, idx = final.max(dim=0)
                results.append((idx_to_class[int(idx)], float(score)))

            return results

        except Exception as e:
            logger.error(f"VideoSwin inference error: {e}")
            torch.cuda.empty_cache()
            return []

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_to_onnx(self, output_path="models/video_swin.onnx", batch_size=-1):
        if not hasattr(self, "model") or self.model is None:
            self.load_model_disk(self.checkpoint_path)
        
        self.model = self.model.float()
        self.model.eval()

        num_frames = self.transform_params["num_frames"]
        crop = self.transform_params["crop_size"]

        # Ensure input dims divisible by patch size (T,H,W)
        patch_size = (2, 4, 4)
        if num_frames % patch_size[0] != 0:
            raise ValueError(f"num_frames {num_frames} not divisible by patch_size {patch_size[0]}")
        if crop % patch_size[1] != 0 or crop % patch_size[2] != 0:
            raise ValueError(f"crop {crop} not divisible by patch_size {patch_size[1]} or {patch_size[2]}")

        # Set fixed input shape (except batch if batch_size < 0)
        input_shape = (
            1 if batch_size < 0 else batch_size,
            3,
            num_frames,
            crop,
            crop,
        )
        dummy_input = torch.randn(*input_shape, device=self.device)

        # Export with safer options for TensorRT
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=15,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            do_constant_folding=True,
        )

        logger.info(f"Video Swin exported to ONNX at {output_path}")



    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def get_num_frames(self):
        return self.transform_params["num_frames"]

    def get_sampling_rate(self):
        return self.transform_params["sampling_rate"]
