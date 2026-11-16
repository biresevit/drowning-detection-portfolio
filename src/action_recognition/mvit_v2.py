import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import logging

from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from action_recognition.custom_transforms import ResizeAndPadVideo
from torchvision.transforms._transforms_video import NormalizeVideo
from action_recognition.model_base import ModelBase

logger = logging.getLogger()


class MViTv2(ModelBase):
    def __init__(self, model_name, device, checkpoint_path, classes, checkpoint_dict=None):
        self.checkpoint_dict = checkpoint_dict

        super().__init__(model_name, device, checkpoint_path, classes)
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.model_transform_params = self.get_model_configs()
        self.transform_params = self.model_transform_params.get(
            self.model_name
        )
        self.val_transform = self.get_validation_transform()
        self.clip_duration = (
            self.transform_params["num_frames"] * self.transform_params["sampling_rate"]
        ) / self.frames_per_second

    # ---------------------------------------------------------------------
    # Model loading
    # ---------------------------------------------------------------------
    def load_model_disk(self, checkpoint_path, use_fp16=True):
        self.model = mvit_v2_s(weights=None)  # don't reload pretrained
        num_classes = len(self.class_dict)
        num_output_nodes = 1 if num_classes <= 2 else num_classes
        self.model.head[1] = nn.Linear(self.model.head[1].in_features, num_output_nodes, bias=True)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(state_dict, strict=False)

        if num_classes == 2:
            self.activation_fn = nn.Sigmoid()
        else:
            self.activation_fn = nn.Softmax(dim=1)

        self.model = self.model.to(self.device)
        # Optional: convert to FP16 for inference
        if use_fp16:
            self.model.half()
        self.model.eval()


    def load_model(self):
        """
        Loads pretrained MViT-v2 backbone from torchvision and adapts head for target classes.
        """
        num_classes = len(self.class_dict)
        weights = MViT_V2_S_Weights.KINETICS400_V1
        self.model = mvit_v2_s(weights=weights)

        # Replace classifier head
        num_output_nodes = 1 if num_classes <= 2 else num_classes
        self.model.head[1] = nn.Linear(self.model.head[1].in_features, num_output_nodes, bias=True)

        # Handle binary classification
        if num_classes == 2:
            self.activation_fn = nn.Sigmoid()
            print("Binary classification: 1 output (sigmoid)")
        else:
            self.activation_fn = nn.Softmax(dim=1)
            print(f"Multi-class classification: {num_classes} outputs (softmax)")

        self.model = self.model.to(self.device)
        self.model.eval()

    # ---------------------------------------------------------------------
    # Validation transforms
    # ---------------------------------------------------------------------
    def get_validation_transform(self):
        """Validation transform for MViT-v2 (single pathway)."""
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

    # ---------------------------------------------------------------------
    # Frame utilities
    # ---------------------------------------------------------------------
    def save_video_frames(self, frames: list):
        os.makedirs("frames", exist_ok=True)
        for idx, frame in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(os.path.join("frames", f"frame_{idx:02d}.png"))

    @staticmethod
    def save_video_tensors(
        transformed_video: torch.Tensor,
        mean=[0.45, 0.45, 0.45],
        std=[0.225, 0.225, 0.225],
    ):
        os.makedirs("transformed_frames", exist_ok=True)
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)

        for idx in range(transformed_video.shape[1]):
            frame = transformed_video[:, idx, :, :]
            frame_cpu = frame.detach().cpu()
            frame_min = frame_cpu.min().item()
            frame_max = frame_cpu.max().item()
            frame = frame * std + mean
            frame = frame.permute(1, 2, 0)
            print(f"Frame {idx:02d}: min={frame_min:.6f}, max={frame_max:.6f}")
            frame_np = (frame * 255).clip(0, 255).byte().cpu().numpy()
            Image.fromarray(frame_np).save(
                os.path.join("transformed_frames", f"transformed_frame_{idx:02d}.png")
            )

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------
    def predict_class(self, frames_batches: list):
        """
        Robust predict_class that avoids GPU memory leaks and cublas errors.
        frames_batches: list of batches, where each batch is a list of HxWxC uint8 frames
        Returns: list of (class_name, confidence)
        """
        results = []
        try:
            transformed_batches = []
            # Transform each video (on CPU)
            for frames in frames_batches:
                frames_np = np.stack(frames).astype(np.float32)  # [T, H, W, C]
                frames_tensor = torch.from_numpy(frames_np)  # CPU
                frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]
                frames_tensor = frames_tensor.contiguous()  # ensure contiguous
                video_data = {"video": frames_tensor}
                transformed_data = self.val_transform(video_data)
                # Ensure float32 and contiguous
                tv = transformed_data["video"].to(dtype=torch.float32).contiguous()
                transformed_batches.append(tv)

            if len(transformed_batches) == 0:
                return []

            # Stack and move to device in one call
            inputs_batch = torch.stack(transformed_batches)  # Stack CPU tensors -> [B, C, T, H, W]
            inputs_batch = inputs_batch.to(self.device)      # Move to GPU (or self.device)
            inputs_batch = inputs_batch.to(dtype=next(self.model.parameters()).dtype)  # Match model dtype

            # Optional: move model & inputs to channels_last memory format (often helps memory)
            # self.model = self.model.to(memory_format=torch.channels_last)
            # inputs_batch = inputs_batch.to(self.device, memory_format=torch.channels_last)

            # Inference (immediately move outputs to CPU)
            with torch.no_grad():
                preds_gpu = self.model(inputs_batch)

            # Move preds to CPU and free GPU memory ASAP
            preds = preds_gpu.detach().cpu()
            del preds_gpu
            del inputs_batch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            num_classes = len(self.class_dict)
            idx_to_class = {k: v for k, v in self.class_dict.items()}

            for batch_pred in preds:
                if num_classes == 2:
                    pos_prob = torch.sigmoid(batch_pred)
                    neg_prob = 1 - pos_prob
                    final_pred = torch.cat([neg_prob, pos_prob], dim=0)
                else:
                    final_pred = torch.softmax(batch_pred, dim=0)

                top_score, top_index = final_pred.max(dim=0)
                top_class_name = idx_to_class[int(top_index)]
                top_confidence = float(top_score)
                results.append((top_class_name, top_confidence))

            # free preds
            del preds
            torch.cuda.empty_cache()
            return results

        except RuntimeError as e:
            logger.error(f"RuntimeError during model inference: {e}")
            # extra diagnostics
            try:
                logger.error("CUDA memory allocated (bytes): %d", torch.cuda.memory_allocated())
                logger.error("CUDA memory reserved (bytes): %d", torch.cuda.memory_reserved())
            except Exception:
                pass
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return []
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return []

    # ---------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------
    def export_to_onnx(self, output_path="models/mvit_v2_s.onnx", batch_size=-1):
        if not hasattr(self, "model") or self.model is None:
            logger.info("Model not loaded. Loading from checkpoint.")
            self.load_model_disk(self.checkpoint_path)

        self.model.eval()  # disable dropout, etc.

        num_frames = self.transform_params["num_frames"]
        crop_size = self.transform_params["crop_size"]
        dummy_input = torch.randn(
            1 if batch_size < 0 else batch_size,
            3, num_frames, crop_size, crop_size,
            device=self.device,
        ).contiguous()

        # Dynamic axes for variable batch size
        dynamic_axes = (
            {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            if batch_size < 0 else {}
        )

        # Export ONNX with opset 17
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=17,  # <--- Changed from 13
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )
        logger.info(f"MViT-v2 exported to ONNX at {output_path}")



    def get_num_frames(self):
        return self.transform_params["num_frames"]

    def get_sampling_rate(self):
        return self.transform_params["sampling_rate"]
