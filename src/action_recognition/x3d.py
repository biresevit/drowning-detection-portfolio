import os
import torch
import numpy as np

from torchvision.transforms import Compose, Lambda, functional as F
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from action_recognition.custom_transforms import ResizeAndPadVideo
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from action_recognition.model_base import ModelBase
from PIL import Image
import torch.nn as nn

import cv2
import logging

logger = logging.getLogger()


class X3d(ModelBase):
    def __init__(self, model_name, device, checkpoint_path, classes, checkpoint_dict=None):
        self.checkpoint_dict = checkpoint_dict

        super().__init__(model_name, device, checkpoint_path, classes)
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.model_transform_params = self.get_model_configs()
        self.transform_params = self.model_transform_params[self.model_name]
        self.val_transform = self.get_validation_transform()
        self.clip_duration = (
            self.transform_params["num_frames"] * self.transform_params["sampling_rate"]
        ) / self.frames_per_second

    def load_model_disk(self, checkpoint_path):
        """
        Loads the X3D model from a local disk file and adjusts the output layer
        based on the number of classes.

        - If binary classification (2 classes), uses 1 output node with sigmoid.
        - If multi-class classification (>2 classes), uses N output nodes with softmax.
        """
        # Load the model architecture without pre-trained weights
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo", self.model_name, pretrained=False
        )

        # Load the checkpoint from the local disk
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract the 'model_state' key if it exists, otherwise use the full checkpoint
        state_dict = checkpoint.get("model_state", checkpoint)

        # Get the number of classes
        num_classes = len(self.class_dict)
        # Get the final "head" block.
        # For X3D from pytorchvideo, it's often self.model.blocks[-1] = ResNetBasicHead
        head = self.model.blocks[-1]

        if num_classes == 2:
            # 1) Remove or override the built-in Softmax in ResNetBasicHead:
            #    Some versions call it 'activation', or you might see 'act' or 'output_pool'.
            #    We'll replace with Identity if it exists.
            if hasattr(head, "activation"):
                head.activation = nn.Identity()

            # 2) Use a single output node
            head.proj = nn.Linear(head.proj.in_features, 1, bias=True)

            # We'll define self.activation_fn for inference (OPTIONAL):
            self.activation_fn = nn.Sigmoid()
            print("Binary classification: removed softmax, set final layer to 1 output")
        else:
            # Use the desired number of output nodes
            head.proj = nn.Linear(head.proj.in_features, num_classes, bias=True)

            self.activation_fn = nn.Softmax(dim=1)
            print(f"Multi-class classification: using {num_classes} outputs + softmax")

        # Load the state dictionary into the model
        self.model.load_state_dict(
            state_dict, strict=False
        )  # strict=False allows non-matching keys

        # Move the model to the specified device
        self.model = self.model.to(self.device)
        self.model.eval()

        print(
            f"Model loaded from disk with {num_classes} output node(s). Using {'sigmoid' if num_classes == 2 else 'softmax'} activation."
        )

    def load_model(self):
        """
        Loads the model, modifies the final layer based on the number of classes,
        and moves it to the specified device.

        - If binary classification (2 classes), uses 1 output node with raw logits (no softmax).
        - If multi-class classification (>2 classes), uses N output nodes with softmax activation.
        """
        # Load pretrained model from PyTorchVideo
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo", self.model_name, pretrained=True
        )

        # Get the number of classes
        num_classes = len(self.class_dict)

        # Get the final "head" block.
        # For X3D from pytorchvideo, it's often self.model.blocks[-1] = ResNetBasicHead
        head = self.model.blocks[-1]

        if num_classes == 2:
            # 1) Remove or override the built-in Softmax in ResNetBasicHead:
            #    Some versions call it 'activation', or you might see 'act' or 'output_pool'.
            #    We'll replace with Identity if it exists.
            if hasattr(head, "activation"):
                head.activation = nn.Identity()

            # 2) Use a single output node
            head.proj = nn.Linear(head.proj.in_features, 1, bias=True)

            # We'll define self.activation_fn for inference (OPTIONAL):
            self.activation_fn = nn.Sigmoid()
            print("Binary classification: removed softmax, set final layer to 1 output")
        else:
            # Use the desired number of output nodes
            head.proj = nn.Linear(head.proj.in_features, num_classes, bias=True)

            self.activation_fn = nn.Softmax(dim=1)
            print(f"Multi-class classification: using {num_classes} outputs + softmax")

        # Move model to device
        self.model = self.model.to(self.device)

        # Print a summary to confirm the final layers
        # summary(self.model, input_size=(32, 3, 13, 180, 180))

    def get_num_frames(self):
        return self.transform_params["num_frames"]

    def get_sampling_rate(self):
        return self.transform_params["sampling_rate"]

    def get_validation_transform(self):
        """Define the video preprocessing transformation for X3D model."""
        # Note: Since NormalizeVideo is a custom or pytorchvideo transform, ensure it uses torchvision.functional
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.transform_params["num_frames"]),
                    Lambda(lambda x: x / 255.0),
                    ResizeAndPadVideo(size=self.transform_params["crop_size"]),
                    NormalizeVideo(
                        self.mean, self.std
                    ),  # Ensure this uses torchvision.transforms.functional
                ]
            ),
        )

    def save_video_frames(self, frames: list):
        os.makedirs("frames", exist_ok=True)
        for idx, frame in enumerate(frames):
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Save as PNG image
            frame_path = os.path.join("frames", f"frame_{idx:02d}.png")
            Image.fromarray(frame_rgb).save(frame_path)

    @staticmethod
    def save_video_tensors(transformed_video: torch.Tensor, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]):
        os.makedirs("transformed_frames", exist_ok=True)
        # Mean and std used during normalization
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)

        for idx in range(
            transformed_video.shape[1]
        ):  # Iterate over T (temporal dimension)
            frame = transformed_video[:, idx, :, :]
            # Move tensor to CPU and detach, then print min/max values for debugging
            frame_cpu = frame.detach().cpu()
            frame_min = frame_cpu.min().item()
            frame_max = frame_cpu.max().item()
            # Undo normalization: multiply by std and add mean
            frame = frame * std + mean
            frame = frame.permute(1, 2, 0)  # Change to [H, W, C] for saving

            print(f"Transformed frame {idx:02d}: min={frame_min:.6f}, max={frame_max:.6f}")
            # Scale back to 0-255 and convert to byte format
            frame_np = (frame * 255).clip(0, 255).byte().cpu().numpy()
            frame_rgb = frame_np
            #frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            # Save the frame as an image
            frame_path = os.path.join(
                "transformed_frames", f"transformed_frame_{idx:02d}.png"
            )
            Image.fromarray(frame_rgb).save(frame_path)

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
            inputs_batch = torch.stack(transformed_batches)  # [B, C, T, H, W] on CPU
            # Optional: move model & inputs to channels_last memory format (often helps memory)
            # self.model = self.model.to(memory_format=torch.channels_last)
            # inputs_batch = inputs_batch.to(self.device, memory_format=torch.channels_last)
            inputs_batch = inputs_batch.to(self.device)

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

    def export_to_onnx(self, output_path="models/drowning_x3d_s.onnx", batch_size=-1):
        """
        Exports the loaded X3D model to ONNX format.

        Args:
            output_path (str): Path where the ONNX file will be saved.
            batch_size (int): Batch size for the dummy input. Use -1 for dynamic batch size (default).

        Returns:
            None
        """
        if not hasattr(self, "model") or self.model is None:
            logger.info("Model not loaded. Loading from checkpoint.")
            self.load_model_disk(self.checkpoint_path)

        # Prepare dummy input with dynamic batch size if batch_size < 0
        num_frames = self.transform_params["num_frames"]
        crop_size = self.transform_params["crop_size"]
        if batch_size < 0:
            # For dynamic batch size, use a placeholder batch size (e.g., 1) but rely on dynamic_axes
            dummy_input = torch.randn(1, 3, num_frames, crop_size, crop_size).to(
                self.device
            )
        else:
            # Static batch size
            dummy_input = torch.randn(
                batch_size, 3, num_frames, crop_size, crop_size
            ).to(self.device)

        self.model.eval()

        # Define dynamic axes for batch dimension if batch_size < 0
        dynamic_axes = (
            {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            if batch_size < 0
            else {}
        )
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            verbose=True,
        )
        logger.info(f"Model exported to ONNX at {output_path}")
