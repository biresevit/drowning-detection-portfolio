import numpy as np
import torch
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from action_recognition.custom_transforms import ResizeAndPadVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from action_recognition.model_base import ModelBase


# Transform for SlowFast model
class PackPathway(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        return [slow_pathway, fast_pathway]


class SlowFast(ModelBase):
    def __init__(self, model_name, device, checkpoint_path, classes):
        super().__init__(model_name, device, checkpoint_path, classes)
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.crop_size = 256
        self.num_frames = 32
        self.alpha = 4
        self.sampling_rate = 2
        self.transform_params = {
            "crop_size": self.crop_size,
            "num_frames": self.num_frames,
            "alpha": self.alpha,
            "sampling_rate": self.sampling_rate,
        }
        self.clip_duration = (
            self.transform_params["num_frames"] * self.transform_params["sampling_rate"]
        ) / self.frames_per_second
        self.val_transform = self.get_validation_transform()

    def predict_class(self, frames: list, overlap_ratio: float = 0.5):
        # Convert frames to tensor and permute to [C, T, H, W] format
        frames_tensor = torch.tensor(np.stack(frames), dtype=torch.float32).permute(
            3, 0, 1, 2
        )

        num_frames = (
            self.get_num_frames()
        )  # Assuming you have a method to get num_frames

        # Calculate the sliding window step size based on the overlap ratio
        step_size = int(num_frames * (1 - overlap_ratio))
        if step_size <= 0:
            raise ValueError(
                "Overlap ratio is too high, resulting in no new frames per batch."
            )

        # List to hold transformed batches
        transformed_batch_list = []

        # Process frames with sliding window and overlap
        for start_idx in range(0, frames_tensor.shape[1] - num_frames + 1, step_size):
            end_idx = start_idx + num_frames
            batch_frames = frames_tensor[:, start_idx:end_idx, :, :]

            # Apply the transformation on CPU for each batch
            video_data = {"video": batch_frames}
            transformed_data = self.val_transform(video_data)
            # For SlowFast, collect transformed frames for both pathways
            transformed_batch_list.append([i for i in transformed_data["video"]])

        # Stack all transformed batches to create a batch of inputs.
        # SlowFast expects two streams (e.g. slow and fast):
        transformed_batch_list_slow = torch.stack(
            [batch[0] for batch in transformed_batch_list]
        ).to(self.device)
        transformed_batch_list_fast = torch.stack(
            [batch[1] for batch in transformed_batch_list]
        ).to(self.device)
        inputs = [transformed_batch_list_slow, transformed_batch_list_fast]

        # Perform inference on all batches in parallel
        with torch.no_grad():
            preds = self.model(inputs)

        # Use the number of classes to decide the branch
        num_classes = len(self.class_dict)

        if num_classes == 2:
            # --- Binary Classification ---
            # For binary classification, the model has 1 output node.
            # Aggregate predictions (here we use the mean, but you could also sum)
            aggregated_logit = torch.mean(preds, dim=0, keepdim=True)
            # Apply sigmoid to get probability for the positive class
            pos_prob = torch.sigmoid(aggregated_logit)
            # Compute probability for the negative class as 1 - pos_prob
            neg_prob = 1 - pos_prob
            # Create a two-element probability vector [P(negative), P(positive)]
            final_preds = torch.cat([neg_prob, pos_prob], dim=1)
            # Since there are only two classes, use topk with k=2
            topk_scores, topk_indices = final_preds.topk(k=2, dim=1)
        else:
            # --- Multi-class Classification ---
            # Aggregate logits (here using sum) and then apply softmax
            aggregated_logits = torch.sum(preds, dim=0, keepdim=True)
            final_preds = torch.softmax(aggregated_logits, dim=1)
            topk_scores, topk_indices = final_preds.topk(k=5, dim=1)

        # Create an inverse mapping from label index to class name
        idx_to_class = {v: k for k, v in self.class_dict.items()}
        topk_class_names = [idx_to_class[int(idx)] for idx in topk_indices[0]]
        topk_confidences = [float(score) for score in topk_scores[0]]

        # Get the top-1 predicted class and its confidence score
        top_class_name = topk_class_names[0]
        top_class_confidence = topk_confidences[0]

        print("Top predicted labels and confidences:")
        for class_name, confidence in zip(topk_class_names, topk_confidences):
            print(f"{class_name}: {confidence:.4f}")

        return top_class_name, top_class_confidence

    def get_num_frames(self):
        return self.num_frames

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_validation_transform(self):
        """Define the video preprocessing transformation."""
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(self.mean, self.std),
                    ResizeAndPadVideo(size=self.crop_size),
                    PackPathway(self.alpha),
                ]
            ),
        )
