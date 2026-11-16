from pyexpat import model
from dotenv import load_dotenv
import torch
import os
import numpy as np
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomHorizontalFlip,
)
from action_recognition.custom_transforms import (
    VideoColorJitter,
    VideoRandomGrayscale,
    VideoGaussianBlur,
    VideoRandomAffine,
    VideoRandomErasing,
    VideoTemporalReverse,
    VideoCompressionArtifacts,
    ResizeAndPadVideo,
    RandomCropResize,
)

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    Normalize,
)
from action_recognition.slowfast import PackPathway
from pytorchvideo.data import LabeledVideoDataset
from pytorchvideo.data.clip_sampling import UniformClipSampler, RandomClipSampler
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split  # Added missing import
import mlflow
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from PIL import Image  # Import PIL for image saving
import json
from torch import amp
from torch.cuda.amp import GradScaler
from utc_logging import get_logger
from action_recognition.x3d import X3d
from action_recognition.slowfast import SlowFast

logger = get_logger(
    log_level="debug",
    filename="log.txt",
    log_type=0,
    max_log_size_mb=5,
    num_log_backup_files=5,
)

FRAMES_PER_SECOND = 30


# Set up argparse to take command-line arguments
def parse_env():
    load_dotenv()
    args = {
        "dataset_path": os.getenv("DATASET_PATH"),
        "model_variant": os.getenv("MODEL_VARIANT"),
        "batch_size": int(os.getenv("BATCH_SIZE")),
        "grad_accum_steps": int(os.getenv("GRAD_ACCUM_STEPS", "1")),
        "alpha_focal": float(os.getenv("ALPHA_FOCAL")),
        "early_stopping_patience": int(os.getenv("EARLY_STOPPING_PATIENCE")),
        "use_colorjitter": os.getenv("USE_COLORJITTER", "true").lower() == "true",
        "use_gaussianblur": os.getenv("USE_GAUSSIANBLUR", "true").lower() == "true",
        "use_randomgrayscale": os.getenv("USE_RANDOMGRAYSCALE", "true").lower() == "true",
        "use_random_affine": os.getenv("USE_RANDOM_AFFINE", "true").lower() == "true",
        "use_random_erasing": os.getenv("USE_RANDOM_ERASING", "true").lower() == "true",
        "use_temporal_reverse": os.getenv("USE_TEMPORAL_REVERSE", "false").lower() == "true",
        "use_compression_artifacts": os.getenv("USE_COMPRESSION_ARTIFACTS", "false").lower() == "true",
        "use_focal_loss": os.getenv("USE_FOCAL_LOSS", "true").lower() == "true",
        "mlflow_tracking_port": int(os.getenv("MLFLOW_TRACKING_PORT")),
        "classes": os.getenv("CLASSES").split(","),  # Parse comma-separated list
        "thresholds": [
            float(threshold) for threshold in os.getenv("THRESHOLDS").split(",")
        ],
        "checkpoint_path": os.getenv("CHECKPOINT_PATH"),
        "chance_random_crop": float(os.getenv("CHANCE_RANDOM_CROP", "0.5")),
    }
    return args

def freeze_backbone_swin(model):
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, alpha=0.33, gamma=2):
        """
        Asymmetric Focal Loss supporting both binary and multi-class classification.

        Args:
            alpha (float): Weighting factor for the positive class.
            gamma (float): Focusing parameter for difficult examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): Raw model outputs of shape [B, C] or [B, 1].
            targets (Tensor): Ground-truth labels:
                              - shape [B] with class indices (multi-class)
                              - or shape [B, 1] / [B, C] for binary or multi-label.

        Returns:
            Scalar loss value.
        """
        # Cast to float32: float16 underflows for probs near 0/1, making log(probs) = -inf → nan
        logits = logits.float()
        # 🔹 Convert logits to probabilities
        probs = torch.sigmoid(logits) if logits.size(1) == 1 else torch.softmax(logits, dim=1)
        probs = probs.clamp(min=1e-7, max=1 - 1e-7)

        # 🔹 Match targets shape
        if targets.ndim == 1:
            # class indices -> one-hot
            num_classes = logits.size(1)
            targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
        elif targets.ndim == 2 and targets.size(1) == 1 and logits.size(1) > 1:
            # single column -> one-hot
            num_classes = logits.size(1)
            targets = torch.nn.functional.one_hot(targets.squeeze(1).long(), num_classes=num_classes).float()
        else:
            targets = targets.float()

        # 🔹 Compute focal weights
        focal_weight = self.alpha * (1.0 - probs) ** self.gamma * targets + \
                       (1.0 - self.alpha) * (probs ** self.gamma) * (1.0 - targets)

        # 🔹 Binary cross-entropy per class
        bce = - (targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs))
        loss = focal_weight * bce
        return loss.mean()



class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# Custom dataset class for video loading
class CustomVideoDataset(LabeledVideoDataset):
    def __init__(
        self,
        duration,
        video_paths,
        labels,
        transform,
        annotations_dir,
        use_random_clip_sampler,
        random_crop_resize=None,
        resize_and_pad=None,
        chance_random_crop=0.5,
        pack_pathway_transf=None,
    ):
        """
        Custom dataset class that applies base transformations first, then bounding box-based RandomCropResize
        or ResizeAndPadVideo based on a probability.

        Args:
            video_paths (list): List of paths to video files.
            labels (list): List of labels for each video.
            transform (Callable): Base transformations to apply before RandomCropResize.
            annotations_dir (str): Directory with YOLO-format annotations.
            random_crop_resize (Callable): Final transformation to apply that respects bounding box (RandomCropResize).
            chance_random_crop (float): Probability of applying RandomCropResize over ResizeAndPadVideo.
        """
        self.pack_pathway_transf = pack_pathway_transf
        self.labeled_video_paths = [
            (video_path, {"label": label})
            for video_path, label in zip(video_paths, labels)
        ]
        self.annotations_dir = annotations_dir
        self.random_crop_resize = random_crop_resize
        self.chance_random_crop = chance_random_crop
        self.resize_and_pad = resize_and_pad
        # Define the clip sampler
        clip_sampler = (
            RandomClipSampler(clip_duration=duration)
            if use_random_clip_sampler
            else UniformClipSampler(clip_duration=duration)
        )

        # Initialize the LabeledVideoDataset with transformations, excluding RandomCropResize
        super().__init__(
            labeled_video_paths=self.labeled_video_paths,
            clip_sampler=clip_sampler,
            video_sampler=torch.utils.data.RandomSampler,
            transform=transform,
            decode_audio=False,
        )

    def load_bbox(self, video_name):
        """
        Load bounding box for the given video name from YOLO-format annotations.

        Args:
            video_name (str): Name of the video file.

        Returns:
            bbox (list): Bounding box in YOLO format [class_id, x_center, y_center, width, height].
        """
        base_filename = os.path.splitext(os.path.basename(video_name))[0]
        annotation_path = os.path.join(self.annotations_dir, f"{base_filename}.txt")

        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as f:
                line = f.readline().strip()
                bbox = [float(x) for x in line.split()]

                # Ensure width and height do not exceed 1
                if bbox[3] > 1:
                    bbox[3] = 1
                if bbox[4] > 1:
                    bbox[4] = 1

                return bbox
        return None

    def __next__(self):
        """
        Retrieve the next sample from LabeledVideoDataset, then apply either RandomCropResize or ResizeAndPadVideo
        based on a probability.

        Returns:
            A dictionary with video, label, and additional transformed data.
        """
        # Retrieve the next sample without RandomCropResize
        sample = super().__next__()

        # Inject bounding box information if available
        video_name = sample["video_name"]
        bbox = self.load_bbox(video_name)

        # Randomly decide between RandomCropResize and ResizeAndPadVideo based on chance_random_crop
        if self.random_crop_resize:
            if random.random() < self.chance_random_crop:
                sample["video"] = self.random_crop_resize(sample["video"], bbox)
            else:
                # Perform ResizeAndPadVideo transformation here (assuming ResizeAndPadVideo is defined)
                sample["video"] = self.resize_and_pad(sample["video"])

        if self.pack_pathway_transf:
            sample["video"] = self.pack_pathway_transf(sample["video"])
        return sample


import random


def seed_worker(worker_id):
    # Seed everything without creating a torch.Generator instance
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# Step 1: Calculate class weights
def calculate_class_weights(labels):
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = [total_samples / count for count in class_counts]
    return torch.FloatTensor(class_weights)


def save_augmented_images(train_loader, output_dir, std, mean, num_samples_to_save=10):
    os.makedirs(output_dir, exist_ok=True)
    sample_counter = 0

    for batch in train_loader:
        videos = batch["video"]  # X3D: Tensor; SlowFast: List of two Tensors
        labels = batch["label"]

        # Handle SlowFast case where videos is a list of two tensors
        if isinstance(videos, list) and len(videos) == 2:
            fastpath_videos = videos[1]  # Use the fastpath tensor
        else:
            fastpath_videos = videos  # Use directly for X3D

        batch_size = fastpath_videos.size(0)
        num_frames = fastpath_videos.size(2)

        for i in range(batch_size):
            video = fastpath_videos[i]  # Shape: (3, num_frames, H, W)
            label = labels[i].item()
            for frame_idx in range(num_frames):
                frame = video[:, frame_idx, :, :]  # Shape: (3, H, W)
                # Denormalize the frame
                frame = frame.numpy()
                for c in range(3):
                    frame[c] = (frame[c] * std[c]) + mean[c]
                # Convert frame from (C, H, W) to (H, W, C)
                frame = np.transpose(frame, (1, 2, 0))
                # Clip values to [0, 1]
                frame = np.clip(frame, 0, 1)
                # Convert to uint8
                frame = (frame * 255).astype(np.uint8)
                # Save the image
                frame_pil = Image.fromarray(frame)
                img_name = f"sample_{sample_counter}_frame_{frame_idx}.jpg"
                save_path = os.path.join(output_dir, img_name)
                frame_pil.save(save_path)
                mlflow.log_artifact(
                    local_path=save_path, artifact_path=f"{output_dir}/{img_name}"
                )
            sample_counter += 1
            if sample_counter >= num_samples_to_save:
                return


def train_1_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
    class_dict,
    scheduler=None,
    scaler: GradScaler = None,
    accum_steps=1
):
    """
    Trains the model for one epoch using gradient accumulation and AMP.
    Supports binary (1-output sigmoid) and multi-class (softmax) classification.

    Args:
        model: PyTorch model
        loader: DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: cuda or cpu
        epoch: current epoch number
        class_dict: dict of class_name -> idx
        scheduler: learning rate scheduler (optional)
        scaler: GradScaler for AMP (optional)
        accum_steps: gradient accumulation steps
    Returns:
        train_loss, train_accuracy, per-class precision
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    num_classes = len(class_dict)

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        videos = batch["video"]
        labels = batch["label"].to(device)

        if isinstance(videos, list):
            videos = [v.to(device) for v in videos]
        else:
            videos = videos.to(device)

        # 🔹 Forward pass under AMP
        with amp.autocast(device_type='cuda'):
            outputs = model(videos) 
            if num_classes == 2:
                labels_ = labels.unsqueeze(1).float()
                loss = criterion(outputs, labels_)
                predicted = (torch.sigmoid(outputs) > 0.5).long()
            else:
                loss = criterion(outputs, labels)
                predicted = torch.argmax(outputs, dim=1).long()

        # 🔹 Scale loss for gradient accumulation
        loss = loss / accum_steps

        # 🔹 Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 🔹 Optimizer step only every `accum_steps`
        if (batch_idx + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        # 🔹 Update metrics
        batch_size = labels.size(0)
        loss_val = loss.item() * accum_steps
        if not (loss_val != loss_val):  # nan check
            running_loss += loss_val * batch_size
        total += batch_size
        correct += (predicted.view(-1) == labels.view(-1).long()).sum().item()

        all_labels.extend(labels.cpu().numpy().flatten())
        all_preds.extend(predicted.cpu().numpy().flatten())

        if (batch_idx + 1) % 10 == 0:
            batch_acc = 100.0 * correct / total if total > 0 else 0.0
            print(
                f"[Epoch {epoch} | Batch {batch_idx + 1}] "
                f"Loss: {loss.item() * accum_steps:.4f} | "
                f"Running Acc: {batch_acc:.2f}% ({correct}/{total})"
            )

    # 🔹 Epoch metrics
    train_loss = running_loss / total
    train_accuracy = correct / total * 100.0

    labels_list = list(range(num_classes))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        train_precision = precision_score(
            all_labels, all_preds, average=None, labels=labels_list, zero_division=0
        )

    if scheduler is not None:
        scheduler.step()

    print(
        f"Epoch {epoch} Summary:\n"
        f"  Train Loss: {train_loss:.4f}\n"
        f"  Train Accuracy: {train_accuracy:.2f}%\n"
        f"  Per-Class Precision: {train_precision}"
    )

    return train_loss, train_accuracy, train_precision




def validate(model, val_loader, criterion, device, class_dict, epoch, thresholds=[0.5]):
    """
    Validation loop supporting binary (1-output sigmoid) and multi-class softmax.
    Returns per-threshold metrics for binary and per-class metrics for multi-class.
    """
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels = []
    all_probs = []

    num_classes = len(class_dict)

    with torch.no_grad():
        for batch in val_loader:
            videos = batch["video"]
            labels = batch["label"].to(device)

            if isinstance(videos, list):
                videos = [v.to(device) for v in videos]
            else:
                videos = videos.to(device)

            with amp.autocast('cuda'):
                outputs = model(videos)
                if num_classes == 2:
                    labels = labels.unsqueeze(1).float()
                loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            loss_val = loss.item()
            if not (loss_val != loss_val):  # nan check
                total_loss += loss_val * batch_size
            total += batch_size

            # Store probabilities and labels
            if num_classes == 2:
                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
            else:
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    mlflow.log_metric("val_loss", avg_loss, step=epoch)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics_by_threshold = {}

    # Binary classification metrics per threshold
    if num_classes == 2:
        for threshold in thresholds:
            predicted = (all_probs > threshold).astype(int)
            accuracy = (predicted == all_labels.astype(int)).sum() / len(all_labels) * 100
            labels_list = list(range(num_classes))

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                precision = precision_score(all_labels, predicted, average=None, labels=labels_list, zero_division=0)
                recall = recall_score(all_labels, predicted, average=None, labels=labels_list, zero_division=0)
                f1 = f1_score(all_labels, predicted, average=None, labels=labels_list, zero_division=0)
                precision_overall = precision_score(all_labels, predicted, average="weighted", zero_division=0)
                recall_overall = recall_score(all_labels, predicted, average="weighted", zero_division=0)
                f1_overall = f1_score(all_labels, predicted, average="weighted", zero_division=0)

            metrics_by_threshold[threshold] = {
                "accuracy": accuracy,
                "precision": precision,
                "precision_overall": precision_overall,
                "recall": recall,
                "recall_overall": recall_overall,
                "f1": f1,
                "f1_overall": f1_overall,
            }

            if threshold == 0.5:
                mlflow.log_metric("val_accuracy", accuracy, step=epoch)
                for i, cls in enumerate(class_dict.keys()):
                    mlflow.log_metric(f"val_precision_{cls}", precision[i], step=epoch)
                    mlflow.log_metric(f"val_recall_{cls}", recall[i], step=epoch)
                    mlflow.log_metric(f"val_f1_{cls}", f1[i], step=epoch)
                mlflow.log_metric("val_f1_overall", f1_overall, step=epoch)

    # Multi-class metrics
    else:
        predicted = np.argmax(all_probs, axis=1)
        all_labels = all_labels.flatten()
        accuracy = (predicted == all_labels.astype(int)).sum() / len(all_labels) * 100
        labels_list = list(range(num_classes))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            precision = precision_score(all_labels, predicted, average=None, labels=labels_list, zero_division=0)
            recall = recall_score(all_labels, predicted, average=None, labels=labels_list, zero_division=0)
            f1 = f1_score(all_labels, predicted, average=None, labels=labels_list, zero_division=0)
            precision_overall = precision_score(all_labels, predicted, average="weighted", zero_division=0)
            recall_overall = recall_score(all_labels, predicted, average="weighted", zero_division=0)
            f1_overall = f1_score(all_labels, predicted, average="weighted", zero_division=0)

        metrics_by_threshold = {
            0.5: {
                "accuracy": accuracy,
                "precision": precision,
                "precision_overall": precision_overall,
                "recall": recall,
                "recall_overall": recall_overall,
                "f1": f1,
                "f1_overall": f1_overall,
            }
        }

        mlflow.log_metric("val_accuracy", accuracy, step=epoch)
        for i, cls in enumerate(class_dict.keys()):
            mlflow.log_metric(f"val_precision_{cls}", precision[i], step=epoch)
            mlflow.log_metric(f"val_recall_{cls}", recall[i], step=epoch)
            mlflow.log_metric(f"val_f1_{cls}", f1[i], step=epoch)
        mlflow.log_metric("val_f1_overall", f1_overall, step=epoch)

    return metrics_by_threshold, avg_loss




def normalize_255(x):
    return x / 255.0


def get_class_dict(root_dir):
    class_names = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
    if "annotations" in class_names:
        class_names.remove("annotations")

    class_dict = {class_name: idx for idx, class_name in enumerate(class_names)}
    return class_dict


# Load dataset from pre-split directories
def load_dataset_from_folders(dataset_path, class_dict):
    """
    Loads dataset from pre-split 'train', 'val', and 'test' directories.

    Args:
        root_dir (str): Path to the dataset root directory (containing 'train', 'val', 'test').

    Returns:
        dict: Dictionary containing train, val, test paths and labels, as well as metadata.
    """
    dataset_splits = ["train", "val", "test"]
    dataset = {split: {"video_paths": [], "labels": []} for split in dataset_splits}

    for split in dataset_splits:
        split_dir = os.path.join(dataset_path, split)
        if not os.path.exists(split_dir):
            raise ValueError(
                f"{split_dir} does not exist. Ensure the dataset is split correctly."
            )

        valid_extensions = (".mp4", ".avi", ".mov", ".mkv")

        for label, class_name in class_dict.items():
            class_dir = os.path.join(split_dir, class_name)
            for video_file in os.listdir(class_dir):
                if video_file.lower().endswith(valid_extensions):
                    dataset[split]["video_paths"].append(
                        os.path.join(class_dir, video_file)
                    )
                    dataset[split]["labels"].append(label)

    # Shuffle each split
    for split in dataset_splits:
        dataset[split]["video_paths"], dataset[split]["labels"] = shuffle(
            dataset[split]["video_paths"], dataset[split]["labels"], random_state=65
        )

    dataset["class_dict"] = class_dict
    dataset["num_classes"] = len(class_dict)
    return dataset


def get_train_transforms(args, num_frames, mean, std):
    # Updated transforms
    transform_list = [
        UniformTemporalSubsample(num_frames),
        Lambda(normalize_255),
        RandomHorizontalFlip(p=0.5),
    ]

    if args["use_colorjitter"]:
        transform_list.append(VideoColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.08,   # ±0.2 turns blue water purple/green; ±0.08 is realistic lighting variation
            p=0.5,      # was 0.33 — augment half of clips
        ))

    if args["use_gaussianblur"]:
        transform_list.append(VideoGaussianBlur(
            kernel_size=(3, 5),
            sigma=(0.1, 1.0),   # was 1.5 — softer blur, avoids destroying person silhouette
            p=0.3,
        ))

    if args["use_randomgrayscale"]:
        # Color is discriminative for pools (blue water); keep this rare
        transform_list.append(VideoRandomGrayscale(p=0.05))

    if args["use_random_affine"]:
        # Pool cameras are fixed; 30° rotation is never realistic, 10° covers slight tilt/mounting variation
        transform_list.append(
            VideoRandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15), p=0.4)
        )

    if args["use_random_erasing"]:
        # Smaller patches (max 8%) — large erasures on small drowning persons destroy the signal
        transform_list.append(VideoRandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0))

    if args["use_temporal_reverse"]:
        # Keep disabled: drowning has a temporal direction, reversing teaches wrong patterns
        transform_list.append(VideoTemporalReverse(p=0.5))

    if args["use_compression_artifacts"]:
        # Realistic CCTV compression; was quality 20-60 which is extreme
        transform_list.append(VideoCompressionArtifacts(quality_range=(40, 80), p=0.3))

    transform_list.append(Normalize(mean=mean, std=std))

    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(transform_list),
    )

    return transform


def get_data_loaders(train_dataset, val_dataset, train_labels, args):
    # Adjust num_workers based on your system
    num_workers = 0  # Adjust this number as per your system

    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args["batch_size"],
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader


def get_datasets(
    duration,
    annotation_dir,
    train_video_paths,
    val_video_paths,
    train_labels,
    val_labels,
    transform,
    transform_val,
    crop_size,
    pack_pathway_transf,
):
    random_crop_resize = RandomCropResize(size=crop_size, perc_bbox_within_crop=0.8)
    resize_and_pad = ResizeAndPadVideo(size=crop_size)
    train_dataset = CustomVideoDataset(
        duration,
        train_video_paths,
        train_labels,
        transform,
        annotation_dir,
        use_random_clip_sampler=True,
        random_crop_resize=random_crop_resize,
        resize_and_pad=resize_and_pad,
        pack_pathway_transf=pack_pathway_transf,
        chance_random_crop=args["chance_random_crop"],
    )
    val_dataset = CustomVideoDataset(
        duration,
        val_video_paths,
        val_labels,
        transform_val,
        annotation_dir,
        use_random_clip_sampler=False,
        pack_pathway_transf=None,
    )
    return train_dataset, val_dataset


def log_mlflow_params(
    args,
    crop_size,
    num_frames,
    sampling_rate,
    train_video_paths,
    val_video_paths,
    classes,
    train_labels=None,
    val_labels=None,
    class_dict=None,
):
    mlflow.log_param("model_variant", args["model_variant"])
    mlflow.log_param("use_gaussianblur", args["use_gaussianblur"])
    mlflow.log_param("use_randomgrayscale", args["use_randomgrayscale"])
    mlflow.log_param("crop_size", crop_size)
    mlflow.log_param("num_frames", num_frames)
    mlflow.log_param("sampling_rate", sampling_rate)
    mlflow.log_param("batch_size", args["batch_size"])
    mlflow.log_param("early_stopping_patience", args["early_stopping_patience"])
    mlflow.log_param("total_videos", len(train_video_paths) + len(val_video_paths))
    mlflow.log_param("use_focal_loss", args["use_focal_loss"])
    class_dict_str = json.dumps(classes)
    mlflow.log_param("classes", class_dict_str)

    if train_labels is not None and class_dict is not None:
        # Determine which integer label corresponds to each class name
        name_to_label = {v: k for k, v in class_dict.items()}
        drowning_lbl = name_to_label.get("drowning")
        normal_lbl   = name_to_label.get("normal")

        for split_name, labels in [("train", train_labels), ("val", val_labels or [])]:
            n_drowning = sum(1 for l in labels if l == drowning_lbl)
            n_normal   = sum(1 for l in labels if l == normal_lbl)
            total      = n_drowning + n_normal
            ratio      = round(n_drowning / total, 3) if total > 0 else float("inf")
            mlflow.log_param(f"{split_name}_count_drowning", n_drowning)
            mlflow.log_param(f"{split_name}_count_normal",   n_normal)
            mlflow.log_param(f"{split_name}_drowning_normal_ratio", ratio)


def get_loss(args, train_labels=None, device=None):
    if args["use_focal_loss"]:
        return AsymmetricFocalLoss(alpha=args["alpha_focal"])
    else:
        if train_labels is not None:
            pos_weight = get_pos_weight(train_labels, device)
            if pos_weight is not None:
                return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return nn.BCEWithLogitsLoss()


def get_pos_weight(labels, device):
    """
    Compute pos_weight for BCEWithLogitsLoss.
    Assumes binary labels {0,1}.
    """
    labels_np = np.array(labels)
    pos = (labels_np == 1).sum()
    neg = (labels_np == 0).sum()
    if pos == 0:
        return None
    return torch.tensor([neg / pos], device=device)



# Main training script
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_variant = args["model_variant"]
    checkpoint_path = args["checkpoint_path"]
    if "slowfast" in model_variant:
        model_wrapper = SlowFast(
            model_name=model_variant,
            device=device,
            checkpoint_path=checkpoint_path,
            classes=args["classes"],
        )
    elif "mvit_v2" in model_variant:
        from action_recognition.mvit_v2 import MViTv2
        model_wrapper = MViTv2(
            model_name=model_variant,
            device=device,
            checkpoint_path=checkpoint_path,
            classes=args["classes"],
        )
    elif "swin" in model_variant:
        from action_recognition.video_swin import VideoSwin
        model_wrapper = VideoSwin(
            model_name=model_variant,
            device=device,
            checkpoint_path=checkpoint_path,
            classes=args["classes"],
        )
    else:
        model_wrapper = X3d(
            model_name=model_variant,
            device=device,
            checkpoint_path=checkpoint_path,
            classes=args["classes"],
        )
    model = model_wrapper.model
    class_dict = model_wrapper.class_dict

    dataset = load_dataset_from_folders(args["dataset_path"], class_dict)
    train_video_paths = dataset["train"]["video_paths"]
    val_video_paths = dataset["val"]["video_paths"]
    train_labels = dataset["train"]["labels"]
    val_labels = dataset["val"]["labels"]
    if len(train_video_paths) == 0 or len(val_video_paths) == 0:
        raise ValueError(f"Dataset is not loaded properly")
    if len(train_video_paths) != len(train_labels) or len(val_video_paths) != len(
        val_labels
    ):
        raise ValueError(f"Dataset labels don't match number of videos")

    # Retrieve model-specific transform parameters
    transform_params = model_wrapper.transform_params
    crop_size = transform_params["crop_size"]
    num_frames = transform_params["num_frames"]
    sampling_rate = transform_params["sampling_rate"]

    mean, std = model_wrapper.get_std_mean()
    train_transform = get_train_transforms(args, num_frames, mean, std)
    val_transform = model_wrapper.val_transform
    annotation_dir = os.path.join(args["dataset_path"], "annotations")
    if not os.path.exists(annotation_dir):
        raise ValueError(f"Annotation dir doesnt exist: {annotation_dir}")

    pack_pathway_transf = (
        PackPathway(model_wrapper.alpha) if "slowfast" in model_variant else None
    )
    train_dataset, val_dataset = get_datasets(
        model_wrapper.clip_duration,
        annotation_dir,
        train_video_paths,
        val_video_paths,
        train_labels,
        val_labels,
        train_transform,
        val_transform,
        crop_size,
        pack_pathway_transf,
    )

    train_loader, val_loader = get_data_loaders(
        train_dataset, val_dataset, train_labels, args
    )


    DEFAULT_FREEZE_EPOCHS = 5

    resuming_training = checkpoint_path is not None and os.path.exists(checkpoint_path)

    FREEZE_EPOCHS = 0 if resuming_training else DEFAULT_FREEZE_EPOCHS

    if "swin" in model_variant and FREEZE_EPOCHS > 0:
        logger.info(f"Freezing Swin backbone for first {FREEZE_EPOCHS} epochs")
        freeze_backbone_swin(model)

    if "mvit_v2" in model_variant:
        # Separate head and backbone (everything except head)
        backbone_params = [p for name, p in model.named_parameters() if not name.startswith("head")]
        head_params = [p for name, p in model.named_parameters() if name.startswith("head")]

        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': head_params,     'lr': 1e-3}
        ])
    elif "swin" in model_variant:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.05)        
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    scheduler = None
    if "swin" in model_variant:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=5
        )

    criterion = get_loss(args, train_labels=train_labels, device=device)
    scaler = GradScaler()

    # MLflow logging setup
    with mlflow.start_run():
        log_mlflow_params(
            args,
            crop_size,
            num_frames,
            sampling_rate,
            train_video_paths,
            val_video_paths,
            args["classes"],
            train_labels=train_labels,
            val_labels=val_labels,
            class_dict=class_dict,
        )
        # Save augmented images from the first 10 samples
        save_augmented_images(train_loader, output_dir='augmented_images', mean=mean, std=std, num_samples_to_save=25)
        # save_augmented_images(val_loader, output_dir='val_augmented_images', mean=mean, std=std, num_samples_to_save=25)

        # Early stopping settings
        best_val_loss = float("inf")
        epochs_no_improve = 0
        early_stopping_patience = args["early_stopping_patience"]
        accum_steps = args.get("grad_accum_steps", 1)
                    
        # Training and validation loop
        for epoch in range(100):  # Max number of epochs
            # Training step
            if "swin" in model_variant and epoch == FREEZE_EPOCHS:
                logger.info("Unfreezing Swin backbone")
                unfreeze_all(model)

            train_1_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                class_dict=class_dict,
                scheduler=scheduler,   # or your actual scheduler
                scaler=scaler,
                accum_steps=accum_steps,
            )


            # Validation step
            val_metrics, val_loss = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                class_dict=class_dict,
                epoch=epoch,
                thresholds=args["thresholds"],
            )
            logger.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")
            for threshold, metric in val_metrics.items():
                logger.info(f"Val_accuracy_{threshold}: {metric['accuracy']:.2f}%")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save the model within MLflow artifacts
                model_dir = "models"
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "best_model.pth")

                # Save the model to disk
                torch.save(model.state_dict(), model_path)

                # Log the saved model as an artifact in MLflow
                mlflow.log_artifact(model_path)

                for threshold, metric in val_metrics.items():
                    # Log metrics in MLflow with the threshold in the metric name
                    mlflow.log_metric(
                        f"best_accuracy_{threshold}", metric["accuracy"], step=epoch
                    )
                    for i, value in enumerate(metric["precision"]):
                        class_name = list(class_dict.keys())[i]
                        mlflow.log_metric(
                            f"best_precision_{class_name}_{threshold}",
                            value,
                            step=epoch,
                        )
                    for i, value in enumerate(metric["recall"]):
                        class_name = list(class_dict.keys())[i]
                        mlflow.log_metric(
                            f"best_recall_{class_name}_{threshold}", value, step=epoch
                        )
                    mlflow.log_metric(
                        f"best_f1_overall_{threshold}", metric["f1_overall"], step=epoch
                    )
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                logger.info("Early stopping")
                break


if __name__ == "__main__":
    args = parse_env()
    port = args["mlflow_tracking_port"]
    mlflow.set_tracking_uri(f"http://localhost:{port}")
    mlflow.set_experiment("Drowning_X3d")
    main(args)
