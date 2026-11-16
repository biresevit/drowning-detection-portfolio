from dotenv import load_dotenv
import torch
import os
from pytorchvideo.data import LabeledVideoDataset
from pytorchvideo.data.clip_sampling import UniformClipSampler
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import mlflow
import mlflow.pytorch
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from torch import autocast
from utc_logging import get_logger
from action_recognition.x3d import X3d
from action_recognition.slowfast import SlowFast
import shutil
from action_recognition.load_models_onnx import GenericModelLoader
from action_recognition.load_models_tensorrt import TensorRTModelLoader
import av


logger = get_logger(
    log_level="debug",
    filename="validate_log.txt",
    log_type=0,
    max_log_size_mb=5,
    num_log_backup_files=5,
)


def parse_env():
    load_dotenv()
    args = {
        "dataset_path": r"path/to/data",
        "model_variant": os.getenv("MODEL_VARIANT"),
        "batch_size": int(os.getenv("MAX_CLUSTERS")),
        "use_focal_loss": os.getenv("USE_FOCAL_LOSS", "true").lower() == "true",
        "mlflow_tracking_port": int(os.getenv("MLFLOW_TRACKING_PORT")),
        "classes": os.getenv("CLASSES").split(","),  # Parse comma-separated list
        "thresholds": [0.3],  # Define thresholds for drowning detection
        "model_extension": os.getenv("MODEL_EXTENSION"),
    }
    return args


class AsymmetricFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.33, gamma=2):
        super(AsymmetricFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        thres_clamp = 5e-4
        probs = torch.sigmoid(logits).clamp(min=thres_clamp, max=1 - thres_clamp)
        focal_weight = self.alpha * (1.0 - probs) ** self.gamma * targets + (
            1.0 - self.alpha
        ) * (probs**self.gamma) * (1.0 - targets)
        bce = -(targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs))
        loss = focal_weight * bce
        loss_mean = loss.mean()
        return loss_mean


def get_class_dict(root_dir):
    class_names = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
    if "annotations" in class_names:
        class_names.remove("annotations")
    class_dict = {class_name: idx for idx, class_name in enumerate(class_names)}
    return class_dict


def is_valid_video(path):
    try:
        with av.open(path) as container:
            return any(s.type == "video" for s in container.streams)
    except Exception:
        return False


def load_validation_data(dataset_path, class_dict):
    val_dir = dataset_path
    if not os.path.exists(val_dir):
        raise ValueError(f"Validation directory {val_dir} does not exist.")

    video_paths = []
    labels = []
    valid_extensions = (".mp4", ".avi", ".mov", ".mkv")
    skipped = 0

    for label, class_name in class_dict.items():
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_dir):
            logger.warning(f"Class directory not found: {class_dir}")
            continue
        for root, _, files in os.walk(class_dir):
            for video_file in sorted(files):
                if video_file.lower().endswith(valid_extensions):
                    full_path = os.path.join(root, video_file)
                    if is_valid_video(full_path):
                        video_paths.append(full_path)
                        labels.append(label)
                    else:
                        logger.warning(f"Skipping corrupt/unreadable video: {full_path}")
                        skipped += 1

    if skipped:
        logger.info(f"Skipped {skipped} invalid video(s) during dataset loading.")

    return video_paths, labels


def get_val_dataset(duration, video_paths, labels, transform):
    clip_sampler = UniformClipSampler(clip_duration=duration)
    dataset = LabeledVideoDataset(
        labeled_video_paths=[
            (path, {"label": label, "video_name": path})
            for path, label in zip(video_paths, labels)
        ],
        clip_sampler=clip_sampler,
        transform=transform,
        decode_audio=False,
    )
    return dataset


def get_data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)


def validate(model, val_loader, device, class_dict, thresholds):
    # TRT wrappers expose infer() and have no eval(); PyTorch models have eval()
    is_trt = hasattr(model, 'infer') and not hasattr(model, 'eval')
    if not is_trt:
        model.eval()

    total = 0
    all_labels = []
    all_probs = []
    saved_videos = {
        threshold: set() for threshold in thresholds
    }  # Track saved videos per threshold

    # Create base directories
    base_output_dir = "output_videos"
    os.makedirs(base_output_dir, exist_ok=True)

    # Create threshold-specific drowning directories
    for threshold in thresholds:
        os.makedirs(
            os.path.join(base_output_dir, f"{threshold}/drowning"), exist_ok=True
        )

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            print(f"Processing batch {batch_idx + 1}")
            videos = batch["video"]
            labels = batch["label"].to(device)
            batch_video_paths = batch["video_name"]  # List of video paths for the batch

            #for b in range(videos.size(0)):
            #    X3d.save_video_tensors(videos[b].cpu())
            labels = labels.unsqueeze(1).float()

            if is_trt:
                np_input = videos.cpu().numpy().astype('float16')
                out_np = model.infer(np_input)
                outputs = torch.from_numpy(out_np.copy()).to(device).float()
            else:
                if isinstance(videos, list):
                    videos = [v.to(device) for v in videos]
                else:
                    videos = videos.to(device)
                with autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(videos)

            total += labels.size(0)

            probabilities = torch.sigmoid(outputs)
            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # print(f'{probabilities.cpu().numpy()}')
            # Save videos immediately if classified as drowning
            for i, video_path in enumerate(batch_video_paths):
                video_name = os.path.basename(video_path)
                prob = probabilities[i].cpu().numpy()

                for threshold in thresholds:
                    if video_path not in saved_videos[threshold]:
                        pred = (prob > threshold).astype(int)
                        true_label = labels[i].cpu().numpy()
                        true_label_idx = int(true_label[0]) if getattr(true_label, "__len__", None) else int(true_label)
                        is_false_prediction = int(pred[0]) != true_label_idx

                        if not is_false_prediction:
                            # do not save correct predictions
                            continue

                        # false prediction -> save using predicted class name instead of "drowning"
                        predicted_idx = int(pred[0])
                        predicted_class = class_dict[predicted_idx]

                        print(
                            "Saving video:",
                            video_name,
                            "predicted:",
                            predicted_class,
                            "probability:",
                            float(prob[0]),
                            "true_label:",
                            true_label_idx,
                        )

                        dest_dir = os.path.join(base_output_dir, f"{threshold}/{predicted_class}")
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_path = os.path.join(dest_dir, video_name)
                        try:
                            shutil.copy(video_path, dest_path)
                            saved_videos[threshold].add(video_path)
                        except Exception as e:
                            logger.warning(f"Failed to copy {video_path} to {dest_path}: {e}")
                            dest_path = os.path.join(
                                base_output_dir, f"{threshold}/drowning", video_name
                            )
                            try:
                                shutil.copy(video_path, dest_path)
                                saved_videos[threshold].add(video_path)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to copy {video_path} to {dest_path}: {e}"
                                )

    all_labels = torch.tensor(all_labels).flatten().numpy()
    all_probs = torch.tensor(all_probs).flatten().numpy()
    print(f'Total videos: {total}')
    metrics_by_threshold = {}

    for threshold in thresholds:
        predicted = (all_probs > threshold).astype(int)
        accuracy = (predicted == all_labels.astype(int)).sum() / total * 100

        labels_list = list(range(len(class_dict)))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            precision = precision_score(
                all_labels, predicted, average=None, labels=labels_list, zero_division=0
            )
            precision_overall = precision_score(
                all_labels, predicted, average="weighted", zero_division=0
            )
            recall = recall_score(
                all_labels, predicted, average=None, labels=labels_list, zero_division=0
            )
            recall_overall = recall_score(
                all_labels, predicted, average="weighted", zero_division=0
            )
            f1 = f1_score(
                all_labels, predicted, average=None, labels=labels_list, zero_division=0
            )
            f1_overall = f1_score(
                all_labels, predicted, average="weighted", zero_division=0
            )

        metrics_by_threshold[threshold] = {
            "accuracy": accuracy,
            "precision": precision,
            "precision_overall": precision_overall,
            "recall": recall,
            "recall_overall": recall_overall,
            "f1": f1,
            "f1_overall": f1_overall,
        }

        mlflow.log_metric(f"val_accuracy_{threshold}", accuracy)
        for i, value in enumerate(precision):
            mlflow.log_metric(f"val_precision_{class_dict[i]}_{threshold}", value)
        for i, value in enumerate(recall):
            mlflow.log_metric(f"val_recall_{class_dict[i]}_{threshold}", value)
        for i, value in enumerate(f1):
            mlflow.log_metric(f"val_f1_{class_dict[i]}_{threshold}", value)
        mlflow.log_metric(f"val_f1_overall_{threshold}", f1_overall)

    return metrics_by_threshold


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_variant = args["model_variant"]

    is_docker = os.path.isdir("/models")
    models_dir = "/models" if is_docker else "models"

    if args["model_extension"] == ".engine":
        loader = TensorRTModelLoader(models_dir=models_dir)
    else:
        loader = GenericModelLoader(models_dir=models_dir)

    model = loader.load_action_model(
        model_variant, ["normal", "drowning"], model_extension=args["model_extension"]
    )        

    class_dict = model.class_dict

    val_video_paths, val_labels = load_validation_data(args["dataset_path"], class_dict)
    if not val_video_paths:
        raise ValueError("Validation dataset is empty.")

    val_transform = model.val_transform
    val_dataset = get_val_dataset(
        model.clip_duration, val_video_paths, val_labels, val_transform
    )
    val_loader = get_data_loader(val_dataset, args["batch_size"])

    # TRT wrapper is passed directly; .pth wrapper exposes .model (raw PyTorch model)
    inference_model = model if args["model_extension"] == ".engine" else model.model

    with mlflow.start_run():
        val_metrics = validate(
            inference_model, val_loader, device, class_dict, args["thresholds"]
        )
        for threshold, metric in val_metrics.items():
            logger.info(f"\n--- Threshold {threshold} ---")
            logger.info(f"Accuracy: {metric['accuracy']:.2f}%")
            for i, cls in class_dict.items():
                logger.info(f"Precision [{cls}]: {metric['precision'][i]:.4f}")
                logger.info(f"Recall    [{cls}]: {metric['recall'][i]:.4f}")
                logger.info(f"F1        [{cls}]: {metric['f1'][i]:.4f}")
            logger.info(f"F1 overall: {metric['f1_overall']:.4f}")


if __name__ == "__main__":
    args = parse_env()
    port = args["mlflow_tracking_port"]
    mlflow.set_tracking_uri(f"http://localhost:{port}")
    mlflow.set_experiment("Drowning_X3d_Validation")
    main(args)
