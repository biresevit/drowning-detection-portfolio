"""
train_classification.py

Trains an image classifier on the head-visibility dataset.

Expects dataset split by split_classification_dataset.py:
    input_dataset/
        train/{class}/  val/{class}/  test/{class}/

Available models (--model):
    efficientnet_v2_s   — SOTA accuracy, fast inference
    efficientnet_v2_m   — higher accuracy, slower
    convnext_small      — ConvNeXt Small
    resnet50            — ResNet-50 baseline
    vit_b_16            — Vision Transformer B/16
    mobilenet_v3_large  — fastest inference (~17x vs ResNet50 at 112px), good for Jetson

Start MLflow server first:
    mlflow server --host localhost --port 5001 --backend-store-uri ./mlflow

Usage:
    python train_classification.py
    python train_classification.py --model convnext_small --epochs 50 --batch_size 64
"""

import argparse
import logging
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from PIL import Image, UnidentifiedImageError
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import amp
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------- model factory

def build_model(model_name: str, num_classes: int) -> nn.Module:
    name = model_name.lower()
    if name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "efficientnet_v2_m":
        m = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "convnext_small":
        m = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Choose: efficientnet_v2_s, efficientnet_v2_m, convnext_small, resnet50, vit_b_16, mobilenet_v3_large"
        )
    return m


# ----------------------------------------------------------------- transforms

def _train_tf(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _val_tf(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ----------------------------------------------------------------- data loaders

def _drop_corrupt_samples(ds: datasets.ImageFolder, split: str) -> None:
    """Remove unreadable images (e.g. zero-byte files from an interrupted copy)."""
    valid = []
    for path, label in ds.samples:
        try:
            with Image.open(path) as img:
                img.verify()
            valid.append((path, label))
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            logger.warning("Skipping corrupt image in %s split: %s (%s)", split, path, e)
    if len(valid) != len(ds.samples):
        logger.warning("%s: dropped %d corrupt image(s), %d remain",
                       split, len(ds.samples) - len(valid), len(valid))
    ds.samples = valid
    ds.targets = [label for _, label in valid]


def build_loaders(
    dataset_dir: Path, batch_size: int, num_workers: int, img_size: int = 224,
    check_corrupt: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_ds = datasets.ImageFolder(str(dataset_dir / "train"), transform=_train_tf(img_size))
    val_ds   = datasets.ImageFolder(str(dataset_dir / "val"),   transform=_val_tf(img_size))
    test_ds  = datasets.ImageFolder(str(dataset_dir / "test"),  transform=_val_tf(img_size))

    if check_corrupt:
        _drop_corrupt_samples(train_ds, "train")
        _drop_corrupt_samples(val_ds,   "val")
        _drop_corrupt_samples(test_ds,  "test")

    # Class imbalance is handled once, via the weighted loss in main() — a
    # WeightedRandomSampler here would double-compensate on top of that,
    # since it already flattens each batch to a near-uniform class mix.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)

    return train_loader, val_loader, test_loader, train_ds.classes


# ----------------------------------------------------------------- train / eval

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    epoch: int,
) -> float:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with amp.autocast(device_type=device.type, enabled=use_amp):
            out  = model(imgs)
            loss = criterion(out, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        preds = out.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total * 100.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    mlflow.log_metric("train_loss",     avg_loss, step=epoch)
    mlflow.log_metric("train_accuracy", accuracy, step=epoch)
    mlflow.log_metric("train_f1",       f1,       step=epoch)
    logger.info("Epoch %3d  train  loss=%.4f  acc=%.2f%%  f1=%.4f", epoch + 1, avg_loss, accuracy, f1)
    return avg_loss


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    class_names: list[str],
    epoch: int,
    split: str = "val",
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with amp.autocast(device_type=device.type, enabled=use_amp):
                out  = model(imgs)
                loss = criterion(out, labels)
            preds = out.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss  = total_loss / total
    accuracy  = correct / total * 100.0
    label_ids = list(range(len(class_names)))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision = precision_score(all_labels, all_preds, average=None,       labels=label_ids, zero_division=0)
        recall    = recall_score   (all_labels, all_preds, average=None,       labels=label_ids, zero_division=0)
        f1        = f1_score       (all_labels, all_preds, average=None,       labels=label_ids, zero_division=0)
        f1_w      = f1_score       (all_labels, all_preds, average="weighted", labels=label_ids, zero_division=0)

    mlflow.log_metric(f"{split}_loss",     avg_loss, step=epoch)
    mlflow.log_metric(f"{split}_accuracy", accuracy, step=epoch)
    mlflow.log_metric(f"{split}_f1",       f1_w,     step=epoch)
    for i, cls in enumerate(class_names):
        mlflow.log_metric(f"{split}_precision_{cls}", float(precision[i]), step=epoch)
        mlflow.log_metric(f"{split}_recall_{cls}",    float(recall[i]),    step=epoch)
        mlflow.log_metric(f"{split}_f1_{cls}",        float(f1[i]),        step=epoch)

    logger.info(
        "         %s  loss=%.4f  acc=%.2f%%  f1=%.4f", split, avg_loss, accuracy, f1_w
    )
    for i, cls in enumerate(class_names):
        logger.info("           %-38s  P=%.3f  R=%.3f  F1=%.3f", cls, precision[i], recall[i], f1[i])

    return avg_loss, accuracy, f1_w


# ----------------------------------------------------------------- entry point

def main() -> None:
    parser = argparse.ArgumentParser(description="Train head-visibility image classifier.")
    parser.add_argument(
        "--input_dataset",
        default=r"path/to/data",
        help="Root directory containing train/val/test splits.",
    )
    parser.add_argument(
        "--model",
        default="efficientnet_v2_s",
        choices=["efficientnet_v2_s", "efficientnet_v2_m", "convnext_small", "resnet50", "vit_b_16", "mobilenet_v3_large"],
        help="Backbone architecture.",
    )
    parser.add_argument("--epochs",          type=int,   default=15,
                        help="Also the length of the cosine LR schedule, so the LR reaches ~0 at the end.")
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--patience",        type=int,   default=4,
                        help="Early-stopping patience in epochs.")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing for the loss. The class borders are fuzzy, so hard "
                             "labels push the model to be overconfident on ambiguous crops.")
    parser.add_argument("--output_dir",      default="models",
                        help="Directory for saved model checkpoints.")
    parser.add_argument("--img_size",        type=int,
                        default=int(os.getenv("HEAD_VISIBILITY_IMG_SIZE", "224")),
                        help="Input image size for training (e.g. 112, 160, 224). "
                             "Defaults to HEAD_VISIBILITY_IMG_SIZE from .env. "
                             "Note: vit_b_16 requires 224.")
    parser.add_argument("--num_workers",     type=int,   default=4,
                        help="DataLoader workers. Each one imports torch in its own process, so "
                             "high counts can exhaust the Windows paging file (WinError 1455).")
    parser.add_argument("--mlflow_port",     type=int,   default=5001)
    parser.add_argument("--experiment_name", default="HeadVisibilityClassifier")
    parser.add_argument("--skip_corrupt_check", action="store_true",
                        help="Skip the startup pass that verifies every image opens cleanly. "
                             "Safe to use once you've already run remove_corrupt_images.py "
                             "on this dataset — saves a full I/O pass over every file at startup.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    logger.info("Device: %s  AMP: %s", device, use_amp)

    dataset_dir = Path(args.input_dataset)
    train_loader, val_loader, test_loader, class_names = build_loaders(
        dataset_dir, args.batch_size, args.num_workers, args.img_size,
        check_corrupt=not args.skip_corrupt_check,
    )
    num_classes = len(class_names)
    logger.info("Classes (%d): %s", num_classes, class_names)
    logger.info("Samples — train: %d  val: %d  test: %d",
                len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    model = build_model(args.model, num_classes).to(device)
    logger.info("Model: %s", args.model)

    # Class-weighted loss to handle imbalanced data.
    # minlength=num_classes guards against a class having zero samples after
    # corrupt-file filtering — without it, bincount silently returns a shorter
    # array and CrossEntropyLoss(weight=...) crashes on the size mismatch.
    counts        = np.bincount([lbl for _, lbl in train_loader.dataset.samples], minlength=num_classes)
    class_weights = torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler(enabled=use_amp)

    output_dir      = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / f"best_{args.model}_head_visibility.pth"

    mlflow.set_tracking_uri(f"http://localhost:{args.mlflow_port}")
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        mlflow.log_params({
            "model":          args.model,
            "epochs":         args.epochs,
            "batch_size":     args.batch_size,
            "lr":             args.lr,
            "weight_decay":   args.weight_decay,
            "patience":       args.patience,
            "label_smoothing": args.label_smoothing,
            "num_classes":    num_classes,
            "classes":        ",".join(class_names),
            "train_samples":  len(train_loader.dataset),
            "val_samples":    len(val_loader.dataset),
            "test_samples":   len(test_loader.dataset),
            "img_size":       args.img_size,
            "device":         device.type,
        })

        best_val_f1       = 0.0
        epochs_no_improve = 0
        last_epoch        = 0

        for epoch in range(args.epochs):
            last_epoch = epoch
            train_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp, epoch)
            val_loss, val_acc, val_f1 = eval_epoch(
                model, val_loader, criterion, device, use_amp, class_names, epoch, "val"
            )
            scheduler.step()
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

            if val_f1 > best_val_f1:
                best_val_f1       = val_f1
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_artifact(str(best_model_path))
                mlflow.log_metric("best_val_f1",       best_val_f1, step=epoch)
                mlflow.log_metric("best_val_loss",     val_loss,    step=epoch)
                mlflow.log_metric("best_val_accuracy", val_acc,     step=epoch)
                logger.info("  -> Best model saved (val_f1=%.4f  acc=%.2f%%  loss=%.4f)",
                            best_val_f1, val_acc, val_loss)
            else:
                epochs_no_improve += 1
                logger.info("  No improvement %d/%d", epochs_no_improve, args.patience)

            if epochs_no_improve >= args.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        # Test evaluation using best checkpoint
        logger.info("--- Test evaluation with best model ---")
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        eval_epoch(model, test_loader, criterion, device, use_amp, class_names, last_epoch, "test")


if __name__ == "__main__":
    main()
