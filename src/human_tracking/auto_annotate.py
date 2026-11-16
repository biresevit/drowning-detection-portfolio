"""
auto_annotate.py

Runs the head-visibility classifier over all unannotated crop images in a
batch directory and writes results into the dataset/ subfolder that
classify_human_ui.py uses.  The human then only needs to do QA.

Supports both:
  .onnx   — ONNX Runtime (Windows / CPU / GPU)
  .engine — TensorRT     (Jetson)

Model is read from HEAD_MODEL_PATH in .env (or --model).
Image size is read from HEAD_VISIBILITY_IMG_SIZE in .env (or --img_size).

Output layout (identical to classify_human_ui.py):
    input_data/
        dataset/
            head_and_shoulders_visible/
                <location>/<video>/<frame>/<crop>.png
                <location>/<video>/<frame>/<crop>_inspect.jpg
            head_clearly_visible/ ...
            head_horizontal_or_under_water/ ...

Already-annotated images are skipped so the script is safe to re-run.
A confidence CSV is written to input_data/dataset/auto_annotations.csv.

Usage:
    python auto_annotate.py
    python auto_annotate.py --input_data path/to/batch_1 --model models/head_visibility.onnx
"""

import argparse
import csv
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    from human_tracking.head_visibility_classes import INFERENCE_CLASS_NAMES
except ImportError:
    from head_visibility_classes import INFERENCE_CLASS_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES   = INFERENCE_CLASS_NAMES           # alphabetically sorted, matches model output

SKIP_DIRS = {"dataset", "dataset_classification", "classification_eval"}


def _ensure_cuda_dlls_on_path():
    """onnxruntime-gpu's CUDA provider needs cuDNN/cuBLAS DLLs on PATH (classic
    LoadLibrary search — os.add_dll_directory is not honored by onnxruntime's
    native loader). Reuses PyTorch's own bundled cuDNN/cuBLAS in torch/lib —
    NOT a separately pip-installed cudnn package, which would be a different
    version and cause DLL export mismatches (WinError 127) when torch loads.
    """
    if os.name != "nt":
        return
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib) and torch_lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")


# ----------------------------------------------------------------- inference backends

class OnnxBackend:
    def __init__(self, model_path: str, img_size: int):
        _ensure_cuda_dlls_on_path()
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session  = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.img_size   = img_size
        active = self.session.get_providers()[0]
        logger.info("ONNX Runtime loaded: %s  provider=%s  img_size=%d", model_path, active, img_size)

    def predict_probs(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        batch = _preprocess(crops_bgr, self.img_size)
        logits = self.session.run(None, {self.input_name: batch})[0].astype(np.float32)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


class TrtBackend:
    def __init__(self, model_path: str, img_size: int, max_batch_size: int):
        try:
            from human_tracking.inference_classification_tensorrt import HeadVisibilityTRT
        except ImportError:
            from inference_classification_tensorrt import HeadVisibilityTRT
        self._model   = HeadVisibilityTRT(model_path, max_batch_size=max_batch_size, img_size=img_size)
        logger.info("TensorRT backend loaded: %s", model_path)

    def predict_probs(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        return self._model.predict_probs(crops_bgr)


def _load_backend(model_path: str, img_size: int, max_batch_size: int):
    ext = Path(model_path).suffix.lower()
    if ext == ".onnx":
        return OnnxBackend(model_path, img_size)
    elif ext == ".engine":
        return TrtBackend(model_path, img_size, max_batch_size)
    else:
        raise ValueError(f"Unsupported model extension '{ext}'. Use .onnx or .engine.")


# ----------------------------------------------------------------- preprocessing

def _preprocess(crops_bgr: list[np.ndarray], img_size: int) -> np.ndarray:
    batch = []
    for crop in crops_bgr:
        img = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        batch.append(img.transpose(2, 0, 1))
    return np.stack(batch, axis=0).astype(np.float32)


# ----------------------------------------------------------------- file helpers

def find_crops(input_dir: Path) -> list[Path]:
    skip = {input_dir / d for d in SKIP_DIRS}
    return sorted(
        p for p in input_dir.rglob("*.png")
        if not p.stem.endswith("_inspect")
        and not any(s in p.parents for s in skip)
        and not any(part.startswith("batch_") for part in p.relative_to(input_dir).parts)
    )


def already_annotated(dataset_dir: Path) -> set[str]:
    """Return set of relative paths already present in any class folder."""
    annotated = set()
    for cls in CLASS_NAMES:
        cls_dir = dataset_dir / cls
        if not cls_dir.exists():
            continue
        for p in cls_dir.rglob("*.png"):
            if not p.stem.endswith("_inspect"):
                annotated.add(str(p.relative_to(cls_dir)))
    return annotated


def copy_to_dataset(src: Path, dataset_dir: Path, cls_name: str, rel: Path) -> None:
    dst = dataset_dir / cls_name / rel.parent / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    inspect = src.with_name(src.stem + "_inspect.jpg")
    if inspect.exists():
        shutil.copy2(inspect, dst.with_name(inspect.name))


# ----------------------------------------------------------------- main logic

def auto_annotate(input_dir: Path, model_path: str, img_size: int,
                  batch_size: int, max_batch_size: int) -> None:
    dataset_dir = input_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    crops = find_crops(input_dir)
    if not crops:
        logger.error("No crop images found in %s", input_dir)
        return

    done = already_annotated(dataset_dir)
    pending = [c for c in crops if str(c.relative_to(input_dir)) not in done]

    logger.info("Total crops: %d | already annotated: %d | pending: %d",
                len(crops), len(done), len(pending))
    if not pending:
        logger.info("Nothing to annotate.")
        return

    backend = _load_backend(model_path, img_size, max_batch_size)

    csv_path = dataset_dir / "auto_annotations.csv"
    csv_exists = csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not csv_exists:
        writer.writerow(["rel_path", "predicted_class", "confidence"])

    total = len(pending)
    annotated_count = 0
    class_counts = {c: 0 for c in CLASS_NAMES}

    for start in range(0, total, batch_size):
        chunk = pending[start : start + batch_size]
        images = []
        for p in chunk:
            img = cv2.imread(str(p))
            if img is None:
                logger.warning("Could not read %s — skipping", p)
                images.append(None)
            else:
                images.append(img)

        valid_idx = [i for i, img in enumerate(images) if img is not None]
        valid_imgs = [images[i] for i in valid_idx]

        if not valid_imgs:
            continue

        probs = backend.predict_probs(valid_imgs)

        for local_i, global_i in enumerate(valid_idx):
            crop_path  = chunk[global_i]
            cls_idx    = int(probs[local_i].argmax())
            cls_name   = CLASS_NAMES[cls_idx]
            confidence = float(probs[local_i, cls_idx])
            rel        = crop_path.relative_to(input_dir)

            copy_to_dataset(crop_path, dataset_dir, cls_name, rel)
            writer.writerow([str(rel), cls_name, f"{confidence:.4f}"])
            class_counts[cls_name] += 1
            annotated_count += 1

        if annotated_count % 500 == 0 or start + batch_size >= total:
            csv_file.flush()
            logger.info("Progress: %d / %d", annotated_count, total)

    csv_file.close()

    logger.info("Done — %d crops annotated.", annotated_count)
    for cls, count in class_counts.items():
        logger.info("  %-40s %d", cls, count)
    logger.info("Confidence CSV: %s", csv_path)


# ----------------------------------------------------------------- entry point

def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-annotate crop images using head-visibility classifier.")
    parser.add_argument(
        "--input_data",
        default=r"path/to/data",
        help="Batch directory to annotate.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("HEAD_MODEL_PATH", "models/head_visibility.onnx"),
        help="Path to .onnx or .engine model file.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=int(os.getenv("HEAD_VISIBILITY_IMG_SIZE", "224")),
        help="Input resolution expected by the model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of crops to process per inference call.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=int(os.getenv("HEAD_MODEL_MAX_BATCH_SIZE", "128")),
        help="TensorRT max batch size (ignored for ONNX).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_data)
    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        return

    model_path = args.model
    if not Path(model_path).is_file():
        logger.error("Model file not found: %s", model_path)
        return

    logger.info("Input:   %s", input_dir)
    logger.info("Model:   %s", model_path)
    logger.info("ImgSize: %d", args.img_size)

    auto_annotate(input_dir, model_path, args.img_size, args.batch_size, args.max_batch_size)


if __name__ == "__main__":
    main()
