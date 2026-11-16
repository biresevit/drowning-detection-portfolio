"""
evaluate_test.py

Runs the head-visibility TensorRT classifier on the test split and saves
each image into correct/ or incorrect/ subfolders with confidence in the
filename.

Output layout:
    save_dir/
        correct/
            {true_class}/
                0000_0.93.png
                ...
        incorrect/
            {true_class}/{predicted_class}/
                0000_0.71.png
                ...

Usage:
    python src/human_tracking/evaluate_test.py
    python src/human_tracking/evaluate_test.py \\
        --engine   models/head_visibility.engine \\
        --img_size 112
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from PIL import Image

try:
    from human_tracking.inference_classification_tensorrt import HeadVisibilityTRT, CLASS_NAMES
except ImportError:
    from inference_classification_tensorrt import HeadVisibilityTRT, CLASS_NAMES

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _load_bgr(path: Path) -> np.ndarray:
    """Load image as BGR uint8 numpy array (as expected by HeadVisibilityTRT)."""
    img_rgb = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    return img_rgb[:, :, ::-1].copy()   # RGB -> BGR


def evaluate(
    input_test_dir: Path,
    save_dir: Path,
    engine_path: Path,
    img_size: int,
    max_batch_size: int,
) -> None:
    # Discover classes from subfolder names (alphabetical = engine output order)
    class_dirs = sorted(d for d in input_test_dir.iterdir() if d.is_dir())
    if not class_dirs:
        logger.error("No class subdirectories found in %s", input_test_dir)
        return

    class_names = [d.name for d in class_dirs]
    num_classes  = len(class_names)
    logger.info("Test classes (%d): %s", num_classes, class_names)
    logger.info("Engine class order: %s", CLASS_NAMES)

    # Validate that engine classes match test classes
    if sorted(class_names) != sorted(CLASS_NAMES):
        logger.warning(
            "Test classes %s do not match engine classes %s — results may be misleading.",
            class_names, CLASS_NAMES,
        )

    # Load TRT engine
    try:
        model = HeadVisibilityTRT(
            engine_path    = str(engine_path),
            max_batch_size = max_batch_size,
            img_size       = img_size,
        )
    except Exception:
        logger.exception("Failed to load TRT engine from %s — aborting.", engine_path)
        sys.exit(1)

    # Clear and recreate output dir
    if save_dir.exists():
        shutil.rmtree(save_dir)

    conf = np.zeros((num_classes, num_classes), dtype=int)

    # Collect all test images
    all_images:     list[Path] = []
    all_true_names: list[str]  = []
    for cls_dir in class_dirs:
        for p in sorted(cls_dir.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS and not p.stem.endswith("_inspect"):
                all_images.append(p)
                all_true_names.append(cls_dir.name)

    total_images = len(all_images)
    logger.info("Total test images: %d", total_images)

    # Process in batches
    for batch_start in range(0, total_images, max_batch_size):
        batch_paths = all_images[batch_start : batch_start + max_batch_size]
        batch_true  = all_true_names[batch_start : batch_start + max_batch_size]

        crops_bgr:    list[np.ndarray] = []
        valid_paths:  list[Path]       = []
        valid_true:   list[str]        = []

        for p, true_cls in zip(batch_paths, batch_true):
            try:
                crops_bgr.append(_load_bgr(p))
                valid_paths.append(p)
                valid_true.append(true_cls)
            except Exception as exc:
                logger.warning("Skipping %s: %s", p, exc)

        if not crops_bgr:
            continue

        probs = model.predict_probs(crops_bgr)   # (N, num_classes)

        for p, true_cls, prob_row in zip(valid_paths, valid_true, probs):
            pred_idx = int(prob_row.argmax())
            pred_cls = CLASS_NAMES[pred_idx]

            # Update confusion matrix (engine class order)
            if true_cls in CLASS_NAMES:
                true_idx = CLASS_NAMES.index(true_cls)
                conf[true_idx][pred_idx] += 1

            correct = (pred_cls == true_cls)
            if correct:
                out_dir = save_dir / "correct" / true_cls
            else:
                out_dir = save_dir / "incorrect" / true_cls / pred_cls

            out_dir.mkdir(parents=True, exist_ok=True)
            Image.open(p).convert("RGB").save(out_dir / p.name)

        done = min(batch_start + max_batch_size, total_images)
        logger.info("Progress: %d / %d", done, total_images)

    # Print summary
    total   = int(conf.sum())
    correct = int(conf.diagonal().sum())
    logger.info("")
    logger.info("=== Test Results ===")
    logger.info("Overall accuracy: %d / %d  = %.2f%%", correct, total, 100 * correct / max(total, 1))
    logger.info("")

    col_w = max(len(c) for c in CLASS_NAMES) + 2
    header = f"{'True / Predicted':<{col_w}}" + "".join(f"{c:<{col_w}}" for c in CLASS_NAMES)
    logger.info(header)
    for i, true_cls in enumerate(CLASS_NAMES):
        row_n = conf[i].sum()
        row   = f"{true_cls:<{col_w}}"
        for j in range(num_classes):
            pct = 100 * conf[i][j] / max(row_n, 1)
            row += f"{conf[i][j]:>4}({pct:4.0f}%)  ".ljust(col_w)
        logger.info(row)

    logger.info("")
    logger.info("Per-class accuracy:")
    for i, cls in enumerate(CLASS_NAMES):
        n = conf[i].sum()
        logger.info("  %-40s  %d / %d  = %.2f%%", cls, conf[i][i], n, 100 * conf[i][i] / max(n, 1))

    logger.info("")
    logger.info("Results saved to: %s", save_dir)


# ----------------------------------------------------------------- entry point

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate head-visibility TRT engine on test split.")
    parser.add_argument(
        "--input_test_dir",
        default=r"path/to/data",
    )
    parser.add_argument(
        "--save_dir",
        default=r"path/to/data",
    )
    parser.add_argument("--engine",         default=os.getenv("HEAD_MODEL_PATH", "models/head_visibility.engine"))
    parser.add_argument("--img_size",       type=int, default=int(os.getenv("HEAD_VISIBILITY_IMG_SIZE", "224")))
    parser.add_argument("--max_batch_size", type=int, default=int(os.getenv("HEAD_MODEL_MAX_BATCH_SIZE", "64")))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    evaluate(
        input_test_dir = Path(args.input_test_dir),
        save_dir       = Path(args.save_dir),
        engine_path    = Path(args.engine),
        img_size       = args.img_size,
        max_batch_size = args.max_batch_size,
    )


if __name__ == "__main__":
    main()
