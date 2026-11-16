"""Prediction | ground-truth images logged to MLflow.

- During training: whenever validation produces a new best model, a fixed set of val images is predicted
  with the new best.pt and logged under best_predictions/epoch_XXX/<image>.png
- After training: log_test_results() evaluates best.pt on the dataset's `test` split and logs test metrics,
  plots and the same kind of images under test_predictions/<image>.png

Each image: left = predictions (class name + conf), right = annotations; same color per class in both.
"""
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths
from ultralytics.utils import LOGGER

# BGR, chosen to contrast with the purple (depth) / green (gray) images; same color per class in both panels
# cord=yellow, fabric=cyan, bowl=orange, uchair=white, then red, blue, green for any extra classes
PALETTE = [(0, 255, 255), (255, 255, 0), (0, 128, 255), (255, 255, 255), (0, 0, 255), (255, 64, 0), (0, 255, 0)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_H = 24
GAP = 4


def short_name(name, max_len=8):
    return name if len(name) <= max_len else name[: max_len - 1] + "."


def read_labels(label_path, w, h):
    """YOLO txt (cls xc yc bw bh, normalized) -> list of (cls, x1, y1, x2, y2) in pixels."""
    p = Path(label_path)
    if not p.is_file():
        return []
    boxes = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, xc, yc, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
        boxes.append((c, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
    return boxes


def draw_boxes(img, boxes, names, scale):
    """boxes: list of (cls, x1, y1, x2, y2, conf_or_None) in original-image pixels."""
    for c, x1, y1, x2, y2, conf in boxes:
        color = PALETTE[c % len(PALETTE)]
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(img, p1, p2, color, 2)
        text = short_name(names.get(c, str(c)))
        if conf is not None:
            text += f" {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
        tx = min(max(p1[0], 0), img.shape[1] - tw - 4)
        ty = p1[1] - 4 if p1[1] - th - 6 >= 0 else p1[1] + th + 6  # above the box, or inside if at top edge
        cv2.rectangle(img, (tx, ty - th - 3), (tx + tw + 4, ty + 3), color, -1)
        cv2.putText(img, text, (tx + 2, ty), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def with_header(img, title):
    header = np.full((HEADER_H, img.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(header, title, (6, HEADER_H - 7), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, img])


def render_pair(im_file, result, names, conf, max_side=960):
    img = cv2.imread(im_file)
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side < 480:  # small ToF frames: integer upscale so labels are readable
        scale = 480 // long_side
        base = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    else:  # large camera images: display at (at most) the training resolution; only affects the logged pictures
        scale = min(1.0, max_side / long_side)
        base = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)

    b = result.boxes
    preds = [(int(c), *xyxy, float(s)) for c, xyxy, s in zip(b.cls.tolist(), b.xyxy.tolist(), b.conf.tolist())]
    gts = [(*box, None) for box in read_labels(img2label_paths([im_file])[0], w, h)]

    left = with_header(draw_boxes(base.copy(), preds, names, scale), f"Prediction ({len(preds)}, conf>={conf})")
    right = with_header(draw_boxes(base.copy(), gts, names, scale), f"Ground truth ({len(gts)})")
    gap = np.full((left.shape[0], GAP, 3), 255, dtype=np.uint8)
    return np.hstack([left, gap, right])


def sample_files(files, num_images, seed=0):
    files = sorted(files)
    return random.Random(seed).sample(files, min(num_images, len(files)))


def list_split_images(split):
    """Image paths of a dataset split as resolved by check_det_dataset: .txt list, directory, or a list of those."""
    files = []
    for s in split if isinstance(split, list) else [split]:
        p = Path(s)
        if p.is_dir():
            files += [str(f) for f in p.rglob("*.*") if f.suffix[1:].lower() in IMG_FORMATS]
        elif p.suffix == ".txt":
            lines = [x.strip() for x in p.read_text().splitlines() if x.strip()]
            files += [x if Path(x).is_absolute() else str((p.parent / x).resolve()) for x in lines]
    return files


def log_prediction_images(model, files, artifact_path, conf, imgsz, device=None):
    """Predict `files` with `model` and log prediction | ground-truth PNGs to the active MLflow run."""
    import mlflow

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # training keeps a large cache; 1280px fp32 inference otherwise OOMs
    with tempfile.TemporaryDirectory() as tmp:
        for im_file in files:  # one image at a time: a list source is batched and can OOM at large imgsz
            r = model.predict(im_file, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
            pair = render_pair(im_file, r, model.names, conf, max_side=max(960, imgsz))
            # PNG for small ToF frames; JPEG for large camera images (PNG would be several MB each)
            if pair.shape[1] <= 1000:
                cv2.imwrite(str(Path(tmp) / f"{Path(im_file).stem}.png"), pair)
            else:
                cv2.imwrite(str(Path(tmp) / f"{Path(im_file).stem}.jpg"), pair, [cv2.IMWRITE_JPEG_QUALITY, 90])
        mlflow.log_artifacts(tmp, artifact_path=artifact_path)


class BestPredictionLogger:
    """on_fit_epoch_end callback. The same val images are used every time so epochs are comparable.
    Also remembers the MLflow run id so log_test_results() can reopen the run after training."""

    def __init__(self, num_images=100, conf=0.25, seed=0):
        self.num_images, self.conf, self.seed = num_images, conf, seed
        self.files = None
        self.logged_fitness = None
        self.run_id = None

    def on_fit_epoch_end(self, trainer):
        try:
            import mlflow

            if mlflow.active_run() is None:
                return
            self.run_id = mlflow.active_run().info.run_id
            if self.num_images <= 0 or trainer.fitness is None or trainer.best_fitness != trainer.fitness:
                return  # not a new best
            if trainer.best_fitness == self.logged_fitness:
                return  # already logged (final_eval re-fires this callback for the same best.pt)
            if self.files is None:
                self.files = sample_files(trainer.test_loader.dataset.im_files, self.num_images, self.seed)

            epoch = trainer.epoch + 1  # 1-based, matches results.csv
            model = YOLO(str(trainer.best))  # best.pt was just written by save_model()
            log_prediction_images(model, self.files, f"best_predictions/epoch_{epoch:03d}", self.conf,
                                  trainer.args.imgsz, trainer.device)
            self.logged_fitness = trainer.best_fitness
            mlflow.set_tag("best_epoch", epoch)
            LOGGER.info(f"MLflow: logged {len(self.files)} best-model prediction images (epoch {epoch})")
        except Exception as e:  # never break training because of visualization
            LOGGER.warning(f"MLflow: failed to log best-model predictions: {e}")


def log_test_results(best_pt, data_yaml, save_dir, run_id=None, num_images=100, conf=0.25, imgsz=224, seed=0):
    """Evaluate best.pt on the dataset's `test` split; log metrics, plots and prediction images to MLflow run `run_id`."""
    data = check_det_dataset(data_yaml)
    if not data.get("test"):
        LOGGER.warning(f"No 'test' split in {data_yaml}; skipping test evaluation")
        return None

    save_dir = Path(save_dir)
    model = YOLO(str(best_pt))
    metrics = model.val(data=data_yaml, split="test", imgsz=imgsz, project=str(save_dir.parent),
                        name=f"{save_dir.name}_test", exist_ok=True)
    if not run_id:
        return metrics

    import mlflow

    with mlflow.start_run(run_id=run_id):
        clean = {f"test/{k.replace('(', '').replace(')', '')}": float(v) for k, v in metrics.results_dict.items()}
        mlflow.log_metrics(clean)
        for f in Path(metrics.save_dir).glob("*.png"):  # confusion matrix, PR curves, ...
            mlflow.log_artifact(str(f), artifact_path="test_eval")
        if num_images > 0:
            files = sample_files(list_split_images(data["test"]), num_images, seed)
            log_prediction_images(model, files, "test_predictions", conf, imgsz)
            LOGGER.info(f"MLflow: logged test metrics and {len(files)} test prediction images")
    return metrics
