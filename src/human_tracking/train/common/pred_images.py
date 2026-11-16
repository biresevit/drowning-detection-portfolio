"""Prediction | ground-truth side-by-side images, framework-neutral (same look as the YOLO callback's).

Left = predictions (class name + conf), right = the YOLO labels; same colour per class in both.
"""
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np

from size_metrics_core import label_path, read_labels

PALETTE = [(0, 255, 255), (255, 255, 0), (0, 128, 255), (255, 255, 255), (0, 0, 255), (255, 64, 0), (0, 255, 0)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_H = 24
GAP = 4


def draw_boxes(img, boxes, names, scale):
    """boxes: list of (cls, x1, y1, x2, y2, conf_or_None) in original-image pixels."""
    for c, x1, y1, x2, y2, conf in boxes:
        color = PALETTE[c % len(PALETTE)]
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(img, p1, p2, color, 2)
        name = names.get(c, str(c))
        text = (name if len(name) <= 8 else name[:7] + ".") + (f" {conf:.2f}" if conf is not None else "")
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
        tx = min(max(p1[0], 0), img.shape[1] - tw - 4)
        ty = p1[1] - 4 if p1[1] - th - 6 >= 0 else p1[1] + th + 6
        cv2.rectangle(img, (tx, ty - th - 3), (tx + tw + 4, ty + 3), color, -1)
        cv2.putText(img, text, (tx + 2, ty), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def with_header(img, title):
    header = np.full((HEADER_H, img.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(header, title, (6, HEADER_H - 7), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, img])


def render_pair(im_file, preds, names, conf, max_side=960):
    """preds: list of (cls, x1, y1, x2, y2, conf) already filtered at `conf`."""
    img = cv2.imread(str(im_file))
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    base = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    gts = [(*box, None) for box in read_labels(label_path(im_file), w, h)]
    left = with_header(draw_boxes(base.copy(), preds, names, scale), f"Prediction ({len(preds)}, conf>={conf})")
    right = with_header(draw_boxes(base.copy(), gts, names, scale), f"Ground truth ({len(gts)})")
    return np.hstack([left, np.full((left.shape[0], GAP, 3), 255, dtype=np.uint8), right])


def sample_files(files, num_images, seed=0):
    files = sorted(files)
    return random.Random(seed).sample(files, min(num_images, len(files)))


def log_prediction_images(predict, files, names, artifact_path, conf, max_side=1920, run_id=None):
    """predict(im_file) -> (xyxy, conf, cls, (h, w)); logs one JPEG per file to MLflow run `run_id`.

    Through MlflowClient, NOT mlflow.start_run(run_id=...): leaving a start_run block ends the run, which marks a
    run that is still training as FINISHED. Without run_id the active run is used."""
    import mlflow

    with tempfile.TemporaryDirectory() as tmp:
        for im_file in files:
            xyxy, cf, cl, _ = predict(im_file)
            keep = np.asarray(cf) >= conf
            preds = [(int(c), *map(float, b), float(s)) for b, s, c in
                     zip(np.asarray(xyxy)[keep], np.asarray(cf)[keep], np.asarray(cl)[keep])]
            pair = render_pair(im_file, preds, names, conf, max_side=max_side)
            cv2.imwrite(str(Path(tmp) / f"{Path(im_file).stem}.jpg"), pair, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if run_id:
            mlflow.MlflowClient().log_artifacts(run_id, tmp, artifact_path=artifact_path)
        else:
            mlflow.log_artifacts(tmp, artifact_path=artifact_path)
