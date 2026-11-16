"""Predict images with a trained RF-DETR checkpoint and write the prediction cache analyse_predictions_ui reads.

Runs in its OWN environment (rfdetr_env), not the app's myenv: rfdetr pulls in PyAV 18, numpy 2.2 and
transformers 5, and the app's video decoding uses PyAV 14. analyse_predictions_ui starts this as a
subprocess and reads the JSON it writes - the same format as its YOLO path:

    {"model": ..., "detector": "rfdetr", "imgsz": <resolution>, "conf": ..., "nms_iou": null, "created": ...,
     "images": {"<file name>": [[x1, y1, x2, y2, conf], ...], ...}}

RF-DETR is NMS-free (one query per object), hence nms_iou null. Progress goes to stdout as
"PROGRESS <done> <total>" lines.

Create the environment once, from the repo root (Windows; torch must be >= the version transformers 5
supports - 2.5 is too old - so it does NOT match myenv's torch, which is the point of a separate env):
    py -3.11 -m venv rfdetr_env
    rfdetr_env\\Scripts\\python -m pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu130
    rfdetr_env\\Scripts\\python -m pip install "rfdetr[plus]==1.11.2"

    rfdetr_env/Scripts/python.exe src/human_tracking/rfdetr_predict.py --weights models/rfdetr_xlarge_1220.pth \
        --images <dir or .txt list> --out predictions.json [--conf 0.05]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

IMG_EXT = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def model_class(weights):
    """The RF-DETR variant class for a checkpoint: from its stored model name, else from the file name."""
    import torch
    import rfdetr

    classes = [rfdetr.RFDETRNano, rfdetr.RFDETRSmall, rfdetr.RFDETRMedium, rfdetr.RFDETRLarge]
    try:
        import rfdetr_plus  # XL / 2XL (PML 1.0 licence)

        classes += [rfdetr_plus.RFDETRXLarge, rfdetr_plus.RFDETR2XLarge]
    except ImportError:
        pass
    stored = torch.load(weights, map_location="cpu", weights_only=False).get("model_name")
    for cls in classes:
        if stored and getattr(cls, "size", None) == stored:
            return cls
    name = Path(weights).stem.lower()  # rfdetr_<variant>_<resolution>.pth, as models/ names them
    for key, cls in (("2xlarge", classes[-1]), ("xlarge", classes[-2] if len(classes) > 4 else None),
                     ("large", rfdetr.RFDETRLarge), ("medium", rfdetr.RFDETRMedium), ("small", rfdetr.RFDETRSmall),
                     ("nano", rfdetr.RFDETRNano)):
        if cls is not None and re.search(rf"(^|_){key}(_|$)", name):
            return cls
    raise SystemExit(f"cannot tell the RF-DETR variant of {weights} (stored model_name={stored!r})")


def list_images(src):
    p = Path(src)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
    return [Path(line.strip()) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="checkpoint_best_total.pth of a train_rfdetr.py run")
    ap.add_argument("--images", required=True, help="image folder, or a .txt file with one image path per line")
    ap.add_argument("--out", required=True, help="prediction cache JSON to write")
    ap.add_argument("--conf", type=float, default=0.05, help="lowest confidence kept in the cache")
    ap.add_argument("--half", action="store_true", help="fp16 inference (faster, scores differ slightly)")
    args = ap.parse_args()

    from PIL import Image
    import numpy as np
    import torch

    images = list_images(args.images)
    print(f"PROGRESS 0 {len(images)}", flush=True)
    model = model_class(args.weights).from_checkpoint(args.weights, trust_checkpoint=True)
    if args.half:
        model.inference(compile=False, dtype=torch.float16)
    resolution = getattr(getattr(model, "model_config", None), "resolution", None)

    out = {}
    for n, path in enumerate(images, 1):
        with Image.open(path) as im:
            img = im.convert("RGB")
        det = model.predict(img, threshold=args.conf)
        xyxy = np.asarray(det.xyxy, dtype=np.float32).reshape(-1, 4)
        conf = np.asarray(det.confidence, dtype=np.float32).reshape(-1)
        order = np.argsort(-conf, kind="stable")
        out[path.name] = [[round(float(v), 1) for v in xyxy[i]] + [round(float(conf[i]), 4)] for i in order]
        print(f"PROGRESS {n} {len(images)}", flush=True)

    data = dict(model=str(args.weights), detector="rfdetr", imgsz=resolution, conf=args.conf, nms_iou=None,
                created=time.strftime("%Y-%m-%d %H:%M:%S"), images=out)
    tmp = Path(args.out).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(args.out)
    print(f"DONE {args.out} {sum(len(v) for v in out.values())} boxes", flush=True)


if __name__ == "__main__":
    sys.exit(main())
