"""Per-object-size metrics (mAP50, precision, recall, F1) for an ultralytics YOLO model.

Thin adapter over ../common/size_metrics_core.py, which holds the bucketing, matching, metrics,
files and MLflow logging, shared with the RF-DETR training code so both detectors are scored by
the same code against the same YOLO labels. See that module for the conventions.

After training this runs from train_yolo26n.py and logs to the same MLflow run. Standalone,
e.g. for an older best.pt:
    python train_code/size_metrics.py --weights runs/detect/X/weights/best.pt \
        --data /path/data.yaml --imgsz 1280 [--run-id <mlflow run id>]
"""
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import size_metrics_core as core  # noqa: E402
from size_metrics_core import collect, compute, kmeans_edges, split_smallest  # noqa: E402,F401  (re-exported)


def yolo_predictor(model, imgsz, min_conf=0.001, device=None):
    """predict(im_file) -> (xyxy, conf, cls, (h, w)) for the shared size metrics."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # training leaves a large cache behind

    def predict(im_file):
        # one image at a time: a batched list source can OOM at large imgsz
        r = model.predict(im_file, conf=min_conf, imgsz=imgsz, device=device, verbose=False, max_det=1000)[0]
        b = r.boxes
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int), r.orig_shape

    return predict


def evaluate_by_size(best_pt, data_yaml, split="test", imgsz=640, buckets=6, edges=None, conf=0.25, iou=0.5,
                     min_conf=0.001, save_dir=None, run_id=None, device=None, split_small=2, min_count=300,
                     extra_edges=(6,)):
    """Compute per-size metrics for `best_pt` on `split`, write CSV/JSON/PNG and log them to MLflow `run_id`."""
    data = check_det_dataset(data_yaml)
    if not data.get(split):
        LOGGER.warning(f"No '{split}' split in {data_yaml}; skipping size metrics")
        return None
    files = sorted(core.list_split_images(data[split]))
    model = YOLO(str(best_pt))
    return core.evaluate(yolo_predictor(model, imgsz, min_conf, device), files, model.names, split, imgsz, buckets,
                         edges, conf, iou, save_dir=save_dir, run_id=run_id, split_small=split_small,
                         min_count=min_count, extra_edges=extra_edges, weights=best_pt, log=LOGGER)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--buckets", type=int, default=6)
    ap.add_argument("--edges", default="", help="comma-separated px edges instead of k-means, e.g. 10,18,32,58,114")
    ap.add_argument("--split-small", type=int, default=2,
                    help="extra equal-count splits of the smallest k-means bucket (0 = plain k-means)")
    ap.add_argument("--min-count", type=int, default=300, help="minimum boxes per split of the smallest bucket")
    ap.add_argument("--extra-edges", default="6",
                    help="comma-separated px cut points always added to the k-means edges ('' = none)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--save-dir", default=None, help="default: <weights dir>/../size_metrics")
    ap.add_argument("--run-id", default=None, help="MLflow run to attach the results to")
    ap.add_argument("--mlflow-uri", default="http://localhost:5001")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.run_id:
        import os

        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
    edges = [float(v) for v in args.edges.split(",") if v.strip()] or None
    save_dir = args.save_dir or str(Path(args.weights).resolve().parent.parent / "size_metrics")
    evaluate_by_size(args.weights, args.data, args.split, args.imgsz, args.buckets, edges, args.conf, args.iou,
                     save_dir=save_dir, run_id=args.run_id, device=args.device,
                     split_small=args.split_small, min_count=args.min_count,
                     extra_edges=[float(v) for v in args.extra_edges.split(",") if v.strip()])


if __name__ == "__main__":
    main()
