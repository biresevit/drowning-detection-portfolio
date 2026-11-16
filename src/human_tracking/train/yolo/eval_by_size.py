"""Evaluate a YOLO detector per object-size bucket, to find where it stops working.

A single mAP number hides the thing that matters here: humans in these pools span two
orders of magnitude in area, and the detector is useless below some size. This splits
the ground truth into size clusters (1-D k-means on sqrt(area), so the buckets follow
the data rather than arbitrary round numbers) and reports mAP50, precision, recall and
F1 for each one. The bucket where recall collapses is the inference filter threshold.

Metrics follow COCO's area-range convention:

  * a prediction matched to a GT box in bucket b is a true positive for b
  * a prediction matched to a GT box in another bucket is IGNORED for b, not a false
    positive - otherwise every bucket is punished for the other buckets' objects
  * an unmatched prediction is a false positive for the bucket its OWN size falls in
  * mAP50 sweeps confidence over that per-bucket list; P/R/F1 are reported at a fixed
    operating point (--conf), which is the number that tells you what to deploy

Sizes are reported in native image pixels, because that is what you would threshold on
at inference time.

    python train_code/eval_by_size.py --weights runs/detect/humans_yolo26s_1280/weights/best.pt
"""

import argparse
import csv
import glob
import json
import os

import numpy as np


def load_gt(label_path, width, height):
    """YOLO label file -> xyxy boxes in pixels."""
    boxes = []
    if not os.path.exists(label_path):
        return np.zeros((0, 4), dtype=np.float32)
    for line in open(label_path, encoding="utf-8"):
        parts = line.split()
        if len(parts) != 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        cx, bw = cx * width, bw * width
        cy, bh = cy * height, bh * height
        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def kmeans_1d(values, k, iters=100):
    """Lloyd's algorithm on log(size). Quantile init keeps it deterministic."""
    x = np.log(np.asarray(values, dtype=np.float64) + 1e-9)
    centres = np.quantile(x, np.linspace(0.5 / k, 1 - 0.5 / k, k))
    for _ in range(iters):
        assign = np.abs(x[:, None] - centres[None, :]).argmin(axis=1)
        moved = False
        for j in range(k):
            member = x[assign == j]
            if len(member):
                new = member.mean()
                if abs(new - centres[j]) > 1e-12:
                    centres[j], moved = new, True
        if not moved:
            break
    centres.sort()
    # Bucket edges are the midpoints between neighbouring centres, back in linear space.
    edges = np.exp((centres[:-1] + centres[1:]) / 2)
    return edges


def average_precision(tp_flags, n_gt):
    """101-point interpolated AP from a confidence-sorted TP/FP list."""
    if n_gt == 0:
        return float("nan")
    if len(tp_flags) == 0:
        return 0.0
    tp = np.cumsum(tp_flags)
    fp = np.cumsum(1 - tp_flags)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    # make precision monotonically decreasing
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    out = np.where(idx < len(precision), precision[np.clip(idx, 0, len(precision) - 1)], 0)
    return float(out.mean())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data",
                    default="/home/ubuntu/projects/data/humans/dataset_grouped/data.yaml")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--buckets", type=int, default=6)
    ap.add_argument("--iou", type=float, default=0.5, help="match threshold")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="operating point for P/R/F1 (mAP sweeps all confidences)")
    ap.add_argument("--min_conf", type=float, default=0.001,
                    help="lowest confidence kept, for the PR curve")
    ap.add_argument("--device", default="0")
    ap.add_argument("--edges", default=None,
                    help="comma-separated sqrt(area) edges, instead of clustering")
    ap.add_argument("--csv", default=None, help="write the table here")
    args = ap.parse_args()

    import yaml
    from ultralytics import YOLO

    cfg = yaml.safe_load(open(args.data, encoding="utf-8"))
    root = cfg.get("path", os.path.dirname(args.data))
    img_dir = os.path.join(root, cfg[args.split])
    images = sorted(p for p in glob.glob(os.path.join(img_dir, "*"))
                    if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
    if not images:
        raise SystemExit(f"no images in {img_dir}")
    print(f"{len(images)} images in {args.split}")

    model = YOLO(args.weights, task="detect")

    # ---- inference + greedy matching, one image at a time ----
    records = []          # (conf, matched_gt_size or None, pred_size)
    gt_sizes_all = []     # every GT size, for clustering
    gt_entries = []       # (image index, size, matched?) filled after matching

    for i, path in enumerate(images):
        res = model.predict(path, imgsz=args.imgsz, conf=args.min_conf, iou=0.7,
                            device=args.device, verbose=False, max_det=1000)[0]
        height, width = res.orig_shape
        stem = os.path.splitext(os.path.basename(path))[0]
        label_path = os.path.join(root, cfg[args.split].replace("images", "labels"),
                                  stem + ".txt")
        gt = load_gt(label_path, width, height)
        gt_size = np.sqrt((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])) if len(gt) else np.zeros(0)
        gt_sizes_all.extend(gt_size.tolist())

        pred = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros(0)
        order = np.argsort(-conf)
        pred, conf = pred[order], conf[order]
        # a degenerate prediction can come back with x2<x1; clip so sqrt stays real
        pred_size = (np.sqrt(np.clip(pred[:, 2] - pred[:, 0], 0, None)
                             * np.clip(pred[:, 3] - pred[:, 1], 0, None))
                     if len(pred) else np.zeros(0))

        ious = iou_matrix(pred, gt)
        taken = np.zeros(len(gt), dtype=bool)
        matched_gt = np.full(len(pred), -1, dtype=int)
        for p in range(len(pred)):
            if len(gt) == 0:
                break
            cand = np.where(~taken & (ious[p] >= args.iou))[0]
            if len(cand):
                best = cand[np.argmax(ious[p, cand])]
                taken[best] = True
                matched_gt[p] = best

        for p in range(len(pred)):
            g = matched_gt[p]
            records.append((float(conf[p]),
                            float(gt_size[g]) if g >= 0 else None,
                            float(pred_size[p])))
        for g in range(len(gt)):
            gt_entries.append((float(gt_size[g]), bool(taken[g]),
                               float(conf[np.where(matched_gt == g)[0][0]]) if taken[g] else 0.0))

        if (i + 1) % 100 == 0 or i + 1 == len(images):
            print(f"  {i + 1}/{len(images)}", flush=True)

    gt_sizes_all = np.array(gt_sizes_all)
    print(f"\n{len(gt_sizes_all)} ground-truth boxes, {len(records)} predictions "
          f"(conf >= {args.min_conf})")

    # ---- size buckets ----
    if args.edges:
        edges = np.array([float(v) for v in args.edges.split(",")])
    else:
        edges = kmeans_1d(gt_sizes_all, args.buckets)
    bounds = [0.0, *edges.tolist(), float("inf")]
    print(f"bucket edges on sqrt(area), px: "
          f"{', '.join(f'{e:.0f}' for e in edges)}")

    def bucket_of(size):
        return int(np.searchsorted(edges, size, side="right"))

    n_b = len(bounds) - 1
    rows = []
    for b in range(n_b):
        gt_b = [g for g in gt_entries if bucket_of(g[0]) == b]
        n_gt = len(gt_b)

        # per-bucket detection list: TPs for this bucket + FPs whose own size is here
        lst = []
        for conf, m_size, p_size in records:
            if m_size is not None:
                if bucket_of(m_size) == b:
                    lst.append((conf, 1))
            elif bucket_of(p_size) == b:
                lst.append((conf, 0))
        lst.sort(key=lambda t: -t[0])
        flags = np.array([t[1] for t in lst], dtype=np.float64)
        ap50 = average_precision(flags, n_gt)

        # operating point
        at = [t for t in lst if t[0] >= args.conf]
        tp = sum(t[1] for t in at)
        fp = len(at) - tp
        fn = n_gt - tp
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / n_gt if n_gt else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if tp and precision + recall > 0 else 0.0)
        lo, hi = bounds[b], bounds[b + 1]
        rows.append(dict(bucket=b, lo_px=round(lo, 1),
                         hi_px=(None if hi == float("inf") else round(hi, 1)),
                         n_gt=n_gt, mAP50=ap50, precision=precision,
                         recall=recall, f1=f1, tp=tp, fp=fp, fn=fn))

    # ---- report ----
    print(f"\nIoU {args.iou}  |  P/R/F1 at conf {args.conf}  |  imgsz {args.imgsz}")
    print(f"{'bucket':<8}{'size px (sqrt area)':<22}{'n_gt':>7}{'mAP50':>9}"
          f"{'prec':>8}{'recall':>8}{'F1':>8}{'TP':>7}{'FP':>7}{'FN':>7}")
    for r in rows:
        rng = f"{r['lo_px']:.0f} - {r['hi_px']:.0f}" if r["hi_px"] else f">= {r['lo_px']:.0f}"
        print(f"{r['bucket']:<8}{rng:<22}{r['n_gt']:>7}{r['mAP50']:>9.3f}"
              f"{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}"
              f"{r['tp']:>7}{r['fp']:>7}{r['fn']:>7}")

    total_gt = sum(r["n_gt"] for r in rows)
    total_tp = sum(r["tp"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    print(f"{'ALL':<8}{'':<22}{total_gt:>7}{'':>9}"
          f"{total_tp / max(1, total_tp + total_fp):>8.3f}"
          f"{total_tp / max(1, total_gt):>8.3f}{'':>8}{total_tp:>7}{total_fp:>7}"
          f"{total_gt - total_tp:>7}")

    print("\ncumulative if you discard every detection below a bucket's lower edge:")
    print(f"{'keep size >=':<16}{'gt kept':>9}{'gt kept %':>11}{'recall':>9}{'prec':>8}{'F1':>8}")
    for b in range(n_b):
        kept = rows[b:]
        n_gt = sum(r["n_gt"] for r in kept)
        tp = sum(r["tp"] for r in kept)
        fp = sum(r["fp"] for r in kept)
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / n_gt if n_gt else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
        print(f"{rows[b]['lo_px']:<16.0f}{n_gt:>9}{100 * n_gt / max(1, total_gt):>10.1f}%"
              f"{recall:>9.3f}{precision:>8.3f}{f1:>8.3f}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")
        meta = os.path.splitext(args.csv)[0] + "_meta.json"
        json.dump(dict(weights=args.weights, split=args.split, imgsz=args.imgsz,
                       iou=args.iou, conf=args.conf, edges=edges.tolist()),
                  open(meta, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
