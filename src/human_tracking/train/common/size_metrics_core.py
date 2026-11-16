"""Detection metrics per object-size bucket: mAP50, precision, recall, F1 - for any detector.

One mAP number hides where a detector stops working. Here the ground-truth boxes of the
evaluated split are clustered by size (1-D k-means on log sqrt(area), so the buckets follow
the data instead of round numbers), and every metric is reported per bucket. The bucket
where recall collapses is the natural minimum-size filter for inference.

Framework-neutral: the detector comes in as a `predict(image_path) -> (xyxy, conf, cls, (h, w))`
function, and ground truth is always read from the YOLO label files of the split. YOLO
(ultralytics) and RF-DETR are therefore scored by the same code against the same labels, so
their numbers are directly comparable. Only numpy is required; matplotlib (plot) and mlflow
(logging) are optional.

Conventions (COCO area ranges):
- matching is per class, greedy by confidence, IoU >= iou
- a prediction matched to a GT box in bucket b is a TP for b; matched to a GT box in another
  bucket it is IGNORED for b, so a bucket is not punished for other buckets' objects
- an unmatched prediction is a FP for the bucket its own size falls in
- mAP50 sweeps all confidences (mean over classes present in the bucket);
  P / R / F1 are at the fixed operating point `conf`, i.e. what a deployment would see
- sizes are sqrt(w*h) in pixels of the dataset images; the median image long side is logged
  so thresholds can be rescaled to the deployment frame size
"""
import csv
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}
LOGGER = logging.getLogger("size_metrics")


# ---------------------------------------------------------------------------------------- dataset (YOLO layout)
def load_data_yaml(data_yaml):
    """YOLO data.yaml -> dict with absolute split paths (str or list) and names {id: name}.

    Resolves splits the way ultralytics' check_det_dataset does: relative to `path`, which is itself
    relative to the yaml's folder when not absolute."""
    import yaml

    p = Path(data_yaml)
    d = yaml.safe_load(p.read_text())
    root = Path(d.get("path") or p.parent)
    if not root.is_absolute():
        root = (p.parent / root).resolve()
    out = dict(d)
    for split in ("train", "val", "test"):
        v = d.get(split)
        if not v:
            continue
        fix = lambda s: str(s if Path(s).is_absolute() else root / s)  # noqa: E731
        out[split] = [fix(s) for s in v] if isinstance(v, list) else fix(v)
    names = d.get("names", {})
    out["names"] = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    return out


def list_split_images(split):
    """Image paths of a split: a directory, a .txt list of images, or a list of those."""
    files = []
    for s in split if isinstance(split, list) else [split]:
        p = Path(s)
        if p.is_dir():
            files += [str(f) for f in p.rglob("*.*") if f.suffix[1:].lower() in IMG_FORMATS]
        elif p.suffix == ".txt":
            lines = [x.strip() for x in p.read_text().splitlines() if x.strip()]
            files += [x if Path(x).is_absolute() else str((p.parent / x).resolve()) for x in lines]
    return files


def label_path(im_file):
    """.../images/x.png -> .../labels/x.txt (the last /images/ in the path), as ultralytics does."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    im_file = str(im_file)
    return sb.join(im_file.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"


def read_labels(path, w, h):
    """YOLO txt (cls xc yc bw bh, normalized) -> list of (cls, x1, y1, x2, y2) in pixels."""
    p = Path(path)
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


# ---------------------------------------------------------------------------------------- matching and metrics
def _iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _size(xyxy):
    """sqrt(area) per box; degenerate boxes (x2 < x1) count as 0 instead of NaN."""
    if len(xyxy) == 0:
        return np.zeros(0)
    w = np.clip(xyxy[:, 2] - xyxy[:, 0], 0, None)
    h = np.clip(xyxy[:, 3] - xyxy[:, 1], 0, None)
    return np.sqrt(w * h)


def kmeans_edges(sizes, k, iters=100):
    """1-D Lloyd's k-means on log(size), quantile-initialised (deterministic).
    Returns the k-1 bucket edges: midpoints between neighbouring centres, in pixels."""
    x = np.log(np.asarray(sizes, dtype=np.float64) + 1e-6)
    centres = np.quantile(x, np.linspace(0.5 / k, 1 - 0.5 / k, k))
    for _ in range(iters):
        assign = np.abs(x[:, None] - centres[None, :]).argmin(axis=1)
        new = np.array([x[assign == j].mean() if np.any(assign == j) else centres[j] for j in range(k)])
        if np.allclose(new, centres):
            break
        centres = new
    centres = np.sort(centres)
    return np.exp((centres[:-1] + centres[1:]) / 2)


def split_smallest(sizes, edges, extra, min_count):
    """Subdivide the smallest bucket into up to `extra`+1 equal-count parts.

    The small end is where the detector fails, so it is worth resolving more finely than k-means
    does. Equal-count quantiles keep every new bucket equally well sampled; the number of parts
    is reduced until each holds >= min_count boxes (at n=300 a recall near 0.5 has a 95% CI of
    about +-0.06, enough to tell neighbouring buckets apart).
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    first = sizes[sizes < edges[0]] if len(edges) else sizes
    for parts in range(extra + 1, 1, -1):
        if len(first) >= parts * min_count:
            cuts = np.unique(np.quantile(first, np.arange(1, parts) / parts))
            return np.unique(np.concatenate([cuts, edges]))
    return edges


def _ap(flags, n_gt):
    """101-point interpolated AP from a confidence-sorted list of 1 (TP) / 0 (FP)."""
    if n_gt == 0:
        return float("nan")
    if len(flags) == 0:
        return 0.0
    tp = np.cumsum(flags)
    recall = tp / n_gt
    precision = tp / np.arange(1, len(flags) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.where(idx < len(precision), precision[np.clip(idx, 0, len(precision) - 1)], 0).mean())


def _prf(tp, fp, n_gt):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / n_gt if n_gt else float("nan")
    f1 = 2 * p * r / (p + r) if tp else 0.0
    return p, r, f1


def match_image(p_xyxy, p_conf, p_cls, g_xyxy, g_cls, iou_thr):
    """Greedy per-class matching of one image -> list of (cls, conf, pred_size, matched_gt_size or None)."""
    p_xyxy = np.asarray(p_xyxy, dtype=np.float32).reshape(-1, 4)
    p_conf = np.asarray(p_conf, dtype=np.float32).reshape(-1)
    p_cls = np.asarray(p_cls).astype(int).reshape(-1)
    order = np.argsort(-p_conf, kind="stable")
    p_xyxy, p_conf, p_cls = p_xyxy[order], p_conf[order], p_cls[order]
    p_size, g_size = _size(p_xyxy), _size(g_xyxy)
    ious = _iou(p_xyxy, g_xyxy)
    taken = np.zeros(len(g_xyxy), dtype=bool)
    out = []
    for i in range(len(p_xyxy)):
        cand = np.where(~taken & (g_cls == p_cls[i]) & (ious[i] >= iou_thr))[0] if len(g_xyxy) else []
        matched = None
        if len(cand):
            j = cand[np.argmax(ious[i, cand])]
            taken[j] = True
            matched = float(g_size[j])
        out.append((int(p_cls[i]), float(p_conf[i]), float(p_size[i]), matched))
    return out


def collect(predict, files, iou_thr, log=LOGGER):
    """Predict every image with `predict(im_file) -> (xyxy, conf, cls, (h, w))` and match it against its labels.

    Returns (preds, gts, long_sides):
      preds: list of (cls, conf, pred_size, matched_gt_size or None)
      gts:   list of (cls, gt_size)
    """
    preds, gts, long_sides = [], [], []
    for n, im_file in enumerate(files, 1):
        p_xyxy, p_conf, p_cls, (h, w) = predict(im_file)
        long_sides.append(max(h, w))
        # drop duplicate label rows, as ultralytics does: a copy can never be matched and would count as a miss
        lab = list(dict.fromkeys(read_labels(label_path(im_file), w, h)))
        g_cls = np.array([b[0] for b in lab], dtype=int)
        g_xyxy = np.array([b[1:] for b in lab], dtype=np.float32).reshape(-1, 4)
        gts += list(zip(g_cls.tolist(), _size(g_xyxy).tolist()))
        preds += match_image(p_xyxy, p_conf, p_cls, g_xyxy, g_cls, iou_thr)
        if n % 200 == 0:
            log.info(f"size metrics: {n}/{len(files)} images")
    return preds, gts, long_sides


def compute(preds, gts, edges, conf, names):
    """Per-bucket rows (class 'all' plus one per class) and the cumulative min-size table."""
    edges = np.asarray(edges, dtype=np.float64)
    bounds = [0.0, *edges.tolist(), float("inf")]
    bucket = lambda s: int(np.searchsorted(edges, s, side="right"))  # noqa: E731
    classes = sorted({c for c, _ in gts} | {c for c, *_ in preds})

    # per (bucket, class): gt count and a conf-sorted TP/FP list under the COCO ignore rule
    n_gt = {}
    lists = {}
    for c, s in gts:
        key = (bucket(s), c)
        n_gt[key] = n_gt.get(key, 0) + 1
    for c, cf, p_size, m_size in preds:
        b = bucket(m_size) if m_size is not None else bucket(p_size)
        lists.setdefault((b, c), []).append((cf, 1 if m_size is not None else 0))
    for v in lists.values():
        v.sort(key=lambda t: -t[0])

    rows = []
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], bounds[b + 1]
        aps, tot = [], dict(n_gt=0, tp=0, fp=0)
        for c in classes:
            lst = lists.get((b, c), [])
            ng = n_gt.get((b, c), 0)
            ap = _ap(np.array([t[1] for t in lst], dtype=np.float64), ng)
            tp = sum(t[1] for t in lst if t[0] >= conf)
            fp = sum(1 - t[1] for t in lst if t[0] >= conf)
            p, r, f1 = _prf(tp, fp, ng)
            if ng:
                aps.append(ap)
            if len(classes) > 1:
                rows.append(dict(bucket=b, cls=names.get(c, str(c)), lo_px=lo, hi_px=hi, n_gt=ng, mAP50=ap,
                                 precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=ng - tp))
            tot["n_gt"] += ng
            tot["tp"] += tp
            tot["fp"] += fp
        p, r, f1 = _prf(tot["tp"], tot["fp"], tot["n_gt"])
        rows.append(dict(bucket=b, cls="all", lo_px=lo, hi_px=hi, n_gt=tot["n_gt"],
                         mAP50=float(np.mean(aps)) if aps else float("nan"), precision=p, recall=r, f1=f1,
                         tp=tot["tp"], fp=tot["fp"], fn=tot["n_gt"] - tot["tp"]))

    # what is left if everything (labels AND detections) below a bucket's lower edge is dropped
    all_rows = [r for r in rows if r["cls"] == "all"]
    total = sum(r["n_gt"] for r in all_rows)
    cumulative = []
    for b, r in enumerate(all_rows):
        kept = all_rows[b:]
        ng, tp, fp = (sum(k[f] for k in kept) for f in ("n_gt", "tp", "fp"))
        p, rc, f1 = _prf(tp, fp, ng)
        cumulative.append(dict(min_px=r["lo_px"], n_gt=ng, gt_share=ng / max(1, total),
                               precision=p, recall=rc, f1=f1))
    return rows, cumulative


def conf_sweep(preds, gts, edges, confs):
    """Recall / precision / FP count per size bucket at each confidence in `confs` (threshold choice)."""
    out = []
    for c in confs:
        rows, _ = compute(preds, gts, edges, c, {})
        for r in (r for r in rows if r["cls"] == "all"):
            out.append(dict(conf=c, bucket=r["bucket"], lo_px=r["lo_px"], hi_px=r["hi_px"], n_gt=r["n_gt"],
                            recall=r["recall"], precision=r["precision"], fp=r["fp"]))
    return out


def _fmt_range(lo, hi):
    return f"{lo:.0f}-{hi:.0f}" if np.isfinite(hi) else f">={lo:.0f}"


def format_tables(rows, cumulative, conf, iou, imgsz, median_long_side):
    out = [f"Size metrics | IoU {iou} | P/R/F1 at conf {conf} | imgsz {imgsz} | "
           f"sizes = sqrt(w*h) px of dataset images (median long side {median_long_side:.0f} px)",
           f"{'bucket':<7}{'size px':<11}{'n_gt':>7}{'mAP50':>8}{'prec':>8}{'recall':>8}{'R±95%':>7}{'F1':>8}"
           f"{'TP':>7}{'FP':>7}{'FN':>7}"]
    for r in (r for r in rows if r["cls"] == "all"):
        # normal-approximation half-width of the recall's 95% interval: how far to trust a small bucket
        ci = 1.96 * np.sqrt(r["recall"] * (1 - r["recall"]) / r["n_gt"]) if r["n_gt"] else float("nan")
        out.append(f"{r['bucket']:<7}{_fmt_range(r['lo_px'], r['hi_px']):<11}{r['n_gt']:>7}{r['mAP50']:>8.3f}"
                   f"{r['precision']:>8.3f}{r['recall']:>8.3f}{ci:>7.3f}{r['f1']:>8.3f}"
                   f"{r['tp']:>7}{r['fp']:>7}{r['fn']:>7}")
    out += ["", "Drop everything (labels and detections) smaller than min_px:",
            f"{'min_px':<9}{'gt kept':>9}{'share':>8}{'prec':>8}{'recall':>8}{'F1':>8}"]
    for c in cumulative:
        out.append(f"{c['min_px']:<9.0f}{c['n_gt']:>9}{100 * c['gt_share']:>7.1f}%{c['precision']:>8.3f}"
                   f"{c['recall']:>8.3f}{c['f1']:>8.3f}")
    return "\n".join(out)


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["cls"] == "all"]
    labels = [_fmt_range(r["lo_px"], r["hi_px"]) for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    ax2.bar(x, [r["n_gt"] for r in rows], color="0.88", zorder=0)
    ax2.set_ylabel("ground-truth boxes")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    for key, style in (("mAP50", "o-"), ("precision", "s--"), ("recall", "^--"), ("f1", "d-")):
        ax.plot(x, [r[key] for r in rows], style, label=key)
    ax.set_xticks(x, labels)
    ax.set_xlabel("object size, sqrt(w*h) px")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Detection metrics by object size")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(predict, files, names, split="test", imgsz=640, buckets=6, edges=None, conf=0.25, iou=0.5,
             save_dir=None, run_id=None, split_small=2, min_count=300, extra_edges=(6,), weights="",
             sweep_confs=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5), log=LOGGER):
    """Per-size metrics of `predict` on `files`; writes CSV/JSON/PNG/TXT and logs them to MLflow `run_id`.

    `predict(im_file)` must return detections down to a LOW confidence (e.g. 0.001): mAP50 and the
    confidence sweep need the whole ranking, P/R/F1 are then read at `conf`."""
    log.info(f"size metrics: predicting {len(files)} {split} images at imgsz {imgsz}")
    preds, gts, long_sides = collect(predict, files, iou, log)
    if not gts:
        log.warning("size metrics: no ground-truth boxes; skipping")
        return None

    if edges is None or len(edges) == 0:  # pinned edges are used as given
        sizes = [s for _, s in gts]
        edges = kmeans_edges(sizes, buckets)
        if split_small > 0:
            edges = split_smallest(sizes, edges, split_small, min_count)
        # explicit cut points are kept even when they leave a bucket short of min_count (warned below)
        edges = np.concatenate([edges, [e for e in (extra_edges or ()) if e > 0]])
    edges = np.unique(np.round(np.asarray(edges, dtype=np.float64), 1))
    rows, cumulative = compute(preds, gts, edges, conf, names)
    for r in (r for r in rows if r["cls"] == "all" and 0 < r["n_gt"] < min_count):
        log.warning(f"size metrics: bucket {_fmt_range(r['lo_px'], r['hi_px'])} px has only {r['n_gt']} boxes "
                    f"(< {min_count}); read its metrics with the wider R±95% interval in mind")
    median_long = float(np.median(long_sides))
    table = format_tables(rows, cumulative, conf, iou, imgsz, median_long)
    log.info("\n" + table)
    sweep = conf_sweep(preds, gts, edges, sweep_confs) if sweep_confs else []

    out_dir = Path(save_dir) if save_dir else Path(tempfile.mkdtemp())
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out_dir / f"size_metrics_{split}.csv", out_dir / f"size_metrics_{split}.json"
    png_path, txt_path = out_dir / f"size_metrics_{split}.png", out_dir / f"size_metrics_{split}.txt"
    sweep_path = out_dir / f"conf_sweep_{split}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if sweep:
        with open(sweep_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep[0]))
            w.writeheader()
            w.writerows(sweep)
    json.dump(dict(weights=str(weights), split=split, imgsz=imgsz, iou=iou, conf=conf, edges_px=edges.tolist(),
                   median_image_long_side=median_long, rows=rows, cumulative=cumulative, conf_sweep=sweep),
              open(json_path, "w"), indent=2, default=lambda v: None if v != v else v)
    txt_path.write_text(table + "\n")
    try:
        plot(rows, png_path)
    except Exception as e:  # a plot must never cost the metrics
        log.warning(f"size metrics: plot failed: {e}")

    if run_id:
        import time

        from mlflow import MlflowClient
        from mlflow.entities import Metric

        # MlflowClient writes into the run without opening/closing it: a mlflow.start_run(run_id=...) block would
        # END the run on exit, i.e. mark a run that is still training (or logged by another process) FINISHED
        client = MlflowClient()
        now = int(time.time() * 1000)
        metrics = []
        for r in (r for r in rows if r["cls"] == "all"):
            tag = f"{split}_size/b{r['bucket']}_{_fmt_range(r['lo_px'], r['hi_px']).replace('>=', 'ge')}px"
            for k in ("mAP50", "precision", "recall", "f1"):
                if r[k] == r[k]:  # skip NaN
                    metrics.append(Metric(f"{tag}/{k}", float(r[k]), now, 0))
            metrics.append(Metric(f"{tag}/n_gt", float(r["n_gt"]), now, 0))
        client.log_batch(run_id, metrics=metrics)
        # a tag, not a param: params are immutable, so re-evaluating a run with new buckets would fail
        client.set_tag(run_id, f"{split}_size_edges_px", ",".join(f"{e:g}" for e in edges))
        for p in (csv_path, json_path, png_path, txt_path, sweep_path):
            if p.exists():
                client.log_artifact(run_id, str(p), artifact_path="size_metrics")
        log.info(f"MLflow: logged size metrics ({len(edges) + 1} buckets) to run {run_id}")
    return rows, cumulative, edges
