"""
analyse_predictions_ui.py

Browse a YOLO dataset split by object size and compare a detector's predictions with the
annotations, to see what the detector misses and where it is wrong, one size bucket at a time.

  Size     Pick a bucket (e.g. 0-6 px): only images containing objects of that size are listed,
           and only predictions of that size are drawn. Size is sqrt(w*h) in image pixels, the
           same measure and default buckets as size_metrics.py, so what you see here is what the
           per-size metrics count.
  Filter annotations
           Off (default): the right panel shows ALL annotations of the image, with only the
           predictions filtered. On: annotations are filtered by size only (never by Match /
           No match), so the right panel shows every annotation of the chosen size, hit or missed.
  Predict  "Run predictions" runs --humans_model_path over every image of the split and caches
           the boxes in a JSON next to the dataset. The cache keeps everything down to
           --save_conf, so the Conf threshold can be changed afterwards without re-running.
           Left panel = predictions, right panel = annotations.
  Show     All | Match | No match, which filters the predictions (left) and decides which images
           are listed; it never hides annotations (right)
             Match    : predictions that hit an annotation (TP); lists images that have one
             No match : predictions that hit nothing (FP); lists images with a false positive
                        or a missed annotation (FN, orange), which Z visits
           Matching is greedy by confidence at IoU >= --iou. A matched prediction belongs to the
           size bucket of the annotation it hit, an unmatched one to the bucket of its own size
           (COCO convention, same as size_metrics.py).

Colours: green = matched (TP / hit annotation), red = false positive, orange = missed annotation,
yellow = annotation while no predictions are loaded. Boxes too small to spot get a dashed ring.

Tiny objects are a few pixels on screen, so both panels share one zoomable view:
  Mouse wheel   zoom at cursor           Drag          pan
  Left / Right  previous / next image    Z             zoom to next shown object
  F / Home      fit image                L             toggle box labels

Editing annotations (right panel)
  Click a box     select it (the smallest one under the cursor)
  Drag inside     move it            Drag a handle   resize it (zoom in first for tiny boxes: Z)
  N               draw a new box (one-shot; then drag on the right panel)
  D / Delete      delete the selected box
  P               Apply prediction: replace the selected box with the shown prediction that
                  overlaps it the most (a dashed cyan frame marks which one on the left panel)
  Ctrl+Z          undo (per image)   Esc  cancel the drag / draw mode / selection
  Delete image... (button, far right) asks first, then moves the image and its label file to
                  <input_dir>/_deleted/<split>/ - out of the dataset, but recoverable.
                  Images deleted here are only unlinked from this split: a copy that is
                  hardlinked elsewhere (e.g. the pool it was split from) stays there.
  Ctrl+S          Save: writes every edited image to its label file. Rows you did not touch are
                  written back unchanged; the first original of each file is kept once under
                  <input_dir>/_label_backups/<split>/, and <split>/labels.cache is deleted
                  (ultralytics trusts it by file size alone, which an edit does not change).
  Auto-save       (checkbox, on by default) writes an image's edits when you leave it with
                  Next / Prev, and when you close the window.

Settings are remembered between sessions in ~/.analyse_predictions_ui.json: confidence, size
filter, Show mode, the checkboxes, window size/position and the image you were on. An explicit
--conf wins over the saved one; --reset_settings ignores them.

Far regions (third toolbar row)
  Far Region [M]  generates the far regions: the part of the image where people are only a few pixels.
                  "small <" (default 15) is that size, sqrt(w*h) px. Regions group the small labels where
                  at least 95% of what they remove is small; they follow a slanted horizon, and are then
                  merged where they overlap. Any number of regions per image is fine. The regions the
                  image had are replaced (Ctrl+Z brings them back).
  expand up       the camera is level, so everything above a far spot is farther still: each region is
                  stretched up to the image top. A region that cannot be (larger labels sit above it)
                  is not a real far region and is dropped.
  expand sideways each region is continued flat to the left and right until just before the first clearly
                  larger label (1.5 x 'small <') it would swallow.
  apply to all images  Far Region and Clear regions then act on every image of the split instead of this
                  one: the same function, so the same result. Nothing is written until Save.
  Only images with regions  limits Next / Prev to those, for reviewing.
  Draw region [R] draws one by hand: click its corners, then Enter, double-click or a click on the first
                  corner closes it (Backspace drops the last corner, Esc cancels); or just drag a
                  rectangle. Hand-drawn regions are used exactly as drawn.
  Editing         click a region (its outline, or its inside where no annotation is) to select it. Drag a
                  corner to reshape, drag inside to move it, double-click an edge to add a corner,
                  right-click a corner to remove it, D deletes the region, Ctrl+Z undoes. Clear regions
                  removes all of them (of every image with 'apply to all images').
  Regions are drawn darkened on both panels, and labels under one are grey: they would be removed. They are
  saved by Save / Auto-save into <input_dir>/far_regions_<split>.json (older rectangle-only files still
  load); nothing in the images or labels changes. A label counts as removed when its centre is inside a
  region or a region covers at least half of its area.

Headless prediction, e.g. on the GPU VM (copy the JSON next to the dataset afterwards):
    python analyse_predictions_ui.py --input_dir <dataset> --humans_model_path best.pt --predict

Usage:
    python analyse_predictions_ui.py --input_dir path/to/data
    python analyse_predictions_ui.py --input_dir <dataset> --humans_model_path models/humans.pt
"""

import argparse
import json
import math
import os
import queue
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:  # the UI needs Tk; --predict must also work on a headless GPU box without it
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from PIL import ImageTk
except ImportError:  # pragma: no cover
    tk = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_EDGES = "6,9,11,13,22,36,61,112"
MODES = ("All", "Match", "No match")

C_TP = "#22dd22"     # matched prediction / annotation that was hit
C_FP = "#ff3333"     # prediction that hit nothing
C_FN = "#ff9900"     # annotation nothing hit
C_GT = "#ffdd00"     # annotation, no predictions loaded
SMALL_ON_SCREEN = 12  # boxes smaller than this (display px) get a ring so they can be found
MAX_ZOOM = 48.0

C_SEL = "#00e5ff"     # selected annotation, and the prediction Apply would copy
HANDLE = 4            # half-size of a drag handle, display px
HANDLE_HIT = 7        # grab distance around a handle, display px
HANDLE_MIN_SIDE = 14  # a selected box smaller than this on screen can be moved but not resized: zoom in (Z)
CLICK_SLOP = 3        # a left-panel press/release that moved less than this (display px) is a click, not a pan
PICK_TOL = 4          # slack when clicking a box, display px, so tiny ones can be hit
MIN_BOX_PX = 1.0      # smallest annotation an edit may create or leave behind, image px
UNDO_DEPTH = 50
HANDLE_CURSOR = {"nw": "size_nw_se", "se": "size_nw_se", "ne": "size_ne_sw", "sw": "size_ne_sw",
                 "n": "size_ns", "s": "size_ns", "e": "size_we", "w": "size_we"}


# --------------------------------------------------------------------------- data

def list_images(img_dir):
    return sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXTS)


def read_gt(label_path, w, h):
    """Label file -> (xyxy pixel boxes, class ids, source rows, editable).

    Duplicate rows are dropped (ultralytics does too; a copy can never be matched and would show up
    as a phantom miss). `src` keeps each row's original text, so saving rewrites only the boxes that
    were edited. `editable` is False when a row is not plain `cls cx cy w h` (segmentation polygon,
    trailing confidence): saving such a file would silently drop information."""
    boxes, cls, src, seen, editable = [], [], [], set(), True
    # A file of nothing but NUL bytes is a write cut short by a sudden shutdown: it holds no rows,
    # so it is read as empty and stays editable - saving replaces it (the original is backed up).
    if label_path.is_file() and label_path.read_bytes().strip(b"\x00\r\n\t "):
        for line in label_path.read_text().splitlines():
            row = line.strip()
            if not row:
                continue
            parts = row.split()
            if len(parts) != 5:
                editable = False
            if len(parts) < 5 or row in seen:
                continue
            seen.add(row)
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
            cls.append(int(float(parts[0])))
            src.append(row)
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(cls, dtype=int), src, editable


def box_sizes(xyxy):
    """sqrt(w*h); degenerate boxes count as 0 instead of NaN."""
    if len(xyxy) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt(np.clip(xyxy[:, 2] - xyxy[:, 0], 0, None) * np.clip(xyxy[:, 3] - xyxy[:, 1], 0, None))


def iou_matrix(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(pred, gt, iou_thr):
    """Predictions sorted by confidence (desc). Returns (pred -> gt index or -1, gt hit mask).

    Greedy matching is prefix-consistent: the matches among predictions above a threshold are the
    same as in the full matching, so this agrees with size_metrics.py at any confidence."""
    p2g = np.full(len(pred), -1, dtype=int)
    hit = np.zeros(len(gt), dtype=bool)
    if len(pred) == 0 or len(gt) == 0:
        return p2g, hit
    ious = iou_matrix(pred, gt)
    for i in range(len(pred)):
        cand = np.where(~hit & (ious[i] >= iou_thr))[0]
        if len(cand):
            j = cand[np.argmax(ious[i, cand])]
            hit[j] = True
            p2g[i] = j
    return p2g, hit


def bucket_labels(edges):
    bounds = [0, *edges]
    return [f"{lo:g}-{hi:g}" for lo, hi in zip(bounds[:-1], bounds[1:])] + [f">={edges[-1]:g}"]


def _polys_equal(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


class Item:
    """One image: annotations and (optionally) predictions, plus the matching at the current conf."""

    def __init__(self, path, w, h, gt, gt_cls, gt_src, editable):
        self.path, self.w, self.h = path, w, h
        self.gt, self.gt_cls, self.gt_src, self.editable = gt, gt_cls, gt_src, editable
        self.dirty, self.undo = False, []             # unsaved edits; snapshots to step back through
        self.far = []                                 # far regions: polygons, (K, 2) float32 arrays, image px
        self.far_dirty = False                        # far regions changed since they were last saved
        self.gt_size = box_sizes(gt)
        self.gt_bucket = None
        self.pred = np.zeros((0, 4), dtype=np.float32)
        self.pred_conf = np.zeros(0, dtype=np.float32)
        self.k = 0                                    # predictions at or above the current conf
        self.p2g = np.zeros(0, dtype=int)
        self.hit = np.zeros(len(gt), dtype=bool)
        self.p_bucket = np.zeros(0, dtype=int)

    def snapshot(self):
        return self.gt.copy(), self.gt_cls.copy(), list(self.gt_src), [p.copy() for p in self.far]

    def restore(self, snap):
        """Back to a snapshot. Returns (labels changed, far regions changed) relative to the state left."""
        old = (self.gt, self.gt_cls, self.gt_src, self.far)
        self.gt, self.gt_cls, self.gt_src = snap[0].copy(), snap[1].copy(), list(snap[2])
        if len(snap) > 3:
            self.far = [p.copy() for p in snap[3]]
        labels = not (np.array_equal(old[0], self.gt) and np.array_equal(old[1], self.gt_cls) and old[2] == self.gt_src)
        return labels, not _polys_equal(old[3], self.far)


def load_items(split_dir):
    items = []
    for p in list_images(split_dir / "images"):
        with Image.open(p) as im:  # header only
            w, h = im.size
        items.append(Item(p, w, h, *read_gt(split_dir / "labels" / f"{p.stem}.txt", w, h)))
    return items


def label_text(it):
    """Label file content for `it`. Rows that were not edited keep their original text byte for byte."""
    rows = []
    for j in range(len(it.gt)):
        if it.gt_src[j] is not None:
            rows.append(it.gt_src[j])
            continue
        x1, y1, x2, y2 = (float(v) for v in it.gt[j])
        rows.append(f"{int(it.gt_cls[j])} {(x1 + x2) / 2 / it.w:.6f} {(y1 + y2) / 2 / it.h:.6f} "
                    f"{(x2 - x1) / it.w:.6f} {(y2 - y1) / it.h:.6f}")
    return "".join(row + "\r\n" for row in rows)  # the dataset's label files use CRLF


def save_label(it, split_dir, backup_dir):
    if not it.editable:
        raise OSError("label file has non-detection rows; refusing to overwrite it")
    path = split_dir / "labels" / f"{it.path.stem}.txt"
    if path.is_file():
        backup = backup_dir / path.name
        if not backup.exists():  # the FIRST original is kept; later saves never replace it
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    # in place, not write-then-rename: the labels are hardlinked into the pool they were split from,
    # and a rename would silently detach this copy so a later re-split brings the old boxes back
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(label_text(it))


# ------------------------------------------------------------------------- far regions
# A far region is a polygon over the part of an image where people are only a few pixels: the labels there are
# blobs whose identity (a head or a rock?) is not in the pixels. Regions are stored next to the dataset and never
# touch the images or labels; building a training set from them means painting the polygons black and dropping the
# labels they cover.

FAR_PX = 15.0        # a label is "small" below this sqrt(w*h), px
FAR_PURITY = 0.95    # a region may only remove labels of which at least this share is small
SLICE_MIN = 6.0  # a slice of a region cut at larger labels above it must be this many x 'small <' wide
SIDE_BLOCK = 1.5  # sideways growth stops at labels of at least SIDE_BLOCK x the 'small <' size
AUTO_MAX = 4  # regions generated per image
FAR_MIN_SMALL = 3    # fewer small labels than this is not a cluster
MIN_FAR_AREA = 64.0  # smallest region, image px^2
COVER_FRAC = 0.5     # a region removes a label whose centre is inside it, or that it covers this much
SNAP_FRAC = 0.20     # a proposal is stretched to a border it comes this close to (if nothing is labelled between)
FAR_DARKEN = 0.25    # brightness of the image under a region in the editor
C_FAR = "#ff44ff"
C_COVERED = "#8a8a8a"


def rect_to_poly(rect):
    x1, y1, x2, y2 = (float(v) for v in rect)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def poly_area(poly):
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x[:-1], y[1:]) + x[-1] * y[0] - np.dot(y[:-1], x[1:]) - y[-1] * x[0]))


def points_in_polygon(px, py, poly):
    """Even-odd rule, vectorised over the points."""
    poly = np.asarray(poly, dtype=np.float64)
    x, y = np.asarray(px, dtype=np.float64)[:, None], np.asarray(py, dtype=np.float64)[:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.concatenate([poly[1:, 0], poly[:1, 0]])[None, :], np.concatenate([poly[1:, 1], poly[:1, 1]])[None, :]
    cross = (y1 > y) != (y2 > y)
    xint = (x2 - x1) * (y - y1) / np.where(y2 != y1, y2 - y1, 1e-12) + x1
    return (np.sum(cross & (x < xint), axis=1) % 2) == 1


def _clip_area(poly, x1, y1, x2, y2):
    """Area of the polygon inside an axis-aligned rectangle (Sutherland-Hodgman against its four sides)."""
    pts = [tuple(p) for p in poly]
    for axis, val, keep_greater in ((0, x1, True), (0, x2, False), (1, y1, True), (1, y2, False)):
        if not pts:
            return 0.0
        out, prev = [], pts[-1]
        for cur in pts:
            pin = (prev[axis] >= val) if keep_greater else (prev[axis] <= val)
            cin = (cur[axis] >= val) if keep_greater else (cur[axis] <= val)
            if pin != cin:
                t = (val - prev[axis]) / (cur[axis] - prev[axis])
                out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            if cin:
                out.append(cur)
            prev = cur
        pts = out
    return poly_area(np.array(pts)) if len(pts) >= 3 else 0.0


def covered_by(boxes, poly, frac=COVER_FRAC):
    """Which boxes a far region removes: the box centre is inside it, or at least `frac` of the box's area is."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    n = len(boxes)
    if n == 0 or len(poly) < 3:
        return np.zeros(n, dtype=bool)
    near = ((boxes[:, 2] >= poly[:, 0].min()) & (boxes[:, 0] <= poly[:, 0].max())
            & (boxes[:, 3] >= poly[:, 1].min()) & (boxes[:, 1] <= poly[:, 1].max()))
    out = np.zeros(n, dtype=bool)
    idx = np.where(near)[0]
    if len(idx) == 0:
        return out
    b = boxes[idx]
    centre = points_in_polygon((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, poly)
    out[idx[centre]] = True
    for i in idx[~centre]:
        x1, y1, x2, y2 = boxes[i]
        if _clip_area(poly, x1, y1, x2, y2) / max((x2 - x1) * (y2 - y1), 1e-9) >= frac:
            out[i] = True
    return out


def covered_by_any(boxes, polys):
    out = np.zeros(len(np.asarray(boxes).reshape(-1, 4)), dtype=bool)
    for p in polys:
        out |= covered_by(boxes, p)
    return out


def _rect_covered(boxes, rect, frac=COVER_FRAC):
    """covered_by for an axis-aligned rectangle, vectorised: used inside the rectangle search."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    x1, y1, x2, y2 = (float(v) for v in rect)
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    inside = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
    iw = np.clip(np.minimum(boxes[:, 2], x2) - np.maximum(boxes[:, 0], x1), 0, None)
    ih = np.clip(np.minimum(boxes[:, 3], y2) - np.maximum(boxes[:, 1], y1), 0, None)
    area = np.clip(boxes[:, 2] - boxes[:, 0], 1e-6, None) * np.clip(boxes[:, 3] - boxes[:, 1], 1e-6, None)
    return inside | (iw * ih / area >= frac)


def _snap_to_edges(rect, boxes, live, w, h, snap):
    """Extend an edge to the image border when it is within `snap` of it and no label lies in the strip in
    between: that strip is farther than the farthest labelled person, so it can only hold people nobody
    labelled, which is exactly what a far region should hide."""
    def blocked(sx1, sy1, sx2, sy2):
        iw = np.minimum(boxes[:, 2], sx2) - np.maximum(boxes[:, 0], sx1)
        ih = np.minimum(boxes[:, 3], sy2) - np.maximum(boxes[:, 1], sy1)
        return bool(((iw > 0) & (ih > 0) & live).any())

    x1, y1, x2, y2 = (float(v) for v in rect)
    if 0 < x1 < snap * w and not blocked(0, y1, x1, y2):
        x1 = 0.0
    if 0 < y1 < snap * h and not blocked(x1, 0, x2, y1):
        y1 = 0.0
    if 0 < w - x2 < snap * w and not blocked(x2, y1, w, y2):
        x2 = float(w)
    if 0 < h - y2 < snap * h and not blocked(x1, y2, x2, h):
        y2 = float(h)
    return np.array([x1, y1, x2, y2])


def _best_prefix(order, sm, bd, purity):
    """Longest run of labels (taken in `order`) that ends on a small label and keeps `purity`."""
    if len(order) == 0:
        return None
    cs, cb = np.cumsum(sm[order]), np.cumsum(bd[order])
    ok = (sm[order] == 1) & (cs / np.maximum(cs + cb, 1) >= purity)
    if not ok.any():
        return None
    return order[:int(np.where(ok)[0][-1]) + 1]  # cs grows at every ok index, so the last one holds the most


def _anchored(boxes, ctr, sm, bd, live, w, h, purity, min_small):
    """Border-anchored rectangles: 4 bands (beyond a line parallel to a border) and 4 corners (beyond a line in
    x and one in y), tight around the small labels they include and stretched to the border(s) they anchor to."""
    out = []
    li = np.where(live)[0]
    for axis, sign in ((1, 1), (1, -1), (0, 1), (0, -1)):
        order = li[np.argsort(ctr[li, axis] * sign, kind="stable")]
        mem = _best_prefix(order, sm, bd, purity)
        if mem is None or sm[mem].sum() < min_small:
            continue
        b = boxes[mem[sm[mem] == 1]]
        r = [0.0, 0.0, float(w), float(h)]
        if axis == 1:
            r[3 if sign == 1 else 1] = float(b[:, 3].max() if sign == 1 else b[:, 1].min())
        else:
            r[2 if sign == 1 else 0] = float(b[:, 2].max() if sign == 1 else b[:, 0].min())
        out.append(np.array(r))
    xs_small = np.unique(np.round(ctr[li][sm[li] == 1][:, 0]))
    for sx in (1, -1):
        for sy in (1, -1):
            best = None
            for xthr in (xs_small if sx == 1 else xs_small[::-1]):
                sub = li[(ctr[li, 0] * sx) <= xthr * sx]
                order = sub[np.argsort(ctr[sub, 1] * sy, kind="stable")]
                mem = _best_prefix(order, sm, bd, purity)
                if mem is None:
                    continue
                n_small = int(sm[mem].sum())
                if n_small < min_small:
                    continue
                b = boxes[mem[sm[mem] == 1]]
                r = [0.0, 0.0, float(w), float(h)]
                r[2 if sx == 1 else 0] = float(b[:, 2].max() if sx == 1 else b[:, 0].min())
                r[3 if sy == 1 else 1] = float(b[:, 3].max() if sy == 1 else b[:, 1].min())
                area = (r[2] - r[0]) * (r[3] - r[1])
                if best is None or (n_small, -area) > best[0]:
                    best = ((n_small, -area), np.array(r))
            if best is not None:
                out.append(best[1])
    return out


def _seeds(ctr, size, small, max_seeds):
    """Small labels to grow clusters from: the ones with the most small neighbours around them."""
    sidx = np.where(small)[0]
    if len(sidx) <= max_seeds:
        return sidx
    r = max(40.0, 8 * float(np.median(size[sidx])))
    d = np.abs(ctr[sidx][:, None, :] - ctr[sidx][None, :, :]).max(axis=2)
    return sidx[np.argsort(-(d < r).sum(axis=1), kind="stable")[:max_seeds]]


def _pockets(boxes, ctr, size, small, bad, purity, min_small, max_seeds):
    """Grow-from-seed rectangles for clusters that touch no border."""
    def counts(rect):
        x1, y1, x2, y2 = rect
        inside = (ctr[:, 0] >= x1) & (ctr[:, 0] <= x2) & (ctr[:, 1] >= y1) & (ctr[:, 1] <= y2)
        return int((inside & small).sum()), int((inside & bad).sum())

    sidx = np.where(small)[0]
    out = []
    for sd in _seeds(ctr, size, small, max_seeds):
        rect = boxes[sd].copy()
        for j in sidx[np.argsort(np.abs(ctr[sidx] - ctr[sd]).max(axis=1), kind="stable")]:
            cand = np.array([min(rect[0], boxes[j, 0]), min(rect[1], boxes[j, 1]),
                             max(rect[2], boxes[j, 2]), max(rect[3], boxes[j, 3])], dtype=np.float32)
            if np.array_equal(cand, rect):
                continue
            ns, nb = counts(cand)
            if ns / max(ns + nb, 1) >= purity:
                rect = cand
        ns, nb = counts(rect)
        if ns >= min_small and ns / max(ns + nb, 1) >= purity:
            out.append(rect.astype(float))
    return out


def _propose_far_rect(boxes, w, h, thr, purity, min_small, size, live, small, bad, ctr, max_seeds=40, pad_frac=0.5):
    """Best rectangle: border-anchored bands and corners, plus grown clusters, padded and stretched to nearby borders."""
    sm, bd = small.astype(int), bad.astype(int)
    cands = _anchored(boxes, ctr, sm, bd, live, w, h, purity, min_small)
    cands += _pockets(boxes, ctr, size, small, bad, purity, min_small, max_seeds)
    best = None
    for tight in cands:
        cov = _rect_covered(boxes, tight) & small
        pad = max(3.0, pad_frac * (float(np.median(size[cov])) if cov.any() else 6.0))
        padded = np.array([max(0.0, tight[0] - pad), max(0.0, tight[1] - pad),
                           min(float(w), tight[2] + pad), min(float(h), tight[3] + pad)])
        for rect in (_snap_to_edges(padded, boxes, live, w, h, SNAP_FRAC), padded, tight):  # most generous that stays pure
            c = _rect_covered(boxes, rect) & live
            ns, nb = int((c & small).sum()), int((c & bad).sum())
            if ns >= min_small and ns / max(ns + nb, 1) >= purity:
                if best is None or (ns, -(rect[2] - rect[0]) * (rect[3] - rect[1])) > best[0]:
                    best = ((ns, -(rect[2] - rect[0]) * (rect[3] - rect[1])), rect)
                break
    return best[1] if best else None


def _hull(points):
    return cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2).astype(np.float64)


def _extend_to_borders(poly, boxes, live, w, h, snap=SNAP_FRAC):
    """Move a vertex onto an image border it is near when that changes nothing about which labels are covered:
    beyond the farthest labelled people there are only unlabelled ones, which a far region should hide."""
    base = covered_by(boxes, poly) & live
    poly = poly.copy()
    for i in range(len(poly)):
        for axis, val in ((0, 0.0), (0, float(w)), (1, 0.0), (1, float(h))):
            dim = w if axis == 0 else h
            gap = abs(poly[i, axis] - val)
            if gap >= snap * dim or gap < 1e-6:
                continue
            trial = poly.copy()
            trial[i, axis] = val
            if np.array_equal(covered_by(boxes, trial) & live, base):
                poly = trial
    return poly


def propose_far_region(boxes, w, h, thr=FAR_PX, purity=FAR_PURITY, min_small=FAR_MIN_SMALL, exclude=None,
                       max_seeds=16, pad_frac=0.5):
    """Polygon that groups the small labels where (almost) nothing larger lives, or None.

    small = sqrt(w*h) < thr. Candidates: the best rectangle (border-anchored bands and corners, grown clusters),
    that rectangle tightened into the convex hull of the small labels it covers (same labels, fewer larger ones),
    and hulls grown directly from seed labels (they follow a slanted horizon where a rectangle cannot). Each is
    padded by half a typical small box and stretched to nearby borders where that changes nothing. The winner
    removes the most small labels while at least `purity` of everything it removes is small (ties: fewer larger
    labels, then less area). Labels in `exclude` (already under another region) take no part.
    Returns dict(poly (K,2), n_small, n_bad, purity, area_frac) or None."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    n = len(boxes)
    if n == 0:
        return None
    excl = np.zeros(n, dtype=bool) if exclude is None else np.asarray(exclude, dtype=bool)
    size = box_sizes(boxes)
    live = ~excl
    small, bad = (size < thr) & live, (size >= thr) & live
    if small.sum() < min_small:
        return None
    ctr = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    sidx = np.where(small)[0]
    pad = max(3.0, pad_frac * float(np.median(size[sidx])))

    def evaluate(poly):
        c = covered_by(boxes, poly) & live
        ns, nb = int((c & small).sum()), int((c & bad).sum())
        return ns >= min_small and ns / max(ns + nb, 1) >= purity, ns, nb

    best = [None]

    def consider(poly):
        if len(poly) < 3 or poly_area(poly) < MIN_FAR_AREA:
            return False
        ok, ns, nb = evaluate(poly)
        if ok:
            key = (ns, -nb, -poly_area(poly))
            if best[0] is None or key > best[0][0]:
                best[0] = (key, dict(poly=np.asarray(poly, dtype=np.float32), n_small=ns, n_bad=nb,
                                     purity=ns / max(ns + nb, 1), area_frac=poly_area(poly) / (w * h)))
        return ok

    def corners(i, p):
        x1, y1, x2, y2 = boxes[i]
        return [[x1 - p, y1 - p], [x2 + p, y1 - p], [x2 + p, y2 + p], [x1 - p, y2 + p]]

    def hull_of(idx):
        """Most generous valid hull of the labels `idx`: padded and stretched to borders, then padded, then bare."""
        for p in (pad, 0.0):
            hull = _hull([c for i in idx for c in corners(i, p)])
            if len(hull) < 3:
                continue
            for poly in (_extend_to_borders(hull, boxes, live, w, h), hull):
                if consider(poly):
                    return

    rect = _propose_far_rect(boxes, w, h, thr, purity, min_small, size, live, small, bad, ctr, pad_frac=pad_frac)
    if rect is not None:
        consider(rect_to_poly(rect))
        hull_of(np.where(_rect_covered(boxes, rect) & small)[0])   # the rectangle, tightened

    for sd in _seeds(ctr, size, small, max_seeds):                  # hulls grown directly from seeds
        mem, pts = [sd], corners(sd, pad)
        cur = _hull(pts)
        for j in sidx[np.argsort(np.abs(ctr[sidx] - ctr[sd]).max(axis=1), kind="stable")]:
            if j == sd:
                continue
            new = _hull(pts + corners(j, pad))
            if poly_area(new) <= poly_area(cur) + 1e-6:             # already inside: keep its corners
                pts, mem = pts + corners(j, pad), mem + [j]
                continue
            inside = points_in_polygon(ctr[:, 0], ctr[:, 1], new)
            ns, nb = int((inside & small).sum()), int((inside & bad).sum())
            if ns / max(ns + nb, 1) >= purity:
                pts, cur, mem = pts + corners(j, pad), new, mem + [j]
        if len(mem) >= min_small:
            hull_of(mem)
    return best[0][1] if best[0] else None


def extend_up(poly, samples=96):
    """Sweep a region up to the image top: the camera is level, so whatever is above a far spot is farther still.
    Keeps the lower edge of the shape and closes it along y = 0."""
    poly = np.asarray(poly, dtype=np.float64)
    x0, x1 = float(poly[:, 0].min()), float(poly[:, 0].max())
    if x1 - x0 < 1e-6:
        return poly.astype(np.float32)
    xs = np.unique(np.concatenate([poly[:, 0], np.linspace(x0, x1, samples)]))
    nxt = np.roll(poly, -1, axis=0)
    low = []
    for x in xs:
        ys = []
        for (ax, ay), (bx, by) in zip(poly, nxt):
            if ax == bx:
                if abs(ax - x) < 1e-9:
                    ys += [ay, by]
            elif min(ax, bx) - 1e-9 <= x <= max(ax, bx) + 1e-9:
                ys.append(ay + (by - ay) * (x - ax) / (bx - ax))
        if ys:
            low.append((x, max(ys)))
    low = np.array(low)
    out = np.vstack([low, [[x1, 0.0], [x0, 0.0]]]).astype(np.float32)
    out = cv2.approxPolyDP(out.reshape(-1, 1, 2), 0.75, True).reshape(-1, 2)
    return out if len(out) >= 3 else poly.astype(np.float32)


def grow_sideways(poly, x_left, x_right, samples=96):
    """Continue a region flat to the left and right: the top and bottom edges found at its two ends are held
    level out to x_left / x_right (a level camera: the far zone has the same height further along)."""
    poly = np.asarray(poly, dtype=np.float64)
    x0, x1 = float(poly[:, 0].min()), float(poly[:, 0].max())
    if x1 - x0 < 1e-6:
        return poly.astype(np.float32)
    xs = np.unique(np.concatenate([poly[:, 0], np.linspace(x0, x1, samples)]))
    nxt = np.roll(poly, -1, axis=0)
    top, low = [], []
    for x in xs:
        ys = []
        for (ax, ay), (bx, by) in zip(poly, nxt):
            if ax == bx:
                if abs(ax - x) < 1e-9:
                    ys += [ay, by]
            elif min(ax, bx) - 1e-9 <= x <= max(ax, bx) + 1e-9:
                ys.append(ay + (by - ay) * (x - ax) / (bx - ax))
        if ys:
            top.append((x, min(ys)))
            low.append((x, max(ys)))
    top, low = np.array(top), np.array(low)
    xl, xr = min(x_left, x0), max(x_right, x1)
    ring = np.vstack([[[xl, top[0, 1]]], top, [[xr, top[-1, 1]]], [[xr, low[-1, 1]]], low[::-1], [[xl, low[0, 1]]]])
    ring = cv2.approxPolyDP(ring.astype(np.float32).reshape(-1, 1, 2), 0.75, True).reshape(-1, 2)
    return ring.astype(np.float32) if len(ring) >= 3 else poly.astype(np.float32)


def _union_outline(pair, w, h, scale, gap):
    """One outline for the union of regions, or None when they are apart or the union has a hole."""
    m = np.zeros((int(np.ceil(h * scale)) + 2, int(np.ceil(w * scale)) + 2), np.uint8)
    for p in pair:
        cv2.fillPoly(m, [np.round(p * scale).astype(np.int32)], 255)
    if gap > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((gap, gap), np.uint8))  # joins regions a few px apart
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or len(cnts) != 1 or (hier[0][:, 3] >= 0).any():
        return None
    q = cv2.approxPolyDP(cnts[0], 1.0, True).reshape(-1, 2).astype(np.float32) / scale
    q[:, 0], q[:, 1] = np.clip(q[:, 0], 0, w), np.clip(q[:, 1], 0, h)
    return q if len(q) >= 3 and poly_area(q) >= MIN_FAR_AREA else None


def merge_polys(polys, w, h, scale=0.5, gap=3, accept=None):
    """Join regions that overlap or touch into one outline (raster union at half resolution, so +-2 px), pair by
    pair. A pair whose union would have a hole stays as it is (one outline cannot show a hole), and so does one
    that accept(union, a, b) refuses."""
    polys = [np.asarray(p, dtype=np.float32) for p in polys]
    changed = len(polys) > 1
    while changed:
        changed = False
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                u = _union_outline([polys[i], polys[j]], w, h, scale, gap)
                if u is not None and (accept is None or accept(u, polys[i], polys[j])):
                    polys = [q for k, q in enumerate(polys) if k not in (i, j)] + [u]
                    changed = True
                    break
            if changed:
                break
    return polys


def region_counts(boxes, polys, thr, taken=()):
    """(small, larger) labels that the polygons remove, not counting the ones `taken` regions already remove."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not len(boxes) or not len(polys):
        return 0, 0
    c = covered_by_any(boxes, polys) & ~covered_by_any(boxes, taken)
    size = box_sizes(boxes)
    return int((c & (size < thr)).sum()), int((c & (size >= thr)).sum())


def _clip_slab(poly, xa, xb):
    """The part of a polygon between the vertical lines x = xa and x = xb (Sutherland-Hodgman), or None."""
    pts = [tuple(map(float, q)) for q in poly]
    for val, keep_greater in ((xa, True), (xb, False)):
        if not pts:
            return None
        out, prev = [], pts[-1]
        for cur in pts:
            pin = prev[0] >= val if keep_greater else prev[0] <= val
            cin = cur[0] >= val if keep_greater else cur[0] <= val
            if pin != cin:
                t = (val - prev[0]) / (cur[0] - prev[0])
                out.append((val, prev[1] + t * (cur[1] - prev[1])))
            if cin:
                out.append(cur)
            prev = cur
        pts = out
    if len(pts) < 3:
        return None
    q = np.array(pts, dtype=np.float32)
    return q if poly_area(q) >= MIN_FAR_AREA else None


def _grow_side(boxes, w, poly, thr, free, size, block):
    """Continue a region flat left and right as far as it goes without taking in a label of >= block x thr."""
    if not len(boxes):
        return grow_sideways(poly, 0.0, float(w))
    lim = thr * block

    def bad(q):
        return int((covered_by(boxes, q) & free & (size >= lim)).sum())

    base = bad(poly)
    x0, x1 = float(poly[:, 0].min()), float(poly[:, 0].max())
    ext = []
    for left in (True, False):
        def reach(f):  # x reached when growing a fraction f of the way to the border
            return x0 - f * x0 if left else x1 + f * (w - x1)

        def grown(f):
            return grow_sideways(poly, reach(f), x1) if left else grow_sideways(poly, x0, reach(f))

        if bad(grown(1.0)) <= base:
            ext.append(reach(1.0))
            continue
        lo, hi = 0.0, 1.0
        for _ in range(9):
            mid = (lo + hi) / 2
            if bad(grown(mid)) <= base:
                lo = mid
            else:
                hi = mid
        ext.append(reach(lo))
    return grow_sideways(poly, ext[0], ext[1])


def stretch_region(boxes, w, h, poly, thr, up, side, taken=(), block=None, min_small=None):
    """Expand a proposed region; returns a LIST of regions (possibly empty).

    up: the region is stretched to the image top. Larger labels (>= thr) above it mean it is not a far region
    there, so the region is cut into vertical slices at those labels, the slices that can reach the top are kept
    (each at least SLICE_MIN x thr wide and with at least min_small small labels) and the rest is dropped. A single stray larger label (at most 5% of
    what is covered, at least 1) is tolerated. side: each region is continued flat to the left and right as far as
    it goes without taking in any label of at least block x thr (default SIDE_BLOCK x thr; smaller ones do not
    stop it). Labels already removed by the `taken` regions are ignored throughout."""
    block = SIDE_BLOCK if block is None else block
    min_small = FAR_MIN_SMALL if min_small is None else min_small
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    poly = np.asarray(poly, dtype=np.float32)
    have = len(boxes) > 0
    free = ~covered_by_any(boxes, taken) if have else None
    size = box_sizes(boxes) if have else None
    out = [poly]
    if up:
        def bad(q):
            return int((covered_by(boxes, q) & free & (size >= thr)).sum())

        def clean(base, ext):
            return bad(ext) <= bad(base) + max(1, int(0.05 * (covered_by(boxes, ext) & free).sum()))

        top = extend_up(poly)
        if not have or clean(poly, top):
            out = [top]
        else:
            above = covered_by(boxes, top) & ~covered_by(boxes, poly) & free & (size >= thr)
            blocks = sorted((float(b[0]) - 2.0, float(b[2]) + 2.0) for b in boxes[above])
            merged = []
            for lo, hi in blocks:
                if merged and lo <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            x0, x1 = float(poly[:, 0].min()), float(poly[:, 0].max())
            edges = [x0] + [v for m in merged for v in m] + [x1]
            out = []
            for xa, xb in zip(edges[0::2], edges[1::2]):
                if xb - xa < SLICE_MIN * thr:
                    continue
                piece = _clip_slab(poly, max(xa, x0), min(xb, x1))
                if piece is None:
                    continue
                ext = extend_up(piece)
                n_small = int((covered_by(boxes, ext) & free & (size < thr)).sum())
                if n_small >= min_small and clean(piece, ext):
                    out.append(ext)
    if side:
        out = [_grow_side(boxes, w, q, thr, free, size, block) for q in out]
    return out


def generate_regions(boxes, w, h, thr, purity, up, side, existing=(), max_regions=None, min_small=None):
    """The whole far-region pipeline for one image: propose regions among the labels `existing` regions do not
    cover yet (up to max_regions), expand each (see stretch_region; with `up`, what has larger labels above it is
    dropped), merge the new ones that overlap. Returns the new polygons only."""
    max_regions = AUTO_MAX if max_regions is None else max_regions
    min_small = FAR_MIN_SMALL if min_small is None else min_small
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    taken = [np.asarray(p, dtype=np.float32) for p in existing]
    new, dropped = [], []
    for _ in range(3 * max_regions):  # dropped proposals use up attempts too, but their labels are not proposed again
        if len(new) >= max_regions:
            break
        res = propose_far_region(boxes, w, h, thr, purity, min_small,
                                 exclude=covered_by_any(boxes, taken + new + dropped))
        if res is None:
            break
        poly = np.asarray(res["poly"], dtype=np.float32)
        out = stretch_region(boxes, w, h, poly, thr, up, side, taken=taken + new, min_small=min_small)
        if out:
            new.extend(out)
        else:
            dropped.append(poly)
    if len(new) > 1:
        big = box_sizes(boxes) >= thr

        def accept(u, a, b):  # a union may not take in more larger labels than the two had (edges are +-2 px)
            return (covered_by(boxes, u) & big).sum() <= (covered_by_any(boxes, [a, b]) & big).sum() + 1

        new = merge_polys(new, w, h, accept=accept if len(boxes) else None)
    return new


def darken(img, polys, x_off, y_off, factor=FAR_DARKEN):
    """Dim the parts of `img` under far regions. `img` is a crop whose top-left corner is (x_off, y_off)."""
    if not len(polys):
        return img
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    for poly in polys:
        draw.polygon([(float(x) - x_off, float(y) - y_off) for x, y in poly], fill=255)
    img.paste(img.point([int(i * factor) for i in range(256)] * 3), (0, 0), mask)
    return img


def far_path(input_dir, split):
    return Path(input_dir) / f"far_regions_{split}.json"


def _as_polygon(entry):
    """A polygon from a sidecar entry: a list of [x, y] corners (version 2) or [x1, y1, x2, y2] (version 1)."""
    try:
        if len(entry) == 4 and all(isinstance(v, (int, float)) for v in entry):
            poly = rect_to_poly(entry)
        else:
            poly = np.array([[float(p[0]), float(p[1])] for p in entry], dtype=np.float32)
    except (TypeError, ValueError, IndexError):
        return None
    return poly if poly.ndim == 2 and poly.shape[1] == 2 and len(poly) >= 3 else None


def load_far(path):
    """{image name: [polygons]} from the sidecar. Missing file: {}. Unparsable file: ValueError. Malformed or
    degenerate entries inside a readable file are skipped; version-1 rectangles load as 4-corner polygons."""
    path = Path(path)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    images = data.get("images", {}) if isinstance(data, dict) else {}
    out = {}
    for name, entries in (images.items() if isinstance(images, dict) else []):
        polys = [p for p in (_as_polygon(e) for e in (entries if isinstance(entries, list) else []))
                 if p is not None and poly_area(p) > 4.0]
        if polys:
            out[name] = polys
    return out


def save_far(path, items):
    """Write every image's regions (atomically); with no regions at all an existing file is emptied, none is created."""
    path = Path(path)
    images = {it.path.name: [[[round(float(x), 1), round(float(y), 1)] for x, y in poly] for poly in it.far]
              for it in items if len(it.far)}
    if not images and not path.exists():
        return
    data = {"version": 2, "images": images,
            "note": "far regions: polygons in image pixels, each a list of [x, y] corners; a label is removed when its "
                    "centre is inside one or at least half of its area is covered"}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def default_pred_path(input_dir, split, model_path):
    p = Path(model_path)
    stem = f"{p.parent.parent.name}_{p.stem}" if p.parent.name == "weights" else p.stem  # runs/x/weights/best.pt
    return Path(input_dir) / f"predictions_{split}_{stem}.json"


def run_predictions(model_path, images, imgsz, conf, device=None, progress=None, stop=None, nms_iou=0.5):
    from ultralytics import YOLO  # slow import, only when predicting

    model = YOLO(str(model_path), task="detect")
    out = {}
    for n, p in enumerate(images, 1):
        if stop is not None and stop.is_set():
            raise InterruptedError("stopped")
        # one image at a time: a list source is batched and can OOM at large imgsz
        r = model.predict(str(p), imgsz=imgsz, conf=conf, iou=nms_iou, max_det=1000, device=device, verbose=False)[0]
        xyxy, cf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        out[p.name] = [[round(float(v), 1) for v in b] + [round(float(c), 4)] for b, c in zip(xyxy, cf)]
        if progress:
            progress(n, len(images))
    return dict(model=str(model_path), imgsz=imgsz, conf=conf, nms_iou=nms_iou, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                images=out)


DETECTORS = ("YOLO", "RF-DETR")


def run_rfdetr_predictions(python, weights, images, conf, out_path, progress=None, stop=None, script=None):
    """RF-DETR predictions through rfdetr_predict.py in its own environment (rfdetr pulls in PyAV 18 / numpy 2.2,
    which must not replace the app's packages). Same cache format as run_predictions(); NMS-free."""
    import subprocess
    import tempfile
    from collections import deque

    script = Path(script) if script else Path(__file__).with_name("rfdetr_predict.py")
    if not Path(python).is_file():
        raise FileNotFoundError(f"RF-DETR environment not found: {python}\n"
                                f"Create it as described in {script.name} (or pass --rfdetr_python).")
    python = Path(python).resolve()  # Windows CreateProcess does not resolve a relative executable path
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(str(p) for p in images))
        listing = f.name
    cmd = [str(python), str(script), "--weights", str(weights), "--images", listing, "--out", str(out_path),
           "--conf", str(conf)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console window popping up on Windows
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            encoding="utf-8", errors="replace", creationflags=flags)
    tail = deque(maxlen=40)
    try:
        for line in proc.stdout:
            tail.append(line)
            if stop is not None and stop.is_set():
                proc.terminate()
                raise InterruptedError("stopped")
            if line.startswith("PROGRESS") and progress:
                _, n, t = line.split()
                if int(n):
                    progress(int(n), int(t))
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
        Path(listing).unlink(missing_ok=True)
    if rc != 0:
        raise RuntimeError("RF-DETR prediction failed:\n" + "".join(list(tail)[-15:]))
    return json.loads(Path(out_path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------- settings

SETTINGS_FILE = Path.home() / ".analyse_predictions_ui.json"  # same convention as review_predictions_ui.py


def _num(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def load_settings(path):
    """Saved UI settings, or {} when there are none or the file is unreadable."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(path, data):
    try:
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass  # remembering settings must never get in the way of the work


def resolve_settings(saved, argv):
    """Saved settings minus the ones the command line overrides: an explicit --conf wins."""
    out = dict(saved)
    if any(a == "--conf" or a.startswith("--conf=") for a in argv):
        out.pop("conf", None)
    if any(a == "--far_px" or a.startswith("--far_px=") for a in argv):
        out.pop("far_small_px", None)
    if any(a == "--detector" or a.startswith("--detector=") for a in argv):
        out.pop("detector", None)
    return out


def restore_geometry(root, geometry, default="1700x950"):
    """Window size from the last session, and its position only if that is still on screen."""
    m = re.fullmatch(r"(\d+)x(\d+)(?:\+(-?\d+)\+(-?\d+))?", str(geometry or ""))
    if not m:
        root.geometry(default)
        return
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = max(600, min(int(m[1]), sw)), max(400, min(int(m[2]), sh))
    if m[3] is not None and -20 <= int(m[3]) <= sw - 120 and -20 <= int(m[4]) <= sh - 120:
        root.geometry(f"{w}x{h}+{int(m[3])}+{int(m[4])}")
    else:
        root.geometry(f"{w}x{h}")


# --------------------------------------------------------------------------- UI

class App:
    def __init__(self, root, args, items, edges, settings=None, settings_path=None):
        self.root, self.args, self.items = root, args, items
        self.edges = np.asarray(edges, dtype=np.float64)
        self.labels = bucket_labels(edges)
        self.split_dir = Path(args.input_dir) / args.split
        self.backup_dir = Path(args.input_dir) / "_label_backups" / args.split
        self.trash_dir = Path(args.input_dir) / "_deleted" / args.split
        self.far_path = far_path(args.input_dir, args.split)
        self._far_global_dirty = False                                # sidecar needs a rewrite (e.g. image deleted)
        self._far_saved = False                                       # the last write included the sidecar
        st = settings if isinstance(settings, dict) else {}
        self.settings, self.settings_path = dict(st), settings_path   # no path (tests): nothing is remembered
        self._persisted, self._persist_job = None, None
        self._key = f"{Path(args.input_dir)}|{args.split}"            # per dataset: which image you were on
        conf0 = min(max(_num(st.get("conf"), args.conf), 0.0), 0.99)
        far0 = min(max(_num(st.get("far_small_px"), args.far_px), 2.0), 200.0)  # own key: the old far_px of 20 is dropped

        self.pred_data, self.pred_path = None, None
        self._conf = conf0
        for it in items:
            self._match_item(it)
        self._load_far()

        self.visible, self.pos = [], 0
        self.scale, self.fit_scale, self.center, self.fit_mode = 1.0, 1.0, (0.0, 0.0), True
        self._zoom_i = -1
        self._drag = None
        self._img_cache = {}
        self._photo = None
        self._render_job = None
        self._worker, self._q, self._stop = None, queue.Queue(), threading.Event()
        # annotation editing
        self.sel, self._sel_item = None, None   # selected annotation: index into _sel_item.gt
        self.sel_far = None                     # selected far region: index into _sel_item.far
        self.sel_pred = None                    # selected prediction (left panel): index into _sel_item.pred
        self._far_draft, self._far_cursor = [], None  # corners clicked so far while drawing a region
        self.draw_mode = self.draw_far_mode = False
        self._edit = None                       # mouse operation in progress on the annotation panel
        self._view = None                       # (x0, y0, scale) of the last render, for hit-testing
        self._cursor = "arrow"

        self.size_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="All")
        self.conf_var = tk.StringVar(value=f"{conf0:.2f}")
        self.show_labels = tk.BooleanVar(value=bool(st.get("labels", True)))
        self.filter_ann = tk.BooleanVar(value=bool(st.get("filter_annotations", False)))
        self.autosave = tk.BooleanVar(value=bool(st.get("autosave", True)))
        self.far_var = tk.StringVar(value=f"{far0:g}")
        det0 = {"yolo": "YOLO", "rfdetr": "RF-DETR"}.get(args.detector) or st.get("detector")
        self.detector_var = tk.StringVar(value=det0 if det0 in DETECTORS else "YOLO")
        self._build_ui()
        self._init_predictions()
        self._apply_settings()
        self._refilter(keep=False)
        self._restore_last_image()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- layout
    def _build_ui(self):
        r = self.root
        bar = tk.Frame(r, bd=1, relief=tk.RAISED)
        bar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(bar, text="< Prev", width=8, command=self.prev).pack(side=tk.LEFT, padx=3, pady=3)
        tk.Button(bar, text="Next >", width=8, command=self.next).pack(side=tk.LEFT, padx=3, pady=3)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)

        tk.Label(bar, text="Size").pack(side=tk.LEFT)
        self.cb_size = ttk.Combobox(bar, state="readonly", width=30, textvariable=self.size_var)
        self.cb_size.pack(side=tk.LEFT, padx=(3, 10))
        self._refresh_size_choices()
        self.cb_size.bind("<<ComboboxSelected>>", lambda e: self._on_filter())

        tk.Label(bar, text="Show").pack(side=tk.LEFT)
        self.cb_mode = ttk.Combobox(bar, state="disabled", width=9, values=MODES, textvariable=self.mode_var)
        self.cb_mode.pack(side=tk.LEFT, padx=(3, 10))
        self.cb_mode.bind("<<ComboboxSelected>>", lambda e: self._on_filter())

        tk.Label(bar, text="Conf").pack(side=tk.LEFT)
        self.sp_conf = ttk.Spinbox(bar, from_=self.args.save_conf, to=0.95, increment=0.05, width=5,
                                   textvariable=self.conf_var, command=self._on_conf, state="disabled")
        self.sp_conf.pack(side=tk.LEFT, padx=(3, 10))
        self.sp_conf.bind("<Return>", lambda e: (self._on_conf(), self.cv_gt.focus_set()))
        self.sp_conf.bind("<FocusOut>", lambda e: self._on_conf())

        tk.Checkbutton(bar, text="Filter annotations", variable=self.filter_ann,
                       command=self._render).pack(side=tk.LEFT, padx=(0, 6))
        tk.Checkbutton(bar, text="Labels", variable=self.show_labels, command=self._render).pack(side=tk.LEFT)
        tk.Button(bar, text="Fit", width=5, command=self.fit).pack(side=tk.LEFT, padx=3)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)
        tk.Label(bar, text="Detector").pack(side=tk.LEFT)
        self.cb_detector = ttk.Combobox(bar, state="readonly", width=8, values=DETECTORS,
                                        textvariable=self.detector_var)
        self.cb_detector.pack(side=tk.LEFT, padx=(3, 6))
        self.cb_detector.bind("<<ComboboxSelected>>", lambda e: self._on_detector())
        self.btn_run = tk.Button(bar, text="Run predictions", width=15, command=self._run_or_stop)
        self.btn_run.pack(side=tk.LEFT, padx=3)
        tk.Button(bar, text="Load predictions...", command=self._load_dialog).pack(side=tk.LEFT, padx=3)
        self.lbl_progress = tk.Label(bar, text="", fg="#0055aa")
        self.lbl_progress.pack(side=tk.LEFT, padx=8)

        bar2 = tk.Frame(r, bd=1, relief=tk.RAISED)
        bar2.pack(side=tk.TOP, fill=tk.X)
        tk.Label(bar2, text="Edit annotations:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(6, 4))
        self.btn_draw = tk.Button(bar2, text="Draw box [N]", width=13, command=self.toggle_draw)
        self.btn_delete = tk.Button(bar2, text="Delete [D]", width=11, command=self.delete_selected)
        self.btn_apply = tk.Button(bar2, text="Apply prediction [P]", width=19, command=self.apply_prediction)
        self.btn_apply_one = tk.Button(bar2, text="Apply one [O]", width=13, command=self.apply_one)
        self.btn_apply_all = tk.Button(bar2, text="Apply all predictions [A]", width=22,
                                       command=self.apply_all_predictions)
        self.btn_undo = tk.Button(bar2, text="Undo [Ctrl+Z]", width=13, command=self.undo)
        self.btn_save = tk.Button(bar2, text="Save (0) [Ctrl+S]", width=17, command=self.save)
        for b in (self.btn_draw, self.btn_delete, self.btn_apply, self.btn_apply_one, self.btn_apply_all,
                  self.btn_undo, self.btn_save):
            b.pack(side=tk.LEFT, padx=3, pady=3)
        tk.Checkbutton(bar2, text="Auto-save", variable=self.autosave).pack(side=tk.LEFT, padx=(6, 0))
        self._btn_bg = self.btn_save.cget("bg")
        self.btn_del_img = tk.Button(bar2, text="Delete image...", width=14, fg="#aa0000", command=self.delete_image)
        self.btn_del_img.pack(side=tk.RIGHT, padx=8, pady=3)
        self.lbl_sel = tk.Label(bar2, anchor="w", fg="#0055aa")
        self.lbl_sel.pack(side=tk.LEFT, padx=10)

        bar3 = tk.Frame(r, bd=1, relief=tk.RAISED)
        bar3.pack(side=tk.TOP, fill=tk.X)
        tk.Label(bar3, text="Far regions:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(6, 4))
        self.btn_far = tk.Button(bar3, text="Far Region [M]", width=14, command=self.far_region)
        self.btn_far_draw = tk.Button(bar3, text="Draw region [R]", width=15, command=self.toggle_draw_far)
        self.btn_far_clear = tk.Button(bar3, text="Clear regions", width=13, command=self.clear_far)
        for b in (self.btn_far, self.btn_far_draw, self.btn_far_clear):
            b.pack(side=tk.LEFT, padx=3, pady=3)
        self.up_var = tk.BooleanVar(value=bool(self.settings.get("far_up", True)))
        self.side_var = tk.BooleanVar(value=bool(self.settings.get("far_side", False)))
        self.all_var = tk.BooleanVar(value=False)
        for text, var in (("expand up", self.up_var), ("expand sideways", self.side_var),
                          ("apply to all images", self.all_var)):
            tk.Checkbutton(bar3, text=text, variable=var).pack(side=tk.LEFT)
        self.only_far = tk.BooleanVar(value=False)
        tk.Checkbutton(bar3, text="Only images with regions", variable=self.only_far,
                       command=self._on_filter).pack(side=tk.LEFT)
        tk.Label(bar3, text="small <").pack(side=tk.LEFT, padx=(10, 2))
        self.sp_far = ttk.Spinbox(bar3, from_=3, to=200, increment=1, width=4, textvariable=self.far_var,
                                  command=self._render)
        self.sp_far.pack(side=tk.LEFT)
        self.sp_far.bind("<Return>", lambda e: (self._render(), self.cv_gt.focus_set()))
        self.sp_far.bind("<FocusOut>", lambda e: self._render())
        tk.Label(bar3, text="px").pack(side=tk.LEFT, padx=(2, 10))
        self.lbl_far = tk.Label(bar3, anchor="w", fg="#aa00aa")
        self.lbl_far.pack(side=tk.LEFT, padx=6)

        self.status = tk.Label(r, anchor="w", relief=tk.SUNKEN, bd=1)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        body = tk.Frame(r)
        body.pack(fill=tk.BOTH, expand=True)
        for c in (0, 1):
            body.columnconfigure(c, weight=1, uniform="panel")
        body.rowconfigure(1, weight=1)
        bold = ("Segoe UI", 10, "bold")
        self.hdr_pred = tk.Label(body, anchor="w", font=bold)
        self.hdr_gt = tk.Label(body, anchor="w", font=bold)
        self.hdr_pred.grid(row=0, column=0, sticky="ew", padx=(4, 2))
        self.hdr_gt.grid(row=0, column=1, sticky="ew", padx=(2, 4))
        self.cv_pred = tk.Canvas(body, bg="#1e1e1e", highlightthickness=0)
        self.cv_gt = tk.Canvas(body, bg="#1e1e1e", highlightthickness=0)
        self.cv_pred.grid(row=1, column=0, sticky="nsew", padx=(0, 2))
        self.cv_gt.grid(row=1, column=1, sticky="nsew", padx=(2, 0))
        for cv in (self.cv_pred, self.cv_gt):
            cv.bind("<MouseWheel>", self._on_wheel)
            cv.bind("<Configure>", lambda e: self._schedule_render())
        self.cv_pred.bind("<ButtonPress-1>", self._on_press)       # left panel: drag pans, a click selects a prediction
        self.cv_pred.bind("<B1-Motion>", self._on_drag)
        self.cv_pred.bind("<ButtonRelease-1>", self._pred_release)
        self.cv_pred.bind("<Double-Button-1>", lambda e: self.fit())
        self.cv_gt.bind("<ButtonPress-1>", self._gt_press)         # right panel: select / move / resize / draw
        self.cv_gt.bind("<B1-Motion>", self._gt_drag)
        self.cv_gt.bind("<ButtonRelease-1>", self._gt_release)
        self.cv_gt.bind("<Motion>", self._gt_hover)
        self.cv_gt.bind("<Double-Button-1>", self._gt_dclick)      # add a corner / close a region being drawn
        self.cv_gt.bind("<Button-3>", self._gt_rclick)             # remove a corner

        def nav(fn):  # not while typing in the confidence box
            return lambda e: None if isinstance(e.widget, ttk.Spinbox) else fn()

        def edit(fn):  # destructive keys also stay out of the dropdowns
            return lambda e: None if isinstance(e.widget, (ttk.Spinbox, ttk.Combobox)) else fn()

        r.bind("<Left>", lambda e: self.prev())   # no guard: they must work from the Conf box too
        r.bind("<Right>", lambda e: self.next())
        for keys, fn in ((("z", "Z"), self._zoom_next), (("f", "F", "<Home>"), self.fit),
                         (("l", "L"), self._toggle_labels)):
            for k in keys:
                r.bind(k, nav(fn))
        for keys, fn in ((("d", "D", "<Delete>"), self.delete_selected), (("n", "N"), self.toggle_draw),
                         (("m", "M"), self.far_region), (("r", "R"), self.toggle_draw_far),
                         (("<Return>",), self.far_draft_finish), (("<BackSpace>",), self.far_draft_back),
                         (("p", "P"), self.apply_prediction), (("o", "O"), self.apply_one),
                         (("a", "A"), self.apply_all_predictions), (("<Control-s>", "<Control-S>"), self.save),
                         (("<Control-z>", "<Control-Z>"), self.undo), (("<Escape>",), self.cancel)):
            for k in keys:
                r.bind(k, edit(fn))

    def _refresh_size_choices(self):
        """Size dropdown with object / image counts; recounted after every edit so they stay true."""
        i = max(0, self.cb_size.current())
        n_obj = np.bincount(np.concatenate([it.gt_bucket for it in self.items] or [np.zeros(0, int)]),
                            minlength=len(self.labels))
        n_img = np.zeros(len(self.labels), dtype=int)
        for it in self.items:
            n_img[np.unique(it.gt_bucket)] += 1
        self.size_choices = [f"All  ({int(n_obj.sum())} obj / {len(self.items)} img)"] + \
            [f"{lab} px  ({int(n_obj[b])} obj / {int(n_img[b])} img)" for b, lab in enumerate(self.labels)]
        self.cb_size.config(values=self.size_choices)
        self.cb_size.current(i)

    # ---------------- predictions
    def _far_px(self):
        return max(2.0, _num(self.far_var.get(), self.args.far_px))

    def _load_far(self):
        """Far regions of every image from the sidecar next to the dataset."""
        try:
            far = load_far(self.far_path)
        except (ValueError, OSError) as ex:
            backup = self.far_path.with_name(self.far_path.name + ".corrupt")
            try:
                shutil.copy2(self.far_path, backup)
            except OSError:
                pass
            messagebox.showwarning("Far regions", f"{self.far_path.name} could not be read ({ex}).\n"
                                                  f"A copy was kept as {backup.name}; saving will replace it.")
            far = {}
        for it in self.items:
            polys = []
            for poly in far.get(it.path.name, []):
                poly = poly.copy()
                poly[:, 0], poly[:, 1] = np.clip(poly[:, 0], 0, it.w), np.clip(poly[:, 1], 0, it.h)
                if poly_area(poly) >= MIN_FAR_AREA / 4:
                    polys.append(poly)
            it.far = polys

    def _rfdetr(self):
        return self.detector_var.get() == "RF-DETR"

    def _weights(self):
        """Weights of the selected detector (YOLO: --humans_model_path, RF-DETR: --rfdetr_weights)."""
        return self.args.rfdetr_weights if self._rfdetr() else self.args.humans_model_path

    def _on_detector(self):
        """Switch detector: show its cached predictions if there are any, so the two can be compared."""
        self.cv_gt.focus_set()
        if not self._weights():
            self._render(note=f"no {self.detector_var.get()} weights set - Run predictions asks for them")
            return
        path = default_pred_path(self.args.input_dir, self.args.split, self._weights())
        if path.is_file():
            self._attach(json.loads(path.read_text()), path)
            self._refilter(keep=True)
            self._render(note=f"{self.detector_var.get()}: loaded {path.name}")
        else:
            self._render(note=f"no {self.detector_var.get()} predictions for this split yet - press Run predictions "
                              f"({Path(self._weights()).name})")
        self._persist_soon()

    def _init_predictions(self):
        a, path = self.args, None
        if a.predictions_json:
            path = Path(a.predictions_json)
        elif self._weights():
            path = default_pred_path(a.input_dir, a.split, self._weights())
        else:  # nothing named: pick up the newest cache for this split, if any
            found = sorted(Path(a.input_dir).glob(f"predictions_{a.split}_*.json"), key=lambda p: p.stat().st_mtime)
            path = found[-1] if found else None
        if path and path.is_file():
            self._attach(json.loads(path.read_text()), path)

    def _attach(self, data, path):
        self.pred_data, self.pred_path = data, Path(path)
        preds = data.get("images", {})
        missing = 0
        for it in self.items:
            boxes = preds.get(it.path.name)
            if boxes is None:
                missing += 1
                boxes = []
            arr = np.array(boxes, dtype=np.float32).reshape(-1, 5)
            order = np.argsort(-arr[:, 4], kind="stable")
            it.pred, it.pred_conf = arr[order, :4], arr[order, 4]
        floor = float(data.get("conf", self.args.save_conf))
        self.sp_conf.config(state="normal", from_=floor)
        if float(self.conf_var.get() or 0) < floor:
            self.conf_var.set(f"{floor:.2f}")
        self.cb_mode.config(state="readonly")
        if missing:
            messagebox.showwarning("Predictions", f"{missing} of {len(self.items)} images have no entry in\n"
                                                  f"{path.name}\n(made from a different dataset?). "
                                                  f"They are treated as having no detections.")
        self._recompute()

    def _match_item(self, it):
        """Sizes, buckets and prediction matching of one image; rerun after any edit or conf change."""
        it.gt_size = box_sizes(it.gt)
        it.gt_bucket = np.searchsorted(self.edges, it.gt_size, side="right")
        it.k = int(np.sum(it.pred_conf >= self._conf))
        it.p2g, it.hit = greedy_match(it.pred[:it.k], it.gt, self.args.iou)
        matched = it.p2g >= 0
        size = box_sizes(it.pred[:it.k])
        size[matched] = it.gt_size[it.p2g[matched]]  # TP: bucket of the annotation it hit
        it.p_bucket = np.searchsorted(self.edges, size, side="right")

    def _recompute(self):
        """Match predictions to annotations at the current confidence threshold."""
        try:
            self._conf = float(self.conf_var.get())
        except ValueError:
            self._conf = self.args.conf
        for it in self.items:
            self._match_item(it)

    def _apply_settings(self):
        """Size filter and Show mode from the last session (Show only once predictions are loaded)."""
        size = self.settings.get("size")
        if size in self.labels:
            self.cb_size.current(self.labels.index(size) + 1)
        mode = self.settings.get("mode")
        if self.pred_data is not None and mode in MODES:
            self.mode_var.set(mode)

    def _restore_last_image(self):
        name = self.settings.get("last_image", {}).get(self._key)
        names = [self.items[i].path.name for i in self.visible]
        if name in names:  # silently the first image when it is gone or filtered out
            self.pos = names.index(name)
            self._new_image()

    def _persist_soon(self):
        if self.settings_path is None:
            return
        if self._persist_job:
            self.root.after_cancel(self._persist_job)
        self._persist_job = self.root.after(600, self._persist)

    def _persist(self):
        """Write the settings if anything changed since the last write."""
        if self._persist_job:
            self.root.after_cancel(self._persist_job)
        self._persist_job = None
        if self.settings_path is None:
            return
        s = dict(self.settings)
        last = dict(s.get("last_image", {}))
        it = self._cur()
        if it is not None:
            last[self._key] = it.path.name
        b = self._bucket()
        s.update(conf=round(_num(self.conf_var.get(), self._conf), 4), size="All" if b is None else self.labels[b],
                 filter_annotations=bool(self.filter_ann.get()), labels=bool(self.show_labels.get()),
                 autosave=bool(self.autosave.get()), far_small_px=round(self._far_px(), 2),
                 far_up=bool(self.up_var.get()), far_side=bool(self.side_var.get()), last_image=last,
                 detector=self.detector_var.get())
        if self.pred_data is not None:  # without predictions the Show box is inert: keep the saved value
            s["mode"] = self.mode_var.get()
        if self.root.state() == "normal":
            s["geometry"] = self.root.geometry()
        if s != self._persisted:
            save_settings(self.settings_path, s)
            self._persisted = self.settings = s

    def _load_dialog(self):
        f = filedialog.askopenfilename(title="Predictions JSON", initialdir=self.args.input_dir,
                                       filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if f:
            self._attach(json.loads(Path(f).read_text()), f)
            self._refilter(keep=True)

    def _run_or_stop(self):
        if self._worker and self._worker.is_alive():
            self._stop.set()
            self.btn_run.config(text="Stopping...")
            return
        rfdetr = self._rfdetr()
        model = self._weights()
        if not model or not Path(model).is_file():
            types = ([("RF-DETR checkpoint", "*.pth"), ("All", "*.*")] if rfdetr
                     else [("YOLO weights", "*.pt *.engine *.onnx"), ("All", "*.*")])
            model = filedialog.askopenfilename(title=f"{self.detector_var.get()} weights", filetypes=types)
            if not model:
                return
            if rfdetr:
                self.args.rfdetr_weights = model
            else:
                self.args.humans_model_path = model
        out = (Path(self.args.predictions_json) if self.args.predictions_json
               else default_pred_path(self.args.input_dir, self.args.split, model))
        if out.exists() and not messagebox.askyesno("Run predictions", f"{out.name} exists. Overwrite?"):
            return
        images = [it.path for it in self.items]
        self._stop.clear()

        def work():
            try:
                progress = lambda n, t: self._q.put(("progress", n, t))  # noqa: E731
                if rfdetr:  # writes the cache itself
                    data = run_rfdetr_predictions(self.args.rfdetr_python, model, images, self.args.save_conf, out,
                                                  progress=progress, stop=self._stop)
                else:
                    data = run_predictions(model, images, self.args.imgsz, self.args.save_conf, self.args.device,
                                           nms_iou=self.args.nms_iou, progress=progress, stop=self._stop)
                    out.write_text(json.dumps(data))
                self._q.put(("done", data, out))
            except InterruptedError:
                self._q.put(("stopped",))
            except Exception as e:  # surface it in the UI rather than dying silently in a thread
                self._q.put(("error", f"{type(e).__name__}: {e}"))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()
        self.btn_run.config(text="Stop")
        self.lbl_progress.config(text="loading model...")
        self._poll()

    def _poll(self):
        while not self._q.empty():
            msg = self._q.get()
            if msg[0] == "progress":
                self.lbl_progress.config(text=f"predicting {msg[1]}/{msg[2]}")
            elif msg[0] == "done":
                self.lbl_progress.config(text=f"saved {msg[2].name}")
                self._attach(msg[1], msg[2])
                self._refilter(keep=True)
            elif msg[0] == "stopped":
                self.lbl_progress.config(text="stopped")
            elif msg[0] == "error":
                self.lbl_progress.config(text="failed")
                messagebox.showerror("Run predictions", msg[1])
        if self._worker and self._worker.is_alive():
            self.root.after(200, self._poll)
        else:
            self.btn_run.config(text="Run predictions")

    # ---------------- filtering
    def _bucket(self):
        i = self.cb_size.current()
        return None if i <= 0 else i - 1

    def _masks(self, it, display=False):
        """Annotations and predictions of `it` selected by the current filters.

        The size bucket and Match / No match always filter the predictions. For the annotations they
        decide which images are listed and which objects Z visits (so No match still finds images with
        missed annotations). What is DRAWN (display=True) depends on "Filter annotations": off, every
        annotation is shown, so the predictions can be judged and the boxes edited against the full
        ground truth; on, annotations are filtered by SIZE only, never by Match / No match, so a
        missed annotation is still seen next to the ones that were hit. The selected annotation is
        always drawn, so a box you resize into another size bucket does not vanish."""
        b, mode = self._bucket(), self.mode_var.get()
        in_bucket = np.ones(len(it.gt), dtype=bool) if b is None else it.gt_bucket == b
        gmask = in_bucket
        pmask = np.ones(it.k, dtype=bool) if b is None else it.p_bucket == b
        if self.pred_data is not None:
            matched = it.p2g >= 0
            if mode == "Match":
                gmask, pmask = in_bucket & it.hit, pmask & matched
            elif mode == "No match":
                gmask, pmask = in_bucket & ~it.hit, pmask & ~matched
        if display:
            gmask = in_bucket.copy() if self.filter_ann.get() else np.ones(len(it.gt), dtype=bool)
            if self._sel_item is it and self.sel is not None and self.sel < len(gmask):
                gmask[self.sel] = True
        return gmask, pmask

    def _on_filter(self):
        self._refilter(keep=True)
        self.cv_gt.focus_set()  # so the arrow keys page images instead of changing the combobox

    def _on_conf(self):
        if self.pred_data is None:
            return
        try:
            c = float(self.conf_var.get())
        except ValueError:
            self.conf_var.set(f"{self._conf:.2f}")
            return
        if abs(c - getattr(self, "_conf", -1)) > 1e-9:
            self._recompute()
            self._refilter(keep=True)

    def _qualifying(self, pin=None):
        out = self._qualifying_base(pin)
        if self.only_far.get():
            out = [i for i in out if len(self.items[i].far) or i == pin]
        return out

    def _qualifying_base(self, pin=None):
        """Indices of the images with at least one annotation or prediction that satisfies the filters
        right now. `pin` (the image on screen) is kept even if it no longer qualifies, so fixing the last
        problem in an image does not yank it away; it drops out of the list as soon as you move on."""
        if self._bucket() is None and self.mode_var.get() == "All":
            return list(range(len(self.items)))  # everything, backgrounds included
        out = []
        for i, it in enumerate(self.items):
            if i == pin:
                out.append(i)
                continue
            g, p = self._masks(it)
            if g.any() or p.any():
                out.append(i)
        return out

    def _sync_list(self):
        """Re-evaluate the list after an edit, so counts and Next / Prev reflect what is left to do."""
        cur = self.visible[self.pos] if self.visible else None
        self.visible = self._qualifying(pin=cur)
        self.pos = self.visible.index(cur) if cur in self.visible else 0

    def _refilter(self, keep):
        current = self.visible[self.pos] if keep and self.visible else None
        self.visible = self._qualifying()
        self.pos = self.visible.index(current) if current in self.visible else 0
        self._new_image()

    # ---------------- navigation
    def _step(self, d):
        if not self.visible:
            return
        ok, saved = self._autosave()
        if not ok:
            self._render(note="Auto-save failed - staying on this image")
            return
        self.pos = (self.pos + d) % len(self.visible)
        cur = self.visible[self.pos]
        self.visible = self._qualifying(pin=cur)  # drops the image just left if it was fixed meanwhile
        self.pos = self.visible.index(cur)
        self._new_image(note=self._saved_note("Auto-saved", saved))

    def prev(self):
        self._step(-1)

    def next(self):
        self._step(1)

    def _new_image(self, note=""):
        self.fit_mode, self._zoom_i = True, -1
        self.sel, self.sel_far, self.sel_pred, self._sel_item, self._edit = None, None, None, None, None
        self.draw_mode = self.draw_far_mode = False
        self._far_draft, self._far_cursor = [], None
        self._render(note=note)

    def fit(self):
        self.fit_mode = True
        self._render()

    def _toggle_labels(self):
        self.show_labels.set(not self.show_labels.get())
        self._render()

    def _panel(self):
        return max(1, self.cv_gt.winfo_width()), max(1, self.cv_gt.winfo_height())

    def _zoom_next(self):
        it = self._cur()
        if it is None:
            return
        g, p = self._masks(it)
        # annotations, plus predictions that are not already on an annotation (those would repeat)
        cand = [("gt", j) for j in np.where(g)[0]] + [("pred", j) for j in np.where(p & (it.p2g < 0))[0]]
        if not cand:
            return
        self._zoom_i = (self._zoom_i + 1) % len(cand)
        kind, j = cand[self._zoom_i]
        x1, y1, x2, y2 = (it.gt if kind == "gt" else it.pred)[j]
        pw, ph = self._panel()
        side = max(x2 - x1, y2 - y1, 1.0)
        # box ~1/16 of the panel: close enough to see the pixels, wide enough to judge if it is a person
        self.scale = min(max(min(pw, ph) / (side * 16), self.fit_scale), MAX_ZOOM)
        self.center, self.fit_mode = ((x1 + x2) / 2, (y1 + y2) / 2), False
        if kind == "gt":  # land on it selected, ready for Delete / Apply prediction
            self._select(it, j)
        else:
            self._deselect()
        self._render(note=f"object {self._zoom_i + 1}/{len(cand)}")

    # ---------------- view
    def _on_wheel(self, e):
        if not self.visible:
            return
        pw, ph = self._panel()
        s, (cx, cy) = self.scale, self.center
        px, py = cx + (e.x - pw / 2) / s, cy + (e.y - ph / 2) / s  # image point under the cursor
        s2 = min(max(s * (1.25 if e.delta > 0 else 0.8), self.fit_scale * 0.5), MAX_ZOOM)
        self.scale, self.center = s2, (px - (e.x - pw / 2) / s2, py - (e.y - ph / 2) / s2)
        self.fit_mode = False
        self._render()

    def _on_press(self, e):
        e.widget.focus_set()
        self._drag = (e.x, e.y, self.center)

    def _pred_release(self, e):
        """Left panel: a release without movement is a click - select the prediction under it (or none)."""
        drag, self._drag = self._drag, None
        if drag is None or abs(e.x - drag[0]) > CLICK_SLOP or abs(e.y - drag[1]) > CLICK_SLOP:
            return  # it was a pan
        it = self._cur()
        if it is None or self._view is None:
            return
        k = self._pick_pred(it, e.x, e.y)
        if k is None:
            self.sel_pred = None
        else:
            self.sel, self.sel_far, self.sel_pred, self._sel_item = None, None, k, it
        self._render()

    def _on_drag(self, e):
        if self._drag:
            x0, y0, (cx, cy) = self._drag
            self.center = (cx - (e.x - x0) / self.scale, cy - (e.y - y0) / self.scale)
            self.fit_mode = False
            self._render()

    # ---------------- annotation editing (right panel)
    def _cur(self):
        return self.items[self.visible[self.pos]] if self.visible else None

    def _to_img(self, ex, ey):
        x0, y0, s = self._view
        return x0 + ex / s, y0 + ey / s

    def _to_scr(self, box):
        x0, y0, s = self._view
        return (box[0] - x0) * s, (box[1] - y0) * s, (box[2] - x0) * s, (box[3] - y0) * s

    @staticmethod
    def _handle_points(rect):
        a, b, c, d = rect
        mx, my = (a + c) / 2, (b + d) / 2
        return {"nw": (a, b), "n": (mx, b), "ne": (c, b), "e": (c, my),
                "se": (c, d), "s": (mx, d), "sw": (a, d), "w": (a, my)}

    def _handle_at(self, rect, ex, ey):
        """Handle of a screen-space rect under the cursor, nearest wins; None if the box is too small to grab."""
        if min(rect[2] - rect[0], rect[3] - rect[1]) < HANDLE_MIN_SIDE:
            return None
        best, best_dist = None, HANDLE_HIT + 1
        for name, (hx, hy) in self._handle_points(rect).items():
            dist = max(abs(ex - hx), abs(ey - hy))
            if dist <= HANDLE_HIT and dist < best_dist:
                best, best_dist = name, dist
        return best

    def _pick(self, it, ex, ey):
        """Smallest shown annotation under the cursor (smallest, so a box nested in another stays reachable)."""
        g, _ = self._masks(it, display=True)
        best, best_area = None, None
        for j in np.where(g)[0]:
            a, b, c, d = self._to_scr(it.gt[j])
            if a - PICK_TOL <= ex <= c + PICK_TOL and b - PICK_TOL <= ey <= d + PICK_TOL:
                area = max(c - a, 1.0) * max(d - b, 1.0)
                if best is None or area < best_area:
                    best, best_area = j, area
        return best

    def _pick_pred(self, it, ex, ey):
        """Smallest shown prediction under the cursor, as an index into it.pred, or None."""
        _, p = self._masks(it, display=True)
        best, best_area = None, None
        for k in np.where(p)[0]:
            a, b, c, d = self._to_scr(it.pred[k])
            if a - PICK_TOL <= ex <= c + PICK_TOL and b - PICK_TOL <= ey <= d + PICK_TOL:
                area = max(c - a, 1.0) * max(d - b, 1.0)
                if best is None or area < best_area:
                    best, best_area = int(k), area
        return best

    def _select(self, it, j):
        self.sel, self.sel_far, self.sel_pred, self._sel_item = j, None, None, it

    def _select_far(self, it, k):
        self.sel, self.sel_far, self.sel_pred, self._sel_item = None, k, None, it

    def _deselect(self):
        self.sel, self.sel_far, self.sel_pred, self._sel_item = None, None, None, None

    def _selected(self):
        """(item, index) of the selected annotation of the image being shown, else None."""
        it = self._cur()
        if it is None or self.sel is None or self._sel_item is not it or self.sel >= len(it.gt):
            return None
        return it, self.sel

    def _selected_pred(self):
        """(item, index) of the selected prediction of the image being shown, if it is still shown, else None."""
        it = self._cur()
        if it is None or self.sel_pred is None or self._sel_item is not it or self.sel_pred >= it.k:
            return None
        _, p = self._masks(it, display=True)
        return (it, self.sel_pred) if p[self.sel_pred] else None

    def _selected_far(self):
        """(item, index) of the selected far region of the image being shown, else None."""
        it = self._cur()
        if it is None or self.sel_far is None or self._sel_item is not it or self.sel_far >= len(it.far):
            return None
        return it, self.sel_far

    def _poly_scr(self, poly):
        x0, y0, s = self._view
        return np.stack([(poly[:, 0] - x0) * s, (poly[:, 1] - y0) * s], 1)

    @staticmethod
    def _nearest_edge(pts, ex, ey):
        """(distance, nearest point, edge index) from (ex, ey) to the closed polygon `pts`, in screen px."""
        a, b = pts, np.roll(pts, -1, axis=0)
        ab = b - a
        t = np.clip(((ex - a[:, 0]) * ab[:, 0] + (ey - a[:, 1]) * ab[:, 1]) / np.maximum((ab ** 2).sum(axis=1), 1e-9), 0, 1)
        proj = a + ab * t[:, None]
        d = np.hypot(proj[:, 0] - ex, proj[:, 1] - ey)
        i = int(np.argmin(d))
        return float(d[i]), proj[i], i

    def _vertex_at(self, poly, ex, ey):
        """Corner of a region under the cursor, or None."""
        sp = self._poly_scr(poly)
        d = np.hypot(sp[:, 0] - ex, sp[:, 1] - ey)
        i = int(np.argmin(d))
        return i if d[i] <= HANDLE_HIT else None

    def _pick_far(self, it, ex, ey, border_only=False):
        """Smallest far region under the cursor. border_only: only a click within PICK_TOL of its outline counts,
        so a region can be grabbed even where annotations lie inside it."""
        best, best_area = None, None
        for k, poly in enumerate(it.far):
            sp = self._poly_scr(poly)
            on_edge = self._nearest_edge(sp, ex, ey)[0] <= PICK_TOL
            if not (on_edge or (not border_only and bool(points_in_polygon([ex], [ey], sp)[0]))):
                continue
            if best is None or poly_area(sp) < best_area:
                best, best_area = k, poly_area(sp)
        return best

    def _can_edit(self, it):
        if it.editable:
            return True
        self._render(note="This label file has rows that are not plain 'class cx cy w h'; "
                          "editing it would drop data, so it is read-only here")
        return False

    def _set_cursor(self, cur):
        if cur != self._cursor:
            try:
                self.cv_gt.config(cursor=cur)
            except tk.TclError:  # the size_* cursors are Windows-only
                self.cv_gt.config(cursor="sizing" if cur.startswith("size") else "arrow")
            self._cursor = cur

    def _gt_hover(self, e):
        it = self._cur()
        cur = "arrow"
        if it is not None and self._view is not None and self._edit is None:
            if self.draw_far_mode:
                cur = "crosshair"
                if self._far_draft:                       # the rubber band follows the cursor
                    self._far_cursor = self._to_img(e.x, e.y)
                    self._render()
            elif self.draw_mode:
                cur = "crosshair"
            else:
                picked, pf = self._selected(), self._selected_far()
                name = self._handle_at(self._to_scr(it.gt[picked[1]]), e.x, e.y) if picked else None
                if name:
                    cur = HANDLE_CURSOR[name]
                elif pf and self._vertex_at(it.far[pf[1]], e.x, e.y) is not None:
                    cur = "hand2"
                elif self._pick(it, e.x, e.y) is not None or self._pick_far(it, e.x, e.y) is not None:
                    cur = "fleur"
        self._set_cursor(cur)

    def _gt_press(self, e):
        self.cv_gt.focus_set()
        it = self._cur()
        self._drag, self._edit = None, None
        if it is None or self._view is None:
            return
        px, py = self._to_img(e.x, e.y)
        if self.draw_far_mode:   # a click adds a corner; a drag before the first corner draws a rectangle
            self._edit = dict(kind="draw_far", target="far", start=(px, py), press=(e.x, e.y), box=None,
                              snap=it.snapshot())
            return
        if self.draw_mode:
            if self._can_edit(it):
                start = (min(max(px, 0.0), it.w), min(max(py, 0.0), it.h))
                self._edit = dict(kind="draw", target="gt", start=start, box=None, snap=it.snapshot())
            return
        picked, pf = self._selected(), self._selected_far()
        if picked:
            name = self._handle_at(self._to_scr(it.gt[picked[1]]), e.x, e.y)
            if name and self._can_edit(it):
                self._edit = dict(kind="resize", target="gt", handle=name, snap=it.snapshot(),
                                  orig=it.gt[picked[1]].copy())
                return
        if pf:
            v = self._vertex_at(it.far[pf[1]], e.x, e.y)
            if v is not None:
                self._edit = dict(kind="vertex", target="far", index=v, snap=it.snapshot(), orig=it.far[pf[1]].copy())
                return
        k = self._pick_far(it, e.x, e.y, border_only=True)      # an outline wins over the annotations inside it
        j = None if k is not None else self._pick(it, e.x, e.y)
        if k is None and j is None:
            k = self._pick_far(it, e.x, e.y)                     # the inside of a region, where no annotation is
        if k is not None:
            self._select_far(it, k)
            self._edit = dict(kind="move", target="far", start=(px, py), snap=it.snapshot(), orig=it.far[k].copy())
            self._render()
            return
        if j is not None:
            self._select(it, j)
            if self._can_edit(it):
                self._edit = dict(kind="move", target="gt", start=(px, py), snap=it.snapshot(), orig=it.gt[j].copy())
            else:
                self._render(note="read-only label file (not plain 'class cx cy w h' rows)")
            if self._edit:
                self._render()
            return
        self._deselect()
        self._drag = (e.x, e.y, self.center)  # empty space: pan, like the left panel
        self._render()

    def _gt_drag(self, e):
        st = self._edit
        if st is None:
            self._on_drag(e)
            return
        it = self._cur()
        rx, ry = self._to_img(e.x, e.y)
        cx, cy = min(max(rx, 0.0), it.w), min(max(ry, 0.0), it.h)
        kind, far = st["kind"], st.get("target") == "far"
        if kind == "draw_far":
            if not self._far_draft and np.hypot(e.x - st["press"][0], e.y - st["press"][1]) > 6:
                x0, y0 = st["start"]
                st["box"] = (min(x0, cx), min(y0, cy), max(x0, cx), max(y0, cy))
        elif kind == "draw":
            x0, y0 = st["start"]
            st["box"] = (min(x0, cx), min(y0, cy), max(x0, cx), max(y0, cy))
        elif far:
            poly = it.far[self.sel_far]
            if kind == "move":  # the whole region, kept inside the image
                orig = st["orig"]
                dx = min(max(rx - st["start"][0], -float(orig[:, 0].min())), it.w - float(orig[:, 0].max()))
                dy = min(max(ry - st["start"][1], -float(orig[:, 1].min())), it.h - float(orig[:, 1].max()))
                poly[:] = orig + np.array([dx, dy], dtype=np.float32)
            else:               # one corner follows the cursor
                poly[st["index"]] = (cx, cy)
        elif kind == "move":
            ox1, oy1, ox2, oy2 = (float(v) for v in st["orig"])
            bw, bh = ox2 - ox1, oy2 - oy1
            nx = min(max(ox1 + rx - st["start"][0], 0.0), max(0.0, it.w - bw))
            ny = min(max(oy1 + ry - st["start"][1], 0.0), max(0.0, it.h - bh))
            it.gt[self.sel] = (nx, ny, nx + bw, ny + bh)
        else:  # resize: the dragged edge(s) follow the cursor; crossing over flips the box
            x1, y1, x2, y2 = (float(v) for v in st["orig"])
            name = st["handle"]
            if "w" in name:
                x1 = cx
            if "e" in name:
                x2 = cx
            if "n" in name:
                y1 = cy
            if "s" in name:
                y2 = cy
            it.gt[self.sel] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self._render()

    def _gt_release(self, e):
        st, self._edit, self._drag = self._edit, None, None
        if st is None:
            return
        it = self._cur()
        kind, far = st["kind"], st.get("target") == "far"
        if kind == "draw_far":
            self._far_click(it, st, e)
            return
        if kind == "draw":
            box = st["box"]
            if box is None or box[2] - box[0] < MIN_BOX_PX or box[3] - box[1] < MIN_BOX_PX:
                self._render(note="Drag to draw a box" if box is None else "Box too small - cancelled")
                return  # stay in draw mode
            it.gt = np.vstack([it.gt, np.array([box], dtype=np.float32)])
            it.gt_cls = np.append(it.gt_cls, self.args.new_class)
            it.gt_src.append(None)
            self._select(it, len(it.gt) - 1)
            self.draw_mode = False
            self._set_cursor("arrow")
            self._commit(it, st["snap"], "Added a box")
            return
        if far:
            poly = it.far[self.sel_far]
            if poly_area(poly) < MIN_FAR_AREA:
                poly[:] = st["orig"]
                self._render(note="Region too small - reverted")
            elif np.abs(poly - st["orig"]).max() > 1e-4:
                self._commit(it, st["snap"], "Moved far region" if kind == "move" else "Moved a corner", far=True)
            else:
                self._render()  # a plain click: selected, nothing changed
            return
        j = self.sel
        x1, y1, x2, y2 = (float(v) for v in it.gt[j])
        if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
            it.gt[j] = st["orig"]
            self._render(note="Box too small - reverted")
        elif np.abs(it.gt[j] - st["orig"]).max() > 1e-4:
            it.gt_src[j] = None  # edited: this row is rewritten on save
            self._commit(it, st["snap"], "Moved box" if kind == "move" else "Resized box")
        else:
            self._render()  # a plain click: selected, nothing changed

    def _far_click(self, it, st, e):
        """Release while drawing a region: a drag made a rectangle; a click adds a corner (or closes the shape)."""
        if st["box"] is not None and not self._far_draft:
            poly = rect_to_poly(st["box"])
            if poly_area(poly) < MIN_FAR_AREA:
                self._render(note="Region too small - cancelled")
                return
            it.far.append(poly)
            self._select_far(it, len(it.far) - 1)
            self.draw_far_mode = False
            self._set_cursor("arrow")
            self._commit(it, st["snap"], "Added a far region", far=True)
            return
        px, py = self._to_img(e.x, e.y)
        if len(self._far_draft) >= 3:  # a click back on the first corner closes the shape
            fx, fy = self._to_scr_pt(self._far_draft[0])
            if np.hypot(e.x - fx, e.y - fy) <= HANDLE_HIT:
                self._finish_far_draft()
                return
        self._far_draft.append((min(max(px, 0.0), it.w), min(max(py, 0.0), it.h)))
        self._far_cursor = None
        n = len(self._far_draft)
        self._render(note=f"{n} corner(s): click to add more, Enter / double-click / click the first corner to close, "
                          f"Backspace removes the last, Esc cancels")

    def _to_scr_pt(self, pt):
        x0, y0, s = self._view
        return (pt[0] - x0) * s, (pt[1] - y0) * s

    def _finish_far_draft(self):
        it = self._cur()
        if it is None:
            return
        if len(self._far_draft) < 3:
            self._render(note="A region needs at least 3 corners")
            return
        poly = np.array(self._far_draft, dtype=np.float32)
        if poly_area(poly) < MIN_FAR_AREA:
            self._render(note="Region too small - add more corners or press Esc")
            return
        snap = it.snapshot()
        it.far.append(poly)
        self._far_draft, self._far_cursor = [], None
        self.draw_far_mode = False
        self._set_cursor("arrow")
        self._select_far(it, len(it.far) - 1)
        self._commit(it, snap, "Added a far region", far=True)

    def far_draft_finish(self):
        """Enter: close the region being drawn."""
        if self.draw_far_mode:
            self._finish_far_draft()

    def far_draft_back(self):
        """Backspace: drop the last corner of the region being drawn."""
        if self.draw_far_mode and self._far_draft:
            self._far_draft.pop()
            self._far_cursor = None
            self._render(note=f"{len(self._far_draft)} corner(s)")

    def _gt_dclick(self, e):
        it = self._cur()
        if it is None or self._view is None:
            return
        if self.draw_far_mode:
            if len(self._far_draft) >= 3:
                self._finish_far_draft()
            return
        best = None  # double-click on an outline: add a corner there
        for k, poly in enumerate(it.far):
            d, pt, i = self._nearest_edge(self._poly_scr(poly), e.x, e.y)
            if d <= PICK_TOL * 1.5 and (best is None or d < best[0]):
                best = (d, k, i, pt)
        if best is None:
            return
        _, k, i, pt = best
        x, y = self._to_img(pt[0], pt[1])
        snap = it.snapshot()
        it.far[k] = np.insert(it.far[k], i + 1, [x, y], axis=0).astype(np.float32)
        self._select_far(it, k)
        self._commit(it, snap, "Added a corner - drag it into place", far=True)

    def _gt_rclick(self, e):
        pf = self._selected_far()
        if not pf or self._view is None:
            return
        it, k = pf
        v = self._vertex_at(it.far[k], e.x, e.y)
        if v is None:
            return
        if len(it.far[k]) <= 3:
            self._render(note="A region needs at least 3 corners - press D to delete it")
            return
        snap = it.snapshot()
        it.far[k] = np.delete(it.far[k], v, axis=0)
        self._commit(it, snap, "Removed a corner", far=True)

    def _commit(self, it, snap, note="", far=False):
        """Record an edit: undo point, unsaved flag, and everything derived from the boxes."""
        it.undo.append(snap)
        del it.undo[:-UNDO_DEPTH]
        if far:
            it.far_dirty = True
        else:
            it.dirty = True
        self._match_item(it)
        self._refresh_size_choices()
        self._sync_list()
        self._render(note=note)

    def _best_pred(self, it, j, pmask):
        """Shown prediction that overlaps annotation j the most -> (index into it.pred, IoU), or (None, 0)."""
        cand = np.where(pmask)[0]
        if len(cand) == 0:
            return None, 0.0
        ious = iou_matrix(it.gt[j][None], it.pred[:it.k][cand])[0]
        k = int(np.argmax(ious))
        return (int(cand[k]), float(ious[k])) if ious[k] > 0 else (None, 0.0)

    def toggle_draw(self):
        it = self._cur()
        if it is None or (not self.draw_mode and not self._can_edit(it)):
            return
        self.draw_mode = not self.draw_mode
        self.draw_far_mode = False
        self._set_cursor("crosshair" if self.draw_mode else "arrow")
        self._render(note="Draw mode: drag a box on the right panel (Esc to cancel)" if self.draw_mode
                     else "Draw mode off")

    def toggle_draw_far(self):
        it = self._cur()
        if it is None:
            return
        self.draw_far_mode = not self.draw_far_mode
        self.draw_mode = False
        self._far_draft, self._far_cursor = [], None
        self._set_cursor("crosshair" if self.draw_far_mode else "arrow")
        self._render(note="Draw a far region: click its corners (Enter / double-click closes it), or drag a "
                          "rectangle. Esc cancels" if self.draw_far_mode else "Draw mode off")

    def _gen_regions(self, it, thr):
        """Generate the regions of one image with the ticked options and append them to it.far.
        Returns (number of regions added, small labels they remove, larger labels they remove)."""
        new = generate_regions(it.gt, it.w, it.h, thr, self.args.far_purity, self.up_var.get(), self.side_var.get(),
                               existing=it.far)
        it.far.extend(new)
        return (len(new),) + region_counts(it.gt, new, thr, taken=it.far[:-len(new)] if new else it.far)

    def _regenerate(self, it, thr):
        """Replace the regions of one image with freshly generated ones. Returns (regions, small, larger removed),
        or None when nothing was found (the image is left as it was)."""
        old = it.far
        it.far = []
        n_new, ns, nb = self._gen_regions(it, thr)
        if not n_new:
            it.far = old
            return None
        return n_new, ns, nb

    def far_region(self):
        """Generate the far regions: propose, expand (up / sideways, as ticked) and merge. On this image only, or on
        every image when 'apply to all images' is ticked - the same function either way, so the same result. The
        regions the image(s) had are replaced; Ctrl+Z brings them back."""
        thr = self._far_px()
        if not self.all_var.get():
            it = self._cur()
            if it is None:
                return
            snap = it.snapshot()
            res = self._regenerate(it, thr)
            if res is None:
                self._render(note=f"No far region found: it needs at least {FAR_MIN_SMALL} labels under {thr:g} px "
                                  f"where {self.args.far_purity:.0%} of the labels are that small"
                                  f"{' and none larger above them' if self.up_var.get() else ''} (raise 'small <' "
                                  f"to group more, or draw one with R)")
                return
            self._select_far(it, len(it.far) - 1)
            self._commit(it, snap, f"{res[0]} far region(s): {res[1]} labels under {thr:g}px (+{res[2]} larger). "
                                   f"Drag a corner or the shape, D deletes", far=True)
            return
        todo = [i for i in self.items if len(i.gt)]
        have = sum(1 for i in todo if len(i.far))
        up, side = self.up_var.get(), self.side_var.get()
        nl = chr(10)
        msg = (f"Generate far regions on all {len(todo)} images?{nl}{nl}"
               f"small < {thr:g} px, expand up {'on' if up else 'off'}, expand sideways {'on' if side else 'off'}.{nl}"
               f"{have} of them already have regions; those are replaced.{nl}"
               f"Nothing is written until you Save; Ctrl+Z undoes it per image.")
        if not messagebox.askyesno("Far regions on all images", msg):
            return
        n_img = n_reg = n_small = n_bad = 0
        self.root.config(cursor="watch")
        for k, it in enumerate(todo):
            snap = it.snapshot()
            res = self._regenerate(it, thr)
            if res is not None:
                it.undo.append(snap)
                del it.undo[:-UNDO_DEPTH]
                it.far_dirty = True
                n_img += 1
                n_reg, n_small, n_bad = n_reg + res[0], n_small + res[1], n_bad + res[2]
            if k % 25 == 0:
                self.status.config(text=f"Generating far regions... {k}/{len(todo)}")
                self.root.update_idletasks()
        self.root.config(cursor="")
        self._deselect()
        self._refilter(keep=True)
        self._render(note=f"{n_reg} far region(s) on {n_img} of {len(todo)} images, removing {n_small} labels under "
                          f"{thr:g}px and {n_bad} larger ones. Review with 'Only images with regions', then Save")

    def clear_far(self):
        """Remove the regions of this image, or of every image when 'apply to all images' is ticked."""
        if self.all_var.get():
            have = [i for i in self.items if len(i.far)]
            if not have:
                self._render(note="No far regions on any image")
                return
            if not messagebox.askyesno("Clear far regions", f"Remove the far regions of all {len(have)} images "
                                                            f"that have them? (Ctrl+Z undoes it per image.)"):
                return
            for it in have:
                it.undo.append(it.snapshot())
                del it.undo[:-UNDO_DEPTH]
                it.far, it.far_dirty = [], True
            self._deselect()
            self._refilter(keep=True)
            self._render(note=f"Cleared the far regions of {len(have)} images")
            return
        it = self._cur()
        if it is None or not len(it.far):
            self._render(note="No far regions on this image")
            return
        snap = it.snapshot()
        it.far = []
        self._deselect()
        self._commit(it, snap, "Cleared the far regions", far=True)

    def delete_selected(self):
        pf = self._selected_far()
        if pf:
            it, k = pf
            snap = it.snapshot()
            it.far = [p for i, p in enumerate(it.far) if i != k]
            self._deselect()
            self._commit(it, snap, "Deleted the far region", far=True)
            return
        picked = self._selected()
        if not picked:
            self._render(note="Nothing selected - click an annotation or a far region first (or press Z)")
            return
        it, j = picked
        if not self._can_edit(it):
            return
        snap = it.snapshot()
        keep = np.ones(len(it.gt), dtype=bool)
        keep[j] = False
        it.gt, it.gt_cls = it.gt[keep], it.gt_cls[keep]
        del it.gt_src[j]
        self._deselect()
        self._commit(it, snap, "Deleted the box")

    def apply_prediction(self):
        """Replace the selected annotation with the shown prediction that overlaps it the most."""
        picked = self._selected()
        if not picked:
            self._render(note="Nothing selected - click an annotation first (or press Z)")
            return
        it, j = picked
        if self.pred_data is None:
            self._render(note="No predictions loaded")
            return
        if not self._can_edit(it):
            return
        _, pmask = self._masks(it, display=True)
        k, iou = self._best_pred(it, j, pmask)
        if k is None:
            self._render(note="No shown prediction overlaps the selected box (try Show: All, or a lower Conf)")
            return
        snap = it.snapshot()
        box = np.clip(it.pred[k], 0, [it.w, it.h, it.w, it.h])
        it.gt[j] = box
        it.gt_src[j] = None
        self._commit(it, snap, f"Applied prediction (conf {it.pred_conf[k]:.2f}, IoU {iou:.2f})")

    def _add_boxes(self, it, boxes):
        """Append boxes (image px) as new annotations of the default class; returns how many were added."""
        boxes = np.clip(np.asarray(boxes, dtype=np.float32).reshape(-1, 4), 0, [it.w, it.h, it.w, it.h])
        boxes = boxes[(boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)]
        if len(boxes):
            it.gt = np.vstack([it.gt.reshape(-1, 4), boxes]).astype(np.float32)
            it.gt_cls = np.append(it.gt_cls, [self.args.new_class] * len(boxes))
            it.gt_src.extend([None] * len(boxes))
        return len(boxes)

    def apply_one(self):
        """Add the prediction selected on the left panel as a new annotation."""
        picked = self._selected_pred()
        if not picked:
            self._render(note="No prediction selected - click a box on the left panel first")
            return
        it, k = picked
        if not self._can_edit(it):
            return
        dup = ""
        if len(it.gt):
            iou = float(iou_matrix(it.pred[k][None], it.gt)[0].max())
            if iou >= self.args.iou:
                dup = f" - overlaps an annotation (IoU {iou:.2f}), Ctrl+Z if it is a duplicate"
        snap = it.snapshot()
        if not self._add_boxes(it, it.pred[k]):
            self._render(note="The prediction lies outside the image - nothing added")
            return
        self._select(it, len(it.gt) - 1)
        self._commit(it, snap, f"Added prediction (conf {it.pred_conf[k]:.2f}){dup}")

    def apply_all_predictions(self):
        """Add every shown prediction that matches no annotation yet, in one undo step.

        Matched predictions (green) are skipped: they would duplicate a box that is already there - use
        Apply prediction [P] to replace an annotation by its prediction. Predictions under a far region
        (grey) are skipped too: apply_far_regions removes those labels anyway."""
        it = self._cur()
        if it is None:
            return
        if self.pred_data is None:
            self._render(note="No predictions loaded")
            return
        if not self._can_edit(it):
            return
        _, p = self._masks(it, display=True)
        p = p.copy()
        n_matched = int((p & (it.p2g >= 0)).sum())
        p &= it.p2g < 0
        covered = covered_by_any(it.pred[:it.k], it.far)
        n_far = int((p & covered).sum())
        p &= ~covered
        skipped = ", ".join(t for t in (f"{n_matched} already matching an annotation" if n_matched else "",
                                        f"{n_far} under a far region" if n_far else "") if t)
        skipped = f" (skipped {skipped})" if skipped else ""
        if not p.any():
            self._render(note=f"No shown prediction to add{skipped}")
            return
        snap = it.snapshot()
        n = self._add_boxes(it, it.pred[:it.k][p])
        self._deselect()
        self._commit(it, snap, f"Added {n} prediction(s){skipped} - Ctrl+Z undoes all of them")

    def undo(self):
        it = self._cur()
        if it is None or not it.undo:
            self._render(note="Nothing to undo on this image")
            return
        labels, far = it.restore(it.undo.pop())
        it.dirty |= labels
        it.far_dirty |= far
        self._deselect()
        self._match_item(it)
        self._refresh_size_choices()
        self._sync_list()
        self._render(note="Undone")

    def cancel(self):
        """Esc: abort the drag in progress, else leave draw mode (dropping a half-drawn region), else deselect."""
        it = self._cur()
        if self._edit is not None and it is not None:
            it.restore(self._edit["snap"])
            self._edit = None
            self._match_item(it)
            self._render(note="Edit cancelled")
        elif self.draw_mode or self.draw_far_mode:
            self.draw_mode = self.draw_far_mode = False
            self._far_draft, self._far_cursor = [], None
            self._set_cursor("arrow")
            self._render(note="Draw mode off")
        else:
            self._deselect()
            self._render()

    def _any_dirty(self):
        return self._far_global_dirty or any(i.dirty or i.far_dirty for i in self.items)

    def _write_dirty(self):
        """Write every image with unsaved edits, and the far-region sidecar if it changed.
        Returns (number of label files written, error messages); self._far_saved says if the sidecar was written."""
        errors, saved = [], 0
        for it in [i for i in self.items if i.dirty]:
            try:
                save_label(it, self.split_dir, self.backup_dir)
                it.dirty = False
                saved += 1
            except OSError as ex:
                errors.append(f"{it.path.name}: {ex}")
        if saved:
            try:  # ultralytics trusts this cache by total file size alone, and a moved box keeps the size
                (self.split_dir / "labels.cache").unlink(missing_ok=True)
            except OSError as ex:
                errors.append(f"labels.cache could not be deleted ({ex}) - delete it before training")
        self._far_saved = False
        if self._far_global_dirty or any(i.far_dirty for i in self.items):
            try:
                save_far(self.far_path, self.items)
                for i in self.items:
                    i.far_dirty = False
                self._far_global_dirty = False
                self._far_saved = True
            except OSError as ex:
                errors.append(f"{self.far_path.name}: {ex}")
        return saved, errors

    def _saved_note(self, verb, saved):
        parts = ([f"{saved} label file(s)"] if saved else []) + (["far regions"] if self._far_saved else [])
        return f"{verb} " + " + ".join(parts) if parts else ""

    def save(self):
        """Write every image with unsaved edits to disk. Returns True if all of them were written."""
        if not self._any_dirty():
            self._render(note="Nothing to save")
            return True
        saved, errors = self._write_dirty()
        if errors:
            messagebox.showerror("Save", "\n".join(errors[:10]))
        self._render(note=(self._saved_note("Saved", saved) or "Saved nothing")
                     + (f", {len(errors)} problem(s)" if errors else ""))
        return not errors

    def _autosave(self):
        """Called before leaving an image. Returns (ok, number written); not ok means stay where you are."""
        self._far_saved = False
        if not self.autosave.get() or not self._any_dirty():
            return True, 0
        saved, errors = self._write_dirty()
        if errors:
            messagebox.showerror("Auto-save", "\n".join(errors[:10]) +
                                 "\n\nStaying on this image. Fix the problem, or untick Auto-save.")
            return False, saved
        return True, saved

    def delete_image(self):
        """Remove the image on screen and its label file from the dataset, after asking.

        Both files are MOVED to <input_dir>/_deleted/<split>/{images,labels}, not erased: ultralytics only
        reads the split's own folders so they are gone from training, but a mis-click is recoverable."""
        it = self._cur()
        if it is None:
            return
        label = self.split_dir / "labels" / f"{it.path.stem}.txt"
        warn = "\nUnsaved edits to this image will be lost." if it.dirty else ""
        if not messagebox.askyesno(
                "Delete image",
                f"Delete this image and its annotation file?\n\n{it.path.name}\n{len(it.gt)} annotation(s){warn}"
                f"\n\nBoth files are moved to\n{self.trash_dir}\n(recoverable; delete that folder to erase "
                f"them for good).", icon="warning", default="no"):
            return
        moved = []
        try:
            for src, sub in ((it.path, "images"), (label, "labels")):
                if not src.is_file():
                    continue  # a background image may have no label file
                dst = self.trash_dir / sub / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    dst.unlink()  # an older copy from a previous delete of the same name
                shutil.move(str(src), str(dst))
                moved.append((src, dst))
        except OSError as ex:
            for src, dst in reversed(moved):  # all or nothing
                shutil.move(str(dst), str(src))
            messagebox.showerror("Delete image", f"Nothing was deleted:\n{ex}")
            return
        (self.split_dir / "labels.cache").unlink(missing_ok=True)

        idx = self.visible[self.pos]
        self.items.pop(idx)
        if len(it.far):
            self._far_global_dirty = True  # the sidecar still lists this image
        self._img_cache.pop(it.path, None)
        self.visible = [v - (v > idx) for v in self.visible if v != idx]  # item indices above idx shift down
        self.pos = min(self.pos, max(0, len(self.visible) - 1))            # the next image takes its place
        self._refresh_size_choices()
        self._new_image()
        self._render(note=f"Deleted {it.path.name} (moved to _deleted/)")

    def _on_close(self):
        n = sum(1 for it in self.items if it.dirty or it.far_dirty)
        if (n or self._far_global_dirty) and self.autosave.get():
            _, errors = self._write_dirty()
            if errors and not messagebox.askyesno(
                    "Auto-save failed", "\n".join(errors[:10]) + "\n\nClose anyway and lose these edits?",
                    icon="warning", default="no"):
                return
        elif n or self._far_global_dirty:
            ans = messagebox.askyesnocancel("Unsaved changes", f"{n} image(s) have unsaved edits.\n"
                                                               f"Save them before closing?")
            if ans is None:
                return
            if ans and not self.save():
                return
        self._persist()
        if self._render_job:
            self.root.after_cancel(self._render_job)
        self.root.destroy()

    def _schedule_render(self):
        if self._render_job:
            self.root.after_cancel(self._render_job)
        self._render_job = self.root.after(40, self._render)

    def _image(self, path):
        img = self._img_cache.get(path)
        if img is None:
            img = Image.open(path).convert("RGB")
            if len(self._img_cache) >= 4:
                self._img_cache.pop(next(iter(self._img_cache)))
            self._img_cache[path] = img
        return img

    def _render(self, note=""):
        self._render_job = None
        self._persist_soon()
        for cv in (self.cv_pred, self.cv_gt):
            cv.delete("all")
        pw, ph = self._panel()
        if not self.visible:
            self._view = None
            for cv in (self.cv_pred, self.cv_gt):
                cv.create_text(pw / 2, ph / 2, text="No images match this filter", fill="white",
                               font=("Segoe UI", 14))
            self.hdr_pred.config(text="")
            self.hdr_gt.config(text="")
            self.status.config(text="0 images")
            return
        if pw < 10 or ph < 10:
            return
        it = self.items[self.visible[self.pos]]
        if self.fit_mode:
            self.scale = self.fit_scale = min(pw / it.w, ph / it.h)
            self.center = (it.w / 2, it.h / 2)
        else:
            self.fit_scale = min(pw / it.w, ph / it.h)
        s, (cx, cy) = self.scale, self.center
        x0, y0 = cx - pw / 2 / s, cy - ph / 2 / s
        self._view = (x0, y0, s)

        # only the visible part of the image is cropped and scaled, so deep zoom stays fast
        ix0, iy0 = max(0, math.floor(x0)), max(0, math.floor(y0))
        ix1, iy1 = min(it.w, math.ceil(x0 + pw / s)), min(it.h, math.ceil(y0 + ph / s))
        if ix1 > ix0 and iy1 > iy0:
            crop = self._image(it.path).crop((ix0, iy0, ix1, iy1))
            if len(it.far):
                crop = darken(crop, it.far, ix0, iy0)  # far regions: dimmed on both panels
            size = (max(1, round((ix1 - ix0) * s)), max(1, round((iy1 - iy0) * s)))
            crop = crop.resize(size, Image.NEAREST if s >= 2 else Image.BILINEAR)  # real pixels when zoomed
            self._photo = ImageTk.PhotoImage(crop)
            for cv in (self.cv_pred, self.cv_gt):
                cv.create_image((ix0 - x0) * s, (iy0 - y0) * s, anchor="nw", image=self._photo)

        g, p = self._masks(it, display=True)
        have = self.pred_data is not None
        picked, pf = self._selected(), self._selected_far()
        best, best_iou = (self._best_pred(it, picked[1], p) if picked and have else (None, 0.0))

        for k, poly in enumerate(it.far):  # far regions: an outline on both panels
            selk = bool(pf) and k == pf[1]
            sp = self._poly_scr(poly)
            top = sp[int(np.argmin(sp[:, 1]))]
            for cv in (self.cv_pred, self.cv_gt):
                cv.create_polygon(sp.ravel().tolist(), outline=C_SEL if selk else C_FAR, fill="",
                                  width=3 if selk else 2, **({} if selk else {"dash": (6, 3)}))
                cv.create_text(top[0] + 4, top[1] + 3, text="far region", anchor="nw",
                               fill=C_SEL if selk else C_FAR, font=("Segoe UI", 9, "bold"))
        cov_g = covered_by_any(it.gt, it.far)   # labels a region would remove are drawn grey
        cov_p = covered_by_any(it.pred[:it.k], it.far)

        tp = fn = 0
        for j in np.where(g)[0]:
            sel = bool(picked) and j == picked[1]
            color = C_SEL if sel else (C_COVERED if cov_g[j] else ((C_TP if it.hit[j] else C_FN) if have else C_GT))
            tp += bool(it.hit[j])
            fn += have and not it.hit[j]
            self._box(self.cv_gt, it.gt[j], color, f"{it.gt_size[j]:.0f}px", x0, y0, s, width=3 if sel else 2)
        if picked:
            self._draw_handles(self.cv_gt, self._to_scr(it.gt[picked[1]]))
        if pf:
            self._draw_corners(self.cv_gt, self._poly_scr(it.far[pf[1]]))
        st = self._edit
        if st and st["kind"] in ("draw", "draw_far") and st["box"]:
            a, b, c, d = self._to_scr(st["box"])
            self.cv_gt.create_rectangle(a, b, c, d, outline=C_FAR if st.get("target") == "far" else C_SEL, width=2,
                                        dash=(4, 3))
        if self.draw_far_mode and self._far_draft:  # the region being drawn: corners so far, rubber band to the cursor
            pts = [self._to_scr_pt(q) for q in self._far_draft]
            if len(pts) > 1:
                self.cv_gt.create_line(*[v for q in pts for v in q], fill=C_FAR, width=2)
            if self._far_cursor is not None:
                self.cv_gt.create_line(*pts[-1], *self._to_scr_pt(self._far_cursor), fill=C_FAR, dash=(4, 3))
            for qx, qy in pts:
                self.cv_gt.create_rectangle(qx - HANDLE, qy - HANDLE, qx + HANDLE, qy + HANDLE, fill="white",
                                            outline=C_FAR)
            if len(pts) >= 3:  # click here to close
                self.cv_gt.create_oval(pts[0][0] - HANDLE_HIT, pts[0][1] - HANDLE_HIT, pts[0][0] + HANDLE_HIT,
                                       pts[0][1] + HANDLE_HIT, outline=C_FAR, width=2)

        n_tp = n_fp = 0
        for j in np.where(p)[0]:
            ok = it.p2g[j] >= 0
            n_tp, n_fp = n_tp + ok, n_fp + (not ok)
            self._box(self.cv_pred, it.pred[j], C_COVERED if cov_p[j] else (C_TP if ok else C_FP),
                      f"{it.pred_conf[j]:.2f}", x0, y0, s)
        if best is not None:  # what Apply prediction would copy
            a, b, c, d = self._to_scr(it.pred[best])
            self.cv_pred.create_rectangle(a - 3, b - 3, c + 3, d + 3, outline=C_SEL, width=2, dash=(4, 3))
        pp = self._selected_pred()
        if pp:  # what Apply one would add
            self._box(self.cv_pred, it.pred[pp[1]], C_SEL, f"{it.pred_conf[pp[1]]:.2f}", x0, y0, s, width=3)

        unsaved = "   * UNSAVED" if (it.dirty or it.far_dirty) else ""
        if have:
            nms = self.pred_data.get("nms_iou", 0.7)  # caches written before nms_iou was recorded used 0.7
            nms_txt = "NMS-free" if nms is None else f"nms {nms:g}"  # RF-DETR: one query per object, no NMS
            self.hdr_pred.config(text=f"Predictions  conf>={self._conf:.2f}  {nms_txt}   shown {n_tp + n_fp}  "
                                      f"(TP {n_tp}, FP {n_fp})   [{self.pred_path.name}]")
            self.hdr_gt.config(text=f"Annotations   shown {int(g.sum())}  (hit {tp}, missed {fn}){unsaved}")
        else:
            self.hdr_pred.config(text="Predictions  none loaded - Run predictions / Load predictions...")
            self.hdr_gt.config(text=f"Annotations   shown {int(g.sum())}{unsaved}")

        if pf:
            poly = it.far[pf[1]]
            cov = covered_by(it.gt, poly)
            small = int((cov & (box_sizes(it.gt) < self._far_px())).sum())
            n_p = int(covered_by(it.pred[:it.k], poly).sum())
            info = (f"far region, {len(poly)} corners, {poly_area(poly) / (it.w * it.h):.0%} of the image: removes "
                    f"{int(cov.sum())} annotation(s), {small} under {self._far_px():g}px, {n_p} prediction(s)  |  "
                    f"drag a corner or the shape; double-click an edge adds a corner, right-click a corner removes it")
            self.lbl_sel.config(text=info)
        elif picked:
            j = picked[1]
            bx = it.gt[j]
            info = f"selected: {it.gt_size[j]:.1f}px  {bx[2] - bx[0]:.0f}x{bx[3] - bx[1]:.0f}"
            if have:
                info += "  hit" if it.hit[j] else "  missed"
                if best is not None:
                    info += f"   |   Apply -> conf {it.pred_conf[best]:.2f}, IoU {best_iou:.2f} (dashed)"
            a, b, c, d = self._to_scr(bx)
            if min(c - a, d - b) < HANDLE_MIN_SIDE:
                info += "   |   zoom in to resize (Z)"
            self.lbl_sel.config(text=info)
        elif pp:
            k = pp[1]
            bx = it.pred[k]
            state = "matches an annotation" if it.p2g[k] >= 0 else "matches no annotation"
            self.lbl_sel.config(text=f"selected prediction: conf {it.pred_conf[k]:.2f}  "
                                     f"{bx[2] - bx[0]:.0f}x{bx[3] - bx[1]:.0f}, {state}   |   O adds it as an annotation")
        else:
            self.lbl_sel.config(text="click an annotation to select it; D deletes, P applies the nearest prediction; "
                                     "click a prediction (left) and O adds it, A adds all unmatched")
        n_dirty = sum(1 for x in self.items if x.dirty or x.far_dirty) or int(self._far_global_dirty)
        self.lbl_far.config(text=self._far_summary(it))
        self.btn_far_draw.config(relief=tk.SUNKEN if self.draw_far_mode else tk.RAISED)
        self.btn_save.config(text=f"Save ({n_dirty}) [Ctrl+S]", bg="#ffd27f" if n_dirty else self._btn_bg)
        self.btn_draw.config(relief=tk.SUNKEN if self.draw_mode else tk.RAISED)

        bucket = "all sizes" if self._bucket() is None else f"{self.labels[self._bucket()]} px"
        why = ""
        if self._bucket() is not None or (have and self.mode_var.get() != "All"):
            ga, pa = self._masks(it)
            why = f"  [{int(ga.sum())} annotation(s), {int(pa.sum())} prediction(s) match" \
                  + ("; none left - dropped when you move on]" if not (ga.any() or pa.any()) else "]")
        self.status.config(text=f"  {self.pos + 1} / {len(self.visible)}   |   {it.path.name}   {it.w}x{it.h}   |   "
                                f"{bucket}, {self.mode_var.get()}{why}   |   zoom {s / self.fit_scale:.1f}x"
                                + (f"   |   {note}" if note else ""))

    def _far_summary(self, it):
        if not len(it.far):
            return "no far region on this image - Far Region (M) proposes one, Draw region (R) draws one"
        cov = covered_by_any(it.gt, it.far)
        n, thr = int(cov.sum()), self._far_px()
        small = int((cov & (box_sizes(it.gt) < thr)).sum())
        share = f", {small / n:.0%} under {thr:g}px" if n else ""
        return f"{len(it.far)} far region(s) remove {n} of {len(it.gt)} annotations{share}"

    def _box(self, cv, box, color, text, x0, y0, s, width=2):
        a, b, c, d = (box[0] - x0) * s, (box[1] - y0) * s, (box[2] - x0) * s, (box[3] - y0) * s
        cv.create_rectangle(a, b, c, d, outline=color, width=width)
        if max(c - a, d - b) < SMALL_ON_SCREEN:
            mx, my = (a + c) / 2, (b + d) / 2
            cv.create_oval(mx - 14, my - 14, mx + 14, my + 14, outline=color, dash=(3, 2))
        if self.show_labels.get():
            cv.create_text(a, b - 2, text=text, anchor="sw", fill=color, font=("Segoe UI", 8))

    def _draw_corners(self, cv, sp):
        """Square handles on every corner of the selected far region (screen coordinates)."""
        for qx, qy in sp:
            cv.create_rectangle(qx - HANDLE, qy - HANDLE, qx + HANDLE, qy + HANDLE, fill="white", outline=C_SEL)

    def _draw_handles(self, cv, rect):
        if min(rect[2] - rect[0], rect[3] - rect[1]) < HANDLE_MIN_SIDE:
            return
        for hx, hy in self._handle_points(rect).values():
            cv.create_rectangle(hx - HANDLE, hy - HANDLE, hx + HANDLE, hy + HANDLE, fill="white", outline=C_SEL)

# --------------------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", default=r"path/to/data", help="YOLO dataset root (contains <split>/images, <split>/labels)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--humans_model_path", default="models/humans.pt", help="detector weights for Run predictions")
    ap.add_argument("--predictions_json", default=None,
                    help="prediction cache (default: <input_dir>/predictions_<split>_<model>.json)")
    ap.add_argument("--detector", choices=["yolo", "rfdetr"], default=None,
                    help="detector for Run predictions (default: the one used last time, else yolo)")
    ap.add_argument("--rfdetr_weights", default="models/rfdetr_xlarge_1220.pth",
                    help="RF-DETR checkpoint for Run predictions with Detector RF-DETR")
    ap.add_argument("--rfdetr_python", default=str(Path("rfdetr_env") / ("Scripts/python.exe" if os.name == "nt"
                                                                        else "bin/python")),
                    help="python of the separate RF-DETR environment that runs rfdetr_predict.py")
    ap.add_argument("--imgsz", type=int, default=1600, help="YOLO input size (models/humans.pt is the 1600 model)")
    ap.add_argument("--conf", type=float, default=0.35, help="initial display threshold")
    ap.add_argument("--save_conf", type=float, default=0.05, help="lowest confidence kept in the cache")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU for a prediction to match an annotation")
    ap.add_argument("--nms_iou", type=float, default=0.5,
                    help="NMS IoU for Run predictions; 0.5 = what object_tracker.py uses in the app "
                         "(ultralytics' own default 0.7 leaves visibly more duplicate boxes)")
    ap.add_argument("--new_class", type=int, default=0, help="class id given to boxes drawn in the editor")
    ap.add_argument("--far_px", type=float, default=FAR_PX,
                    help="Far Region: labels with sqrt(w*h) below this are small (the toolbar box overrides it)")
    ap.add_argument("--far_purity", type=float, default=FAR_PURITY,
                    help="Far Region: share of the labels a region removes that must be small")
    ap.add_argument("--size_edges", default=DEFAULT_EDGES, help="bucket edges, sqrt(w*h) px")
    ap.add_argument("--device", default=None)
    ap.add_argument("--predict", action="store_true", help="only run predictions and write the JSON (no UI)")
    ap.add_argument("--reset_settings", action="store_true",
                    help="ignore the saved UI settings (they are overwritten as you work)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    split_dir = Path(args.input_dir) / args.split
    if not (split_dir / "images").is_dir():
        sys.exit(f"no {split_dir / 'images'}")
    edges = sorted(float(v) for v in args.size_edges.split(",") if v.strip())

    if args.predict:
        rfdetr = args.detector == "rfdetr"
        weights = args.rfdetr_weights if rfdetr else args.humans_model_path
        if not weights:
            sys.exit("--predict needs --rfdetr_weights" if rfdetr else "--predict needs --humans_model_path")
        images = list_images(split_dir / "images")
        out = (Path(args.predictions_json) if args.predictions_json
               else default_pred_path(args.input_dir, args.split, weights))
        progress = lambda n, t: (n % 100 == 0 or n == t) and print(f"  {n}/{t}", flush=True)  # noqa: E731
        if rfdetr:
            data = run_rfdetr_predictions(args.rfdetr_python, weights, images, args.save_conf, out, progress=progress)
        else:
            data = run_predictions(weights, images, args.imgsz, args.save_conf, args.device,
                                   nms_iou=args.nms_iou, progress=progress)
            out.write_text(json.dumps(data))
        print(f"wrote {out}  ({sum(len(v) for v in data['images'].values())} boxes, {len(images)} images)")
        return

    if tk is None:
        sys.exit("tkinter is not available; use --predict for headless prediction")
    try:  # crisp rendering on high-DPI Windows displays
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    print(f"loading {split_dir} ...")
    items = load_items(split_dir)
    root = tk.Tk()
    root.title(f"Analyse predictions - {split_dir}")
    saved = {} if args.reset_settings else resolve_settings(load_settings(SETTINGS_FILE),
                                                            argv if argv is not None else sys.argv[1:])
    restore_geometry(root, saved.get("geometry"))
    App(root, args, items, edges, settings=saved, settings_path=SETTINGS_FILE)
    root.mainloop()


if __name__ == "__main__":
    main()
