"""Pre-annotate humans in still images with SAM3, writing YOLO-format labels.

SAM3 is prompted with free text, so no class-specific detector is needed. The default
query is the bare "person": on a wide pool shot with ~60 swimmers it returned 112 boxes
at conf 0.25, where the more descriptive "humans in the water" returned 6 and needed the
threshold dropped to 0.10 to find them. Longer prompts score lower, not better.
The output is a starting point for manual correction, not a finished label set - review
it before folding it into a training split.

The weights are not auto-downloadable. models/sam3.pt has to be requested from Hugging
Face and placed there by hand.

    python src/dataset/auto_annotate_humans.py \
        --input_images_dir  path/to/data \
        --annotations_dir   path/to/data

Images with no detection still get an empty .txt, which is how YOLO marks a background
image. Pass --preview_dir to also write boxed JPEGs for eyeballing the result.

Tiny people: SAM3 resizes everything to 1008x1008, so on a 4K frame a 10 px swimmer is ~3 px.
--sahi cuts the frame into overlapping --slice_size tiles (SAHI), runs SAM3 on every tile at
native resolution, adds one full-frame pass for the large people, maps the boxes back and
merges them. It is slower (about one SAM3 call per tile) and needs `pip install sahi`
(--no-deps is enough: it only needs numpy, opencv, pillow, shapely, tqdm, which this script
already has or that install separately).
"""

import argparse
import glob
import os
import sys
import time

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# SAM3's ViT is built for 1008x1008 with a patch size of 14; other sizes silently
# degrade the features rather than erroring, so this is not a free knob.
SAM3_IMGSZ = 1008


def find_images(root, recursive):
    pattern = os.path.join(root, "**", "*") if recursive else os.path.join(root, "*")
    return sorted(p for p in glob.glob(pattern, recursive=recursive)
                  if p.lower().endswith(IMAGE_EXTS))


def suppress(boxes, iou_thr, contain_thr):
    """Greedy single-class NMS with an extra containment rule.

    SAM3's semantic head emits up to 200 queries and nothing merges them, so one
    swimmer often comes back as several nested boxes - a tight crop plus a loose one
    around the same body. Plain IoU misses that pair, because a small box inside a
    large one scores low. The containment test compares the overlap against the
    SMALLER of the two areas instead, which catches nesting in either direction.
    """
    order = sorted(range(len(boxes)), key=lambda i: float(boxes[i][4]), reverse=True)
    kept = []
    for i in order:
        ax1, ay1, ax2, ay2 = (float(v) for v in boxes[i][:4])
        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        if a_area <= 0:
            continue
        for j in kept:
            bx1, by1, bx2, by2 = (float(v) for v in boxes[j][:4])
            iw = min(ax2, bx2) - max(ax1, bx1)
            ih = min(ay2, by2) - max(ay1, by1)
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            b_area = (bx2 - bx1) * (by2 - by1)
            if inter / (a_area + b_area - inter) > iou_thr:
                break
            if contain_thr and inter / min(a_area, b_area) > contain_thr:
                break
        else:
            kept.append(i)
    return [boxes[i] for i in sorted(kept)], len(boxes) - len(kept)


def to_yolo_lines(boxes, width, height, cls, min_rel_area, max_rel_area):
    """Convert xyxy pixel boxes to normalised YOLO rows, dropping implausible ones."""
    lines, dropped = [], 0
    for x1, y1, x2, y2, conf, _ in boxes:
        x1, x2 = sorted((max(0.0, float(x1)), min(float(width), float(x2))))
        y1, y2 = sorted((max(0.0, float(y1)), min(float(height), float(y2))))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            dropped += 1
            continue
        rel_area = (bw * bh) / (width * height)
        if not (min_rel_area <= rel_area <= max_rel_area):
            dropped += 1
            continue
        cx, cy = (x1 + bw / 2) / width, (y1 + bh / 2) / height
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}")
    return lines, dropped


def draw_preview(image, boxes, path):
    import cv2

    canvas = image.copy()
    for x1, y1, x2, y2, conf, _ in boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(canvas, p1, p2, (0, 0, 255), 2)
        cv2.putText(canvas, f"{float(conf):.2f}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(path, canvas)


def make_sahi_model(model_path, text_query, conf, device, imgsz):
    """SAHI DetectionModel that runs SAM3's text-prompted predictor on one tile at a time."""
    import numpy as np
    import torch
    from sahi.models.base import DetectionModel

    class Sam3Sahi(DetectionModel):
        def load_model(self):
            from ultralytics.models.sam.predict import SAM3SemanticPredictor

            self.predictor = SAM3SemanticPredictor(overrides=dict(
                task="segment", mode="predict", model=self.model_path, imgsz=imgsz,
                conf=self.confidence_threshold, device=self.device, save=False, verbose=False))
            self.model = self.predictor
            self.category_mapping = {"0": "person"}

        def set_model(self, model, **kwargs):
            self.model = model

        @property
        def category_names(self):
            return ["person"]

        @property
        def num_categories(self):
            return 1

        @property
        def has_mask(self):
            return False

        @property
        def is_obb(self):
            return False

        def perform_inference(self, image):
            # SAHI hands over RGB; ultralytics expects BGR arrays
            self.predictor.set_prompts({"text": [text_query]})  # inference() pops it every call
            result = self.predictor(source=np.ascontiguousarray(image[:, :, ::-1]))[0]
            boxes = result.boxes.data if result.boxes is not None else torch.empty((0, 6))
            self._original_predictions = [boxes.cpu()]
            self._original_shapes = [image.shape]

        def _create_object_prediction_list_from_original_predictions(self, shift_amount_list=[[0, 0]],
                                                                      full_shape_list=None):
            from sahi.prediction import ObjectPrediction
            from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

            shifts = fix_shift_amount_list(shift_amount_list)  # a flat [x, y] for one image becomes [[x, y]]
            fulls = fix_full_shape_list(full_shape_list)
            out = []
            for k, boxes in enumerate(self._original_predictions):
                shift = [int(v) for v in shifts[k]]
                full = None if fulls is None else [int(v) for v in fulls[k]]
                preds = []
                for x1, y1, x2, y2, score, _ in boxes.numpy().tolist():
                    box = [max(0.0, x1), max(0.0, y1), max(0.0, x2), max(0.0, y2)]
                    if full is not None:
                        box = [min(full[1], box[0]), min(full[0], box[1]), min(full[1], box[2]), min(full[0], box[3])]
                    if not (box[0] < box[2] and box[1] < box[3]):
                        continue
                    preds.append(ObjectPrediction(bbox=box, category_id=0, category_name="person", score=score,
                                                  shift_amount=shift, full_shape=full))
                out.append(preds)
            self._object_prediction_list_per_image = out

    return Sam3Sahi(model_path=model_path, confidence_threshold=conf, device=device, load_at_init=True)


def label_done(label_path):
    """An existing label file counts as done - unless it holds nothing but NUL bytes, which is what a
    write cut short by a sudden shutdown leaves behind. Such a file is redone."""
    if not os.path.exists(label_path):
        return False
    with open(label_path, "rb") as f:
        data = f.read()
    return not (data and not data.strip(b"\x00"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_images_dir", default=r"path/to/data")
    ap.add_argument("--annotations_dir", default=r"path/to/data")
    ap.add_argument("--model", default="models/sam3.pt",
                    help="SAM3 weights; must be downloaded by hand")
    ap.add_argument("--text_query", default="person",
                    help="what SAM3 should segment")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.7,
                    help="NMS IoU threshold; 0 disables merging entirely")
    ap.add_argument("--contain", type=float, default=0.8,
                    help="drop a box whose overlap with a higher-scoring one "
                         "exceeds this fraction of the smaller box; 0 disables")
    ap.add_argument("--cls", type=int, default=0, help="YOLO class index to write")
    ap.add_argument("--device", default="0")
    ap.add_argument("--min_rel_area", type=float, default=0.0,
                    help="drop boxes smaller than this fraction of the image")
    ap.add_argument("--max_rel_area", type=float, default=0.5,
                    help="drop boxes larger than this fraction of the image - a text "
                         "query can latch onto the whole pool instead of a swimmer")
    ap.add_argument("--preview_dir", default=r"path/to/data",
                    help="also write boxed images here for review")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--sahi", action="store_true",
                    help="tile each image with SAHI so tiny people are seen at native resolution (slower)")
    ap.add_argument("--slice_size", type=int, default=SAM3_IMGSZ,
                    help="--sahi tile size in px; keep near 1008 so a tile is not resized")
    ap.add_argument("--overlap", type=float, default=0.2, help="--sahi overlap between tiles (0-1)")
    ap.add_argument("--no_full_pass", action="store_true",
                    help="--sahi: skip the extra full-frame pass (large people may then be cut at tile edges)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N images (0 = all)")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo images that already have a label file")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}\n"
                 f"SAM3 cannot be auto-downloaded - request it on Hugging Face and "
                 f"place the .pt there.")

    images = find_images(args.input_images_dir, args.recursive)
    if not images:
        sys.exit(f"no images under {args.input_images_dir}")
    if args.limit:
        images = images[:args.limit]

    os.makedirs(args.annotations_dir, exist_ok=True)
    if args.preview_dir:
        os.makedirs(args.preview_dir, exist_ok=True)

    if args.sahi:
        import cv2
        from sahi.predict import get_sliced_prediction

        sahi_model = make_sahi_model(args.model, args.text_query, args.conf,
                                     f"cuda:{args.device}" if str(args.device).isdigit() else args.device,
                                     SAM3_IMGSZ)
        predictor = None
    else:
        from ultralytics.models.sam.predict import SAM3SemanticPredictor

        predictor = SAM3SemanticPredictor(overrides=dict(
            task="segment", mode="predict", model=args.model, imgsz=SAM3_IMGSZ,
            conf=args.conf, device=args.device, save=False, verbose=False))

    print(f"{len(images)} images  |  query={args.text_query!r}  conf={args.conf}")
    print(f"labels -> {args.annotations_dir}")

    started = time.perf_counter()
    n_done = n_skipped = n_empty = n_boxes = n_dropped = n_merged = 0
    for i, image_path in enumerate(images, 1):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(args.annotations_dir, stem + ".txt")
        if label_done(label_path) and not args.overwrite:
            n_skipped += 1
            continue

        if args.sahi:
            orig_img = cv2.imread(image_path)
            height, width = orig_img.shape[:2]
            sliced = get_sliced_prediction(
                orig_img[:, :, ::-1], sahi_model, slice_height=args.slice_size, slice_width=args.slice_size,
                overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
                perform_standard_pred=not args.no_full_pass, postprocess_type="GREEDYNMM",
                postprocess_match_metric="IOS", postprocess_match_threshold=args.contain,
                auto_slice_resolution=False, verbose=0)
            boxes = [[*p.bbox.to_xyxy(), p.score.value, 0] for p in sliced.object_prediction_list]
        else:
            # Re-set on every image: inference() pops the prompt off self.prompts.
            predictor.set_prompts({"text": [args.text_query]})
            result = predictor(source=image_path)[0]
            orig_img = result.orig_img
            height, width = result.orig_shape
            boxes = result.boxes.data.tolist() if result.boxes is not None else []
        if args.iou or args.contain:
            boxes, merged = suppress(boxes, args.iou, args.contain)
            n_merged += merged
        lines, dropped = to_yolo_lines(boxes, width, height, args.cls,
                                       args.min_rel_area, args.max_rel_area)
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
            # on disk before the next image: a laptop that shuts down from heat otherwise leaves
            # files whose size was reserved but whose bytes are all NUL
            f.flush()
            os.fsync(f.fileno())

        n_done += 1
        n_boxes += len(lines)
        n_dropped += dropped
        n_empty += not lines
        if args.preview_dir:
            kept = [b for b in boxes
                    if to_yolo_lines([b], width, height, args.cls,
                                     args.min_rel_area, args.max_rel_area)[0]]
            draw_preview(orig_img, kept,
                         os.path.join(args.preview_dir, stem + ".jpg"))

        if i % 10 == 0 or i == len(images):
            rate = (time.perf_counter() - started) / max(1, n_done)
            print(f"  {i}/{len(images)}  boxes={n_boxes}  empty={n_empty}  "
                  f"{rate:.2f}s/img", flush=True)

    print(f"\nannotated {n_done} images in {time.perf_counter() - started:.0f}s")
    print(f"  boxes written : {n_boxes} ({n_boxes / max(1, n_done):.2f} per image)")
    print(f"  no detection  : {n_empty}")
    print(f"  merged by NMS : {n_merged} (iou>{args.iou}, containment>{args.contain})")
    print(f"  filtered out  : {n_dropped} (outside "
          f"{args.min_rel_area}-{args.max_rel_area} of frame area)")
    if n_skipped:
        print(f"  already done  : {n_skipped} (use --overwrite to redo)")


if __name__ == "__main__":
    main()
