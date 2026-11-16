"""Convert a YOLO detection dataset (data.yaml + images/ + labels/) to the COCO layout RF-DETR trains on.

    <out>/train/_annotations.coco.json   + the train images
    <out>/valid/_annotations.coco.json   + the val images     (RF-DETR calls the split "valid")
    <out>/test/_annotations.coco.json    + the test images

Images are HARDLINKED by default (--link hard): no extra disk space, and the YOLO dataset stays the
single source of truth. Use --link copy across filesystems, --link sym for symlinks.

Labels are read the way ultralytics reads them, so both detectors train and are scored on the same
boxes: duplicate rows are dropped (36 duplicates in one test image of dataset_grouped), boxes are
clipped to the image, and zero-area boxes are skipped. Images without a label file (or with an
empty one) are kept as background images with no annotations.

One COCO category per YOLO class, ids = YOLO class ids, each its own supercategory (the COCO
convention for a top-level class, which RF-DETR keeps as a real class).

    python yolo_to_coco.py --data /home/ubuntu/projects/data/humans/dataset_grouped/data.yaml \
        --out /home/ubuntu/projects/data/humans/dataset_grouped_coco
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from size_metrics_core import label_path, list_split_images, load_data_yaml  # noqa: E402

SPLITS = {"train": "train", "val": "valid", "test": "test"}


def place(src, dst, mode):
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return
        dst.unlink()
    if mode == "hard":
        os.link(src, dst)
    elif mode == "sym":
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def read_yolo_boxes(path, w, h):
    """YOLO rows -> list of (cls, x, y, bw, bh) in pixels, deduplicated and clipped like ultralytics."""
    p = Path(path)
    if not p.is_file():
        return []
    rows = list(dict.fromkeys(r.strip() for r in p.read_text().splitlines() if r.strip()))
    out = []
    for r in rows:
        parts = r.split()
        if len(parts) < 5:
            continue
        c, xc, yc, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
        x1, y1 = max(0.0, (xc - bw / 2) * w), max(0.0, (yc - bh / 2) * h)
        x2, y2 = min(float(w), (xc + bw / 2) * w), min(float(h), (yc + bh / 2) * h)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        out.append((c, x1, y1, x2 - x1, y2 - y1))
    return out


def convert_split(files, out_dir, names, mode, limit=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    images, anns, seen = [], [], set()
    files = sorted(files)[:limit] if limit else sorted(files)
    for i, f in enumerate(files):
        src = Path(f)
        name = src.name
        if name in seen:  # flat COCO folder: two images with one name would overwrite each other
            raise SystemExit(f"duplicate image name {name} in {out_dir.name}; rename it in the YOLO dataset first")
        seen.add(name)
        with Image.open(src) as im:  # header only
            w, h = im.size
        place(src, out_dir / name, mode)
        images.append(dict(id=i, file_name=name, width=w, height=h))
        for c, x, y, bw, bh in read_yolo_boxes(label_path(f), w, h):
            anns.append(dict(id=len(anns), image_id=i, category_id=c, bbox=[round(x, 2), round(y, 2),
                             round(bw, 2), round(bh, 2)], area=round(bw * bh, 2), iscrowd=0))
    cats = [dict(id=k, name=v, supercategory=v) for k, v in sorted(names.items())]
    (out_dir / "_annotations.coco.json").write_text(json.dumps(dict(images=images, annotations=anns, categories=cats)))
    return len(images), len(anns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="YOLO data.yaml")
    ap.add_argument("--out", required=True, help="output folder (RF-DETR dataset_dir)")
    ap.add_argument("--link", choices=["hard", "sym", "copy"], default="hard")
    ap.add_argument("--limit", type=int, default=0, help="first N images per split only (smoke tests)")
    args = ap.parse_args()

    data = load_data_yaml(args.data)
    out = Path(args.out)
    for yolo_split, coco_split in SPLITS.items():
        if not data.get(yolo_split):
            print(f"{yolo_split}: not in {args.data}, skipped")
            continue
        n_img, n_ann = convert_split(list_split_images(data[yolo_split]), out / coco_split, data["names"],
                                     args.link, args.limit)
        print(f"{yolo_split:>5} -> {coco_split:<5} {n_img:6d} images  {n_ann:7d} boxes")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
