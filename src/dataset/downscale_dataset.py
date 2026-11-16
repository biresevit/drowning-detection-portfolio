"""Shrink the images of a YOLO dataset whose long side is larger than --max_side, in place.

Why: the model sees at most --imgsz (1280 / 1920) pixels on the long side, so the extra pixels of 4K frames
are thrown away at load time - after costing disk, upload time and (slow PNG decoding of) data-loader CPU.

Labels are normalised (0-1) and the aspect ratio is kept, so the label files are NOT touched: a box stays
on the same person. Only images whose long side is > --max_side are rewritten, with an area filter
(cv2.INTER_AREA), under the same file name; images that are already small are left alone, so running it
twice changes nothing more. Each image is written as a new file and swapped in (a hardlink elsewhere keeps
the full-size original). <dataset>/<split>/labels.cache is not affected by this (it stores normalised boxes),
but is removed anyway when any image changed, to be safe.

Caveat: anything measured in native pixels changes for the shrunk images (box sizes sqrt(w*h), the size
buckets of the evaluation, 'small < N px' rules): a 10 px person in a 3840 px frame is 5 px in a 1920 px one.

Without --apply nothing is written (dry run).

    python src/dataset/downscale_dataset.py --dataset <dataset root with train/ val/ test/>
    python src/dataset/downscale_dataset.py --dataset <...> --max_side 1920 --apply
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def target_size(w, h, max_side):
    """(new w, new h) with the long side at max_side and the aspect ratio kept, or None if no shrink is needed."""
    if max(w, h) <= max_side:
        return None
    s = max_side / max(w, h)
    return max(1, round(w * s)), max(1, round(h * s))


def shrink_file(path, max_side, jpeg_quality, apply):
    """-> (old bytes, new bytes, new size) of a shrunk image, or None when it is left as it is."""
    with Image.open(path) as im:
        w, h = im.size
    new = target_size(w, h, max_side)
    if new is None:
        return None
    old_bytes = path.stat().st_size
    if not apply:
        return old_bytes, old_bytes, new
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read {path.name}")
    small = cv2.resize(img, new, interpolation=cv2.INTER_AREA)
    suffix = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if suffix in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(suffix, small, params)
    if not ok:
        raise RuntimeError(f"cannot encode {path.name}")
    tmp = path.with_name(path.name + ".tmp_small")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)
    return old_bytes, len(buf), new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset root; its train/ val/ test/ each hold images/ and labels/")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--max_side", type=int, default=1920, help="largest allowed long side, px")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--apply", action="store_true", help="write the shrunk images (default: dry run)")
    args = ap.parse_args()

    root = Path(args.dataset)
    files = []
    for split in args.splits.split(","):
        d = root / split / "images"
        if d.is_dir():
            files += [(split, p) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if not files:
        sys.exit(f"no images under {root}/<split>/images")
    print(f"{len(files)} images, long side limit {args.max_side}px, {args.workers} workers")

    t0 = time.perf_counter()
    done = shrunk = 0
    before = after = 0
    changed, errors = set(), []

    def work(item):
        split, p = item
        try:
            return split, p, shrink_file(p, args.max_side, args.jpeg_quality, args.apply), None
        except Exception as ex:  # noqa: BLE001 - report and carry on with the rest
            return split, p, None, ex

    with ThreadPoolExecutor(args.workers) as ex:
        for split, p, res, err in ex.map(work, files):
            done += 1
            if err:
                errors.append(f"{p.name}: {err}")
            elif res:
                shrunk += 1
                before += res[0]
                after += res[1]
                changed.add(split)
            if done % 500 == 0:
                print(f"  {done}/{len(files)}  shrunk so far {shrunk}", flush=True)

    print(f"\n{'applied' if args.apply else 'DRY RUN - nothing written'} in {time.perf_counter() - t0:.0f}s")
    print(f"  images to shrink: {shrunk} of {len(files)} (long side > {args.max_side}px)")
    if args.apply and shrunk:
        print(f"  size of those: {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB (saved {(before - after) / 1e9:.2f} GB)")
        for split in changed:
            (root / split / "labels.cache").unlink(missing_ok=True)
    if errors:
        print(f"  ERRORS: {len(errors)} (first: {errors[0]})")
        sys.exit(1)
    if not args.apply:
        print("  run again with --apply to shrink them")


if __name__ == "__main__":
    main()
