"""Apply the far regions drawn in analyse_predictions_ui to a dataset split, in place.

For every image that has far regions in the editor's sidecar file (far_regions_<split>.json):
  * the pixels under the regions are painted black in the image file, and
  * the labels of that image that are mostly under a region are removed.

A label is removed when MORE than --min_cover (default 0.8) of its box area lies in the painted
pixels (the union of all regions of the image, measured on the same pixel mask that is painted, so
overlapping regions are not counted twice). Labels only partly under a region (up to 80%) stay: their
person is still mostly visible. This is stricter than the editor's grey "would be removed" markers,
which also count a label whose centre is inside, or half of it - expect a few that are grey in the
editor to be kept here.

Layout (the same as the editor): <input_dir> contains images/ and labels/. The sidecar is looked
for as <input_dir>/far_regions_*.json, then as <parent of input_dir>/far_regions_<name of input_dir>.json
(where the editor writes it when started with --input_dir <parent> --split <name>), or give --far_file.

This edits the dataset IN PLACE: make a backup first. Images are rewritten as new files (a
hardlink to another copy of the dataset keeps the original) and PNG stays PNG, JPEG is re-encoded at
--jpeg_quality. Running it twice is harmless: black stays black, removed labels stay removed.
Kept label rows are copied byte for byte (line endings preserved); rows that are not plain
`cls cx cy w h` are kept untouched. <input_dir>/labels.cache is deleted when a label file changed.
A report of every removed label is written next to the sidecar: removed_far_labels_<time>.csv.

Without --apply nothing is written (dry run): the counts are printed only.

    python src/dataset/apply_far_regions.py --input_dir <dir with images and labels>            # dry run
    python src/dataset/apply_far_regions.py --input_dir <dir with images and labels> --apply
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_small_labels import editor_running  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_far_file(input_dir, explicit=None):
    if explicit:
        return Path(explicit)
    for d, pat in ((input_dir, "far_regions_*.json"), (input_dir.parent, f"far_regions_{input_dir.name}.json")):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    return None


def load_regions(path):
    """{image file name: [polygon (K x 2 float), ...]}; version-1 rectangles [x1, y1, x2, y2] become 4 corners."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, polys in (data.get("images") or {}).items():
        good = []
        for p in polys if isinstance(polys, list) else []:
            try:
                a = np.asarray(p, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if a.shape == (4,):
                x1, y1, x2, y2 = a
                a = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            if a.ndim == 2 and a.shape[1] == 2 and len(a) >= 3 and np.isfinite(a).all():
                good.append(a)
        if good:
            out[name] = good
    return out


def region_mask(polys, w, h):
    """uint8 mask (h x w), 1 where a region is: the union of the polygons."""
    mask = np.zeros((h, w), np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [np.round(p).astype(np.int32)], 1)
    return mask


def covered_fractions(rows, mask):
    """Fraction of each box (x1, y1, x2, y2 pixels) that lies in the mask, via an integral image."""
    h, w = mask.shape
    integral = cv2.integral(mask, sdepth=cv2.CV_32S)
    out = []
    for x1, y1, x2, y2 in rows:
        a, b = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        c, d = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        area = (c - a) * (d - b)
        if area <= 0:
            out.append(0.0)
            continue
        s = int(integral[d, c]) - int(integral[b, c]) - int(integral[d, a]) + int(integral[b, a])
        out.append(s / area)
    return out


def filter_label_text(raw, w, h, mask, min_cover):
    """-> (new text, removed [(row text, cls, bw, bh, fraction)], kept rows that are partly under a region).
    Rows that do not parse as `cls cx cy w h` are kept."""
    lines = raw.splitlines(keepends=True)
    parsed = []
    for i, line in enumerate(lines):
        t = line.split()
        if len(t) != 5:
            continue
        try:
            cls, (cx, cy, bw, bh) = int(float(t[0])), (float(v) for v in t[1:])
        except ValueError:
            continue
        parsed.append((i, cls, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h,
                       bw * w, bh * h))
    fr = covered_fractions([p[2:6] for p in parsed], mask) if parsed else []
    drop, removed, partial = set(), [], 0
    for p, f in zip(parsed, fr):
        if f > min_cover:
            drop.add(p[0])
            removed.append((lines[p[0]].strip(), p[1], p[6], p[7], f))
        elif f > 0:
            partial += 1
    return "".join(l for i, l in enumerate(lines) if i not in drop), removed, partial


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def write_new_file(path, data):
    """Write as a new file and swap it in (a hardlink elsewhere keeps the old content)."""
    tmp = path.with_name(path.name + ".tmp_far")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def encode_image(img, suffix, jpeg_quality):
    suffix = suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if suffix in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(suffix, img, params)
    if not ok:
        raise RuntimeError("could not encode " + suffix)
    return buf.tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, help="folder with images/ and labels/ (edited in place)")
    ap.add_argument("--far_file", default=None, help="far_regions_<split>.json (default: found next to --input_dir)")
    ap.add_argument("--min_cover", type=float, default=0.8,
                    help="remove a label when more than this share of its area is under the regions")
    ap.add_argument("--jpeg_quality", type=int, default=95, help="quality for rewritten .jpg images")
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run, nothing written)")
    ap.add_argument("--ignore_running_editor", action="store_true",
                    help="skip the check for a running label editor")
    args = ap.parse_args()

    root = Path(args.input_dir)
    img_dir, lab_dir = root / "images", root / "labels"
    if not img_dir.is_dir() or not lab_dir.is_dir():
        sys.exit(f"{root} must contain images/ and labels/")
    far_file = find_far_file(root, args.far_file)
    if far_file is None or not far_file.exists():
        sys.exit(f"no far_regions_*.json found next to {root} (pass --far_file)")
    regions = load_regions(far_file)
    print(f"far regions: {far_file}  ({len(regions)} images with regions)")

    if args.apply and not args.ignore_running_editor:
        pids = editor_running()
        if pids:
            sys.exit(f"the label editor (analyse_predictions_ui) is running, pid {pids}.\n"
                     f"Close it first: its auto-save would write the old labels back "
                     f"(--ignore_running_editor skips this check).")

    on_disk = {p.name: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    missing = sorted(n for n in regions if n not in on_disk)
    report, n_img = [], 0
    n_labels_files = n_removed = n_kept_partial = n_rows = 0
    black_px = 0
    t0 = time.perf_counter()
    for k, (name, polys) in enumerate(sorted(regions.items()), 1):
        if name not in on_disk:
            continue
        img_path = on_disk[name]
        img = read_image(img_path)
        if img is None:
            print(f"  cannot read {name}, skipped")
            continue
        h, w = img.shape[:2]
        mask = region_mask(polys, w, h)
        painted = int(mask.sum())
        if painted:
            n_img += 1
            black_px += painted
            if args.apply and img[mask.astype(bool)].any():  # already black: leave the file alone
                img[mask.astype(bool)] = 0
                write_new_file(img_path, encode_image(img, img_path.suffix, args.jpeg_quality))
        lab_path = lab_dir / (img_path.stem + ".txt")
        if lab_path.exists():
            raw = lab_path.read_bytes().decode("utf-8")
            new, removed, partial = filter_label_text(raw, w, h, mask, args.min_cover)
            n_rows += len(raw.splitlines())
            n_kept_partial += partial
            if removed:
                n_labels_files += 1
                n_removed += len(removed)
                report += [(name, row, cls, round(bw, 1), round(bh, 1), round(f, 3)) for row, cls, bw, bh, f in removed]
                if args.apply:
                    write_new_file(lab_path, new.encode("utf-8"))
        if k % 50 == 0:
            print(f"  {k}/{len(regions)}", flush=True)

    print(f"\n{'applied' if args.apply else 'DRY RUN - nothing written'} in {time.perf_counter() - t0:.0f}s")
    print(f"  images with painted regions : {n_img}  ({black_px / max(1, n_img) / 1e6:.2f} Mpx black on average)")
    print(f"  labels removed (> {args.min_cover:.0%} covered): {n_removed} in {n_labels_files} label files "
          f"(of {n_rows} rows in those images)")
    print(f"  labels kept although partly under a region: {n_kept_partial}")
    if missing:
        print(f"  in the far file but no image on disk: {len(missing)} (e.g. {missing[0]})")
    if args.apply:
        if n_labels_files:
            (root / "labels.cache").unlink(missing_ok=True)  # ultralytics trusts it by total size
        rep = far_file.parent / f"removed_far_labels_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(rep, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["image", "label_row", "class", "width_px", "height_px", "share_covered"])
            wr.writerows(report)
        print(f"  report: {rep}")
    else:
        print("  run again with --apply to write it")


if __name__ == "__main__":
    main()
