"""Remove the labels that are too small to be a person from a YOLO detection dataset.

A label is dropped when, in native image pixels,
    sqrt(w*h) < --min_size   (default 4)   - below this the object is a few pixels: the 1280 model
                                              finds ~3% of 3-4 px labels, and about half of them show
                                              nothing identifiable
    min(w, h) < --min_side   (default 2)   - a 1-2 px thick sliver is not a person (6x1, 12x1 px boxes
                                              turn up, e.g. along the edge of the water mask)
Nothing larger is touched on purpose: 10-12 px heads look the same as 13-20 px ones, so deleting
there would teach the model that visible swimmers are background (see the size-bucket metrics).

The analyse_predictions_ui editor prints sizes rounded to whole pixels, so a box shown as "4px" can
be anywhere in 3.5-4.5. To remove everything the editor labels 4px or smaller, use --min_size 4.5.

Two modes:

  copy (default)   writes <input_dir>_clean: images hardlinked, label files new independent copies.
                   The master is only read.
  --in_place       edits the label files of <input_dir> itself. For every file that changes:
                     * its first original is kept once under <input_dir>/_label_backups/<split>/
                       (the same place and rule as the editor's Save: later runs never replace it)
                     * the new content is written as a NEW file, so a hardlink to the pool the split
                       was made from keeps the original labels (a rename detaches this copy)
                     * <split>/labels.cache is deleted: ultralytics trusts it by total file size, and
                       removing rows does change sizes, but a stale cache is not worth the risk
                   Files with nothing to remove are not touched. A report of every removed row is
                   written to <input_dir>/removed_small_labels_<time>.csv. Close the label editor
                   first: it holds labels in memory and its auto-save would write the boxes back.

In both modes kept rows are copied byte for byte (line endings preserved), and rows that are not
plain `cls cx cy w h` are kept untouched rather than guessed at.

    python src/dataset/filter_small_labels.py --in_place --min_size 4.5 --dry_run
    python src/dataset/filter_small_labels.py --in_place --min_size 4.5
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def drop_reason(bw, bh, min_size, min_side):
    """Why a box of bw x bh native pixels is dropped, or None to keep it."""
    if min(bw, bh) < min_side:
        return "side"
    if (bw * bh) ** 0.5 < min_size:
        return "size"
    return None


def filter_label_text(raw, w, h, min_size, min_side):
    """Split a label file into (kept rows, removed [(row, bw, bh, reason)], unparsed row count, eol)."""
    eol = "\r\n" if "\r\n" in raw else "\n"
    kept, removed, unparsed = [], [], 0
    for line in raw.split("\n"):
        row = line.strip()
        if not row:
            continue
        p = row.split()
        try:
            if len(p) != 5:
                raise ValueError
            bw, bh = float(p[3]) * w, float(p[4]) * h
        except ValueError:
            kept.append(row)  # not a plain detection row: keep it exactly as it is
            unparsed += 1
            continue
        reason = drop_reason(bw, bh, min_size, min_side)
        if reason:
            removed.append((row, bw, bh, reason))
        else:
            kept.append(row)
    return kept, removed, unparsed, eol


def editor_running():
    """PIDs of a running label editor (best effort, Windows). Its auto-save would undo an in-place run."""
    if os.name != "nt":
        return []
    cmd = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and "
           "$_.CommandLine -match 'analyse_predictions_u[i]' } | ForEach-Object { $_.ProcessId }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                             timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return []  # cannot tell: the docstring and the message still say to close it
    return [int(x) for x in out if x.isdigit() and int(x) != os.getpid()]


def process_split(src, dst, split, args, report):
    img_dir, lbl_dir = src / split / "images", src / split / "labels"
    stats = dict(images=0, rows=0, removed_side=0, removed_size=0, kept=0, unparsed=0, emptied=0, copied_imgs=0,
                 files_changed=0)
    if not args.in_place and not args.dry_run:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
    for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        stats["images"] += 1
        label = lbl_dir / f"{img.stem}.txt"
        with Image.open(img) as im:  # header only
            w, h = im.size
        if label.is_file():
            kept, removed, unparsed, eol = filter_label_text(label.read_bytes().decode("utf-8"), w, h,
                                                             args.min_size, args.min_side)
            stats["rows"] += len(kept) + len(removed)
            stats["kept"] += len(kept)
            stats["unparsed"] += unparsed
            for row, bw, bh, reason in removed:
                stats["removed_" + reason] += 1
                report.append((split, img.name, row, f"{bw:.2f}", f"{bh:.2f}", f"{(bw * bh) ** 0.5:.2f}", reason))
            if removed and not kept:
                stats["emptied"] += 1  # the image now has no label at all: a background image
            new_text = "".join(r + eol for r in kept).encode("utf-8")
            if args.in_place:
                if removed and not args.dry_run:
                    backup = src / "_label_backups" / split / label.name
                    if not backup.exists():  # the FIRST original is kept, never an intermediate state
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(label, backup)
                    tmp = label.with_name("." + label.name + ".tmp")
                    tmp.write_bytes(new_text)
                    os.replace(tmp, label)  # a new file: any hardlink elsewhere keeps the old content
                    stats["files_changed"] += 1
            elif not args.dry_run:
                (dst / split / "labels" / label.name).write_bytes(new_text)
        if not args.in_place and not args.dry_run:
            target = dst / split / "images" / img.name
            try:
                os.link(img, target)
            except OSError:  # another volume, or no hardlink support
                shutil.copy2(img, target)
                stats["copied_imgs"] += 1
    if args.in_place and stats["files_changed"]:
        (src / split / "labels.cache").unlink(missing_ok=True)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", default=r"path/to/data")
    ap.add_argument("--output_dir", default=None, help="copy mode only; default: <input_dir>_clean")
    ap.add_argument("--in_place", action="store_true", help="edit the label files of --input_dir itself")
    ap.add_argument("--min_size", type=float, default=4.0, help="drop labels with sqrt(w*h) below this, px")
    ap.add_argument("--min_side", type=float, default=2.0, help="drop labels with a side below this, px")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--overwrite", action="store_true", help="copy mode: replace an existing output (its splits only)")
    ap.add_argument("--dry_run", action="store_true", help="count only, write nothing")
    ap.add_argument("--ignore_running_editor", action="store_true",
                    help="in-place: skip the check for a running label editor")
    args = ap.parse_args()

    src = Path(args.input_dir)
    splits = [s for s in args.splits.split(",") if s]
    for s in splits:
        if not (src / s / "images").is_dir():
            sys.exit(f"no {src / s / 'images'}")

    if args.in_place:
        if args.output_dir:
            sys.exit("--in_place edits the input dataset itself; it cannot be combined with --output_dir")
        dst = src
        if not args.dry_run and not args.ignore_running_editor:
            pids = editor_running()
            if pids:
                sys.exit(f"the label editor (analyse_predictions_ui) is running, pid {pids}.\n"
                         f"It keeps labels in memory and its auto-save would write the removed boxes back.\n"
                         f"Close it, then run this again (--ignore_running_editor skips this check).")
    else:
        dst = Path(args.output_dir) if args.output_dir else src.with_name(src.name + "_clean")
        if not args.dry_run:
            busy = [s for s in splits if (dst / s).exists()]
            if busy and not args.overwrite:
                sys.exit(f"{dst} already has {busy}; pass --overwrite to replace them")
            for s in busy:
                shutil.rmtree(dst / s)  # only unlinks this copy; hardlinked images in the master stay
            dst.mkdir(parents=True, exist_ok=True)

    report, totals = [], {}
    head = f"{'in place: ' if args.in_place else ''}{'dry run: ' if args.dry_run else ''}{src}"
    print(head if args.in_place else f"{head} -> {dst}")
    print(f"drop: sqrt(w*h) < {args.min_size:g} px, or a side < {args.min_side:g} px (native image pixels)\n")
    print(f"{'split':<6}{'images':>8}{'labels':>9}{'side<2':>8}{'sqrt<4':>8}{'removed':>9}{'share':>8}{'left':>9}"
          f"{'images emptied':>16}" + (f"{'files changed':>15}" if args.in_place else ""))
    for s in splits:
        st = totals[s] = process_split(src, dst, s, args, report)
        gone = st["removed_side"] + st["removed_size"]
        print(f"{s:<6}{st['images']:>8}{st['rows']:>9}{st['removed_side']:>8}{st['removed_size']:>8}{gone:>9}"
              f"{100 * gone / max(1, st['rows']):>7.2f}%{st['kept']:>9}{st['emptied']:>16}"
              + (f"{st['files_changed']:>15}" if args.in_place else ""))
        if st["unparsed"]:
            print(f"      ({st['unparsed']} rows were not plain 'cls cx cy w h' and were kept as they are)")
        if st["copied_imgs"]:
            print(f"      ({st['copied_imgs']} images were copied because hardlinking failed)")

    if args.dry_run:
        print("\ndry run - nothing written")
        return

    if args.in_place:
        if report:
            path = src / f"removed_small_labels_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["split", "image", "row", "w_px", "h_px", "sqrt_px", "reason"])
                w.writerows(report)
            print(f"\nremoved {len(report)} labels from {sum(t['files_changed'] for t in totals.values())} files")
            print(f"  report: {path}")
            print(f"  first originals: {src / '_label_backups'}   |   labels.cache deleted in the changed splits")
        else:
            print("\nnothing to remove - no file was touched")
        return

    with open(dst / "removed_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "row", "w_px", "h_px", "sqrt_px", "reason"])
        w.writerows(report)
    yaml_src = src / "data.yaml"
    if yaml_src.is_file():
        text = re.sub(r"(?m)^path:.*$", lambda m: f"path: {dst}", yaml_src.read_text(encoding="utf-8"))
        (dst / "data.yaml").write_text(text, encoding="utf-8")
    print(f"\nwrote {dst}")
    print(f"  removed_labels.csv: {len(report)} rows   |   train with: {dst / 'data.yaml'}")
    print("  images emptied = images whose only labels were dropped; they are now background images")


if __name__ == "__main__":
    main()
