"""Tests of src/dataset/apply_far_regions.py: black pixels under far regions, labels > 80% covered removed."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "dataset"))
import apply_far_regions as A  # noqa: E402

W, H = 400, 200
SCRIPT = ROOT / "src" / "dataset" / "apply_far_regions.py"


def label(cls, x1, y1, x2, y2):
    return f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def make_dataset(tmp_path, regions):
    """<tmp>/split/{images,labels} plus <tmp>/far_regions_split.json, like the editor writes it."""
    split = tmp_path / "split"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir()
    rng = np.random.default_rng(0)
    for name in ("a", "b", "c"):
        img = rng.integers(40, 255, (H, W, 3), dtype=np.uint8)  # never black, so black pixels are ours
        cv2.imwrite(str(split / "images" / f"{name}.png"), img)
    rows_a = [label(0, 10, 10, 30, 30),        # fully inside the region (x < 100, y < 50)
              label(0, 80, 10, 120, 30),       # exactly half inside: kept
              label(0, 90, 10, 110, 30),       # 50% inside
              label(0, 5, 5, 105, 45),         # 95% of the box inside: removed
              label(0, 200, 100, 240, 140)]    # far away
    (split / "labels" / "a.txt").write_bytes(("\r\n".join(rows_a) + "\r\n").encode())
    (split / "labels" / "b.txt").write_bytes((label(0, 10, 10, 30, 30) + "\n").encode())  # image b has no region
    (split / "labels" / "c.txt").write_bytes(b"")
    (tmp_path / "far_regions_split.json").write_text(json.dumps({"version": 2, "images": regions}))
    return split


REGION = [[[0, 0], [100, 0], [100, 50], [0, 50]]]


def run(split, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--input_dir", str(split), "--ignore_running_editor", *extra],
                          capture_output=True, text=True)


def read_labels(split, name):
    return (split / "labels" / f"{name}.txt").read_bytes().decode().splitlines()


def test_dry_run_writes_nothing(tmp_path):
    split = make_dataset(tmp_path, {"a.png": REGION})
    before = {p.name: p.read_bytes() for p in list((split / "images").iterdir()) + list((split / "labels").iterdir())}
    r = run(split)
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout and "labels removed (> 80% covered): 2" in r.stdout, r.stdout
    after = {p.name: p.read_bytes() for p in list((split / "images").iterdir()) + list((split / "labels").iterdir())}
    assert before == after
    assert not list(tmp_path.glob("removed_far_labels_*.csv"))


def test_apply_blackens_the_region_and_nothing_else(tmp_path):
    split = make_dataset(tmp_path, {"a.png": REGION})
    orig = cv2.imread(str(split / "images" / "a.png"))
    others = {n: (split / "images" / f"{n}.png").read_bytes() for n in ("b", "c")}
    r = run(split, "--apply")
    assert r.returncode == 0, r.stderr
    new = cv2.imread(str(split / "images" / "a.png"))
    assert new.shape == orig.shape
    inside = np.zeros((H, W), bool)
    inside[:51, :101] = True                        # polygon incl. its border pixels
    assert (new[:49, :99] == 0).all(), "the region is black"
    assert (new[inside == False] == orig[inside == False]).all(), "outside the region nothing changes"  # noqa: E712
    assert {n: (split / "images" / f"{n}.png").read_bytes() for n in ("b", "c")} == others, "other images untouched"


def test_labels_over_the_threshold_are_removed_the_rest_kept_byte_for_byte(tmp_path):
    split = make_dataset(tmp_path, {"a.png": REGION})
    kept_before = read_labels(split, "a")
    r = run(split, "--apply")
    assert r.returncode == 0, r.stderr
    rows = read_labels(split, "a")
    assert rows == [kept_before[1], kept_before[2], kept_before[4]], "boxes 0 (100%) and 3 (95%) are gone"
    raw = (split / "labels" / "a.txt").read_bytes()
    assert raw.endswith(b"\r\n") and b"\r\n" in raw, "line endings preserved"
    assert read_labels(split, "b") == [label(0, 10, 10, 30, 30)], "an image without a region keeps its labels"


def test_share_covered_decides_not_the_centre(tmp_path):
    split = make_dataset(tmp_path, {"a.png": REGION})
    # region is y < 50: a 100 px wide box of height 66.7 is 75% covered (kept), height 58.8 is 85% (removed)
    (split / "labels" / "a.txt").write_text("\n".join([label(0, 0, 0, 100, 66.7), label(0, 0, 0, 100, 58.8)]) + "\n")
    run(split, "--apply")
    rows = read_labels(split, "a")
    assert len(rows) == 1 and abs(float(rows[0].split()[4]) * H - 66.7) < 0.1


def test_overlapping_regions_are_a_union(tmp_path):
    two = [[[0, 0], [60, 0], [60, 50], [0, 50]], [[40, 0], [100, 0], [100, 50], [40, 50]]]
    split = make_dataset(tmp_path, {"a.png": two})
    (split / "labels" / "a.txt").write_text(label(0, 20, 5, 80, 45) + "\n")     # spans both: fully covered
    run(split, "--apply")
    assert read_labels(split, "a") == []


def test_polygon_not_just_rectangles(tmp_path):
    tri = [[[0, 0], [200, 0], [0, 100]]]
    split = make_dataset(tmp_path, {"a.png": tri})
    (split / "labels" / "a.txt").write_text("\n".join([label(0, 10, 10, 30, 30),     # inside the triangle
                                                       label(0, 150, 60, 190, 90)]) + "\n")  # outside it
    run(split, "--apply")
    rows = read_labels(split, "a")
    assert len(rows) == 1 and abs(float(rows[0].split()[1]) * W - 170) < 0.5
    img = cv2.imread(str(split / "images" / "a.png"))
    assert (img[10:30, 10:30] == 0).all() and (img[60:90, 150:190] != 0).any()


def test_applying_twice_changes_nothing_more(tmp_path):
    split = make_dataset(tmp_path, {"a.png": REGION})
    run(split, "--apply")
    snap = {p.name: p.read_bytes() for p in list((split / "images").iterdir()) + list((split / "labels").iterdir())}
    r = run(split, "--apply")
    assert r.returncode == 0, r.stderr
    assert snap == {p.name: p.read_bytes() for p in list((split / "images").iterdir()) + list((split / "labels").iterdir())}
    assert "labels removed (> 80% covered): 0" in r.stdout


def test_report_and_cache(tmp_path):
    split = make_dataset(tmp_path, {"a.png": REGION})
    (split / "labels.cache").write_bytes(b"x")
    run(split, "--apply")
    assert not (split / "labels.cache").exists(), "stale ultralytics cache is deleted"
    (rep,) = tmp_path.glob("removed_far_labels_*.csv")
    lines = rep.read_text().strip().splitlines()
    assert len(lines) == 3 and lines[0].startswith("image,label_row") and all(l.startswith("a.png,") for l in lines[1:])


def test_version_1_rectangles_and_bad_entries_load(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"version": 1, "images": {"r.png": [[10, 20, 300, 400]], "bad.png": "nope",
                                                       "x.png": [[[0, 0], [5, 5]], [[0, 0], [10, 0], [10, 10]]]}}))
    got = A.load_regions(p)
    assert sorted(got) == ["r.png", "x.png"] and got["r.png"][0].shape == (4, 2) and len(got["x.png"]) == 1


def test_region_mask_and_fractions():
    mask = A.region_mask([np.array([[0, 0], [100, 0], [100, 50], [0, 50]], float)], W, H)
    f = A.covered_fractions([(0, 0, 100, 50), (50, 0, 150, 50), (200, 100, 240, 140), (-20, -20, 20, 20)], mask)
    assert f[0] > 0.97 and abs(f[1] - 0.5) < 0.03 and f[2] == 0
    assert f[3] == 1.0, 'a box sticking out of the image is measured on its visible part'


def test_missing_far_file_and_layout_errors(tmp_path):
    split = tmp_path / "s"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir()
    r = run(split)
    assert r.returncode != 0 and "no far_regions" in (r.stdout + r.stderr)
    r = run(tmp_path / "nope")
    assert r.returncode != 0 and "images/ and labels/" in (r.stdout + r.stderr)


def test_far_file_found_in_the_folder_or_its_parent(tmp_path):
    split = make_dataset(tmp_path, {"a.png": REGION})
    assert A.find_far_file(split) == tmp_path / "far_regions_split.json"
    (split / "far_regions_x.json").write_text("{}")
    assert A.find_far_file(split) == split / "far_regions_x.json"
    assert A.find_far_file(split, "z.json") == Path("z.json")
