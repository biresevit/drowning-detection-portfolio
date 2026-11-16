"""Tests of src/dataset/downscale_dataset.py."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "src" / "dataset" / "downscale_dataset.py"
sys.path.insert(0, str(ROOT / "src" / "dataset"))
import downscale_dataset as D  # noqa: E402


def make(tmp_path):
    ds = tmp_path / "ds"
    for split in ("train", "val"):
        (ds / split / "images").mkdir(parents=True)
        (ds / split / "labels").mkdir()
    rng = np.random.default_rng(0)
    for split, name, w, h, ext in (("train", "big", 3840, 1800, ".png"), ("train", "small", 1280, 600, ".png"),
                                   ("val", "edge", 1920, 900, ".png"), ("val", "bigjpg", 2400, 1000, ".jpg")):
        img = cv2.resize(rng.integers(0, 255, (9, 16, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(ds / split / "images" / (name + ext)), img)
        (ds / split / "labels" / (name + ".txt")).write_text("0 0.5 0.5 0.2 0.3\r\n")
    (ds / "train" / "labels.cache").write_bytes(b"x")
    return ds


def run(ds, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--dataset", str(ds), *extra], capture_output=True, text=True)


def shape(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR).shape[:2]


def test_target_size_keeps_the_aspect_ratio():
    assert D.target_size(3840, 1800, 1920) == (1920, 900)
    assert D.target_size(1800, 3840, 1920) == (900, 1920)
    assert D.target_size(1920, 900, 1920) is None and D.target_size(1280, 600, 1920) is None
    assert D.target_size(3201, 1599, 1920) == (1920, 959)


def test_dry_run_writes_nothing(tmp_path):
    ds = make(tmp_path)
    before = {p: p.read_bytes() for p in ds.rglob("*") if p.is_file()}
    r = run(ds)
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout and "images to shrink: 2 of 4" in r.stdout
    assert before == {p: p.read_bytes() for p in ds.rglob("*") if p.is_file()}


def test_only_large_images_shrink_and_labels_are_untouched(tmp_path):
    ds = make(tmp_path)
    labels = {p: p.read_bytes() for p in ds.rglob("*.txt")}
    small_bytes = (ds / "train" / "images" / "small.png").read_bytes()
    edge_bytes = (ds / "val" / "images" / "edge.png").read_bytes()
    r = run(ds, "--apply")
    assert r.returncode == 0, r.stderr
    assert shape(ds / "train" / "images" / "big.png") == (900, 1920)
    assert shape(ds / "val" / "images" / "bigjpg.jpg") == (800, 1920)
    assert (ds / "train" / "images" / "small.png").read_bytes() == small_bytes, "small images are left alone"
    assert (ds / "val" / "images" / "edge.png").read_bytes() == edge_bytes, "exactly the limit is left alone"
    assert labels == {p: p.read_bytes() for p in ds.rglob("*.txt")}, "labels are byte for byte the same"
    assert not (ds / "train" / "labels.cache").exists()
    assert not list(ds.rglob("*.tmp_small"))


def test_applying_twice_changes_nothing_more(tmp_path):
    ds = make(tmp_path)
    run(ds, "--apply")
    snap = {p: p.read_bytes() for p in ds.rglob("*") if p.is_file()}
    r = run(ds, "--apply")
    assert r.returncode == 0 and "images to shrink: 0 of 4" in r.stdout
    assert snap == {p: p.read_bytes() for p in ds.rglob("*") if p.is_file()}


def test_a_hardlinked_original_keeps_its_full_size(tmp_path):
    ds = make(tmp_path)
    link = tmp_path / "pool_big.png"
    import os
    os.link(ds / "train" / "images" / "big.png", link)
    run(ds, "--apply")
    assert shape(link) == (1800, 3840) and shape(ds / "train" / "images" / "big.png") == (900, 1920)


def test_max_side_option_and_an_unreadable_file_is_reported(tmp_path):
    ds = make(tmp_path)
    (ds / "val" / "images" / "broken.png").write_bytes(b"not an image")
    r = run(ds, "--apply", "--max_side", "1000")
    assert r.returncode == 1 and "ERRORS: 1" in r.stdout and "broken.png" in r.stdout
    assert shape(ds / "train" / "images" / "big.png") == (469, 1000), "the other images were still shrunk"
