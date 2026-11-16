"""
Detector choice (YOLO / RF-DETR) in analyse_predictions_ui.py.

  * run_rfdetr_predictions drives rfdetr_predict.py in its own environment: progress lines, the cache it
    writes, a failing run, a missing environment (a fake script stands in for RF-DETR here)
  * switching the Detector box loads that detector's own prediction cache, so the two can be compared
  * the header of an RF-DETR cache says NMS-free

Run:  python -m pytest src/human_tracking/tests/test_detector_choice.py -q
"""

import json
import sys
import threading
from pathlib import Path

import pytest

from .test_far_regions import U, goto, horizon, scenes, ui  # noqa: F401  (fixtures)

FAKE = '''
import argparse, json, sys
ap = argparse.ArgumentParser()
for a in ("--weights", "--images", "--out", "--conf"):
    ap.add_argument(a)
x = ap.parse_args()
imgs = [l for l in open(x.images, encoding="utf-8").read().splitlines() if l]
print(f"PROGRESS 0 {len(imgs)}", flush=True)
for i, _ in enumerate(imgs, 1):
    print(f"PROGRESS {i} {len(imgs)}", flush=True)
if "fail" in x.weights:
    print("boom: something broke", flush=True)
    sys.exit(3)
names = [l.replace("\\\\", "/").rsplit("/", 1)[-1] for l in imgs]
json.dump(dict(model=x.weights, detector="rfdetr", imgsz=1220, conf=float(x.conf), nms_iou=None,
               images={n: [[10, 10, 50, 90, 0.9]] for n in names}), open(x.out, "w"))
'''


@pytest.fixture
def fake_script(tmp_path):
    p = tmp_path / "fake_rfdetr_predict.py"
    p.write_text(FAKE)
    return p


def test_runner_reports_progress_and_reads_the_cache(tmp_path, fake_script):
    images = [tmp_path / f"im{i}.png" for i in range(3)]
    seen = []
    data = U.run_rfdetr_predictions(sys.executable, "w.pth", images, 0.05, tmp_path / "out.json",
                                    progress=lambda n, t: seen.append((n, t)), script=fake_script)
    assert seen == [(1, 3), (2, 3), (3, 3)]
    assert data["nms_iou"] is None and set(data["images"]) == {"im0.png", "im1.png", "im2.png"}


def test_runner_failure_shows_the_output(tmp_path, fake_script):
    with pytest.raises(RuntimeError, match="boom: something broke"):
        U.run_rfdetr_predictions(sys.executable, "fail.pth", [tmp_path / "a.png"], 0.05, tmp_path / "o.json",
                                 script=fake_script)


def test_runner_stop(tmp_path, fake_script):
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        U.run_rfdetr_predictions(sys.executable, "w.pth", [tmp_path / "a.png"], 0.05, tmp_path / "o.json",
                                 stop=stop, script=fake_script)


def test_missing_environment_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="RF-DETR environment not found"):
        U.run_rfdetr_predictions(tmp_path / "nope" / "python.exe", "w.pth", [], 0.05, tmp_path / "o.json")


def test_switching_detector_loads_its_own_cache(ui, tmp_path):
    app = ui(extra=["--rfdetr_weights", str(tmp_path / "rfdetr_xlarge_1220.pth")])
    it = goto(app, "b")
    cache = U.default_pred_path(app.args.input_dir, app.args.split, app.args.rfdetr_weights)
    assert cache.name == "predictions_test_rfdetr_xlarge_1220.json"
    data = {"detector": "rfdetr", "conf": 0.05, "nms_iou": None,
            "images": {i.path.name: [] for i in app.items}}
    data["images"]["b.png"] = [[1000.0, 200.0, 1060.0, 300.0, 0.8]]
    cache.write_text(json.dumps(data))

    app.detector_var.set("RF-DETR")
    app._on_detector()
    app.root.update()
    assert app.pred_path == cache
    assert len(it.pred) == 1
    assert "NMS-free" in app.hdr_pred.cget("text")

    app.detector_var.set("YOLO")                      # no YOLO cache here: the RF-DETR one stays shown
    app._on_detector()
    assert app.pred_path == cache
    assert "no YOLO predictions" in app.status.cget("text")


def test_detector_from_command_line(ui):
    app = ui(extra=["--detector", "rfdetr"])
    assert app.detector_var.get() == "RF-DETR"
    assert app._weights() == app.args.rfdetr_weights
