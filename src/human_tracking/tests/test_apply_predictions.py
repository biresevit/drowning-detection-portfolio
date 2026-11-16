"""
Apply one [O] and Apply all predictions [A] in analyse_predictions_ui.py.

  * a click (no drag) on the left panel selects the prediction under it; a drag still pans
  * Apply one adds the selected prediction as a new annotation and selects it on the right
  * Apply all adds every shown prediction that matches no annotation, skipping matched ones and
    ones under a far region, in a single undo step

Run:  python -m pytest src/human_tracking/tests/test_apply_predictions.py -q
"""

from types import SimpleNamespace

import numpy as np

from .test_far_regions import U, goto, horizon, scenes, ui  # noqa: F401  (fixtures)

NEW = [1000.0, 200.0, 1060.0, 300.0]       # matches no annotation of scene "b"
FAR = [1500.0, 100.0, 1540.0, 140.0]       # under the far region added below


def _with_predictions(app):
    it = goto(app, "b")
    gt0 = [float(v) for v in it.gt[0]]
    data = {"conf": 0.05, "images": {i.path.name: [] for i in app.items}}
    data["images"]["b.png"] = [gt0 + [0.9], NEW + [0.8], FAR + [0.7]]
    app._attach(data, app.split_dir / "preds.json")
    app.root.update()
    return goto(app, "b")


def _click_pred(app, box, dx=0):
    a, b, c, d = app._to_scr(box)
    x, y = (a + c) / 2, (b + d) / 2
    app._on_press(SimpleNamespace(x=x, y=y, widget=app.cv_pred))
    app._pred_release(SimpleNamespace(x=x + dx, y=y, widget=app.cv_pred))


def test_click_selects_prediction_and_apply_one_adds_it(ui):
    app = ui()
    it = _with_predictions(app)
    n0 = len(it.gt)
    _click_pred(app, NEW)
    assert app._selected_pred() == (it, 1)
    app.apply_one()
    assert len(it.gt) == n0 + 1 and it.dirty
    assert np.allclose(it.gt[-1], NEW) and it.gt_cls[-1] == app.args.new_class and it.gt_src[-1] is None
    assert app._selected() == (it, n0)                # the new annotation is selected on the right
    app.undo()
    assert len(it.gt) == n0


def test_drag_on_left_panel_pans_and_does_not_select(ui):
    app = ui()
    _with_predictions(app)
    _click_pred(app, NEW, dx=40)
    assert app._selected_pred() is None


def test_apply_one_without_selection_changes_nothing(ui):
    app = ui()
    it = _with_predictions(app)
    n0 = len(it.gt)
    app.apply_one()
    assert len(it.gt) == n0 and not it.dirty


def test_apply_all_adds_only_unmatched_outside_far_regions_in_one_undo(ui):
    app = ui()
    it = _with_predictions(app)
    it.far = [np.array([[1450, 50], [1600, 50], [1600, 200], [1450, 200]], np.float32)]
    app._match_item(it)
    n0 = len(it.gt)
    app.apply_all_predictions()
    assert len(it.gt) == n0 + 1                       # matched one and the one under the far region skipped
    assert np.allclose(it.gt[-1], NEW)
    assert bool(it.hit[-1])                           # the added box is now matched by its prediction
    app.apply_all_predictions()                       # second press: nothing left to add
    assert len(it.gt) == n0 + 1
    app.undo()
    assert len(it.gt) == n0


def test_apply_all_respects_the_confidence_threshold(ui):
    app = ui()
    it = _with_predictions(app)
    app.conf_var.set("0.85")
    app._on_conf()
    app.apply_all_predictions()                       # only the 0.9 prediction is shown, and it is matched
    assert not it.dirty


def test_nul_only_label_file_is_empty_and_editable(tmp_path):
    """What a sudden shutdown leaves: the file's size reserved, every byte NUL. It holds no rows."""
    f = tmp_path / "x.txt"
    f.write_bytes(b"\x00" * 273)
    boxes, cls, src, editable = U.read_gt(f, 100, 100)
    assert len(boxes) == 0 and src == [] and editable
    f.write_bytes(b"0 0.5 0.5 0.1 0.1\r\n0 0.5 0.5 0.1\r\n")      # a real non-detection row stays read-only
    assert U.read_gt(f, 100, 100)[3] is False


def test_image_without_annotations(ui, tmp_path):
    """No annotations at all (empty label file, or none): Apply one and Apply all still add boxes."""
    app = ui()
    it = goto(app, "b")
    for missing_file in (False, True):
        it.gt, it.gt_cls, it.gt_src = np.zeros((0, 4), np.float32), np.zeros(0, int), []
        data = {"conf": 0.05, "images": {i.path.name: [] for i in app.items}}
        data["images"]["b.png"] = [NEW + [0.8], FAR + [0.7]]
        app._attach(data, app.split_dir / "preds.json")
        it = goto(app, "b")
        assert len(it.gt) == 0
        _click_pred(app, NEW)
        app.apply_one()
        assert len(it.gt) == 1, app.status.cget("text")
        app.undo()
        app.apply_all_predictions()
        assert len(it.gt) == 2, app.status.cget("text")
