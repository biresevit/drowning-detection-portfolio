"""
Swimming-speed gate on the direction exclusion.

Replays real tracks from tests/speed_gate/*.json through get_drowning_candidates with
the frame numbers and head predictions they were recorded with. Each file carries
"expected": "candidate" or "excluded_direction".

The slow tracks (20260625_205127 ids 18 and 49) passed the direction test - their
paths really are straight - but moved at 0.20-0.22 extents/s. The 0fcc12e2 ids 2 and 6
are swimmers filmed from above moving along image Y (0.35-0.41 extents/s) and must stay
excluded; the fast swimmers of both videos move at 0.70-1.12.
"""

import json
from pathlib import Path

import pytest

from src.human_tracking.behavior_analysis import BehaviorAnalyses

_CASES = sorted((Path(__file__).parent / "speed_gate").glob("*.json"))


def _make_ba():
    # any non-None classifier enables the head check; predictions come from the fixture
    return BehaviorAnalyses(head_classifier=object())


def _classify(case: dict, with_frames: bool = True):
    ba = _make_ba()
    tid = "0"
    preds = [tuple(p) for p in case["head_predictions"]]
    candidates, excl_dir, excl_change, excl_head = ba.get_drowning_candidates(
        {tid: case["bboxes"]},
        head_preds_cache={tid: preds} if preds else None,
        head_visible_frame_count=0,
        head_visibility_min_frames=16,
        frame_history={tid: case["frames"]} if with_frames else None,
        processed_fps=case["processed_fps"] if with_frames else None,
    )
    for name, ids in (("excluded_direction", excl_dir), ("excluded_direction_change", excl_change),
                      ("excluded_head_visible", excl_head)):
        if tid in ids:
            return name, ba
    return "candidate", ba


@pytest.mark.parametrize("path", _CASES, ids=[p.stem for p in _CASES])
def test_speed_gate(path: Path):
    case = json.loads(path.read_text())
    result, ba = _classify(case)
    assert result == case["expected"], (
        f"{path.name} ({case['note']}): expected {case['expected']}, got {result} | "
        f"motion={ba.motion_stats.get('0')}"
    )


@pytest.mark.parametrize("path", _CASES, ids=[p.stem for p in _CASES])
def test_all_pass_direction_test_without_frames(path: Path):
    """Every fixture is linear: without frame numbers the gate is skipped and all are excluded."""
    case = json.loads(path.read_text())
    result, _ = _classify(case, with_frames=False)
    assert result == "excluded_direction"


def test_head_under_requires_longer_run():
    """Fast enough, but face-down and short: kept until the run reaches the stricter bar."""
    ba = BehaviorAnalyses()
    bboxes = [[100 + 10 * i, 100, 140 + 10 * i, 140] for i in range(20)]   # 5 bbox over 19 frames
    frames = list(range(20))
    assert ba._swimming_confirmed("t", bboxes, frames, 6.0, head_under_frac=1.0) is True

    short = bboxes[:8]                                                     # 1.75 extents, 1.5 extents/s
    assert ba._swimming_confirmed("t", short, frames[:8], 6.0, head_under_frac=1.0) is False
    assert ba.motion_stats["t"]["kept_because"].startswith("head under water")
    assert ba._swimming_confirmed("t", short, frames[:8], 6.0, head_under_frac=0.2) is True
    assert ba._swimming_confirmed("t", short, frames[:8], 6.0, head_under_frac=None) is True


def test_speed_uses_extent_along_travel():
    """A long thin box moving along its long side is judged by its length, not sqrt(w*h)."""
    ba = BehaviorAnalyses()
    # 160x40 box moving 4 px/frame along x at 6 fps: 0.15 lengths/s, but 0.6 sqrt(w*h)/s
    bboxes = [[100 + 4 * i, 100, 260 + 4 * i, 140] for i in range(30)]
    assert ba._swimming_confirmed("t", bboxes, list(range(30)), 6.0, None) is False
    assert ba.motion_stats["t"]["extent_px"] == 160.0
    # the same box moving along y is judged by its height (40 px): 0.6 extents/s
    bboxes = [[100, 100 + 4 * i, 260, 140 + 4 * i] for i in range(30)]
    assert ba._swimming_confirmed("t", bboxes, list(range(30)), 6.0, None) is True


def test_gate_skipped_when_frames_misaligned():
    ba = BehaviorAnalyses()
    bboxes = [[100 + i, 100, 140 + i, 140] for i in range(20)]            # very slow
    assert ba._swimming_confirmed("t", bboxes, list(range(19)), 6.0, None) is True
    assert ba._swimming_confirmed("t", bboxes, list(range(20)), None, None) is True
    assert ba._swimming_confirmed("t", bboxes, list(range(20)), 6.0, None) is False
