"""
Data-driven tests: load every *.json file under tests/{candidate,direction,direction_change}/
and assert the expected BehaviorAnalyses classification.

Folder            → expected result
----------------    -----------------------------------------
candidate          all checks return False                    → candidate
direction          _is_consistent_direction returns True      → excluded_direction
direction_change   _is_direction_change returns True          → excluded_direction_change

Add new scenarios by dropping a .json bbox-sequence file into the appropriate folder.
Each file must be a JSON array of [x1, y1, x2, y2] bboxes in frame order.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.human_tracking.behavior_analysis import BehaviorAnalyses  # noqa: E402

_TESTS_DIR = Path(__file__).parent
_BA = BehaviorAnalyses()


def _load(subfolder: str):
    folder = _TESTS_DIR / subfolder
    if not folder.exists():
        return []
    cases = sorted(folder.glob("*.json"))
    return [pytest.param(p, id=f"{subfolder}/{p.name}") for p in cases]


def _classify(bboxes):
    candidates, excl_dir, excl_dir_change, _ = _BA.get_drowning_candidates({"0": bboxes})
    if "0" in excl_dir:
        return "excluded_direction"
    if "0" in excl_dir_change:
        return "excluded_direction_change"
    return "candidate"


@pytest.mark.parametrize("path", _load("candidate"))
def test_candidate(path: Path):
    bboxes = json.loads(path.read_text())
    result = _classify(bboxes)
    assert result == "candidate", (
        f"{path.name}: expected candidate, got {result}  "
        f"(frames={len(bboxes)})"
    )


@pytest.mark.parametrize("path", _load("direction"))
def test_direction_excluded(path: Path):
    bboxes = json.loads(path.read_text())
    result = _classify(bboxes)
    assert result == "excluded_direction", (
        f"{path.name}: expected excluded_direction, got {result}  "
        f"(frames={len(bboxes)})"
    )


@pytest.mark.parametrize("path", _load("direction_change"))
def test_direction_change_excluded(path: Path):
    bboxes = json.loads(path.read_text())
    result = _classify(bboxes)
    assert result == "excluded_direction_change", (
        f"{path.name}: expected excluded_direction_change, got {result}  "
        f"(frames={len(bboxes)})"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
