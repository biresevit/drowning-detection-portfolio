import math
import sys
import pytest
from src.human_tracking.behavior_analysis import BehaviorAnalyses


def make_bbox(cx, cy, w=40, h=80):
    """Helper: centred bbox at (cx, cy) with given width/height."""
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def straight_line_bboxes(n=20, dx=10, dy=0, w=40, h=80):
    """n bboxes moving dx,dy per step from origin (200, 200)."""
    return [make_bbox(200 + i * dx, 200 + i * dy, w, h) for i in range(n)]


def zigzag_bboxes(n=20, amplitude=30, w=40, h=80):
    """n bboxes oscillating left/right with no net progress."""
    bboxes = []
    for i in range(n):
        cx = 200 + amplitude * (1 if i % 2 == 0 else -1)
        bboxes.append(make_bbox(cx, 200, w, h))
    return bboxes


def stationary_bboxes(n=20, w=40, h=80):
    """n identical bboxes — person not moving."""
    return [make_bbox(200, 200, w, h) for _ in range(n)]


def stability_spike_bboxes(n=20, spike_fraction=0.0):
    """
    n bboxes where spike_fraction of frames are replaced with one identical
    spike bbox; all remaining frames have a unique position.

    Normal frame i: make_bbox(200 + i, 200, 40, 80)  — each distinct
    Spike frame:    make_bbox(200, 200, 80, 40)        — all identical

    spike_fraction=0.0  → 0 spike frames, all 20 distinct normals
    spike_fraction=0.10 → 2 identical spikes, 18 distinct normals
    spike_fraction=0.50 → 10 identical spikes, 10 distinct normals
    spike_fraction=1.0  → 20 identical spikes
    """
    spike_bbox = make_bbox(200, 200, 80, 40)
    k = round(n * spike_fraction)
    if k == 0:
        spike_idx = set()
    elif k == n:
        spike_idx = set(range(n))
    else:
        step = n / k
        spike_idx = {round(i * step) % n for i in range(k)}
    return [
        spike_bbox if i in spike_idx
        else make_bbox(200 + i, 200, 40, 80)
        for i in range(n)
    ]


def back_and_forth_bboxes(n=40, forward_fraction=0.5, dx=10, w=40, h=80):
    """
    n bboxes moving dx right for forward_fraction of frames, then dx left.

    With n=40 bboxes (39 steps after diff) the raw forward-step fraction is:
        round(n * forward_fraction) / (n - 1)
    which closely tracks forward_fraction and stays well clear of the 0.70
    threshold for the cases tested here.
    """
    bboxes = []
    cx = 200.0
    n_forward = round(n * forward_fraction)
    n_back = n - n_forward
    for _ in range(n_forward):
        bboxes.append(make_bbox(cx, 200, w, h))
        cx += dx
    for _ in range(n_back):
        bboxes.append(make_bbox(cx, 200, w, h))
        cx -= dx
    return bboxes


# ---------------------------------------------------------------------------
# _is_consistent_direction
# ---------------------------------------------------------------------------

class TestIsConsistentDirection:
    ba = BehaviorAnalyses()

    def test_horizontal_swimmer_excluded(self):
        bboxes = straight_line_bboxes(n=20, dx=15, dy=0)
        assert self.ba._is_consistent_direction(bboxes, "t1") is True

    def test_vertical_swimmer_excluded(self):
        bboxes = straight_line_bboxes(n=20, dx=0, dy=15)
        assert self.ba._is_consistent_direction(bboxes, "t2") is True

    def test_diagonal_swimmer_excluded(self):
        bboxes = straight_line_bboxes(n=20, dx=10, dy=10)
        assert self.ba._is_consistent_direction(bboxes, "t3") is True

    def test_zigzag_not_excluded(self):
        bboxes = zigzag_bboxes(n=20, amplitude=30)
        assert self.ba._is_consistent_direction(bboxes, "t4") is False

    def test_stationary_not_excluded(self):
        # Displacement below MIN_DISPLACEMENT_BBOX_FRACTION * bbox_size
        bboxes = stationary_bboxes(n=20)
        assert self.ba._is_consistent_direction(bboxes, "t5") is False

    def test_single_bbox_not_excluded(self):
        # len(raw_cx) < 2 early-exit path
        bboxes = straight_line_bboxes(n=1, dx=15)
        assert self.ba._is_consistent_direction(bboxes, "t6") is False

    def test_zigzag_then_straight_excluded(self):
        # First 6 frames zigzag, then 14 frames straight — net direction clear
        zigzag = zigzag_bboxes(n=6, amplitude=8)
        straight = straight_line_bboxes(n=14, dx=15, dy=0)
        # offset straight part to start from where zigzag ends
        last_cx = (zigzag[-1][0] + zigzag[-1][2]) / 2
        offset = last_cx - 200
        straight = [make_bbox(200 + offset + (i + 1) * 15, 200) for i in range(14)]
        bboxes = zigzag + straight
        assert self.ba._is_consistent_direction(bboxes, "t7") is True

    def test_nearly_straight_with_noise(self):
        # Straight path with small random-looking offsets; should still be excluded
        import numpy as np
        rng = np.random.default_rng(42)
        bboxes = []
        for i in range(20):
            cx = 200 + i * 12 + rng.uniform(-2, 2)
            cy = 200 + rng.uniform(-2, 2)
            bboxes.append(make_bbox(cx, cy))
        assert self.ba._is_consistent_direction(bboxes, "t8") is True

    def test_slow_mover_below_displacement_threshold_not_excluded(self):
        # Very small displacement — well below 0.33 * bbox_size
        bboxes = straight_line_bboxes(n=20, dx=0.5, dy=0)  # total ~9.5 px vs bbox_size ~56
        assert self.ba._is_consistent_direction(bboxes, "t9") is False

    # --- back-and-forth threshold boundary tests ---
    # DIRECTION_CONSISTENCY_THRESHOLD = 0.70
    # With n=40 bboxes (39 steps), raw forward-step fractions are:
    #   50% → 0 displacement → fails guard → False
    #   66% → 26/39 ≈ 0.667 < 0.70 → False
    #   75% → 30/39 ≈ 0.769 > 0.70 → True
    #   85% → 34/39 ≈ 0.872 > 0.70 → True

    def test_back_and_forth_50pct_not_excluded(self):
        # Person goes right then returns to near-start: displacement ≈ 0,
        # fails the minimum-displacement guard → not excluded.
        bboxes = back_and_forth_bboxes(n=40, forward_fraction=0.50)
        assert self.ba._is_consistent_direction(bboxes, "bf50") is False

    def test_back_and_forth_66pct_not_excluded(self):
        # 66 % forward, 34 % back: consistency ≈ 0.67 < threshold → not excluded.
        bboxes = back_and_forth_bboxes(n=40, forward_fraction=0.66)
        assert self.ba._is_consistent_direction(bboxes, "bf66") is False

    def test_back_and_forth_75pct_not_excluded(self):
        # 75 % forward then 25 % reversal: peak-reversal ratio ≈ -0.34 → candidate.
        bboxes = back_and_forth_bboxes(n=40, forward_fraction=0.75)
        assert self.ba._is_consistent_direction(bboxes, "bf75") is False

    def test_back_and_forth_85pct_excluded(self):
        # 85 % forward, 15 % back: consistency ≈ 0.87 > threshold → excluded.
        bboxes = back_and_forth_bboxes(n=40, forward_fraction=0.85)
        assert self.ba._is_consistent_direction(bboxes, "bf85") is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
