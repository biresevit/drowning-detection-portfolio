"""
Tests of the far-region feature of analyse_predictions_ui.py.

A far region is a polygon over the part of an image where people are only a few pixels; annotations under it
are meant to be removed and its pixels painted black. These tests cover, on synthetic scenes:

  * geometry          polygon area, point-in-polygon, clipping, "is this label covered" rule
  * proposal          propose_far_region groups small labels, follows a slanted horizon, respects the threshold
  * expand up         stretch to the image top, unless that would swallow clearly more larger labels
  * expand sideways   continue flat to both borders, stopping before the first clearly larger label
  * generate_regions  the whole pipeline (propose + expand + merge) - what Far Region [M] runs, for one image
                      or for all of them
  * merging           overlapping regions join, a union with a hole does not
  * sidecar file      far_regions_<split>.json round trip, version 1 (rectangles) still loads
  * UI                (needs a display) one image vs "apply to all images" give identical results, options,
                      replace / undo / clear, hand-drawn regions, remembered settings

Run:  python -m pytest src/human_tracking/tests/test_far_regions.py -q
"""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.human_tracking import analyse_predictions_ui as U  # noqa: E402

W, H = 1920, 1080
THR = 15.0
PURITY = 0.95


# ---------------------------------------------------------------------------------------------- helpers
def boxes_at(xs, ys, size):
    """Square boxes of the given sqrt(w*h) size centred on (xs, ys)."""
    xs, ys = np.atleast_1d(np.asarray(xs, float)), np.atleast_1d(np.asarray(ys, float))
    size = np.broadcast_to(np.asarray(size, float), xs.shape)
    return np.stack([xs - size / 2, ys - size / 2, xs + size / 2, ys + size / 2], 1).astype(np.float32)


def grid(x0, x1, y0, y1, nx, ny, size):
    gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    return boxes_at(gx.ravel(), gy.ravel(), size)


def sizes(b):
    return U.box_sizes(np.asarray(b, np.float32).reshape(-1, 4))


def covered(boxes, polys):
    return U.covered_by_any(np.asarray(boxes, np.float32).reshape(-1, 4), list(polys))


def reaches_top(poly):
    return float(np.min(poly[:, 1])) <= 0.5


def span(polys):
    xs = np.concatenate([p[:, 0] for p in polys])
    return float(xs.min()), float(xs.max())


@pytest.fixture
def horizon():
    """A pool seen from the side: far people (8 px) in a band well away from every border (the proposal itself
    snaps to a border it is within 20% of, which would hide what 'expand' does), near people (60 px) lower
    down. Returns (boxes, n_small)."""
    small = grid(600, 1300, 420, 540, 14, 4, 8)
    near = boxes_at(np.linspace(100, 1800, 12), np.linspace(700, 1000, 12), 60)
    return np.vstack([small, near]), len(small)


def gen(boxes, up=False, side=False, thr=THR, existing=(), **kw):
    return U.generate_regions(boxes, W, H, thr, PURITY, up, side, existing=existing, **kw)


# ---------------------------------------------------------------------------------------------- geometry
def test_polygon_area():
    assert U.poly_area(np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)) == 100
    assert U.poly_area(np.array([[0, 0], [10, 0], [0, 10]], float)) == 50
    assert U.poly_area(np.array([[0, 0], [10, 0]], float)) == 0.0


def test_point_in_polygon_even_odd_with_a_concave_notch():
    L = np.array([[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]], float)
    inside = U.points_in_polygon(np.array([2, 8, 8, -1, 5]), np.array([2, 2, 8, 2, 2]), L)
    assert list(inside) == [True, True, False, False, True]  # (8, 8) is in the notch


def test_clip_area():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert abs(U._clip_area(sq, 5, 5, 20, 20) - 25) < 1e-9
    assert U._clip_area(sq, 20, 20, 30, 30) == 0
    assert abs(U._clip_area(sq, 2, 2, 4, 4) - 4) < 1e-9


def test_covered_rule_centre_inside_or_half_the_area():
    tri = np.array([[0, 0], [100, 0], [0, 100]], float)
    b = np.array([[10, 10, 20, 20],      # centre inside
                  [60, 60, 80, 80],      # entirely outside
                  [95, 0, 105, 4]], float)  # mostly outside
    assert list(U.covered_by(b, tri)) == [True, False, False]
    assert not U.covered_by(b, tri[:2]).any(), "fewer than 3 corners removes nothing"
    assert len(U.covered_by_any(np.zeros((0, 4)), [tri])) == 0


def test_polygon_rule_equals_the_rectangle_rule_for_rectangles():
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, y = np.sort(rng.uniform(0, 400, 2)), np.sort(rng.uniform(0, 400, 2))
        rect = [x[0], y[0], x[1], y[1]]
        bx, by = np.sort(rng.uniform(0, 400, (30, 2)), axis=1), np.sort(rng.uniform(0, 400, (30, 2)), axis=1)
        boxes = np.stack([bx[:, 0], by[:, 0], bx[:, 1], by[:, 1]], 1)
        assert (U.covered_by(boxes, U.rect_to_poly(rect)) == U._rect_covered(boxes, rect)).all()


def test_extend_up_keeps_the_lower_edge_and_notches():
    tri = np.array([[400, 300], [600, 300], [500, 340]], float)
    up = U.extend_up(tri)
    assert up[:, 1].min() == 0 and abs(up[:, 0].min() - 400) < 1e-3 and abs(up[:, 0].max() - 600) < 1e-3
    inside = np.array([[480, 5, 520, 15], [450, 100, 460, 110]], float)
    outside = np.array([[700, 100, 710, 110], [410, 335, 420, 345]], float)  # beside / below the lower edge
    assert U.covered_by(inside, up).all() and not U.covered_by(outside, up).any()
    L = np.array([[100, 200], [300, 200], [300, 400], [200, 400], [200, 300], [100, 300]], float)
    uL = U.extend_up(L)  # a notch below the left arm stays open
    assert U.covered_by(np.array([[140, 250, 150, 260]], float), uL)[0]
    assert not U.covered_by(np.array([[140, 340, 150, 350]], float), uL)[0]
    assert U.covered_by(np.array([[240, 350, 250, 360]], float), uL)[0]


def test_grow_sideways_holds_the_end_heights_level():
    r = np.array([[500, 100], [800, 100], [800, 200], [500, 200]], float)
    g = U.grow_sideways(r, 100, 1500)
    assert abs(g[:, 0].min() - 100) < 1e-3 and abs(g[:, 0].max() - 1500) < 1e-3
    assert abs(g[:, 1].min() - 100) < 1e-3 and abs(g[:, 1].max() - 200) < 1e-3
    assert abs(U.poly_area(g) - 1400 * 100) < 1e-3
    same = U.grow_sideways(r, 500, 800)  # nothing to grow into
    assert abs(U.poly_area(same) - 300 * 100) < 1e-3
    slope = np.array([[500, 100], [800, 100], [800, 200], [500, 150]], float)  # lower edge slopes: ends held flat
    gs = U.grow_sideways(slope, 0, 1000)
    assert U.covered_by(np.array([[100, 120, 110, 130]], float), gs)[0]       # left of the shape: its left height
    assert not U.covered_by(np.array([[100, 180, 110, 190]], float), gs)[0]


# ---------------------------------------------------------------------------------------------- proposal
def test_proposal_covers_the_far_band_and_nothing_larger(horizon):
    boxes, n_small = horizon
    r = U.propose_far_region(boxes, W, H, THR, PURITY, U.FAR_MIN_SMALL)
    assert r is not None and r["n_small"] == n_small and r["n_bad"] == 0 and r["purity"] == 1.0
    poly = r["poly"]
    assert len(poly) >= 3 and poly[:, 0].min() >= 0 and poly[:, 0].max() <= W and poly[:, 1].max() <= H
    c = covered(boxes, [poly])
    assert c[:n_small].all() and not c[n_small:].any()


def test_proposal_needs_enough_small_labels():
    assert U.propose_far_region(boxes_at([100, 200], [100, 100], 8), W, H, THR) is None  # 2 < FAR_MIN_SMALL
    assert U.propose_far_region(np.zeros((0, 4)), W, H, THR) is None
    assert U.propose_far_region(boxes_at([100, 200, 300], [100] * 3, 8), W, H, THR) is not None


def test_proposal_threshold_is_strictly_below():
    s = boxes_at([300, 400, 500, 600], [100] * 4, 15.0)
    assert U.propose_far_region(s, W, H, thr=15.0) is None       # 15 is not < 15
    assert U.propose_far_region(s, W, H, thr=15.5) is not None


def test_threshold_decides_which_labels_are_far():
    mid = grid(500, 1400, 100, 200, 10, 3, 20)                    # 20 px people
    assert gen(mid, thr=15) == []                                 # not small at 15
    new = gen(mid, thr=25)
    assert new and covered(mid, new).all()                        # small at 25


def test_proposal_purity_keeps_larger_labels_out():
    band = grid(600, 1300, 100, 250, 14, 3, 8)
    stray = boxes_at([950], [180], 60)                            # one near person inside the band
    r = U.propose_far_region(np.vstack([band, stray]), W, H, THR, PURITY, U.FAR_MIN_SMALL)
    assert r["purity"] >= PURITY and r["n_bad"] <= max(1, 0.05 * (r["n_small"] + r["n_bad"]))


def test_proposal_follows_a_slanted_horizon():
    rng = np.random.default_rng(3)
    xs = rng.uniform(50, 1870, 90)
    ys = rng.uniform(0.05, 1, 90) * (150 + 0.25 * xs - 30) + 20  # above the line y = 150 + 0.25 x
    bx = rng.uniform(50, 1870, 25)
    near = boxes_at(bx, 150 + 0.25 * bx + 60 + rng.uniform(0, 500, 25), 70)
    boxes = np.vstack([boxes_at(xs, ys, 9), near])
    r = U.propose_far_region(boxes, W, H, THR, PURITY, U.FAR_MIN_SMALL)
    assert r and r["n_small"] >= 85 and r["n_bad"] <= 1
    c = covered(boxes, [r["poly"]])
    assert c[:90].sum() >= 85 and c[90:].sum() <= 1


def test_proposal_exclude_gives_the_next_cluster():
    c1 = grid(100, 400, 100, 300, 5, 4, 7)
    c2 = grid(1400, 1700, 700, 900, 4, 3, 7)
    boxes = np.vstack([c1, c2, boxes_at(np.linspace(600, 1200, 10), 600, 70)])
    a = U.propose_far_region(boxes, W, H, THR)
    b = U.propose_far_region(boxes, W, H, THR, exclude=U.covered_by(boxes, a["poly"]))
    n_small = len(c1) + len(c2)
    assert a["n_small"] + b["n_small"] == n_small and b["n_small"] > 0
    c = covered(boxes, [a["poly"], b["poly"]])
    assert c[:n_small].all() and c[n_small:].sum() <= 1          # purity allows a stray larger label (5%)
    assert U.propose_far_region(boxes, W, H, THR, exclude=U.covered_by_any(boxes, [a["poly"], b["poly"]])) is None


def test_proposal_is_deterministic(horizon):
    boxes, _ = horizon
    a = U.propose_far_region(boxes, W, H, THR)["poly"]
    b = U.propose_far_region(boxes, W, H, THR)["poly"]
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------------------------- expand up
def test_expand_up_reaches_the_top_and_keeps_the_same_labels(horizon):
    boxes, n_small = horizon
    flat = gen(boxes, up=False)
    up = gen(boxes, up=True)
    assert len(flat) == 1 and len(up) == 1
    assert not reaches_top(flat[0]), "the band does not touch the top, so without 'expand up' it must not"
    assert reaches_top(up[0])
    assert U.poly_area(up[0]) > U.poly_area(flat[0])
    for polys in (flat, up):
        c = covered(boxes, polys)
        assert c[:n_small].all() and not c[n_small:].any()


def test_region_with_larger_boxes_above_it_is_dropped_when_expanding_up(horizon):
    boxes, n_small = horizon
    wall = boxes_at(np.linspace(600, 1300, 8), [130, 150, 140, 160, 135, 155, 145, 150], 70)  # real-size people above
    above = np.vstack([boxes, wall])
    assert len(gen(above, up=False)) == 1, "without 'expand up' the small band is still a region"
    assert gen(above, up=True) == [], "it cannot reach the top without covering them: not a real far region"
    assert gen(above, up=True, side=True) == []


def test_only_the_part_that_can_reach_the_top_is_kept(horizon):
    boxes, n_small = horizon                                                   # band x 600-1300, small labels in 14 columns
    above = np.vstack([boxes, boxes_at([700, 800, 900], [130, 140, 135], 60)])  # people above the left part only
    new = gen(above, up=True)
    assert new and all(reaches_top(p) for p in new)
    c = covered(above, new)
    xs = (above[:, 0] + above[:, 2]) / 2
    band = np.arange(len(above)) < n_small
    assert c[band & (xs >= 950)].all(), "the part with nothing above it is a far region"
    assert not c[band & (xs >= 660) & (xs <= 940)].any(), "under the larger people it is not"
    assert not c[n_small:].any()


def test_slices_narrower_than_slice_min_are_not_kept():
    blocked = grid(1200, 1600, 420, 500, 8, 3, 7)
    gaps = boxes_at(np.linspace(1190, 1610, 6), [150, 200, 160, 210, 170, 190], 70)        # narrow gaps between larger people
    near = boxes_at(np.linspace(100, 1800, 12), np.linspace(750, 1000, 12), 60)
    boxes = np.vstack([blocked, gaps, near])
    assert gen(boxes, up=True) == []
    assert len(gen(boxes, up=False)) >= 1


def test_only_the_regions_that_cannot_reach_the_top_are_dropped():
    clean = grid(300, 700, 420, 500, 8, 3, 7)                                  # nothing above: kept
    blocked = grid(1200, 1600, 420, 500, 8, 3, 7)
    people_above = boxes_at(np.linspace(1190, 1610, 6), [150, 200, 160, 210, 170, 190], 70)  # real-size people above it
    near = boxes_at(np.linspace(100, 1800, 12), np.linspace(750, 1000, 12), 60)
    boxes = np.vstack([clean, blocked, people_above, near])
    kept = gen(boxes, up=True)
    assert len(kept) == 1 and reaches_top(kept[0])
    c = covered(boxes, kept)
    assert c[:len(clean)].all() and not c[len(clean):].any()
    plain = gen(boxes, up=False)
    assert len(plain) >= 1 and covered(boxes, plain)[len(clean):len(clean) + len(blocked)].any()  # kept when not expanding up


def test_dropped_regions_do_not_stop_other_regions_from_being_found():
    blocked = grid(1200, 1600, 420, 500, 8, 3, 7)
    people_above = boxes_at(np.linspace(1190, 1610, 6), [150, 200, 160, 210, 170, 190], 70)
    clean = grid(300, 700, 420, 500, 8, 3, 7)
    more = grid(800, 1000, 600, 660, 4, 3, 7)
    near = boxes_at(np.linspace(100, 1800, 12), np.linspace(750, 1000, 12), 60)
    boxes = np.vstack([blocked, people_above, clean, more, near])             # the blocked group is proposed first
    kept = gen(boxes, up=True)
    c = covered(boxes, kept)
    start = len(blocked) + len(people_above)
    assert c[start:start + len(clean)].all()
    assert not c[:len(blocked)].any()


def test_expand_up_tolerates_a_single_stray_label(horizon):
    boxes, _ = horizon
    stray = np.vstack([boxes, boxes_at([900], [140], 60)])
    up = gen(stray, up=True)
    assert len(up) == 1 and reaches_top(up[0])                                # one label is noise, not a veto


# ---------------------------------------------------------------------------------------------- expand sideways
def test_expand_sideways_reaches_both_borders_when_nothing_blocks(horizon):
    boxes, n_small = horizon
    none = gen(boxes, up=False, side=False)
    side = gen(boxes, up=False, side=True)
    x0, x1 = span(none)
    assert x0 > 100 and x1 < W - 100, "the cluster does not touch the borders without 'expand sideways'"
    assert len(side) == 1
    sx0, sx1 = span(side)
    assert sx0 <= 0.5 and sx1 >= W - 0.5
    c = covered(boxes, side)
    assert c[:n_small].all()


def test_expand_sideways_stops_before_a_clearly_larger_label():
    band = grid(700, 1300, 420, 540, 12, 4, 8)
    blocker_left = boxes_at([300], [480], 60)       # 60 px person in the rows of the band, left of it
    boxes = np.vstack([band, blocker_left, boxes_at(np.linspace(100, 1800, 12), np.linspace(700, 1000, 12), 60)])
    side = gen(boxes, side=True)
    assert len(side) == 1
    x0, x1 = span(side)
    assert x1 >= W - 0.5, "the right side is free"
    assert not covered(blocker_left, side)[0], "the 60 px person must not be taken along"
    assert x0 > 0.5, "the left side must stop"


def test_expand_sideways_is_not_stopped_by_borderline_labels():
    # 20 px people are not "small" at 15 but are below SIDE_BLOCK x 15 = 22.5: they do not block the growth
    band = grid(700, 1300, 420, 540, 12, 4, 8)
    border = boxes_at([300, 400, 1600], [480, 450, 500], 20)
    boxes = np.vstack([band, border, boxes_at(np.linspace(100, 1800, 12), np.linspace(700, 1000, 12), 60)])
    side = gen(boxes, side=True)
    x0, x1 = span(side)
    assert x0 <= 0.5 and x1 >= W - 0.5


def test_sideways_block_factor_is_relative_to_the_threshold():
    band = grid(700, 1300, 420, 540, 12, 4, 8)
    p = boxes_at([300], [480], 26)                                            # 26 px: blocks at 1.5 x 15, not at 2 x 15
    boxes = np.vstack([band, p, boxes_at(np.linspace(100, 1800, 12), np.linspace(700, 1000, 12), 60)])
    poly = U.propose_far_region(boxes, W, H, THR)["poly"]
    (tight,) = U.stretch_region(boxes, W, H, poly, THR, up=False, side=True, block=1.5)
    (loose,) = U.stretch_region(boxes, W, H, poly, THR, up=False, side=True, block=2.0)
    assert span([tight])[0] > 0.5 and not covered(p, [tight])[0]
    assert span([loose])[0] <= 0.5 and covered(p, [loose])[0]


def test_sideways_without_any_labels_still_spans_the_image():
    r = np.array([[500, 100], [800, 100], [800, 200], [500, 200]], float)
    out = U.stretch_region(np.zeros((0, 4)), W, H, r, THR, up=False, side=True)
    assert len(out) == 1 and span(out) == (0.0, float(W))


def test_up_and_sideways_together_make_a_full_width_strip_to_the_top(horizon):
    boxes, n_small = horizon
    new = gen(boxes, up=True, side=True)
    assert len(new) == 1
    x0, x1 = span(new)
    assert reaches_top(new[0]) and x0 <= 0.5 and x1 >= W - 0.5
    c = covered(boxes, new)
    assert c[:n_small].all() and not c[n_small:].any()


# ---------------------------------------------------------------------------------------------- properties
def random_scene(seed):
    g = np.random.default_rng(seed)
    far = boxes_at(g.uniform(200, 1700, 70), g.uniform(60, 330, 70), g.uniform(5, 13, 70))
    mid = boxes_at(g.uniform(0, W, 25), g.uniform(250, 700, 25), g.uniform(16, 40, 25))
    near = boxes_at(g.uniform(0, W, 25), g.uniform(500, H, 25), g.uniform(45, 120, 25))
    return np.vstack([far, mid, near])


@pytest.mark.parametrize("seed", range(12))
def test_options_never_lose_small_labels_and_never_take_clearly_larger_ones(seed):
    boxes = random_scene(seed)
    base = gen(boxes, up=False, side=False)
    up = gen(boxes, up=True, side=False)
    both = gen(boxes, up=True, side=True)
    small = sizes(boxes) < THR
    block = THR * U.SIDE_BLOCK
    clearly = sizes(boxes) >= block
    n = lambda polys: int((covered(boxes, polys) & small).sum()) if polys else 0
    assert n(up) <= n(base) + 2, "expanding up never finds more small labels than not expanding (it only drops)"
    for polys in (up, both):
        assert all(reaches_top(p) for p in polys), "with 'expand up' every region that is kept reaches the top"
        if polys:  # what is kept contains next to no larger label
            cov = covered(boxes, polys)
            assert int((cov & (sizes(boxes) >= THR)).sum()) <= 0.1 * int(cov.sum()) + 2
    # growing sideways may add labels below `block`, never clearly larger ones beyond what up already had
    extra = int((covered(boxes, both) & clearly).sum()) - int((covered(boxes, up) & clearly).sum()) if both else 0
    assert extra <= 3
    for polys in (base, up, both):
        for p in polys:
            assert p[:, 0].min() >= -1e-3 and p[:, 0].max() <= W + 1e-3 and p[:, 1].min() >= -1e-3 and p[:, 1].max() <= H + 1e-3
            assert U.poly_area(p) >= U.MIN_FAR_AREA


@pytest.mark.parametrize("seed", range(6))
def test_generation_is_deterministic_and_idempotent(seed):
    boxes = random_scene(seed)
    a = gen(boxes, up=True, side=True)
    b = gen(boxes, up=True, side=True)
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))
    # everything the regions can take is taken: generating again among what is left finds nothing (or little)
    again = gen(boxes, up=True, side=True, existing=a)
    left = int((covered(boxes, again) & ~covered(boxes, a) & (sizes(boxes) < THR)).sum()) if again else 0
    assert left <= 3


def test_existing_regions_are_not_proposed_again(horizon):
    boxes, _ = horizon
    first = gen(boxes, up=True)
    assert first
    assert gen(boxes, up=True, existing=first) == []


# ---------------------------------------------------------------------------------------------- merging
def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def test_merge_joins_overlapping_regions():
    a, b, c = rect(100, 0, 300, 100), rect(250, 0, 500, 120), rect(900, 0, 1000, 50)
    out = U.merge_polys([a, b, c], W, H)
    assert len(out) == 2
    assert abs(sum(U.poly_area(p) for p in out) - (200 * 100 + 250 * 120 - 50 * 100 + 100 * 50)) < 800
    assert len(U.merge_polys([a], W, H)) == 1 and U.merge_polys([], W, H) == []


def test_merge_joins_regions_a_few_pixels_apart():
    out = U.merge_polys([rect(0, 0, 400, 100), rect(402, 0, 800, 100)], W, H)
    assert len(out) == 1


def test_merge_never_makes_a_hole():
    ring = [rect(100, 100, 400, 140), rect(100, 360, 400, 400), rect(100, 100, 140, 400), rect(360, 100, 400, 400)]
    out = U.merge_polys(ring, W, H)
    inner = np.array([[250, 250, 260, 260]], np.float32)
    assert not U.covered_by_any(inner, out).any(), "the pocket in the middle must stay open"


def test_merge_respects_the_accept_callback():
    a, b = rect(100, 0, 300, 100), rect(250, 0, 500, 120)
    assert len(U.merge_polys([a, b], W, H, accept=lambda u, p, q: False)) == 2
    assert len(U.merge_polys([a, b], W, H, accept=lambda u, p, q: True)) == 1


def test_generation_merges_bands_that_overlap_after_expanding():
    a = grid(500, 800, 380, 440, 6, 3, 7)
    b = grid(1100, 1400, 520, 580, 6, 3, 7)
    wall = boxes_at([880, 940, 1000, 1060], 480, 60)   # between the groups, in the gap between their rows
    near = boxes_at(np.linspace(100, 1800, 12), np.linspace(750, 1000, 12), 60)
    boxes = np.vstack([a, b, wall, near])
    apart = gen(boxes, up=False, side=False)
    joined = gen(boxes, up=True, side=True)
    assert len(apart) == 2
    assert len(joined) == 1, "after expanding, the two strips overlap and are one region"
    c = covered(boxes, joined)
    assert c[:len(a) + len(b)].all()
    assert c[len(a) + len(b):].sum() <= 2, "the people between the groups are at most partly taken"


# ---------------------------------------------------------------------------------------------- counts, sidecar
def test_region_counts():
    boxes = np.vstack([boxes_at([100, 120], [100, 100], 8), boxes_at([140], [100], 40), boxes_at([900], [900], 8)])
    poly = rect(50, 50, 200, 200)
    assert U.region_counts(boxes, [poly], THR) == (2, 1)
    assert U.region_counts(boxes, [poly], THR, taken=[poly]) == (0, 0)
    assert U.region_counts(np.zeros((0, 4)), [poly], THR) == (0, 0) and U.region_counts(boxes, [], THR) == (0, 0)


def test_sidecar_roundtrip_and_legacy_rectangles(tmp_path):
    p = tmp_path / "far_regions_test.json"
    item = SimpleNamespace(path=Path("a.png"), far=[np.array([[1.04, 2.0], [30.0, 4.0], [10, 40]], np.float32)])
    U.save_far(p, [item])
    d = json.loads(p.read_text())
    assert d["version"] == 2 and d["images"]["a.png"] == [[[1.0, 2.0], [30.0, 4.0], [10.0, 40.0]]]
    assert np.allclose(U.load_far(p)["a.png"][0], [[1, 2], [30, 4], [10, 40]], atol=0.06)
    p.write_text(json.dumps({"version": 1, "images": {"r.png": [[10, 20, 300, 400]]}}))
    got = U.load_far(p)["r.png"][0]
    assert got.shape == (4, 2) and np.allclose(got, U.rect_to_poly([10, 20, 300, 400]))


def test_sidecar_skips_bad_entries_and_raises_on_corrupt(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"version": 2, "images": {
        "a.png": [[[10, 10], [200, 10], [10, 150]], [[0, 0], [5, 5]], "x", [[1, 2], ["u", 3], [4, 4]], [[0, 0]] * 3],
        "b.png": "nope"}}))
    got = U.load_far(p)
    assert list(got) == ["a.png"] and len(got["a.png"]) == 1
    p.write_text("{ not json")
    with pytest.raises(ValueError):
        U.load_far(p)
    assert U.load_far(tmp_path / "missing.json") == {}
    none = tmp_path / "none.json"
    U.save_far(none, [SimpleNamespace(path=Path("x.png"), far=[])])
    assert not none.exists()


def test_darken_paints_only_inside_the_polygon():
    img = Image.new("RGB", (100, 100), (200, 200, 200))
    U.darken(img, [np.array([[60, 60], [100, 60], [60, 100]], float)], 50, 50)   # crop starts at (50, 50)
    assert img.getpixel((12, 12)) == (50, 50, 50)
    assert img.getpixel((45, 45)) == (200, 200, 200) and img.getpixel((5, 5)) == (200, 200, 200)


# ---------------------------------------------------------------------------------------------- UI (needs a display)
tk = pytest.importorskip("tkinter")
try:
    _probe = tk.Tk()
    _probe.destroy()
except tk.TclError:  # pragma: no cover
    pytest.skip("no display for Tk", allow_module_level=True)

EDGES = [float(v) for v in U.DEFAULT_EDGES.split(",")]


def new_root(tries=5):
    """tk.Tk() with retries: on Windows, opening windows in quick succession occasionally fails with
    'Can't find a usable init.tcl' (the file is briefly locked)."""
    import time
    for k in range(tries):
        try:
            return tk.Tk()
        except tk.TclError:
            if k == tries - 1:
                raise
            time.sleep(0.3)


def label_rows(boxes):
    out = []
    for b in boxes:
        cx, cy, w, h = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H
        out.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\r\n".join(out) + "\r\n"


def build_dataset(root, scenes):
    shutil.rmtree(root, ignore_errors=True)
    for sub in ("images", "labels"):
        (root / "test" / sub).mkdir(parents=True)
    grad = np.tile(np.linspace(40, 220, W, dtype=np.uint8), (H, 1))
    for name, boxes in scenes.items():
        Image.fromarray(np.stack([grad, grad[::-1], grad], 2)).save(root / "test" / "images" / f"{name}.png")
        (root / "test" / "labels" / f"{name}.txt").write_bytes(label_rows(boxes).encode())


@pytest.fixture
def scenes(horizon):
    boxes, _ = horizon
    two = np.vstack([grid(100, 400, 100, 300, 5, 4, 7), grid(1400, 1700, 700, 900, 4, 3, 7),
                     boxes_at(np.linspace(600, 1200, 10), 600, 70)])
    return {"a": boxes, "b": boxes_at([300, 900], [500, 600], 80), "c": two}


@pytest.fixture
def ui(tmp_path, scenes, monkeypatch):
    from tkinter import messagebox
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: True)
    monkeypatch.setattr(messagebox, "showwarning", lambda *a, **k: None)
    monkeypatch.setattr(messagebox, "showerror", lambda *a, **k: None)
    made = []

    def make(settings=None, settings_path=None, raw=False, extra=()):
        for r in made:  # one Tk root at a time: images (PhotoImage) belong to the first root created
            try:
                r.destroy()
            except tk.TclError:
                pass
        made.clear()
        build_dataset(tmp_path / "ds", scenes)
        items = U.load_items(tmp_path / "ds" / "test")
        root = new_root()
        root.geometry("1500x900+20+20")
        app = U.App(root, U.parse_args(["--input_dir", str(tmp_path / "ds"), *extra]), items, EDGES,
                    settings=settings, settings_path=settings_path)
        app.autosave.set(False)
        root.update()
        made.append(root)
        return app

    yield make
    for r in made:
        try:
            r.destroy()
        except tk.TclError:
            pass


def goto(app, stem):
    app.cb_size.current(0)
    app.mode_var.set("All")
    app._refilter(keep=False)
    app.pos = app.visible.index([i.path.stem for i in app.items].index(stem))
    app._new_image()
    app.root.update()
    return app._cur()


def item(app, stem):
    return [i for i in app.items if i.path.stem == stem][0]


@pytest.mark.parametrize("up,side", [(False, False), (True, False), (False, True), (True, True)])
def test_ui_one_image_equals_apply_to_all(ui, up, side):
    app1 = ui()
    app1.up_var.set(up); app1.side_var.set(side); app1.all_var.set(False)
    for stem in ("a", "c"):
        goto(app1, stem)
        app1.far_region()
    single = {stem: [p.copy() for p in item(app1, stem).far] for stem in ("a", "b", "c")}
    app2 = ui()
    app2.up_var.set(up); app2.side_var.set(side); app2.all_var.set(True)
    goto(app2, "b")
    app2.far_region()
    assert single["a"] and single["c"] and single["b"] == []
    for stem in ("a", "b", "c"):
        x, y = single[stem], item(app2, stem).far
        assert len(x) == len(y) and all(np.array_equal(p, q) for p, q in zip(x, y)), (stem, up, side)


def test_ui_options_change_the_result(ui):
    app = ui()
    it = goto(app, "a")
    app.up_var.set(False); app.side_var.set(False); app.far_region()
    plain = [p.copy() for p in it.far]
    assert not reaches_top(plain[0]) and span(plain)[0] > 100
    app.up_var.set(True); app.far_region()
    assert reaches_top(it.far[0]) and span(it.far)[0] > 100
    app.side_var.set(True); app.far_region()
    assert reaches_top(it.far[0]) and span(it.far) == (0.0, float(W))


def test_ui_far_region_replaces_and_undo_restores(ui):
    app = ui()
    it = goto(app, "a")
    app.up_var.set(True); app.side_var.set(False)
    app.far_region()
    first = [p.copy() for p in it.far]
    app.far_region()                                                   # again: replaced by the same, not doubled
    assert len(it.far) == len(first) and all(np.array_equal(p, q) for p, q in zip(first, it.far))
    app.side_var.set(True); app.far_region()
    assert span(it.far)[1] >= W - 0.5
    app.undo()
    assert all(np.array_equal(p, q) for p, q in zip(first, it.far))
    app.far_var.set("5"); app.far_region()                              # nothing found: the old regions stay
    assert all(np.array_equal(p, q) for p, q in zip(first, it.far)) and "No far region found" in app.status.cget("text")


def test_ui_threshold_box_default_and_use(ui):
    app = ui()
    assert app.far_var.get() == "15" and U.FAR_PX == 15.0
    it = goto(app, "a")
    app.up_var.set(False); app.side_var.set(False)
    app.far_var.set("5"); app.far_region()
    assert it.far == []
    app.far_var.set("15"); app.far_region()
    assert len(it.far) == 1


def test_ui_apply_to_all_and_clear_all(ui):
    app = ui()
    app.up_var.set(True); app.side_var.set(False); app.all_var.set(True)
    goto(app, "b")
    app.far_region()
    assert [i.path.stem for i in app.items if i.far] == ["a", "c"]       # b has no small labels
    assert all(i.far_dirty for i in app.items if i.far) and not any(i.dirty for i in app.items)
    assert not (app.far_path).exists(), "nothing is written until Save"
    app.clear_far()
    assert all(i.far == [] for i in app.items)
    app.all_var.set(False)
    it = goto(app, "a")
    it.far = []
    app.far_region(); assert it.far
    app.clear_far(); assert it.far == []


def test_ui_save_writes_the_sidecar_not_the_labels(ui):
    app = ui()
    labels = {f.name: f.read_bytes() for f in (app.split_dir / "labels").glob("*.txt")}
    app.all_var.set(True); app.up_var.set(True); app.side_var.set(True)
    goto(app, "a"); app.far_region(); app.save()
    d = json.loads(app.far_path.read_text())["images"]
    assert set(d) == {"a.png", "c.png"} and d["a.png"]
    assert labels == {f.name: f.read_bytes() for f in (app.split_dir / "labels").glob("*.txt")}
    assert not (app.split_dir.parent / "_label_backups").exists()
    app2 = ui()                                                          # the file is read back on the next start
    # (ui() rebuilds the dataset, so write the sidecar again to check the reload)
    app2.far_path.write_text(json.dumps({"version": 2, "images": d}))
    items = U.load_items(app2.split_dir)
    for it in items:
        it.far = U.load_far(app2.far_path).get(it.path.name, [])
    assert [i.path.stem for i in items if i.far] == ["a", "c"]


def test_ui_hand_drawn_region_is_used_exactly_as_drawn(ui):
    app = ui()
    app.up_var.set(True); app.side_var.set(True)
    it = goto(app, "b")
    app.toggle_draw_far()
    pts = [(400, 400), (600, 400), (500, 450)]
    for q in pts:
        app._gt_press(SimpleNamespace(x=q[0], y=q[1], widget=app.cv_gt, delta=0))
        app._gt_release(SimpleNamespace(x=q[0], y=q[1], widget=app.cv_gt, delta=0))
    app.far_draft_finish()
    want = np.array([app._to_img(*q) for q in pts])
    assert len(it.far) == 1 and np.allclose(it.far[0], want, atol=0.01)


def test_ui_settings_are_remembered_and_the_old_threshold_is_ignored(ui, tmp_path):
    path = tmp_path / "settings.json"
    app = ui(settings={}, settings_path=path)
    assert app.up_var.get() and not app.side_var.get() and not app.all_var.get()
    app.side_var.set(True); app.up_var.set(False); app.far_var.set("12"); app._persist()
    d = json.loads(path.read_text())
    assert d["far_small_px"] == 12 and d["far_up"] is False and d["far_side"] is True and "far_all" not in d
    app.root.destroy()
    app = ui(settings=U.load_settings(path))
    assert app.far_var.get() == "12" and not app.up_var.get() and app.side_var.get() and not app.all_var.get()
    app.root.destroy()
    app = ui(settings={"far_px": 20})                                    # the pre-polygon key is not used any more
    assert app.far_var.get() == "15"
    assert "far_small_px" not in U.resolve_settings({"far_small_px": 33, "conf": 0.3}, ["--far_px", "25"])
