"""
End-to-end evaluation: runs the full drowning detection pipeline on every video
in --input_dir, checks detected drowning bboxes against ground-truth annotations,
and reports per-class and overall statistics.

Structure expected:
  input_dir/
    drowning/<sub>/<video>.mp4
    normal/<sub>/<video>.mp4
  annotations_dir/
    {video_stem}_drowning_0_{uuid}.txt   # YOLO: class cx cy w h  (normalized)
    {video_stem}.txt                     # human bboxes (optional)

Usage:
  python src/end_to_end_tests.py
  python src/end_to_end_tests.py --model_extension .engine
"""

import argparse
import glob
import os
import collections
import queue
import random
import shutil
import sys
import threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from action_recognition.processing import processing_thread
from classes_enum import ClassesEnum
from human_tracking.head_visibility_classes import NOT_DETECTED, HeadVisibilityClass as HVC
from utc_logging import get_logger
from utils_lib.draw_helpers import draw_results

logger = get_logger(
    log_level="info",
    filename="e2e_test_log.txt",
    log_type=0,
    max_log_size_mb=10,
    num_log_backup_files=3,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
OVERLAP_THRESHOLD = 0.1  # overlap threshold to count a detection as hitting the annotated person
# A person detection with at least this much of ITS OWN area inside an annotation box is
# taken to be the annotated person, however large that box is (see track_on_annotation).
ANNOTATION_INSIDE_FRAC = 0.5

# Drowning sub-type, decided from how much of the time the annotated person's head was
# classified under water. Passive drowning (unconscious, floating) is almost always
# "under"; active drowning still breaks the surface to breathe.
PASSIVE_UNDER_FRAC = 0.90       # >= this share of frames "under" -> passive
DROWNING_TYPE_MIN_FRAMES = 10   # below this there is not enough evidence to call it
DROWNING_TYPES = ["passive", "active", "unknown"]


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def load_drowning_bboxes(annotations_dir: str, video_stem: str) -> list[tuple]:
    """
    Returns list of (cx, cy, w, h) normalized bboxes.

    Tries two naming conventions:
      1. {stem}.txt          – split-dataset style (one YOLO line per person)
      2. {stem}_drowning_0_*.txt – per-person file style (one bbox per file)
    """
    def _parse_file(path):
        bboxes = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cx, cy, w, h = map(float, parts[1:5])
                    bboxes.append((cx, cy, w, h))
        return bboxes

    # Style 1: plain {stem}.txt
    simple = os.path.join(annotations_dir, f"{video_stem}.txt")
    if os.path.isfile(simple):
        return _parse_file(simple)

    # Style 2: {stem}_drowning_0_{uuid}.txt
    bboxes = []
    for ann_file in glob.glob(os.path.join(annotations_dir, f"{video_stem}_drowning_0_*.txt")):
        bboxes.extend(_parse_file(ann_file))
    return bboxes


def annotation_files(annotations_dir: str, video_stem: str) -> list[str]:
    """The annotation file(s) load_drowning_bboxes read — recorded so the CSV points at
    the file that has to be extended with the active/passive label."""
    simple = os.path.join(annotations_dir, f"{video_stem}.txt")
    if os.path.isfile(simple):
        return [simple]
    return sorted(glob.glob(os.path.join(annotations_dir, f"{video_stem}_drowning_0_*.txt")))


def yolo_to_abs(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def bbox_overlap_parts(detected, gt):
    """Returns (iou, gt_coverage, detected_coverage) — the three ways two boxes can agree.

    gt_coverage = intersection/area_gt: the prediction is much LARGER than the GT and
    contains it (a cluster bbox over the whole pool area).
    detected_coverage = intersection/area_detected: the prediction is much SMALLER than
    the GT and sits inside it (one person box inside a wide annotation box).
    """
    ix1, iy1 = max(detected[0], gt[0]), max(detected[1], gt[1])
    ix2, iy2 = min(detected[2], gt[2]), min(detected[3], gt[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0, 0.0, 0.0
    area_detected = (detected[2] - detected[0]) * (detected[3] - detected[1])
    area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
    union = area_detected + area_gt - inter
    return (inter / union if union > 0 else 0.0,
            inter / area_gt if area_gt > 0 else 0.0,
            inter / area_detected if area_detected > 0 else 0.0)


def detection_overlap(detected, gt):
    """
    Returns max(IoU, gt_coverage) where gt_coverage = intersection/area_gt.

    gt_coverage handles the case where the prediction bbox is much larger than
    the GT (e.g. a cluster bbox covering the whole pool area) but still fully
    contains the annotated drowning person — IoU would be near zero in that case
    even though the person is correctly located inside the detection.
    """
    iou, gt_coverage, _ = bbox_overlap_parts(detected, gt)
    return max(iou, gt_coverage)


def track_on_annotation(bbox, ann_abs):
    """Is this person detection the annotated person? Returns (matched, rank_key).

    Qualifies either by the usual overlap rule, or by sitting (mostly) INSIDE an
    annotation box. The second case is what the overlap rule alone misses: annotation
    boxes are often far larger than a single person detection — a wide box drawn around
    a floating body measured 1024x302 px against a 186x116 px detection, 14x the area.
    A detection fully inside it scores IoU 0.07 and gt_coverage 0.07, under the 0.1 bar,
    so the annotated drowning person was skipped in most frames.

    rank_key sorts by how much of the detection is inside the box first, IoU second, so
    that when several people share one box the best-framed one is taken as the annotated
    person rather than whichever has the largest raw intersection.
    """
    best = (0.0, 0.0, 0.0)
    for g in ann_abs:
        parts = bbox_overlap_parts(bbox, g)
        if (parts[2], parts[0]) > (best[2], best[0]):
            best = parts
    iou, gt_cov, det_cov = best
    matched = max(iou, gt_cov) >= OVERLAP_THRESHOLD or det_cov >= ANNOTATION_INSIDE_FRAC
    return matched, (det_cov, iou)


def bbox_iou(a, b):
    """Plain IoU — used to pick WHICH track is the annotated person."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1])) + ((b[2] - b[0]) * (b[3] - b[1])) - inter
    return inter / union if union > 0 else 0.0


def new_behavior_acc():
    return {"samples": 0, "with_track": 0, "with_candidate": 0,
            # frames where a track sat in the annotation box but behaviour analysis had
            # not classified that id yet (new id after a re-surface) — not a discard
            "with_track_unclassified": 0, "reasons": {},
            # which track ids were counted as "the annotated person", and how each
            # fared. The annotation is a STATIC box while people move, so without this
            # a low candidate% cannot be told apart from a passing swimmer drifting
            # through the box while the annotated person is elsewhere.
            "per_track": {},
            # Frames in which a CLUSTER covered the annotated person. Clusters are what
            # the action model is actually handed, and a person can sit inside a cluster
            # that somebody else anchored, so this is the real reachability question -
            # the candidate vote only explains why.
            "with_cluster": 0, "cluster_frames": 0,
            # head-visibility evidence on the annotated person, one entry per
            # (behaviour window, track) so overlapping windows are not double counted
            "head_windows": {}, "head_ambiguous": 0}


def head_class_counts(stats):
    """Per-class frame counts from one head_stats record (see BehaviorAnalyses._head_excluded)."""
    counts = {"n": 0, "under": 0, "hs": 0, "clear": 0, "missing": 0}
    for entry in stats.get("predictions") or []:
        if not entry:
            continue
        cls = entry[0]
        counts["n"] += 1
        if cls == HVC.HEAD_HORIZONTAL_OR_UNDER_WATER.value:
            counts["under"] += 1
        elif cls == HVC.HEAD_AND_SHOULDERS_VISIBLE.value:
            counts["hs"] += 1
        elif cls == HVC.HEAD_CLEARLY_VISIBLE.value:
            counts["clear"] += 1
        elif cls == NOT_DETECTED:
            counts["missing"] += 1
    return counts


def sample_head_evidence(human_tracker, inside_ranked, acc):
    """Record the annotated person's head-visibility evidence, once per behaviour window.

    inside_ranked is [(rank_key, track_id)] for the tracks in the annotation box, ranked
    by track_on_annotation. When several people share the box the best-ranked one is
    taken as the annotated person — mixing a passing swimmer's frames in would wreck the
    under-water fraction.
    """
    # one load: (run_id, stats) is published as a single tuple so the id and the dict
    # always belong to the same behaviour window
    run_id, stats_by_track = getattr(human_tracker, "behavior_head_snapshot", (0, {}))
    if not stats_by_track:
        return
    _, track_id = max(inside_ranked)
    key = (run_id, track_id)
    if key in acc["head_windows"]:
        return                          # this window already counted
    stats = stats_by_track.get(track_id)
    if not stats or not stats.get("checked"):
        return
    acc["head_windows"][key] = head_class_counts(stats)
    if len(inside_ranked) > 1:
        acc["head_ambiguous"] += 1


def summarize_head_evidence(acc, passive_under_frac=PASSIVE_UNDER_FRAC,
                            min_frames=DROWNING_TYPE_MIN_FRAMES):
    """Aggregate the per-window head evidence into an active/passive verdict.

    under_frac counts only frames where the person was actually detected; frames with
    no detection are reported separately as under_or_missing_frac, since a missed
    detection in a water zone often means submerged but is not proof of it.
    """
    tot = {"n": 0, "under": 0, "hs": 0, "clear": 0, "missing": 0}
    for counts in acc.get("head_windows", {}).values():
        for k in tot:
            tot[k] += counts[k]
    detected = tot["n"] - tot["missing"]
    out = {
        "n_frames": tot["n"], "n_detected": detected,
        "n_windows": len(acc.get("head_windows", {})),
        "n_tracks": len({tid for _, tid in acc.get("head_windows", {})}),
        "ambiguous_windows": acc.get("head_ambiguous", 0),
        "under_frac": (tot["under"] / detected) if detected else None,
        "under_or_missing_frac": ((tot["under"] + tot["missing"]) / tot["n"]) if tot["n"] else None,
        "hs_frac": (tot["hs"] / detected) if detected else None,
        "clear_frac": (tot["clear"] / detected) if detected else None,
    }
    if detected < min_frames or out["under_frac"] is None:
        out["drowning_type"] = "unknown"
    elif out["under_frac"] >= passive_under_frac:
        out["drowning_type"] = "passive"
    else:
        out["drowning_type"] = "active"
    return out


def sample_behavior_on_annotation(human_tracker, ann_bboxes_norm, img_w, img_h, acc):
    """Record whether the annotated drowning person still has a drowning candidate.

    One annotation box can contain several people (overlapping swimmers), so the
    question is not "was every track kept" but "did AT LEAST ONE track inside the
    box survive behaviour analysis". If none did, the drowning person cannot reach
    the X3D model at all, however good that model is.

    A track the behaviour categories say nothing about is NOT a discarded track. The
    categories are rebuilt once per behaviour window (~6 s), while a person who submerges
    and resurfaces comes back with a fresh track id, so between windows the live id is
    often in no set at all. Counting that as "not a candidate" reported people as
    discarded who were never judged — those frames go to with_track_unclassified instead.

    acc is mutated in place: samples / with_track / with_candidate /
    with_track_unclassified, plus a tally of the exclusion reason when every classified
    track in the box was discarded.
    """
    if not ann_bboxes_norm:
        return
    try:
        detections = human_tracker.tracker.get_last_frame_tracking()
        if not detections:
            return
        cats = human_tracker.behavior_categories or {}
        if not cats or not any(cats.values()):
            return                      # behaviour analysis has not run yet this video
        # keyed by category NAME: the enum can be imported under two module paths
        # (human_tracking.behavior_analysis vs behavior_analysis), and then the same
        # member is two distinct objects that neither match nor hash alike
        by_name = {cat.name: ids for cat, ids in cats.items()}
        candidates = by_name.get("CANDIDATE", set())

        # Judge PEOPLE, not track ids. One person is often several stitched fragments,
        # analysed separately, so they can disagree: on the annotated swimmer of
        # 20260622_150454, fragment 46 was EXCLUDED_HEAD_VISIBLE in two windows while
        # fragment 17 was a CANDIDATE in those same windows. Clustering keeps a person
        # when ANY fragment survives, so counting per track reported a discard where the
        # pipeline had in fact kept them.
        canonical = getattr(human_tracker, "_canonical_track", None)

        def person_of(track_id):
            return canonical(track_id) if canonical else track_id

        candidate_people = {person_of(t) for t in candidates}

        ann_abs = [yolo_to_abs(cx, cy, w, h, img_w, img_h) for cx, cy, w, h in ann_bboxes_norm]
        inside, inside_candidates, inside_ranked = [], 0, []
        for track_id, obj in detections.items():
            bbox = obj.get("bbox")
            if bbox is None:
                continue
            matched, rank = track_on_annotation(bbox, ann_abs)
            if not matched:
                continue
            inside.append(track_id)
            inside_ranked.append((rank, track_id))
            if person_of(track_id) in candidate_people:
                inside_candidates += 1

        acc["samples"] += 1
        if not inside:
            return
        sample_head_evidence(human_tracker, inside_ranked, acc)

        classified = set().union(*by_name.values())
        classified_people = {person_of(t) for t in classified}
        if not any(person_of(t) in classified_people for t in inside):
            # behaviour analysis has never seen these ids — no verdict either way
            acc["with_track_unclassified"] += 1
            return

        # was the annotated person inside a cluster this frame?
        clusters = getattr(human_tracker, "merged_clusters", None) or {}
        if clusters:
            acc["cluster_frames"] += 1
            best_box = None
            best_rank = None
            for rank, tid in inside_ranked:
                if best_rank is None or rank > best_rank:
                    best_rank, best_box = rank, detections[tid].get("bbox")
            if best_box is not None:
                for c in clusters.values():
                    cb = c.get("bbox")
                    if cb and bbox_overlap_parts(best_box, cb)[2] >= ANNOTATION_INSIDE_FRAC:
                        acc["with_cluster"] += 1
                        break

        acc["with_track"] += 1
        counted = set()
        for _, tid in inside_ranked:
            key = person_of(tid)
            if key not in classified_people or key in counted:
                continue
            counted.add(key)
            rec = acc["per_track"].setdefault(key, {"frames": 0, "candidate": 0, "reasons": {}})
            rec["frames"] += 1
            if key in candidate_people:
                rec["candidate"] += 1
            else:
                for name, ids in by_name.items():
                    if name != "CANDIDATE" and any(person_of(i) == key for i in ids):
                        rec["reasons"][name] = rec["reasons"].get(name, 0) + 1
        if inside_candidates:
            acc["with_candidate"] += 1
        else:
            # every fragment of the person in the box was discarded — by what?
            inside_people = {person_of(t) for t in inside}
            for name, ids in by_name.items():
                if name == "CANDIDATE":
                    continue
                if any(person_of(i) in inside_people for i in ids):
                    acc["reasons"][name] = acc["reasons"].get(name, 0) + 1
    except Exception as e:                      # metrics must never break a run
        logger.debug(f"behaviour sampling failed: {e}")


def bbox_hits_annotation(detected_bbox, ann_bboxes_norm, img_w, img_h):
    """True if detected_bbox [x1 y1 x2 y2] overlaps any annotated bbox."""
    if not detected_bbox or not ann_bboxes_norm:
        return False
    for cx, cy, w, h in ann_bboxes_norm:
        ann_abs = yolo_to_abs(cx, cy, w, h, img_w, img_h)
        if detection_overlap(detected_bbox, ann_abs) >= OVERLAP_THRESHOLD:
            return True
    return False


# ---------------------------------------------------------------------------
# Threading helpers
# ---------------------------------------------------------------------------

def _get_video_meta(video_path, max_w, max_h):
    """Returns (w, h, fps, duration_s) capped by max_w/max_h."""
    cap = cv2.VideoCapture(video_path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    duration = n_frames / fps if fps > 0 else 0.0
    return min(w, max_w), min(h, max_h), fps, duration


def _get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return n / fps if fps > 0 else 0.0


def _compute_sample_interval(fps: float, sampling_rate: int, factor_tracking: float) -> int:
    """
    Mirror of VideoStream's sample_interval calculation.
    Only every sample_interval-th frame is pushed into frame_queue.
    """
    target_fps = 30
    interval = sampling_rate
    if fps != target_fps and fps > 0:
        interval = int(interval * (fps / target_fps))
    interval = max(1, int(interval / factor_tracking))
    return interval


def capture_and_write_thread(
    video_path, frame_queue, shared_data_queue, detections, stop_event,
    max_w, max_h, fps, sample_interval,
    human_tracker, conf_high_thres,
    ann_bboxes_norm, frame_w, frame_h, behavior_acc,
    video_writer, out_w, out_h,
):
    """
    Mirrors frame_capture_live_thread: reads every frame from the video file,
    feeds sampled frames to the processing thread, and in the SAME loop iteration
    reads the latest prediction from shared_data_queue (non-blocking, sticky).
    This keeps annotations exactly in sync with the displayed frame — no display_queue lag.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open: {video_path}")
        stop_event.set()
        return

    sleep_time = 1.0 / max(fps, 1.0)
    frame_counter = 0
    predicted_class = ClassesEnum.normal
    info_predictions = []
    ann_bboxes_norm = ann_bboxes_norm or []

    try:
        while not stop_event.is_set():
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if w > max_w or h > max_h:
                scale = min(max_w / w, max_h / h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            frame_copy = frame.copy()  # copy before process_frame mutates in-place (water mask)
            sampled = frame_counter % sample_interval == 0

            # Feed sampled frame to processing thread (drop if full — keeps processing real-time).
            if sampled:
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

            # Did the annotated drowning person survive behaviour analysis this frame?
            if behavior_acc is not None:
                sample_behavior_on_annotation(
                    human_tracker, ann_bboxes_norm, frame_w, frame_h, behavior_acc
                )

            # Read latest prediction (non-blocking; previous value is kept when nothing new arrived).
            try:
                final_class, final_conf, final_bbox, info_predictions = shared_data_queue.get_nowait()
                predicted_class = final_class
                if final_class == ClassesEnum.drowning:
                    detections.append((final_conf, final_bbox))
            except queue.Empty:
                pass

            # Draw and write only the sampled frames, so every saved frame carries its
            # own predictions and the file matches the rate the pipeline actually sees.
            # Writing all of them repeated each prediction sample_interval times and made
            # the file that many times longer than the source.
            if video_writer is not None and sampled:
                annotated = draw_results(
                    frame_copy, info_predictions, conf_high_thres,
                    False, True, True,
                    human_tracker, predicted_class, None,
                )
                write_frame = annotated if annotated is not None else frame_copy
                if write_frame.ndim == 3 and write_frame.shape[2] == 4:
                    alpha = write_frame[:, :, 3:4].astype(np.float32) / 255.0
                    write_frame = (write_frame[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)

                if ann_bboxes_norm and frame_w and frame_h:
                    for cx, cy, bw, bh in ann_bboxes_norm:
                        x1 = int((cx - bw / 2) * frame_w)
                        y1 = int((cy - bh / 2) * frame_h)
                        x2 = int((cx + bw / 2) * frame_w)
                        y2 = int((cy + bh / 2) * frame_h)
                        cv2.rectangle(write_frame, (x1, y1), (x2, y2), (128, 0, 128), 3)
                        cv2.putText(write_frame, "GT", (x1 + 5, y1 + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 2)

                if out_w and out_h and (write_frame.shape[1] != out_w or write_frame.shape[0] != out_h):
                    write_frame = cv2.resize(write_frame, (out_w, out_h))
                video_writer.write(write_frame)
                if frame_counter % 100 == 0:
                    logger.info(f"  [video write] frame {frame_counter} written to video")

            frame_counter += 1
            elapsed = time.perf_counter() - t0
            remaining = sleep_time - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except Exception as e:
        logger.error(f"Exception in capture_and_write_thread: {e}", exc_info=True)
    finally:
        cap.release()
        stop_event.set()  # always signal processing_thread to exit, even on exception


def process_video(video_path, action_model, human_tracker, args, ann_bboxes_norm=None):
    """
    Run the full pipeline on one video.
    Returns (detections, frame_w, frame_h, fps, duration).
    """
    frame_queue = queue.Queue(maxsize=args.max_queue_size)
    shared_data_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    processing_done_event = threading.Event()

    frame_w, frame_h, fps, duration = _get_video_meta(video_path, args.max_frame_width, args.max_frame_height)
    detections = []
    behavior_acc = new_behavior_acc()

    # Must be called before process_frame() — sets tile_cols/tile_rows on the tracker.
    sample_interval = _compute_sample_interval(fps, action_model.get_sampling_rate(), args.factor_tracking)

    human_tracker.set_video_metadata(
        video_width=frame_w,
        video_height=frame_h,
        fps=min(fps, 30),
        sample_interval=sample_interval,
    )

    logger.info(f"  sample_interval={sample_interval} (fps={fps:.1f}, sampling_rate={action_model.get_sampling_rate()}, factor_tracking={args.factor_tracking})")

    vid_writer = None
    out_w, out_h = frame_w, frame_h
    if getattr(args, "save_video", False):
        stem = Path(video_path).stem
        os.makedirs("saved_videos", exist_ok=True)
        out_path = os.path.join("saved_videos", f"{stem}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if frame_w > 1920 or frame_h > 1080:
            out_w, out_h = frame_w // 2, frame_h // 2
        # Only sampled frames are written, so the file's rate is the sampled rate.
        # A float: integer division would stretch the clip (30/7 -> 4 instead of 4.29).
        saved_fps = max(1.0, min(fps, 30) / max(1, sample_interval))
        vid_writer = cv2.VideoWriter(out_path, fourcc, saved_fps, (out_w, out_h))
        if vid_writer.isOpened():
            logger.info(f"  Saving annotated video -> {out_path} "
                        f"({out_w}x{out_h} @ {saved_fps:.2f}fps, every {sample_interval} frames)")
        else:
            logger.error(f"  VideoWriter failed to open: {out_path}")
            vid_writer = None

    proc = threading.Thread(
        target=processing_thread,
        args=(
            frame_queue, action_model, human_tracker,
            action_model.get_num_frames(), shared_data_queue,
            args.slide_window_perc, processing_done_event, stop_event,
            args.conf_high_thres, args.conf_low_thres,
            args.factor_tracking, args.batch_size, frame_w, frame_h,
        ),
        daemon=False,
    )
    capture = threading.Thread(
        target=capture_and_write_thread,
        args=(
            video_path, frame_queue, shared_data_queue, detections, stop_event,
            args.max_frame_width, args.max_frame_height, min(fps, 30), sample_interval,
            human_tracker, args.conf_high_thres,
            ann_bboxes_norm, frame_w, frame_h, behavior_acc,
            vid_writer, out_w, out_h,
        ),
        daemon=False,
    )

    proc.start()
    capture.start()

    capture.join()

    # Drain any remaining results that arrived after the last frame was read.
    drain_deadline = time.perf_counter() + 120  # give processing up to 2 min to finish
    while time.perf_counter() < drain_deadline:
        if processing_done_event.is_set() and shared_data_queue.empty():
            break
        try:
            final_class, final_conf, final_bbox, _ = shared_data_queue.get(timeout=0.5)
            if final_class == ClassesEnum.drowning:
                detections.append((final_conf, final_bbox))
        except queue.Empty:
            if processing_done_event.is_set():
                break
    else:
        logger.warning("Drain loop timed out — processing thread may be stuck")

    proc.join(timeout=300)
    if proc.is_alive():
        logger.error("processing_thread did not exit within 5 min — continuing anyway")

    if vid_writer:
        vid_writer.release()

    frame_queue.queue.clear()
    shared_data_queue.queue.clear()

    return detections, frame_w, frame_h, fps, duration, behavior_acc


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def _infer_label(input_dir: str) -> str | None:
    """Try to infer the ground-truth label from the path components."""
    parts = Path(input_dir).parts
    for part in reversed(parts):
        if part.lower() == "drowning":
            return "drowning"
        if part.lower() == "normal":
            return "normal"
    return None


def collect_videos(input_dir, label_override: str | None = None):
    """
    Returns list of (video_path, label) where label in {'drowning', 'normal'}.

    If input_dir contains drowning/ and normal/ subdirs the labels come from
    those folder names.  If a flat directory of videos is given, the label is
    taken from label_override (or auto-detected from the path).
    """
    has_class_subdirs = any(
        os.path.isdir(os.path.join(input_dir, d)) and d in ("drowning", "normal")
        for d in os.listdir(input_dir)
    )

    videos = []
    if has_class_subdirs and label_override is None:
        for label in ("drowning", "normal"):
            class_dir = os.path.join(input_dir, label)
            if not os.path.isdir(class_dir):
                continue
            for root, _, files in os.walk(class_dir):
                for fname in files:
                    if Path(fname).suffix.lower() in VIDEO_EXTS:
                        videos.append((os.path.join(root, fname), label))
    else:
        label = label_override or _infer_label(input_dir)
        if label is None:
            raise ValueError(
                "Cannot determine label. Pass --label drowning or --label normal."
            )
        for root, _, files in os.walk(input_dir):
            for fname in files:
                if Path(fname).suffix.lower() in VIDEO_EXTS:
                    videos.append((os.path.join(root, fname), label))

    videos.sort()
    return videos


def sample_videos(videos, perc: float, seed: int = 0):
    """Take a stratified random sample of `perc` of the videos.

    Stratified by (label, location) rather than sampled flat: at 1% of 1905 videos a
    flat draw can miss whole locations or skew the drowning/normal balance, which makes
    the run useless for comparing against a full one. Each group contributes its own
    share, and the leftovers go to the groups with the largest fractional remainder so
    the total is exactly round(len(videos) * perc).

    Deterministic for a given seed, so two runs compare the same videos - the point of
    sampling here is usually to A/B a setting, which a different subset each time
    would defeat.
    """
    if perc >= 1.0:
        return list(videos)
    if perc <= 0.0:
        return []

    groups = {}
    for item in videos:
        path, label = item[0], item[1]
        groups.setdefault((label, Path(path).parent.name), []).append(item)

    target = round(len(videos) * perc)
    quotas, remainders = {}, []
    for key, items in groups.items():
        exact = len(items) * perc
        quotas[key] = int(exact)
        remainders.append((exact - int(exact), key))
    # hand the rounding leftovers to the groups that lost the most to truncation
    for _, key in sorted(remainders, reverse=True)[:max(0, target - sum(quotas.values()))]:
        quotas[key] += 1

    rng = random.Random(seed)
    sampled = []
    for key in sorted(groups):
        take = min(quotas[key], len(groups[key]))
        if take:
            sampled.extend(rng.sample(groups[key], take))
    sampled.sort()
    logger.info(
        "Sampled %d of %d videos (%.1f%%) from %d groups, seed=%d",
        len(sampled), len(videos), 100 * perc, len(groups), seed,
    )
    return sampled


# Judged frames below which no verdict is given. A 22s video that produced 7 judged
# frames was being filed as "never a candidate, unreachable by X3D" on 8% coverage,
# while the action model had in fact run on that person.
MIN_FRAMES_FOR_BEHAVIOR_VERDICT = 20

BEHAVIOR_OUTCOMES = ["ALWAYS_CANDIDATE", "SOMETIMES_DISCARDED", "DISCARDED_BY_BEHAVIOR",
                     "NEVER_ANALYSED", "TOO_FEW_FRAMES", "NO_TRACK_IN_ANNOTATION"]
BEHAVIOR_PROBLEMS = ["DISCARDED_BY_BEHAVIOR", "SOMETIMES_DISCARDED", "NEVER_ANALYSED",
                     "TOO_FEW_FRAMES", "NO_TRACK_IN_ANNOTATION"]


def print_behavior_summary(per_video, n_total):
    """Running tally of whether the annotated drowning person survived behaviour analysis."""
    rows = [v for v in per_video if v.get("behavior_outcome")]
    if not rows:
        return
    counts = {k: 0 for k in BEHAVIOR_OUTCOMES}
    for v in rows:
        counts[v["behavior_outcome"]] = counts.get(v["behavior_outcome"], 0) + 1
    n = len(rows)
    logger.info("")
    logger.info(f"  Behaviour analysis on annotated drowning person "
          f"[{n} annotated | {len(per_video)}/{n_total} processed]:")
    for key in BEHAVIOR_OUTCOMES:
        mark = "  <-- unreachable by X3D" if key == "DISCARDED_BY_BEHAVIOR" and counts[key] else ""
        logger.info(f"    {key:<24s} {counts[key]:4d}  ({100 * counts[key] / n:5.1f}%){mark}")
    reachable = counts["ALWAYS_CANDIDATE"] + counts["SOMETIMES_DISCARDED"]
    logger.info(f"    {'reachable at all':<24s} {reachable:4d}  ({100 * reachable / n:5.1f}%)")
    logger.info("")
    print_behavior_location_table(rows)


SHORT_OUTCOME = {"ALWAYS_CANDIDATE": "ALWAYS", "SOMETIMES_DISCARDED": "SOMET",
                 "DISCARDED_BY_BEHAVIOR": "DISCARD", "NEVER_ANALYSED": "NOJUDG",
                 "TOO_FEW_FRAMES": "FEWFR", "NO_TRACK_IN_ANNOTATION": "NOTRK"}


def print_behavior_location_table(rows):
    """Same behaviour outcomes as print_behavior_summary, broken down per location.

    Locations differ in water colour, camera angle and distance, so a discard rate that
    looks acceptable overall can be concentrated in one site.
    """
    if not rows:
        return
    loc = {}
    for v in rows:
        s = loc.setdefault(Path(v["path"]).parent.name,
                           {k: 0 for k in BEHAVIOR_OUTCOMES} | {"n": 0, "passive": 0,
                                                                "active": 0, "unknown": 0})
        s["n"] += 1
        s[v["behavior_outcome"]] += 1
        s[v.get("drowning_type") or "unknown"] += 1

    cols = [SHORT_OUTCOME[k] for k in BEHAVIOR_OUTCOMES]
    hdr = (f"{'Location':<22} {'Ann':>4}" + "".join(f" {c:>7}" for c in cols)
           + f" {'Reach%':>7} | {'Pass':>5} {'Act':>5} {'Unk':>5}")
    bar = "-" * len(hdr)
    logger.info(f"  Behaviour per location:")
    logger.info(f"  {hdr}")
    logger.info(f"  {bar}")
    tot = {k: 0 for k in BEHAVIOR_OUTCOMES} | {"n": 0, "passive": 0, "active": 0, "unknown": 0}
    for name in sorted(loc):
        s = loc[name]
        reach = s["ALWAYS_CANDIDATE"] + s["SOMETIMES_DISCARDED"]
        logger.info(f"  {name:<22} {s['n']:>4}"
              + "".join(f" {s[k]:>7}" for k in BEHAVIOR_OUTCOMES)
              + f" {reach / s['n'] * 100:>6.1f}% | {s['passive']:>5} {s['active']:>5} {s['unknown']:>5}")
        for k in tot:
            tot[k] += s[k]
    reach_t = tot["ALWAYS_CANDIDATE"] + tot["SOMETIMES_DISCARDED"]
    logger.info(f"  {bar}")
    logger.info(f"  {'TOTAL':<22} {tot['n']:>4}"
          + "".join(f" {tot[k]:>7}" for k in BEHAVIOR_OUTCOMES)
          + f" {reach_t / tot['n'] * 100:>6.1f}% | {tot['passive']:>5} {tot['active']:>5} {tot['unknown']:>5}")
    logger.info("")


def print_drowning_type_table(per_video):
    """Detection performance split by drowning sub-type.

    This is the number that says whether one model can carry both cases: if passive and
    active detect at very different rates, a single "drowning" class is averaging over
    two different problems.
    """
    rows = [v for v in per_video if v.get("drowning_type")]
    if not rows:
        return
    by_type = {}
    for v in rows:
        s = by_type.setdefault(v["drowning_type"], {"n": 0, "TP": 0, "NOVL": 0, "MISS": 0,
                                                    "reach": 0, "under": []})
        s["n"] += 1
        key = {"TP": "TP", "DETECTED_NO_OVERLAP": "NOVL", "MISSED": "MISS"}.get(v["outcome"])
        if key:
            s[key] += 1
        if v.get("behavior_outcome") in ("ALWAYS_CANDIDATE", "SOMETIMES_DISCARDED"):
            s["reach"] += 1
        uf = (v.get("head") or {}).get("under_frac")
        if uf is not None:
            s["under"].append(uf)

    hdr = (f"{'Drowning type':<16} {'Vids':>5} {'TP':>4} {'NOVL':>5} {'MISS':>5} {'Det%':>7}"
           f" {'Reach%':>7} {'mean under%':>12}")
    logger.info(f"  Detection by drowning type (passive = head under water >= "
          f"{PASSIVE_UNDER_FRAC:.0%} of detected frames):")
    logger.info(f"  {hdr}")
    logger.info(f"  {'-' * len(hdr)}")
    for t in DROWNING_TYPES:
        s = by_type.get(t)
        if not s:
            continue
        det = s["TP"] + s["NOVL"]
        mu = f"{100 * sum(s['under']) / len(s['under']):.1f}%" if s["under"] else "-"
        logger.info(f"  {t:<16} {s['n']:>5} {s['TP']:>4} {s['NOVL']:>5} {s['MISS']:>5}"
              f" {det / s['n'] * 100:>6.1f}% {s['reach'] / s['n'] * 100:>6.1f}% {mu:>12}")
    logger.info("")


# Share of the annotated person's frames excluded for head visibility above which the
# clip is flagged for review. The head classifier firing this often on someone who is
# supposed to be drowning usually means the acting is wrong - a person leaping clear of
# the water with their shoulders out looks exactly like a swimmer to the classifier,
# because that is what it is. Flagged, never auto-removed: it is a review queue.
BAD_ACTING_HEAD_EXCLUDED_FRAC = 0.20

# How often the annotated person was kept, bucketed so a clip discarded 80% of the time
# is not filed alongside one discarded 5% of the time. SOMETIMES_DISCARDED covers both.
DISCARD_LEVELS = [(0.0, "ALWAYS_DISCARDED"), (50.0, "MOSTLY_DISCARDED"),
                  (95.0, "PARTLY_DISCARDED"), (100.0, "RARELY_DISCARDED"),
                  (float("inf"), "NEVER_DISCARDED")]


def discard_level(candidate_pct):
    """Bucket a candidate percentage. None when the person was never tracked."""
    if candidate_pct is None:
        return "NO_TRACK"
    for bar, name in DISCARD_LEVELS:
        if candidate_pct <= bar:
            return name
    return "NEVER_DISCARDED"


def annotated_person_stats(v):
    """Per-frame record for the person the annotation points at.

    The metric samples whoever is inside the annotation box, and in a busy pool other
    swimmers drift through it, so the person counted in the MOST frames is taken as the
    annotated one and the rest are reported separately as contamination.
    """
    per_track = v.get("behavior_per_track") or {}
    if not per_track:
        return {}
    pid, rec = max(per_track.items(), key=lambda kv: kv[1]["frames"])
    frames = max(rec["frames"], 1)
    head_excl = rec["reasons"].get("EXCLUDED_HEAD_VISIBLE", 0)
    others = sum(r["frames"] for t, r in per_track.items() if t != pid)
    return {
        "person": pid,
        "frames": rec["frames"],
        "candidate_pct": 100 * rec["candidate"] / frames,
        "head_excluded_pct": 100 * head_excl / frames,
        "reasons": rec["reasons"],
        "other_people_frames": others,
    }


ALERT_GT_LEVELS = ["ALWAYS_IN_GT", "SOMETIMES_IN_GT", "NEVER_IN_GT", "NO_ALERT"]


def write_outcomes_csv(per_video, path):
    """One row per video: what the pipeline decided, with no behaviour-analysis fields.

    Written even when USE_BEHAVIOR_ANALYSIS is off, which drowning_meta.csv cannot be -
    that one needs head evidence and would come out empty. This is the file to diff
    between a run with behaviour analysis and one without.
    """
    import csv

    if not per_video:
        return
    fields = ["video_path", "location", "video", "label", "detection_outcome",
              "alerts", "alerts_in_gt", "alert_gt_level", "alert_gt_pct",
              "max_conf", "duration_s", "resolution", "fps"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in sorted(per_video, key=lambda x: x["path"]):
            pct = v.get("alert_gt_pct")
            w.writerow({
                "video_path": v["path"],
                "location": Path(v["path"]).parent.name,
                "video": Path(v["path"]).name,
                "label": v["label"],
                "detection_outcome": v["outcome"],
                "alerts": v.get("alerts", 0),
                "alerts_in_gt": v.get("alerts_in_gt", 0),
                "alert_gt_level": v.get("alert_gt_level", ""),
                "alert_gt_pct": "" if pct is None else f"{pct:.1f}",
                "max_conf": f"{v.get('max_conf') or 0:.3f}",
                "duration_s": f"{v.get('duration') or 0:.1f}",
                "resolution": v.get("resolution") or "",
                "fps": f"{v.get('fps') or 0:.2f}",
            })
    try:
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("Could not update %s (%s) - rows kept in %s", path, e, tmp)


def print_alert_gt_table(per_video):
    """Share of videos whose alerts landed on the annotated person.

    Says nothing about which track was the drowning person, so it survives a crowded
    annotation box - and it is defined identically with behaviour analysis on or off,
    which makes the two runs directly comparable.
    """
    rows = [v for v in per_video if v.get("alert_gt_level")]
    if not rows:
        return
    counts = collections.Counter(v["alert_gt_level"] for v in rows)
    logger.info("")
    logger.info(f"  Alerts landing on the annotated person  ({len(rows)} videos):")
    for lvl in ALERT_GT_LEVELS:
        c = counts.get(lvl, 0)
        logger.info(f"    {lvl:<18} {c:4d}  ({100 * c / len(rows):5.1f}%)")
    fired = len(rows) - counts.get("NO_ALERT", 0)
    if fired:
        always = counts.get("ALWAYS_IN_GT", 0)
        logger.info(f"    {'of those that fired':<18} {always:4d} always in GT "
                    f"({100 * always / fired:.1f}% of {fired})")


def write_drowning_meta_csv(per_video, path, passive_under_frac=PASSIVE_UNDER_FRAC,
                            verbose=True):
    """One row per annotated drowning video: where it is, and active or passive.

    Written so the rows can be QA'd by hand and then folded back into the annotations
    as a third class. under_frac is the evidence behind drowning_type, so a reviewer can
    see how marginal each call was instead of trusting the verdict blind.

    Rewritten in full after every video (a run over 1900 videos takes hours — the CSV is
    useful while it is still going), so it is written to a temp file and moved into place:
    a half-written CSV would be indistinguishable from a finished one. verbose=False
    keeps the per-video rewrites quiet.
    """
    import csv

    rows = [v for v in per_video if v.get("head")]
    if not rows:
        if verbose:
            logger.warning("No head-visibility evidence collected — %s not written "
                           "(is USE_BEHAVIOR_ANALYSIS=True and HEAD_MODEL_PATH set?)", path)
        return
    fields = ["video_path", "location", "video", "annotation_files", "drowning_type",
              "under_frac", "under_or_missing_frac", "hs_frac", "clear_frac",
              "n_frames", "n_detected", "n_windows", "n_tracks", "ambiguous_windows",
              "detection_outcome", "behavior_outcome", "behavior_candidate_pct",
              # the annotated person specifically, rather than whoever was in the box
              "person_id", "person_frames", "person_candidate_pct",
              "person_head_excluded_pct", "other_people_frames",
              "cluster_covered_pct", "cluster_frames",
              "discard_level", "bad_acting_flag", "behavior_reasons",
              "duration_s", "resolution", "passive_threshold"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in sorted(rows, key=lambda x: ((x.get("head") or {}).get("under_frac") or 0)):
            h = v["head"]
            ps = annotated_person_stats(v)
            bad_acting = (ps.get("head_excluded_pct", 0) / 100.0) >= BAD_ACTING_HEAD_EXCLUDED_FRAC

            def _pct(key):
                val = h.get(key)
                return "" if val is None else f"{val:.4f}"

            w.writerow({
                "video_path": v["path"],
                "location": Path(v["path"]).parent.name,
                "video": Path(v["path"]).name,
                "annotation_files": ";".join(v.get("annotation_files") or []),
                "drowning_type": h["drowning_type"],
                "under_frac": _pct("under_frac"),
                "under_or_missing_frac": _pct("under_or_missing_frac"),
                "hs_frac": _pct("hs_frac"),
                "clear_frac": _pct("clear_frac"),
                "n_frames": h["n_frames"],
                "n_detected": h["n_detected"],
                "n_windows": h["n_windows"],
                "n_tracks": h["n_tracks"],
                "ambiguous_windows": h["ambiguous_windows"],
                "detection_outcome": v["outcome"],
                "behavior_outcome": v.get("behavior_outcome") or "",
                "behavior_candidate_pct": ("" if v.get("behavior_candidate_pct") is None
                                           else f"{v['behavior_candidate_pct']:.1f}"),
                "person_id": ps.get("person", ""),
                "person_frames": ps.get("frames", ""),
                "person_candidate_pct": ("" if "candidate_pct" not in ps
                                         else f"{ps['candidate_pct']:.1f}"),
                "person_head_excluded_pct": ("" if "head_excluded_pct" not in ps
                                             else f"{ps['head_excluded_pct']:.1f}"),
                "other_people_frames": ps.get("other_people_frames", ""),
                "cluster_covered_pct": ("" if v.get("cluster_covered_pct") is None
                                        else f"{v['cluster_covered_pct']:.1f}"),
                "cluster_frames": v.get("cluster_frames", ""),
                "discard_level": discard_level(ps.get("candidate_pct",
                                                      v.get("behavior_candidate_pct"))),
                "bad_acting_flag": int(bad_acting),
                "behavior_reasons": ";".join(f"{k}={n_}" for k, n_ in
                                             sorted((v.get("behavior_reasons") or {}).items())),
                "duration_s": f"{v.get('duration') or 0:.1f}",
                "resolution": v.get("resolution") or "",
                "passive_threshold": f"{passive_under_frac:.2f}",
            })
    try:
        os.replace(tmp, path)
    except OSError as e:
        # typically the CSV is open in Excel, which locks it on Windows. Losing the
        # metadata file must not kill a multi-hour evaluation run.
        logger.warning("Could not update %s (%s) — rows kept in %s", path, e, tmp)
        return

    counts = {}
    for v in rows:
        counts[v["head"]["drowning_type"]] = counts.get(v["head"]["drowning_type"], 0) + 1
    summary = ", ".join(f"{k}={counts[k]}" for k in DROWNING_TYPES if counts.get(k))
    logger.info("Wrote %d rows -> %s  (%s)", len(rows), path, summary)
    if verbose:
        logger.info(f"  Drowning metadata: {len(rows)} rows -> {path}  ({summary})\n")


def print_location_table(per_video, n_total, min_duration=15.0):
    """Print running per-location detection stats; also prints a >=min_duration-filtered table."""

    def _build_loc(videos):
        loc = {}
        def _empty():
            return {"D": 0, "TP": 0, "NO_OVL": 0, "MISS": 0, "N": 0, "TN": 0, "FA": 0}
        for v in videos:
            loc_name = Path(v["path"]).parent.name
            s = loc.setdefault(loc_name, _empty())
            if v["label"] == "drowning":
                s["D"] += 1
                if v["outcome"] == "TP":
                    s["TP"] += 1
                elif v["outcome"] == "DETECTED_NO_OVERLAP":
                    s["NO_OVL"] += 1
                elif v["outcome"] == "MISSED":
                    s["MISS"] += 1
            else:
                s["N"] += 1
                if v["outcome"] == "TN":
                    s["TN"] += 1
                elif v["outcome"] == "FALSE_ALARM":
                    s["FA"] += 1
        return loc

    def _print_table(loc, label, n_shown):
        hdr = (f"{'Location':<22} {'D.Tot':>5} {'TP':>4} {'NOVL':>4} {'MISS':>4} {'Det%':>7}"
               f" | {'N.Tot':>5} {'TN':>4} {'FA':>4} {'Spec%':>6}")
        bar = "-" * len(hdr)
        logger.info("")
        logger.info(f"  Per-location {label} [{n_shown}/{n_total}]:")
        logger.info(f"  {hdr}")
        logger.info(f"  {bar}")
        tot = {"D": 0, "TP": 0, "NO_OVL": 0, "MISS": 0, "N": 0, "TN": 0, "FA": 0}
        for loc_name in sorted(loc.keys()):
            s = loc[loc_name]
            dt, nt = s["D"], s["N"]
            det = s["TP"] + s["NO_OVL"]
            dp  = f"{det / dt * 100:.1f}%" if dt else "-"
            sp  = f"{s['TN'] / nt * 100:.1f}%" if nt else "-"
            logger.info(f"  {loc_name:<22} {dt:>5} {s['TP']:>4} {s['NO_OVL']:>4} {s['MISS']:>4} {dp:>7}"
                  f" | {nt:>5} {s['TN']:>4} {s['FA']:>4} {sp:>6}")
            for k in tot:
                tot[k] += s[k]
        det_t = tot["TP"] + tot["NO_OVL"]
        dp_t  = f"{det_t / tot['D'] * 100:.1f}%" if tot["D"] else "-"
        sp_t  = f"{tot['TN'] / tot['N'] * 100:.1f}%" if tot["N"] else "-"
        logger.info(f"  {bar}")
        logger.info(f"  {'TOTAL':<22} {tot['D']:>5} {tot['TP']:>4} {tot['NO_OVL']:>4} {tot['MISS']:>4} {dp_t:>7}"
              f" | {tot['N']:>5} {tot['TN']:>4} {tot['FA']:>4} {sp_t:>6}")
        logger.info("")

    _print_table(_build_loc(per_video), "(all)", len(per_video))

    if min_duration > 0:
        filtered = [v for v in per_video if (v.get("duration") or 0) >= min_duration]
        if filtered:
            _print_table(_build_loc(filtered), f"(>={min_duration:g}s)", len(filtered))


def relocate_saved_videos(per_video):
    """Move saved_videos/{stem}.mp4 into saved_videos/correct/ or saved_videos/wrong/."""
    correct_outcomes = {"TP", "TN"}
    for v in per_video:
        stem = Path(v["path"]).stem
        src = os.path.join("saved_videos", f"{stem}.mp4")
        if not os.path.isfile(src):
            continue
        subfolder = "correct" if v["outcome"] in correct_outcomes else "wrong"
        dst_dir = os.path.join("saved_videos", subfolder)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, os.path.join(dst_dir, f"{stem}.mp4"))
    logger.info("Relocated saved videos into saved_videos/correct/ and saved_videos/wrong/")


def evaluate(videos, action_model, human_tracker, annotations_dir, args, min_duration=0.0):
    stats = {
        "drowning": {"total": 0, "true_positive": 0, "detected_no_overlap": 0, "missed": 0},
        "normal":   {"total": 0, "true_negative": 0, "false_alarm": 0},
        # did behaviour analysis leave the annotated drowning person reachable at all?
        "behavior": {"with_annotation": 0, "never_candidate": 0, "sometimes_candidate": 0,
                     "always_candidate": 0, "no_track_found": 0, "never_analysed": 0,
                     "too_few_frames": 0, "reasons": {}},
        "drowning_type": {k: 0 for k in DROWNING_TYPES},
    }
    per_video = []

    for idx, (video_path, label) in enumerate(videos):
        stem = Path(video_path).stem
        logger.info(f"[{idx+1}/{len(videos)}] {label} | {video_path}")

        if min_duration > 0:
            dur = _get_video_duration(video_path)
            if dur < min_duration:
                logger.info(f"  SKIP (duration {dur:.1f}s < {min_duration:.0f}s)")
                continue

        ann_bboxes = load_drowning_bboxes(annotations_dir, stem)
        if label == "drowning" and not ann_bboxes:
            logger.warning(f"  No drowning annotation found for {stem}")

        detections, frame_w, frame_h, fps, duration, behavior_acc = process_video(
            video_path, action_model, human_tracker, args, ann_bboxes)
        drowning_fired = len(detections) > 0

        # Behaviour analysis on the annotated person: the box may hold several people,
        # so it counts as survived if ANY track inside it was still a candidate.
        behavior_outcome, cand_pct, head = None, None, None
        if label == "drowning" and ann_bboxes:
            stats["behavior"]["with_annotation"] += 1
            wt, wc = behavior_acc["with_track"], behavior_acc["with_candidate"]
            wu = behavior_acc["with_track_unclassified"]
            if 0 < wt < MIN_FRAMES_FOR_BEHAVIOR_VERDICT and wc == 0:
                # too little to call: saying "unreachable" on a handful of frames has
                # been wrong every time it was checked against the video
                behavior_outcome = "TOO_FEW_FRAMES"
                stats["behavior"]["too_few_frames"] = stats["behavior"].get("too_few_frames", 0) + 1
                cand_pct = 100.0 * wc / wt
            elif wt == 0 and wu > 0:
                # tracked, but every id was new to behaviour analysis when sampled
                behavior_outcome = "NEVER_ANALYSED"
                stats["behavior"]["never_analysed"] += 1
            elif wt == 0:
                behavior_outcome = "NO_TRACK_IN_ANNOTATION"
                stats["behavior"]["no_track_found"] += 1
            else:
                cand_pct = 100.0 * wc / wt
                if wc == 0:
                    behavior_outcome = "DISCARDED_BY_BEHAVIOR"
                    stats["behavior"]["never_candidate"] += 1
                elif wc == wt:
                    behavior_outcome = "ALWAYS_CANDIDATE"
                    stats["behavior"]["always_candidate"] += 1
                else:
                    behavior_outcome = "SOMETIMES_DISCARDED"
                    stats["behavior"]["sometimes_candidate"] += 1
            for reason, cnt in behavior_acc["reasons"].items():
                stats["behavior"]["reasons"][reason] = stats["behavior"]["reasons"].get(reason, 0) + cnt
            per_track = behavior_acc.get("per_track") or {}
            if per_track:
                ranked = sorted(per_track.items(), key=lambda kv: -kv[1]["frames"])
                logger.info(f"  tracks counted inside the annotation ({len(ranked)}):")
                for tid, rec in ranked[:8]:
                    pct = 100 * rec["candidate"] / max(rec["frames"], 1)
                    logger.info(f"    track {tid:<5} frames={rec['frames']:<5} "
                                f"candidate={rec['candidate']:<5} ({pct:5.1f}%)  {rec['reasons']}")
            logger.info(
                f"  behaviour: {behavior_outcome}"
                + (f"  candidate in {wc}/{wt} sampled frames ({cand_pct:.1f}%)" if cand_pct is not None else "")
                + (f"  unclassified_frames={wu}" if wu else "")
                + (f"  cluster_covered={behavior_acc['with_cluster']}/"
                   f"{behavior_acc['cluster_frames']}"
                   f" ({100 * behavior_acc['with_cluster'] / behavior_acc['cluster_frames']:.1f}%)"
                   if behavior_acc["cluster_frames"] else "")
                + (f"  discard reasons={behavior_acc['reasons']}" if behavior_acc["reasons"] else "")
            )

            # active vs passive drowning, from the head-visibility model on that person
            head = summarize_head_evidence(
                behavior_acc, getattr(args, "passive_under_frac", PASSIVE_UNDER_FRAC))
            stats["drowning_type"][head["drowning_type"]] = \
                stats["drowning_type"].get(head["drowning_type"], 0) + 1
            uf = head["under_frac"]
            logger.info(
                f"  drowning type: {head['drowning_type']}"
                f"  under={(f'{uf:.1%}' if uf is not None else '-')}"
                f"  frames={head['n_detected']}/{head['n_frames']} detected"
                f"  windows={head['n_windows']}"
                + (f"  ambiguous_windows={head['ambiguous_windows']}" if head["ambiguous_windows"] else "")
            )

        # How many of this video's alerts landed on the annotated person, independent
        # of behaviour analysis and of which track the person was. "Always" vs
        # "sometimes" here is about the ALERTS, so it can be compared directly between a
        # run with behaviour analysis and one without.
        alerts_in_gt = sum(1 for _, bbox in detections
                           if bbox_hits_annotation(bbox, ann_bboxes, frame_w, frame_h))
        if not detections:
            alert_gt_level = "NO_ALERT"
        elif alerts_in_gt == len(detections):
            alert_gt_level = "ALWAYS_IN_GT"
        elif alerts_in_gt:
            alert_gt_level = "SOMETIMES_IN_GT"
        else:
            alert_gt_level = "NEVER_IN_GT"
        stats.setdefault("alert_gt", {}).setdefault(alert_gt_level, 0)
        stats["alert_gt"][alert_gt_level] += 1

        if label == "drowning":
            stats["drowning"]["total"] += 1
            if drowning_fired:
                hit = alerts_in_gt > 0
                if hit:
                    stats["drowning"]["true_positive"] += 1
                    outcome = "TP"
                else:
                    stats["drowning"]["detected_no_overlap"] += 1
                    outcome = "DETECTED_NO_OVERLAP"
            else:
                stats["drowning"]["missed"] += 1
                outcome = "MISSED"
        else:  # normal
            stats["normal"]["total"] += 1
            if drowning_fired:
                stats["normal"]["false_alarm"] += 1
                outcome = "FALSE_ALARM"
            else:
                stats["normal"]["true_negative"] += 1
                outcome = "TN"

        n_alerts = len(detections)
        max_conf = max((c for c, _ in detections), default=0.0)
        logger.info(f"  -> {outcome}  alerts={n_alerts}  max_conf={max_conf:.3f}  ann_bboxes={len(ann_bboxes)}")
        per_video.append({"path": video_path, "label": label, "outcome": outcome,
                          "alerts": n_alerts, "max_conf": max_conf,
                          "resolution": f"{frame_w}x{frame_h}",
                          "fps": fps, "duration": duration,
                          "behavior_outcome": behavior_outcome,
                          "behavior_candidate_pct": cand_pct,
                          "behavior_reasons": behavior_acc["reasons"],
                          "behavior_unclassified_frames": behavior_acc["with_track_unclassified"],
                          "behavior_per_track": behavior_acc.get("per_track") or {},
                          "alerts_in_gt": alerts_in_gt,
                          "alert_gt_level": alert_gt_level,
                          "alert_gt_pct": (100.0 * alerts_in_gt / len(detections)
                                           if detections else None),
                          "cluster_covered_pct": (
                              100.0 * behavior_acc["with_cluster"] / behavior_acc["cluster_frames"]
                              if behavior_acc["cluster_frames"] else None),
                          "cluster_frames": behavior_acc["cluster_frames"],
                          "annotation_files": annotation_files(annotations_dir, stem),
                          "drowning_type": (head or {}).get("drowning_type"),
                          "head": head})

        print_location_table(per_video, len(videos))
        print_behavior_summary(per_video, len(videos))
        print_drowning_type_table(per_video)
        print_alert_gt_table(per_video)

        # refreshed every video, so the CSV can be QA'd while the run is still going
        outcomes_path = getattr(args, "save_outcomes", None)
        if outcomes_path:
            write_outcomes_csv(per_video, outcomes_path)

        meta_path = getattr(args, "save_drowning_meta", None)
        if meta_path:
            write_drowning_meta_csv(per_video, meta_path,
                                    getattr(args, "passive_under_frac", PASSIVE_UNDER_FRAC),
                                    verbose=False)

    return stats, per_video


def print_report(stats, per_video):
    print_location_table(per_video, len(per_video))
    print_behavior_location_table([v for v in per_video if v.get("behavior_outcome")])
    print_drowning_type_table(per_video)
    print_alert_gt_table(per_video)

    d = stats["drowning"]
    n = stats["normal"]

    tp = d["true_positive"]
    fn_overlap = d["detected_no_overlap"]   # detected but bbox missed annotation
    fn_missed  = d["missed"]
    tn = n["true_negative"]
    fp = n["false_alarm"]

    total_drowning = d["total"]
    total_normal   = n["total"]

    sensitivity = tp / total_drowning if total_drowning else 0.0
    specificity  = tn / total_normal  if total_normal  else 0.0
    precision    = tp / (tp + fp)     if (tp + fp)     else 0.0
    f1           = (2 * precision * sensitivity / (precision + sensitivity)
                    if (precision + sensitivity) else 0.0)

    sep = "=" * 60
    logger.info("")
    logger.info(f"{sep}")
    logger.info("  END-TO-END EVALUATION RESULTS")
    logger.info(sep)
    logger.info("")
    logger.info(f"  DROWNING videos  ({total_drowning} total)")
    logger.info(f"    True Positives (detected + bbox overlap):  {tp}")
    logger.info(f"    Detected but bbox missed annotation:       {fn_overlap}")
    logger.info(f"    Missed (not detected at all):              {fn_missed}")
    logger.info("")
    logger.info(f"  NORMAL videos  ({total_normal} total)")
    logger.info(f"    True Negatives (correctly not alerted):   {tn}")
    logger.info(f"    False Alarms   (incorrectly alerted):     {fp}")
    b = stats.get("behavior", {})
    if b.get("with_annotation"):
        n = b["with_annotation"]
        logger.info("")
        logger.info(f"  BEHAVIOUR ANALYSIS on the annotated drowning person  ({n} annotated videos)")
        logger.info(f"    Candidate in every sampled frame:          {b['always_candidate']}")
        logger.info(f"    Candidate only part of the time:           {b['sometimes_candidate']}")
        logger.info(f"    NEVER a candidate (discarded, unreachable):{b['never_candidate']}"
              f"   <-- {100*b['never_candidate']/n:.1f}% of annotated drowning videos")
        logger.info(f"    No track ever found inside the annotation: {b['no_track_found']}"
              f"   (detection/tracking, not behaviour)")
        logger.info(f"    Tracked but never judged (id kept changing):{b['never_analysed']}"
              f"   (tracking, not behaviour)")
        if b["reasons"]:
            logger.info("    Discard reasons (frames where every track in the box was excluded):")
            for reason, cnt in sorted(b["reasons"].items(), key=lambda kv: -kv[1]):
                logger.info(f"      {reason:<32s} {cnt}")

    logger.info("")
    logger.info(f"  METRICS")
    logger.info(f"    Sensitivity (recall on drowning): {sensitivity:.1%}")
    logger.info(f"    Specificity (recall on normal):   {specificity:.1%}")
    logger.info(f"    Precision:                        {precision:.1%}")
    logger.info(f"    F1 score:                         {f1:.4f}")
    logger.info("")
    logger.info(f"{sep}\n")

    # Videos to review, grouped by what went wrong. DISCARDED_BY_BEHAVIOR first:
    # those are unreachable by X3D no matter how good the model is.
    for outcome in BEHAVIOR_PROBLEMS:
        rows = [v for v in per_video if v.get("behavior_outcome") == outcome]
        if not rows:
            continue
        note = {
            "DISCARDED_BY_BEHAVIOR":  "drowning person never a candidate — X3D never saw them",
            "SOMETIMES_DISCARDED":    "reachable only part of the time",
            "NEVER_ANALYSED":         "tracked, but the id kept changing — behaviour analysis "
                                      "never judged it either way (tracking, not behaviour)",
            "NO_TRACK_IN_ANNOTATION": "nothing ever tracked there — detection/tracking, not behaviour",
        }[outcome]
        logger.info(f"  {outcome}  ({len(rows)})  — {note}")
        logger.info(f"  {'Location':<20} {'Video':<44} {'candidate%':>10}  reasons")
        logger.info("  " + "-" * 110)
        for v in sorted(rows, key=lambda x: (x.get("behavior_candidate_pct") or 0)):
            pct = v.get("behavior_candidate_pct")
            loc, name = Path(v["path"]).parent.name, Path(v["path"]).name
            reasons = v.get("behavior_reasons") or ""
            logger.info(f"  {loc:<20} {name:<44} "
                  f"{(f'{pct:.1f}%' if pct is not None else '-'):>10}  {reasons}")
            # also to the log file, so these stay greppable after the console scrolls
            logger.info(f"BEHAVIOUR {outcome} | [{loc}] {name} | "
                        f"candidate%={pct if pct is not None else '-'} | reasons={reasons}")
        logger.info("")

    # Clips where the annotated drowning person was discarded a lot, whatever the rule.
    # These are the review queue for bad acting: a person who is supposed to be drowning
    # but reads as a swimmer to every rule is usually acting wrong, not analysed wrong.
    buckets = {}
    for v in per_video:
        if not v.get("behavior_outcome"):
            continue
        ps = annotated_person_stats(v)
        pct = ps.get("candidate_pct", v.get("behavior_candidate_pct"))
        buckets.setdefault(discard_level(pct), []).append((pct if pct is not None else -1, v, ps))
    worst = [lvl for lvl in ("ALWAYS_DISCARDED", "MOSTLY_DISCARDED", "PARTLY_DISCARDED")
             if buckets.get(lvl)]
    if worst:
        logger.info("")
        logger.info("  ANNOTATED PERSON DISCARDED - review these for bad acting")
        logger.info(f"  {'Level':<18}{'Location':<14}{'Video':<42}{'kept%':>7}"
                    f"{'head-excl%':>11}{'type':>9}{'det':>8}")
        logger.info("  " + "-" * 110)
        for lvl in worst:
            # sort on the number only: ties would otherwise fall through to
            # comparing the per-video dicts, which raises TypeError
            for pct, v, ps in sorted(buckets[lvl], key=lambda row: row[0]):
                hp = ps.get("head_excluded_pct")
                head_pct = "-" if hp is None else f"{hp:.1f}"
                logger.info(f"  {lvl:<18}{Path(v['path']).parent.name[:12]:<14}"
                            f"{Path(v['path']).name[:40]:<42}"
                            f"{(f'{pct:.1f}' if pct is not None and pct >= 0 else '-'):>7}"
                            f"{head_pct:>11}"
                            f"{(v.get('drowning_type') or '-'):>9}{v['outcome']:>8}")
                logger.info(f"DISCARDED {lvl} | [{Path(v['path']).parent.name}] "
                            f"{Path(v['path']).name} | kept%={pct} | reasons={ps.get('reasons', {})}")
        counts = {lvl: len(buckets.get(lvl, [])) for lvl in
                  ("ALWAYS_DISCARDED", "MOSTLY_DISCARDED", "PARTLY_DISCARDED",
                   "RARELY_DISCARDED", "NEVER_DISCARDED")}
        logger.info("  " + "-" * 110)
        logger.info("  " + "  ".join(f"{k}={v_}" for k, v_ in counts.items() if v_))
        logger.info("")

    # Clips where the head classifier kept firing on the annotated "drowning" person.
    # Almost always the acting rather than the model: someone leaping clear of the water
    # with their shoulders out IS a person whose head is visible.
    flagged = []
    for v in per_video:
        ps = annotated_person_stats(v)
        if ps.get("head_excluded_pct", 0) / 100.0 >= BAD_ACTING_HEAD_EXCLUDED_FRAC:
            flagged.append((ps["head_excluded_pct"], v, ps))
    if flagged:
        flagged.sort(reverse=True, key=lambda r: r[0])
        logger.info("")
        logger.info(f"  REVIEW FOR BAD ACTING  ({len(flagged)})  - the annotated person was "
                    f"excluded as head-visible in >= {BAD_ACTING_HEAD_EXCLUDED_FRAC:.0%} of frames")
        logger.info(f"  {'Location':<18}{'Video':<44}{'head-excl%':>11}{'cand%':>8}"
                    f"{'hs%':>7}{'type':>9}{'det':>8}")
        logger.info("  " + "-" * 106)
        for pct, v, ps in flagged:
            h = v.get("head") or {}
            hs = h.get("hs_frac")
            logger.info(f"  {Path(v['path']).parent.name:<18}{Path(v['path']).name[:42]:<44}"
                        f"{pct:>10.1f}%{ps['candidate_pct']:>7.1f}%"
                        f"{(f'{100*hs:.0f}%' if hs is not None else '-'):>7}"
                        f"{(v.get('drowning_type') or '-'):>9}{v['outcome']:>8}")
            logger.info(f"BAD_ACTING | [{Path(v['path']).parent.name}] {Path(v['path']).name} | "
                        f"head_excluded={pct:.1f}% | reasons={ps['reasons']}")
        logger.info("")

    # Per-video breakdown for failures
    failures = [v for v in per_video if v["outcome"] not in ("TP", "TN")]
    if failures:
        logger.info("  FAILURES / EDGE CASES")
        for v in failures:
            loc  = Path(v["path"]).parent.name
            name = Path(v["path"]).name
            logger.info(f"    [{v['outcome']:24s}]  alerts={v['alerts']}  conf={v['max_conf']:.3f}  [{loc}] {name}")
        logger.info("")

    # Missed drowning videos with video metadata
    missed = [v for v in per_video if v["outcome"] == "MISSED"]
    if missed:
        logger.info("  MISSED DROWNING VIDEOS")
        logger.info(f"  {'Location':<20} {'Video':<45} {'Resolution':>12}  {'FPS':>6}  {'Duration':>10}")
        logger.info("  " + "-" * 100)
        for v in missed:
            dur = v["duration"]
            dur_str = f"{int(dur)//60}m{int(dur)%60:02d}s" if dur >= 60 else f"{dur:.1f}s"
            loc  = Path(v["path"]).parent.name
            name = Path(v["path"]).name
            logger.info(f"  {loc:<20} {name:<45} {v['resolution']:>12}  {v['fps']:>6.2f}  {dur_str:>10}")
        logger.info("")


def parse_env():
    return {
        "slide_window_perc": float(os.getenv("SLIDE_WINDOW_PERC", "0.5")),
        "max_queue_size":     int(os.getenv("MAX_QUEUE_SIZE",     "10")),
        "batch_size":         int(os.getenv("LIVE_BATCH_SIZE",    "4")),
        "conf_high_thres":    float(os.getenv("CONF_HIGH_THRES",  "0.6")),
        "conf_low_thres":     float(os.getenv("CONF_LOW_THRES",   "0.4")),
        "model_variant":      os.getenv("MODEL_VARIANT",          "swin_t"),
        "model_extension":    os.getenv("MODEL_EXTENSION",        ".pth"),
        "max_frame_width":    int(os.getenv("MAX_FRAME_WIDTH",    "10000")),
        "max_frame_height":   int(os.getenv("MAX_FRAME_HEIGHT",   "10000")),
        "max_clusters":       int(os.getenv("MAX_CLUSTERS",       "12")),
        "factor_tracking":    float(os.getenv("FACTOR_TRACKING",  "2")),
        "imgsz":              int(os.getenv("IMGSZ",              "1280")),
        "skip_water_process_frames": int(os.getenv("SKIP_WATER_PROCESS_FRAMES", "150")),
        "max_water_deque_size":      int(os.getenv("MAX_WATER_DEQUE_SIZE",      "10")),
        "detect_water":       os.getenv("DETECT_WATER", "True").lower() == "true",
        "classes":            os.getenv("CLASSES", "normal,drowning").split(","),
        "save_video":         os.getenv("SAVE_VIDEO", "False").lower() == "true",
        "passive_under_frac": float(os.getenv("PASSIVE_UNDER_FRAC", str(PASSIVE_UNDER_FRAC))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default=r"path/to/data",
    )
    parser.add_argument(
        "--annotations_dir",
        default=r"path/to/data",
    )
    parser.add_argument("--model_extension", default=None,
                        help="Override MODEL_EXTENSION from .env (.pth or .engine)")
    parser.add_argument("--label", default=None, choices=["drowning", "normal"],
                        help="Ground-truth label when input_dir is a flat video directory.")
    parser.add_argument("--save_video", type=lambda x: x.lower() == "true", default=True,
                        help="Save annotated videos to saved_videos/ (default: True).")
    parser.add_argument("--min_duration", type=float, default=15.0,
                        help="Skip videos shorter than this many seconds (default: 15). Set 0 to disable.")
    parser.add_argument("--save_outcomes", nargs="?", const="outcomes.csv",
                        default="outcomes.csv", metavar="CSV",
                        help="CSV of per-video detection outcomes, written whether or not "
                             "behaviour analysis is on (default: outcomes.csv). This is "
                             "the file to diff between a run with USE_BEHAVIOR_ANALYSIS "
                             "on and one with it off. Pass an empty string to disable.")
    parser.add_argument("--video_filter", default=None, metavar="TEXT",
                        help="Only evaluate videos whose path contains this text. "
                             "Applied before sampling, for investigating one video.")
    parser.add_argument("--sample_perc_videos", type=float, default=0.2, metavar="FRAC",
                        help="Evaluate only this fraction of the videos, 0.0-1.0 "
                             "(1.0 = all, 0.5 = half, 0.01 = 1%%). Stratified by label "
                             "and location, and deterministic for a given --sample_seed, "
                             "so two runs compare the same videos.")
    parser.add_argument("--sample_seed", type=int, default=0,
                        help="Seed for --sample_perc_videos (default: 0).")
    parser.add_argument("--save_drowning_meta", nargs="?", const="drowning_meta.csv", default="drowning_meta.csv",
                        metavar="CSV",
                        help="CSV for per-video active/passive drowning metadata, rewritten "
                             "after every video (default: drowning_meta.csv). Pass an empty "
                             "string to disable. Needs behaviour analysis and the "
                             "head-visibility model enabled.")
    parser.add_argument("--passive_under_frac", type=float, default=PASSIVE_UNDER_FRAC,
                        help=f"Share of detected frames classified head-under-water above "
                             f"which a video is called passive drowning "
                             f"(default: {PASSIVE_UNDER_FRAC}).")
    cli = parser.parse_args()

    env = parse_env()
    if cli.model_extension:
        env["model_extension"] = cli.model_extension
    env["save_video"] = cli.save_video
    env["passive_under_frac"] = cli.passive_under_frac
    env["save_drowning_meta"] = cli.save_drowning_meta
    env["save_outcomes"] = cli.save_outcomes

    args = argparse.Namespace(**env)

    is_docker = os.path.isdir("/models")
    models_dir = "/models" if is_docker else "models"

    from utils_lib.path_helpers import normalize_path

    logger.info(f"Loading models (variant={args.model_variant}, ext={args.model_extension})")

    if args.model_extension == ".engine":
        from action_recognition.load_models_tensorrt import TensorRTModelLoader
        loader = TensorRTModelLoader(models_dir=normalize_path(models_dir))
    else:
        from action_recognition.load_models_onnx import GenericModelLoader
        loader = GenericModelLoader(models_dir=normalize_path(models_dir))

    action_model = loader.load_action_model(
        args.model_variant, args.classes, model_extension=args.model_extension
    )

    from action_recognition.load_models_tensorrt import TensorRTModelLoader as TrtLoader
    trt_loader = TrtLoader(models_dir=normalize_path(models_dir))
    human_tracker = trt_loader.load_human_tracker(
        imgsz=args.imgsz,
        max_clusters=args.max_clusters,
        skip_water_process_frames=args.skip_water_process_frames,
        max_water_deque_size=args.max_water_deque_size,
        detect_water=args.detect_water,
        use_behavior_analysis=os.getenv("USE_BEHAVIOR_ANALYSIS", "False").lower() == "true",
        head_classifier_path=os.getenv("HEAD_MODEL_PATH"),
        head_classifier_max_batch_size=int(os.getenv("HEAD_MODEL_MAX_BATCH_SIZE", "64")),
        head_visibility_img_size=int(os.getenv("HEAD_VISIBILITY_IMG_SIZE", "112")),
        head_visible_frame_count=int(os.getenv("HEAD_VISIBLE_FRAME_COUNT", "16")),
        head_visibility_min_frames=int(os.getenv("HEAD_VISIBILITY_MIN_FRAMES", "8")),
        head_visibility_sample_rate=int(os.getenv("SAMPLE_HEAD_VISIBILITY", "2")),
        downsample_head_frames=os.getenv("DOWNSAMPLE_HEAD_FRAMES", "True").lower() == "true",
        head_visibility_min_bbox_size=int(os.getenv("HEAD_VISIBILITY_MIN_BBOX_SIZE", "20")),
    )

    sample_interval = action_model.get_sampling_rate()
    num_frames = action_model.get_num_frames()
    human_tracker.set_deque_length(int(args.factor_tracking * num_frames))

    if not 0.0 <= cli.sample_perc_videos <= 1.0:
        parser.error(f"--sample_perc_videos must be between 0.0 and 1.0, "
                     f"got {cli.sample_perc_videos}")

    videos = collect_videos(cli.input_dir, label_override=cli.label)
    logger.info(f"Found {len(videos)} videos in {cli.input_dir}")
    if cli.video_filter:
        videos = [v for v in videos if cli.video_filter.lower() in v[0].lower()]
        logger.info(f"--video_filter={cli.video_filter!r} -> {len(videos)} videos")
        if not videos:
            logger.error("No video matched --video_filter")
            human_tracker.cleanup()
            action_model.cleanup()
            return
    # An explicit --video_filter is a deliberate selection, so it is not sampled down:
    # filtering to one video and then taking 20% of it selects nothing.
    if cli.sample_perc_videos < 1.0 and not cli.video_filter:
        videos = sample_videos(videos, cli.sample_perc_videos, cli.sample_seed)
        if not videos:
            logger.warning("--sample_perc_videos=%s selected no videos, nothing to do",
                           cli.sample_perc_videos)
            human_tracker.cleanup()
            action_model.cleanup()
            return
    elif cli.video_filter and cli.sample_perc_videos < 1.0:
        logger.info("--video_filter given, so --sample_perc_videos=%s is not applied",
                    cli.sample_perc_videos)

    stats, per_video = evaluate(videos, action_model, human_tracker, cli.annotations_dir, args,
                                min_duration=cli.min_duration)
    print_report(stats, per_video)

    if cli.save_outcomes:
        write_outcomes_csv(per_video, cli.save_outcomes)
        logger.info("Wrote per-video outcomes -> %s", cli.save_outcomes)

    if cli.save_drowning_meta:
        write_drowning_meta_csv(per_video, cli.save_drowning_meta, cli.passive_under_frac)

    if args.save_video:
        relocate_saved_videos(per_video)

    human_tracker.cleanup()
    action_model.cleanup()


if __name__ == "__main__":
    main()
