import os
import time
import numpy as np
import logging
from enum import Enum, auto

from human_tracking.head_visibility_classes import NOT_DETECTED, HeadVisibilityClass as HVC

logger = logging.getLogger()


class BehaviorCategory(Enum):
    CANDIDATE = auto()
    EXCLUDED_DIRECTION = auto()
    EXCLUDED_DIRECTION_CHANGE = auto()
    EXCLUDED_HEAD_VISIBLE = auto()


class BehaviorAnalyses:
    """
    Filters tracked humans down to drowning candidates by eliminating individuals
    whose behavior is inconsistent with drowning.

    Approach
    --------
    Two independent exclusion tests are applied to each track's per-frame bbox history:

    1. **Path efficiency** – A freestyle swimmer travels in a sustained direction so
       their centroid traces a nearly straight line.  Path efficiency is defined as
       ``total_displacement / total_path_length`` (range 0–1).  A value near 1 means
       all motion went toward the same destination; a low value means the person
       thrashed in place or changed direction repeatedly.  A track is excluded when
       efficiency exceeds ``PATH_EFFICIENCY_THRESHOLD`` AND the total displacement
       exceeds ``MIN_DISPLACEMENT_BBOX_FRACTION × mean_bbox_size`` (guards against near-stationary noise).

    A bbox aspect-ratio stability test used to sit here, excluding tracks whose
    silhouette barely changed, on the assumption that a drowning person thrashes.
    It was removed: passive drowning (an unconscious person floating) produces the
    MOST stable bbox there is, so the rule discarded exactly the case it most needed
    to keep. Observed in practice on a track the head classifier rated under water
    in 16/16 frames — ratio_diff_std 0.003 against a 0.12 bar.

    A track becomes a **candidate** only when it is NOT excluded by any test.
    If every track is excluded (e.g. a quiet scene with only swimmers), the method
    falls back to returning all tracks so the downstream model is never starved of
    input.

    Constants (class-level, override per-instance if needed)
    ---------------------------------------------------------
    MIN_FRAMES_FOR_ANALYSIS_DIRECTION : int
        Minimum frames for direction analysis when displacement > mean bbox size.
    MIN_FRAMES_FOR_ANALYSIS_DIRECTION_STATIONARY : int
        Minimum frames for direction analysis when displacement ≤ mean bbox size.
    PATH_EFFICIENCY_THRESHOLD : float
        total_displacement / total_path_length above which motion is "linear"
        (0–1; lower = easier to exclude as linear swimmer).
    PATH_CROOKEDNESS_THRESHOLD : float
        Maximum allowed ratio of (max lateral deviation from start→end line) /
        total_displacement.  Paths that bend significantly (L-shape, V-shape)
        exceed this and are kept as candidates.
    PROJ_CLUSTER_THRESHOLD : float
        Maximum allowed fraction of centroid positions in the bottom quartile
        of the projected-position range.  A high fraction means the person
        spent most of the track stationary near the start and only moved in
        a brief final burst.
    MIN_DISPLACEMENT_BBOX_FRACTION : float
        Minimum end-to-end displacement as a fraction of the mean bbox size
        (mean of max(w, h) across frames) before the path efficiency test fires.
        Scale-invariant: a distant small person and a close large person both
        need to travel a proportional distance relative to their own size.
    """

    # Head-visibility exclusion thresholds
    # Checked first (predictions are pre-computed per-frame, no extra cost).
    HEAD_SHOULDERS_EXCLUSION_FRAC: float = 0.20   # >= 20 % HEAD_AND_SHOULDERS_VISIBLE → exclude
    HEAD_CLEARLY_VISIBLE_EXCLUSION_FRAC: float = 0.75  # >= 75 % HEAD_CLEARLY_VISIBLE → exclude
    MIN_FRAMES_FOR_HEAD_CLASSIFICATION: int = 5
    # Minimum model confidence for a prediction to count toward exclusion.
    # Predictions below the threshold are treated as uncertain and ignored.
    HEAD_SHOULDERS_CONF_THRESHOLD: float = 0.70
    HEAD_CLEARLY_CONF_THRESHOLD: float = 0.75

    # --- weighted head-visibility score -------------------------------------
    # "weighted": one score, each frame contributing according to how strong its
    # evidence is. "legacy": the two independent fractions above.
    #
    # The two rules agree at the extremes by construction: an all-h&s track needs
    # 20 % of frames either way, and an all-clear track needs 75 % (0.75 * 0.267 =
    # 0.20). They differ when evidence is SPLIT across the two classes, which the
    # legacy rule cannot represent — a track can be visible in every frame and
    # still clear neither bar.
    HEAD_EXCLUSION_MODE: str = "weighted"
    HEAD_WEIGHT_SHOULDERS: float = 1.0
    HEAD_WEIGHT_CLEAR: float = 0.267              # = 0.20 / 0.75, matches the legacy bars
    HEAD_EXCLUSION_SCORE: float = 0.20
    # Partial credit for "visible, but unsure which kind". Measured on the test set:
    # h&s at 0.50-0.70 confidence was truly under water 0/13 times, while clear at
    # 0.50-0.75 was under water 4/27 (14.8 %) — so h&s earns partial credit and
    # clear earns none by default.
    HEAD_LOW_CONF_FLOOR: float = 0.50
    HEAD_LOW_CONF_SHOULDERS_FACTOR: float = 0.5
    HEAD_LOW_CONF_CLEAR_FACTOR: float = 0.0
    # Fraction of the window in which the person was actually detected. Below this we
    # refuse to exclude: undetected frames in water often mean submerged, so a track
    # seen only intermittently is one we never really observed, not one we cleared.
    HEAD_MIN_COVERAGE: float = 0.60

    MIN_FRAMES_FOR_ANALYSIS_DIRECTION: int = 10
    MIN_FRAMES_FOR_ANALYSIS_DIRECTION_STATIONARY: int = 20
    DIRECTION_CONSISTENCY_THRESHOLD: float = 0.70
    PATH_LINEARITY_THRESHOLD: float = 0.80
    PATH_CROOKEDNESS_THRESHOLD: float = 0.37
    PROJ_CLUSTER_THRESHOLD: float = 0.60
    MIN_ACTIVE_STEP_FRACTION: float = 0.50
    PATH_REVERSAL_THRESHOLD: float = 0.20
    PATH_CROOKEDNESS_SOFT_THRESHOLD: float = 0.27
    ANGULAR_CONSISTENCY_THRESHOLD: float = 0.65
    ANGULAR_CONSISTENCY_FLOOR: float = 0.50
    ANGULAR_FLOOR_MAX_PVE: float = 0.90
    MAX_RUN_FRACTION: float = 0.25
    SUSTAINED_CONSISTENCY_THRESHOLD: float = 0.95
    FRAGMENTED_RUN_CONSISTENCY_THRESHOLD: float = 0.91
    TAIL_MIN_DISPLACEMENT_FACTOR: float = 1.5
    TAIL_MIN_ACTIVE_FRACTION: float = 0.65
    TAIL_ANGULAR_CONSISTENCY_THRESHOLD: float = 0.67
    TAIL_MAX_PREFIX_STEP_FRACTION: float = 0.15
    END_INACTIVE_FRAC_THRESHOLD: float = 0.40
    MIN_ACTIVE_PROJ_STEP_SIZE: float = 0.055
    MIN_DISPLACEMENT_BBOX_FRACTION: float = 1.5
    STATIONARY_SPREAD_FRACTION: float = 0.15
    SMOOTHING_WINDOW: int = 5
    MIN_ESTABLISHED_PAUSES: int = 4
    BREASTSTROKE_PAUSE_RATIO_MAX: float = 0.40
    MIN_FRAMES_FOR_DIRECTION_CHANGE: int = 20
    DIRECTION_CHANGE_LEG_LINEARITY: float = 0.50
    DIRECTION_CHANGE_COS_THRESHOLD: float = -0.50
    DIRECTION_CHANGE_REVERSAL_RATIO: float = 0.50

    # --- swimming-speed gate on the direction exclusion ----------------------
    # Straight-line motion alone does not prove swimming: a face-down body drifting on
    # a current also moves in a straight line, and the head classifier rates it the
    # same as a freestyle swimmer (horizontal / under water). What separates the two is
    # speed.
    #
    # Speed is end-to-end displacement over elapsed WALL time, taken from the frames
    # the track was detected in. The direction test itself only sees the detected
    # frames, so a gap of six frames between two detections looks like one ordinary
    # stroke - it cannot tell slow from fast. The gate needs frame numbers and the
    # processed fps; without them it is skipped.
    #
    # Displacement is measured in "extents": the box's size ALONG THE DIRECTION OF
    # TRAVEL (|ux|*w + |uy|*h), not sqrt(w*h). sqrt(w*h) depends on the camera: a
    # side-on swimmer's box is ~4:1 (182x42 px), whose geometric mean is half the body
    # length, so speeds came out twice as high as for the same swimmer filmed from
    # above (square box). The extent along travel also cancels ground-plane
    # foreshortening: moving along image Y, both the displacement and the box height
    # are squashed by the same perspective factor.
    #
    # Measured in extents/s at 6 processed fps (20260625_205127, 0fcc12e2):
    #   drifting / slow, kept       0.19-0.24  (a face-down drowning person at 0.20-0.22)
    #   swimmers along image Y      0.35-0.41  (top-down camera)
    #   fast swimmers               0.70-1.12
    MIN_SWIM_SPEED_BBOX_PER_S: float = 0.30
    # Head under water in at least this fraction of DETECTED frames → also require a
    # longer straight run (in extents) before excluding, since the head gives no
    # reassurance. 2.0 still admits the top-down Y swimmers after 6 s (2.1-2.3).
    HEAD_UNDER_STRICT_FRAC: float = 0.60
    HEAD_UNDER_MIN_DISPLACEMENT_BBOX: float = 2.0
    save_jsons: bool = False
    # per-run record of the head-visibility decision, dumped when save_jsons is on.
    # Populated by get_drowning_candidates/_head_excluded; empty otherwise.
    head_stats: dict = {}
    # per-run record of the swimming-speed gate, for tracks that passed the direction test.
    motion_stats: dict = {}
    # tracks that moved in a straight line but too slowly (or too briefly) to count as
    # swimming. They stay candidates; this set only lets the overlay tell them apart.
    slow_linear: set = set()

    def __init__(self, head_classifier=None,
                 head_shoulders_conf_threshold: float | None = None,
                 head_clearly_conf_threshold: float | None = None):
        """
        Args:
            head_classifier: Optional HeadVisibilityTRT instance.
            head_shoulders_conf_threshold: Override HEAD_SHOULDERS_CONF_THRESHOLD.
            head_clearly_conf_threshold:   Override HEAD_CLEARLY_CONF_THRESHOLD.
        """
        self.head_classifier = head_classifier
        if head_shoulders_conf_threshold is not None:
            self.HEAD_SHOULDERS_CONF_THRESHOLD = head_shoulders_conf_threshold
        if head_clearly_conf_threshold is not None:
            self.HEAD_CLEARLY_CONF_THRESHOLD = head_clearly_conf_threshold

        self.HEAD_EXCLUSION_MODE = os.getenv("HEAD_EXCLUSION_MODE", self.HEAD_EXCLUSION_MODE).lower()
        for name in ("HEAD_WEIGHT_SHOULDERS", "HEAD_WEIGHT_CLEAR", "HEAD_EXCLUSION_SCORE",
                     "HEAD_LOW_CONF_FLOOR", "HEAD_LOW_CONF_SHOULDERS_FACTOR",
                     "HEAD_LOW_CONF_CLEAR_FACTOR", "HEAD_MIN_COVERAGE",
                     "MIN_SWIM_SPEED_BBOX_PER_S", "HEAD_UNDER_STRICT_FRAC",
                     "HEAD_UNDER_MIN_DISPLACEMENT_BBOX"):
            val = os.getenv(name)
            if val is not None:
                setattr(self, name, float(val))

    def _frame_weight(self, cls, conf: float) -> float:
        """Evidence that this one frame shows a head above water.

        1.0 = confident head-and-shoulders (strongest "not drowning" signal),
        down to 0.0 for under-water or too-uncertain frames.
        """
        if cls == HVC.HEAD_AND_SHOULDERS_VISIBLE:
            if conf >= self.HEAD_SHOULDERS_CONF_THRESHOLD:
                return self.HEAD_WEIGHT_SHOULDERS
            if conf >= self.HEAD_LOW_CONF_FLOOR:
                return self.HEAD_WEIGHT_SHOULDERS * self.HEAD_LOW_CONF_SHOULDERS_FACTOR
        elif cls == HVC.HEAD_CLEARLY_VISIBLE:
            if conf >= self.HEAD_CLEARLY_CONF_THRESHOLD:
                return self.HEAD_WEIGHT_CLEAR
            if conf >= self.HEAD_LOW_CONF_FLOOR:
                return self.HEAD_WEIGHT_CLEAR * self.HEAD_LOW_CONF_CLEAR_FACTOR
        return 0.0

    @staticmethod
    def _select_crops(crops: list, n_select: int, downsample: bool) -> list:
        """Return n_select crops: uniformly subsampled or last-N tail."""
        n = len(crops)
        if n <= n_select:
            return crops
        if downsample:
            indices = np.round(np.linspace(0, n - 1, n_select)).astype(int)
            return [crops[i] for i in indices]
        return crops[-n_select:]

    def get_drowning_candidates(
        self,
        bbox_history: dict,
        crop_history: dict | None = None,
        head_preds_cache: dict | None = None,
        head_visible_frame_count: int = 16,
        head_visibility_min_frames: int = 10,
        downsample_head_frames: bool = False,
        head_evidence_samples: int = 0,
        frame_history: dict | None = None,
        processed_fps: float | None = None,
    ) -> tuple:
        """
        Classify each track into one of five categories.

        Args:
            bbox_history  (dict): track_id → list[bbox]  ([x1, y1, x2, y2])
            crop_history  (dict | None): track_id → list[np.ndarray] (BGR crops).
                Required when a head_classifier is set; ignored otherwise.
            frame_history (dict | None): track_id → list[int], the frame each bbox was
                detected in (same length as bbox_history[track_id]).
            processed_fps (float | None): rate of those frame numbers. With
                frame_history, enables the swimming-speed gate on the direction
                exclusion.

        Returns:
            tuple:
                candidates (set)                – suspicious tracks kept for inference
                excluded_direction (set)        – excluded: consistent direction
                excluded_direction_change (set) – excluded: lap / reversal
                excluded_head_visible (set)     – excluded: head clearly above water
            Falls back to (all tracks, empty, …) if analysis would exclude everyone.
        """
        t_start = time.perf_counter()
        candidates = set()
        excluded_direction = set()
        excluded_direction_change = set()
        excluded_head_visible = set()
        head_classification_count = 0

        self.head_stats = {}          # track_id → why the head check did or didn't exclude
        self.motion_stats = {}        # track_id → the swimming-speed gate's numbers
        self.slow_linear = set()
        for track_id, bboxes in bbox_history.items():
            n = len(bboxes)
            head_under_frac = None    # under water among DETECTED frames, if known
            # ── Head-visibility first: predictions are pre-computed per-frame, no extra cost ──
            if (
                self.head_classifier is not None
                and head_preds_cache is not None
                and track_id in head_preds_cache
            ):
                # Evidence is the tail of the cache covering the action model's clip:
                # the head verdict should describe the same footage the action model will
                # judge. head_visible_frame_count is that span in processed frames.
                preds = list(head_preds_cache[track_id])[-head_visible_frame_count:]                     if head_visible_frame_count > 0 else list(head_preds_cache[track_id])
                head_under_frac = self._head_under_frac(preds)
                if len(preds) < head_visibility_min_frames:
                    self.head_stats[track_id] = {
                        "checked": False,
                        "reason": f"{len(preds)} predictions < head_visibility_min_frames "
                                  f"({head_visibility_min_frames})",
                        "n_predictions": len(preds),
                    }
                if len(preds) >= head_visibility_min_frames:
                    head_classification_count += 1
                    # By default every prediction in the window counts. Subsampling was a
                    # compute saving from when the classifier ran on demand at decision
                    # time; predictions are now cached per frame, so dropping some only
                    # coarsens the fractions - at 16 samples one frame is 6.25% of a
                    # score compared against a 0.20 bar, a third of the bar per frame.
                    selected_preds = preds
                    if head_evidence_samples and len(preds) > head_evidence_samples:
                        selected_preds = self._select_crops(
                            preds, head_evidence_samples, downsample_head_frames)
                    n_select = len(selected_preds)
                    mean_w, mean_h = 0, 0
                    if crop_history is not None and track_id in crop_history:
                        crops = list(crop_history[track_id])[-head_visible_frame_count:]                             if head_visible_frame_count > 0 else list(crop_history[track_id])
                        if crops:
                            selected_crops = self._select_crops(crops, n_select, downsample_head_frames)
                            mean_w = int(np.mean([c.shape[1] for c in selected_crops]))
                            mean_h = int(np.mean([c.shape[0] for c in selected_crops]))
                    if self._head_excluded(track_id, selected_preds, n_select, excluded_head_visible,
                                           mean_w=mean_w, mean_h=mean_h):
                        continue

            # ── Direction checks ──────────────────────────────────────────────
            if n >= self.MIN_FRAMES_FOR_ANALYSIS_DIRECTION_STATIONARY:
                direction_ready = True
            elif n >= self.MIN_FRAMES_FOR_ANALYSIS_DIRECTION:
                cxs = np.array([(b[0] + b[2]) / 2 for b in bboxes])
                cys = np.array([(b[1] + b[3]) / 2 for b in bboxes])
                disp = float(np.hypot(cxs[-1] - cxs[0], cys[-1] - cys[0]))
                bsz = float(np.median([np.sqrt(max(1, b[2] - b[0]) * max(1, b[3] - b[1])) for b in bboxes]))
                direction_ready = disp > bsz
            else:
                direction_ready = False

            if not direction_ready:
                logger.debug(f"Too few frames ({n}), track_id: {track_id}")
                candidates.add(track_id)
                continue

            if direction_ready and self._is_consistent_direction(bboxes, track_id):
                frames = (frame_history or {}).get(track_id)
                if self._swimming_confirmed(track_id, bboxes, frames, processed_fps, head_under_frac):
                    logger.debug(f"Track {track_id} excluded: consistent directional movement")
                    excluded_direction.add(track_id)
                    continue
                # Straight but too slow (or too short with the head under): drift, not
                # swimming. Kept as a candidate - the direction-change test is not tried,
                # since a straight path is not a lap either.
                self.slow_linear.add(track_id)
                candidates.add(track_id)
                continue

            if n >= self.MIN_FRAMES_FOR_DIRECTION_CHANGE and self._is_direction_change(bboxes):
                logger.debug(f"Track {track_id} excluded: direction change (lap/reversal)")
                excluded_direction_change.add(track_id)
                continue

            candidates.add(track_id)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "BehaviorAnalysis %.1f ms | tracks=%d | candidates=%d | "
            "excl_dir=%d excl_dir_change=%d excl_head=%d | "
            "head_classified=%d",
            elapsed_ms, len(bbox_history), len(candidates),
            len(excluded_direction), len(excluded_direction_change),
            len(excluded_head_visible),
            head_classification_count,
        )
        return candidates, excluded_direction, excluded_direction_change, excluded_head_visible

    def _head_excluded(
        self,
        track_id,
        predictions: list,
        n_crops: int,
        excluded_head_visible: set,
        mean_w: int = 0,
        mean_h: int = 0,
    ) -> bool:
        """Return True and record in excluded_head_visible if head is clearly above water.

        Each entry in predictions is a (class_name, confidence) tuple.
        A prediction only counts toward exclusion if its confidence meets the threshold.
        """
        shoulders_frac = sum(
            1 for cls, conf in predictions
            if cls == HVC.HEAD_AND_SHOULDERS_VISIBLE and conf >= self.HEAD_SHOULDERS_CONF_THRESHOLD
        ) / max(n_crops, 1)
        clearly_frac = sum(
            1 for cls, conf in predictions
            if cls == HVC.HEAD_CLEARLY_VISIBLE and conf >= self.HEAD_CLEARLY_CONF_THRESHOLD
        ) / max(n_crops, 1)

        logger.debug(
            "Head classifier track %s | crops=%d | bbox avg %dx%d | "
            "h&s=%.0f%% (thr=%.0f%%) clearly=%.0f%% (thr=%.0f%%)",
            track_id, n_crops, mean_w, mean_h,
            shoulders_frac * 100, self.HEAD_SHOULDERS_CONF_THRESHOLD * 100,
            clearly_frac * 100, self.HEAD_CLEARLY_CONF_THRESHOLD * 100,
        )

        under_frac = sum(
            1 for cls, _ in predictions if cls == HVC.HEAD_HORIZONTAL_OR_UNDER_WATER
        ) / max(n_crops, 1)
        score = sum(self._frame_weight(c, p) for c, p in predictions) / max(n_crops, 1)
        n_detected = sum(1 for c, _ in predictions if c != NOT_DETECTED)
        coverage = n_detected / max(n_crops, 1)
        stats = {
            "checked": True,
            "mode": self.HEAD_EXCLUSION_MODE,
            "n_frames_used": n_crops,
            "n_detected": n_detected,
            "coverage": round(coverage, 3),
            "bbox_avg": [mean_w, mean_h],
            "weighted_score": round(score, 3),
            "shoulders_frac": round(shoulders_frac, 3),
            "clearly_frac": round(clearly_frac, 3),
            "combined_frac": round(shoulders_frac + clearly_frac, 3),
            "under_frac_any_conf": round(under_frac, 3),
            "bars": {
                "weighted_score": self.HEAD_EXCLUSION_SCORE,
                "shoulders_frac": self.HEAD_SHOULDERS_EXCLUSION_FRAC,
                "clearly_frac": self.HEAD_CLEARLY_VISIBLE_EXCLUSION_FRAC,
                "shoulders_conf": self.HEAD_SHOULDERS_CONF_THRESHOLD,
                "clearly_conf": self.HEAD_CLEARLY_CONF_THRESHOLD,
            },
            # every frame that fed the fractions, so a decision can be replayed offline
            "predictions": [[getattr(c, "value", str(c)), round(float(p), 3)] for c, p in predictions],
            "excluded": False,
            "excluded_by": None,
        }
        stats["bars"]["min_coverage"] = self.HEAD_MIN_COVERAGE
        self.head_stats[track_id] = stats

        # Known and deliberate: when a person is re-acquired under a new track id, the
        # old fragment keeps recording "not detected" placeholders for frames in which
        # that same person WAS detected under the new id. Coverage then falls below the
        # floor and a correct head-visible exclusion is revoked, so a swimmer with his
        # head up the whole time stays a candidate and reaches the action model.
        #
        # Left alone on purpose. Object detection is weak on MJPEG 4K footage and drops
        # many frames of a person who is plainly visible, so an absence cannot be read as
        # a submersion: the same gap is produced by a detector miss and by someone
        # genuinely going under. Erring toward keeping the candidate costs inference time;
        # erring the other way loses a drowning person. Revisit only once detection is
        # reliable on this footage — and note that by then, a person neither the detector
        # nor a human can see underwater is also a person the action model cannot judge,
        # since they may surface anywhere.
        if coverage < self.HEAD_MIN_COVERAGE:
            logger.debug(
                "Track %s not excluded: only %d/%d frames detected (coverage %.2f < %.2f)",
                track_id, n_detected, n_crops, coverage, self.HEAD_MIN_COVERAGE,
            )
            stats["skip_reason"] = (f"coverage {coverage:.2f} < {self.HEAD_MIN_COVERAGE} "
                                    f"({n_detected}/{n_crops} frames detected)")
            return False

        if self.HEAD_EXCLUSION_MODE == "weighted":
            if score >= self.HEAD_EXCLUSION_SCORE:
                logger.debug(
                    "Track %s excluded by weighted head score %.3f >= %.2f "
                    "(h&s %.0f%%, clear %.0f%% of %d frames)",
                    track_id, score, self.HEAD_EXCLUSION_SCORE,
                    shoulders_frac * 100, clearly_frac * 100, n_crops,
                )
                stats["excluded"], stats["excluded_by"] = True, "weighted_score"
                excluded_head_visible.add(track_id)
                return True
            return False

        if shoulders_frac >= self.HEAD_SHOULDERS_EXCLUSION_FRAC:
            logger.debug(
                "Track %s excluded by %s (%.0f%% of %d frames)",
                track_id, HVC.HEAD_AND_SHOULDERS_VISIBLE.value, shoulders_frac * 100, n_crops,
            )
            stats["excluded"], stats["excluded_by"] = True, "shoulders_frac"
            excluded_head_visible.add(track_id)
            return True
        if clearly_frac >= self.HEAD_CLEARLY_VISIBLE_EXCLUSION_FRAC:
            logger.debug(
                "Track %s excluded by %s (%.0f%% of %d frames)",
                track_id, HVC.HEAD_CLEARLY_VISIBLE.value, clearly_frac * 100, n_crops,
            )
            stats["excluded"], stats["excluded_by"] = True, "clearly_frac"
            excluded_head_visible.add(track_id)
            return True
        return False

    def _head_under_frac(self, predictions: list) -> float | None:
        """Fraction of DETECTED frames rated head-horizontal-or-under-water.

        Undetected frames are left out: they are neither evidence of a head above water
        nor of one below it. None when too few frames were detected to say anything.
        """
        detected = [c for c, _ in predictions if c != NOT_DETECTED]
        if len(detected) < self.MIN_FRAMES_FOR_HEAD_CLASSIFICATION:
            return None
        return sum(1 for c in detected if c == HVC.HEAD_HORIZONTAL_OR_UNDER_WATER) / len(detected)

    def _swimming_confirmed(self, track_id, bboxes: list, frames: list | None,
                            processed_fps: float | None, head_under_frac: float | None) -> bool:
        """Is a track that passed the direction test fast and far enough to be swimming?

        Returns True (exclusion stands) when the gate cannot be evaluated - no frame
        numbers, no fps, or frame numbers that do not line up with the bboxes.
        """
        if not frames or not processed_fps or len(frames) != len(bboxes) or frames[-1] <= frames[0]:
            return True

        b0, b1 = bboxes[0], bboxes[-1]
        dx = (b1[0] + b1[2] - b0[0] - b0[2]) / 2
        dy = (b1[1] + b1[3] - b0[1] - b0[3]) / 2
        disp = float(np.hypot(dx, dy))
        if disp < 1e-6:
            return True               # cannot happen after the direction test; nothing to judge
        ux, uy = abs(dx) / disp, abs(dy) / disp
        extent = float(np.median([ux * max(1.0, b[2] - b[0]) + uy * max(1.0, b[3] - b[1])
                                  for b in bboxes]))
        disp_bsz = disp / extent
        elapsed_s = (frames[-1] - frames[0]) / processed_fps
        speed = disp_bsz / elapsed_s

        head_under = head_under_frac is not None and head_under_frac >= self.HEAD_UNDER_STRICT_FRAC
        reason = None
        if speed < self.MIN_SWIM_SPEED_BBOX_PER_S:
            reason = f"speed {speed:.2f} < {self.MIN_SWIM_SPEED_BBOX_PER_S} extents/s"
        elif head_under and disp_bsz < self.HEAD_UNDER_MIN_DISPLACEMENT_BBOX:
            reason = (f"head under water in {head_under_frac:.0%} of detected frames and "
                      f"displacement {disp_bsz:.2f} < {self.HEAD_UNDER_MIN_DISPLACEMENT_BBOX} extents")

        self.motion_stats[track_id] = {
            "displacement_bbox": round(disp_bsz, 3),     # in extents along travel
            "extent_px": round(extent, 1),
            "elapsed_s": round(elapsed_s, 2),
            "speed_bbox_per_s": round(speed, 3),
            "head_under_frac_detected": None if head_under_frac is None else round(head_under_frac, 3),
            "bars": {
                "min_speed_bbox_per_s": self.MIN_SWIM_SPEED_BBOX_PER_S,
                "head_under_strict_frac": self.HEAD_UNDER_STRICT_FRAC,
                "head_under_min_displacement_bbox": self.HEAD_UNDER_MIN_DISPLACEMENT_BBOX,
            },
            "swimming_confirmed": reason is None,
            "kept_because": reason,
        }
        if reason:
            logger.debug(f"Track {track_id} linear but kept as candidate: {reason}")
        return reason is None

    def _is_consistent_direction(self, bboxes: list, track_id:str) -> bool:
        """
        Return True if centroid motion is consistently directed (linear swimmer).

        Approach
        --------
        The centroid trajectory is analyzed to determine whether motion shows
        persistent progress in one dominant direction, which is characteristic
        of intentional swimming.

        A PCA-based dominant motion axis is estimated from the centroid cloud.
        Consecutive centroid steps are then projected onto this axis and checked
        for directional consistency.

        Key idea:
        - Intentional swimmers:
            mostly advance in one direction
            → projected steps keep the same sign
        - Wave drift / oscillation:
            frequently reverses direction
            → projected steps alternate sign

        Steps
        -----
        1. Compute raw centroid trajectory from bbox history.
        2. Compute raw total displacement (unsmoothed) for minimum-motion gating.
        3. Estimate person scale using the median geometric-mean bbox size.
        4. Smooth centroid trajectory to suppress detector jitter.
        5. Fit PCA to the smoothed centroid cloud to obtain dominant motion axis.
        6. Project centroid steps onto that axis.
        7. Remove tiny projected motions caused by bbox jitter.
        8. Compute directional consistency:

            abs(sum(projected_steps)) / sum(abs(projected_steps))

        This measures how much motion contributed to net forward progress.
        Values near 1 indicate persistent motion in one direction.

        Returns
        -------
        bool
            True if motion is consistently directional and therefore likely
            corresponds to a normal swimmer rather than a drowning candidate.
        """

        # ------------------------------------------------------------------
        # Raw centroids (used for physical displacement measurement)
        # ------------------------------------------------------------------
        raw_cx = np.array([(b[0] + b[2]) / 2.0 for b in bboxes], dtype=np.float32)
        raw_cy = np.array([(b[1] + b[3]) / 2.0 for b in bboxes], dtype=np.float32)

        if len(raw_cx) < 2:
            return False

        # ------------------------------------------------------------------
        # Minimum displacement guard (use RAW trajectory, not smoothed)
        # ------------------------------------------------------------------
        total_displacement = float(
            np.hypot(raw_cx[-1] - raw_cx[0],
                    raw_cy[-1] - raw_cy[0])
        )

        # Robust person-scale estimate
        bbox_sizes = []
        for b in bboxes:
            w = max(1.0, b[2] - b[0])
            h = max(1.0, b[3] - b[1])
            bbox_sizes.append(np.sqrt(w * h))

        mean_bbox_size = float(np.median(bbox_sizes))
        if total_displacement < (
            self.MIN_DISPLACEMENT_BBOX_FRACTION * mean_bbox_size
        ):
            return False

        
        # ------------------------------------------------------------------
        # Smooth trajectory to suppress bbox jitter
        # ------------------------------------------------------------------
        cx = raw_cx.copy()
        cy = raw_cy.copy()

        window = min(self.SMOOTHING_WINDOW, len(cx) // 2)

        if window >= 2:
            # Median smoothing with edge-padding: suppresses detector jitter
            # without smearing large positional jumps across adjacent frames
            # (mean smoothing would spread a sudden jump over ~window frames,
            # making every neighbouring step look directionally biased).
            pad_l = (window - 1) // 2
            pad_r = window - 1 - pad_l
            cx_pad = np.concatenate([[cx[0]] * pad_l, cx, [cx[-1]] * pad_r])
            cy_pad = np.concatenate([[cy[0]] * pad_l, cy, [cy[-1]] * pad_r])
            cx = np.array(
                [np.median(cx_pad[i: i + window]) for i in range(len(cx))],
                dtype=np.float32,
            )
            cy = np.array(
                [np.median(cy_pad[i: i + window]) for i in range(len(cy))],
                dtype=np.float32,
            )

        # ------------------------------------------------------------------
        # Path crookedness: max perpendicular deviation from the start→end
        # line, normalised by total displacement.  A swimmer stays close
        # to the direct line (low crookedness); an L-shape or V-shape
        # deviates by roughly half the shorter arm (~0.5+).
        # ------------------------------------------------------------------
        crookedness = 0.0
        seg = np.array([cx[-1] - cx[0], cy[-1] - cy[0]])
        seg_len = float(np.linalg.norm(seg))
        if seg_len > 1e-6:
            seg_unit = seg / seg_len
            perp = np.array([-seg_unit[1], seg_unit[0]])
            pts_2d = np.column_stack([cx, cy])
            lateral = np.abs((pts_2d - np.array([cx[0], cy[0]])) @ perp)
            crookedness = float(lateral.max() / seg_len)
            if crookedness > self.PATH_CROOKEDNESS_THRESHOLD:
                logger.debug(f'crookedness:{crookedness} | self.PATH_CROOKEDNESS_THRESHOLD:{self.PATH_CROOKEDNESS_THRESHOLD}')
                return False

        # ------------------------------------------------------------------
        # PCA: dominant motion direction
        # ------------------------------------------------------------------
        points = np.column_stack([cx, cy])

        centered = points - points.mean(axis=0)

        cov = np.cov(centered.T)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Principal eigenvector = dominant motion axis
        main_dir = eigenvectors[:, np.argmax(eigenvalues)]

        # Align PCA direction with overall displacement
        overall_motion = np.array([
            cx[-1] - cx[0],
            cy[-1] - cy[0]
        ])

        if np.dot(main_dir, overall_motion) < 0:
            main_dir = -main_dir

        # ------------------------------------------------------------------
        # Signed projected step motion
        # ------------------------------------------------------------------
        steps = np.diff(points, axis=0)

        proj_steps = steps @ main_dir

        eps = 0.02 * mean_bbox_size

        # ------------------------------------------------------------------
        # Minimum active-step fraction: if fewer than MIN_ACTIVE_STEP_FRACTION
        # of all steps have a meaningful projected magnitude, the person was
        # mostly stationary and should not be excluded as a swimmer.
        # ------------------------------------------------------------------
        active_fraction = float(np.sum(np.abs(proj_steps) > eps) / len(proj_steps))
        if active_fraction < self.MIN_ACTIVE_STEP_FRACTION:
            # Fallback: "stationary start → linear tail" — person was still, then swam
            return self._is_linear_tail(cx, cy, mean_bbox_size)

        # Path linearity: fraction of variance explained by the first PC.
        # A curved or arch-shaped path has significant variance on both axes
        # (PVE near 0.5), while a straight swimmer has PVE near 1.0.
        # Reject non-linear paths before the consistency test so that arch
        # paths whose steps happen to project consistently (e.g. always
        # moving right along the x-axis of an arch) are not wrongly excluded.
        # ------------------------------------------------------------------
        total_variance = float(eigenvalues.sum())
        pve = float(eigenvalues.max() / total_variance) if total_variance > 0 else 1.0

        # Moderately crooked path + significant inactivity → not a swimmer.
        # Exception: skip when PVE is high AND active fraction is sufficient —
        # a zigzag swimmer's lateral oscillations inflate crookedness even though
        # the overall path is genuinely linear, but they still have clearly
        # active steps (>= 0.65 fraction).  A slow-drifting person has both
        # moderate crookedness AND low active fraction, so it is kept as candidate.
        if crookedness > self.PATH_CROOKEDNESS_SOFT_THRESHOLD and active_fraction < 0.90:
            if active_fraction < self.TAIL_MIN_ACTIVE_FRACTION or pve < self.PATH_LINEARITY_THRESHOLD:
                return False

        if pve < self.PATH_LINEARITY_THRESHOLD:
            logger.debug(f'pve:{pve}, PATH_LINEARITY_THRESHOLD:{self.PATH_LINEARITY_THRESHOLD}')
            return self._is_linear_tail(cx, cy, mean_bbox_size)

        # ------------------------------------------------------------------
        # Projection cluster ratio: if most centroid positions project into
        # the bottom quartile of the projection range, the person spent the
        # majority of the track stationary near the start and only moved in
        # a brief final burst — not characteristic of sustained swimming.
        # ------------------------------------------------------------------
        proj_positions = points @ main_dir
        proj_range = float(proj_positions.max() - proj_positions.min())
        if proj_range > 1e-6:
            q1 = float(proj_positions.min()) + 0.25 * proj_range
            cluster_ratio = float(np.sum(proj_positions < q1) / len(proj_positions))
            if cluster_ratio > self.PROJ_CLUSTER_THRESHOLD:
                return self._is_linear_tail(cx, cy, mean_bbox_size)

        # ------------------------------------------------------------------
        # Angular consistency: coherence of 2D step directions for active steps.
        # Values near 1 mean all steps point roughly the same way (straight swimmer).
        # Low value + partial inactivity → erratic / multi-direction movement.
        # ------------------------------------------------------------------
        active_step_vecs = steps[np.abs(proj_steps) > eps]
        if len(active_step_vecs) > 0:
            norms = np.linalg.norm(active_step_vecs, axis=1, keepdims=True)
            unit_vecs = active_step_vecs / (norms + 1e-9)
            angular_consistency = float(np.linalg.norm(unit_vecs.mean(axis=0)))
        else:
            angular_consistency = 1.0
        logger.debug(f"angular_consistency:{angular_consistency:.3f} | active_fraction:{active_fraction:.3f}")
        # Near-random step directions (circular/chaotic motion): active_fraction cannot
        # rescue a track whose steps point in all directions.  The PVE guard excludes
        # reversal tracks (go-left-then-right), which also have low angular consistency
        # but very high PVE (all motion on one axis) and are blocked later by the
        # consistency check.
        if angular_consistency < self.ANGULAR_CONSISTENCY_FLOOR and pve < self.ANGULAR_FLOOR_MAX_PVE:
            return self._is_linear_tail(cx, cy, mean_bbox_size)
        if angular_consistency < self.ANGULAR_CONSISTENCY_THRESHOLD and active_fraction < 0.75:
            return self._is_linear_tail(cx, cy, mean_bbox_size)

        # ------------------------------------------------------------------
        # Peak-reversal check: if cumulative projected displacement drops
        # significantly from its peak, the person reversed direction rather
        # than swimming through.  A pure swimmer's cumulative grows
        # monotonically; a path that goes forward then turns back will have
        # (final - peak) / peak below -PATH_REVERSAL_THRESHOLD.
        # ------------------------------------------------------------------
        cum = np.cumsum(proj_steps)
        peak = float(cum.max())
        if peak > eps:
            reversal = (float(cum[-1]) - peak) / peak
            if reversal < -self.PATH_REVERSAL_THRESHOLD:
                return False

        # ------------------------------------------------------------------
        # Directional consistency: fraction of active steps that advance
        # forward.  Counting by number (not magnitude) prevents large early
        # zigzags from dominating the score.
        # ------------------------------------------------------------------
        proj_steps_active = proj_steps[np.abs(proj_steps) > eps]
        consistency = float(np.sum(proj_steps_active > 0) / len(proj_steps_active))
        logger.debug(f"consistency:{consistency}")

        # ------------------------------------------------------------------
        # Active step size: if the mean projected magnitude of active steps is
        # tiny relative to bbox size, the "movement" is detector noise, not
        # purposeful swimming (genuine swimmers cover meaningful ground each step).
        # ------------------------------------------------------------------
        mean_active_proj = float(np.mean(np.abs(proj_steps_active)))
        logger.debug(f"mean_active_proj:{mean_active_proj:.3f} ({mean_active_proj/mean_bbox_size:.4f} bsz)")
        if mean_active_proj < self.MIN_ACTIVE_PROJ_STEP_SIZE * mean_bbox_size:
            return False

        # ------------------------------------------------------------------
        # Fragmented active runs: a genuine swimmer has a dominant continuous
        # active period; wave-drifted clustering shows many short bursts with
        # no single run accounting for a meaningful fraction of active steps.
        # Guard requires imperfect consistency to avoid rejecting long but
        # occasionally-interrupted tracks (cons≈1.0 means every active step
        # is forward, regardless of run structure).
        # ------------------------------------------------------------------
        active_run_lengths = []
        cur = 0
        for v in (np.abs(proj_steps) > eps):
            if v:
                cur += 1
            else:
                if cur > 0:
                    active_run_lengths.append(cur)
                    cur = 0
        if cur > 0:
            active_run_lengths.append(cur)
        if active_run_lengths:
            max_run_frac = max(active_run_lengths) / len(proj_steps_active)
            if max_run_frac < self.MAX_RUN_FRACTION and consistency < self.FRAGMENTED_RUN_CONSISTENCY_THRESHOLD:
                return False

        # ------------------------------------------------------------------
        # End-concentrated inactive steps: a person who swam then stopped will
        # have a large fraction of their inactive steps clustered at the tail.
        # Skip this check if the track ENDS with movement that is comparable
        # in scale to normal swimming steps — a mid-track pause followed by
        # genuine resumed swimming is still consistent direction.  Small-scale
        # oscillations at the end (stopped jitter) do NOT qualify as "resumed".
        # ------------------------------------------------------------------
        inactive_mask = np.abs(proj_steps) <= eps
        total_inactive = int(inactive_mask.sum())
        if total_inactive / len(proj_steps) > 0.10:
            last_few_n = max(3, round(len(proj_steps) * 0.10))
            last_few_proj_mean = float(np.mean(np.abs(proj_steps[-last_few_n:])))
            tail_n = max(1, round(len(proj_steps) * 0.30))
            inactive_in_tail = int(inactive_mask[-tail_n:].sum())
            inactive_before_tail = total_inactive - inactive_in_tail
            end_inactive_frac = inactive_in_tail / total_inactive
            # Path 1: high-consistency swimmer ending with active steps
            path1 = (
                consistency >= self.SUSTAINED_CONSISTENCY_THRESHOLD
                and last_few_proj_mean > 0.30 * mean_active_proj
            )
            # Path 2: established rhythm + active end + sufficient overall activity
            # (handles swimmers with consistency just below 0.95)
            path2 = (
                inactive_before_tail >= self.MIN_ESTABLISHED_PAUSES
                and end_inactive_frac < self.END_INACTIVE_FRAC_THRESHOLD
                and last_few_proj_mean > 0.30 * mean_active_proj
                and active_fraction > 0.65
            )
            # Path 3: established rhythm + genuinely quiet end (track ended mid-pause)
            # last_few small confirms a clean pause rather than stopped-swimmer jitter.
            # Uses <= 0.50 (not strict <) because consistency >= 0.90 is the real gate.
            path3 = (
                inactive_before_tail >= self.MIN_ESTABLISHED_PAUSES
                and end_inactive_frac <= 0.50
                and last_few_proj_mean <= self.BREASTSTROKE_PAUSE_RATIO_MAX * mean_active_proj
                and consistency >= 0.90
            )
            # Path 4: high-consistency swimmer (≥ 0.90) ending with moderate activity.
            # Covers the gap between path3 (quiet end) and path1 (very high consistency):
            # a zig-zag or oscillating swimmer whose last steps are at partial intensity
            # but consistency is clearly above the 0.90 threshold.
            # Strict < 0.50 (not <=): if exactly half the pauses cluster at the end
            # that is already too many to trust as "still swimming".
            path4 = (
                consistency >= 0.90
                and end_inactive_frac < 0.50
                and last_few_proj_mean > 0.30 * mean_active_proj
                and active_fraction > 0.65
            )
            # Path 5: high-consistency swimmer whose track ends very quietly (≤ 0.35×
            # mean active step) — characteristic of a swimmer who reached the pool wall
            # and slowed to a stop.  Tail must also be less than half inactive so a
            # long stopped-then-barely-moved pattern does not qualify.
            path5 = (
                consistency >= 0.90
                and last_few_proj_mean <= 0.35 * mean_active_proj
                and active_fraction > 0.65
                and inactive_in_tail / tail_n < 0.50
            )
            ends_with_movement = inactive_before_tail >= 2 and (path1 or path2 or path3 or path4 or path5)
            if not ends_with_movement:
                logger.debug(f"end_inactive_frac:{end_inactive_frac:.3f} total_inactive:{total_inactive}")
                if end_inactive_frac > self.END_INACTIVE_FRAC_THRESHOLD:
                    return False

        return consistency > self.DIRECTION_CONSISTENCY_THRESHOLD

    def _is_linear_tail(self, cx: np.ndarray, cy: np.ndarray, mean_bbox_size: float) -> bool:
        """
        Detect 'stationary start → linear swimming tail' pattern.

        Scans forward from the beginning to find the earliest onset index where
        the remaining trajectory (tail) satisfies:
          - end-to-end displacement > TAIL_MIN_DISPLACEMENT_FACTOR * mean_bbox_size
          - active step fraction > TAIL_MIN_ACTIVE_FRACTION
          - directional consistency > DIRECTION_CONSISTENCY_THRESHOLD

        This handles the case where a swimmer was stationary then began moving
        linearly: the full-track active_fraction is diluted by the stationary
        prefix, but the tail alone is clearly directional.

        The tail must contain at least 30% of the total steps so a brief
        end-of-track burst never triggers the exclusion.
        """
        n = len(cx)
        eps = 0.02 * mean_bbox_size
        min_tail_steps = max(4, round((n - 1) * 0.30))

        for onset in range(1, n - min_tail_steps):
            tail_cx = cx[onset:]
            tail_cy = cy[onset:]

            tail_disp = float(np.hypot(tail_cx[-1] - tail_cx[0], tail_cy[-1] - tail_cy[0]))
            if tail_disp < self.TAIL_MIN_DISPLACEMENT_FACTOR * mean_bbox_size:
                continue

            tail_steps = np.diff(np.column_stack([tail_cx, tail_cy]), axis=0)
            step_mags = np.linalg.norm(tail_steps, axis=1)
            tail_active_frac = float(np.sum(step_mags > eps) / len(step_mags))
            if tail_active_frac < self.TAIL_MIN_ACTIVE_FRACTION:
                continue

            # Prefix must be genuinely stationary: large oscillations (e.g. >0.15*bbox
            # per step) indicate erratic in-place struggle, not a quiet stationary phase.
            prefix_steps = np.diff(np.column_stack([cx[:onset + 1], cy[:onset + 1]]), axis=0)
            if len(prefix_steps) > 0:
                prefix_mean_step = float(np.linalg.norm(prefix_steps, axis=1).mean())
                if prefix_mean_step > self.TAIL_MAX_PREFIX_STEP_FRACTION * mean_bbox_size:
                    continue

            # PCA on tail to get the dominant direction of the tail alone
            tail_pts = np.column_stack([tail_cx, tail_cy])
            if np.std(tail_cx) < 1e-6 and np.std(tail_cy) < 1e-6:
                continue
            centered = tail_pts - tail_pts.mean(axis=0)
            evals, evecs = np.linalg.eigh(np.cov(centered.T))
            tail_dir = evecs[:, np.argmax(evals)]
            overall = np.array([tail_cx[-1] - tail_cx[0], tail_cy[-1] - tail_cy[0]])
            if np.dot(tail_dir, overall) < 0:
                tail_dir = -tail_dir

            tail_proj = tail_steps @ tail_dir
            tail_active_proj = tail_proj[np.abs(tail_proj) > eps]
            if len(tail_active_proj) < 3:
                continue

            # 2D angular coherence of the tail's active steps.  Projecting onto
            # the best PCA direction can make random movement look consistent;
            # this check ensures the raw step vectors genuinely point the same way.
            active_mask = step_mags > eps
            tail_unit = tail_steps / (step_mags[:, None] + 1e-9)
            tail_angular_cons = float(np.linalg.norm(tail_unit[active_mask].mean(axis=0)))
            if tail_angular_cons < self.TAIL_ANGULAR_CONSISTENCY_THRESHOLD:
                continue

            tail_consistency = float(np.sum(tail_active_proj > 0) / len(tail_active_proj))
            if tail_consistency > self.DIRECTION_CONSISTENCY_THRESHOLD:
                logger.debug(
                    f"tail_linear: onset={onset}/{n-1} "
                    f"tail_disp={tail_disp:.1f} "
                    f"tail_active={tail_active_frac:.2f} "
                    f"tail_cons={tail_consistency:.2f}"
                )
                return True

        return False

    def _is_direction_change(self, bboxes: list) -> bool:
        """
        Return True if the track shows a lap/reversal pattern: person swims in
        one direction, turns around, and swims back.  Both legs must cover at
        least mean_bbox_size in displacement so a simple stop-at-end never
        qualifies.
        """
        raw_cx = np.array([(b[0] + b[2]) / 2.0 for b in bboxes], dtype=np.float32)
        raw_cy = np.array([(b[1] + b[3]) / 2.0 for b in bboxes], dtype=np.float32)
        mean_bbox_size = float(np.median([
            np.sqrt(max(1, b[2] - b[0]) * max(1, b[3] - b[1])) for b in bboxes
        ]))

        cx, cy = raw_cx.copy(), raw_cy.copy()
        window = min(self.SMOOTHING_WINDOW, len(cx) // 2)
        if window >= 2:
            pad_l = (window - 1) // 2
            pad_r = window - 1 - pad_l
            cx_pad = np.concatenate([[cx[0]] * pad_l, cx, [cx[-1]] * pad_r])
            cy_pad = np.concatenate([[cy[0]] * pad_l, cy, [cy[-1]] * pad_r])
            cx = np.array([np.median(cx_pad[i:i + window]) for i in range(len(cx))], dtype=np.float32)
            cy = np.array([np.median(cy_pad[i:i + window]) for i in range(len(cy))], dtype=np.float32)

        points = np.column_stack([cx, cy])
        steps = np.diff(points, axis=0)
        path_length = float(np.sum(np.linalg.norm(steps, axis=1)))
        net_displacement = float(np.linalg.norm(points[-1] - points[0]))

        if path_length < 2 * mean_bbox_size:
            return False

        # Net displacement must be small — person must have reversed
        if net_displacement > self.DIRECTION_CHANGE_REVERSAL_RATIO * path_length:
            return False

        # Find turning point: index in points where cumulative PCA projection peaks
        cov = np.cov((points - points.mean(axis=0)).T)
        vals, vecs = np.linalg.eigh(cov)
        main_dir = vecs[:, np.argmax(vals)]
        proj = steps @ main_dir
        cum_proj = np.cumsum(proj)
        turn_idx = int(np.argmax(np.abs(cum_proj))) + 1  # index into points

        # Each leg needs at least 3 steps (4 points)
        min_pts = 4
        if turn_idx < min_pts or turn_idx > len(points) - min_pts:
            return False

        seg1 = points[:turn_idx + 1]
        seg2 = points[turn_idx:]

        def _seg(seg):
            disp = float(np.linalg.norm(seg[-1] - seg[0]))
            seg_steps = np.diff(seg, axis=0)
            plen = float(np.sum(np.linalg.norm(seg_steps, axis=1)))
            linearity = disp / plen if plen > 0 else 0.0
            return disp, linearity, seg[-1] - seg[0]

        disp1, lin1, dir1 = _seg(seg1)
        disp2, lin2, dir2 = _seg(seg2)

        # Both legs must exceed mean_bbox_size
        if disp1 < mean_bbox_size or disp2 < mean_bbox_size:
            return False

        # Both legs must be reasonably linear
        if lin1 < self.DIRECTION_CHANGE_LEG_LINEARITY or lin2 < self.DIRECTION_CHANGE_LEG_LINEARITY:
            return False

        # Legs must point in roughly opposite directions
        n1, n2 = np.linalg.norm(dir1), np.linalg.norm(dir2)
        if n1 < 1e-6 or n2 < 1e-6:
            return False
        cos_angle = float(np.dot(dir1 / n1, dir2 / n2))
        if cos_angle > self.DIRECTION_CHANGE_COS_THRESHOLD:
            return False

        return True
