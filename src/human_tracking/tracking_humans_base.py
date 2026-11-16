import cv2
import json
import logging
import numpy as np
import os
import threading
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger()

from human_tracking.cluster_humans import ClusterHumans
from human_tracking.object_tracker import ObjectTracker
from human_tracking.behavior_analysis import BehaviorAnalyses, BehaviorCategory
from human_tracking.inference_classification_tensorrt import HeadVisibilityTRT, CLASS_NAMES as HEAD_CLASS_NAMES
from human_tracking.head_visibility_classes import CLASS_SHORT_NAMES, NOT_DETECTED, class_color_bgr
from utils import resize_big_frame, get_adjusted_size


class TrackingHumansBase:
    def_video_width = 1920
    def_img_area = 1920 * 1080
    threshold_area_small_perc = 62500 / def_img_area
    threshold_area_big_perc = 150000 / def_img_area
    threshold_huge_size_perc = 1200 / def_img_area
    min_object_size_perc = 10 / def_video_width
    # How much of one cluster must lie inside another before the two are merged.
    # These were permissive for small boxes (0.33) because one person split across
    # several track ids produced several small clusters, and merging them back was the
    # only way to get one crop per person. Track stitching now does that at the source,
    # using time-disjointness and motion, so the only merges still wanted are
    # near-duplicate crops. A low bar now mostly fuses DIFFERENT swimmers into one crop,
    # which forces the action model to call drowning on a picture that also contains a
    # normal swimmer. Merging is chained and each merge enlarges the box, so at 0.33 a
    # row of swimmers could snowball into one cluster.
    threshold_overlap_small = float(os.getenv("CLUSTER_OVERLAP_SMALL", "0.85"))
    threshold_overlap_mid = float(os.getenv("CLUSTER_OVERLAP_MID", "0.90"))
    threshold_overlap_big = float(os.getenv("CLUSTER_OVERLAP_BIG", "0.95"))
    classes = 0
    # Detector thresholds. FP-tolerant use case: a missed person is far worse than an extra
    # candidate, so these sit low; ByteTrack's own thresholds (custom_tracker.yaml) decide
    # which detections may START a track.
    iou = float(os.getenv("DET_IOU", "0.5"))     # NMS IoU
    conf = float(os.getenv("DET_CONF", "0.3"))   # minimum detection confidence
    tile_rows = 1
    tile_cols = 1
    water_class = 1
    trajectory_deque_multiplier = 1.5

    def __init__(
        self,
        human_model_path,
        threshold_frames=30,
        num_total_crops=12,
        skip_water_process_frames=10,
        max_water_deque_size=50,
        tracking_deque_length=12,
        imgsz=960,
        detect_water=True,
        min_bbox_size=None,
        head_classifier_path=None,
        head_classifier_max_batch_size=64,
        head_visibility_img_size=224,
        head_visible_frame_count=16,
        head_visibility_min_frames=10,
        head_visibility_sample_rate=2,
        downsample_head_frames=True,
        head_visibility_min_bbox_size=30,
        use_behavior_analysis: bool | None = None,
    ):
        """
        Initializes the TrackingHumansBase instance.

        Args:
            human_model_path (str): Path to the YOLO model for human detection.
            threshold_frames (int, optional): Minimum number of frames an object must be tracked to be considered valid. Default is 30.
            num_total_crops (int, optional): Total number of crops for clustering. Default is 12.
            skip_water_process_frames (int, optional): Number of frames to skip between water mask updates. Default is 10.
            max_water_deque_size (int, optional): Maximum number of water masks to keep for majority voting. Default is 50.
            tracking_deque_length (int, optional): Length of the tracking deque for object tracking. Default is 12.
            imgsz (int, optional): Image size for detection. Default is 960.
            detect_water (bool, optional): Whether to enable water detection. Default is True.
            min_bbox_size (int, optional): Minimum bounding box size. Default is 182.
        """
        # Floor on a person's box, purely against degenerate crops. It used to be 224,
        # the action model's input size, on the reasoning that cropping smaller wastes an
        # upscale. That reasoning does not hold: resizing a 100x100 crop to 224 keeps
        # every pixel of the person, while cropping 224 around them keeps the same person
        # pixels PLUS background — and in a pool that background is other swimmers. A far
        # away person was being framed at 16x their own area, so the crop handed to the
        # action model could contain several people around the candidate.
        # Framing is now done per cluster in _frame_clusters, proportionally.
        self.min_bbox_size = (min_bbox_size if min_bbox_size is not None
                              else int(os.getenv("CLUSTER_MIN_SIZE", "64")))
        # Crop framing: a margin of background on ALL FOUR sides, as a fraction of the
        # people's pixel size. A fraction rather than a pixel count so a distant swimmer
        # and a close one are framed alike; equal on every side because a limb leaves its
        # detection box in any direction, not preferentially along the longer axis.
        # Content-aware: it grows toward _MAX but stops short of pulling another detected
        # person into the crop, because a second swimmer in a clip labelled "drowning" is
        # label noise - the label no longer says which person it describes.
        self.cluster_pad_margin = float(os.getenv("CLUSTER_PAD_MARGIN_MIN", "0.15"))
        self.cluster_pad_margin_max = float(os.getenv("CLUSTER_PAD_MARGIN_MAX", "0.25"))
        self.cluster_pad_avoid_humans = os.getenv("CLUSTER_PAD_AVOID_HUMANS", "True").lower() == "true"
        # Off by default. Every action model preprocesses with ResizeAndPadVideo, which
        # scales the longer side and letterboxes the rest with black, so the crop is
        # never squashed. Padding the short side here instead would fill that same gap
        # with real pixels - which in a pool means other swimmers - and would differ from
        # what the model was trained through. Set it only to match a model that stretches.
        self.cluster_aspect = float(os.getenv("CLUSTER_ASPECT", "0"))
        self.detect_water = detect_water
        self.num_total_crops = num_total_crops
        self.counter_process_water = 0
        self.skip_water_process_frames = skip_water_process_frames
        self.threshold_frames = threshold_frames
        self.trajectory_deque_size = int(tracking_deque_length * TrackingHumansBase.trajectory_deque_multiplier)
        # How long a track survives without being detected, in processed frames. Matched
        # to the tracking deque, which is the window the action model reasons over: a
        # track whose last detection is older than that can no longer contribute to any
        # cluster, so keeping it longer would only produce ghosts.
        self.track_retention_frames = tracking_deque_length
        logger.info(f"Trajectories initialized with size: {self.trajectory_deque_size}")
        self.tracked_objects = {}
        self.behavior_categories = {
            BehaviorCategory.CANDIDATE: set(),
            BehaviorCategory.EXCLUDED_DIRECTION: set(),
            BehaviorCategory.EXCLUDED_DIRECTION_CHANGE: set(),
            BehaviorCategory.EXCLUDED_HEAD_VISIBLE: set(),
        }
        # candidates that moved straight but too slowly to be swimming (shown as "slow")
        self.behavior_slow_linear = set()
        self.trajectories = {}
        self.behavior_bbox_history = {}
        self.behavior_crop_history = {}
        self.behavior_head_preds_cache = {}   # track_id → deque of per-frame predictions
        # Head-visibility evidence from the LAST behaviour analysis window, as one
        # (run_id, {track_id: stats}) tuple — see BehaviorAnalyses._head_excluded.
        # run_id increments once per window so a reader on another thread can consume
        # each window exactly once instead of once per frame. It is ONE attribute on
        # purpose: two (counter, dict) assignments would let a reader pair the new dict
        # with the old id and count a window twice.
        self.behavior_head_snapshot = (0, {})
        self.behavior_run_id = 0
        self.head_classifier_max_batch_size = head_classifier_max_batch_size

        self.head_classifier = None
        self.head_visible_frame_count = head_visible_frame_count
        self.head_visibility_min_frames = head_visibility_min_frames
        self.head_visibility_sample_rate = max(1, head_visibility_sample_rate)
        self.downsample_head_frames = downsample_head_frames
        self.head_visibility_min_bbox_size = head_visibility_min_bbox_size
        # Head-visibility evidence is configured in SECONDS, not frames. The pipeline
        # samples frames at a rate derived from the source fps, and that derivation is
        # not fps-invariant: 4 fps footage is processed at 4 fps while 30 fps footage is
        # processed at ~6, and 29.97 vs 30.00 differ by 25% through integer truncation.
        # A fixed frame count therefore meant anywhere from 0.8 s to 2.0 s of evidence
        # depending on the camera.
        #
        # The window matches the action model's clip (num_frames * sampling_rate / 30,
        # 5.33 s for swin_t / x3d_m): the head verdict should describe the same footage
        # the action model is about to judge.
        self.head_evidence_window_s = float(os.getenv("HEAD_EVIDENCE_WINDOW_SECONDS", "5.5"))
        self.head_min_evidence_s = float(os.getenv("HEAD_VISIBILITY_MIN_SECONDS",
                                                   str(self.head_evidence_window_s / 2)))
        # 0 = use every prediction in the window (see BehaviorAnalyses).
        self.head_evidence_samples = int(os.getenv("HEAD_EVIDENCE_SAMPLES", "0"))
        # Frames actually handed to the pipeline per second. Set from the real sample
        # interval by set_video_metadata; this is the value the standard pipeline lands
        # on for almost every source rate, used until a caller supplies the exact one.
        self.processed_fps = float(os.getenv("PROCESSED_FPS_FALLBACK", "6.0"))
        if head_classifier_path:
            self.head_classifier = HeadVisibilityTRT(
                head_classifier_path,
                max_batch_size=head_classifier_max_batch_size,
                img_size=head_visibility_img_size,
            )
            logger.info("HeadVisibilityTRT loaded from %s", head_classifier_path)
        self.trajectories_lock = threading.Lock()
        self.merged_clusters = {}
        self.clusters = {}
        self.tracking_results = None
        self.frame_counter = 0
        self.water_mask_bool = None
        self.water_mask_3c = None
        self.water_mask_lock = threading.Lock()
        self.water_deque = deque(maxlen=max_water_deque_size)
        self._save_behavior_jsons = os.getenv("SAVE_BEHAVIOR_JSONS", "False").lower() == "true"
        # draw the per-frame head-visibility class as an inner box inside each person box
        self._visualize_head_visibility = (
            os.getenv("VISUALIZE_HEAD_VISIBILITY", "False").lower() == "true"
        )
        # How long a missing person keeps being recorded as "not detected" before the
        # track is assumed gone (left the zone / lost) rather than submerged. In seconds
        # for the same reason as the rest: the frame-based version meant 0.8s of absence
        # on one camera and 2s on another. HEAD_MAX_MISS_PLACEHOLDERS still overrides it
        # with an exact frame count.
        self._head_max_miss_s = float(os.getenv("HEAD_MAX_MISS_SECONDS", "2.0"))
        self._head_max_miss_override = (int(os.getenv("HEAD_MAX_MISS_PLACEHOLDERS"))
                                        if os.getenv("HEAD_MAX_MISS_PLACEHOLDERS") else None)
        self.behavior_miss_counts = {}
        # Track stitching: a person who submerges and re-surfaces gets a new track id
        # because the new detection overlaps nothing. These ids are joined back into one
        # person FOR CLUSTERING ONLY, so the action model sees one whole movement instead
        # of two or three part-crops. Behaviour analysis still judges each fragment alone.
        self.track_aliases = {}
        self.stitch_tracks = os.getenv("STITCH_TRACKS", "True").lower() == "true"
        # How long a person may be missing and still be recognised on re-appearance.
        # In SECONDS, like the head thresholds and for the same reason: the pipeline's
        # frame rate depends on the source fps, so a fixed frame count meant 0.8s of
        # tolerance on one camera and 1.5s on another - shorter than a single dive.
        self.stitch_max_gap_s = float(os.getenv("STITCH_MAX_GAP_SECONDS", "3.0"))
        # Exact frame count override. Parsed once, because os.getenv returns the STRING
        # "0", which is truthy - testing the raw value silently pins the gap.
        _gap = (os.getenv("STITCH_MAX_GAP_FRAMES") or "").strip()
        self._stitch_gap_override = int(_gap) if _gap.lstrip("-").isdigit() and int(_gap) > 0 else None
        self.stitch_radius_bodies = float(os.getenv("STITCH_RADIUS_BODIES", "1.0"))
        self.stitch_drift_bodies = float(os.getenv("STITCH_DRIFT_BODIES", "0.3"))
        self.stitch_max_size_ratio = float(os.getenv("STITCH_MAX_SIZE_RATIO", "2.0"))
        # The radius grows with the gap, so it has to be capped: at 0.3 bodies per frame
        # a 13-frame gap reached 4.9 bodies and allowed a 630px jump in a crowded lane,
        # which lands on somebody else. And a floor in pixels, because for a 25px
        # detection a couple of bodies is 60px - smaller than the jitter of the box
        # itself, which was rejecting genuine re-appearances of distant swimmers.
        self.stitch_max_bodies = float(os.getenv("STITCH_MAX_BODIES", "2.5"))
        self.stitch_min_radius_px = float(os.getenv("STITCH_MIN_RADIUS_PX", "80"))
        self.use_behavior_analysis = (
            use_behavior_analysis
            if use_behavior_analysis is not None
            else os.getenv("USE_BEHAVIOR_ANALYSIS", "True").lower() == "true"
        )
        if not self.use_behavior_analysis:
            logger.info("Behavior analysis disabled (USE_BEHAVIOR_ANALYSIS=False)")

        # Predefine colors (fixed single color for clusters, random for detections)
        self.colors_detections = [
            (
                np.random.randint(0, 100),
                np.random.randint(200, 255),
                np.random.randint(0, 100),
            )
            for _ in range(50)
        ]
        self.colors_clusters = [(0, 255, 255)]

        # Water model is initialized in derived classes
        self.water_model = None

        self.tracker = ObjectTracker(
            human_model_path=human_model_path,
            conf=TrackingHumansBase.conf,
            iou=TrackingHumansBase.iou,
            classes=TrackingHumansBase.classes,
            deque_length=tracking_deque_length,
            imgsz=imgsz,
        )

    def cleanup(self):
        if hasattr(self, "water_model") and self.water_model is not None:
            self.water_model.cleanup()

    @property
    def head_window_frames(self) -> int:
        """Head-evidence window in processed frames, from seconds and the real rate."""
        return max(1, int(round(self.head_evidence_window_s * self.processed_fps)))

    @property
    def stitch_max_gap(self) -> int:
        """Frames a track may be unseen and still be stitched to a new id."""
        if self._stitch_gap_override is not None:
            return self._stitch_gap_override
        return max(4, int(round(self.stitch_max_gap_s * self.processed_fps)))

    @property
    def _head_max_miss_placeholders(self) -> int:
        """Missed frames still recorded as 'not detected', as a duration."""
        if self._head_max_miss_override is not None:
            return self._head_max_miss_override
        return max(1, int(round(self._head_max_miss_s * self.processed_fps)))

    @property
    def head_cache_size(self) -> int:
        """Prediction cache length: never shorter than the evidence window.

        The bbox/crop deques are sized for tracking and hold image data, so they stay as
        they are; predictions are a few bytes each and must cover the whole window or the
        oldest part of it would be silently missing.
        """
        return max(self.trajectory_deque_size, self.head_window_frames + 8)

    @property
    def head_min_frames(self) -> int:
        """Minimum evidence before the head check may exclude, in processed frames.

        Kept strictly below the window: a threshold at or above it could never be met.
        """
        return max(2, min(int(round(self.head_min_evidence_s * self.processed_fps)),
                          self.head_window_frames))

    def set_video_metadata(self, video_width, video_height, fps, sample_interval=None):
        """
        Sets the metadata for the video.

        Args:
            video_width (int): Width of the video.
            video_height (int): Height of the video.
            fps (int): Frames per second of the video.
            sample_interval (int, optional): Only every Nth frame reaches the pipeline.
                Given it, the tracker knows its true processed frame rate and can turn
                the head-visibility settings from seconds into frames exactly. Without
                it the fallback rate is used.
        """
        self.reset()
        self.video_width = video_width
        self.video_height = video_height
        video_area = video_width * video_height
        self.threshold_huge_size = max(int(self.threshold_huge_size_perc * video_area), 800)
        self.threshold_area_big = int(self.threshold_area_big_perc * video_area)
        self.threshold_area_small = int(self.threshold_area_small_perc * video_area)
        self.min_object_size = int(self.min_object_size_perc * video_width)
        self.fps = fps
        if sample_interval and sample_interval > 0 and fps and fps > 0:
            self.processed_fps = fps / sample_interval
        logger.info(
            "Head evidence: window %.1fs, minimum %.1fs at %.2f processed fps "
            "-> %d / %d frames",
            self.head_evidence_window_s, self.head_min_evidence_s, self.processed_fps,
            self.head_window_frames, self.head_min_frames,
        )
        self.tile_cols, self.tile_rows = (
            TrackingHumansBase.tile_cols,
            TrackingHumansBase.tile_rows,
        )
        self.tracker.initialize(
            tile_rows=self.tile_rows,
            tile_cols=self.tile_cols,
            frame_width=self.video_width,
            frame_height=self.video_height,
        )

    @staticmethod
    def expand_water_area(mask, perc=0.1):
        """
        Expands the area of water pixels in a binary mask.

        Parameters:
            mask (numpy.ndarray): Binary mask with 0 for background and 1 for water.
            perc (float): Percentage of expansion. Determines the size of the dilation kernel.

        Returns:
            numpy.ndarray: Expanded binary mask.

        Raises:
            Exception: If an error occurs during the expansion process, it is logged.
        """
        try:
            # Calculate kernel size based on the mask size and percentage
            height, width = mask.shape
            kernel_size = max(1, int(min(height, width) * perc))

            # Create a square kernel for dilation
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Perform dilation to expand the water area
            expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

            return expanded_mask
        except Exception as e:
            logger.error(f"Exception in expand_water_area: {e}")
            return None

    def get_water_mask(self, frame):
        """
        Generates a water mask for the given frame using a majority voting mechanism.
        Args:
            frame (numpy.ndarray): The input frame for which the water mask is to be generated.
        Returns:
            numpy.ndarray: A boolean mask where True indicates the presence of water.
        Steps:
            1. Get the current water mask from the model.
            2. Append the water mask to the deque.
            3. Stack all stored masks to form a 3D array.
            4. Perform a majority vote to determine the final water mask.
            5. Optionally expand the water area in the mask.
            6. Convert the final mask to a boolean mask.
        Note:
            The function prints the elapsed time for water detection.
        """
        try:
            if self.water_model is None:
                logger.warning(
                    "Water model not initialized. Skipping water mask generation."
                )
                return None
            start = time.perf_counter()
            # 1) Get the current water mask from the model
            water_mask = self.water_model.get_pred(frame)
            # 2) Append to the deque
            self.water_deque.append(water_mask)

            # 3) Stack all stored masks: shape => (deque_length, H, W)
            stacked_masks = np.stack(self.water_deque, axis=0)

            # 4) Perform majority vote:
            #    - We sum up all “water pixels” (==1) along the 0th axis.
            #    - If # of times “water” >= half of the deque length => 1, else 0
            threshold = len(self.water_deque) / 2.0
            vote_mask = np.sum(stacked_masks, axis=0)  # shape => (H, W)
            majority_mask = (vote_mask >= threshold).astype(np.uint8)

            # 5) Optionally expand the water area
            expanded_mask = TrackingHumansBase.expand_water_area(majority_mask)

            # 6) Convert to boolean mask (True = water_class)
            water_mask_bool = expanded_mask == self.water_class

            logger.info(
                f"Water detection elapsed: {(time.perf_counter() - start):.03f}"
            )
            return water_mask_bool
        except Exception as e:
            logger.error(f"Exception in get_water_mask: {e}")
            return None

    def set_deque_length(self, deque_length):
        """
        Sets the length of the tracking deque and resizes trajectory deques accordingly.
        """
        if self.tracker is not None:
            self.tracker.set_deque_lenght(deque_length)
        self.trajectory_deque_size = int(deque_length * TrackingHumansBase.trajectory_deque_multiplier)
        self.track_retention_frames = deque_length
        logger.info(f"Trajectories initialized with size: {self.trajectory_deque_size} | "
                    f"track retention: {self.track_retention_frames} frames")
        for track_id, pts in self.trajectories.items():
            self.trajectories[track_id] = deque(pts, maxlen=self.trajectory_deque_size)
        for track_id, bboxes in self.behavior_bbox_history.items():
            self.behavior_bbox_history[track_id] = deque(bboxes, maxlen=self.trajectory_deque_size)

    def apply_transparent_water_mask(self, frame, alpha=0.2):
        """
        Applies the water mask to the frame, making non-water pixels semi-transparent.

        Args:
            frame (numpy.ndarray): The input frame to be processed.
        Returns:
            numpy.ndarray: The processed frame with non-water pixels made semi-transparent.
        """
        with self.water_mask_lock:
            if self.water_mask_bool is not None:

                water_mask = self.water_mask_bool

                # Create an alpha channel based on the water mask
                alpha_channel = np.full_like(
                    frame[:, :, 0], int(alpha * 255), dtype=np.uint8
                )  # Set default alpha to 128 (50% transparency)
                alpha_channel[water_mask] = (
                    255  # Set alpha to 255 (fully opaque) for water pixels
                )

                # Add the alpha channel to the frame
                frame_with_alpha = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                frame_with_alpha[:, :, 3] = alpha_channel

                return frame_with_alpha
            else:
                return frame

    def apply_water_mask(self, frame):
        try:
            if self.water_mask_3c is None:
                return
            with self.water_mask_lock:
                frame *= (
                    self.water_mask_3c
                )  # In-place multiplication is faster than indexing
        except Exception as e:
            logger.error(f"Exception in apply_water_mask: {e}")

    def process_water(self, frame):
        """
        Processes the given frame to isolate water regions.
        This method processes the input frame to create a mask that identifies water regions.
        It updates the water mask at intervals defined by `skip_water_process_frames` and
        sets non-water pixels in the frame to black.
        Args:
            frame (numpy.ndarray): The input frame to be processed.
        Returns:
            numpy.ndarray: The processed frame with non-water pixels set to black.
        """
        try:
            if not self.detect_water:
                return frame

            if self.counter_process_water % self.skip_water_process_frames == 0:
                # Calculate the water mask
                water_mask = self.get_water_mask(frame)
                if water_mask is not None:
                    # Use the lock to update the shared resource
                    with self.water_mask_lock:
                        self.water_mask_bool = water_mask
                        self.water_mask_3c = water_mask.astype(np.uint8)[..., None]
                else:
                    logger.error("Water mask is None.")

            self.counter_process_water += 1
            self.apply_water_mask(frame)
            return frame
        except Exception as e:
            logger.error(f"Exception in process_water: {e}")
            return None

    def process_frame(self, frame):
        """
        Processes a single frame by resizing it, applying water processing, and tracking.

        Args:
            frame (numpy.ndarray): The input frame to be processed.

        Returns:
            numpy.ndarray: The processed frame.
        """
        try:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]
            frame = self.process_water(frame)
            start = time.perf_counter()
            self.tracker.process_frame(frame)
            logger.debug(f"process_frame elapsed: {(time.perf_counter() - start):.03f}s")
        except Exception as e:
            logger.error(f"Exception in process_frame (tracking): {e}")
            return None
        self._update_tracking_state(frame)
        self.frame_counter += 1
        return frame

    @staticmethod
    def _track_order(track_id):
        """Sort key for track ids: they are issued in increasing order, so smallest is
        oldest. Falls back to string order for non-numeric ids."""
        try:
            return (0, int(track_id))
        except (TypeError, ValueError):
            return (1, str(track_id))

    def _canonical_track(self, track_id):
        """Resolve a raw track id to the id of the person it belongs to (path-compressed)."""
        seen = []
        while track_id in self.track_aliases:
            seen.append(track_id)
            track_id = self.track_aliases[track_id]
        for t in seen:
            self.track_aliases[t] = track_id
        return track_id

    @staticmethod
    def _median_step(points):
        """Median per-frame (dx, dy) of a trajectory of (frame, cx, cy) points.

        Median rather than the last step: a person about to submerge produces one or two
        erratic detections, and the last step is exactly the one that is wrong. On the
        observed swimmer the last-step estimate put the prediction 113 px off while the
        median put it 49 px off, under half a body.
        """
        if len(points) < 2:
            return 0.0, 0.0
        dxs, dys = [], []
        for (f0, x0, y0), (f1, x1, y1) in zip(points, list(points)[1:]):
            df = max(1, f1 - f0)
            dxs.append((x1 - x0) / df)
            dys.append((y1 - y0) / df)
        return float(np.median(dxs)), float(np.median(dys))

    def _stitch_new_track(self, new_id, bbox, alive_canonical):
        """Alias a brand-new track id onto a person who vanished a moment ago.

        A swimmer who submerges is not detected, and when they surface again the new
        detection overlaps nothing, so the tracker issues a fresh id. Each fragment then
        clusters separately and the action model sees the same person as two or three
        part-crops of a movement instead of one whole one.

        The match requires all of:
          - the old track was NOT detected this frame (ids visible together are people
            seen together, so they cannot be the same person)
          - it vanished no more than stitch_max_gap frames ago
          - the new detection is near where the old track was heading, allowing the gap
          - the two are of comparable size

        Only clustering uses the result: behaviour analysis keeps judging each fragment
        on its own, so stitching cannot lengthen a track into a direction-rule exclusion.
        """
        if not self.stitch_tracks:
            return
        nx, ny = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        new_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        if new_size <= 0:
            return

        best, best_score = None, None
        rejected = []                         # (old_id, why) — logged when nothing matched
        for old_id, misses in self.behavior_miss_counts.items():
            if misses < 1:
                continue                      # detected this frame: a different person
            if misses > self.stitch_max_gap:
                rejected.append((old_id, f"gone {misses}f > max {self.stitch_max_gap}"))
                continue
            canonical = self._canonical_track(old_id)
            if canonical in alive_canonical:
                rejected.append((old_id, f"person {canonical} already on screen"))
                continue
            history = self.behavior_bbox_history.get(old_id)
            if not history:
                continue

            boxes = list(history)
            sizes = [max(b[2] - b[0], b[3] - b[1]) for b in boxes]
            body = float(np.median(sizes))
            if body <= 0:
                continue
            ratio = new_size / body
            if not (1 / self.stitch_max_size_ratio) <= ratio <= self.stitch_max_size_ratio:
                rejected.append((old_id, f"size ratio {ratio:.2f}"))
                continue

            last = boxes[-1]
            lx, ly = (last[0] + last[2]) / 2, (last[1] + last[3]) / 2
            vx, vy = self._median_step(self.trajectories.get(old_id, ()))
            px, py = lx + vx * misses, ly + vy * misses
            # nearest of "where it was heading" and "where it was last seen": a person
            # who stopped moving when they went under is still the same person.
            dist = min(float(np.hypot(nx - px, ny - py)), float(np.hypot(nx - lx, ny - ly)))
            limit = max(
                body * min(self.stitch_radius_bodies + self.stitch_drift_bodies * misses,
                           self.stitch_max_bodies),
                self.stitch_min_radius_px,
            )
            if dist > limit:
                rejected.append((old_id, f"{dist:.0f}px > {limit:.0f}px after {misses}f"))
                continue

            score = dist / limit
            if best_score is None or score < best_score:
                best, best_score = (canonical, misses, dist, limit), score

        if best is None:
            if rejected:
                logger.debug("New track %s not stitched; candidates: %s", new_id,
                             ", ".join(f"{t}({why})" for t, why in rejected))
            return
        canonical, misses, dist, limit = best
        self.track_aliases[new_id] = canonical
        alive_canonical.add(canonical)        # one new id per vanished person per frame
        logger.info(
            "Stitched track %s -> %s (gap=%d frames, dist=%.0f px, limit=%.0f px)",
            new_id, canonical, misses, dist, limit,
        )

    def _merge_aliased_objects(self, objects):
        """Union the bboxes of track fragments that belong to the same person."""
        merged = {}
        for track_id, obj in objects.items():
            canonical = self._canonical_track(track_id)
            bbox = list(obj["bbox"])
            if canonical not in merged:
                merged[canonical] = {**obj, "bbox": bbox, "members": [track_id]}
                continue
            m = merged[canonical]
            mb = m["bbox"]
            m["bbox"] = [min(mb[0], bbox[0]), min(mb[1], bbox[1]),
                         max(mb[2], bbox[2]), max(mb[3], bbox[3])]
            m["frames_count"] = m.get("frames_count", 1) + obj.get("frames_count", 1)
            m["members"].append(track_id)
        for canonical, m in merged.items():
            if len(m["members"]) > 1:
                logger.info("Merged tracks %s into one person (%s): %dx%d px, %d frames",
                            m["members"], canonical,
                            int(m["bbox"][2] - m["bbox"][0]), int(m["bbox"][3] - m["bbox"][1]),
                            m.get("frames_count", 1))
        return merged

    def _update_tracking_state(self, frame):
        """Update trajectories, bbox/crop history, age out missing tracks, and run per-frame head visibility."""
        t_total = time.perf_counter()
        try:
            frame_pending: list = []  # (track_id, crop) for per-frame head inference
            with self.trajectories_lock:
                detected_ids = set(self.tracker.tracking_per_frame[-1].keys()) if self.tracker.tracking_per_frame else set()
                # A stitch claims two ids are one person. Seeing both in the SAME frame
                # disproves it - and that can only be discovered afterwards, because at
                # stitch time the old id was by definition absent. Observed on a drowning
                # swimmer: track 42 was joined to person 27 while 27 was off screen, then
                # 37 and 42 appeared together for six frames, putting two people in one
                # cluster. The newer id is released; the oldest keeps the identity.
                by_person = {}
                for rid in detected_ids:
                    by_person.setdefault(self._canonical_track(rid), []).append(rid)
                for canonical, rids in by_person.items():
                    if len(rids) < 2:
                        continue
                    keeper = canonical if canonical in rids else min(rids, key=self._track_order)
                    for rid in rids:
                        if rid != keeper and rid in self.track_aliases:
                            del self.track_aliases[rid]
                            logger.info(
                                "Unstitched track %s from person %s: both detected in "
                                "frame %d, so they are two people",
                                rid, canonical, self.frame_counter,
                            )

                # Canonical ids alive this frame. A vanished track can only be stitched to
                # a new id if nothing else is already reporting it: two ids visible at the
                # same time are two people, whatever their positions.
                alive_canonical = {self._canonical_track(t) for t in detected_ids}

                for track_id, obj in (self.tracker.tracking_per_frame[-1].items() if self.tracker.tracking_per_frame else []):
                    b = obj["bbox"]
                    cx = int((b[0] + b[2]) / 2)
                    cy = int((b[1] + b[3]) / 2)
                    if track_id not in self.behavior_bbox_history:
                        # brand-new id: is it a person who just re-surfaced under a new id?
                        self._stitch_new_track(track_id, b, alive_canonical)
                    if track_id not in self.trajectories:
                        self.trajectories[track_id] = deque(maxlen=self.trajectory_deque_size)
                        logger.debug(f"Trajectory initialized for track_id={track_id}, maxlen={self.trajectory_deque_size}")
                    self.trajectories[track_id].append((self.frame_counter, cx, cy))
                    if track_id not in self.behavior_bbox_history:
                        self.behavior_bbox_history[track_id] = deque(maxlen=self.trajectory_deque_size)
                    self.behavior_bbox_history[track_id].append(b)

                    self.behavior_miss_counts[track_id] = 0      # seen this frame
                    if self.head_classifier is not None:
                        if track_id not in self.behavior_crop_history:
                            self.behavior_crop_history[track_id] = deque(maxlen=self.trajectory_deque_size)
                            self.behavior_head_preds_cache[track_id] = deque(maxlen=self.head_cache_size)
                        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        crop = frame[y1:y2, x1:x2]
                        if (crop.size > 0
                                and crop.shape[1] > self.head_visibility_min_bbox_size
                                and crop.shape[0] > self.head_visibility_min_bbox_size):
                            self.behavior_crop_history[track_id].append(crop.copy())
                            frame_pending.append((track_id, crop.copy()))

                # Age out tracks not seen this frame. A track is kept until its LAST
                # detection falls outside the action-recognition window, then dropped
                # whole — its history is never eroded while it is still inside it.
                #
                # This used to drop one historic entry per missed frame, so a track's
                # history shrank by one for every frame the person was not detected and
                # the track was deleted once it hit zero. For a drowning person that is
                # backwards: they are submerged (undetected) more often than visible, so
                # history went net-negative and the track was deleted after a couple of
                # seconds. Object detection then re-acquired them with a NEW id (too
                # little overlap to re-match), which also started from nothing, so no id
                # ever survived to the next behaviour window. The person vanished from
                # the candidate set entirely and X3D never ran on them — not excluded by
                # any rule, simply never analysed. Staleness is still bounded, by the
                # same window the action model itself looks at.
                for track_id in list(self.behavior_bbox_history.keys()):
                    if track_id not in detected_ids:
                        # Record the miss instead of letting it vanish: undetected frames
                        # belong in the denominator, since in water they often mean
                        # submerged. Capped, so someone who left the zone stops
                        # accumulating placeholders while the track itself lives on.
                        misses = self.behavior_miss_counts.get(track_id, 0) + 1
                        self.behavior_miss_counts[track_id] = misses
                        # placeholders follow the SAME cadence as real predictions,
                        # otherwise a sample rate above 1 fills the cache with more
                        # "not detected" slots than prediction slots and drags coverage
                        # down for a person who is simply being classified less often
                        if (track_id in self.behavior_head_preds_cache
                                and misses <= self._head_max_miss_placeholders
                                and self.frame_counter % self.head_visibility_sample_rate == 0):
                            self.behavior_head_preds_cache[track_id].append((NOT_DETECTED, 0.0))
                        if misses > self.track_retention_frames:
                            self.behavior_bbox_history.pop(track_id, None)
                            self.trajectories.pop(track_id, None)
                            self.behavior_crop_history.pop(track_id, None)
                            self.behavior_head_preds_cache.pop(track_id, None)
                            self.behavior_miss_counts.pop(track_id, None)
                            self.track_aliases = {
                                raw: canon for raw, canon in self.track_aliases.items()
                                if raw in self.behavior_bbox_history
                                or canon in self.behavior_bbox_history
                            }
                            logger.debug(
                                f"Track {track_id} aged out: unseen for {misses} frames "
                                f"(> {self.track_retention_frames})"
                            )
        except Exception as e:
            logger.error(f"Exception in _update_tracking_state: {e}")
            return

        # Run head visibility inference outside the lock in max_batch_size chunks
        if (self.use_behavior_analysis
                and frame_pending and self.head_classifier is not None
                and self.frame_counter % self.head_visibility_sample_rate == 0):
            try:
                t0 = time.perf_counter()
                all_crops = [c for _, c in frame_pending]
                all_probs: list[np.ndarray] = []
                for i in range(0, len(all_crops), self.head_classifier_max_batch_size):
                    chunk = all_crops[i:i + self.head_classifier_max_batch_size]
                    all_probs.append(self.head_classifier.predict_probs(chunk))
                probs_cat = np.concatenate(all_probs, axis=0)   # (N, num_classes)
                infer_ms = (time.perf_counter() - t0) * 1000
                logger.debug(
                    "Head visibility per-frame frame=%d | tracks=%d crops=%d infer=%.1f ms",
                    self.frame_counter, len(frame_pending), len(all_crops), infer_ms,
                )
                with self.trajectories_lock:
                    for (track_id, _), probs in zip(frame_pending, probs_cat):
                        if track_id in self.behavior_head_preds_cache:
                            cls_idx    = int(probs.argmax())
                            confidence = float(probs[cls_idx])
                            self.behavior_head_preds_cache[track_id].append(
                                (HEAD_CLASS_NAMES[cls_idx], confidence)
                            )
            except Exception as e:
                logger.warning("Per-frame head visibility failed: %s", e)

        logger.debug(
            "_update_tracking_state frame=%d elapsed=%.1f ms",
            self.frame_counter, (time.perf_counter() - t_total) * 1000,
        )

    def compute_clusters(self):
        """
        Finalizes the processing of tracked objects by performing the following steps:

        1. Merges bounding boxes with the same object ID.
        2. Filters out objects that have been tracked for fewer frames than the threshold.
        4. Adjusts small objects.
        5. Clusters human objects based on the video dimensions and total number of crops.
        6. Merges overlapping clusters.
        7. Resets the tracker for future processing.

        Attributes:
            tracked_objects (dict): Dictionary of tracked objects after merging and filtering.
            clusters (list): List of clusters after processing.
        """
        try:
            self.reset_tracking()
            self.tracked_objects = self.tracker.merge_bboxes_same_object_id()
            # Join fragments of one person before the frames_count filter, so three
            # one-frame sightings count as one three-frame person rather than being
            # dropped individually.
            self.tracked_objects = self._merge_aliased_objects(self.tracked_objects)
            self.tracked_objects = {
                id: obj
                for id, obj in self.tracked_objects.items()
                if obj["frames_count"] > self.threshold_frames
            }
            self.remove_huge_objects()
            self.adjust_small_objects()
            with self.trajectories_lock:
                bbox_history = {tid: list(dq) for tid, dq in self.behavior_bbox_history.items()}
                crop_history = {tid: list(dq) for tid, dq in self.behavior_crop_history.items()}
                head_preds_cache = {tid: list(dq) for tid, dq in self.behavior_head_preds_cache.items()}
                traj_snapshot = {tid: list(dq) for tid, dq in self.trajectories.items()}

            if self.use_behavior_analysis:
                ba = BehaviorAnalyses(
                    head_classifier=self.head_classifier,
                    head_shoulders_conf_threshold=float(os.getenv("HEAD_SHOULDERS_CONF_THRESHOLD", "0.70")),
                    head_clearly_conf_threshold=float(os.getenv("HEAD_CLEARLY_CONF_THRESHOLD", "0.75")),
                )
                ba.save_jsons = self._save_behavior_jsons
                candidates, excluded_direction, excluded_direction_change, excluded_head_visible = \
                    ba.get_drowning_candidates(
                        bbox_history,
                        crop_history=crop_history,
                        head_preds_cache=head_preds_cache,
                        head_visible_frame_count=self.head_window_frames,
                        head_visibility_min_frames=self.head_min_frames,
                        downsample_head_frames=self.downsample_head_frames,
                        head_evidence_samples=self.head_evidence_samples,
                        frame_history={tid: [int(f) for f, _, _ in pts]
                                       for tid, pts in traj_snapshot.items()},
                        processed_fps=self.processed_fps,
                    )
                if ba.save_jsons:
                    self._dump_bbox_history(
                        bbox_history,
                        trajectories=traj_snapshot,
                        head_stats=ba.head_stats,
                        motion_stats=ba.motion_stats,
                        categories={
                            BehaviorCategory.CANDIDATE: candidates,
                            BehaviorCategory.EXCLUDED_DIRECTION: excluded_direction,
                            BehaviorCategory.EXCLUDED_DIRECTION_CHANGE: excluded_direction_change,
                            BehaviorCategory.EXCLUDED_HEAD_VISIBLE: excluded_head_visible,
                        },
                    )
                logger.info(
                    f"candidates: {len(candidates)} | "
                    f"excluded_direction: {len(excluded_direction)} | excluded_direction_change: {len(excluded_direction_change)} | "
                    f"excluded_head_visible: {len(excluded_head_visible)}"
                )
                self.behavior_categories = {
                    BehaviorCategory.CANDIDATE: candidates,
                    BehaviorCategory.EXCLUDED_DIRECTION: excluded_direction,
                    BehaviorCategory.EXCLUDED_DIRECTION_CHANGE: excluded_direction_change,
                    BehaviorCategory.EXCLUDED_HEAD_VISIBLE: excluded_head_visible,
                }
                self.behavior_slow_linear = set(ba.slow_linear)
                self.behavior_run_id += 1
                self.behavior_head_snapshot = (self.behavior_run_id, ba.head_stats)
            else:
                candidates = set(self.tracked_objects.keys())
                logger.info("Behavior analysis disabled — all %d tracked objects are candidates", len(candidates))

            cluster_humans = ClusterHumans(
                self.num_total_crops, tracked_objects=self.tracked_objects
            )
            # candidates are raw track ids; tracked_objects is keyed by person. A person
            # is kept if ANY of their fragments survived behaviour analysis.
            self.clusters = cluster_humans.process(
                candidate_ids={self._canonical_track(t) for t in candidates}
            )
            self.merge_overlapping_clusters()
            # how large is each person in pixels, and where are they? The margin scales
            # with the person, not with the cluster, which is only their path envelope.
            sizes_by_person, person_boxes = {}, {}
            for tid, boxes in bbox_history.items():
                if not boxes:
                    continue
                pid = self._canonical_track(tid)
                sizes_by_person.setdefault(pid, []).extend(
                    max(b[2] - b[0], b[3] - b[1]) for b in boxes)
                xs = [c for b in boxes for c in (b[0], b[2])]
                ys = [c for b in boxes for c in (b[1], b[3])]
                box = [min(xs), min(ys), max(xs), max(ys)]
                prev = person_boxes.get(pid)
                person_boxes[pid] = box if prev is None else [
                    min(prev[0], box[0]), min(prev[1], box[1]),
                    max(prev[2], box[2]), max(prev[3], box[3])]
            body_sizes = {pid: float(np.median(v)) for pid, v in sizes_by_person.items()}
            self._frame_clusters(body_sizes, person_boxes)
        except Exception as e:
            logger.error(f"Exception in compute_clusters: {e}")

    def _dump_bbox_history(self, bbox_history: dict, head_stats: dict | None = None,
                           motion_stats: dict | None = None,
                           categories: dict | None = None,
                           trajectories: dict | None = None) -> None:
        """Dump one behaviour-analysis call: per-track bboxes plus a summary.json.

        The summary carries the head-visibility numbers behind each decision (the
        fractions, the bars they were compared against, and every per-frame
        prediction), so a track that was or wasn't excluded can be explained after
        the fact instead of only from a DEBUG log line.

        It also carries the frames each track was DETECTED in, plus the stitching
        result. Without the frame numbers two fragments cannot be told apart from two
        people offline: whether they overlapped in time is the whole question, and a
        bare list of boxes cannot answer it.
        """
        runs_dir = Path(__file__).parent / "tests" / "runs"
        existing = sorted(runs_dir.glob("run_*")) if runs_dir.exists() else []
        run_dir = runs_dir / f"run_{len(existing):02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for track_id, bboxes in bbox_history.items():
            data = [[float(v) for v in b] for b in bboxes]
            (run_dir / f"{str(track_id).zfill(3)}.json").write_text(json.dumps(data, indent=2))

        category_of = {}
        for cat, ids in (categories or {}).items():
            for tid in ids:
                category_of[str(tid)] = cat.name

        # track_id -> frames it was actually detected in (gaps are submerged / missed)
        frames_of = {tid: [int(f) for f, _, _ in pts] for tid, pts in (trajectories or {}).items()}
        # canonical person -> the track ids stitched into them
        people = {}
        for tid in bbox_history:
            people.setdefault(self._canonical_track(tid), []).append(tid)

        def _track_entry(tid, bboxes):
            frames = frames_of.get(tid, [])
            canonical = self._canonical_track(tid)
            entry = {
                "category": category_of.get(str(tid), "UNKNOWN"),
                "n_bboxes": len(bboxes),
                "detected_frames": frames,
                "first_frame": frames[0] if frames else None,
                "last_frame": frames[-1] if frames else None,
                "gaps": [b - a for a, b in zip(frames, frames[1:]) if b - a > 1],
                "person": canonical,
                "stitched": canonical != tid,
                "head": (head_stats or {}).get(tid, {"checked": False, "reason": "no head predictions"}),
            }
            if tid in (motion_stats or {}):
                entry["motion"] = motion_stats[tid]
            return entry

        summary = {
            "frame": self.frame_counter,
            # every id that was joined to another, and the resulting groups
            "aliases": {str(k): v for k, v in self.track_aliases.items()},
            "people": {str(person): sorted(ids) for person, ids in people.items()},
            "tracks": {
                str(tid): _track_entry(tid, bboxes)
                for tid, bboxes in bbox_history.items()
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        groups = {p: ids for p, ids in people.items() if len(ids) > 1}
        logger.info(
            f"Saved bbox history to {run_dir} ({len(bbox_history)} tracks, "
            f"{len(people)} people"
            + (f", stitched: {groups}" if groups else "")
            + ")"
        )

    def remove_huge_objects(self):
        """
        Removes objects from the tracked_objects dictionary that have a bounding box
        width or height exceeding the threshold_huge_size.

        This method iterates through the tracked_objects dictionary, calculates the
        width and height of each object's bounding box, and retains only those objects
        whose dimensions are below the threshold_huge_size. The filtered objects are
        then stored back in the tracked_objects dictionary.

        Returns:
            None
        """
        adjusted_objects = {}
        for id, obj in self.tracked_objects.items():
            bbox = list(map(int, obj["bbox"]))
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < self.threshold_huge_size and h < self.threshold_huge_size:
                adjusted_objects[id] = obj
            else:
                logger.info(f"Removing huge object: {id} | {w}x{h}")
        self.tracked_objects = adjusted_objects

    def reset_tracking(self):
        """
        Resets tracking results. Trajectories are kept (bounded deques handle rollover).
        """
        self.tracked_objects = {}
        self.behavior_categories = {
            BehaviorCategory.CANDIDATE: set(),
            BehaviorCategory.EXCLUDED_DIRECTION: set(),
            BehaviorCategory.EXCLUDED_DIRECTION_CHANGE: set(),
            BehaviorCategory.EXCLUDED_HEAD_VISIBLE: set(),
        }
        # candidates that moved straight but too slowly to be swimming (shown as "slow")
        self.behavior_slow_linear = set()
        self.merged_clusters = {}
        self.clusters = {}
        self.tracking_results = None

    def reset(self):
        self.reset_tracking()
        self.trajectories = {}
        self.behavior_bbox_history = {}
        self.behavior_crop_history = {}
        self.behavior_head_preds_cache = {}
        self.behavior_miss_counts = {}
        self.track_aliases = {}
        self.behavior_head_snapshot = (0, {})
        self.behavior_run_id = 0
        self.frame_counter = 0
        self.counter_process_water = 0
        self.water_mask_bool = None
        self.water_mask_3c = None
        self.water_deque.clear()

    def get_color_id(self, id):
        """
        Returns a color ID for a given object ID.

        Args:
            id (int): The object ID.

        Returns:
            int: The color ID.
        """
        try:
            if isinstance(id, int):
                return id
            else:
                if "_" in id:
                    id = id.split("_")[1]
                return int(id)
        except Exception as e:
            logger.error(f"Exception in get_color_id: {e}")

    def get_predictions(self):
        """
        Returns the current tracking results.

        Returns:
            dict: The current tracking results.
        """
        if self.tracker is not None:
            return self.tracker.get_last_frame_tracking()
        else:
            return None

    def draw_object_detections(self, frame, font_scale=1, thickness=2):
        """
        Custom annotation of frames with adjustable font size and bounding box thickness.

        Args:
            frame: The frame to annotate.
            font_scale: Font size for text annotations.
            thickness: Thickness of bounding box and text.
        """
        try:
            detections = self.tracker.get_last_frame_tracking()
            # if tracking hasn't started yet
            if detections is None:
                return

            for id, obj in detections.items():
                bbox = obj["bbox"]
                x_min, y_min, x_max, y_max = map(int, bbox)

                if id in self.behavior_categories[BehaviorCategory.EXCLUDED_DIRECTION]:
                    color = (128, 0, 128)     # purple  – consistent direction
                elif id in self.behavior_categories[BehaviorCategory.EXCLUDED_DIRECTION_CHANGE]:
                    color = (128, 128, 0)     # teal    – lap / direction change
                elif id in self.behavior_categories[BehaviorCategory.EXCLUDED_HEAD_VISIBLE]:
                    color = (255, 255, 0)     # cyan    – head clearly above water
                else:
                    color = (255, 0, 0)       # blue    – candidate (or not yet analysed)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

                if self._visualize_head_visibility:
                    self._draw_head_visibility(
                        frame, id, x_min, y_min, x_max, y_max, font_scale, thickness
                    )

                label = f"ID: {id}" if id is not None else ""
                if id in self.behavior_slow_linear:
                    label += " slow"          # straight path, but too slow to be swimming
                cv2.putText(
                    frame,
                    label,
                    (x_min + 5, y_min + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                )

                with self.trajectories_lock:
                    raw_pts = self.trajectories.get(id)
                    pts = list(raw_pts) if raw_pts else None
                if pts and len(pts) >= 2:
                    coords = [(cx, cy) for _, cx, cy in pts]
                    pts_array = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts_array], isClosed=False, color=color, thickness=2)
                    for i, (frame_idx, cx, cy) in enumerate(pts):
                        cv2.circle(frame, (cx, cy), radius=3, color=color, thickness=-1)
                        if i == 0 or i == len(pts) - 1:
                            cv2.putText(frame, str(frame_idx), (cx + 4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        except Exception as e:
            logger.error(f"Exception in draw_object_detections: {e}")

    def _draw_head_visibility(self, frame, track_id, x_min, y_min, x_max, y_max,
                              font_scale, thickness):
        """Colour-coded label with this frame's head-visibility class for one person.

        The person box is the track-level behaviour decision; this label is the
        per-frame evidence behind it, so a track excluded as "head visible" can be
        checked against what the classifier actually saw frame by frame. Colours match
        review_predictions_ui (both come from head_visibility_classes).

        Sits above the person box, or below it when the box is too close to the top
        edge to fit. White on a filled colour patch, since coloured text alone is hard
        to read against water.

        The classifier only runs every head_visibility_sample_rate frames, so between
        samples this shows the most recent prediction rather than a fresh one.
        """
        with self.trajectories_lock:
            preds = self.behavior_head_preds_cache.get(track_id)
            # Skip back past "not detected" placeholders to the last real prediction.
            # Those are bookkeeping for the coverage denominator, not something the
            # classifier saw, and drawing them labelled a plainly visible person
            # "not_detected 0.00" whenever they were missed for a frame or two and the
            # classifier had not run again yet.
            last = None
            for entry in reversed(preds or ()):
                if entry and entry[0] != NOT_DETECTED:
                    last = entry
                    break
        if last is None:
            return
        cls_name, confidence = last
        color = class_color_bgr(cls_name)

        text = f"{CLASS_SHORT_NAMES.get(cls_name, cls_name)} {confidence:.2f}"
        scale = max(0.3, font_scale * 0.5)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        pad = 3
        box_h = th + base + pad * 2
        frame_h, frame_w = frame.shape[:2]

        # above the person box when it fits, otherwise below it
        top = y_min - box_h - thickness
        if top < 0:
            top = min(y_max + thickness, frame_h - box_h)
        top = max(0, top)
        left = max(0, min(x_min, frame_w - (tw + pad * 2)))

        cv2.rectangle(frame, (left, top), (left + tw + pad * 2, top + box_h), color, -1)
        cv2.putText(frame, text, (left + pad, top + th + pad),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _frame_clusters(self, body_sizes=None, person_boxes=None):
        """Add a margin of background around each cluster.

        The margin is the SAME on all four sides. It used to be a multiplier on the
        cluster box, which padded each axis in proportion to its own length: a 1429x597
        cluster gained 214px horizontally but only 90px vertically. Nothing about a limb
        leaving its detection box is horizontal - a hand strays the same distance in any
        direction - so the asymmetry was an artefact of the arithmetic, not the problem.

        The margin scales with the PERSON, not with the cluster. Apparent size is what
        governs how far a limb can stray: a swimmer 20m away occupies 100px and their
        hand cannot be 50px outside the box, while one close to the camera occupies 500px
        and it can. The cluster box is a poor proxy for that, because it is the union of
        everywhere the person has been and so grows with their path: a distant swimmer
        measured 108px across yet spanned 948x229, which would earn a 34px margin from
        the cluster and 16px from their own size. min(w, h) is used when no person can be
        matched to the cluster.

        Clusters are framed after merging, so merge decisions still see the true extent
        of the people in them and padding cannot cascade into further merges.
        """
        if not self.merged_clusters:
            return
        body_sizes = body_sizes or {}
        person_boxes = person_boxes or {}
        people = ([obj["bbox"] for obj in (self.tracked_objects or {}).values()]
                  if self.cluster_pad_avoid_humans else [])
        framed = {}
        for cluster_id, obj in self.merged_clusters.items():
            x1, y1, x2, y2 = (float(v) for v in obj["bbox"])
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue

            # how big are the people in this cluster, in pixels?
            inside_bodies = [body_sizes[pid] for pid, pb in person_boxes.items()
                             if pid in body_sizes
                             and pb[0] < x2 and pb[2] > x1 and pb[1] < y2 and pb[3] > y1]
            scale = float(np.median(inside_bodies)) if inside_bodies else min(w, h)

            # people already inside the cluster cannot be avoided; only those outside
            # constrain how far it may grow
            outsiders = [b for b in people
                         if not (b[0] < x2 and b[2] > x1 and b[1] < y2 and b[3] > y1)]
            frac = (self._free_margin_frac((x1, y1, x2, y2), outsiders, scale)
                    if self.cluster_pad_avoid_humans else self.cluster_pad_margin)
            margin = frac * scale

            nx1, ny1 = x1 - margin, y1 - margin
            nx2, ny2 = x2 + margin, y2 + margin
            if self.cluster_aspect > 0:
                cw, ch = nx2 - nx1, ny2 - ny1
                cx, cy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
                if cw / ch < self.cluster_aspect:
                    cw = ch * self.cluster_aspect
                else:
                    ch = cw / self.cluster_aspect
                nx1, ny1, nx2, ny2 = cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2
            # floor against degenerate crops, grown about the centre
            if nx2 - nx1 < self.min_bbox_size:
                cx = (nx1 + nx2) / 2
                nx1, nx2 = cx - self.min_bbox_size / 2, cx + self.min_bbox_size / 2
            if ny2 - ny1 < self.min_bbox_size:
                cy = (ny1 + ny2) / 2
                ny1, ny2 = cy - self.min_bbox_size / 2, cy + self.min_bbox_size / 2

            # slide back inside the frame before clipping, so a cluster at the edge keeps
            # its margin on the other side instead of losing it
            if nx1 < 0:
                nx2, nx1 = nx2 - nx1, 0
            if ny1 < 0:
                ny2, ny1 = ny2 - ny1, 0
            if nx2 > self.video_width:
                nx1, nx2 = nx1 - (nx2 - self.video_width), self.video_width
            if ny2 > self.video_height:
                ny1, ny2 = ny1 - (ny2 - self.video_height), self.video_height
            bbox = [int(max(0, nx1)), int(max(0, ny1)),
                    int(min(self.video_width, nx2)), int(min(self.video_height, ny2))]
            if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                continue
            framed[cluster_id] = {**obj, "bbox": bbox}
        self.merged_clusters = framed

    def _free_margin_frac(self, bbox, others, scale):
        """Largest margin fraction that does not pull another person into the crop.

        Returns a value in [cluster_pad_margin, cluster_pad_margin_max], as a fraction of
        `scale` (the people's pixel size). The minimum always wins: a crop with no margin
        is worse than one that clips a neighbour, because the person's own box jitters
        frame to frame.

        A box grown by an equal margin m on every side first touches another box once
        BOTH axes overlap, so the margin at which that happens is the larger of the two
        per-axis gaps. The smallest such margin over all outside people is the ceiling.
        """
        lo, hi = self.cluster_pad_margin, self.cluster_pad_margin_max
        x1, y1, x2, y2 = bbox
        if hi <= lo or scale <= 0:
            return lo
        ceiling = hi * scale                    # nobody nearby: take the whole allowance
        for ox1, oy1, ox2, oy2 in others:
            gap_x = max(ox1 - x2, x1 - ox2, 0.0)
            gap_y = max(oy1 - y2, y1 - oy2, 0.0)
            # stop just short of touching, so the neighbour stays out of the crop
            ceiling = min(ceiling, max(gap_x, gap_y) * 0.98)
        return max(lo, min(hi, ceiling / scale))

    def validate_small_objects(self):
        """
        Validates the size of tracked objects' bounding boxes.

        This method checks each tracked object's bounding box to ensure that its
        width and height are not smaller than the minimum bounding box size
        (`self.min_bbox_size`). Additionally, it verifies that the area of the
        bounding box is not smaller than the maximum allowed area (`max_area`).

        Raises:
            ValueError: If the width or height of the bounding box is smaller than
                        `self.min_bbox_size`, or if the area of the bounding box
                        is smaller than `max_area`.

        """
        try:
            max_area = self.min_bbox_size**2
            for _, obj in self.tracked_objects.items():
                bbox = obj["bbox"]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if width < self.min_bbox_size or height < self.min_bbox_size:
                    raise ValueError(f"Bbox is too small: {width}x{height}")
                area = width * height
                if area < max_area:
                    raise ValueError(f"Bbox is too small: {width}x{height} | {area}")
        except Exception as e:
            logger.error(f"Exception in validate_small_objects: {e}")

    def adjust_small_objects(self):
        """
        Adjusts the bounding boxes of small tracked objects to ensure they meet the minimum size requirements.
        This method performs the following steps:
        1. Iterates through the tracked objects.
        2. Calculates the width and height of each object's bounding box.
        3. Removes objects that are smaller than the minimum object size.
        4. Adjusts the bounding boxes that are smaller than the minimum width or height to meet the minimum size requirements.
           - Adjustments are made symmetrically on both sides (left-right or top-bottom).
           - If the bounding box is close to the edge of the video frame, adjustments are limited to the available space.
        5. Adds the adjusted objects to the final list of tracked objects.
        After adjusting the bounding boxes, the method updates the `tracked_objects` attribute with the adjusted objects
        and calls `validate_small_objects` to perform any additional validation.
        Attributes:
            tracked_objects (dict): A dictionary of tracked objects with their bounding boxes and frame counts.
            min_object_size (int): The minimum size (width or height) for an object to be considered valid.
            min_bbox_size (int): The minimum size (width or height) for a bounding box to be adjusted.
            video_width (int): The width of the video frame.
            video_height (int): The height of the video frame.
        """
        try:
            adjusted_objects = {}
            for id, obj in self.tracked_objects.items():
                bbox = list(map(int, obj["bbox"]))
                frames_count = obj["frames_count"]

                # Calculate object size (area)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                # Remove objects smaller than the minimum size
                if width < self.min_object_size or height < self.min_object_size:
                    continue

                # Adjust bounding boxes smaller than the minimum width
                if width < self.min_bbox_size:
                    total_adjustment = self.min_bbox_size - width
                    left_adjustment = int(total_adjustment / 2)
                    right_adjustment = total_adjustment - left_adjustment

                    # Adjust left side
                    if bbox[0] - left_adjustment < 0:
                        left_adjustment = bbox[
                            0
                        ]  # Adjust as much as possible to the left

                    # Adjust right side
                    if bbox[2] + right_adjustment > self.video_width:
                        right_adjustment = (
                            self.video_width - bbox[2]
                        )  # Adjust as much as possible to the right

                    # Redistribute any remaining adjustment
                    remaining_adjustment = total_adjustment - (
                        left_adjustment + right_adjustment
                    )
                    if remaining_adjustment > 0:
                        # Try to adjust more on the left if possible
                        if bbox[0] - remaining_adjustment >= 0:
                            left_adjustment += remaining_adjustment
                        else:
                            left_adjustment = bbox[0]  # Use all remaining space

                        # Adjust remaining to the right
                        remaining_adjustment = total_adjustment - (
                            left_adjustment + right_adjustment
                        )
                        right_adjustment += remaining_adjustment

                    # Apply adjustments
                    bbox[0] -= left_adjustment
                    bbox[2] += right_adjustment

                # Adjust bounding boxes smaller than the minimum height
                if height < self.min_bbox_size:
                    total_adjustment = self.min_bbox_size - height
                    top_adjustment = int(total_adjustment / 2)
                    bottom_adjustment = total_adjustment - top_adjustment

                    # Adjust top side
                    if bbox[1] - top_adjustment < 0:
                        top_adjustment = bbox[
                            1
                        ]  # Adjust as much as possible to the top

                    # Adjust bottom side
                    if bbox[3] + bottom_adjustment > self.video_height:
                        bottom_adjustment = (
                            self.video_height - bbox[3]
                        )  # Adjust as much as possible to the bottom

                    # Redistribute any remaining adjustment
                    remaining_adjustment = total_adjustment - (
                        top_adjustment + bottom_adjustment
                    )
                    if remaining_adjustment > 0:
                        # Try to adjust more on the top if possible
                        if bbox[1] - remaining_adjustment >= 0:
                            top_adjustment += remaining_adjustment
                        else:
                            top_adjustment = bbox[1]  # Use all remaining space

                        # Adjust remaining to the bottom
                        remaining_adjustment = total_adjustment - (
                            top_adjustment + bottom_adjustment
                        )
                        bottom_adjustment += remaining_adjustment

                    # Apply adjustments
                    bbox[1] -= top_adjustment
                    bbox[3] += bottom_adjustment

                # Ensure no negative values and not exceeding video dimensions
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = min(self.video_width, bbox[2])
                bbox[3] = min(self.video_height, bbox[3])

                # Add the adjusted object to the final list
                adjusted_objects[id] = {"bbox": bbox, "frames_count": frames_count}

            self.tracked_objects = adjusted_objects
            # self.validate_small_objects()
        except Exception as e:
            logger.error(f"Exception in adjust_small_objects: {e}")

    def merge_overlapping_clusters(self):
        """
        Merges overlapping clusters of tracked objects.

        This function iterates over the tracked objects and merges those that have overlapping bounding boxes.
        The merged objects are stored in `self.tracked_objects`.

        Raises:
            Exception: If an error occurs during the merging process, it is logged.
        """
        try:
            # Copy tracked objects to start the merging process
            merged_objects = [
                {"id": id, "bbox": obj["bbox"], "frames_count": obj["frames_count"]}
                for id, obj in self.clusters.items()
            ]
            updated = True

            while updated:
                updated = False
                new_merged_objects = []

                # Process each object and group overlapping objects
                while merged_objects:
                    current = merged_objects.pop(0)
                    current_group = [current]

                    for other in merged_objects[
                        :
                    ]:  # Iterate over a copy of merged_objects
                        # Check if 'other' overlaps with any object in 'current_group'
                        if any(
                            self.is_overlapping(other["bbox"], obj["bbox"])
                            for obj in current_group
                        ):
                            current_group.append(other)
                            merged_objects.remove(other)

                    # Merge the current group into a single object
                    if len(current_group) > 1:
                        merged_bbox = self.merge_bboxes(
                            [obj["bbox"] for obj in current_group]
                        )
                        merged_bbox[0] = max(0, merged_bbox[0])
                        merged_bbox[1] = max(0, merged_bbox[1])
                        merged_bbox[2] = min(self.video_width, merged_bbox[2])
                        merged_bbox[3] = min(self.video_height, merged_bbox[3])
                        frames_count = sum(obj["frames_count"] for obj in current_group)
                        new_merged_objects.append(
                            {
                                "id": current["id"],
                                "bbox": merged_bbox,
                                "frames_count": frames_count,
                            }
                        )
                        updated = True
                    else:
                        new_merged_objects.append(current_group[0])

                merged_objects = new_merged_objects

            # Update tracked objects with merged results
            self.merged_clusters = {obj["id"]: obj for obj in merged_objects}
        except Exception as e:
            logger.error(f"Error in merge_overlapping_clusters: {e}")

    def draw_clusters(self, frame, tracked_objects, show_frames_count=False):
        """
        Draws the merged bounding boxes on the frame.

        Args:
            frame (numpy.ndarray): The frame to draw the bounding boxes on.
            tracked_objects (dict): Dictionary of tracked objects with their bounding boxes.
            show_frames_count (bool): Whether to show the frames count in the bounding box label.
        """
        try:
            if tracked_objects is None:
                return
            for id, obj in tracked_objects.items():
                x_min, y_min, x_max, y_max = map(int, obj["bbox"])
                frames_count = obj["frames_count"]
                color_id = self.get_color_id(id)
                color = self.colors_clusters[
                    color_id % len(self.colors_clusters)
                ]  # Use predefined colors for consistent ID coloring
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                text = (
                    f"ID: {id} | {frames_count}" if show_frames_count else f"ID: {id}"
                )
                cv2.putText(
                    frame,
                    text,
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        except Exception as e:
            logger.error(f"Exception in draw_clusters: {e}")

    def save_video(
        self, title, tracked_objects, video_path, save_path, show_frames_count
    ):
        """
        Saves the video with merged bounding boxes.

        Args:
            tracked_objects (dict): Dictionary of tracked objects with their bounding boxes.
            video_path (str): Path to the input video.
            save_path (str): Path to save the output video.
            show_frames_count (bool): Whether to show the frames count in the bounding box label.
        """
        try:
            # Replay the video with merged bounding boxes
            cap = cv2.VideoCapture(video_path)
            new_width, new_height = get_adjusted_size(
                self.video_width, self.video_height
            )
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(save_path, fourcc, self.fps, (new_width, new_height))

            while cap.isOpened():
                success, frame = cap.read()

                if success:
                    frame = resize_big_frame(frame)
                    self.draw_clusters(frame, tracked_objects, show_frames_count)

                    # Write the frame to the output video
                    out.write(frame)

                    # Display the frame
                    cv2.imshow(title, frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Exception in save_video: {e}")

    def is_overlapping(self, box1, box2) -> bool:
        """
        Determines if two bounding boxes overlap based on a threshold.
        Args:
            box1 (tuple): A tuple of four integers (x1, y1, x2, y2) representing the first bounding box.
            box2 (tuple): A tuple of four integers (x1, y1, x2, y2) representing the second bounding box.
        Returns:
            bool: True if the boxes overlap based on the defined threshold, False otherwise.
        Raises:
            Exception: If an error occurs during the calculation.
        The method calculates the intersection area of the two boxes and compares it to the areas of the individual boxes.
        Depending on the sizes of the boxes, different overlap thresholds are used to determine if the boxes are considered overlapping.
        """
        try:
            # Compute intersection coordinates
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            # Calculate areas
            intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

            # Determine the threshold based on the sizes of both boxes
            if (
                box1_area < self.threshold_area_small
                and box2_area < self.threshold_area_small
            ):
                threshold_overlap = TrackingHumansBase.threshold_overlap_small
            elif (
                box1_area >= self.threshold_area_big
                or box2_area >= self.threshold_area_big
            ):
                threshold_overlap = TrackingHumansBase.threshold_overlap_big
            else:
                threshold_overlap = TrackingHumansBase.threshold_overlap_mid

            # Determine if one box is significantly inside or overlaps sufficiently
            return (
                intersection_area / box1_area > threshold_overlap
                or intersection_area / box2_area > threshold_overlap
            )
        except Exception as e:
            logger.error(f"Exception in is_overlapping: {e}")
            return False

    def get_last_frame_tracking(self):
        """
        Returns the last frame tracking results.

        Returns:
            dict: The last frame tracking results.
        """
        return self.tracker.get_last_frame_tracking()

    def merge_bboxes(self, bboxes):
        """
        Merges a list of bounding boxes into a single bounding box.

        Args:
            bboxes (list): List of bounding boxes to merge.

        Returns:
            list: The merged bounding box [x1, y1, x2, y2].
        """
        try:
            x1 = min(bbox[0] for bbox in bboxes)
            y1 = min(bbox[1] for bbox in bboxes)
            x2 = max(bbox[2] for bbox in bboxes)
            y2 = max(bbox[3] for bbox in bboxes)

            return [x1, y1, x2, y2]
        except Exception as e:
            logger.error(f"Exception in merge_bboxes: {e}")
