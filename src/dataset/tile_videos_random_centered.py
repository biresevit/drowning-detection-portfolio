import os
import sys
import cv2
import logging
from moviepy.editor import VideoFileClip
import numpy as np
from dotenv import load_dotenv
import uuid
import argparse
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from action_recognition.load_models_tensorrt import TensorRTModelLoader
from action_recognition.model_base import get_model_configs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
DESIRED_FPS = 30


# ------------------- Helpers -------------------

def _overlaps(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)


def _drowning_abs(bboxes_norm, video_width, video_height):
    result = []
    for b in bboxes_norm:
        bx1 = int((b[0] - b[2] / 2) * video_width)
        by1 = int((b[1] - b[3] / 2) * video_height)
        bx2 = int((b[0] + b[2] / 2) * video_width)
        by2 = int((b[1] + b[3] / 2) * video_height)
        result.append((bx1, by1, bx2, by2))
    return result


# ------------------- Crop functions -------------------

def generate_drowning_crop(bbox, video_width, video_height, min_crop_size, max_crop_size, factor_resize_bbox):
    """
    Generate a random crop that fully contains the bbox.
    Crop size is limited to factor_resize_bbox * bbox dimensions.
    bbox: [x_center_rel, y_center_rel, w_rel, h_rel]
    Returns (crop_x1, crop_y1, crop_x2, crop_y2)
    """
    bbox_x_center = bbox[0] * video_width
    bbox_y_center = bbox[1] * video_height
    bbox_width = bbox[2] * video_width
    bbox_height = bbox[3] * video_height

    # Absolute bbox corners
    bbox_x1 = int(round(bbox_x_center - bbox_width / 2.0))
    bbox_y1 = int(round(bbox_y_center - bbox_height / 2.0))
    bbox_x2 = int(round(bbox_x_center + bbox_width / 2.0))
    bbox_y2 = int(round(bbox_y_center + bbox_height / 2.0))

    if bbox_width >= video_width or bbox_height >= video_height:
        return 0, 0, video_width, video_height

    # Limit crop size to factor_resize_bbox * bbox dimensions.
    # If bbox exceeds max_crop_size, allow the crop to grow beyond it so the bbox always fits.
    max_crop_w = min(int(bbox_width * factor_resize_bbox), video_width)
    max_crop_h = min(int(bbox_height * factor_resize_bbox), video_height)
    if bbox_width <= max_crop_size:
        max_crop_w = min(max_crop_w, max_crop_size)
    if bbox_height <= max_crop_size:
        max_crop_h = min(max_crop_h, max_crop_size)

    min_crop_w = max(min_crop_size, int(math.ceil(bbox_width)))
    min_crop_h = max(min_crop_size, int(math.ceil(bbox_height)))

    # Ensure min <= max
    min_crop_w = min(min_crop_w, max_crop_w)
    min_crop_h = min(min_crop_h, max_crop_h)

    # Random crop size
    crop_width = int(np.random.randint(min_crop_w, max_crop_w + 1))
    crop_height = int(np.random.randint(min_crop_h, max_crop_h + 1))

    # Determine feasible top-left ranges so the crop fully contains bbox
    low_x = max(0, int(bbox_x2 - crop_width))
    high_x = min(int(bbox_x1), video_width - crop_width)
    low_y = max(0, int(bbox_y2 - crop_height))
    high_y = min(int(bbox_y1), video_height - crop_height)

    crop_x1 = int(np.random.randint(low_x, high_x + 1)) if low_x <= high_x else max(0, min(int(bbox_x1), video_width - crop_width))
    crop_y1 = int(np.random.randint(low_y, high_y + 1)) if low_y <= high_y else max(0, min(int(bbox_y1), video_height - crop_height))

    return crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height


def _generate_crop_around_human(hx1, hy1, hx2, hy2, video_width, video_height,
                                 min_crop_size, drowning_abs, max_tries=20):
    """
    Generate a crop centered on the human's merged trajectory bbox.
    Crop size = max(min_crop_size, merged_bbox_dim * (1 + U[0, 10%])), clamped to frame.
    Padding is always split equally on both sides of the bbox center.
    Returns (cx1,cy1,cx2,cy2) or None.
    """
    bbox_w  = hx2 - hx1
    bbox_h  = hy2 - hy1
    bbox_cx = (hx1 + hx2) // 2
    bbox_cy = (hy1 + hy2) // 2

    for _ in range(max_tries):
        crop_w = min(video_width,  max(min_crop_size, int(bbox_w * (1.0 + np.random.uniform(0.0, 0.10)))))
        crop_h = min(video_height, max(min_crop_size, int(bbox_h * (1.0 + np.random.uniform(0.0, 0.10)))))

        # Center on bbox center — extra space is split equally on both sides
        cx1 = max(0, min(bbox_cx - crop_w // 2, video_width  - crop_w))
        cy1 = max(0, min(bbox_cy - crop_h // 2, video_height - crop_h))
        cx2 = cx1 + crop_w
        cy2 = cy1 + crop_h

        if not any(_overlaps(cx1, cy1, cx2, cy2, *d) for d in drowning_abs):
            return cx1, cy1, cx2, cy2

    return None


def generate_normal_crop(drowning_bboxes_norm, video_width, video_height, min_crop_size,
                          max_crop_size, human_bboxes_abs=None):
    """
    Generate a crop that does not overlap any drowning annotation.

    If human_bboxes_abs is provided, tries to center the crop around a randomly chosen
    human bbox that itself doesn't overlap any drowning annotation.
    Falls back to a fully random crop if no valid human-centered crop can be found.
    """
    drowning_abs = _drowning_abs(drowning_bboxes_norm, video_width, video_height)

    if human_bboxes_abs:
        # Keep only humans that don't overlap any drowning bbox
        valid_humans = [
            h for h in human_bboxes_abs
            if not any(_overlaps(*h, *d) for d in drowning_abs)
        ]

        if valid_humans:
            # Shuffle and try each candidate so we don't always prefer the first one
            indices = list(range(len(valid_humans)))
            np.random.shuffle(indices)
            for idx in indices:
                result = _generate_crop_around_human(
                    *valid_humans[idx], video_width, video_height,
                    min_crop_size, drowning_abs
                )
                if result is not None:
                    return result

    # Fallback: fully random crop that avoids drowning annotations
    for _ in range(100):
        effective_max_w = min(video_width, max_crop_size)
        effective_max_h = min(video_height, max_crop_size)
        crop_w = 2 * np.random.randint(min_crop_size // 2, (effective_max_w // 2) + 1)
        crop_h = 2 * np.random.randint(min_crop_size // 2, (effective_max_h // 2) + 1)
        crop_x1 = np.random.randint(0, video_width - crop_w + 1)
        crop_y1 = np.random.randint(0, video_height - crop_h + 1)
        crop_x2, crop_y2 = crop_x1 + crop_w, crop_y1 + crop_h

        if not any(_overlaps(crop_x1, crop_y1, crop_x2, crop_y2, *d) for d in drowning_abs):
            return crop_x1, crop_y1, crop_x2, crop_y2

    return None


def transform_bbox_for_crop(bbox, crop_coords, video_width, video_height):
    """Transform bbox coordinates relative to cropped frame."""
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    bx_center = bbox[0] * video_width
    by_center = bbox[1] * video_height
    bw = bbox[2] * video_width
    bh = bbox[3] * video_height

    new_x = (bx_center - crop_x1) / crop_w
    new_y = (by_center - crop_y1) / crop_h
    new_w = min(bw / crop_w, 1.0)
    new_h = min(bh / crop_h, 1.0)

    return [1, new_x, new_y, new_w, new_h]  # Class ID = 1 for drowning


def save_annotation_file(annotation_path, bboxes):
    """Save bboxes to txt."""
    with open(annotation_path, "w") as f:
        for bbox in bboxes:
            f.write(" ".join(map(str, bbox)) + "\n")


# ------------------- Human detection via tracker -------------------

def detect_human_bboxes(video_path, tracker, skip_frames=6, min_track_frames=20):
    """
    Run YOLO+ByteTrack over the entire video (sampled every skip_frames frames).
    For each track_id, accumulate a merged bbox (union of all positions across frames).
    Return merged bboxes only for tracks that appeared in >= min_track_frames sampled frames.
    Returns [(x1, y1, x2, y2), ...] in absolute pixels.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open for human detection: {video_path}")
        return []

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracker.set_video_metadata(video_width=width, video_height=height, fps=fps)

    factor_fps = fps / 30.0
    skip_adj   = max(1, int(skip_frames * factor_fps))

    # {track_id: {"count": N, "x1": min_x1, "y1": min_y1, "x2": max_x2, "y2": max_y2}}
    tracks = {}

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_adj == 0:
            tracker.process_frame(frame.copy())

            inner     = getattr(tracker, "tracker", None)
            per_frame = getattr(inner, "tracking_per_frame", None)
            last      = per_frame[-1] if per_frame else {}

            for track_id, obj in last.items():
                b = obj.get("bbox")
                if b is None:
                    continue
                x1, y1, x2, y2 = map(int, b[:4])
                if x2 <= x1 or y2 <= y1:
                    continue
                if track_id not in tracks:
                    tracks[track_id] = {"count": 0, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                else:
                    t = tracks[track_id]
                    t["x1"] = min(t["x1"], x1)
                    t["y1"] = min(t["y1"], y1)
                    t["x2"] = max(t["x2"], x2)
                    t["y2"] = max(t["y2"], y2)
                tracks[track_id]["count"] += 1

        frame_count += 1

    cap.release()

    valid = [
        (t["x1"], t["y1"], t["x2"], t["y2"])
        for t in tracks.values()
        if t["count"] >= min_track_frames
    ]

    logger.info(
        f"  Human tracking: {len(tracks)} track(s), "
        f"{len(valid)} with >= {min_track_frames} frames  [{os.path.basename(video_path)}]"
    )
    return valid


# ------------------- Processing -------------------

def process_video_file(video_path, split, class_name, location_name, input_dir, save_dir,
                        min_crop_size, max_crop_size, num_drowning_crops, num_normal_crops,
                        factor_resize_bbox, tracker=None, skip_frames=6, min_track_frames=20):
    try:
        clip = VideoFileClip(video_path)
        w, h = clip.size

        annotation_path = os.path.join(input_dir, "annotations", os.path.splitext(os.path.basename(video_path))[0] + ".txt")
        bboxes = []
        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as f:
                bboxes = [list(map(float, line.split()[1:])) for line in f]

        # output dirs
        base_out_dir = os.path.join(save_dir, split, class_name, location_name)
        os.makedirs(base_out_dir, exist_ok=True)
        drowning_out_dir = os.path.join(base_out_dir, "drowning")
        normal_out_dir   = os.path.join(base_out_dir, "normal")
        os.makedirs(drowning_out_dir, exist_ok=True)
        os.makedirs(normal_out_dir,   exist_ok=True)
        annotations_out_dir = os.path.join(save_dir, "annotations")
        os.makedirs(annotations_out_dir, exist_ok=True)

        # Drowning crops
        for i in range(num_drowning_crops):
            for bbox in bboxes:
                crop_coords = generate_drowning_crop(bbox, w, h, min_crop_size, max_crop_size, factor_resize_bbox)
                crop_clip = clip.fl_image(lambda f, c=crop_coords: f[c[1]:c[3], c[0]:c[2]])
                unique_id = str(uuid.uuid4())
                crop_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_drowning_{i}_{unique_id}.mp4"
                crop_path = os.path.join(drowning_out_dir, crop_name)
                crop_clip.write_videofile(crop_path, codec="libx264", audio_codec="aac", fps=DESIRED_FPS, preset="ultrafast")
                ann = transform_bbox_for_crop(bbox, crop_coords, w, h)
                crop_ann_filename = os.path.splitext(crop_name)[0] + ".txt"
                crop_ann_path = os.path.join(annotations_out_dir, crop_ann_filename)
                save_annotation_file(crop_ann_path, [ann])
                crop_clip.close()

        # Normal crops — detect humans, pre-filter and shuffle so each crop gets a unique human
        valid_humans = []
        if tracker is not None and num_normal_crops > 0:
            raw_humans = detect_human_bboxes(video_path, tracker, skip_frames=skip_frames, min_track_frames=min_track_frames)
            drowning_abs = _drowning_abs(bboxes, w, h)
            valid_humans = [h for h in raw_humans if not any(_overlaps(*h, *d) for d in drowning_abs)]
            np.random.shuffle(valid_humans)
            logger.info(f"  {len(valid_humans)} valid human track(s) available for {num_normal_crops} normal crop(s)")

        for i in range(num_normal_crops):
            # Each crop gets its own unique human; fall back to random if exhausted
            human_for_crop = [valid_humans[i]] if i < len(valid_humans) else None
            crop_coords = generate_normal_crop(bboxes, w, h, min_crop_size, max_crop_size,
                                               human_bboxes_abs=human_for_crop)
            if crop_coords is None:
                logger.warning(f"  Could not find valid normal crop for {video_path} (iteration {i})")
                continue
            crop_clip = clip.fl_image(lambda f, c=crop_coords: f[c[1]:c[3], c[0]:c[2]])
            unique_id = str(uuid.uuid4())
            crop_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_normal_{i}_{unique_id}.mp4"
            crop_path = os.path.join(normal_out_dir, crop_name)
            crop_clip.write_videofile(crop_path, codec="libx264", audio_codec="aac", fps=DESIRED_FPS, preset="ultrafast")
            crop_clip.close()

    finally:
        clip.close() if "clip" in locals() else None


def process_split(input_dir, save_dir, min_crop_size, max_crop_size, num_drowning_crops,
                   num_normal_crops, factor_resize_bbox, tracker=None, skip_frames=6, min_track_frames=20):
    splits = ["train", "val", "test"]
    for split in splits:
        split_dir = os.path.join(input_dir, split)
        if not os.path.exists(split_dir):
            continue
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for location_name in os.listdir(class_dir):
                loc_dir = os.path.join(class_dir, location_name)
                if not os.path.isdir(loc_dir):
                    continue
                video_files = [os.path.join(loc_dir, f) for f in os.listdir(loc_dir) if f.lower().endswith(VIDEO_EXTENSIONS)]
                for vf in video_files:
                    process_video_file(
                        vf, split, class_name, location_name, input_dir, save_dir,
                        min_crop_size, max_crop_size, num_drowning_crops, num_normal_crops,
                        factor_resize_bbox, tracker=tracker, skip_frames=skip_frames,
                        min_track_frames=min_track_frames,
                    )


if __name__ == "__main__":
    load_dotenv()
    _model_variant   = os.getenv("MODEL_VARIANT", "x3d_s")
    _configs         = get_model_configs()
    _default_crop    = _configs.get(_model_variant, {}).get("crop_size", 182)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",          default=r"path/to/data")
    parser.add_argument("--save_dir",           default=r"path/to/data")
    parser.add_argument("--factor_resize_bbox", type=float, default=1.5,  help="Max crop size factor relative to bbox size")
    parser.add_argument("--min_crop_size",      type=int,   default=_default_crop, help=f"Min crop size (default from MODEL_VARIANT={_model_variant}: {_default_crop})")
    parser.add_argument("--max_crop_size",      type=int,   default=600)
    parser.add_argument("--num_drowning_crops", type=int,   default=0)
    parser.add_argument("--num_normal_crops",   type=int,   default=2)
    parser.add_argument("--skip_frames",        type=int,   default=6,    help="Frame sampling interval for human detection")
    parser.add_argument("--min_track_frames",   type=int,   default=20,   help="Min sampled frames a track_id must appear in to be considered a valid human")
    parser.add_argument("--imgsz",              type=int,   default=1280)
    parser.add_argument("--deque_length",       type=int,   default=26)
    parser.add_argument("--max_clusters",       type=int,   default=12)
    parser.add_argument("--skip_water_process_frames", type=int, default=3)
    parser.add_argument("--max_water_deque_size",      type=int, default=3)
    args = parser.parse_args()

    is_docker_env = os.path.exists("/.dockerenv")
    models_dir = "/models" if is_docker_env else "models"
    loader  = TensorRTModelLoader(models_dir=models_dir)
    tracker = loader.load_human_tracker(
        imgsz=args.imgsz,
        max_clusters=args.max_clusters,
        skip_water_process_frames=args.skip_water_process_frames,
        max_water_deque_size=args.max_water_deque_size,
        detect_water=True,
        deque_length=args.deque_length,
        use_behavior_analysis=os.getenv("USE_BEHAVIOR_ANALYSIS", "True").lower() == "true",
    )

    if tracker is None:
        raise RuntimeError("Tracker failed to initialize")

    process_split(
        args.input_dir, args.save_dir, args.min_crop_size, args.max_crop_size,
        args.num_drowning_crops, args.num_normal_crops, args.factor_resize_bbox,
        tracker=tracker, skip_frames=args.skip_frames, min_track_frames=args.min_track_frames,
    )
