#!/usr/bin/env python3
"""
process_clusters.py

Process videos arranged in splits (train/val/test), detect clusters from tracker,
and save cropped cluster videos + annotations while preserving split/class/location
folder structure.

Expected input layout (video_dir):
    <video_dir>/
        train/
            drowning/
                lake/
                    vid1.mp4
                river/
            normal/
        val/
        test/

Output:
    save_videos_dir/<split>/<class>/<location>/<label>/<video>.mp4
    save_annotations_dir/<split>/<class>/<location>/<video>.txt  (only for drowning labels)
"""

import os
import sys
import cv2
import argparse
import logging
from collections import deque
from tqdm import tqdm

# add project parent to path so imports like action_recognition.load_models_tensorrt work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from action_recognition.load_models_tensorrt import TensorRTModelLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

show_video = False


def is_bbox_within_cluster(cluster_bbox, bbox, threshold_overlap):
    """Return True if bbox overlaps cluster_bbox by at least threshold_overlap (relative to bbox area)."""
    x1_min, y1_min, x1_max, y1_max = cluster_bbox
    x2_min, y2_min, x2_max, y2_max = bbox

    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    overlap_area = x_overlap * y_overlap

    bbox_area = (x2_max - x2_min) * (y2_max - y2_min)
    if bbox_area <= 0:
        return False
    overlap_perc = overlap_area / bbox_area
    return overlap_perc >= threshold_overlap


def save_cluster_videos(
    accumulated_clusters,
    frame_store,
    save_videos_dir,
    video_path,
    fps,
    drowning_bboxes,
    threshold_overlap,
    tiny_overlap_threshold,
    small_overlap_threshold,
    mid_overlap_threshold,
    save_annotations_dir,
    split_name,
    class_name,
    location_name,
):
    """
    accumulated_clusters: list of (frame_count, clusters) where clusters is a dict {id: {'bbox':[...] , ...}}
    frame_store: list of frames corresponding to original frames captured (should map to indices used when clustering)
    """

    base_video_name = os.path.splitext(os.path.basename(video_path))[0]

    for cluster_count, (frame_count, clusters) in enumerate(accumulated_clusters):
        for obj_id, obj in clusters.items():
            try:
                x_min, y_min, x_max, y_max = map(int, obj["bbox"])
                new_width = max(1, x_max - x_min)
                new_height = max(1, y_max - y_min)
                cluster_bbox = (x_min, y_min, x_max, y_max)

                is_drowning = any(is_bbox_within_cluster(cluster_bbox, bbox, threshold_overlap) for bbox in drowning_bboxes)
                is_mid_overlap = not is_drowning and any(is_bbox_within_cluster(cluster_bbox, bbox, mid_overlap_threshold) for bbox in drowning_bboxes)
                is_small_overlap = not is_drowning and not is_mid_overlap and any(is_bbox_within_cluster(cluster_bbox, bbox, small_overlap_threshold) for bbox in drowning_bboxes)
                is_tiny_overlap = not is_drowning and not is_mid_overlap and not is_small_overlap and any(is_bbox_within_cluster(cluster_bbox, bbox, tiny_overlap_threshold) for bbox in drowning_bboxes)

                if is_drowning:
                    label = "drowning"
                elif is_mid_overlap:
                    label = "mid_overlap"
                elif is_small_overlap:
                    label = "small_overlap"
                elif is_tiny_overlap:
                    label = "tiny_overlap"
                else:
                    label = "normal"

                # build target path: save_videos_dir/<split>/<class>/<location>/<label>/
                video_save_dir = os.path.join(save_videos_dir, split_name, class_name, location_name, label)
                os.makedirs(video_save_dir, exist_ok=True)

                video_name = f"{base_video_name}_cluster_{cluster_count}_{obj_id}"
                video_filename = os.path.join(video_save_dir, f"{video_name}.mp4")

                # write mp4 with mp4v
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(video_filename, fourcc, max(1.0, fps), (new_width, new_height))

                # Write frames from frame_store. frame_store is list of (frame_index, frame)
                for idx, frm in frame_store:
                    # use the stored frames that correspond to the captured frames
                    cluster_frame = frm[y_min:y_max, x_min:x_max]
                    if cluster_frame.shape[1] != new_width or cluster_frame.shape[0] != new_height:
                        # ensure correct shape (can happen due to cropping edges)
                        cluster_frame = cv2.resize(cluster_frame, (new_width, new_height))
                    out.write(cluster_frame)
                out.release()

                # Save annotation if drowning
                if is_drowning:
                    annotation_save_dir = os.path.join(save_annotations_dir, split_name, class_name, location_name)
                    os.makedirs(annotation_save_dir, exist_ok=True)
                    annotation_file = os.path.join(annotation_save_dir, f"{video_name}.txt")
                    with open(annotation_file, "w") as f:
                        for bbox in drowning_bboxes:
                            if is_bbox_within_cluster(cluster_bbox, bbox, threshold_overlap):
                                x1, y1, x2, y2 = bbox
                                adj_x1 = max(0, x1 - x_min)
                                adj_y1 = max(0, y1 - y_min)
                                adj_x2 = min(new_width, x2 - x_min)
                                adj_y2 = min(new_height, y2 - y_min)
                                adj_w = max(0, adj_x2 - adj_x1)
                                adj_h = max(0, adj_y2 - adj_y1)
                                if adj_w <= 0 or adj_h <= 0:
                                    continue
                                x_center = (adj_x1 + adj_x2) / 2 / new_width
                                y_center = (adj_y1 + adj_y2) / 2 / new_height
                                width = adj_w / new_width
                                height = adj_h / new_height
                                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            except Exception as e:
                logger.exception(f"Error saving cluster video for {video_path}: {e}")


def load_annotation(video_path, annotation_dir, video_width, video_height):
    """Load YOLO-like normalized annotations (class x_center y_center w h) and convert to pixel bboxes."""
    annotations = []
    video_file_name = os.path.splitext(os.path.basename(video_path))[0]
    annotation_file = os.path.join(annotation_dir, f"{video_file_name}.txt")
    if not os.path.exists(annotation_file):
        return []

    with open(annotation_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                annotations.append([float(x) for x in parts])
            except Exception:
                continue

    bboxes = []
    for row in annotations:
        # row: [class, x_center, y_center, width, height]
        x_center = row[1] * video_width
        y_center = row[2] * video_height
        width = row[3] * video_width
        height = row[4] * video_height
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)
        bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes


def process_video(
    tracker,
    video_path,
    video_index,
    total_videos,
    skip_frames,
    skip_cluster_compute,
    save_frames_dir,
    save_videos_dir,
    annotation_dir,
    threshold_overlap,
    tiny_overlap_threshold,
    small_overlap_threshold,
    mid_overlap_threshold,
    save_annotations_dir,
    max_cluster_compute,
    split_name,
    class_name,
    location_name,
    max_store_frames=8000,
):
    """
    Process a single video and save cluster crops.
    all_frames_store keeps all original frames (for saving videos)
    sampled_frames_store keeps frames at skip_frames intervals (for tracker/cluster computation)
    """
    try:
        os.makedirs(save_frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = total_frame_count / fps if fps > 0 else 0.0

        window_title = f"{os.path.basename(video_path)} - {width}x{height} - {fps:.2f} FPS - {duration:.2f} sec"
        drowning_bboxes = load_annotation(video_path, annotation_dir, width, height)

        factor_fps = fps / 30.0
        skip_frames_adj = max(1, int(skip_frames * factor_fps))
        skip_cluster_compute_adj = max(1, int(skip_cluster_compute * factor_fps))

        tracker.set_video_metadata(video_width=width, video_height=height, fps=fps)

        frame_count = 0
        all_frames_store = []       # store all original frames for saving clusters
        sampled_frames_store = []   # store sampled frames for tracker/cluster computation
        accumulated_clusters = []
        count_cluster_compute = 0

        if show_video:
            cv2.namedWindow(window_title)
            cv2.moveWindow(window_title, 0, 0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # store all frames for final video saving
            all_frames_store.append(frame.copy())
            if len(all_frames_store) > max_store_frames:
                logger.warning(f"Video too long: keeping only last {max_store_frames} frames")
                all_frames_store = all_frames_store[-max_store_frames:]

            # store sampled frames for cluster computation
            if frame_count % skip_frames_adj == 0:
                sampled_frames_store.append((frame_count, frame.copy()))

                # process sampled frame with tracker
                proc_frame = tracker.process_frame(frame.copy())
                tracker.draw_object_detections(proc_frame)

                is_cluster_computed = False
                if (
                    count_cluster_compute < max_cluster_compute
                    and (frame_count % skip_cluster_compute_adj == 0 or len(tracker.merged_clusters) == 0)
                    and hasattr(tracker, "tracker")
                    and len(getattr(tracker.tracker, "tracking_per_frame", [])) == getattr(tracker.tracker, "deque_lenght", 0)
                ):
                    tracker.compute_clusters()
                    count_cluster_compute += 1
                    is_cluster_computed = True

                tracker.draw_clusters(proc_frame, getattr(tracker, "merged_clusters", {}))

                if is_cluster_computed:
                    # store a snapshot of clusters
                    clusters_copy = {}
                    merged = getattr(tracker, "merged_clusters", {})
                    for k, v in merged.items():
                        clusters_copy[k] = v.copy() if isinstance(v, dict) else v
                    accumulated_clusters.append((frame_count, clusters_copy))

                    # optional: save example frame
                    if save_frames_dir:
                        frame_filename = os.path.join(
                            save_frames_dir,
                            f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count}.jpg",
                        )
                        cv2.imwrite(frame_filename, proc_frame)

                if show_video:
                    cv2.imshow(window_title, proc_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frame_count += 1

        cap.release()
        if show_video:
            cv2.destroyAllWindows()

        # save clusters using ALL frames
        if not all_frames_store:
            logger.warning(f"No frames captured for {video_path}, skipping cluster saving.")
        else:
            save_cluster_videos(
                accumulated_clusters,
                [(i, all_frames_store[i]) for i in range(len(all_frames_store))],
                save_videos_dir,
                video_path,
                fps,
                drowning_bboxes,
                threshold_overlap,
                tiny_overlap_threshold,
                small_overlap_threshold,
                mid_overlap_threshold,
                save_annotations_dir,
                split_name,
                class_name,
                location_name,
            )

    except Exception as e:
        logger.exception(f"Exception in process_video {video_path}: {e}")



def find_video_files(video_dir):
    """Find all video files under video_dir, return list of absolute paths."""
    exts = (".mp4", ".avi", ".mov", ".mkv")
    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            if f.lower().endswith(exts):
                video_files.append(os.path.join(root, f))
    return sorted(video_files)


def extract_split_class_location(video_path, video_dir):
    """
    Given a video_path under video_dir (which should be the splits root OR a specific split),
    return (split_name, class_name, location_name).

    Works for:
      video_dir/.../splits              => video_path = splits/train/drowning/lake/vid.mp4
      video_dir/.../splits/train        => video_path = train/drowning/lake/vid.mp4
    """
    rel = os.path.relpath(video_path, video_dir)
    parts = rel.split(os.sep)

    # if video_dir ends at splits root: rel = train/drowning/lake/vid.mp4
    if len(parts) >= 4:
        split_name = parts[0]
        class_name = parts[1]
        location_name = parts[2]
    # if video_dir already points to a split (train/val/test): rel = drowning/lake/vid.mp4
    elif len(parts) >= 3:
        split_name = os.path.basename(video_dir)
        class_name = parts[0]
        location_name = parts[1]
    else:
        raise ValueError(f"Cannot extract split/class/location from path: {video_path} relative to {video_dir}")

    return split_name, class_name, location_name


def resolve_annotation_root(video_dir):
    """
    Determine annotation root as the parent-of-parent of video_dir when video_dir points
    to a split folder or splits root.

    Example:
      video_dir = /data/splits            -> annotation_root = /data/annotations
      video_dir = /data/splits/train      -> annotation_root = /data/annotations
    """
    splits_dir = video_dir
    # if video_dir ends with train/val/test we want its parent (splits)
    base_name = os.path.basename(video_dir)
    if base_name in ("train", "val", "test"):
        splits_dir = os.path.dirname(video_dir)

    dataset_root = os.path.dirname(splits_dir)
    annotation_dir = os.path.join(dataset_root, "annotations")
    return annotation_dir


def main(
    tracker,
    video_dir,
    skip_frames,
    skip_cluster_compute,
    save_frames_dir,
    save_videos_dir,
    threshold_overlap,
    tiny_overlap_threshold,
    small_overlap_threshold,
    mid_overlap_threshold,
    save_annotations_dir,
    max_cluster_compute,
    max_store_frames,
):
    annotation_dir = resolve_annotation_root(video_dir)
    os.makedirs(save_annotations_dir, exist_ok=True)

    video_files = find_video_files(video_dir)
    total_videos = len(video_files)
    logger.info(f"Found {total_videos} videos under {video_dir}. Annotation root: {annotation_dir}")

    for idx, video_path in enumerate(tqdm(video_files, desc="Processing videos"), start=1):
        try:
            split_name, class_name, location_name = extract_split_class_location(video_path, video_dir)
        except ValueError as e:
            logger.error(str(e))
            continue

        logger.info(f"[{idx}/{total_videos}] {split_name}/{class_name}/{location_name} -> {video_path}")

        process_video(
            tracker,
            video_path,
            idx,
            total_videos,
            skip_frames,
            skip_cluster_compute,
            save_frames_dir,
            save_videos_dir,
            annotation_dir,
            threshold_overlap,
            tiny_overlap_threshold,
            small_overlap_threshold,
            mid_overlap_threshold,
            save_annotations_dir,
            max_cluster_compute,
            split_name,
            class_name,
            location_name,
            max_store_frames,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos in a directory tree and save cluster crops.")
    parser.add_argument("--video_dir", type=str, default=r"path/to/data", help="Root folder containing splits (train/val/test) or a single split folder.")
    parser.add_argument("--save_videos_dir", type=str, default=r"path/to/data", help="Directory to save processed cluster videos.")    
    parser.add_argument("--save_annotations_dir", type=str, default=r"path/to/data", help="Directory to save cluster annotations.")    
    parser.add_argument("--deque_length", type=int, default=26, help="Length of the deque for tracking.")
    parser.add_argument("--threshold_frames", type=int, default=1, help="Threshold frames for tracking (unused here).")
    parser.add_argument("--max_cluster_compute", type=int, default=1, help="Max times to compute clusters per video.")
    parser.add_argument("--num_total_crops", type=int, default=12, help="Max clusters to track.")
    parser.add_argument("--skip_frames", type=int, default=6, help="Number of frames to skip during tracking.")
    parser.add_argument("--skip_cluster_compute", type=int, default=150, help="Frames to skip between cluster computations.")
    parser.add_argument("--save_frames_dir", type=str, default="cluster_frames", help="Directory to save debug frames.")
    parser.add_argument("--threshold_overlap", type=float, default=0.8, help="Overlap threshold to mark drowning.")
    parser.add_argument("--small_overlap_threshold", type=float, default=0.4, help="small overlap threshold.")
    parser.add_argument("--mid_overlap_threshold", type=float, default=0.6, help="mid overlap threshold.")
    parser.add_argument("--tiny_overlap_threshold", type=float, default=0.2, help="tiny overlap threshold.")
    parser.add_argument("--skip_water_process_frames", type=int, default=3)
    parser.add_argument("--max_water_deque_size", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max_store_frames", type=int, default=8000, help="Max sampled frames to keep in memory per video (safeguard).")

    args = parser.parse_args()

    is_docker_env = os.path.exists("/.dockerenv")
    models_dir = "/models" if is_docker_env else "models"
    loader = TensorRTModelLoader(models_dir=models_dir)
    tracker = loader.load_human_tracker(
        imgsz=args.imgsz,
        max_clusters=args.num_total_crops,
        skip_water_process_frames=args.skip_water_process_frames,
        max_water_deque_size=args.max_water_deque_size,
        detect_water=True,
        deque_length=args.deque_length,
        use_behavior_analysis=os.getenv("USE_BEHAVIOR_ANALYSIS", "True").lower() == "true",
    )

    if tracker is None:
        raise RuntimeError("Tracker failed to initialize")

    main(
        tracker,
        args.video_dir,
        args.skip_frames,
        args.skip_cluster_compute,
        args.save_frames_dir,
        args.save_videos_dir,
        args.threshold_overlap,
        args.tiny_overlap_threshold,
        args.small_overlap_threshold,
        args.mid_overlap_threshold,
        args.save_annotations_dir,
        args.max_cluster_compute,
        args.max_store_frames,
    )
