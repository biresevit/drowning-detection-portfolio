import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from ultralytics.utils.checks import cuda_is_available
from utils import resize_big_frame, adjust_cols_rows_human_detection, get_adjusted_size
import cv2
import logging
from src.action_recognition.load_models_ import load_human_tracker

logger = logging.getLogger()


def main():
    if cuda_is_available():
        logger.info("CUDA is available!")
    else:
        logger.info("CUDA is not available.")
    # Open the video file
    video_path = r"path/to/data"
    if not os.path.isfile(video_path):
        raise ValueError("Video is not found")
    scale = 1

    cap = cv2.VideoCapture(video_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width, new_height = get_adjusted_size(video_width, video_height)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        f"Video resolution: {video_width}x{video_height}, FPS: {fps}, Total frames: {total_frames}"
    )

    imgsz = int(max(video_width * scale, video_height * scale))
    skip_frames = 1
    deque_lenght = 26
    tracker = load_human_tracker(
        imgsz=960,
        max_clusters=12,
        skip_water_process_frames=3,
        max_water_deque_size=3,
        deque_lenght=deque_lenght,
    )

    tracker.set_video_metadata(video_width, video_height, fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("detections.mp4", fourcc, fps, (new_width, new_height))
    frame_id = 0
    # skip_clusters = deque_lenght // 2
    skip_clusters = 5
    counter_cluster = 0
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if not success:
            break

        if frame_id % skip_frames == 0:
            # Run YOLO tracking on the frame
            frame = tracker.process_frame(frame)

            # Visualize the results on the frame
            tracker.draw_object_detections(frame)
            counter_cluster += 1
            if (
                counter_cluster % skip_clusters == 0
                and len(tracker.tracker.tracking_per_frame) == deque_lenght
            ):
                counter_cluster = 0
                tracker.compute_clusters()

            tracker.draw_clusters(frame, tracker.merged_clusters)

            out.write(frame)
            # Display the annotated frame
            cv2.imshow("YOLO Tracking", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_id += 1

    cv2.destroyAllWindows()
    out.release()


if __name__ == "__main__":
    main()
