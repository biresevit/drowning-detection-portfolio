import glob
import logging
import os
import queue
import threading
import time
import uuid
import av
import cv2
from numpy import ndarray

from utils_lib.path_helpers import normalize_path
from utils_lib.draw_helpers import draw_alert_notification
from utils_lib.drowning_detection_helpers import is_drowning_detected
logger = logging.getLogger()

def save_frame(frame, save_frames, num_saved_frames, last_save_frame, total_seconds_save_frames):
    """Save a frame to disk if conditions are met."""
    if last_save_frame is None:
        last_save_frame = 0
    if save_frames and num_saved_frames < 300 and can_save_frames(last_save_frame, total_seconds_save_frames):
        last_save_frame = time.perf_counter()
        frames_folder = "frames"
        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)
        cv2.imwrite(os.path.join(frames_folder, f"{uuid.uuid4().hex}.png"), frame)
        num_saved_frames += 1
    return num_saved_frames, last_save_frame

def can_save_frames(last_save, total_seconds):
    return last_save is None or time.perf_counter() - last_save > total_seconds

def save_drowning_frame(frame_alert, video_source, is_live_video, frames_folder="frames"):
    """
    Save a frame to disk when drowning is detected for non-live videos.
    """
    if is_live_video:
        return  # Do nothing for live video

    try:
        # Determine base name for the file
        base_name = (
            os.path.splitext(os.path.basename(str(video_source)))[0]
            if isinstance(video_source, str)
            else "live_video"
        ).replace(" ", "_")  # Replace spaces with underscores

        # Ensure the folder exists
        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)

        # Build image path and save the frame
        img_path = os.path.join(frames_folder, f"{base_name}.png")
        cv2.imwrite(img_path, frame_alert)

        logger.info(f"Drowning detected, wrote image: {img_path}")

    except Exception as e:
        logger.error(f"Failed to save drowning frame: {e}")

def process_and_store_frame(
    frame: ndarray,
    video_stream,
    target_width: int,
    target_height: int,
    disable_alerts: bool,
    is_uploading_video,
    use_cloud: bool,
    save_alerts_disk: bool,
    frame_queue_storage,
    frame_queue_storage_lock: threading.Lock,
    is_live_video: bool,
    put_in_queue: bool,
    frame_queue: queue.Queue,
    last_frame,
):
    """
    Process a video frame: resize, convert to YUV420p, store in frame queue,
    and optionally keep a copy for non-live video or put in a queue for display.

    Returns updated last_frame and frame_copy.
    """
    frame_copy = None  # Ensure frame_copy exists even if an exception occurs

    try:
        # Make a copy of the frame to avoid modifying the original
        frame_copy = frame.copy()

        # Resize frame if target dimensions are different from stream
        if video_stream.width != target_width or video_stream.height != target_height:
            frame_copy = cv2.resize(frame_copy, (target_width, target_height))

        # Convert frame to YUV420p and store if alerts are enabled
        if not disable_alerts and not is_uploading_video.is_set():
            av_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2YUV_I420)
            av_frame = av.VideoFrame.from_ndarray(av_frame, format="yuv420p")

            if use_cloud or save_alerts_disk:
                with frame_queue_storage_lock:
                    frame_queue_storage.append(av_frame)

        # Keep a copy of the last frame for non-live videos
        if not is_live_video:
            last_frame = frame_copy

        # Optionally put the frame into a queue for display
        if put_in_queue:
            try:
                frame_queue.put_nowait(frame_copy)
            except queue.Full:
                logger.warning("Frame queue full, skipping frame")

        return last_frame, frame_copy

    except Exception as e:
        # Log any exception without breaking the processing loop
        logger.exception(f"Failed to process frame: {e}")
        return last_frame, frame_copy

def delete_temp_frames(temp_dir):
    # Remove remaining files in temp_dir
    for file_path in glob.glob(os.path.join(temp_dir, "*.jpg")):
        try:
            os.remove(file_path)
        except Exception:
            pass
    try:
        os.rmdir(temp_dir)
    except Exception as e:
        logger.warning(f"Error removing temp directory {temp_dir}: {e}")

def get_required_frames(samples_1_batch, batch_size, slide_window_perc):
    return samples_1_batch + (batch_size - 1) * int(slide_window_perc * samples_1_batch)

def wait_for_last_results(
    last_frame,
    processing_done_event,
    shared_data_queue,
    human_tracker,
    display_water_mask,
    conf_high_thres,
    last_alert_time,
    total_seconds,
    video_source,
):
    from utils_lib.processing_drowning_helpers import is_enough_time_elapsed

    predicted_class = None
    info_predictions = None
    confidence = None
    bbox = None
    if last_frame is not None:
        while not processing_done_event.is_set():
            try:
                try:
                    predicted_class, confidence, bbox, info_predictions = (
                        shared_data_queue.get_nowait()
                    )
                except queue.Empty:
                    pass

                if predicted_class is not None and predicted_class != "":
                    is_drowning = is_drowning_detected(
                        predicted_class, confidence, conf_high_thres, info_predictions
                    )

                    if is_drowning and is_enough_time_elapsed(
                        last_alert_time, total_seconds
                    ):
                        frame_alert = draw_alert_notification(
                            last_frame.copy(),
                            predicted_class,
                            confidence,
                            bbox,
                            human_tracker,
                            display_water_mask,
                        )
                        
                        save_drowning_frame(frame_alert, video_source, False)

                        logger.info(
                            f"Drowning is detected in wait_for_last_results, writing image: {video_source}"
                        )
                        return
            except Exception as e:
                logger.error(f"Error during post-processing display: {e}")
                break


def save_image_to_disk(uuid, frame_alert):
    try:
        output_dir = normalize_path("/alert_images")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{uuid}.jpg")
        cv2.imwrite(filename, frame_alert)
        logger.info(f"Alert image saved to {filename}")
    except:
        logger.error(f"Error in save_alert_image_to_disk. Failed to save alert frame {uuid}.")

def save_video_buffer_to_disk(uuid_name, buffer_original):
    try:
        output_dir = normalize_path("/video_alerts")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{uuid_name}.mp4")

        with open(filename, "wb") as f:
            f.write(buffer_original.getbuffer())

        logger.info(f"Video saved locally to {filename}")
    except:
        logger.error(f"Error in save_video_to_disk. Failed to save video alert {uuid_name}")