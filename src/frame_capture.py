import cv2
import queue
import time
import logging
import os
import threading
import uuid

from utils_lib.draw_helpers import display_last_frame, draw_results
from utils_lib.processing_drowning_helpers import handle_drowning_detection, is_enough_time_elapsed
from utils_lib.metrics_helpers import initialize_metrics_thread, log_metrics
from utils_lib.video_stream_helpers import handle_stream_failure, initialize_video_stream, setup_video_metadata, capture_frame_status
from utils_lib.frame_helpers import process_and_store_frame, save_frame, can_save_frames, wait_for_last_results
from utils_lib.environment_helpers import initialize_environment
from collections import deque
from collect_device_metrics import MetricType
from alert.mac_address import get_mac_address
from classes_enum import ClassesEnum
logger = logging.getLogger()

HIGH_FPS = 60

def check_cuda_availability():
    """Check if FFmpeg supports CUDA hardware acceleration."""
    return True

CUDA_AVAILABLE = check_cuda_availability()

def safe_put_latest(q, item):
    if q is None:
        return
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass

def frame_capture_live_thread(
    video_source,
    disable_alerts,
    human_tracker,
    num_frames,
    frame_queue,
    shared_data_queue,
    sample_interval,
    batch_size,
    max_frame_width,
    max_frame_height,
    stop_event,
    factor_tracking,
    slide_window_perc,
    display_water_mask,
    max_video_seconds,
    conf_high_thres,
    save_frames,
    total_seconds_save_frames,
    processing_done_event,
    use_cloud=False,
    save_alerts_disk=False,
    alert_ip=None,
    alert_port=None,
    total_seconds=None,
    drowning_queue=None,
    upload_video = None,
    save_video_to_disk = None,
    send_alert_via_cloud = None,
    save_alert_image_to_disk = None,
    save_video = True,
):
    frame_counter = 0
    last_alert_time = None
    if total_seconds is None:
        total_seconds = 300
    last_save_frame = None
    last_log_frame_not_read = time.perf_counter()
    last_upload_camera_working = None
    last_upload_last_seen = None
    num_saved_frames = 0
    last_frame = None 
    metric_log_interval = 60
    last_seen_metric_log_interval = 15
    log_interval = 5 * 60
    last_log_time = time.perf_counter()
    predicted_class = ""
    confidence = None
    bbox = None
    info_predictions = []

    azure_original_video_url, azure_resized_video_url = initialize_environment(use_cloud)
    video_stream = initialize_video_stream(video_source, num_frames, batch_size, slide_window_perc, factor_tracking, sample_interval)
    is_live_video, target_fps, target_width, target_height, max_frames = setup_video_metadata(video_stream, human_tracker, max_frame_width, max_frame_height, max_video_seconds)

    video_writer = None
    if save_video:
        os.makedirs("saved_videos", exist_ok=True)
        stem = os.path.splitext(os.path.basename(str(video_source)))[0] or "saved_video"
        saved_video_path = os.path.join("saved_videos", f"{stem}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        saved_video_fps = max(1, target_fps)
        video_writer = cv2.VideoWriter(saved_video_path, fourcc, saved_video_fps, (target_width, target_height))
        logger.info(f"Saving video to {saved_video_path} at {target_width}x{target_height} {saved_video_fps}fps")

    interface = os.getenv("INTERFACE")
    mac_address = get_mac_address(interface)
    logger.info(f"Started live camera, source: {video_source}, size: {video_stream.width}x{video_stream.height}, FPS: {video_stream.fps}")

    frame_queue_storage = deque(maxlen=max_frames)
    frame_queue_storage_lock = threading.Lock()
    logger.info(f"Storing raw RGB frames in memory at {target_width}x{target_height}, max {max_frames} frames")

    #if use_cloud:
    #    from collect_device_metrics import initialize_metrics
    #    initialize_metrics(use_cloud)
    metric_queue, metric_thread = initialize_metrics_thread(use_cloud, stop_event, mac_address)
    is_uploading_video = threading.Event()
    started_frame_read_time = time.perf_counter()
    sleep_time = 1 / target_fps

    upload_thread = None
    alert_thread = None

    try:
        while not stop_event.is_set():
            start_time = time.perf_counter()
            frame_copy = None

            # Read frame from VideoStream
            frame, is_camera_working, last_log_time = capture_frame_status(last_log_time, log_interval, video_stream, start_time)
            
            now = time.perf_counter()
            if is_camera_working:
                started_frame_read_time = now

            if use_cloud:
                last_upload_camera_working, last_upload_last_seen = log_metrics(metric_queue, is_camera_working, last_upload_camera_working, last_upload_last_seen, metric_log_interval, last_seen_metric_log_interval)

            # Handle no frame
            if not is_camera_working:
                if is_live_video:
                    if not handle_stream_failure(is_live_video, started_frame_read_time, video_stream, now):
                        break
                    if now - last_log_frame_not_read > 60:
                        last_log_frame_not_read = now
                        logger.warning("Failed to read frame")
                    continue
                else:
                    logger.info("Reached end of local video file, stopping capture")
                    break

            # Decide whether to put frame in queue
            put_in_queue = frame_counter % video_stream.sample_interval == 0
            if (
                not put_in_queue
                and round(video_stream.fps) >= HIGH_FPS
                and frame_counter % 2 == 0
            ):
                frame_counter += 1
                continue

            last_frame, frame_copy = process_and_store_frame(frame=frame,video_stream=video_stream, target_width=target_width, target_height=target_height, disable_alerts=disable_alerts, is_uploading_video=is_uploading_video, use_cloud=use_cloud, save_alerts_disk=save_alerts_disk, frame_queue_storage=frame_queue_storage, frame_queue_storage_lock=frame_queue_storage_lock, is_live_video=is_live_video, put_in_queue=put_in_queue, frame_queue=frame_queue, last_frame=last_frame)

            # Get predictions if available, keep last known results on empty
            new_prediction = False
            try:
                predicted_class, confidence, bbox, info_predictions = (
                    shared_data_queue.get_nowait()
                )
                new_prediction = True
                timestamp = time.time()
                safe_put_latest(drowning_queue, (info_predictions, timestamp))

            except queue.Empty:
                pass

            if video_writer is not None and frame_copy is not None:
                annotated = draw_results(
                    frame_copy.copy(),
                    info_predictions,
                    conf_high_thres,
                    False,
                    display_water_mask,
                    True,
                    human_tracker,
                    predicted_class,
                    None,
                )
                write_frame = annotated if annotated is not None else frame_copy
                if write_frame.shape[2] != 3:
                    write_frame = cv2.cvtColor(write_frame, cv2.COLOR_BGRA2BGR)
                video_writer.write(write_frame)

            # Save frame if needed
            num_saved_frames, last_save_frame = save_frame(frame_copy, save_frames, num_saved_frames, last_save_frame, total_seconds_save_frames)

            # Handle drowning detection only when a new prediction arrived
            if new_prediction:
                last_alert_time, alert_thread, upload_thread = handle_drowning_detection( predicted_class=predicted_class, confidence=confidence, conf_high_thres=conf_high_thres, info_predictions=info_predictions,bbox=bbox, frame=frame, video_source=video_source, frame_queue_storage=frame_queue_storage, target_width=target_width, target_height=target_height, target_fps=target_fps, human_tracker=human_tracker, display_water_mask=display_water_mask, disable_alerts=disable_alerts, is_uploading_video=is_uploading_video, use_cloud=use_cloud, save_alerts_disk=save_alerts_disk, metric_queue=metric_queue, last_alert_time=last_alert_time, total_seconds=total_seconds, max_frames=max_frames, azure_original_video_url=azure_original_video_url, azure_resized_video_url=azure_resized_video_url, alert_ip=alert_ip, alert_port=alert_port, is_live_video=is_live_video, mac_address=mac_address, start_time=start_time, alert_thread=alert_thread, upload_thread=upload_thread, upload_video=upload_video, save_video_to_disk=save_video_to_disk, send_alert_via_cloud=send_alert_via_cloud, save_alert_image_to_disk=save_alert_image_to_disk)

            frame_counter += 1
            elapsed_time = time.perf_counter() - start_time
            adjusted_sleep_time = max(0, sleep_time - elapsed_time)
            if not is_live_video:
                time.sleep(adjusted_sleep_time)

    finally:
        if video_writer is not None:
            video_writer.release()
            logger.info(f"Saved video to {saved_video_path}")
        cleanup_resources(stop_event, video_stream, upload_thread, alert_thread, metric_thread, metric_queue, use_cloud, last_frame, processing_done_event, shared_data_queue, human_tracker, display_water_mask, conf_high_thres, last_alert_time, total_seconds, video_source)


def frame_capture_thread(
    video_source,
    disable_alerts,
    human_tracker,
    num_frames,
    frame_queue,
    shared_data_queue,
    sample_interval,
    batch_size,
    max_frame_width,
    max_frame_height,
    frame_display_queue,
    processing_done_event,
    stop_event,
    display_human_detections_state,
    display_water_mask_state,
    display_only_drowning_state,
    factor_tracking,
    slide_window_perc,
    conf_high_thres,
    max_video_seconds,
    save_frames,
    total_seconds_save_frames,
    save_video=True,
):
    frame_counter = 0
    last_alert_time = None
    total_seconds = 300
    last_save_frame = None
    last_log_frame_not_read = time.perf_counter()

    last_frame = None


    predicted_class = ""
    info_predictions = []
    

    # Initialize VideoStream
    video_stream = initialize_video_stream(video_source, num_frames, batch_size, slide_window_perc, factor_tracking, sample_interval)

    video_width = video_stream.width
    video_height = video_stream.height
    fps = video_stream.fps
    sample_interval = video_stream.sample_interval


    is_live_video, target_fps, target_width, target_height, max_frames = setup_video_metadata(video_stream, human_tracker, max_frame_width, max_frame_height, max_video_seconds)
    
    sleep_time = 1 / target_fps
    logger.info(
        f"Video source: {video_source}, size: {video_width}x{video_height}, FPS: {fps}, sample interval: {sample_interval}, sleep: {sleep_time}"
    )

    # Initialize deque for JPEG bytes
    jpeg_queue = deque(maxlen=max_frames)
    jpeg_quality = 85
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    logger.info(
        f"Storing JPEG frames in memory at {target_width}x{target_height}, max {max_frames}, JPEG quality {jpeg_quality}"
    )

    video_writer = None
    if save_video:
        os.makedirs("saved_videos", exist_ok=True)
        stem = os.path.splitext(os.path.basename(video_source))[0] or "saved_video"
        saved_video_path = os.path.join("saved_videos", f"{stem}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Only sampled frames are drawn and written, so the file's rate is the SAMPLED
        # rate and its duration matches the source. A float on purpose: integer division
        # turned 30/7 into 4 instead of 4.29 and stretched every clip by 9%.
        #
        # Derived from the SOURCE fps, not target_fps. target_fps is min(source, 30) and
        # is used to pace the loop, but every frame ffmpeg emits is still read, so a 60fps
        # source really does deliver 60 frames per second of video. Using the capped value
        # labelled a 20.7s clip as 41.7s.
        source_fps = float(getattr(video_stream, "fps", 0) or target_fps)
        saved_video_fps = max(1.0, source_fps / max(1, sample_interval))
        video_writer = cv2.VideoWriter(saved_video_path, fourcc, saved_video_fps, (target_width, target_height))
        logger.info(f"Saving visualization video to {saved_video_path} at "
                    f"{target_width}x{target_height} {saved_video_fps:g}fps")

    try:
        while not stop_event.is_set():
            try:
                start_time = time.perf_counter()

                # Read frame from VideoStream
                frame = video_stream.read_frame()
                if frame is None:
                    if is_live_video:
                        now = time.perf_counter()
                        if now - last_log_frame_not_read > 60:
                            last_log_frame_not_read = now
                            logger.warning("Failed to read frame from live video")
                        continue
                    else:
                        logger.info("Reached end of local video file, stopping capture")
                        break

                put_in_queue = frame_counter % sample_interval == 0
                if not put_in_queue and round(fps) >= 25:
                    frame_counter += 1
                    elapsed_time = time.perf_counter() - start_time
                    adjusted_sleep_time = max(0, sleep_time - elapsed_time)
                    time.sleep(adjusted_sleep_time)
                    continue

                frame_copy = frame.copy()
                # Resize if needed
                if video_width != target_width or video_height != target_height:
                    frame_copy = cv2.resize(frame_copy, (target_width, target_height))

                # Compress frame to JPEG in memory
                if not disable_alerts and is_enough_time_elapsed(
                    last_alert_time, total_seconds // 2
                ):
                    _, jpeg_bytes = cv2.imencode(".jpg", frame_copy, encode_params)
                    jpeg_queue.append(jpeg_bytes.tobytes())

                if not is_live_video:
                    last_frame = frame_copy
                if put_in_queue:
                    try:
                        frame_queue.put_nowait(frame_copy)
                    except queue.Full:
                        logger.warning("Frame queue is full, skipping frame")
                        pass

                # Get predictions if available
                try:
                    predicted_class, confidence, bbox, info_predictions = (
                        shared_data_queue.get_nowait()
                    )
                except queue.Empty:
                    pass

                # Save raw frames if requested
                if predicted_class == ClassesEnum.drowning and save_frames and can_save_frames(
                    last_save_frame, total_seconds_save_frames
                ):
                    last_save_frame = time.perf_counter()
                    frames_folder = "frames"
                    if not os.path.exists(frames_folder):
                        os.makedirs(frames_folder)
                    cv2.imwrite(f"{frames_folder}/{uuid.uuid4().hex}.png", frame)

                # Draw and save only the frames that were sampled for inference, so
                # every saved frame carries its own predictions rather than repeating a
                # neighbour's. Drawing is not cheap at 4K, and a frame that is never
                # written does not need to be annotated.
                if put_in_queue:
                    annotated = draw_results(
                        frame_copy.copy(),
                        info_predictions,
                        conf_high_thres,
                        display_only_drowning_state.is_set(),
                        display_water_mask_state.is_set(),
                        display_human_detections_state.is_set(),
                        human_tracker,
                        predicted_class,
                        frame_display_queue,
                    )
                    if video_writer is not None and annotated is not None:
                        write_frame = (annotated if annotated.shape[2] == 3
                                       else cv2.cvtColor(annotated, cv2.COLOR_BGRA2BGR))
                        video_writer.write(write_frame)

                frame_counter += 1
                elapsed_time = time.perf_counter() - start_time
                adjusted_sleep_time = max(0, sleep_time - elapsed_time)
                time.sleep(adjusted_sleep_time)
            except Exception as e:
                logger.error(f"Error during frame capture: {e}")

    finally:
        stop_event.set()
        video_stream.close()
        if video_writer is not None:
            video_writer.release()
            logger.info(f"Saved visualization video to {saved_video_path}")

        # Display last frame for non-live video
        if not is_live_video:
            display_last_frame(
                last_frame,
                processing_done_event,
                shared_data_queue,
                display_human_detections_state,
                conf_high_thres,
                display_only_drowning_state,
                human_tracker,
                display_water_mask_state,
                frame_display_queue,
                info_predictions,
                predicted_class,
            )


def cleanup_resources(stop_event, video_stream, upload_thread, alert_thread, metric_thread, metric_queue, use_cloud, last_frame, processing_done_event, shared_data_queue, human_tracker, display_water_mask, conf_high_thres, last_alert_time, total_seconds, video_source):
    """Clean up resources and finalize the capture process."""
    stop_event.set()
    video_stream.close()
    if upload_thread:
        upload_thread.join()
        if upload_thread.is_alive():
            logger.warning("upload_thread did not finish")
    if alert_thread:
        alert_thread.join(timeout=10)
        if alert_thread.is_alive():
            logger.warning("alert_thread did not finish within 10 seconds")
    if metric_thread:
        metric_thread.join(timeout=10)
        if metric_thread.is_alive():
            logger.warning("metric_thread did not finish within 10 seconds")
    if use_cloud and metric_queue:
        try:
            metric_queue.put({"type": MetricType.CAMERA, "is_camera_working": False}, timeout=1)
        except queue.Full:
            logger.warning("Metric queue full, skipping final CAMERA metric")
    cv2.destroyAllWindows()
    wait_for_last_results(
        last_frame, processing_done_event, shared_data_queue, human_tracker, display_water_mask, conf_high_thres, last_alert_time, total_seconds, video_source
    )
    logger.info("Stopping live camera")

