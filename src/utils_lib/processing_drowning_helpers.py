import time
import logging
import threading
import uuid

from alert.send_notification import send_alert_rest_api, send_image_and_notification
from alert.video_upload import save_alert_video_locally, upload_video
from utils_lib.draw_helpers import draw_alert_notification
from utils_lib.drowning_detection_helpers import is_drowning_detected
from utils_lib.frame_helpers import save_drowning_frame

logger = logging.getLogger()

def start_alert_thread(
    use_cloud: bool,
    frame_alert,
    mac_address: str,
    unique_name: str,
    metric_queue,
    frame,
    bbox,
    target_width: int,
    target_height: int,
    confidence: float,
    alert_ip: str,
    alert_port: int,
    save_alerts_disk: bool,
    send_alert_via_cloud = None,
    save_alert_image_to_disk = None
) -> threading.Thread | None:
    """
    Start an alert thread (cloud or REST) and return it.
    Returns the Thread object, or None if the thread could not be started.

    :param use_cloud: Whether to use cloud alert
    :param frame_alert: Frame for cloud alert
    :param mac_address: Device MAC address
    :param unique_name: Unique identifier for the alert
    :param metric_queue: Queue for metrics (used in cloud alert)
    :param frame: Frame for REST alert
    :param bbox: Bounding box coordinates for REST alert
    :param target_width: Width for REST alert frame
    :param target_height: Height for REST alert frame
    :param confidence: Confidence score for REST alert
    :param alert_ip: IP for REST alert
    :param alert_port: Port for REST alert
    :param save_alerts_disk: Whether to save REST alert locally
    :return: Thread object or None
    """
    try:
        if use_cloud:
            # Start thread for sending alert to cloud
            alert_thread = threading.Thread(
                target=send_image_and_notification,
                args=(frame_alert, mac_address, unique_name, metric_queue, send_alert_via_cloud, save_alert_image_to_disk),
                daemon=False,
            )
            alert_thread.start()
            return alert_thread

        # If not cloud, ensure REST alert parameters are provided
        if alert_port is None or alert_ip is None:
            logger.error("REST alert port or IP not provided")
            return None

        # Start thread for sending alert via REST API
        alert_thread = threading.Thread(
            target=send_alert_rest_api,
            args=(
                frame,
                bbox,
                target_width,
                target_height,
                confidence,
                alert_ip,
                alert_port,
                save_alerts_disk,
                unique_name,
                frame_alert
            ),
            daemon=False,
        )
        alert_thread.start()
        return alert_thread

    except Exception as e:
        # Log any exception that occurs while starting the thread
        logger.exception("Failed to start alert thread: %s", e)
        return None

def start_video_upload(
    use_cloud: bool,
    frame_queue_storage,
    target_fps,
    confidence,
    pred_class_int,
    mac_address,
    target_width,
    target_height,
    unique_name,
    bbox,
    metric_queue,
    azure_original_video_url,
    azure_resized_video_url,
    is_uploading_video,
    save_alerts_disk: bool,
    use_cloud_upload_video: bool,
    save_video_to_disk: bool
):
    """
    Start a thread to upload or save the alert video depending on the configuration.
    Returns the Thread object or None if nothing was started.
    """
    try:
        if use_cloud:
            upload_thread = threading.Thread(
                target=upload_video,
                args=(
                    frame_queue_storage,
                    target_fps,
                    confidence,
                    pred_class_int,
                    mac_address,
                    target_width,
                    target_height,
                    unique_name,
                    bbox,
                    metric_queue,
                    azure_original_video_url,
                    azure_resized_video_url,
                    is_uploading_video,
                    use_cloud_upload_video,
                    save_video_to_disk
                ),
                daemon=False,
            )
        elif save_alerts_disk:
            upload_thread = threading.Thread(
                target=save_alert_video_locally,
                args=(
                    frame_queue_storage,
                    target_fps,
                    target_width,
                    target_height,
                    unique_name,
                    is_uploading_video,
                ),
                daemon=False,
            )
        else:
            return None  # Neither cloud nor local saving enabled

        upload_thread.start()
        return upload_thread

    except Exception as e:
        logger.error(f"Failed to start video upload thread: {e}")
        return None

def handle_drowning_detection(
    predicted_class,
    confidence,
    conf_high_thres,
    info_predictions,
    bbox,
    frame,
    video_source,
    frame_queue_storage,
    target_width,
    target_height,
    target_fps,
    human_tracker,
    display_water_mask,
    disable_alerts,
    is_uploading_video: threading.Event,
    use_cloud: bool,
    save_alerts_disk: bool,
    metric_queue,
    last_alert_time,
    total_seconds,
    max_frames,
    azure_original_video_url,
    azure_resized_video_url,
    alert_thread,
    upload_thread,
    alert_ip=None,
    alert_port=None,
    is_live_video=True,
    mac_address=None,
    start_time=None,
    upload_video = None,
    save_video_to_disk = None,
    send_alert_via_cloud = None,
    save_alert_image_to_disk = None
    
):
    """
    Handle drowning detection logic:
    - Check if predicted_class indicates drowning
    - Draw alert frame if necessary
    - Start alert thread (cloud or REST)
    - Start video upload if conditions are met
    - Log warnings if upload cannot start
    Returns updated last_alert_time, alert_thread, and upload_thread.
    """

    try:
        if predicted_class is None:
            return last_alert_time, alert_thread, upload_thread

        # Determine if drowning is detected
        is_drowning = is_drowning_detected(predicted_class, confidence, conf_high_thres, info_predictions)
        unique_name = str(uuid.uuid4())

        # Check if enough time has passed since last alert
        enough_time_elapsed = is_enough_time_elapsed(last_alert_time, total_seconds)

        frame_alert = None
        if is_drowning and enough_time_elapsed:
            # Draw alert notification frame
            frame_alert = draw_alert_notification(
                frame.copy(),
                predicted_class,
                confidence,
                bbox,
                human_tracker,
                display_water_mask,
            )
            logger.info(f"Frame alert drawn in {time.perf_counter() - start_time:.2f}s" if start_time else "Frame alert drawn")

            # Save alert frame for non-live video
            save_drowning_frame(frame_alert, video_source, is_live_video)

            # Update last alert time
            last_alert_time = time.perf_counter()

            # Start alert thread if alerts are enabled
            if not disable_alerts:
                alert_thread = start_alert_thread(
                    use_cloud,
                    frame_alert,
                    mac_address,
                    unique_name,
                    metric_queue,
                    frame,
                    bbox,
                    target_width,
                    target_height,
                    confidence,
                    alert_ip,
                    alert_port,
                    save_alerts_disk,
                    send_alert_via_cloud,
                    save_alert_image_to_disk
                )

        # Check if video upload should start
        is_set_is_uploading_video = is_uploading_video.is_set()
        if (
            not disable_alerts
            and is_drowning
            and not is_set_is_uploading_video
            and len(frame_queue_storage) >= 0.2 * max_frames
            and enough_time_elapsed
        ):
            logger.info(
                f"Uploading video with drowning prediction: {predicted_class.name} ({confidence:.2f}%)"
            )
            pred_class_int = (
                predicted_class.value
                if isinstance(predicted_class.value, int)
                else predicted_class.value[0]
            )

            # Start upload thread
            upload_thread = start_video_upload(
                use_cloud,
                frame_queue_storage,
                target_fps= target_fps,
                confidence=confidence,
                pred_class_int=pred_class_int,
                mac_address=mac_address,
                target_width=target_width,
                target_height=target_height,
                unique_name=unique_name,
                bbox=bbox,
                metric_queue=metric_queue,
                azure_original_video_url=azure_original_video_url,
                azure_resized_video_url=azure_resized_video_url,
                is_uploading_video=is_uploading_video,
                save_alerts_disk=save_alerts_disk,
                use_cloud_upload_video=upload_video,
                save_video_to_disk=save_video_to_disk
            )
        else:
            log_drowning_upload_warning(is_drowning, is_uploading_video, disable_alerts, frame_queue_storage, enough_time_elapsed, use_cloud)

    except Exception as e:
        # Catch any exception and log without breaking main loop
        logger.exception(f"Error in drowning detection handling: {e}")

    return last_alert_time, alert_thread, upload_thread

def log_drowning_upload_warning(is_drowning: bool, is_uploading_video, disable_alerts: bool, frame_queue_storage, enough_time_elapsed: bool, use_cloud: bool):
    """
    Log warnings related to upload conditions when drowning is detected.
    """
    if is_drowning:
        logger.warning(
            f"Not starting upload thread: Upload conditions - is_uploading_video: {is_uploading_video.is_set()}, "
            f"disable_alerts: {disable_alerts}, is_drowning: {is_drowning}, "
            f"stored frames: {len(frame_queue_storage)}, "
            f"enough_time_elapsed: {enough_time_elapsed}, cloud use: {use_cloud}"
        )

def is_enough_time_elapsed(last_alert_time, total_seconds):
    if last_alert_time is None:
        return True

    now = time.perf_counter()
    elapsed = now - last_alert_time

    if elapsed > total_seconds:
        return True
    else:
        return False