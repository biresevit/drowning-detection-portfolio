import argparse
from action_recognition.processing import processing_thread
from frame_capture import frame_capture_live_thread
from dotenv import load_dotenv
import os
from alert.camera_details import camera_type__to_enum, load_camera_details
from alert.mac_address import get_mac_address
from action_recognition.load_models_onnx import GenericModelLoader
from utils_lib.path_helpers import normalize_path
from utils_lib.converters import int2bool
from utils_lib.rtsp_helpers import get_rtsp_url
from utils_lib.camera_source import is_live_camera_source
from utc_logging import get_logger
import threading
import queue
from tqdm import tqdm

log_path = "/logs/log.txt"
is_docker_env = os.path.exists("/.dockerenv")
# Check if running in Docker
if not is_docker_env:
    log_path = log_path.lstrip("/")
logger = get_logger(
    log_level="info",
    filename=log_path,
    log_type=0,
    max_log_size_mb=5,
    num_log_backup_files=5,
)


def parse_env():
    load_dotenv()
    interface = os.getenv("INTERFACE")
    mac_address = get_mac_address(interface)
    azure_camera_details_url = None
    camera_ip, camera_type, camera_username, camera_password = None, None, None, None
    camera_type_enum = None
    use_cloud = True if os.getenv("USE_CLOUD").lower() == "true" else False

    if use_cloud:
        azure_camera_details_url = os.getenv("LOAD_CAMERA_DETAILS")

        if not azure_camera_details_url:
            logger.error("LOAD_CAMERA_DETAILS environment variable is not defined!")
            camera_ip, camera_type, camera_username, camera_password = (
                None,
                None,
                None,
                None,
            )
        else:
            camera_ip, camera_type, camera_username, camera_password = (
                load_camera_details(mac_address, azure_camera_details_url)
            )

        camera_type_enum = camera_type__to_enum(camera_type)
        if camera_type_enum is None:
            logger.warning(f"Unknown camera type {camera_type}, defaulting to None")
    else:
        logger.info("Paring environment variables for local processing (no cloud)")
    args = {
        "slide_window_perc": float(os.getenv("SLIDE_WINDOW_PERC")),
        "max_queue_size": int(os.getenv("MAX_QUEUE_SIZE")),
        "val_batch_size": int(os.getenv("VAL_BATCH_SIZE")),
        "overlap_tiles": float(os.getenv("OVERLAP_TILES")),
        "conf_high_thres": float(os.getenv("CONF_HIGH_THRES")),
        "conf_low_thres": float(os.getenv("CONF_LOW_THRES")),
        "classes": os.getenv("CLASSES").split(","),
        "model_variant": os.getenv("MODEL_VARIANT"),
        "cols": int(os.getenv("COLS")),
        "rows": int(os.getenv("ROWS")),
        "max_frame_width": int(os.getenv("MAX_FRAME_WIDTH")),
        "max_frame_height": int(os.getenv("MAX_FRAME_HEIGHT")),
        "max_clusters": int(os.getenv("MAX_CLUSTERS")),
        "factor_tracking": float(os.getenv("FACTOR_TRACKING")),
        "imgsz": int(os.getenv("IMGSZ")),
        "batch_size": int(os.getenv("LIVE_BATCH_SIZE")),
        "display_water_mask": bool(os.getenv("DISPLAY_WATER_MASK")),
        "max_video_seconds": int(os.getenv("MAX_VIDEO_SECONDS")),
        "skip_water_process_frames": int(os.getenv("SKIP_WATER_PROCESS_FRAMES")),
        "max_water_deque_size": int(os.getenv("MAX_WATER_DEQUE_SIZE")),
        "disable_alerts": (
            True if os.getenv("DISABLE_ALERTS").lower() == "true" else False
        ),
        "save_frames": True if os.getenv("SAVE_FRAMES").lower() == "true" else False,
        "total_seconds_save_frames": int(os.getenv("TOTAL_SECONDS_SAVE_FRAMES")),
        "save_video": True if os.getenv("SAVE_VIDEO", "False").lower() == "true" else False,
        "ip_address": camera_ip if camera_ip is not None else "",
        "detect_water": True if os.getenv("DETECT_WATER").lower() == "true" else False,
        "camera_type": camera_type_enum if camera_type_enum is not None else None,
        "camera_username": camera_username if camera_username is not None else "",
        "camera_password": camera_password if camera_password is not None else "",
        "use_cloud": use_cloud,
        "model_extension": (
            os.getenv("MODEL_EXTENSION") if os.getenv("MODEL_EXTENSION") else ".pth"
        ),
    }
    return args


def read_video_metadata(video_source, max_frame_width, max_frame_height):
    import cv2

    cap = cv2.VideoCapture(video_source)

    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        width = min(width, max_frame_width)
        height = min(height, max_frame_height)
        cap.release()
        return width, height, fps
    else:
        logger.error(f"Failed to open video source: {video_source}")
        cap.release()
        return None, None, None


def start_processing(
    video_source,
    disable_alerts,
    action_model,
    human_tracker,
    num_frames,
    max_queue_size,
    sample_interval,
    batch_size,
    factor_tracking,
    max_frame_width,
    max_frame_height,
    slide_window_perc,
    display_water_mask,
    conf_high_thres,
    conf_low_thres,
    max_video_seconds,
    save_frames,
    total_seconds_save_frames,
    stop_event,
    use_cloud=True,
    port=None,
    save_alerts_disk=False,
    alert_ip: str = None,
    alert_port: int = None,
    min_alert_interval_seconds=None,
    drowning_queue: queue.Queue = None,
    upload_video = None,
    save_video_to_disk = None,
    send_alert_via_cloud = None,
    save_alert_image_to_disk = None,
    save_video = False,
):
    frame_queue = queue.Queue(maxsize=max_queue_size)
    shared_data_queue = queue.Queue(maxsize=1)
    processing_done_event = threading.Event()

    width, height, _ = read_video_metadata(
        video_source, max_frame_width, max_frame_height
    )

    logger.info(f"use_cloud: {use_cloud}, port: {port} in start_processing")
    logger.info(f"conf_high_thres: {conf_high_thres}")
    logger.info(f"max_queue_size: {max_queue_size}")

    capture_thread = threading.Thread(
        target=frame_capture_live_thread,
        args=(
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
            use_cloud,
            save_alerts_disk,
            alert_ip,
            alert_port,
            min_alert_interval_seconds,
            drowning_queue,
            upload_video,
            save_video_to_disk,
            send_alert_via_cloud,
            save_alert_image_to_disk,
            save_video,
        ),
        daemon=False,
    )
    proc_thread = threading.Thread(
        target=processing_thread,
        args=(
            frame_queue,
            action_model,
            human_tracker,
            num_frames,
            shared_data_queue,
            slide_window_perc,
            processing_done_event,
            stop_event,
            conf_high_thres,
            conf_low_thres,
            factor_tracking,
            batch_size,
            width,
            height,
        ),
        daemon=False,
    )
    proc_thread.start()    
    capture_thread.start()

    capture_thread.join()
    stop_event.set()
    proc_thread.join()
    logger.info("Threads have stopped. Program exiting.")
    frame_queue.queue.clear()
    shared_data_queue.queue.clear()
    logger.info("Cuda Cleanup done.")


def main(args):
    is_live_camera = is_live_camera_source(args.video_source)
    logger.info(f"Is live camera: {is_live_camera}")
    model_variant = args.model_variant
    classes = args.classes
    imgsz = args.imgsz
    max_clusters = args.max_clusters
    factor_tracking = args.factor_tracking
    slide_window_perc = args.slide_window_perc
    conf_low_thres = args.conf_low_thres
    conf_high_thres = args.conf_high_thres
    batch_size = args.batch_size
    max_queue_size = args.max_queue_size
    max_frame_width = args.max_frame_width
    max_frame_height = args.max_frame_height
    display_water_mask = args.display_water_mask
    max_video_seconds = args.max_video_seconds
    ip_address = args.ip_address
    detect_water = args.detect_water
    camera_type = args.camera_type
    camera_username = args.camera_username
    camera_password = args.camera_password
    upload_video = args.upload_video
    save_video_to_disk = args.save_video_to_disk
    save_video = args.save_video
    send_alert_via_cloud = args.send_alert_via_cloud
    save_alert_image_to_disk = args.save_alert_image_to_disk


    logger.info(
        f"Using camera type: {camera_type}, IP address: {ip_address}, username: {camera_username}, password: {'******' if camera_password else ''}"
    )
    logger.info(
        f"Loading all models with: model variant: {model_variant}, classes: {classes}, imgsz: {imgsz}, max clusters: {max_clusters}, skip water process frames: {args.skip_water_process_frames}, max water deque size: {args.max_water_deque_size}, detect water: {detect_water}, extension:{args.model_extension}"
    )
    
    model_dir = "/models" if is_docker_env else "models"
    
    if args.model_extension == ".engine":
        logger.info("Using TensorRTModelLoader based on model extension .engine")
        from action_recognition.load_models_tensorrt import TensorRTModelLoader
        loader = TensorRTModelLoader(models_dir=normalize_path(model_dir))
    else:
        logger.info("Using GenericModelLoader based on model extension .pth")
        loader = GenericModelLoader(models_dir=normalize_path(model_dir))

    # Load models based on extension dynamically
    action_model = loader.load_action_model(
        model_variant, classes, model_extension=args.model_extension
    )

    from action_recognition.load_models_tensorrt import TensorRTModelLoader
    loader_trt = TensorRTModelLoader(models_dir=normalize_path(model_dir))
    human_tracker = loader_trt.load_human_tracker(
        imgsz=imgsz,
        max_clusters=max_clusters,
        skip_water_process_frames=args.skip_water_process_frames,
        max_water_deque_size=args.max_water_deque_size,
        detect_water=detect_water,
        use_behavior_analysis=os.getenv("USE_BEHAVIOR_ANALYSIS", "True").lower() == "true",
        head_classifier_path=os.getenv("HEAD_MODEL_PATH"),
        head_classifier_max_batch_size=int(os.getenv("HEAD_MODEL_MAX_BATCH_SIZE", "64")),
        head_visibility_img_size=int(os.getenv("HEAD_VISIBILITY_IMG_SIZE", "224")),
        head_visible_frame_count=int(os.getenv("HEAD_VISIBLE_FRAME_COUNT", "16")),
        head_visibility_min_frames=int(os.getenv("HEAD_VISIBILITY_MIN_FRAMES", "10")),
        head_visibility_sample_rate=int(os.getenv("SAMPLE_HEAD_VISIBILITY", "2")),
        downsample_head_frames=os.getenv("DOWNSAMPLE_HEAD_FRAMES", "True").lower() == "true",
        head_visibility_min_bbox_size=int(os.getenv("HEAD_VISIBILITY_MIN_BBOX_SIZE", "30")),
    )

    sample_interval = action_model.get_sampling_rate()
    num_frames = action_model.get_num_frames()
    human_tracker.set_deque_length(int(factor_tracking * num_frames))

    if is_live_camera:
        rtsp_url = get_rtsp_url(
            ip_address, camera_type, camera_username, camera_password
        )
        logger.info(
            f"Using camera type: {camera_type}, IP address: {ip_address}, rtsp_url: {rtsp_url}"
        )

        stop_event = threading.Event()
        start_processing(
            rtsp_url,
            args.disable_alerts,
            action_model,
            human_tracker,
            num_frames,
            max_queue_size,
            sample_interval,
            batch_size,
            factor_tracking,
            max_frame_width,
            max_frame_height,
            slide_window_perc,
            display_water_mask,
            conf_high_thres,
            conf_low_thres,
            max_video_seconds,
            args.save_frames,
            args.total_seconds_save_frames,
            stop_event,
            upload_video = upload_video,
            save_video_to_disk = save_video_to_disk,
            send_alert_via_cloud = send_alert_via_cloud,
            save_alert_image_to_disk = save_alert_image_to_disk,
            save_video = save_video,
        )
    else:
        logger.info(f"Processing video files from directory: {args.video_source}")
        input_dir = args.video_source
        os.makedirs("frames", exist_ok=True)
        video_files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".mpg", ".mpeg"}
        video_files = []

        for root, _, files in os.walk(input_dir):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in video_extensions:
                    full_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(full_path, input_dir)
                    video_files.append(rel_path)

        video_files.sort()
        if not video_files:
            logger.warning(f"No video files found under directory: {input_dir}")
        for video_file in tqdm(video_files, desc="Processing videos"):
            video_source = os.path.join(input_dir, video_file)
            logger.info(f"Processing video: {video_source}")
            stop_event = threading.Event()
            start_processing(
                video_source,
                args.disable_alerts,
                action_model,
                human_tracker,
                num_frames,
                max_queue_size,
                sample_interval,
                batch_size,
                factor_tracking,
                max_frame_width,
                max_frame_height,
                slide_window_perc,
                display_water_mask,
                conf_high_thres,
                conf_low_thres,
                max_video_seconds,
                args.save_frames,
                args.total_seconds_save_frames,
                stop_event,
                upload_video = upload_video,
                save_video_to_disk = save_video_to_disk,
                send_alert_via_cloud = send_alert_via_cloud,
                save_alert_image_to_disk = save_alert_image_to_disk,
                save_video = save_video,
            )

    logger.info("Done processing all videos")
    human_tracker.cleanup()
    action_model.cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Process video files or streams")
    parser.add_argument(
        "--video_source",
        default=r"path/to/data",
        required=False,
        help="Path to directory containing MP4 files",
    )

    parser.add_argument(
        "--upload_video",
        type=int2bool,
        default=1,
        help="Upload video to the cloud when use_cloud=True (1=yes, 0=no, default=1).",
    )

    parser.add_argument(
        "--save_video_to_disk",
        type=int2bool,
        default=0,
        help="Save video to the local disk (1=yes, 0=no, default=0).",
    )

    parser.add_argument(
        "--send_alert_via_cloud",
        type=int2bool,
        default=1,
        help="Send alert via the cloud when use_cloud=True (1=yes, 0=no, default=1).",
    )

    parser.add_argument(
        "--save_alert_image_to_disk",
        type=int2bool,
        default=0,
        help="Save alert image to the local disk (1=yes, 0=no, default=0).",
    )

    return parser.parse_args()


def combine_args(env_args, cli_args):
    args = env_args.copy()
    args.update(
        {
            "video_source": cli_args.video_source,
            "upload_video": cli_args.upload_video,
            "save_video_to_disk":  cli_args.save_video_to_disk,
            "send_alert_via_cloud": cli_args.send_alert_via_cloud,
            "save_alert_image_to_disk": cli_args.save_alert_image_to_disk
        }
    )
    return argparse.Namespace(**args)


if __name__ == "__main__":
    args = parse_env()
    cli_args = parse_args()
    args = combine_args(args, cli_args)
    main(args)
