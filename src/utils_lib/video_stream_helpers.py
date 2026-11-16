import logging
import time
import os

from video_stream import VideoStream
logger = logging.getLogger()



def initialize_video_stream(video_source, num_frames, batch_size, slide_window_perc, factor_tracking, sample_interval, use_cuda=True, restart_timeout=5.0):
    """Initialize the video stream with provided parameters."""
    is_docker_env = os.path.exists("/.dockerenv")
    if not is_docker_env and os.name != 'nt':
        use_cuda = False
        logger.info("Non-Docker environment detected; disabling CUDA for video stream.")
    video_stream = VideoStream(
        video_source=video_source,
        num_frames=num_frames,
        batch_size=batch_size,
        slide_window_perc=slide_window_perc,
        factor_tracking=factor_tracking,
        sample_interval=sample_interval,
        use_cuda=use_cuda,
        restart_timeout=restart_timeout,
    )
    return video_stream

def setup_video_metadata(video_stream, human_tracker, max_frame_width, max_frame_height, max_video_seconds):
    """Set up video metadata and calculate target dimensions and frame limits."""
    is_live_video = "rtsp" in str(video_stream.video_source) or "youtube" in str(video_stream.video_source) or "googlevideo" in str(video_stream.video_source)
    target_fps = min(video_stream.fps, 30)
    human_tracker.set_video_metadata(
        video_width=video_stream.width, video_height=video_stream.height, fps=target_fps,
        sample_interval=getattr(video_stream, "sample_interval", None),
    )
    target_width = min(video_stream.width, max_frame_width)
    target_height = min(video_stream.height, max_frame_height)
    max_frames = max_video_seconds * int(target_fps)
    return is_live_video, target_fps, target_width, target_height, max_frames

def capture_frame_status(last_log_time, log_interval, video_stream: VideoStream, start_time):
    now = time.perf_counter()
    should_log = now - last_log_time >= log_interval

    if should_log:
        logger.info("Before read_frame()")

    try:
        # Try to read a frame from the video stream
        frame = video_stream.read_frame()
    except Exception as e:
        # Log the error if reading the frame fails and set frame to None
        logger.error(f"Error while reading frame from camera: {e}")
        frame = None

    if should_log:
        # Log timing and whether the frame is valid or None
        logger.info(
            f"After read_frame() - duration={time.perf_counter() - start_time:.2f}s, "
            f"frame={'valid' if frame is not None else 'None'}"
        )
        last_log_time = now

    # Determine if the camera is working based on frame validity
    is_camera_working = frame is not None
    return frame, is_camera_working, last_log_time

def handle_stream_failure(is_live_video, started_frame_read_time, video_stream: VideoStream, now):
    """Handle video stream failure and attempt to reinitialize if necessary."""
    if is_live_video and now - started_frame_read_time > 5:
        logger.warning("No valid frame for 5 seconds, attempting to reinitialize stream")
        try:
            video_stream.restart_stream()
        except Exception as e:
            logger.error(f"Failed to reinitialize stream: {e}")
            if now - started_frame_read_time > 60:
                logger.error("Camera inactive for over 1 minute, stopping capture")
                return False  # signal to break the loop
    return True


