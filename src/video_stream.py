import ffmpeg
import threading
import time
import numpy as np
import logging
import os
from resource_monitor import ResourceMonitor
from fractions import Fraction
import cv2

logger = logging.getLogger()


class VideoStream:
    def __init__(
        self,
        video_source: str,
        num_frames: int,
        batch_size: int,
        slide_window_perc: float,
        factor_tracking: float,
        sample_interval: int,
        use_cuda: bool = False,
        restart_timeout: float = 5.0,
    ):
        """
        Initialize video stream with given parameters.
        video_source: RTSP/HTTP URL or local file path
        num_frames, batch_size, slide_window_perc, factor_tracking, sample_interval: used for frame reading logic
        use_cuda: enable hwaccel if available
        restart_timeout: seconds without valid frames before restarting
        """
        self.video_source = video_source
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.slide_window_perc = slide_window_perc
        self.factor_tracking = factor_tracking
        self.sample_interval = sample_interval
        self.use_cuda = use_cuda
        self.restart_timeout = restart_timeout

        self.width = 0
        self.height = 0
        self.fps = 0
        self.stream_proc = None
        self.last_valid_frame_time = time.time()
        self.lock = threading.Lock()
        self.transport_protocol = "tcp"  # Default to TCP for RTSP
        self.is_local_file = self._is_local_file(video_source)
        self.invalid_frame_count = 0

        self._init_stream()

        self.resource_monitor = ResourceMonitor(interval=1.0)
        self.resource_monitor.start()

    def _is_local_file(self, video_source: str) -> bool:
        """Check if video_source is a local file based on extension or prefix"""
        local_extensions = (".mp4", ".avi", ".mkv", ".mov")
        return (
            video_source.startswith("file://")
            or video_source.lower().endswith(local_extensions)
            or os.path.isfile(video_source)
        )

    def _init_stream(self):
        """Initialize FFmpeg stream and probe info"""
        try:
            logger.info(
                f"Probing video source {self.video_source} {'as local file' if self.is_local_file else f'with {self.transport_protocol}'}"
            )
            probe_kwargs = (
                {}
                if self.is_local_file
                else {"rtsp_transport": self.transport_protocol}
            )
            probe = ffmpeg.probe(self.video_source, **probe_kwargs)
            video_streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
            if not video_streams:
                raise RuntimeError(f"No video streams found in {self.video_source}")
            video_info = video_streams[0]
            self.width = int(video_info.get("width", 0))
            self.height = int(video_info.get("height", 0))
            if self.is_local_file:
                try:
                    cap = cv2.VideoCapture(self.video_source)
                    fps_val = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    if fps_val <= 0:
                        raise ValueError("OpenCV returned invalid FPS")
                except Exception as e:
                    logger.warning(f"Failed to read FPS with OpenCV: {e}, falling back to FFmpeg")
                    r_frame_rate = video_info.get("r_frame_rate", "30/1")
                    fps_val = float(Fraction(r_frame_rate))
            else:
                r_frame_rate = video_info.get("r_frame_rate", "30/1")
                fps_val = float(Fraction(r_frame_rate))

            self.fps = int(round(fps_val))
            codec_name = video_info.get("codec_name", "")

            # Validate metadata
            if self.width <= 0 or self.height <= 0:
                logger.error(
                    f"Invalid video dimensions: width={self.width}, height={self.height}"
                )
                logger.info(f"Probe output: {probe}")
                if not self.is_local_file and self.transport_protocol == "tcp":
                    logger.info(f"TCP probe failed, trying UDP for {self.video_source}")
                    self.transport_protocol = "udp"
                    return self._init_stream()
                raise ValueError("Video stream has invalid dimensions")

            # Ensure minimum FPS
            min_fps = 4
            if self.fps < min_fps:
                logger.warning(f"Low FPS detected: {self.fps}. Setting to {min_fps}.")
                self.fps = min_fps
            if self.fps <= 0 or self.fps > 1000:
                logger.warning(f"Unrealistic FPS detected: {self.fps}. Setting to 30.")
                self.fps = 30.0

            # Adjust sample interval based on target FPS
            target_fps = 30
            if self.fps != target_fps and self.fps > 0:
                factor_fps = self.fps / target_fps
                self.sample_interval = int(self.sample_interval * factor_fps)

            # Apply tracking factor
            self.sample_interval = max(
                1, int(self.sample_interval / self.factor_tracking)
            )

            logger.info(
                f"Video info - {self.width}x{self.height} @ {self.fps:.2f}fps, sample_interval={self.sample_interval}"
            )

            # Setup FFmpeg stream
            input_kwargs = {}
            if not self.is_local_file and codec_name != "av1":
                input_kwargs.update(
                    {
                        "rtsp_transport": self.transport_protocol,
                        "rtsp_flags": "prefer_tcp",
                        "buffer_size": str(4 * 1024 * 1024),  # 4MB buffer
                        "flags": "low_delay",
                        "fflags": "+discardcorrupt",
                        "err_detect": "ignore_err",
                    }
                )

            # Use CUDA only if supported codec
            if self.use_cuda and codec_name in ["h264", "hevc", "mpeg4", "mjpeg"]:
                input_kwargs.update(
                    {"hwaccel": "cuda", "hwaccel_output_format": "nv12"}
                )
            elif self.use_cuda and codec_name == "av1":
                logger.warning(
                    "AV1 hardware decoding not supported, falling back to CPU decoding"
                )
            elif self.use_cuda:
                logger.warning(
                    f"CUDA hwaccel not supported for codec {codec_name}, using CPU decoding"
                )

            ff_input = ffmpeg.input(self.video_source, **input_kwargs)
            self.stream_proc = (
                ffmpeg.output(
                    ff_input, "pipe:", format="rawvideo", pix_fmt="bgr24", vsync="0"
                )
                .global_args("-stats_period", "60")
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )

            threading.Thread(
                target=self._stderr_reader, args=(self.stream_proc,), daemon=True
            ).start()
            logger.info("Video stream initialized successfully")

        except ffmpeg.Error as e:
            logger.error(
                f"FFmpeg error with {self.transport_protocol if not self.is_local_file else 'local file'}: {e.stderr.decode()}"
            )
            if not self.is_local_file and self.transport_protocol == "tcp":
                logger.info(f"TCP failed, trying UDP for {self.video_source}")
                self.transport_protocol = "udp"
                return self._init_stream()
            raise
        except Exception as e:
            logger.error(f"Error initializing video stream: {e}")
            logger.info(f"Probe output: {probe if 'probe' in locals() else 'N/A'}")
            if not self.is_local_file and self.transport_protocol == "tcp":
                logger.info(
                    f"TCP failed (general error), trying UDP for {self.video_source}"
                )
                self.transport_protocol = "udp"
                return self._init_stream()
            raise

    def _stderr_reader(self, proc):
        """Continuously read FFmpeg stderr and restart stream on frequent DTS errors."""
        try:
            dts_error_count = 0
            last_dts_error_time = time.time()

            for line in proc.stderr:
                line_decoded = line.decode().strip()
                logger.info(f"FFmpeg: {line_decoded}")

                if "non-monotonically increasing DTS" in line_decoded:
                    current_time = time.time()

                    # Reset count if last error was too long ago
                    if current_time - last_dts_error_time > 10:
                        dts_error_count = 0
                        last_dts_error_time = current_time

                    # Increment error count
                    dts_error_count += 1
                    last_dts_error_time = current_time

                    # Restart stream if too many errors occur in short period
                    if dts_error_count >= 5:
                        logger.warning(
                            "Frequent DTS errors detected, restarting stream"
                        )
                        self.restart_stream()
                        dts_error_count = 0
                        last_dts_error_time = current_time

        except Exception as e:
            logger.error(f"Error in stderr reader: {e}")

    def read_frame(self) -> np.ndarray | None:
        with self.lock:
            if self.width <= 0 or self.height <= 0:
                logger.error("Cannot read frame: invalid dimensions")
                return None
            try:
                expected_bytes = self.width * self.height * 3
                start_time = time.time()
                in_bytes = b""
                while len(in_bytes) < expected_bytes:
                    chunk = self.stream_proc.stdout.read(expected_bytes - len(in_bytes))
                    if not chunk:
                        break
                    in_bytes += chunk
                    if time.time() - start_time > 12:
                        logger.warning(
                            f"read_frame timeout after {time.time() - start_time:.2f}s, got {len(in_bytes)} bytes"
                        )
                        self.restart_stream()
                        return None

                if len(in_bytes) < expected_bytes:
                    logger.warning(
                        f"Invalid frame: expected {expected_bytes} bytes, got {len(in_bytes)}"
                    )
                    self.invalid_frame_count += 1
                    return None

                self.last_valid_frame_time = time.time()
                self.invalid_frame_count = 0

                return np.frombuffer(in_bytes, np.uint8).reshape(
                    (self.height, self.width, 3)
                )
            except Exception as e:
                logger.error(f"Error reading frame: {e}")
                return None

    def restart_stream(self):
        """Terminate current FFmpeg process and restart"""
        try:
            logger.info("Restarting video stream")
            if self.stream_proc:
                self.stream_proc.terminate()
                try:
                    self.stream_proc.wait(timeout=5)
                except:
                    self.stream_proc.kill()
            time.sleep(1)  # Prevent rapid restart loops
            self._init_stream()
            self.last_valid_frame_time = time.time()
        except Exception as e:
            logger.error(f"Failed to restart stream: {e}")
            raise

    def close(self):
        """Terminate stream process"""
        if self.stream_proc:
            self.stream_proc.terminate()
            try:
                self.stream_proc.wait(timeout=5)
            except:
                self.stream_proc.kill()
        logger.info("Video stream closed")

    @property
    def video_info(self):
        """Return video width, height, fps, and sample interval"""
        return self.width, self.height, self.fps, self.sample_interval
