from alert.camera_details import camera_type__to_enum, load_camera_details
from alert.mac_address import get_mac_address
import torch
import threading
import queue
import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
from dotenv import load_dotenv
import os
import shutil
from action_recognition.load_models_onnx import GenericModelLoader
from action_recognition.load_models_tensorrt import TensorRTModelLoader

from utc_logging import get_logger
from action_recognition.processing import processing_thread
from frame_capture import frame_capture_thread
from dataset.download_yt_video import get_youtube_stream_url, is_youtube_url
from start_processing import read_video_metadata
from utils_lib.rtsp_helpers import get_rtsp_url

logger = get_logger(
    log_level="info",
    filename="log.txt",
    log_type=0,
    max_log_size_mb=5,
    num_log_backup_files=5,
)
logger.info(f"Is cuda available: {torch.cuda.is_available()}")


class ActionRecognitionApp:
    def __init__(
        self,
        root,
        slide_window_perc,
        max_queue_size,
        batch_size,
        conf_high_thres,
        conf_low_thres,
        max_frame_width,
        max_frame_height,
        max_clusters,
        factor_tracking,
        classes,
        imgsz,
        max_video_seconds,
        save_frames,
        total_seconds_save_frames,
        save_video,
        skip_water_process_frames,
        max_water_deque_size,
        disable_alerts,
        ip_address,
        camera_type,
        camera_username,
        camera_password,
        detect_water,
        model_extension,
        model_variant,
        head_model_path=None,
        head_model_max_batch_size=64,
        head_visibility_img_size=224,
        head_visible_frame_count=16,
        head_visibility_min_frames=10,
        head_visibility_sample_rate=2,
        downsample_head_frames=True,
        head_visibility_min_bbox_size=30,
        use_behavior_analysis=True,
    ):
        self.ip_address = ip_address
        self.camera_type = camera_type
        self.camera_username = camera_username
        self.camera_password = camera_password
        self.skip_water_process_frames = skip_water_process_frames
        self.max_water_deque_size = max_water_deque_size
        self.root = root
        self.total_seconds_save_frames = total_seconds_save_frames
        self.save_frames = save_frames
        self.save_video = save_video
        self.max_video_seconds = max_video_seconds
        self.factor_tracking = factor_tracking
        self.max_clusters = max_clusters
        self.max_frame_width = max_frame_width
        self.max_frame_height = max_frame_height
        self.conf_high_thres = conf_high_thres
        self.conf_low_thres = conf_low_thres
        self.slide_window_perc = slide_window_perc
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.shared_data_queue = queue.Queue(maxsize=1)
        self.frame_display_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.processing_done_event = threading.Event()
        self.video_ended = False
        self.capture_thread = None
        self.proc_thread = None
        self.disable_alerts = disable_alerts

        self.trt_loader = TensorRTModelLoader(models_dir="/models")
        action_loader = (
            GenericModelLoader(models_dir="/models")
            if model_extension != ".engine"
            else self.trt_loader
        )
        self.model = action_loader.load_action_model(
            model_variant, classes, model_extension=model_extension
        )
        if self.model is None:
            logger.error("Failed to load action recognition model.")
            raise RuntimeError("Failed to load action recognition model.")
        self.human_tracker = self.trt_loader.load_human_tracker(
            imgsz=imgsz,
            max_clusters=max_clusters,
            skip_water_process_frames=skip_water_process_frames,
            max_water_deque_size=max_water_deque_size,
            detect_water=detect_water,
            head_classifier_path=head_model_path,
            head_classifier_max_batch_size=head_model_max_batch_size,
            head_visibility_img_size=head_visibility_img_size,
            head_visible_frame_count=head_visible_frame_count,
            head_visibility_min_frames=head_visibility_min_frames,
            head_visibility_sample_rate=head_visibility_sample_rate,
            downsample_head_frames=downsample_head_frames,
            head_visibility_min_bbox_size=head_visibility_min_bbox_size,
            use_behavior_analysis=use_behavior_analysis,
        )
        if self.human_tracker is None:
            logger.error("Failed to load human tracker model.")
            raise RuntimeError("Failed to load human tracker model.")
        num_frames = self.model.get_num_frames()
        self.human_tracker.set_deque_length(int(factor_tracking * num_frames))

        self.sample_interval = self.model.get_sampling_rate()
        logger.info("Loading models success")

        self.batch_size = batch_size
        self.zoom_levels = ["25%", "50%", "75%", "100%", "125%", "150%", "200%", "300%"]
        self.zoom_factor = 0.5
        self.display_human_detections = tk.BooleanVar(value=True)
        self.display_only_drowning = tk.BooleanVar(value=True)
        self.display_water_mask = tk.BooleanVar(value=True)
        self.display_human_detections_state = threading.Event()
        self.display_water_mask_state = threading.Event()
        self.display_only_drowning_state = threading.Event()
        self.display_human_detections_state.set()
        self.display_only_drowning_state.set()
        self.display_water_mask_state.set()
        self.after_id = None

        self.setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_frame()

    def setup_gui(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.video_frame = tk.Frame(self.main_frame)
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.load_button = tk.Button(
            self.control_frame,
            text="Load Video",
            command=self.load_video,
            width=20,
            height=2,
        )
        self.load_button.pack(pady=10)
        self.live_button = tk.Button(
            self.control_frame,
            text="Live Camera",
            command=self.start_live,
            width=20,
            height=2,
        )
        self.live_button.pack(pady=10)
        self.youtube_button = tk.Button(
            self.control_frame,
            text="YouTube",
            command=self.load_youtube,
            width=20,
            height=2,
        )
        self.youtube_button.pack(pady=10)
        self.batch_size_label = tk.Label(self.control_frame, text="Batch Size:")
        self.batch_size_label.pack(pady=5)
        self.batch_size_input = tk.Entry(self.control_frame)
        self.batch_size_input.pack(pady=5)
        self.batch_size_input.insert(0, str(self.batch_size))
        self.zoom_label = tk.Label(self.control_frame, text="Zoom:")
        self.zoom_label.pack(pady=5)
        self.zoom_var = tk.StringVar(self.control_frame)
        self.zoom_var.set("75%")
        self.zoom_dropdown = tk.OptionMenu(
            self.control_frame,
            self.zoom_var,
            *self.zoom_levels,
            command=self.update_zoom,
        )
        self.zoom_dropdown.pack(pady=5)
        self.human_detections_checkbox = tk.Checkbutton(
            self.control_frame,
            text="Display Human Detections",
            variable=self.display_human_detections,
            command=self.update_display_human_detections_state,
        )

        self.human_detections_checkbox.pack(pady=5)

        self.only_drowning_checkbox = tk.Checkbutton(
            self.control_frame,
            text="Display Only Drowning",
            variable=self.display_only_drowning,
            command=self.update_display_only_drowning_state,
        )

        self.only_drowning_checkbox.pack(pady=5)
        self.water_mask_checkbox = tk.Checkbutton(
            self.control_frame,
            text="Display Water Mask",
            variable=self.display_water_mask,
            command=self.update_display_water_mask_state,
        )
        self.water_mask_checkbox.pack(pady=5)
        self.quit_button = tk.Button(
            self.control_frame, text="Quit", command=self.quit_app, width=20, height=2
        )
        self.quit_button.pack(pady=10)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(pady=10)

    def update_display_only_drowning_state(self):
        if self.display_only_drowning.get():
            self.display_only_drowning_state.set()
        else:
            self.display_only_drowning_state.clear()

    def update_display_human_detections_state(self):
        if self.display_human_detections.get():
            self.display_human_detections_state.set()
        else:
            self.display_human_detections_state.clear()

    def update_display_water_mask_state(self):
        if self.display_water_mask.get():
            self.display_water_mask_state.set()
        else:
            self.display_water_mask_state.clear()

    def update_zoom(self, value):
        zoom_percent = int(value.strip("%"))
        self.zoom_factor = zoom_percent / 100.0
        logger.info(f"Zoom factor updated to {self.zoom_factor}")

    def update_frame(self):
        try:
            frame = self.frame_display_queue.get_nowait()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            img = Image.fromarray(frame)
            width, height = img.size
            new_width = int(width * self.zoom_factor)
            new_height = int(height * self.zoom_factor)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except queue.Empty:
            pass
        self.after_id = self.root.after(15, self.update_frame)

    def on_closing(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.stop_threads()
        self.human_tracker.cleanup()
        self.model.cleanup()
        self.root.destroy()

    def load_video(self):
        video_path = filedialog.askopenfilename()

        if video_path:
            self.start_frame_capture(video_path)

    def load_youtube(self):
        url = simpledialog.askstring("YouTube URL", "Enter YouTube URL:")
        if not url:
            return
        if not is_youtube_url(url):
            tk.messagebox.showerror(
                "Invalid URL", "The URL entered is not a valid YouTube URL."
            )
            return
        stream_url = get_youtube_stream_url(url)
        logger.info(f"YouTube URL verified. Stream URL: {stream_url}")
        self.start_frame_capture(stream_url)

    def start_live(self):
        # rtsp = "rtsp://localhost:8554/drowning_alerts"
        rtsp = "rtsp://localhost/live"
        
        self.start_frame_capture(rtsp)

    def _poll_processing_done(self):
        if self.processing_done_event.is_set():
            self.load_button.config(state=tk.NORMAL)
        else:
            self.root.after(500, self._poll_processing_done)

    def start_frame_capture(self, video_source):
        _runs_dir = os.path.join(os.path.dirname(__file__), "human_tracking", "tests", "runs")
        if os.path.exists(_runs_dir):
            shutil.rmtree(_runs_dir)

        if self.capture_thread or self.proc_thread:
            self.stop_threads()
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        with self.shared_data_queue.mutex:
            self.shared_data_queue.queue.clear()
        with self.frame_display_queue.mutex:
            self.frame_display_queue.queue.clear()
        num_frames = self.model.get_num_frames()
        self.processing_done_event.clear()

        width, height, fps = read_video_metadata(
            video_source, self.max_frame_width, self.max_frame_height
        )

        self.batch_size = int(self.batch_size_input.get())
        self.capture_thread = threading.Thread(
            target=frame_capture_thread,
            args=(
                video_source,
                self.disable_alerts,
                self.human_tracker,
                num_frames,
                self.frame_queue,
                self.shared_data_queue,
                self.sample_interval,
                self.batch_size,
                self.max_frame_width,
                self.max_frame_height,
                self.frame_display_queue,
                self.processing_done_event,
                self.stop_event,
                self.display_human_detections_state,
                self.display_water_mask_state,
                self.display_only_drowning_state,
                self.factor_tracking,
                self.slide_window_perc,
                self.conf_high_thres,
                self.max_video_seconds,
                self.save_frames,
                self.total_seconds_save_frames,
                self.save_video,
            ),
            daemon=True,
        )
        self.proc_thread = threading.Thread(
            target=processing_thread,
            args=(
                self.frame_queue,
                self.model,
                self.human_tracker,
                num_frames,
                self.shared_data_queue,
                self.slide_window_perc,
                self.processing_done_event,
                self.stop_event,
                self.conf_high_thres,
                self.conf_low_thres,
                self.factor_tracking,
                self.batch_size,
                width,
                height,
            ),
            daemon=True,
        )
        self.load_button.config(state=tk.DISABLED)
        self.proc_thread.start()
        self.capture_thread.start()
        self.root.after(500, self._poll_processing_done)


    def stop_threads(self):
        self.stop_event.set()
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                logger.warning("Capture thread did not terminate cleanly")
        if self.proc_thread and self.proc_thread.is_alive():
            self.proc_thread.join(timeout=2.0)
            if self.proc_thread.is_alive():
                logger.warning("Processing thread did not terminate cleanly")
        self.stop_event.clear()
        self.processing_done_event.set()

    def quit_app(self):
        self.stop_threads()
        self.on_closing()


def parse_env():
    load_dotenv()
    interface = os.getenv("INTERFACE")
    mac_address = get_mac_address(interface)
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
        try:
            camera_ip, camera_type, camera_username, camera_password = load_camera_details(
                mac_address, azure_camera_details_url
            )
        except:
            logger.error("LOAD_CAMERA_DETAILS failed")
            camera_ip, camera_type, camera_username, camera_password = (
                        None,
                        None,
                        None,
                        None,)
                        
    # Map camera_type (integer) to CAMERA_TYPE enum
    camera_type_enum = camera_type__to_enum(camera_type)
    if camera_type_enum is None:
        logger.warning(f"Unknown camera type {camera_type}, defaulting to None")
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
        "max_video_seconds": int(os.getenv("MAX_VIDEO_SECONDS")),
        "save_frames": True if os.getenv("SAVE_FRAMES").lower() == "true" else False,
        "total_seconds_save_frames": int(os.getenv("TOTAL_SECONDS_SAVE_FRAMES")),
        "save_video": True if os.getenv("SAVE_VIDEO", "False").lower() == "true" else False,
        "skip_water_process_frames": int(os.getenv("SKIP_WATER_PROCESS_FRAMES")),
        "max_water_deque_size": int(os.getenv("MAX_WATER_DEQUE_SIZE")),
        "disable_alerts": (
            True if os.getenv("DISABLE_ALERTS").lower() == "true" else False
        ),
        "ip_address": camera_ip,
        "detect_water": True if os.getenv("DETECT_WATER").lower() == "true" else False,
        "camera_type": camera_type_enum,
        "camera_username": camera_username,
        "camera_password": camera_password,
        "model_extension": (
            os.getenv("MODEL_EXTENSION") if os.getenv("MODEL_EXTENSION") else ".pth"
        ),
        "head_model_path": os.getenv("HEAD_MODEL_PATH"),
        "head_model_max_batch_size": int(os.getenv("HEAD_MODEL_MAX_BATCH_SIZE", "64")),
        "head_visibility_img_size": int(os.getenv("HEAD_VISIBILITY_IMG_SIZE", "224")),
        "head_visible_frame_count": int(os.getenv("HEAD_VISIBLE_FRAME_COUNT", "16")),
        "head_visibility_min_frames": int(os.getenv("HEAD_VISIBILITY_MIN_FRAMES", "10")),
        "head_visibility_sample_rate": int(os.getenv("SAMPLE_HEAD_VISIBILITY", "2")),
        "downsample_head_frames": os.getenv("DOWNSAMPLE_HEAD_FRAMES", "True").lower() == "true",
        "head_visibility_min_bbox_size": int(os.getenv("HEAD_VISIBILITY_MIN_BBOX_SIZE", "30")),
        "use_behavior_analysis": os.getenv("USE_BEHAVIOR_ANALYSIS", "True").lower() == "true",
    }
    return args


def main():
    args = parse_env()
    root = tk.Tk()
    root.title("Real-Time Action Recognition")

    app = ActionRecognitionApp(
        root,
        args["slide_window_perc"],
        args["max_queue_size"],
        args["val_batch_size"],
        args["conf_high_thres"],
        args["conf_low_thres"],
        args["max_frame_width"],
        args["max_frame_height"],
        args["max_clusters"],
        args["factor_tracking"],
        args["classes"],
        args["imgsz"],
        args["max_video_seconds"],
        args["save_frames"],
        args["total_seconds_save_frames"],
        args["save_video"],
        args["skip_water_process_frames"],
        args["max_water_deque_size"],
        args["disable_alerts"],
        args["ip_address"],
        args["camera_type"],
        args["camera_username"],
        args["camera_password"],
        args["detect_water"],
        args["model_extension"],
        args["model_variant"],
        head_model_path=args["head_model_path"],
        head_model_max_batch_size=args["head_model_max_batch_size"],
        head_visibility_img_size=args["head_visibility_img_size"],
        head_visible_frame_count=args["head_visible_frame_count"],
        head_visibility_min_frames=args["head_visibility_min_frames"],
        head_visibility_sample_rate=args["head_visibility_sample_rate"],
        downsample_head_frames=args["downsample_head_frames"],
        head_visibility_min_bbox_size=args["head_visibility_min_bbox_size"],
        use_behavior_analysis=args["use_behavior_analysis"],
    )
    root.mainloop()


if __name__ == "__main__":
    main()
