import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from action_recognition.base_model_loader import BaseModelLoader
from action_recognition.x3d_tensorrt import X3dTensorRT
from action_recognition.swin_tensorrt import SwinTensorRT
from human_tracking.tracking_humans_engine import TrackingHumansEngine
import pycuda.driver as cuda
import logging

logger = logging.getLogger()


class TensorRTModelLoader(BaseModelLoader):
    def __init__(self, models_dir="/models"):
        is_docker_env = os.path.exists("/.dockerenv")
        if not is_docker_env and models_dir.startswith("/"):
            models_dir = models_dir.lstrip("/")
        super().__init__(models_dir)
        try:            
            cuda.init()
            self.cuda_context = cuda.Device(0).make_context()
            logger.info("CUDA context initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize CUDA: {e}")
            self.cuda_context = None

    def load_action_model(self, model_variant, classes, model_extension=".engine"):
        try:
            action_model_path = self.get_model_path(
                f"drowning_{model_variant}", model_extension
            )
            logger.info(f"Loading X3dTensorRT model from {action_model_path}...")
            if "swin" in model_variant:
                action_model = SwinTensorRT(
                    model_name=model_variant,
                    engine_path=action_model_path,
                    classes=classes,
                    shared_cuda_context=self.cuda_context,
                )
            else:
                action_model = X3dTensorRT(
                    model_name=model_variant,
                    engine_path=action_model_path,
                    classes=classes,
                    shared_cuda_context=self.cuda_context,
                )
            logger.info(f"Action model {model_variant} loaded successfully.")
            return action_model
        except Exception as e:
            logger.exception(f"Failed to load action model {model_variant}: {e}")
            return None

    def load_human_tracker(
        self,
        imgsz,
        max_clusters,
        skip_water_process_frames,
        max_water_deque_size,
        deque_length=12,
        detect_water=True,
        head_classifier_path=None,
        head_classifier_max_batch_size=64,
        head_visibility_img_size=224,
        head_visible_frame_count=16,
        head_visibility_min_frames=10,
        head_visibility_sample_rate=2,
        downsample_head_frames=True,
        head_visibility_min_bbox_size=30,
        use_behavior_analysis: bool | None = None,
    ):
        """
        Load the human tracker model for TensorRT.

        Args:
            imgsz (int): Image size for detection.
            max_clusters (int): Maximum number of clusters.
            skip_water_process_frames (int): Number of frames to skip for water mask updates.
            max_water_deque_size (int): Maximum size of the water mask deque.
            deque_length (int): Length of the tracking deque for ObjectTracker (default: 12).
            detect_water (bool): Whether to enable water detection (default: True).
            head_classifier_path (str | None): Path to HeadVisibility TensorRT engine.
            head_classifier_max_batch_size (int): Max batch size the engine was built with.
            head_visible_frame_count (int): Number of crops to classify per track.
            downsample_head_frames (bool): Uniformly subsample crops if True, else use last N.

        Returns:
            TrackingHumansEngine: Initialized human tracker.
        """
        try:
            model_extension=".engine"
            human_tracker = TrackingHumansEngine(
                human_model_path=self.get_model_path("humans", model_extension),
                water_model_path=self.get_model_path("water_640", model_extension),
                threshold_frames=1,
                tracking_deque_length=deque_length,
                num_total_crops=max_clusters,
                shared_cuda_context=self.cuda_context,
                imgsz=imgsz,
                skip_water_process_frames=skip_water_process_frames,
                max_water_deque_size=max_water_deque_size,
                detect_water=detect_water,
                head_classifier_path=head_classifier_path,
                head_classifier_max_batch_size=head_classifier_max_batch_size,
                head_visibility_img_size=head_visibility_img_size,
                head_visible_frame_count=head_visible_frame_count,
                head_visibility_min_frames=head_visibility_min_frames,
                head_visibility_sample_rate=head_visibility_sample_rate,
                downsample_head_frames=downsample_head_frames,
                head_visibility_min_bbox_size=head_visibility_min_bbox_size,
                use_behavior_analysis=use_behavior_analysis,
            )
            logger.info("Human tracker loaded successfully.")
            return human_tracker
        except Exception as e:
            logger.exception(f"Failed to load human tracker: {e}")
            return None
