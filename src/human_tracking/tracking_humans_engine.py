from water_detection.tensorrt_inference import SegmentationTrtInfer
from human_tracking.tracking_humans_base import TrackingHumansBase


class TrackingHumansEngine(TrackingHumansBase):
    def __init__(
        self, human_model_path, water_model_path, shared_cuda_context=None, **kwargs
    ):
        super().__init__(human_model_path, **kwargs)
        if self.detect_water:
            self.water_model = SegmentationTrtInfer(
                model_path=water_model_path, shared_cuda_context=shared_cuda_context
            )
