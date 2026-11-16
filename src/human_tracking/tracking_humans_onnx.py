from water_detection.onnx_inference import SegmentationOnnxInfer
from human_tracking.tracking_humans_base import TrackingHumansBase


class TrackingHumansOnnx(TrackingHumansBase):
    def __init__(self, human_model_path, water_model_path, **kwargs):
        super().__init__(human_model_path, **kwargs)
        if self.detect_water:
            self.water_model = SegmentationOnnxInfer(model_path=water_model_path)
