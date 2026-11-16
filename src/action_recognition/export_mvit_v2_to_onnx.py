from mvit_v2 import MViTv2
import os

model_path = "/models/drowning_mvit_v2_s.pth"
if os.name == "nt" and model_path.startswith("/"):
    model_path = model_path[1:]
model_variant = "mvit_v2_s"
device = "cuda"
model = MViTv2(
    model_name=model_variant,
    device=device,
    classes=["normal", "drowning"],
    checkpoint_path=model_path,
)
model.export_to_onnx(output_path=model_path.replace(".pth", ".onnx"))
