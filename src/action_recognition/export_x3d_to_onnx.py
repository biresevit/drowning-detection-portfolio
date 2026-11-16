from x3d import X3d
import os

model_path = "/models/drowning_x3d_s.pth"
if os.name == "nt" and model_path.startswith("/"):
    model_path = model_path[1:]
model_variant = "x3d_s"
device = "cuda"
model = X3d(
    model_name=model_variant,
    device=device,
    classes=["normal", "drowning"],
    checkpoint_path=model_path,
)
model.export_to_onnx(output_path=model_path.replace(".pth", ".onnx"))
