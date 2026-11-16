import torch
import torchvision
import argparse
import os
import network
import utils
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
parser = argparse.ArgumentParser()

parser.add_argument(
    "-i", "--input", default="models/model.pth", help="Path to the input model"
)
parser.add_argument(
    "-s", "--size", type=int, default=640, help="Input size of the model"
)
parser.add_argument(
    "-c", "--num_classes", type=int, default=2, help="Number of classes"
)
parser.add_argument(
    "-o",
    "--output_stride",
    type=int,
    default=16,
    help="Output stride of deeplabv3+ model",
)
parser.add_argument(
    "-m", "--model_type", type=str, default="deeplabv3plus_mobilenet", help="Model type"
)
parser.add_argument(
    "--separable_conv",
    action="store_true",
    default=True,
    help="apply separable conv to decoder and aspp",
)


def load_weights_from_pth(pth_model_path, model):
    checkpoint = torch.load(pth_model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"])


def convert_model(path, size, model_type, num_classes, output_stride, separable_conv):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    dst_path = path.replace(".pth", ".onnx")

    # Load the model
    model = network.modeling.__dict__[model_type](
        num_classes=num_classes, output_stride=output_stride
    )
    if separable_conv:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    print("Loaded model architecture")

    load_weights_from_pth(path, model)
    print("Loaded weights from .pth file")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()

    # Create a dummy input tensor and move it to the same device
    dummy_input = torch.randn(1, 3, size, size, device=device)

    # Export the trained model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        dst_path,
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=11,  # Specify the ONNX opset version to use
        do_constant_folding=True,  # Optimize constant folding for inference
        input_names=["input"],  # Specify input tensor names
        output_names=["output"],  # Specify output tensor names
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },  # Enable dynamic axes
    )
    print(f"Model successfully converted to ONNX and saved at {dst_path}")


if __name__ == "__main__":
    args = parser.parse_args()

    convert_model(
        args.input,
        args.size,
        args.model_type,
        args.num_classes,
        args.output_stride,
        args.separable_conv,
    )
