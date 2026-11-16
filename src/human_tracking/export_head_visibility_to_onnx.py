"""
export_head_visibility_to_onnx.py

Exports the trained head-visibility PyTorch checkpoint to ONNX,
simplifies it with onnxslim, then converts to a TensorRT engine.

Usage (from project root):
    python src/human_tracking/export_head_visibility_to_onnx.py
    python src/human_tracking/export_head_visibility_to_onnx.py \
        --checkpoint models/best_efficientnet_v2_s_head_visibility.pth \
        --output     models/head_visibility.onnx \
        --engine     models/head_visibility.engine \
        --model      efficientnet_v2_s \
        --num_classes 4
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int) -> nn.Module:
    name = model_name.lower()
    if name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "efficientnet_v2_m":
        m = models.efficientnet_v2_m()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "convnext_small":
        m = models.convnext_small()
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "vit_b_16":
        m = models.vit_b_16()
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large()
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model '{model_name}'")
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     default="models/head_visibility.pth")
    parser.add_argument("--output",         default="models/head_visibility.onnx")
    parser.add_argument("--engine",         default="models/head_visibility.engine")
    parser.add_argument("--model",          default="efficientnet_v2_s")
    parser.add_argument("--num_classes",    type=int, default=3)
    parser.add_argument("--img_size",       type=int, default=int(os.getenv("HEAD_VISIBILITY_IMG_SIZE", "224")))
    parser.add_argument("--max_batch_size", type=int, default=int(os.getenv("HEAD_MODEL_MAX_BATCH_SIZE", "128")))
    parser.add_argument("--opt_batch_size", type=int, default=16)
    parser.add_argument("--opset",          type=int, default=17)
    args = parser.parse_args()

    # ---- Export to ONNX ----
    device = torch.device("cpu")
    model  = build_model(args.model, args.num_classes)
    state  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1, 3, args.img_size, args.img_size)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"Exported → {args.output}")

    # ---- Simplify with onnxslim ----
    slim_path = args.output.replace(".onnx", "_slim.onnx")
    print("Simplifying ONNX model with onnxslim...")
    subprocess.run(
        [sys.executable, "-m", "onnxslim", args.output, slim_path],
        check=True,
    )
    print(f"Simplified → {slim_path}")

    # ---- Convert to TensorRT engine ----
    sz = args.img_size
    min_shape = f"input:1x3x{sz}x{sz}"
    opt_shape = f"input:{args.opt_batch_size}x3x{sz}x{sz}"
    max_shape = f"input:{args.max_batch_size}x3x{sz}x{sz}"

    print("Converting to TensorRT engine with trtexec...")
    subprocess.run(
        [
            "trtexec",
            f"--onnx={slim_path}",
            f"--saveEngine={args.engine}",
            "--fp16",
            f"--minShapes={min_shape}",
            f"--optShapes={opt_shape}",
            f"--maxShapes={max_shape}",
        ],
        check=True,
    )
    print(f"Engine saved → {args.engine}")


if __name__ == "__main__":
    main()
