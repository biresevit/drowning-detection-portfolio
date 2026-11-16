"""
Measures RAM and GPU memory used by the action recognition model.

On Jetson, GPU and CPU share unified LPDDR memory, so system RAM delta
is the most accurate single number. GPU stats are reported separately
for non-Jetson machines.

Usage:
    python src/action_recognition/model_memory_check.py --model_extension .pth
    python src/action_recognition/model_memory_check.py --model_extension .engine
"""

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    print("psutil not installed — system RAM tracking disabled. pip install psutil")


def _ram_mb() -> float:
    """Current process RSS in MB (works on both Windows and Linux/Jetson)."""
    if _HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1024 ** 2
    return 0.0


def _sys_available_mb() -> float:
    """System-wide available RAM in MB — best signal on Jetson unified memory."""
    if _HAS_PSUTIL:
        return psutil.virtual_memory().available / 1024 ** 2
    return 0.0


def _gpu_mb() -> tuple[float, float]:
    """Returns (allocated_MB, reserved_MB) for the CUDA device."""
    if torch.cuda.is_available():
        return (
            torch.cuda.memory_allocated() / 1024 ** 2,
            torch.cuda.memory_reserved() / 1024 ** 2,
        )
    return 0.0, 0.0


def _snapshot(label: str) -> dict:
    snap = {
        "label": label,
        "proc_rss_mb": _ram_mb(),
        "sys_avail_mb": _sys_available_mb(),
        "gpu_alloc_mb": _gpu_mb()[0],
        "gpu_reserved_mb": _gpu_mb()[1],
    }
    print(
        f"  [{label:30s}]  "
        f"proc RSS {snap['proc_rss_mb']:7.1f} MB  |  "
        f"sys avail {snap['sys_avail_mb']:7.1f} MB  |  "
        f"GPU alloc {snap['gpu_alloc_mb']:7.1f} MB"
    )
    return snap


def _run_dummy_inference(model, model_extension: str, batch_size: int, model_variant: str):
    """One forward pass with a dummy batch to trigger activation allocation."""
    num_frames = 16
    crop_size = 224

    if model_extension == ".engine":
        dummy = np.random.rand(batch_size, 3, num_frames, crop_size, crop_size).astype(np.float16)
        model.infer(dummy)
    else:
        dummy = torch.randn(batch_size, 3, num_frames, crop_size, crop_size, device="cuda")
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model.model(dummy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_variant",  default=os.getenv("MODEL_VARIANT", "swin_t"))
    parser.add_argument("--model_extension", default=".engine", choices=[".pth", ".engine"])
    parser.add_argument("--batch_sizes", default="1,4,8", help="Comma-separated batch sizes to test")
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    classes = ["normal", "drowning"]

    is_docker = os.path.isdir("/models")
    models_dir = "/models" if is_docker else "models"

    print(f"\n{'='*60}")
    print(f"  Model: {args.model_variant}  |  Extension: {args.model_extension}")
    print(f"{'='*60}\n")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    snap_baseline = _snapshot("baseline (before load)")

    # --- Load model ---
    if args.model_extension == ".engine":
        from action_recognition.load_models_tensorrt import TensorRTModelLoader
        loader = TensorRTModelLoader(models_dir=models_dir)
    else:
        from action_recognition.load_models_onnx import GenericModelLoader
        loader = GenericModelLoader(models_dir=models_dir)

    model = loader.load_action_model(
        args.model_variant, classes, model_extension=args.model_extension
    )

    snap_loaded = _snapshot("after model load")

    load_rss_delta   = snap_loaded["proc_rss_mb"]  - snap_baseline["proc_rss_mb"]
    load_avail_delta = snap_baseline["sys_avail_mb"] - snap_loaded["sys_avail_mb"]
    load_gpu_delta   = snap_loaded["gpu_alloc_mb"]  - snap_baseline["gpu_alloc_mb"]

    print(f"\n  >> Load cost:  process RSS +{load_rss_delta:.1f} MB  |  "
          f"sys RAM consumed {load_avail_delta:.1f} MB  |  GPU +{load_gpu_delta:.1f} MB\n")

    # --- Inference at each batch size ---
    inference_model = model if args.model_extension == ".engine" else model.model

    for bs in batch_sizes:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        snap_pre = _snapshot(f"before inference bs={bs}")
        _run_dummy_inference(model, args.model_extension, bs, args.model_variant)
        snap_post = _snapshot(f"after  inference bs={bs}")

        infer_rss   = snap_post["proc_rss_mb"]  - snap_pre["proc_rss_mb"]
        infer_avail = snap_pre["sys_avail_mb"]  - snap_post["sys_avail_mb"]
        infer_gpu   = snap_post["gpu_alloc_mb"] - snap_pre["gpu_alloc_mb"]

        if torch.cuda.is_available():
            peak_gpu = torch.cuda.max_memory_allocated() / 1024 ** 2
        else:
            peak_gpu = snap_post["gpu_alloc_mb"]

        print(f"  >> bs={bs}: activation RSS +{infer_rss:.1f} MB  |  "
              f"sys RAM +{infer_avail:.1f} MB  |  GPU alloc +{infer_gpu:.1f} MB  |  "
              f"GPU peak {peak_gpu:.1f} MB\n")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({args.model_variant} {args.model_extension})")
    print(f"{'='*60}")
    print(f"  Model load (sys RAM consumed): {load_avail_delta:.1f} MB")
    print(f"  Model load (GPU allocated):    {load_gpu_delta:.1f} MB")
    print(f"  Note: on Jetson these two numbers are the same pool.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
