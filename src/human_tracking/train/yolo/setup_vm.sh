#!/usr/bin/env bash
# One-time environment setup on the training VM (Ubuntu, NVIDIA driver preinstalled; nvcc not needed).
# Safe to re-run. Usage: cd ~/projects/yolo && bash setup_vm.sh
set -euo pipefail
cd "$(dirname "$0")"

# System packages: tmux for long runs, portaudio for tflite-support
sudo apt-get update -y && sudo apt-get install -y tmux libportaudio2

# uv (Python + venv manager)
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# The old ultralytics 8.2.2 fork would shadow the pip package; keep it only as a reference
if [ -d ultralytics ]; then
    mv ultralytics ultralytics_8.2.2_fork
fi

# Main env: preprocessing, YOLO26 training, TFLite export, S3 download
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv ultralytics opencv-python-headless matplotlib numpy boto3 tqdm mlflow
uv pip install --python .venv "tensorflow>=2.16" onnx onnxslim onnxruntime onnx2tf tf_keras sng4onnx onnx_graphsurgeon ai-edge-litert \
    || echo "WARNING: export packages failed to install; ultralytics will install them on first export"

# Metadata env: only for data_processing/convert_script/generate_meta_4cls.py (tflite-support needs Python <= 3.11)
[ -d .venv-meta ] || uv venv --python 3.11 .venv-meta
uv pip install --python .venv-meta tflite-support==0.4.4 "protobuf==3.20.3" "numpy<2"
# tflite-support imports its audio module (sounddevice/PortAudio) on import, which fails on a headless server.
# The metadata writer does not need audio, so replace sounddevice with an empty stub.
uv pip uninstall --python .venv-meta sounddevice || true
SP=$(.venv-meta/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo '# stub: tflite-support audio not needed on headless server' > "$SP/sounddevice.py"

# Checks
.venv/bin/python -c "import torch, ultralytics; print('ultralytics', ultralytics.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
.venv-meta/bin/python -c "from tflite_support.metadata_writers import object_detector; print('tflite-support OK')"
echo "Done. Activate with: source .venv/bin/activate"
