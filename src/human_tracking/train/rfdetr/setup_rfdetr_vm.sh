#!/usr/bin/env bash
# One-time RF-DETR environment on a training VM, separate from the YOLO venv (~/projects/yolo/.venv) so neither
# can break the other. Safe to re-run.
#
#   mkdir -p ~/projects/rfdetr/code   # copy src/human_tracking/train/{common,rfdetr,yolo} there
#   cd ~/projects/rfdetr && bash code/rfdetr/setup_rfdetr_vm.sh
#
# Then convert the dataset once (hardlinks, no extra disk):
#   .venv/bin/python code/rfdetr/yolo_to_coco.py --data /home/ubuntu/projects/data/humans/dataset_grouped/data.yaml \
#       --out /home/ubuntu/projects/data/humans/dataset_grouped_coco
#
# rfdetr[plus] adds RF-DETR XL / 2XL, which are under the PML 1.0 licence (Nano..Large are Apache 2.0).
set -euo pipefail
cd "$(dirname "$0")/../.."          # ~/projects/rfdetr
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv "rfdetr[plus,train,loggers]==1.11.2" mlflow matplotlib opencv-python-headless pyyaml
.venv/bin/python -c "import rfdetr, rfdetr_plus, torch, pytorch_lightning; print('rfdetr OK | torch', torch.__version__, '| cuda', torch.cuda.is_available())"
