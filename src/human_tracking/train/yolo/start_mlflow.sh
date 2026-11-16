#!/usr/bin/env bash
# Start the MLflow tracking server on the VM (localhost:${MLFLOW_PORT:-5001}) in a detached tmux session.
# View it from the PC through an SSH tunnel:  ssh -L 5001:localhost:5001 yolo_aws  ->  http://localhost:5001
set -euo pipefail
cd "$(dirname "$0")"

MLFLOW_DIR="$HOME/projects/mlflow"   # EBS, persists across stop/start
mkdir -p "$MLFLOW_DIR/artifacts"

if tmux has-session -t mlflow 2>/dev/null; then
    echo "MLflow already running (tmux session 'mlflow')"
    exit 0
fi

tmux new-session -d -s mlflow "$PWD/.venv/bin/mlflow server --host 127.0.0.1 --port ${MLFLOW_PORT:-5001} \
    --backend-store-uri sqlite:///$MLFLOW_DIR/mlflow.db --artifacts-destination $MLFLOW_DIR/artifacts"
sleep 5
curl -sf http://127.0.0.1:${MLFLOW_PORT:-5001}/health >/dev/null && echo "MLflow running on http://127.0.0.1:5001" \
    || echo "MLflow not responding yet; check with: tmux attach -t mlflow"
