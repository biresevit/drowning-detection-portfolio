#!/usr/bin/env bash
# Train the human detector 6 times, one after the other: imgsz 1280 / 1600 / 1920, each with and without the
# extra P2 (stride-4) head. When one run ends - finished, failed, or out of memory twice - the next starts.
#
#   plain : train_yolo26n.py     --model yolo26s.pt       (the humans_yolo26s_<imgsz> baseline recipe)
#   p2    : train_yolo26_p2.py   --model yolo26s-p2.yaml  (weights transferred from yolo26s.pt)
#
# Run it on the VM from ~/projects/yolo, detached, so it survives the SSH session:
#   tmux new-session -d -s queue 'bash train_code/run_training_queue.sh'     (the train scripts are taken from the
#                                                                             folder this script is in)
#   tmux attach -t queue        # watch;  Ctrl-b d to leave it running
#
# Re-running the same command later is safe: finished runs (marker in queue_done/) are skipped and an
# interrupted run continues from its last.pt (--resume). Per-run logs: queue_logs/<name>.log.
#
# Settings (environment variables, all optional):
#   EPOCHS=100            epochs per run
#   TAG=v2                suffix of the run names (humans_yolo26s_<imgsz>[_p2]_<TAG>)
#   EXPERIMENT=humans_yolo26   MLflow experiment
#   MLFLOW_URI=http://localhost:5001   MLflow server of this VM (use another port, e.g. 5002, on a second machine so both
#                         can be opened in the browser through their own SSH tunnels)
#   DATA=.../dataset_grouped/data.yaml
#   ONLY=1920_p2          run only the entries whose key matches this regex (keys: 1280_p2 1280_plain ...), e.g.
#                         ONLY='^(1920_p2|1600_plain|1280_plain)$' to share the six runs between two machines
#   FRACTION=1.0          share of the train set (0.02 for a smoke test)
#   DRY_RUN=1             print what would run, start nothing
#   SHUTDOWN_WHEN_DONE=1  `sudo shutdown -h +3` after the last run (stops the instance: no more GPU billing)
#
# Batch sizes are per imgsz and variant (the P2 head needs more memory); ultralytics keeps the effective
# batch at nbs=64 by accumulating gradients, so a smaller batch changes speed and BatchNorm noise, not the step.
# If a run dies with CUDA out of memory it is retried with half the batch, up to twice.

set -u
CODE="$(basename "$(dirname "$(readlink -f "$0")")")"   # the folder this script lives in (train_code, train_code_humans)
cd "$(dirname "$0")/.."            # ~/projects/yolo
PY=.venv/bin/python
DATA="${DATA:-/home/ubuntu/projects/data/humans/dataset_grouped/data.yaml}"
EPOCHS="${EPOCHS:-100}"
TAG="${TAG:-v2}"
EXPERIMENT="${EXPERIMENT:-humans_yolo26}"
MLFLOW_URI="${MLFLOW_URI:-http://localhost:5001}"
FRACTION="${FRACTION:-1.0}"
ONLY="${ONLY:-.}"
DRY_RUN="${DRY_RUN:-0}"
SHUTDOWN_WHEN_DONE="${SHUTDOWN_WHEN_DONE:-0}"

mkdir -p queue_logs queue_done
QLOG=queue_logs/queue.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }

# imgsz variant batch   (order = order of execution)
ENTRIES=(
  "1280 p2    8"
  "1280 plain 8"
  "1600 p2    4"
  "1600 plain 6"
  "1920 p2    3"
  "1920 plain 4"
)

run_cmd() {  # imgsz variant batch name [resume]
  local imgsz=$1 variant=$2 batch=$3 name=$4 resume=${5:-}
  local common=(--data "$DATA" --name "$name" --imgsz "$imgsz" --epochs "$EPOCHS" --batch "$batch"
                --fraction "$FRACTION" --no-amp --mlflow-experiment "$EXPERIMENT" --mlflow-uri "$MLFLOW_URI")
  if [ "$variant" = p2 ]; then
    echo "$PY $CODE/train_yolo26_p2.py --model yolo26s-p2.yaml --weights yolo26s.pt ${common[*]} ${resume:+--resume}"
  else
    echo "$PY $CODE/train_yolo26n.py --model yolo26s.pt --aug default ${common[*]} ${resume:+--resume}"
  fi
}

for entry in "${ENTRIES[@]}"; do
  read -r imgsz variant batch <<<"$entry"
  key="${imgsz}_${variant}"
  [[ "$key" =~ $ONLY ]] || continue
  if [ "$variant" = p2 ]; then name="humans_yolo26s_${imgsz}_p2_${TAG}"; else name="humans_yolo26s_${imgsz}_${TAG}"; fi
  marker="queue_done/$name"

  if [ -f "$marker" ]; then say "SKIP $name (already finished)"; continue; fi

  attempt=0
  while :; do
    resume=""
    if [ -d "runs/detect/$name" ]; then
      if [ -f "runs/detect/$name/weights/last.pt" ]; then
        resume=1
      else  # a run folder without weights is a failed start: keep it aside so ultralytics does not name the next one -2
        mv "runs/detect/$name" "runs/detect/${name}_failed_$(date +%s)"
      fi
    fi
    cmd=$(run_cmd "$imgsz" "$variant" "$batch" "$name" "$resume")
    say "START $name  imgsz=$imgsz batch=$batch epochs=$EPOCHS attempt=$((attempt + 1))${resume:+ (resuming)}"
    say "  $cmd"
    if [ "$DRY_RUN" = 1 ]; then break; fi
    log="queue_logs/${name}.log"
    start=$(date +%s)
    bash -c "$cmd" >>"$log" 2>&1
    rc=$?
    mins=$(( ($(date +%s) - start) / 60 ))
    if [ $rc -eq 0 ]; then
      date >"$marker"
      say "DONE  $name in ${mins} min"
      break
    fi
    if tail -n 200 "$log" | grep -qi "out of memory" && [ $attempt -lt 2 ] && [ "$batch" -gt 1 ]; then
      batch=$(( (batch + 1) / 2 )); attempt=$((attempt + 1))
      say "OOM   $name after ${mins} min -> retrying with batch=$batch"
      [ -d "runs/detect/$name" ] && mv "runs/detect/$name" "runs/detect/${name}_oom_$(date +%s)"
      continue
    fi
    say "FAIL  $name (exit $rc after ${mins} min) - see $log; continuing with the next run"
    break
  done
done

say "queue finished"
if [ "$SHUTDOWN_WHEN_DONE" = 1 ] && [ "$DRY_RUN" != 1 ]; then
  say "shutting the instance down in 3 minutes (cancel: sudo shutdown -c)"
  sudo shutdown -h +3
fi
