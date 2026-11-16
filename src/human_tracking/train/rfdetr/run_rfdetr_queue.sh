#!/usr/bin/env bash
# Train RF-DETR on the human dataset for several (model, resolution) pairs, one after the other, like
# yolo/run_training_queue.sh. When one run ends - finished, failed, or out of memory - the next starts.
#
# Run it on the VM from ~/projects/rfdetr, detached, so it survives the SSH session:
#   tmux new-session -d -s queue 'bash code/rfdetr/run_rfdetr_queue.sh'
#   tmux attach -t queue        # watch;  Ctrl-b d to leave it running
#
# Re-running is safe: finished runs (marker in queue_done/) are skipped and an interrupted run continues from its
# last.ckpt (--resume). Per-run logs: queue_logs/<name>.log, queue log: queue_logs/queue.log.
#
# Settings (environment variables, all optional):
#   EPOCHS=50             epochs per run (early stopping on val mAP50:95 usually ends it sooner)
#   PATIENCE=10           early-stopping patience in epochs
#   TAG=v2                suffix of the run names (humans_rfdetr_<model>_<res>_<TAG>); v1 = first runs, flat LR (superseded)
#   EXPERIMENT=humans_rfdetr   MLflow experiment
#   MLFLOW_URI=http://localhost:5001   MLflow server of this VM (5002 on the second machine, as for YOLO)
#   ONLY=regex            run only entries whose key <model>_<res> matches, e.g. ONLY='^(large_1216|nano_.*)$'
#                         to share the entries between two machines
#   DRY_RUN=1             print what would run, start nothing
#   KEEP_RESUME=1         keep last.ckpt / last_ema.pth / checkpoint_best_ema.pth of finished runs (default: delete
#                         them, ~4x the size of the best weights; checkpoint_best_total.pth is always kept)
#   MIN_FREE_GB=5         do not start a run with less free disk than this
#   NUM_WORKERS=6         data-loader workers per run
#   SHUTDOWN_WHEN_DONE=1  `sudo shutdown -h +3` after the last run (stops the instance: no more GPU billing)
#
# Effective batch = batch x grad_accum = 16 everywhere (RF-DETR's default target), so runs differ in model and
# resolution only. On CUDA out of memory a run is retried with half the batch and twice the accumulation (same
# effective batch); at batch 1 the retry turns on gradient checkpointing instead.

set -u
cd "$(dirname "$(readlink -f "$0")")/../.."     # ~/projects/rfdetr
PY=.venv/bin/python
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
TAG="${TAG:-v2}"
EXPERIMENT="${EXPERIMENT:-humans_rfdetr}"
MLFLOW_URI="${MLFLOW_URI:-http://localhost:5001}"
ONLY="${ONLY:-.}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_RESUME="${KEEP_RESUME:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
NUM_WORKERS="${NUM_WORKERS:-6}"   # data loading is the bottleneck at low resolution (8 vCPUs on the VMs)
SHUTDOWN_WHEN_DONE="${SHUTDOWN_WHEN_DONE:-0}"

mkdir -p queue_logs queue_done runs
QLOG=queue_logs/queue.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }

# model resolution batch   (order = order of execution; grad_accum = 16 / batch)
# Batches from bench_rfdetr.sh on an RTX PRO 4500 (32 GB): peak 10-22 GB, so crowded images still have headroom.
# Estimated hours for 50 epochs in brackets; the two halves are split by ONLY so each VM gets ~45 h:
#   yolo_aws2: ONLY='^(large_1216|small_512|nano_384|xlarge_1220|2xlarge_880)$'
#   yolo_aws:  ONLY='^(small_1216|nano_1216|large_704|xlarge_700|2xlarge_1200)$'
ENTRIES=(
  "large   1216 4"    # (7)   the main question: does ~1200 px make RF-DETR competitive with YOLO26s@1600?
  "small   1216 4"    # (7)
  "small   512  16"   # (4)   native resolutions: how much the input size matters
  "nano    1216 4"    # (6.5)
  "nano    384  16"   # (5)
  "large   704  8"    # (4.5)
  "xlarge  1220 4"    # (16)  rfdetr-plus, PML 1.0 licence
  "xlarge  700  8"    # (12)
  "2xlarge 880  4"    # (13)
  "2xlarge 1200 2"    # (16)
)

for entry in "${ENTRIES[@]}"; do
  read -r model res batch <<<"$entry"
  key="${model}_${res}"
  [[ "$key" =~ $ONLY ]] || continue
  name="humans_rfdetr_${model}_${res}_${TAG}"
  marker="queue_done/$name"
  if [ -f "$marker" ]; then say "SKIP $name (already finished)"; continue; fi

  attempt=0
  ckpt=""
  while :; do
    free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
      say "STOP  only ${free_gb} GB free (< MIN_FREE_GB=$MIN_FREE_GB); free disk space and re-run the queue"
      exit 1
    fi
    accum=$(( 16 / batch )); [ $accum -lt 1 ] && accum=1
    resume=""; [ -f "runs/$name/last.ckpt" ] && resume="--resume"
    cmd="$PY code/rfdetr/train_rfdetr.py --model $model --resolution $res --name $name --epochs $EPOCHS \
--batch $batch --grad-accum $accum --patience $PATIENCE --num-workers $NUM_WORKERS --mlflow-experiment $EXPERIMENT --mlflow-uri $MLFLOW_URI \
$ckpt $resume"
    say "START $name  res=$res batch=$batch x accum=$accum epochs=$EPOCHS attempt=$((attempt + 1))${resume:+ (resuming)}${ckpt:+ (gradient checkpointing)}"
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
      if [ "$KEEP_RESUME" != 1 ]; then
        rm -f "runs/$name/last.ckpt" "runs/$name/last_ema.pth" "runs/$name/checkpoint_best_ema.pth" \
              "runs/$name/checkpoint_best_regular.pth"
      fi
      break
    fi
    if tail -n 300 "$log" | grep -qi "out of memory" && [ $attempt -lt 3 ]; then
      attempt=$((attempt + 1))
      if [ "$batch" -gt 1 ]; then
        batch=$(( batch / 2 ))
        say "OOM   $name after ${mins} min -> retrying with batch=$batch"
      elif [ -z "$ckpt" ]; then
        ckpt="--gradient-checkpointing"
        say "OOM   $name after ${mins} min at batch 1 -> retrying with gradient checkpointing"
      else
        say "FAIL  $name: out of memory even at batch 1 with gradient checkpointing; continuing with the next run"
        break
      fi
      rm -rf "runs/$name"           # an OOM'd start: begin clean with the new batch
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
