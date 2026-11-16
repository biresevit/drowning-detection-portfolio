#!/usr/bin/env bash
# Time one short training epoch per (model, resolution, batch) and record peak GPU memory, to size the queue.
# Uses a small COCO copy (yolo_to_coco.py --limit 200) so each config takes a few minutes.
#
#   cd ~/projects/rfdetr && bash code/rfdetr/bench_rfdetr.sh            # -> bench/bench.csv
#   CONFIGS="large:1216:2 nano:384:8" bash code/rfdetr/bench_rfdetr.sh   # only these
set -u
cd "$(dirname "$0")/../.."          # ~/projects/rfdetr
PY=.venv/bin/python
D=/home/ubuntu/projects/data/humans
SMOKE="${SMOKE:-$D/dataset_grouped_coco_200}"
[ -d "$SMOKE" ] || $PY code/rfdetr/yolo_to_coco.py --data $D/dataset_grouped/data.yaml --out "$SMOKE" --limit 200
CONFIGS="${CONFIGS:-nano:384:16 nano:1216:4 small:512:16 small:1216:4 large:704:8 large:1216:4 xlarge:700:8 xlarge:1220:4 2xlarge:880:4 2xlarge:1200:2}"
mkdir -p bench
OUT=bench/bench.csv
[ -f "$OUT" ] || echo "model,resolution,batch,status,s_per_epoch_200img,peak_mem_mib,wall_s" > "$OUT"
EPOCHS="${EPOCHS:-3}"   # early stopping logs one line per epoch: the gap between the last two is one epoch's time

for c in $CONFIGS; do
  IFS=: read -r model res batch <<<"$c"
  name="bench_${model}_${res}_b${batch}"
  rm -rf "bench/$name"
  log="bench/$name.log"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "bench/$name.mem" &
  smi=$!
  start=$(date +%s)
  $PY code/rfdetr/train_rfdetr.py --model "$model" --resolution "$res" --batch "$batch" --grad-accum 1 \
      --epochs "$EPOCHS" --name "$name" --runs-dir bench --dataset-dir "$SMOKE" --mlflow-uri "" \
      --size-buckets 0 --size-edges "" --num-prediction-images 0 --patience 100 >"$log" 2>&1
  rc=$?
  wall=$(( $(date +%s) - start ))
  kill $smi 2>/dev/null; wait $smi 2>/dev/null
  peak=$(sort -n "bench/$name.mem" | tail -1)
  status=ok
  [ $rc -ne 0 ] && status=fail && grep -qi "out of memory" "$log" && status=oom
  # one epoch (train + val on the small copy) = time between the last two per-epoch early-stopping lines
  s_epoch=$(grep -a -E "Metric __rfdetr_effective_map__|did not improve" "$log" | cut -c1-23 | tail -2 | \
      $PY -c "import sys, datetime as d
t = [d.datetime.strptime(l.strip(), '%Y-%m-%d %H:%M:%S,%f') for l in sys.stdin if l.strip()]
print(f'{(t[1] - t[0]).total_seconds():.1f}' if len(t) == 2 else '')")
  echo "$model,$res,$batch,$status,${s_epoch:-},$peak,$wall" | tee -a "$OUT"
  rm -rf "bench/$name"             # weights of a short run are useless; keep the log and memory trace
done
