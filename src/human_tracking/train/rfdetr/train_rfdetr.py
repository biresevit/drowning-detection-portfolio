"""Train RF-DETR on the human dataset and evaluate it exactly like the YOLO runs.

Mirrors yolo/train_yolo26n.py: one MLflow run per training (experiment --mlflow-experiment, run
name --name), then on the held-out test split
  * RF-DETR's own COCO test metrics (run_test=True),
  * prediction | ground-truth images (test_predictions/),
  * the per-object-size metrics of ../common/size_metrics_core.py - the SAME code and the SAME YOLO
    label files the YOLO runs are scored with, so the numbers compare directly,
  * mean PyTorch inference latency per image at this resolution (test/latency_ms_pytorch; relative
    speed only - deployment would be TensorRT).

Data: --dataset-dir is the COCO copy made by yolo_to_coco.py (train/ valid/ test/); --data is the
original YOLO data.yaml, whose labels the size metrics read.

Resolution: RF-DETR trains on SQUARE inputs (a 1920x1080 frame is resized, not letterboxed). It must
be a multiple of patch_size * num_windows: 32 for nano/small/medium/large, 20 for xlarge, 40 for
2xlarge. xlarge / 2xlarge come from rfdetr-plus (PML 1.0 licence, not Apache 2.0).

    python train_rfdetr.py --model large --resolution 1216 --name humans_rfdetr_l_1216 --epochs 50 \
        --batch 4 --grad-accum 4
    python train_rfdetr.py --model large --resolution 1216 --name humans_rfdetr_l_1216 --resume   # continue
    python train_rfdetr.py --model large --resolution 1216 --name humans_rfdetr_l_1216 --eval-only
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import size_metrics_core as core  # noqa: E402
from pred_images import log_prediction_images, sample_files  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("train_rfdetr")

DEFAULT_DATA = "/home/ubuntu/projects/data/humans/dataset_grouped/data.yaml"
DEFAULT_COCO = "/home/ubuntu/projects/data/humans/dataset_grouped_coco"


def model_class(name):
    import rfdetr

    classes = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}
    if name in classes:
        return getattr(rfdetr, classes[name])
    import rfdetr_plus  # xlarge / 2xlarge: pip install "rfdetr[plus]" (PML 1.0 licence)

    return {"xlarge": rfdetr_plus.RFDETRXLarge, "2xlarge": rfdetr_plus.RFDETR2XLarge}[name]


def check_resolution(model, resolution):
    cfg = model.model_config
    block = cfg.patch_size * cfg.num_windows
    if resolution % block:
        lo, hi = resolution // block * block, (resolution // block + 1) * block
        raise SystemExit(f"--resolution {resolution} is not a multiple of patch_size*num_windows = {block} "
                         f"for this model; use {lo} or {hi}")


def rfdetr_predictor(model, min_conf, single_class):
    """predict(im_file) -> (xyxy, conf, cls, (h, w)) for the shared size metrics."""
    from PIL import Image

    def predict(im_file):
        with Image.open(im_file) as im:
            img = im.convert("RGB")
        det = model.predict(img, threshold=min_conf)
        cls = np.zeros(len(det), dtype=int) if single_class else np.asarray(det.class_id, dtype=int)
        return (np.asarray(det.xyxy, dtype=np.float32).reshape(-1, 4), np.asarray(det.confidence, dtype=np.float32),
                cls, (img.height, img.width))

    return predict


def find_run_id(experiment, run_name):
    """RF-DETR creates the MLflow run inside train(); find it (latest run of that name) to log into it."""
    import mlflow

    runs = mlflow.search_runs(experiment_names=[experiment], filter_string=f"tags.mlflow.runName = '{run_name}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    return None if runs.empty else runs.iloc[0]["run_id"]


def measure_latency(predict, files, warmup=5, n=50):
    import torch

    files = files[: warmup + n]
    for f in files[:warmup]:
        predict(f)
    times = []
    for f in files[warmup:]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        predict(f)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times)) if times else float("nan")


def evaluate(args, out_dir, run_id):
    """Test-split evaluation of checkpoint_best_total.pth with the shared YOLO-compatible metrics."""
    import torch

    best = out_dir / "checkpoint_best_total.pth"
    if not best.exists():
        raise SystemExit(f"no {best}; nothing to evaluate")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    data = core.load_data_yaml(args.data)
    files = sorted(core.list_split_images(data["test"]))
    names = data["names"]
    # from_checkpoint restores the trained class count and resolution; metrics in fp32, as the YOLO runs were
    model = model_class(args.model).from_checkpoint(str(best), trust_checkpoint=True)
    predict = rfdetr_predictor(model, args.min_conf, single_class=len(names) == 1)

    latency = {"fp32": measure_latency(predict, files)}
    # all MLflow writes go through MlflowClient(run_id): a mlflow.start_run(run_id=...) block ends the run on exit
    if run_id and args.num_prediction_images > 0:
        log_prediction_images(predict, sample_files(files, args.num_prediction_images), names,
                              "test_predictions", args.pred_conf, max_side=args.test_image_side, run_id=run_id)
        LOGGER.info(f"MLflow: logged {args.num_prediction_images} test prediction images")

    size_edges = [float(v) for v in args.size_edges.split(",") if v.strip()] or None
    if args.size_buckets > 0 or size_edges:
        core.evaluate(predict, files, names, "test", args.resolution, args.size_buckets, size_edges,
                      conf=args.pred_conf, save_dir=out_dir / "size_metrics", run_id=run_id,
                      split_small=args.size_split_small, min_count=args.size_min_count,
                      extra_edges=[float(v) for v in args.size_extra_edges.split(",") if v.strip()],
                      weights=best, log=LOGGER)

    try:  # relative speed in fp16 (deployment would be TensorRT, faster still); last, as it changes the model
        model.inference(compile=False, dtype=torch.float16)
        latency["fp16"] = measure_latency(predict, files)
    except Exception as e:
        LOGGER.warning(f"fp16 latency not measured: {e}")
    LOGGER.info(f"latency (PyTorch, {args.resolution}px, median of 50 test images): "
                + ", ".join(f"{k} {v:.1f} ms" for k, v in latency.items()))
    if run_id:
        from mlflow import MlflowClient

        for k, v in latency.items():
            MlflowClient().log_metric(run_id, f"test/latency_ms_pytorch_{k}", v)


ARCH_FIELDS = ("encoder", "patch_size", "num_windows", "hidden_dim", "dec_layers", "num_queries", "num_select",
               "projector_scale", "positional_encoding_size", "resolution")


def model_config(name, resolution):
    """The variant's ModelConfig at `resolution`, without building the network or loading weights."""
    import rfdetr.config as C

    if name in ("xlarge", "2xlarge"):
        from rfdetr_plus.models import detection as P

        cls = {"xlarge": P.RFDETRXLargeConfig, "2xlarge": P.RFDETR2XLargeConfig}[name]
    else:
        cls = {"nano": C.RFDETRNanoConfig, "small": C.RFDETRSmallConfig, "medium": C.RFDETRMediumConfig,
               "large": C.RFDETRLargeConfig}[name]
    return cls(resolution=resolution)


def count_params(model):
    """Total parameters of an RFDETR wrapper's torch module, or None if its layout is not the expected one."""
    for path in ("model.model", "model"):
        m = model
        try:
            for part in path.split("."):
                m = getattr(m, part)
            return int(sum(p.numel() for p in m.parameters()))
        except Exception:
            continue
    return None


def run_settings(args, cfg, n_params=None):
    """Everything needed to compare runs side by side: training recipe + architecture of the variant."""
    s = {"model": f"rfdetr-{args.model}", "resolution": args.resolution, "epochs": args.epochs, "batch": args.batch,
         "grad_accum": args.grad_accum, "effective_batch": args.batch * args.grad_accum, "lr": args.lr,
         "lr_encoder": args.lr_encoder, "lr_scheduler": args.lr_scheduler,
         "lr_min_factor" if args.lr_scheduler == "cosine" else "lr_drop":
             args.lr_min_factor if args.lr_scheduler == "cosine" else (args.lr_drop or max(1, round(0.7 * args.epochs))),
         "warmup_epochs": args.warmup_epochs, "early_stopping": args.early_stopping,
         "early_stopping_patience": args.patience, "num_workers": args.num_workers,
         "gradient_checkpointing": args.gradient_checkpointing, "dataset_dir": args.dataset_dir,
         "input": "square resize (not letterboxed)", "framework": "rfdetr",
         "licence": "PML-1.0 (rfdetr-plus)" if args.model in ("xlarge", "2xlarge") else "Apache-2.0"}
    for f in ARCH_FIELDS:
        v = getattr(cfg, f, None)
        if v is not None:
            s[f"arch_{f}"] = ",".join(v) if isinstance(v, (list, tuple)) else v
    s["arch_resolution_multiple"] = cfg.patch_size * cfg.num_windows
    s["arch_tokens_per_side"] = args.resolution // cfg.patch_size
    if n_params:
        s["num_params"] = n_params
        s["num_params_M"] = round(n_params / 1e6, 1)
    return s


def log_run_settings(run_id, settings):
    """Params (the MLflow compare-view columns) plus the headline ones as tags. Params are immutable, so a value
    already logged is left alone; a resumed run logs into a new MLflow run anyway."""
    from mlflow import MlflowClient

    client = MlflowClient()
    have = client.get_run(run_id).data.params
    for k, v in settings.items():
        if k not in have:
            client.log_param(run_id, k, v)
    for k in ("model", "resolution", "effective_batch", "num_params_M", "licence"):
        if k in settings:
            client.set_tag(run_id, k, settings[k])


def log_settings_when_run_appears(experiment, run_name, started_ms, settings, timeout_s=7200):
    """RF-DETR creates its MLflow run inside train(); log the settings into it as soon as it exists."""
    import threading

    def watch():
        import mlflow

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                runs = mlflow.search_runs(experiment_names=[experiment],
                                          filter_string=f"tags.mlflow.runName = '{run_name}' and "
                                                        f"attributes.start_time >= {started_ms}",
                                          order_by=["attributes.start_time DESC"], max_results=1)
                if not runs.empty:
                    log_run_settings(runs.iloc[0]["run_id"], settings)
                    LOGGER.info(f"MLflow: logged {len(settings)} run settings to {runs.iloc[0]['run_id']}")
                    return
            except Exception as e:  # experiment not created yet, server busy, ...
                LOGGER.debug(f"waiting for the MLflow run: {e}")
            time.sleep(15)
        LOGGER.warning("MLflow run did not appear; settings are logged after training instead")

    threading.Thread(target=watch, daemon=True).start()


def watch_best(args):
    """Separate process during training: whenever RF-DETR rewrites checkpoint_best_ema.pth (a new best on val),
    predict a fixed sample of val images with it and log them to best_predictions/epoch_XXX - like the YOLO runs'
    BestPredictionLogger. A separate process so that a failure here can never cost the training run."""
    import mlflow
    import torch

    out_dir = Path(args.runs_dir).resolve() / args.name
    best = out_dir / "checkpoint_best_ema.pth"
    data = core.load_data_yaml(args.data)
    names = data["names"]
    files = sample_files(core.list_split_images(data["val"]), args.num_val_images)   # the same images every time
    started_s = args.watch_started_ms / 1000
    done_mtime = best.stat().st_mtime if best.exists() and best.stat().st_mtime < started_s else 0.0  # resumed run
    run_id = None

    def parent_alive():
        if (out_dir / ".training_done").exists():   # written by the trainer after train(): one last check, then exit
            return False
        try:
            os.kill(args.watch_parent_pid, 0)
            return True
        except OSError:
            return False

    while True:
        alive = parent_alive()
        try:
            mtime = best.stat().st_mtime if best.exists() else 0.0
            if mtime > done_mtime and time.time() - mtime > 5:          # new best, and fully written
                if run_id is None:
                    runs = mlflow.search_runs(experiment_names=[args.mlflow_experiment],
                                              filter_string=f"tags.mlflow.runName = '{args.name}' and "
                                                            f"attributes.start_time >= {args.watch_started_ms}",
                                              order_by=["attributes.start_time DESC"], max_results=1)
                    run_id = None if runs.empty else runs.iloc[0]["run_id"]
                free_gb = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else float("inf")
                if run_id and free_gb < args.val_min_free_gb:
                    # the trainer shares this GPU: never risk ITS out-of-memory for a few logged images
                    LOGGER.warning(f"only {free_gb:.1f} GB GPU memory free (< {args.val_min_free_gb}); "
                                   f"skipping the val images of this best, trying again at the next one")
                    done_mtime = mtime
                elif run_id:
                    epoch = int(torch.load(best, map_location="cpu", weights_only=False).get("epoch", -1)) + 1
                    model = model_class(args.model).from_checkpoint(str(best), trust_checkpoint=True)
                    predict = rfdetr_predictor(model, args.pred_conf, single_class=len(names) == 1)
                    # MlflowClient, not start_run: the run belongs to the trainer and must stay RUNNING
                    log_prediction_images(predict, files, names, f"best_predictions/epoch_{epoch:03d}",
                                          args.pred_conf, max_side=args.val_image_side, run_id=run_id)
                    mlflow.MlflowClient().set_tag(run_id, "best_epoch", epoch)
                    LOGGER.info(f"MLflow: logged {len(files)} val prediction images of the best model (epoch {epoch})")
                    del model, predict
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    done_mtime = mtime
        except Exception as e:  # never fatal: try again on the next new best
            LOGGER.warning(f"val prediction logging failed: {e}")
            done_mtime = max(done_mtime, mtime if "mtime" in locals() else 0.0)
        if not alive:
            return
        time.sleep(20)


def start_best_watcher(args, out_dir):
    """Launch watch_best() as a child process (same args, plus --watch-best)."""
    import subprocess

    (out_dir / ".training_done").unlink(missing_ok=True)   # left over from an earlier attempt of this run
    argv = [a for a in sys.argv[1:] if a != "--resume"]
    cmd = [sys.executable, str(Path(__file__).resolve()), *argv, "--watch-best",
           "--watch-parent-pid", str(os.getpid()), "--watch-started-ms", str(int(time.time() * 1000))]
    log = open(out_dir / "val_predictions.log", "a")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["nano", "small", "medium", "large", "xlarge", "2xlarge"])
    p.add_argument("--resolution", type=int, required=True, help="square input side; see the divisibility rule")
    p.add_argument("--name", required=True, help="run name: output folder runs/<name> and MLflow run name")
    p.add_argument("--data", default=DEFAULT_DATA, help="YOLO data.yaml (labels for the size metrics)")
    p.add_argument("--dataset-dir", default=DEFAULT_COCO, help="COCO copy made by yolo_to_coco.py")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch x grad-accum (16 is RF-DETR's)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-encoder", type=float, default=1.5e-4)
    # RF-DETR's default is "step" with lr_drop=100 epochs: with <= 100 epochs the LR NEVER decays (the first v1
    # runs trained at a flat 1e-4 and plateaued after a few epochs). Cosine anneals over the planned --epochs.
    p.add_argument("--lr-scheduler", choices=["cosine", "step"], default="cosine")
    p.add_argument("--lr-min-factor", type=float, default=0.05, help="cosine: final LR as a fraction of --lr")
    p.add_argument("--lr-drop", type=int, default=0, help="step: epoch of the 10x LR drop (0 = 70%% of --epochs)")
    p.add_argument("--warmup-epochs", type=float, default=0.0, help="linear LR warmup (the pretrained start is stable)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gradient-checkpointing", action="store_true", help="trade speed for memory at high resolution")
    p.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience in epochs (val mAP50:95)")
    p.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.ckpt")
    p.add_argument("--eval-only", action="store_true", help="skip training; evaluate runs/<name>/checkpoint_best_total.pth")
    p.add_argument("--log-params-only", action="store_true",
                   help="only log the run settings (recipe + architecture) into the latest MLflow run named --name, "
                        "e.g. for a run started before settings were logged; pass the same args it was started with")
    p.add_argument("--num-params", type=int, default=0, help="with --log-params-only: parameter count, if known")
    p.add_argument("--mlflow-uri", default="http://localhost:5001", help="MLflow tracking server ('' disables MLflow)")
    p.add_argument("--mlflow-experiment", default="humans_rfdetr")
    p.add_argument("--num-prediction-images", type=int, default=50, help="test images logged after training; 0 disables")
    p.add_argument("--test-image-side", type=int, default=1280, help="long side of each panel of a logged test image")
    p.add_argument("--num-val-images", type=int, default=10,
                   help="val images logged on every new best model during training (fixed sample); 0 disables")
    p.add_argument("--val-image-side", type=int, default=1280, help="long side of each panel of a logged val image")
    p.add_argument("--val-min-free-gb", type=float, default=6.0,
                   help="the val-image watcher shares the GPU with training: it skips a new best when less than "
                        "this much GPU memory is free, so it can never push the training run out of memory")
    p.add_argument("--watch-best", action="store_true", help=argparse.SUPPRESS)       # internal: watcher process
    p.add_argument("--watch-parent-pid", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--watch-started-ms", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--pred-conf", type=float, default=0.25, help="operating point of P/R/F1 and of the logged images")
    p.add_argument("--min-conf", type=float, default=0.001, help="lowest score kept for mAP50 / the confidence sweep")
    p.add_argument("--size-buckets", type=int, default=6)
    p.add_argument("--size-edges", default="6,10,12,14,23,38,62,113",
                   help="fixed px edges (sqrt(w*h)); default = the edges the YOLO comparison used. '' for k-means")
    p.add_argument("--size-split-small", type=int, default=2)
    p.add_argument("--size-min-count", type=int, default=300)
    p.add_argument("--size-extra-edges", default="")
    args = p.parse_args()

    if args.mlflow_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
    out_dir = Path(args.runs_dir).resolve() / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.watch_best:
        watch_best(args)
        return

    if args.log_params_only:
        run_id = find_run_id(args.mlflow_experiment, args.name)
        if not run_id:
            raise SystemExit(f"no MLflow run named {args.name} in {args.mlflow_experiment}")
        settings = run_settings(args, model_config(args.model, args.resolution), args.num_params or None)
        log_run_settings(run_id, settings)
        LOGGER.info(f"MLflow: logged {len(settings)} run settings to {run_id}")
        return

    settings = run_settings(args, model_config(args.model, args.resolution))
    if not args.eval_only:
        model = model_class(args.model)(resolution=args.resolution,
                                        **({"gradient_checkpointing": True} if args.gradient_checkpointing else {}))
        check_resolution(model, args.resolution)
        settings = run_settings(args, model.model_config, count_params(model))
        watcher = None
        if args.mlflow_uri:
            log_settings_when_run_appears(args.mlflow_experiment, args.name, int(time.time() * 1000), settings)
            if args.num_val_images > 0:
                watcher = start_best_watcher(args, out_dir)
        last = out_dir / "last.ckpt"
        if args.resume and not last.exists():
            LOGGER.warning(f"--resume but no {last}; starting from scratch")
        model.train(
            dataset_dir=args.dataset_dir, dataset_file="roboflow", output_dir=str(out_dir),
            epochs=args.epochs, batch_size=args.batch, grad_accum_steps=args.grad_accum,
            lr=args.lr, lr_encoder=args.lr_encoder, num_workers=args.num_workers,
            lr_scheduler=args.lr_scheduler, warmup_epochs=args.warmup_epochs,
            lr_scheduler_kwargs=({"min_factor": args.lr_min_factor} if args.lr_scheduler == "cosine"
                                 else {"lr_drop": args.lr_drop or max(1, round(0.7 * args.epochs))}),
            early_stopping=args.early_stopping, early_stopping_patience=args.patience,
            checkpoint_interval=10_000,   # no periodic archives: disk is tight; best + last.ckpt are kept
            run_test=True, eval_max_dets=500, tensorboard=False,
            mlflow=bool(args.mlflow_uri), project=args.mlflow_experiment, run=args.name,
            resume=str(last) if args.resume and last.exists() else None,
        )
        del model
        if watcher is not None:  # let it log the last new best, then it exits by itself
            (out_dir / ".training_done").touch()
            try:
                watcher.wait(timeout=120)
            except Exception:
                watcher.terminate()

    run_id = find_run_id(args.mlflow_experiment, args.name) if args.mlflow_uri else None
    if run_id:  # again after training, in case the watcher missed the run (already-logged params are skipped)
        log_run_settings(run_id, settings)
    evaluate(args, out_dir, run_id)


if __name__ == "__main__":
    main()
