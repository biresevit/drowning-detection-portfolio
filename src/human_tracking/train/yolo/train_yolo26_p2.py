"""Train YOLO26 with an extra P2/4 detection head, for small humans at a low input size.

The stock yolo26 detects at strides 8/16/32 (P3-P5). At imgsz 1280 the finest cell is
8 px, and 25% of the boxes in dataset_grouped are smaller than that once letterboxed -
they cannot be localised, only guessed at. Raising imgsz fixes it by brute force and
costs 2.25x the compute at 1920, which the Jetson Orin Nano cannot afford.

yolo26-p2 adds a stride-4 head instead. The finest cell becomes 4 px, so the same boxes
land on 2+ cells at the SAME input size, for ~1.22x the compute of the 1280 baseline.

    GFLOPs, yolo26s      P3-P5 @1280   91.4   <- baseline
                         P3-P5 @1600  142.7
                         P3-P5 @1920  205.5
                         P2-P5 @1280  111.2   <- this script

Pretrained yolo26s.pt has no P2 layers, so the transfer is partial: `.load()` copies
every matching backbone and neck tensor and leaves the new head random. Expect the
first few epochs to trail the baseline before catching up.

Everything else mirrors train_yolo26n.py so the run is comparable to humans_yolo26s_1280.
"""

import argparse
import os

from ultralytics import YOLO, settings
from ultralytics.utils import LOGGER

from best_predictions_callback import BestPredictionLogger, log_test_results
from train_yolo26n import CollapseGuard


def main():
    parser = argparse.ArgumentParser(
        description="Train yolo26-p2 (P2/4-P5/32) for small-object detection",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--model", default="yolo26s-p2.yaml",
                        help="architecture yaml; the -p2 variants add the stride-4 head")
    parser.add_argument("--weights", default="yolo26s.pt",
                        help="pretrained weights to transfer into it ('' = train from scratch)")
    parser.add_argument("--data",
                        default="/home/ubuntu/projects/data/humans/dataset_grouped/data.yaml")
    parser.add_argument("--name", default="humans_yolo26s_p2_1280")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cls", type=float, default=0.5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                        help="baseline runs used --no-amp; keep it off to stay comparable")
    parser.add_argument("--collapse-ratio", type=float, default=0.5)
    parser.add_argument("--collapse-epochs", type=int, default=3)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mlflow-uri", default="http://localhost:5001")
    parser.add_argument("--mlflow-experiment", default="humans_yolo26")
    parser.add_argument("--num-prediction-images", "--num_prediction_images",
                        type=int, default=100)
    parser.add_argument("--pred-conf", type=float, default=0.25)
    args = parser.parse_args()

    settings.update({"mlflow": bool(args.mlflow_uri)})
    if args.mlflow_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment

    if args.resume:
        yolo = YOLO(f"runs/detect/{args.name}/weights/last.pt")
    else:
        yolo = YOLO(args.model)
        if args.weights:
            # Matches by name and shape: backbone and neck carry over, the P2 head does not.
            yolo = yolo.load(args.weights)
            LOGGER.info(f"transferred weights from {args.weights} into {args.model}")

    pred_logger = BestPredictionLogger(args.num_prediction_images, args.pred_conf)
    yolo.add_callback("on_fit_epoch_end", pred_logger.on_fit_epoch_end)
    if args.collapse_ratio > 0:
        yolo.add_callback("on_fit_epoch_end",
                          CollapseGuard(args.collapse_ratio, args.collapse_epochs).on_fit_epoch_end)

    if args.resume:
        yolo.train(resume=True)
    else:
        yolo.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                   fraction=args.fraction, optimizer=args.optimizer, lr0=args.lr0,
                   cos_lr=args.cos_lr, cls=args.cls, amp=args.amp,
                   save_period=50, name=args.name)

    trainer = yolo.trainer
    log_test_results(trainer.best, trainer.args.data, trainer.save_dir, pred_logger.run_id,
                     args.num_prediction_images, args.pred_conf, trainer.args.imgsz)


if __name__ == "__main__":
    main()
