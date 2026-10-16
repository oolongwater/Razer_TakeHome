"""Fine-tune YOLO11n on the CocoDoom training maps."""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

EPOCHS = 12
IMAGE_SIZE = 320
BATCH = 16
PATIENCE = 4
# A full pass is about 18 minutes per epoch on an M2 (3.5 it/s, 3819 steps).
# CocoDoom training frames are taken every 5 tics, so neighbours are near
# duplicates. 15% still covers every training map and finishes in one sitting.
FRACTION = 0.15


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11n for Doom sprites.")
    parser.add_argument("--data", type=Path, default=Path("data/yolo/data.yaml"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--weights-out", type=Path, default=Path("models/yolo11n-doom.pt"))
    args = parser.parse_args()

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        device=args.device,
        project=str(Path("runs").resolve()),
        name="doom",
        patience=PATIENCE,
        workers=2,
        exist_ok=True,
        fraction=FRACTION,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    args.weights_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.weights_out)
    print(f"copied {best} -> {args.weights_out}")


if __name__ == "__main__":
    main()
