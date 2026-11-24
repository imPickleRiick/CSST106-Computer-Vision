"""Lab 6, Parts 2–4: train and evaluate a custom YOLO11 model.

Dataset: Ultralytics "African Wildlife" (4 classes: buffalo, elephant, rhino, zebra),
already annotated in YOLO format:
    datasets/african-wildlife/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt   (class_id x_center y_center width height, normalised)

Usage (from the project root, inside the virtual environment):
    python training/train_lab6.py            # train + validate + export artifacts
    python training/train_lab6.py --epochs 50 --imgsz 640

The best weights and evaluation plots are copied to models/lab6/ so the web app
can load them without the dataset or the runs/ folder.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "lab6"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default=str(ROOT / "models" / "yolo11n.pt"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from ultralytics import YOLO, settings
    from ultralytics.data.utils import check_det_dataset

    settings.update({"datasets_dir": str(ROOT / "datasets"), "sync": False})

    # Part 2: dataset (downloaded and unpacked automatically the first time)
    info = check_det_dataset("african-wildlife.yaml", autodownload=True)
    data_yaml = ROOT / "training" / "data.yaml"
    data_yaml.write_text(
        "# Lab 6 custom dataset (YOLO format)\n"
        f"path: {Path(info['path']).as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(info['names'])}\n"
        f"names: {list(info['names'].values())}\n",
        encoding="utf-8",
    )

    # Part 3: training (transfer learning from COCO-pretrained YOLO11n)
    model = YOLO(args.model)
    t0 = time.time()
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=4, project=str(ROOT / "runs" / "detect"),
                name="lab6_wildlife", exist_ok=True, seed=0, deterministic=True, plots=True, verbose=True)
    train_minutes = (time.time() - t0) / 60
    run_dir = ROOT / "runs" / "detect" / "lab6_wildlife"

    # Part 4: evaluation on the validation and test splits with the best weights
    best = YOLO(str(run_dir / "weights" / "best.pt"))
    metrics = {}
    for split in ("val", "test"):
        m = best.val(data=str(data_yaml), split=split, imgsz=args.imgsz, device=args.device,
                     project=str(ROOT / "runs" / "detect"), name=f"lab6_eval_{split}", exist_ok=True,
                     plots=True, verbose=False)
        per_class = {}
        for i, c in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            per_class[best.names[int(c)]] = {"precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap}
        metrics[split] = {
            "precision": float(m.box.mp), "recall": float(m.box.mr),
            "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
            "per_class": per_class,
            "speed_ms_per_image": {k: float(v) for k, v in m.speed.items()},
        }

    # Export artifacts used by the web app
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "weights" / "best.pt", OUT / "best.pt")
    for f in ("results.csv", "results.png", "labels.jpg", "args.yaml", "train_batch0.jpg"):
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, OUT / f)
    for split in ("val", "test"):
        ev = ROOT / "runs" / "detect" / f"lab6_eval_{split}"
        for f in ("confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png",
                  "val_batch0_pred.jpg", "val_batch0_labels.jpg"):
            if (ev / f).exists():
                shutil.copy(ev / f, OUT / f"{split}_{f}")
    summary = {
        "dataset": "Ultralytics African Wildlife",
        "classes": list(info["names"].values()),
        "base_model": Path(args.model).name,
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "device": args.device,
        "train_minutes": round(train_minutes, 1),
        "metrics": metrics,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
