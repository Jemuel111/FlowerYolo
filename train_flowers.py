"""
train_flowers.py
────────────────────────────────────────────────────────────────────────────────
Train a YOLOv8 classification model on the Kaggle Flowers dataset.
"""

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# ─── CONFIG ─────────────────────────────────────────────────────────────────
RAW_DATASET_DIR = Path("flowers_raw/flowers")
PREPARED_DIR    = Path("flowers_dataset")

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

IMAGE_SIZE  = 224
EPOCHS      = 5
BATCH_SIZE  = 32
SEED        = 42

MODEL_BASE  = "yolov8n-cls.pt"
RUN_NAME    = "flowers_train"

# ─── DATASET PREP ───────────────────────────────────────────────────────────
def prepare_dataset():
    random.seed(SEED)

    if PREPARED_DIR.exists():
        print(f"[INFO] Dataset already prepared at '{PREPARED_DIR}'. Skipping split.")
        return

    classes = [d.name for d in RAW_DATASET_DIR.iterdir() if d.is_dir()]
    print(f"[INFO] Found {len(classes)} classes: {classes}")

    for split in ["train", "val", "test"]:
        for cls in classes:
            (PREPARED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        images = list((RAW_DATASET_DIR / cls).glob("*.*"))
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:]
        }

        for split, files in splits.items():
            for f in files:
                shutil.copy(f, PREPARED_DIR / split / cls / f.name)

        print(f"  {cls}: {len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test")

    print(f"[INFO] Dataset prepared at '{PREPARED_DIR}'")


# ─── TRAINING ───────────────────────────────────────────────────────────────
def train():
    model = YOLO(MODEL_BASE)
    print(f"[INFO] Training {MODEL_BASE} for {EPOCHS} epochs...")

    results = model.train(
        data=str(PREPARED_DIR),
        task="classify",
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name=RUN_NAME,      # ✅ ONLY name (NO project to avoid duplication)
        exist_ok=True,
        patience=10,
        lr0=0.001,
        dropout=0.2,
        seed=SEED,
        verbose=True,
    )

    return results


# ─── EVALUATION ─────────────────────────────────────────────────────────────
def evaluate(best_weights: Path):
    model = YOLO(str(best_weights))
    metrics = model.val(data=str(PREPARED_DIR), split="test")

    print("\n[RESULTS]")
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")


# ─── SAMPLE INFERENCE ───────────────────────────────────────────────────────
def run_sample_inference(best_weights: Path):
    model = YOLO(str(best_weights))

    output_dir = Path("sample_detections")
    output_dir.mkdir(exist_ok=True)

    test_images = []
    for cls_dir in (PREPARED_DIR / "test").iterdir():
        test_images += list(cls_dir.glob("*.*"))[:2]

    if not test_images:
        print("[WARN] No test images found.")
        return

    for img in test_images[:10]:
        results = model(str(img))
        results[0].save(filename=str(output_dir / img.name))

        top_class = results[0].names[results[0].probs.top1]
        conf = results[0].probs.top1conf.item()

        print(f"{img.name} → {top_class} ({conf:.2%})")

    print(f"[INFO] Saved to {output_dir}")


# ─── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(" FloraLens — YOLOv8 Flower Classification")
    print("=" * 60)

    prepare_dataset()

    train()

    # ✅ FIXED PATH (NO DUPLICATION)
    best_pt = Path("runs/classify/flowers_train/weights/best.pt")

    if not best_pt.exists():
        raise FileNotFoundError(f"Missing: {best_pt}")

    print(f"\n[INFO] Best model: {best_pt}")

    evaluate(best_pt)
    run_sample_inference(best_pt)

    print("\n[DONE] Training complete!")
    print(f"Use this in Flask: YOLO('{best_pt}')")