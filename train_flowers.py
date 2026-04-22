"""
train_flowers.py
────────────────────────────────────────────────────────────────────────────────
Train a YOLOv8 classification model on the Kaggle Flowers Recognition dataset.

Dataset: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
Structure after unzip:
    flowers/
        daisy/        ← ~769 images
        dandelion/    ← ~1052 images
        rose/         ← ~784 images
        sunflower/    ← ~734 images
        tulip/        ← ~984 images

Steps:
    1. pip install ultralytics kaggle
    2. kaggle datasets download -d alxmamaev/flowers-recognition
    3. unzip flowers-recognition.zip -d flowers_raw
    4. python train_flowers.py
    5. Trained weights → runs/classify/flowers_train/weights/best.pt
    6. Copy best.pt to your project root and update app.py model path.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# ─── CONFIG ─────────────────────────────────────────────────────────────────
RAW_DATASET_DIR = Path("flowers_raw/flowers")   # adjust if needed
PREPARED_DIR    = Path("flowers_dataset")       # YOLOv8-ready split
TRAIN_RATIO     = 0.80
VAL_RATIO       = 0.10
TEST_RATIO      = 0.10
IMAGE_SIZE      = 224
EPOCHS          = 50
BATCH_SIZE      = 32
SEED            = 42
MODEL_BASE      = "yolov8n-cls.pt"             # nano classification model
PROJECT_NAME    = "runs/classify"
RUN_NAME        = "flowers_train"

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def prepare_dataset():
    """Split raw images into train/val/test folders for YOLOv8 classification."""
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
        images = list((RAW_DATASET_DIR / cls).glob("*.jpg")) + \
                 list((RAW_DATASET_DIR / cls).glob("*.jpeg")) + \
                 list((RAW_DATASET_DIR / cls).glob("*.png"))
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

        print(f"  {cls}: {n_train} train / {n_val} val / {n - n_train - n_val} test")

    print(f"[INFO] Dataset prepared at '{PREPARED_DIR}'")


def train():
    """Fine-tune YOLOv8 classification on the flowers dataset."""
    model = YOLO(MODEL_BASE)
    print(f"[INFO] Starting training with {MODEL_BASE} for {EPOCHS} epochs...")

    results = model.train(
        data    = str(PREPARED_DIR),
        task    = "classify",
        epochs  = EPOCHS,
        imgsz   = IMAGE_SIZE,
        batch   = BATCH_SIZE,
        project = PROJECT_NAME,
        name    = RUN_NAME,
        exist_ok= True,
        patience= 10,            # early stopping
        lr0     = 0.001,
        dropout = 0.2,
        seed    = SEED,
        verbose = True,
    )
    return results


def evaluate(best_weights: Path):
    """Validate the trained model on the test split."""
    model = YOLO(str(best_weights))
    metrics = model.val(data=str(PREPARED_DIR), split="test")
    print("\n[RESULTS] Test set evaluation:")
    print(f"  Top-1 Accuracy : {metrics.top1:.4f}")
    print(f"  Top-5 Accuracy : {metrics.top5:.4f}")
    return metrics


def run_sample_inference(best_weights: Path):
    """Run inference on a few test images and save results."""
    model = YOLO(str(best_weights))
    output_dir = Path("sample_detections")
    output_dir.mkdir(exist_ok=True)

    test_images = []
    for cls_dir in (PREPARED_DIR / "test").iterdir():
        imgs = list(cls_dir.glob("*.jpg"))[:2]
        test_images.extend(imgs)

    if not test_images:
        print("[WARN] No test images found for sample inference.")
        return

    for img_path in test_images[:10]:
        results = model(str(img_path))
        out_path = output_dir / img_path.name
        results[0].save(filename=str(out_path))
        top_cls  = results[0].names[results[0].probs.top1]
        top_conf = results[0].probs.top1conf.item()
        print(f"  {img_path.name} → {top_cls} ({top_conf:.2%})")

    print(f"[INFO] Sample detections saved to '{output_dir}'")


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  FloraLens — YOLOv8 Flower Classification Training")
    print("=" * 60)

    # 1. Prepare dataset
    prepare_dataset()

    # 2. Train
    train_results = train()

    # 3. Locate best weights
    best_pt = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[ERROR] best.pt not found at {best_pt}")
        raise FileNotFoundError(str(best_pt))

    print(f"\n[INFO] Best weights saved at: {best_pt}")

    # 4. Evaluate on test set
    evaluate(best_pt)

    # 5. Sample inference
    run_sample_inference(best_pt)

    print("\n[DONE] Training complete!")
    print(f"  → Copy '{best_pt}' to your Flask project root.")
    print("  → Update app.py:  model = YOLO('best.pt')")
    print("=" * 60)
