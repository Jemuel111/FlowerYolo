# 🌸 FloraLens — Flower Recognition System
### ITST 303 · Group Performance Task #3

An AI-powered flower classification web application built with **Ultralytics YOLOv8** and **Flask**, trained on the [Kaggle Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).

---

## 📁 Project Structure

```
flower_yolo_app/
├── app.py                  ← Flask application (main entry point)
├── train_flowers.py        ← YOLOv8 training script
├── requirements.txt        ← Python dependencies
├── best.pt                 ← (place trained model weights here)
├── templates/
│   ├── index.html          ← Upload form + results display (Jinja2)
│   └── about.html          ← About / tech info page
└── static/
    ├── uploads/            ← User-uploaded images (auto-created)
    └── results/            ← YOLO-annotated output images (auto-created)
```

---

## 🚀 Setup & Installation

### 1. Clone / Download the project

```bash
cd flower_yolo_app
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🌼 Dataset Setup (Kaggle)

### Option A — Kaggle CLI (fastest)

```bash
pip install kaggle

# Place your kaggle.json API key in ~/.kaggle/
kaggle datasets download -d alxmamaev/flowers-recognition
unzip flowers-recognition.zip -d flowers_raw
```

### Option B — Manual Download

1. Go to https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
2. Click **Download**
3. Unzip and place the `flowers/` folder inside a `flowers_raw/` directory

**Expected structure after unzip:**
```
flowers_raw/
└── flowers/
    ├── daisy/       (~769 images)
    ├── dandelion/   (~1052 images)
    ├── rose/        (~784 images)
    ├── sunflower/   (~734 images)
    └── tulip/       (~984 images)
```

---

## 🤖 Model Training

Run the training script to fine-tune YOLOv8 on your flowers dataset:

```bash
python train_flowers.py
```

**What happens:**
1. Splits images → `flowers_dataset/train/`, `val/`, `test/` (80/10/10)
2. Fine-tunes `yolov8n-cls.pt` for 50 epochs
3. Saves best weights to `runs/classify/flowers_train/weights/best.pt`
4. Evaluates on test set (Top-1 & Top-5 accuracy)
5. Runs sample inference and saves annotated images

**After training:**

```bash
# Copy trained weights to project root
cp runs/classify/flowers_train/weights/best.pt .
```

Then update `app.py` line 16:
```python
# Before (pretrained)
model = YOLO("yolov8n-cls.pt")

# After (custom trained)
model = YOLO("best.pt")
```

---

## 🌐 Running the Flask App

```bash
python app.py
```

Visit: **http://127.0.0.1:5000**

---

## 📸 How It Works

```
User uploads image
        ↓
Flask saves to static/uploads/
        ↓
YOLO runs inference (model(filepath))
        ↓
Result image saved to static/results/
        ↓
Top-5 predictions extracted with confidence scores
        ↓
Jinja2 template renders results + images
```

---

## 🌺 Recognizable Flowers

| Flower     | Emoji | Training Images | Description                          |
|------------|-------|-----------------|--------------------------------------|
| Daisy      | 🌼    | ~769            | White petals, yellow center          |
| Dandelion  | 💛    | ~1,052          | Bright yellow wildflower             |
| Rose       | 🌹    | ~784            | Classic bloom, many colors           |
| Sunflower  | 🌻    | ~734            | Large, bold, heliotropic             |
| Tulip      | 🌷    | ~984            | Cup-shaped spring bloom              |

---

## ⚙️ Key Flask Routes

| Route       | Method   | Description                            |
|-------------|----------|----------------------------------------|
| `/`         | GET      | Home page with upload form             |
| `/predict`  | POST     | Accepts image, runs YOLO, returns results |
| `/about`    | GET      | About page with tech info              |

---

## 📋 Task Checklist (ITST 303 GPT #3)

- [x] Install Ultralytics YOLOv8
- [x] Load YOLO model
- [x] Custom dataset (Kaggle Flowers Recognition)
- [x] Training script with 80/10/10 split
- [x] Training results (loss, accuracy via `model.train()`)
- [x] Run detection on new images
- [x] Save output images with YOLO annotations
- [x] Flask web app with file upload form (`enctype="multipart/form-data"`)
- [x] Jinja2 templates (`index.html`, `about.html`)
- [x] Display original + detected image side by side
- [x] Show top-5 predictions with confidence bars
- [x] Secure file handling (`werkzeug.utils.secure_filename`)

---

## 👥 Group Members

> *(Add your group members here)*

---

## 📚 References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [Kaggle Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- [Flask Documentation](https://flask.palletsprojects.com)
- [Jinja2 Documentation](https://jinja.palletsprojects.com)
