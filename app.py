import os
import time
import json
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join('static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Load YOLO model (custom trained or pretrained)
# If you have trained a custom model, replace with: YOLO("runs/classify/train/weights/best.pt")
model = YOLO("yolov8n-cls.pt")  # classification model for flowers

# Flower class names (update after training with your dataset)
FLOWER_CLASSES = {
    0: "daisy",
    1: "dandelion",
    2: "rose",
    3: "sunflower",
    4: "tulip"
}

FLOWER_EMOJIS = {
    "daisy": "🌼",
    "dandelion": "🌻",
    "rose": "🌹",
    "sunflower": "🌻",
    "tulip": "🌷"
}

FLOWER_DESCRIPTIONS = {
    "daisy": "A cheerful white flower with a yellow center, symbolizing innocence and purity.",
    "dandelion": "A bright yellow wildflower known for its iconic seed puffs and resilience.",
    "rose": "The classic symbol of love and beauty, available in countless colors and varieties.",
    "sunflower": "A tall, bold flower that always turns to face the sun, representing warmth and loyalty.",
    "tulip": "An elegant cup-shaped spring bloom that comes in nearly every color of the rainbow."
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_dirs():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    ensure_dirs()

    if 'image' not in request.files:
        return render_template('index.html', error="No file part in the request.")

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    if not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, BMP, WEBP).")

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        unique_filename = f"{timestamp}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(upload_path)

        # Run YOLO inference
        results = model(upload_path)
        result = results[0]

        # Save result image with bounding boxes / overlays
        result_filename = f"result_{unique_filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        result.save(filename=result_path)

        # Parse detection data
        detections = []

        # For classification models
        if hasattr(result, 'probs') and result.probs is not None:
            probs = result.probs
            top5_indices = probs.top5
            top5_conf = probs.top5conf.tolist()

            for idx, conf in zip(top5_indices, top5_conf):
                class_name = result.names[idx]
                detections.append({
                    'label': class_name,
                    'confidence': round(conf * 100, 2),
                    'emoji': FLOWER_EMOJIS.get(class_name.lower(), '🌸'),
                    'description': FLOWER_DESCRIPTIONS.get(class_name.lower(), 'A beautiful flower species.')
                })

        # For detection models (bounding boxes)
        elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[class_id]
                xyxy = box.xyxy[0].tolist()
                detections.append({
                    'label': class_name,
                    'confidence': round(conf * 100, 2),
                    'emoji': FLOWER_EMOJIS.get(class_name.lower(), '🌸'),
                    'description': FLOWER_DESCRIPTIONS.get(class_name.lower(), 'A detected flower species.'),
                    'bbox': [round(x) for x in xyxy]
                })

        if not detections:
            detections = [{
                'label': 'Unknown',
                'confidence': 0,
                'emoji': '❓',
                'description': 'No flower detected with sufficient confidence. Try a clearer image.'
            }]

        top_prediction = detections[0] if detections else None

        return render_template(
            'index.html',
            original_image=upload_path,
            result_image=result_path,
            detections=detections,
            top_prediction=top_prediction,
            filename=filename
        )

    except Exception as e:
        return render_template('index.html', error=f"Detection failed: {str(e)}")


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    ensure_dirs()
    app.run(debug=True)
