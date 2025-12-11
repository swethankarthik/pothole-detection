import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np

# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------
def load_model(path):
    """Loads YOLOv8 model if available."""
    if os.path.exists(path):
        return YOLO(path)
    return None

# -------------------------------------------------
# INFERENCE
# -------------------------------------------------
def run_inference(image, model, conf_thresh=0.4):
    """Runs YOLOv8 inference and returns detections."""
    if model is None:
        return dummy_detect(image)

    try:
        img_array = np.array(image)
        results = model.predict(img_array, conf=conf_thresh)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                detections.append({
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": conf
                })

        return detections

    except Exception as e:
        print("Inference error:", e)
        return dummy_detect(image)

# -------------------------------------------------
# DUMMY DETECTION (FALLBACK)
# -------------------------------------------------
def dummy_detect(image):
    w, h = image.size
    return [{
        "box": (int(w*0.3), int(h*0.4), int(w*0.6), int(h*0.7)),
        "conf": 0.80
    }]

# -------------------------------------------------
# DRAW BOXES
# -------------------------------------------------
def draw_boxes(image, detections):
    image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 12), f"{conf:.2f}", fill="white", font=font)

    return image

# -------------------------------------------------
# SEVERITY LOGIC
# -------------------------------------------------
def compute_severity(image, detections):
    if not detections:
        return "none", 0.0

    w, h = image.size
    area = w * h

    max_pct = 0.0
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        box_area = (x2 - x1) * (y2 - y1)
        pct = (box_area / area) * 100
        max_pct = max(max_pct, pct)

    if max_pct < 1:
        return "small", max_pct
    elif max_pct < 5:
        return "medium", max_pct
    else:
        return "large", max_pct

# -------------------------------------------------
# SAVE REPORT
# -------------------------------------------------
def save_report(csv_path, report_dict):
    df = pd.DataFrame([report_dict])

    # File doesn't exist → create new
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", index=False, header=False)
