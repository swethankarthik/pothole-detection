import os
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from geopy.geocoders import Nominatim

# --------------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------------
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print("Model load error:", e)
        return None


# --------------------------------------------------------
# DEFAULT INFERENCE
# --------------------------------------------------------
def run_inference(image, model, conf_thresh=0.4):
    if model is None:
        return []

    results = model(image, conf=conf_thresh)[0]

    detections = []
    for box in results.boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        detections.append({
            "bbox": xyxy,     # [x1, y1, x2, y2]
            "conf": conf
        })
    return detections


# --------------------------------------------------------
# TUNED INFERENCE (Better accuracy)
# --------------------------------------------------------
def run_inference_tuned(image, model, conf_thresh=0.28, imgsz=1280, augment=True):
    if model is None:
        return []

    results = model(
        image,
        conf=conf_thresh,
        imgsz=imgsz,
        augment=augment
    )[0]

    detections = []
    for box in results.boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        detections.append({
            "bbox": xyxy,
            "conf": conf
        })
    return detections


# --------------------------------------------------------
# TEST TIME AUGMENTATION (TTA)
# --------------------------------------------------------
def tta_inference(image, model, conf_thresh=0.25, imgsz=1280):
    if model is None:
        return []

    results = model(
        image,
        conf=conf_thresh,
        imgsz=imgsz,
        augment=True
    )[0]

    detections = []
    for box in results.boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        detections.append({
            "bbox": xyxy,
            "conf": conf
        })
    return detections


# --------------------------------------------------------
# TILED INFERENCE (Best accuracy for big potholes)
# --------------------------------------------------------
def tile_inference(image, model, tile_size=768, overlap=0.25, conf_thresh=0.25, imgsz=1280):
    if model is None:
        return []

    img_w, img_h = image.size
    step = int(tile_size * (1 - overlap))
    detections = []

    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            tile = image.crop((x, y, x + tile_size, y + tile_size))

            results = model(
                tile,
                conf=conf_thresh,
                imgsz=imgsz,
                augment=True
            )[0]

            for box in results.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                # Adjust box back to full image coordinates
                adjusted = [
                    xyxy[0] + x,
                    xyxy[1] + y,
                    xyxy[2] + x,
                    xyxy[3] + y
                ]

                detections.append({
                    "bbox": adjusted,
                    "conf": conf
                })

    return detections


# --------------------------------------------------------
# DRAW BOXES ON IMAGE
# --------------------------------------------------------
def draw_boxes(image, detections):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"{conf:.2f}", fill="white")

    return img


# --------------------------------------------------------
# SEVERITY CALCULATION
# --------------------------------------------------------
def compute_severity(image, detections):
    if len(detections) == 0:
        return "none", 0

    img_w, img_h = image.size
    img_area = img_w * img_h

    total_area = 0
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        area = (x2 - x1) * (y2 - y1)
        total_area += area

    pct = (total_area / img_area) * 100

    if pct < 3:
        label = "small"
    elif pct < 8:
        label = "medium"
    else:
        label = "large"

    return label, pct


# --------------------------------------------------------
# SAVE REPORT TO CSV
# --------------------------------------------------------
def save_report(csv_path, data):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


# --------------------------------------------------------
# REVERSE GEOCODING → GET CITY NAME
# --------------------------------------------------------
geolocator = Nominatim(user_agent="pothole_app")

def get_city_name(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)

        location = geolocator.reverse(f"{lat}, {lon}", timeout=10)
        addr = location.raw.get("address", {})

        city = (
            addr.get("city") or
            addr.get("town") or
            addr.get("village") or
            addr.get("state_district") or
            addr.get("state")
        )

        return city if city else "Unknown"
    except:
        return "Unknown"
