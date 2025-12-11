import os
import csv
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from geopy.geocoders import Nominatim

# --------------------------------------------------------
# MODEL LOAD
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
    results = model(np.array(image), conf=conf_thresh)
    # results may be a list/Results; handle first
    res = results[0] if isinstance(results, (list, tuple)) else results
    detections = []
    for box in res.boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        detections.append({"bbox": xyxy, "conf": conf})
    return detections

# --------------------------------------------------------
# TUNED INFERENCE
# --------------------------------------------------------
def run_inference_tuned(image, model, conf_thresh=0.28, iou_thresh=0.5, imgsz=1280, augment=True):
    if model is None:
        return []
    results = model(np.array(image), conf=conf_thresh, iou=iou_thresh, imgsz=imgsz, augment=augment)
    res = results[0] if isinstance(results, (list, tuple)) else results
    detections = []
    for box in res.boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        detections.append({"bbox": xyxy, "conf": conf})
    return detections

# --------------------------------------------------------
# TTA INFERENCE (simple augment inside ultralytics)
# --------------------------------------------------------
def tta_inference(image, model, conf_thresh=0.25, imgsz=1280):
    if model is None:
        return []
    results = model(np.array(image), conf=conf_thresh, imgsz=imgsz, augment=True)
    res = results[0] if isinstance(results, (list, tuple)) else results
    detections = []
    for box in res.boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        detections.append({"bbox": xyxy, "conf": conf})
    return detections

# --------------------------------------------------------
# TILED INFERENCE
# --------------------------------------------------------
def tile_inference(image, model, tile_size=768, overlap=0.25, conf_thresh=0.25, imgsz=1280):
    if model is None:
        return []
    img_w, img_h = image.size
    step = int(tile_size * (1 - overlap))
    detections = []
    for y in range(0, max(1, img_h - tile_size + 1), max(1, step)):
        for x in range(0, max(1, img_w - tile_size + 1), max(1, step)):
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            tile = image.crop((x, y, x2, y2))
            results = model(np.array(tile), conf=conf_thresh, imgsz=imgsz, augment=True)
            res = results[0] if isinstance(results, (list, tuple)) else results
            for box in res.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                # adjust coords to full image
                adjusted = [xyxy[0] + x, xyxy[1] + y, xyxy[2] + x, xyxy[3] + y]
                detections.append({"bbox": adjusted, "conf": conf})
    return detections

# --------------------------------------------------------
# DRAW BOXES
# --------------------------------------------------------
def draw_boxes(image, detections):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((max(0, x1), max(0, y1-12)), f"{conf:.2f}", fill="red")
    return img

# --------------------------------------------------------
# SEVERITY: combined area + count scoring
# --------------------------------------------------------
def compute_severity(image, detections):
    if len(detections) == 0:
        return "none", 0.0

    img_w, img_h = image.size
    img_area = img_w * img_h

    total_area = 0.0
    used_boxes = []

    for det in detections:
        try:
            x1, y1, x2, y2 = map(float, det["bbox"])
        except:
            continue
        # skip invalid
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        # skip hallucination boxes that are too big
        if area > img_area * 0.25:
            continue
        # check duplicate (overlapping/close origin)
        duplicate = False
        for ux1, uy1, ux2, uy2 in used_boxes:
            if abs(x1 - ux1) < 10 and abs(y1 - uy1) < 10:
                duplicate = True
                break
        if not duplicate:
            used_boxes.append((x1, y1, x2, y2))
            total_area += area

    pct = (total_area / img_area) * 100.0
    pct = min(pct, 20.0)  # clamp

    # AREA score (1..3)
    if pct < 3:
        area_score = 1
    elif pct < 8:
        area_score = 2
    else:
        area_score = 3

    # COUNT score (0..3)
    count = len(used_boxes)
    if count == 0:
        count_score = 0
    elif count <= 3:
        count_score = 1
    elif count <= 7:
        count_score = 2
    else:
        count_score = 3

    final_score = max(area_score, count_score)

    if final_score == 0:
        label = "none"
    elif final_score == 1:
        label = "small"
    elif final_score == 2:
        label = "medium"
    else:
        label = "large"

    return label, pct

# --------------------------------------------------------
# SAVE REPORT
# --------------------------------------------------------
def save_report(csv_path, data):
    # write headers based on keys, append if exists
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# --------------------------------------------------------
# REVERSE GEOCODING (city lookup)
# --------------------------------------------------------
geolocator = Nominatim(user_agent="pothole_app")

def get_city_name(lat, lon):
    try:
        latf = float(lat)
        lonf = float(lon)
    except:
        return "Unknown"
    try:
        location = geolocator.reverse(f"{latf}, {lonf}", timeout=10)
        address = location.raw.get("address", {})
        city = address.get("city") or address.get("town") or address.get("village") or address.get("state_district") or address.get("state")
        return city if city else "Unknown"
    except Exception as e:
        # network or rate-limit etc.
        return "Unknown"
