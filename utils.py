import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd

# ============================================
#  MODEL LOADING
# ============================================

def load_model(model_path):
    try:
        from ultralytics import YOLO
        return YOLO(model_path)
    except Exception as e:
        print("Model load failed:", e)
        return None


# ============================================
#  DEFAULT INFERENCE (WHAT YOU HAD BEFORE)
# ============================================

def run_inference(image, model, conf_thresh=0.4):
    """Basic inference function used earlier."""
    if model is None:
        return dummy_detect(image)

    try:
        arr = np.array(image)
        results = model.predict(arr, conf=conf_thresh)
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
        print("Default inference error:", e)
        return dummy_detect(image)


# ============================================
#  TUNED INFERENCE (HIGHER ACCURACY)
# ============================================

def run_inference_tuned(
    image,
    model,
    conf_thresh=0.30,
    iou_thresh=0.5,
    imgsz=1280,
    augment=True
):
    """Higher resolution + lower conf + ultralytics TTA augment."""
    if model is None:
        return dummy_detect(image)

    try:
        arr = np.array(image)
        results = model.predict(
            arr, conf=conf_thresh, iou=iou_thresh,
            imgsz=imgsz, augment=augment
        )

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
        print("run_inference_tuned error:", e)
        return dummy_detect(image)


# ============================================
#  NMS UTILITY (USED BY TTA + TILING)
# ============================================

def non_max_suppression_numpy(boxes, iou_threshold=0.45, score_threshold=0.25):
    """
    boxes = list of [x1, y1, x2, y2, score]
    """
    if len(boxes) == 0:
        return []

    boxes_arr = np.array(boxes)
    idxs = np.argsort(boxes_arr[:, 4])[::-1]  # sort by confidence desc
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(boxes_arr[i].tolist())

        if len(idxs) == 1:
            break

        rest = idxs[1:]
        new_idxs = []

        for j in rest:
            # compute IoU
            boxA = boxes_arr[i]
            boxB = boxes_arr[j]

            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            inter = max(0, xB - xA) * max(0, yB - yA)
            areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

            iou = inter / (areaA + areaB - inter + 1e-6)

            if iou <= iou_threshold:
                new_idxs.append(j)

        idxs = np.array(new_idxs)

    # filter out low scores
    return [b for b in keep if b[4] >= score_threshold]


# ============================================
#  TTA INFERENCE (FLIP-BASED)
# ============================================

def tta_inference(image, model, conf_thresh=0.25, iou_thresh=0.45, imgsz=1280):
    """Runs inference on image + H-flip + V-flip. Merges via NMS."""
    if model is None:
        return dummy_detect(image)

    variants = [
        image,
        image.transpose(Image.FLIP_LEFT_RIGHT),
        image.transpose(Image.FLIP_TOP_BOTTOM)
    ]

    all_boxes = []

    for idx, im in enumerate(variants):
        arr = np.array(im)
        results = model.predict(arr, conf=conf_thresh, imgsz=imgsz)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                # unflip
                if idx == 1:  # H-flip
                    x1, x2 = im.width - x2, im.width - x1
                if idx == 2:  # V-flip
                    y1, y2 = im.height - y2, im.height - y1

                all_boxes.append([x1, y1, x2, y2, conf])

    # merge boxes
    kept = non_max_suppression_numpy(all_boxes, iou_threshold=iou_thresh, score_threshold=conf_thresh)

    return [
        {"box": (int(x1), int(y1), int(x2), int(y2)), "conf": float(s)}
        for x1, y1, x2, y2, s in kept
    ]


# ============================================
#  TILED INFERENCE (BEST FOR SMALL POTHOLES)
# ============================================

def tile_inference(image, model, tile_size=768, overlap=0.25,
                   conf_thresh=0.25, iou_thresh=0.45, imgsz=1280):

    if model is None:
        return dummy_detect(image)

    w, h = image.size
    stride = int(tile_size * (1 - overlap))
    boxes_all = []

    for y in range(0, max(1, h - tile_size + 1), max(1, stride)):
        for x in range(0, max(1, w - tile_size + 1), max(1, stride)):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)

            crop = image.crop((x, y, x2, y2))
            arr = np.array(crop)

            results = model.predict(arr, conf=conf_thresh, imgsz=imgsz)

            for r in results:
                for b in r.boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    conf = float(b.conf[0])

                    boxes_all.append([
                        x + bx1, y + by1, x + bx2, y + by2, conf
                    ])

    kept = non_max_suppression_numpy(
        boxes_all, iou_threshold=iou_thresh, score_threshold=conf_thresh
    )

    return [
        {"box": (int(x1), int(y1), int(x2), int(y2)), "conf": float(s)}
        for x1, y1, x2, y2, s in kept
    ]


# ============================================
#  DRAW BOXES
# ============================================

def draw_boxes(image, detections):
    im = image.copy()
    draw = ImageDraw.Draw(im)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"{conf:.2f}", fill="red")

    return im


# ============================================
#  SEVERITY CALCULATION
# ============================================

def compute_severity(image, detections):
    """Simple severity metric based on area of potholes."""
    if len(detections) == 0:
        return "none", 0

    im_area = image.width * image.height
    areas = []

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        areas.append((x2 - x1) * (y2 - y1))

    pct = (sum(areas) / im_area) * 100

    if pct < 1:
        label = "small"
    elif pct < 3:
        label = "medium"
    else:
        label = "large"

    return label, pct


# ============================================
#  SAVE REPORT
# ============================================

def save_report(csv_path, entry):
    # Ensure dir exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    df.to_csv(csv_path, index=False)


# ============================================
#  FALLBACK DUMMY DETECTION
# ============================================

def dummy_detect(image):
    """Fallback when model is missing."""
    w, h = image.size
    return [{
        "box": (w//4, h//4, w//2, h//2),
        "conf": 0.5
    }]
