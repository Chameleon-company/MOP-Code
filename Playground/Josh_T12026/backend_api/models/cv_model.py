import os
from ultralytics import YOLO
import cv2
import numpy as np
import base64

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# load YOLO model ONCE
model = YOLO(MODEL_PATH)

# -----------------------------
# preprocessing functions
# -----------------------------

def brighten_image(image, alpha=1.5, beta=30):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    l_clahe = clahe.apply(l)

    merged = cv2.merge((l_clahe, a, b))

    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def brighten_then_clahe(image):
    bright = brighten_image(image)
    return apply_clahe(bright)

# temporary substitute if missing
def double_check_preprocess(image):
    return brighten_then_clahe(image)

# -----------------------------
# classification
# -----------------------------

def classify_streetlight_state(
    crop,
    off_threshold=80,
    dim_threshold=160
):

    if crop is None or crop.size == 0:
        return "unknown", 0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    pixels = gray.flatten()
    pixels = np.sort(pixels)

    top_pixels = pixels[int(0.9 * len(pixels)):]

    mean_brightness = float(np.mean(top_pixels))

    if mean_brightness < off_threshold:
        state = "off"

    elif mean_brightness < dim_threshold:
        state = "dim"

    else:
        state = "on"

    return state, mean_brightness


# -----------------------------
# MAIN ANALYSIS FUNCTION
# -----------------------------

def analyse_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return {
            "error": "Could not load image"
        }

    processed_img = double_check_preprocess(image)

    results = model(
        processed_img,
        conf=0.10,
        verbose=False
    )

    boxes = results[0].boxes
    
    if boxes is None or len(boxes) == 0: 
        return {
            "streetlight_count": 0, 
            "on": 0,
            "dim": 0,
            "off": 0,
            "details": []
        }

    on_count = 0 
    dim_count = 0 
    off_count = 0 

    details = []

    for box in boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        crop = image[y1:y2, x1:x2]

        state, brightness = classify_streetlight_state(crop)

        if state == "on":
            on_count += 1

        elif state == "dim":
            dim_count += 1

        elif state == "off":
            off_count += 1

        details.append({
            "bbox": [x1, y1, x2, y2],
            "state": state,
            "brightness": brightness
        })

    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(
            img_file.read()
        ).decode("utf-8")

    return {
        "uploaded_img": encoded_image,
        "streetlight_count": len(boxes),
        "on": on_count,
        "dim": dim_count,
        "off": off_count,
        "details": details
    }