import cv2
import numpy as np

def extract_crack_features(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    output_img = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold detection
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

    # Strengthen cracks
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_length = 0
    max_width = 0
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 100:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w == 0 or h == 0:
            continue

        aspect_ratio = max(w, h) / (min(w, h) + 1)

        if aspect_ratio < 2:
            continue

        valid_contours.append(cnt)
        total_length += cv2.arcLength(cnt, False)
        max_width = max(max_width, w, h)

    cv2.drawContours(output_img, valid_contours, -1, (0, 0, 255), 2)

    crack_length = round(total_length / 100, 2)
    crack_width = round(max_width / 10, 2)

    if crack_width > 5:
        severity = "High"
    elif crack_width > 2:
        severity = "Medium"
    else:
        severity = "Low"

    return {
        "crack_length": crack_length,
        "crack_width": crack_width,
        "location": "surface",
        "severity": severity,
        "processed_image": output_img
    }