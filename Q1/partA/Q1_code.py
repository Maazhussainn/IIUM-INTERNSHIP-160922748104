# Q1: License Plate Character Break Detection in Colab
import cv2
import os
import pandas as pd
from ultralytics import YOLO
import numpy as np

# -----------------------------
# Paths
front_folder = "/content/drive/MyDrive/Assignment internship/Q1/partA/data/front"
rear_folder = "/content/drive/MyDrive/Assignment internship/Q1/partA/data/rear"
output_folder = "/content/drive/MyDrive/Assignment internship/Q1/partA/output"
model_path = "/content/drive/MyDrive/Assignment internship/Q1/partA/LP-detection.pt"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Load model
model = YOLO(model_path)

# -----------------------------
# Functions
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def detect_license_plate(img):
    results = model.predict(img, conf=0.25, iou=0.5, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            ar = w / h
            if 2 <= ar <= 6 and w > 50 and h > 15:
                boxes.append((x1, y1, x2, y2))
    return boxes

def detect_broken_characters(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    broken = 0
    annotated = plate_img.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if 20 < area < 500 and 5 < w < 50 and 10 < h < 60:
            broken += 1
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return broken, annotated

def crop_plate(img, bbox):
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2]

def stitch_images(front_img, rear_img):
    h = min(front_img.shape[0], rear_img.shape[0])
    f_resized = cv2.resize(front_img, (int(front_img.shape[1] * h / front_img.shape[0]), h))
    r_resized = cv2.resize(rear_img, (int(rear_img.shape[1] * h / rear_img.shape[0]), h))
    return cv2.hconcat([f_resized, r_resized])

# -----------------------------
# Main processing
report = []
front_files = sorted([f for f in os.listdir(front_folder) if f.endswith(('.jpg', '.png'))])
rear_files = sorted([f for f in os.listdir(rear_folder) if f.endswith(('.jpg', '.png'))])

for f_file, r_file in zip(front_files, rear_files):
    front_img = cv2.imread(os.path.join(front_folder, f_file))
    rear_img = cv2.imread(os.path.join(rear_folder, r_file))

    f_proc = preprocess_image(front_img)
    r_proc = preprocess_image(rear_img)

    f_boxes = detect_license_plate(f_proc)
    r_boxes = detect_license_plate(r_proc)

    broken_front, broken_rear = 0, 0

    # Front plate
    if f_boxes:
        plate = crop_plate(front_img, f_boxes[0])
        broken_front, _ = detect_broken_characters(plate)
    else:
        h, w, _ = front_img.shape
        x1, y1, x2, y2 = int(w*0.3), int(h*0.7), int(w*0.7), int(h*0.85)
        plate = front_img[y1:y2, x1:x2]
        broken_front, _ = detect_broken_characters(plate)

    # Rear plate
    if r_boxes:
        plate = crop_plate(rear_img, r_boxes[0])
        broken_rear, _ = detect_broken_characters(plate)
    else:
        h, w, _ = rear_img.shape
        x1, y1, x2, y2 = int(w*0.3), int(h*0.7), int(w*0.7), int(h*0.85)
        plate = rear_img[y1:y2, x1:x2]
        broken_rear, _ = detect_broken_characters(plate)

    if broken_front > 0 or broken_rear > 0:
        stitched = stitch_images(front_img, rear_img)
        out_path = os.path.join(output_folder, f"broken_{f_file}")
        cv2.imwrite(out_path, stitched)
        report.append({
            "Car Image": f_file,
            "Broken Front": "Yes" if broken_front else "No",
            "Broken Rear": "Yes" if broken_rear else "No"
        })
        print(f"Saved: {out_path}")

# Save CSV
import pandas as pd
df = pd.DataFrame(report)
csv_path = os.path.join(output_folder, "broken_license_plate_report.csv")
df.to_csv(csv_path, index=False)
print("Report saved to:", csv_path)
