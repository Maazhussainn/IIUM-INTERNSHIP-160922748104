# Q4 - Face Blurring in Video (Colab / Script)

import cv2
import os
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from IPython.display import Video

BASE_DIR = "/content/drive/MyDrive/Assignment internship"
MODEL_PATH = os.path.join(BASE_DIR, "Q4", "yolov8m-face.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "data", "sample2_video.mp4")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "result", "Q4", "output_blurred_video.mp4")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

# ---- Check files ----
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"[ERROR] Model not found at {MODEL_PATH}")
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"[ERROR] Video not found at {VIDEO_PATH}")

print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("[INFO] Initializing Mediapipe FaceMesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    cap.get(cv2.CAP_PROP_FPS),
    (int(cap.get(3)), int(cap.get(4)))
)

print("\n[INFO] Processing video... Please wait.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    results = model(frame, conf=0.4, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            results_mesh = face_mesh.process(rgb_face)

            if results_mesh.multi_face_landmarks:
                for landmarks in results_mesh.multi_face_landmarks:
                    points = []
                    for lm in landmarks.landmark:
                        px, py = int(lm.x * (x2 - x1)) + x1, int(lm.y * (y2 - y1)) + y1
                        points.append([px, py])

                    hull = cv2.convexHull(np.array(points, dtype=np.int32))
                    cv2.fillConvexPoly(mask, hull, 255)

    if mask.sum() > 0:
        blurred = cv2.GaussianBlur(frame, (99, 99), 30)
        face_only = cv2.bitwise_and(blurred, blurred, mask=mask)
        background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        output = cv2.add(background, face_only)
    else:
        output = frame

    out.write(output)

cap.release()
out.release()

print(f"\n[INFO] Done! Blurred video saved at: {OUTPUT_VIDEO}")

# Play in notebook
Video(OUTPUT_VIDEO, embed=True, width=640, height=480)
