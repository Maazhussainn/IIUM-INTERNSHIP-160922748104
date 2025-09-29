# Q3. Face Detection and Feature Localization (Colab-Ready)

import cv2
import mediapipe as mp
import os
import time
import matplotlib.pyplot as plt

# ===============================
# 🔹 Set Base Path for Google Drive
# ===============================
BASE_DIR = "/content/drive/MyDrive/Assignment internship"
RESULT_DIR = os.path.join(BASE_DIR, "result/Q3")
os.makedirs(RESULT_DIR, exist_ok=True)

# Landmark indexes from MediaPipe Face Mesh
LEFT_EYE_IDX = 468
RIGHT_EYE_IDX = 473
NOSE_TIP_IDX = 1

def get_landmarks(img, face_landmarks):
    """Convert normalized coordinates (0–1) to pixel coordinates."""
    h, w, _ = img.shape

    def to_px(idx):
        pt = face_landmarks.landmark[idx]
        return int(pt.x * w), int(pt.y * h)

    return {
        "left_eye": to_px(LEFT_EYE_IDX),
        "right_eye": to_px(RIGHT_EYE_IDX),
        "nose_tip": to_px(NOSE_TIP_IDX),
    }


def draw_landmarks(img, lm):
    """Draw circles, lines, and labels for eyes and nose."""

    # Left eye
    cv2.circle(img, lm["left_eye"], 8, (0, 255, 0), -1)
    cv2.putText(img, "Left Eye", (lm["left_eye"][0] - 70, lm["left_eye"][1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Right eye
    cv2.circle(img, lm["right_eye"], 8, (0, 255, 0), -1)
    cv2.putText(img, "Right Eye", (lm["right_eye"][0] + 20, lm["right_eye"][1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Nose tip
    cv2.circle(img, lm["nose_tip"], 8, (0, 0, 255), -1)
    cv2.putText(img, "Nose", (lm["nose_tip"][0] - 30, lm["nose_tip"][1] + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Lines
    cv2.line(img, lm["left_eye"], lm["right_eye"], (255, 255, 0), 2)
    cv2.line(img, lm["left_eye"], lm["nose_tip"], (255, 255, 0), 2)
    cv2.line(img, lm["right_eye"], lm["nose_tip"], (255, 255, 0), 2)


def process_image(image_path):
    """Detect face and landmarks in an image."""
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Could not read image:", image_path)
        return

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        print("[INFO] No face detected in image.")
        return

    # Process first face only
    landmarks = get_landmarks(img, results.multi_face_landmarks[0])
    draw_landmarks(img, landmarks)

    # Save annotated image
    base_name = os.path.basename(image_path)
    out_path = os.path.join(RESULT_DIR, os.path.splitext(base_name)[0] + "_annotated.jpg")
    cv2.imwrite(out_path, img)

    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("Annotated Image")
    plt.show()

    print(f"Landmarks detected in {image_path}:")
    print(" Left eye:", landmarks["left_eye"])
    print(" Right eye:", landmarks["right_eye"])
    print(" Nose tip:", landmarks["nose_tip"])
    print(" Faces detected:", len(results.multi_face_landmarks))
    print(f" Saved annotated image at: {out_path}")


def process_webcam():
    """Detect face and landmarks in real-time webcam with FPS + face count."""
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    prev_time = 0

    with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            display = frame.copy()

            face_count = 0
            if results.multi_face_landmarks:
                face_count = len(results.multi_face_landmarks)
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = get_landmarks(frame, face_landmarks)
                    draw_landmarks(display, landmarks)

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            # Show FPS and face count
            cv2.putText(display, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, f"Faces: {face_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Webcam - Press 'q' to quit", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# ===============================
# 🔹 Choose Mode (Colab-Friendly)
# ===============================
MODE = "image"   # change to "webcam" if you want webcam mode
IMAGE_PATH = "/content/drive/MyDrive/Assignment internship/data/sample_face.jpg"  # <-- upload your test image here

if MODE == "image":
    process_image(IMAGE_PATH)
elif MODE == "webcam":
    process_webcam()
else:
    print("Invalid MODE. Choose 'image' or 'webcam'.")
