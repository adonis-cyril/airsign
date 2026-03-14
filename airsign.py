"""AirSign — Draw your signature in the air using hand tracking."""

import os
import sys
import cv2
import numpy as np

# MediaPipe Tasks API (0.10+)
import mediapipe as mp
from mediapipe.tasks.python import vision

# Camera
CAM_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Zima Blue colors (BGR)
LINE_COLOR = (255, 150, 0)       # Zima Blue - bright
GLOW_COLOR = (180, 100, 0)      # Zima Blue - dimmer for glow
LINE_THICKNESS = 3
GLOW_THICKNESS = 8

# Smoothing
SMOOTHING_WINDOW = 5

# MediaPipe hand landmark indices (same as legacy API)
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18

# Hand landmarker model (downloaded on first run if missing)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_model():
    """Download hand_landmarker.task if not present."""
    if os.path.isfile(MODEL_PATH):
        return MODEL_PATH
    print("Downloading hand landmarker model (one-time)...")
    try:
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model saved to", MODEL_PATH)
        return MODEL_PATH
    except Exception as e:
        print("Could not download model:", e, file=sys.stderr)
        print("Download manually from:", MODEL_URL, file=sys.stderr)
        print("Save as:", MODEL_PATH, file=sys.stderr)
        sys.exit(1)


def is_finger_extended(landmarks, tip_idx, pip_idx):
    """Check if a finger is extended (tip above PIP in image coords; y is down)."""
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    return tip.y < pip.y


def is_drawing_mode(landmarks):
    """Return True when only the index finger is extended (pointing gesture)."""
    index_up = is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP)
    middle_up = is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring_up = is_finger_extended(landmarks, RING_TIP, RING_PIP)
    pinky_up = is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP)
    return index_up and not middle_up and not ring_up and not pinky_up


def smooth_point(raw_points):
    """Return a smoothed point from the last few raw points."""
    window = raw_points[-SMOOTHING_WINDOW:]
    x = int(np.mean([p[0] for p in window]))
    y = int(np.mean([p[1] for p in window]))
    return (x, y)


def draw_strokes(canvas, strokes):
    """Render all strokes onto the canvas with a glow effect."""
    for stroke in strokes:
        if len(stroke) > 1:
            pts = np.array(stroke, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, GLOW_COLOR, GLOW_THICKNESS, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], False, LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)


def main():
    ensure_model()

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        print("On macOS, grant camera access in System Settings > Privacy & Security > Camera.")
        return

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    landmarker = HandLandmarker.create_from_options(options)

    strokes = []
    current_stroke = []
    raw_points = []
    was_drawing = False
    frame_timestamp_ms = 0

    print("AirSign running. Controls:")
    print("  Point index finger → draw")
    print("  Open hand → pause")
    print("  'c' → clear canvas")
    print("  'q' → quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Tasks expect RGB numpy (height, width, 3) uint8
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += int(1000 / 30)  # assume ~30 fps

        drawing = False
        hand_landmarks = None

        if result.hand_landmarks:
            # First hand: list of NormalizedLandmark (x, y, z in [0,1] for x,y)
            hand_landmarks = result.hand_landmarks[0]
            if is_drawing_mode(hand_landmarks):
                drawing = True
                ix = int(hand_landmarks[INDEX_TIP].x * w)
                iy = int(hand_landmarks[INDEX_TIP].y * h)
                raw_points.append((ix, iy))
                pt = smooth_point(raw_points)
                current_stroke.append(pt)

        if was_drawing and not drawing and current_stroke:
            strokes.append(current_stroke)
            current_stroke = []
            raw_points = []

        was_drawing = drawing

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        all_strokes = strokes + ([current_stroke] if current_stroke else [])
        draw_strokes(canvas, all_strokes)
        frame = cv2.add(frame, canvas)

        if drawing:
            status, color = "DRAWING", (0, 255, 0)
        elif hand_landmarks:
            status, color = "PAUSED", (128, 128, 128)
        else:
            status, color = "", (80, 80, 80)

        if status:
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2, cv2.LINE_AA)

        cv2.imshow("AirSign", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            strokes.clear()
            current_stroke.clear()
            raw_points.clear()

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
