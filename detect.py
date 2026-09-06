import os
import cv2
import time
import threading
from ultralytics import YOLO

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "runs/detect/train/weights/best.pt"
CONFIDENCE = 0.65
ALERT_DELAY = 3

# =========================================================
# GLOBAL VARIABLES
# =========================================================

model = None
running = False
detection_thread = None
last_alert_time = 0

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise Exception(f"Pothole model not found at {MODEL_PATH}")
        print("Loading pothole model...")
        model = YOLO(MODEL_PATH)
        print("Pothole model loaded.")
    return model

def pothole_camera(speak_function):
    global running
    global last_alert_time

    try:
        detector = load_model()
    except Exception as e:
        print(f"Pothole Model Error: {e}")
        speak_function("Pothole model file not found. Please check your weights folder.")
        running = False
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak_function("Could not access the camera for road safety.")
        running = False
        return

    speak_function("Pothole detection and road safety mode active.")

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        roi_start = int(height * 0.45)

        results = detector.predict(frame, conf=CONFIDENCE, verbose=False)
        detected = False
        detected_position = None
        highest_confidence = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if center_y < roi_start:
                    continue

                detected = True
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    if center_x < width * 0.33:
                        detected_position = "left"
                    elif center_x > width * 0.66:
                        detected_position = "right"
                    else:
                        detected_position = "center"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(frame, f"Pothole {confidence:.2f}", (x1, max(y1 - 10, 25)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        current_time = time.time()
        if detected and (current_time - last_alert_time >= ALERT_DELAY):
            if detected_position == "left":
                speak_function("Warning. Pothole on the left.")
            elif detected_position == "right":
                speak_function("Warning. Pothole on the right.")
            else:
                speak_function("Warning. Pothole ahead.")
            last_alert_time = current_time

        status_text = "POTHOLE DETECTED" if detected else "ROAD CLEAR"
        color = (0, 0, 255) if detected else (0, 255, 0)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow("Nova - Road Safety Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()
    running = False

def start_pothole_detection(speak_function):
    global running
    global detection_thread

    if running:
        speak_function("Road safety mode is already running.")
        return

    running = True
    detection_thread = threading.Thread(target=pothole_camera, args=(speak_function,), daemon=True)
    detection_thread.start()

def stop_pothole_detection(speak_function):
    global running
    if not running:
        speak_function("Road safety mode is not active.")
        return
    running = False
    speak_function("Stopping road safety mode.")
