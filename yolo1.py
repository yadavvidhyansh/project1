print("========================================")
print("NEW YOLO.PY CODE IS RUNNING")
print("========================================")
import os
import re
import cv2
import time
import threading

from yolo_detector import YoloDetector
from direction_helper import get_direction


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"yolo11n.onnx"

CAMERA_INDEX = 0

# YOLO detection happens every 5 seconds
DETECTION_INTERVAL = 5.0

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


# ============================================================
# QUERY ENGINE
# ============================================================

class QueryEngine:

    def __init__(self, model_path=MODEL_PATH):

        print("========================================")
        print("        NovaAI Object Detection")
        print("========================================")

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        print("Checking model...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found:\n{model_path}"
            )

        print("Model found:")
        print(model_path)

        # ----------------------------------------------------
        # Load detector
        # ----------------------------------------------------

        print("Loading YOLO model...")

        self.detector = YoloDetector(
            model_path,
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD
        )

        print("YOLO model loaded successfully.")

        # ----------------------------------------------------
        # Latest detection
        # ----------------------------------------------------

        self.latest_summary = []

        self.frame_width = 640
        self.frame_height = 480

        self.camera_running = False
        self._camera_thread = None
        self._lock = threading.Lock()

    # ========================================================
    # START CAMERA (NON-BLOCKING)
    # ========================================================

    def start_camera(self, speak_callback=None):
        with self._lock:
            if self.camera_running:
                return "Object detection is already running."
            
            self.camera_running = True
            self._camera_thread = threading.Thread(
                target=self._camera_loop, 
                args=(speak_callback,),
                daemon=True
            )
            self._camera_thread.start()
            return "Starting object detection."

    def stop_camera(self):
        with self._lock:
            if not self.camera_running:
                return "Object detection is not running."
            self.camera_running = False
            return "Stopping object detection."

    def _camera_loop(self, speak_callback):
        print()
        print("Opening camera...")

        cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():
            print("ERROR: Camera could not be opened.")
            self.camera_running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Camera opened successfully.")
        last_detection_time = 0

        try:
            while self.camera_running:
                ret, frame = cap.read()
                if not ret:
                    break

                height, width = frame.shape[:2]
                self.frame_width = width
                self.frame_height = height

                current_time = time.time()
                if current_time - last_detection_time >= DETECTION_INTERVAL:
                    try:
                        result = self.detector.detect(frame)
                        summary = self.detector.get_detection_summary(result, detect_color=True)
                        self.latest_summary = summary
                        
                        # AUTOMATED SPEECH
                        if speak_callback and summary:
                            answer = self._create_detection_answer(summary, width, height)
                            speak_callback(answer)

                    except Exception as e:
                        print("Detection error:", e)
                    last_detection_time = time.time()

                # Display logic
                display_frame = frame.copy()
                if last_detection_time == 0:
                    countdown_text = "Detecting..."
                else:
                    remaining = max(0, DETECTION_INTERVAL - (time.time() - last_detection_time))
                    countdown_text = f"Next detection: {remaining:.1f}s"

                cv2.putText(display_frame, countdown_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                y = 65
                for item in self.latest_summary:
                    label = item.get("label", "unknown")
                    color = item.get("color", "unknown")
                    cv2.putText(display_frame, f"{label} - {color}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y += 30

                cv2.imshow("NovaAI - Object Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.camera_running = False
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.camera_running = False
            print("Camera closed.")

    # ========================================================
    # DETECTION ANSWER
    # ========================================================

    def _create_detection_answer(
        self,
        summary,
        width,
        height
    ):

        objects = []

        for item in summary:

            label = item.get(
                "label",
                "unknown"
            )

            color = item.get(
                "color",
                "unknown"
            )

            box = item.get("box")

            try:

                direction = get_direction(
                    box,
                    width,
                    height
                )

            except Exception:

                direction = "unknown"

            if color != "unknown":

                objects.append(
                    f"{color} {label} "
                    f"on the {direction}"
                )

            else:

                objects.append(
                    f"{label} "
                    f"on the {direction}"
                )

        if not objects:

            return "I don't see any objects."

        if len(objects) == 1:

            return f"I see {objects[0]}."

        return (
            "I see "
            + ", ".join(objects[:-1])
            + " and "
            + objects[-1]
            + "."
        )

    # ========================================================
    # FIND OBJECT
    # ========================================================

    def _answer_find(self, text):

        if not self.latest_summary:

            return (
                "I don't have a recent detection. "
                "Please start object detection first."
            )

        known_labels = [
            item.get("label", "")
            for item in self.latest_summary
        ]

        matched = self._extract_object_from_text(
            text,
            known_labels
        )

        if not matched:

            return "I cannot find that object."

        for item in self.latest_summary:

            if item.get("label") == matched:

                box = item.get("box")

                try:

                    direction = get_direction(
                        box,
                        self.frame_width,
                        self.frame_height
                    )

                except Exception:

                    direction = "unknown"

                return (
                    f"The {matched} "
                    f"is on the {direction}."
                )

        return "I cannot find that object."

    # ========================================================
    # COLOR
    # ========================================================

    def _answer_color(self, text):

        if not self.latest_summary:

            return (
                "I don't have a recent detection. "
                "Please start object detection first."
            )

        known_labels = [
            item.get("label", "")
            for item in self.latest_summary
        ]

        matched = self._extract_object_from_text(
            text,
            known_labels
        )

        if not matched:

            return "I couldn't find that object."

        for item in self.latest_summary:

            if item.get("label") == matched:

                color = item.get(
                    "color",
                    "unknown"
                )

                return (
                    f"The color of the "
                    f"{matched} is {color}."
                )

        return "I couldn't determine the color."

    # ========================================================
    # OBJECT NAME EXTRACTION
    # ========================================================

    def _extract_object_from_text(
        self,
        text,
        known_labels
    ):

        query_words = set(
            re.findall(
                r"\w+",
                text.lower()
            )
        )

        for label in known_labels:

            label_words = set(
                re.findall(
                    r"\w+",
                    label.lower()
                )
            )

            if label_words.issubset(query_words):

                return label

        return None

    # ========================================================
    # COMMAND HANDLER
    # ========================================================

    def get_answer(self, query_text):

        if not query_text:

            return "I didn't catch that."

        text = query_text.lower().strip()

        # ----------------------------------------------------
        # START CAMERA
        # ----------------------------------------------------

        if (
            "start object detection" in text
            or "start detection" in text
            or "start camera" in text
            or "detect objects" in text
            or "detect object" in text
            or "scan objects" in text
            or "scan object" in text
            or "look around" in text
            or "what do you see" in text
        ):

            return self.start_camera()

        # ----------------------------------------------------
        # STOP CAMERA
        # ----------------------------------------------------

        if (
            "stop object detection" in text
            or "stop detection" in text
            or "stop camera" in text
        ):

            if self.camera_running:

                return (
                    "Press Q in the camera window "
                    "to stop detection."
                )

            return "Object detection is not running."

        # ----------------------------------------------------
        # COLOR
        # ----------------------------------------------------

        if (
            "color" in text
            or "colour" in text
        ):

            return self._answer_color(text)

        # ----------------------------------------------------
        # FIND
        # ----------------------------------------------------

        if (
            "find" in text
            or "where is" in text
            or "where's" in text
            or "locate" in text
        ):

            return self._answer_find(text)

        return "Sorry, I didn't understand that command."


# ============================================================
# CONSOLE
# ============================================================

def run_console():

    try:

        engine = QueryEngine()

    except Exception as e:

        print()
        print("ERROR:")
        print(e)
        print()

        return

    print()
    print("NovaAI ready.")
    print()
    print("Type:")
    print("  detect objects")
    print("  find chair")
    print("  where is the bottle")
    print("  what is the color of the bottle")
    print("  exit")
    print()

    while True:

        try:

            user_input = input(">> ").strip()

        except KeyboardInterrupt:

            print()
            print("Program stopped.")

            break

        if user_input.lower() == "exit":

            if engine.camera_running:

                print(
                    "Press Q in the camera window "
                    "to close the camera."
                )

            print("Exiting NovaAI.")

            break

        if not user_input:

            continue

        answer = engine.get_answer(
            user_input
        )

        print()
        print("Answer:", answer)
        print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_console()