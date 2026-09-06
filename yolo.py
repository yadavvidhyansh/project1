"""
main.py
NovaAI Object Detection - Query Engine.
Camera runs in a background thread only when triggered.
Provides a single entry point: get_answer(text) -> returns a text answer string.
No voice/STT/TTS included here — integrate your own voice assistant on top of this.
"""

import os
import re
import cv2
import threading
from yolo_detector import YoloDetector
from direction_helper import get_direction

MODEL_PATH = "yolo11n.onnx"


class QueryEngine:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")

        self.detector = YoloDetector(model_path, conf_threshold=0.25, iou_threshold=0.45)

        self.latest_summary = []
        self.frame_width = 640
        self.frame_height = 480
        self._summary_lock = threading.Lock()

        self._camera_running = threading.Event()
        self._camera_thread = None

    # ---------- Camera control ----------
    def start_camera(self):
        if self._camera_running.is_set():
            return "Object detection is already running"

        self._camera_running.set()
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        return "Starting object detection"

    def stop_camera(self):
        if not self._camera_running.is_set():
            return "Object detection is not running"

        self._camera_running.clear()
        return "Stopping object detection"

    def is_camera_running(self):
        return self._camera_running.is_set()

    def _camera_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not access the webcam.")
            self._camera_running.clear()
            return

        print("Camera opened.")

        while self._camera_running.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            try:
                result = self.detector.detect(frame)
                annotated_frame = self.detector.get_annotated_frame(result)
                summary = self.detector.get_detection_summary(result, detect_color=True)

                with self._summary_lock:
                    self.latest_summary = summary
                    self.frame_width = w
                    self.frame_height = h

            except Exception as e:
                print(f"Detection error: {e}")
                annotated_frame = frame

            cv2.imshow("NovaAI - Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._camera_running.clear()
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")

    # ---------- Main entry point ----------
    def get_answer(self, query_text):
        """
        Pass any recognized text query here (typed, or from your own voice
        recognition). Returns a plain text answer string.

        Supported commands:
          - "start object detection" / "start camera"
          - "stop object detection" / "stop camera"
          - "what is the color of <object>"
          - "find <object>" / "where is <object>"
        """
        if not query_text:
            return "I didn't catch that"

        text = query_text.lower()

        if "start" in text and ("detection" in text or "camera" in text):
            return self.start_camera()

        if "stop" in text and ("detection" in text or "camera" in text):
            return self.stop_camera()

        if "color" in text or "colour" in text:
            return self._answer_color(text)

        if "find" in text or "where is" in text or "where's" in text or "locate" in text:
            return self._answer_find(text)

        return "Sorry, I didn't understand that command"

    # ---------- Internal helpers ----------
    def _get_current_summary(self):
        with self._summary_lock:
            return list(self.latest_summary), self.frame_width, self.frame_height

    def _extract_object_from_text(self, text, known_labels):
        query_words = set(re.findall(r"\w+", text.lower()))
        for label in known_labels:
            label_words = set(re.findall(r"\w+", label.lower()))
            if label_words.issubset(query_words):
                return label
        return None

    def _answer_color(self, text):
        if not self.is_camera_running():
            return "Please start object detection first"

        summary, _, _ = self._get_current_summary()
        if not summary:
            return "I don't see anything right now"

        known_labels = [item["label"] for item in summary]
        matched = self._extract_object_from_text(text, known_labels)

        if not matched:
            return "I couldn't find that object"

        for item in summary:
            if item["label"] == matched:
                return f"The color of the {matched} is {item['color']}"

        return "I couldn't determine the color"

    def _answer_find(self, text):
        if not self.is_camera_running():
            return "Please start object detection first"

        summary, fw, fh = self._get_current_summary()
        if not summary:
            return "I don't see anything right now"

        known_labels = [item["label"] for item in summary]
        matched = self._extract_object_from_text(text, known_labels)

        if not matched:
            return "I cannot find that object right now"

        for item in summary:
            if item["label"] == matched:
                direction = get_direction(item["box"], fw, fh)
                return f"The {matched} is {direction}"

        return "I cannot find that object right now"


# ---------- Console test runner (optional, for local testing only) ----------
def run_console():
    try:
        engine = QueryEngine()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("NovaAI ready. Type commands like:")
    print("  - start object detection")
    print("  - find chair")
    print("  - what is the color of the bottle")
    print("  - stop object detection")
    print("  - exit\n")

    while True:
        user_input = input(">> ").strip()

        if user_input.lower() == "exit":
            if engine.is_camera_running():
                engine.stop_camera()
            print("Exiting.")
            break

        answer = engine.get_answer(user_input)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    run_console()