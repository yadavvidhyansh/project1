"""
yolo_detector.py
Core detection engine — loads ONNX model, runs inference, and detects dominant color
of each detected object using HSV color space.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path="yolo11n.onnx", conf_threshold=0.25, iou_threshold=0.45):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(model_path, task="detect")

    def detect(self, image):
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        return results[0]

    def get_detection_summary(self, result, detect_color=True):
        summary = []
        names = result.names
        original_frame = result.orig_img

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            item = {
                "label": names[class_id],
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
            }

            if detect_color:
                item["color"] = self._get_dominant_color(original_frame, x1, y1, x2, y2)

            summary.append(item)

        return summary

    def get_annotated_frame(self, result):
        return result.plot()

    def _get_dominant_color(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            return "unknown"

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "unknown"

        crop_h, crop_w = roi.shape[:2]
        pad_h, pad_w = int(crop_h * 0.30), int(crop_w * 0.30)
        center_roi = roi[pad_h:crop_h - pad_h, pad_w:crop_w - pad_w]
        if center_roi.size == 0:
            center_roi = roi

        hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        v_channel = pixels[:, 2]
        mask = (v_channel > 40) & (v_channel < 250)
        filtered = pixels[mask]

        if len(filtered) < 10:
            filtered = pixels

        k = min(3, len(filtered))
        if k < 1:
            return "unknown"

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(filtered, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

        counts = np.bincount(labels.flatten())
        dominant_hsv = centers[np.argmax(counts)]

        return self._hsv_to_color_name(dominant_hsv)

    def _hsv_to_color_name(self, hsv):
        h, s, v = hsv

        if v < 50:
            return "black"
        if s < 40:
            return "white" if v > 200 else "gray"

        if h < 5 or h >= 175:
            return "red"
        elif h < 22:
            return "orange"
        elif h < 33:
            return "yellow"
        elif h < 78:
            return "green"
        elif h < 100:
            return "cyan"
        elif h < 130:
            return "blue"
        elif h < 150:
            return "purple"
        elif h < 175:
            return "pink"

        return "unknown"