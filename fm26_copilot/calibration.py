from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .contracts import TrackedObject


@dataclass
class PitchCalibration:
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float


class PitchCalibrator:
    def estimate(self, frame_bgr: object) -> PitchCalibration | None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return None

        if frame_bgr is None:
            return None

        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return None

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([35, 30, 30], dtype=np.uint8)
        upper = np.array([95, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        area_ratio = (bw * bh) / float(w * h)
        conf = min(0.98, max(0.3, area_ratio))
        return PitchCalibration(
            x0=x / w,
            y0=y / h,
            x1=(x + bw) / w,
            y1=(y + bh) / h,
            confidence=conf,
        )

    def map_to_pitch(self, objects: list[TrackedObject], calibration: PitchCalibration | None) -> Dict[int, Tuple[float, float]]:
        coordinates: Dict[int, Tuple[float, float]] = {}
        for obj in objects:
            cx = obj.bbox.x + obj.bbox.w * 0.5
            cy = obj.bbox.y + obj.bbox.h * 0.5
            if calibration is None:
                coordinates[obj.object_id] = (cx, cy)
                continue
            nx = (cx - calibration.x0) / max(1e-4, (calibration.x1 - calibration.x0))
            ny = (cy - calibration.y0) / max(1e-4, (calibration.y1 - calibration.y0))
            coordinates[obj.object_id] = (max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny)))
        return coordinates

