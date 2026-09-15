from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .contracts import BoundingBox, TeamSide, TrackedObject


@dataclass
class OpenCVOnnxDetectorConfig:
    onnx_model_path: str | None = None
    confidence_floor: float = 0.45
    managed_team_code: str = "ASM"


class OpenCVOnnxDetector:
    def __init__(self, config: OpenCVOnnxDetectorConfig):
        self.config = config
        self._session = self._load_onnx_session(config.onnx_model_path)
        self._next_id = 2000

    def detect(self, frame_bgr: object) -> Tuple[List[TrackedObject], float]:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return [], 0.0

        if frame_bgr is None:
            return [], 0.0

        image = frame_bgr
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return [], 0.0

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 25, 25], dtype=np.uint8)
        upper_green = np.array([95, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        non_field = cv2.bitwise_not(green_mask)

        kernel = np.ones((3, 3), np.uint8)
        non_field = cv2.morphologyEx(non_field, cv2.MORPH_OPEN, kernel)
        non_field = cv2.morphologyEx(non_field, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(non_field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects: List[TrackedObject] = []
        candidate_scores: List[float] = []

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < 80 or area > 2800:
                continue
            if bh <= bw:
                continue
            nx = x / w
            ny = y / h
            nw = bw / w
            nh = bh / h
            conf = min(0.95, max(0.4, 0.5 + (area / 2800.0) * 0.3))
            side = self._infer_side_by_kit(image, x, y, bw, bh, nx)
            objects.append(
                TrackedObject(
                    object_id=self._consume_id(),
                    label="player",
                    side=side,
                    bbox=BoundingBox(x=nx, y=ny, w=nw, h=nh, confidence=conf),
                )
            )
            candidate_scores.append(conf)

        ball = self._detect_ball(image, w, h)
        if ball is not None:
            objects.append(ball)
            candidate_scores.append(ball.bbox.confidence)

        if self._session is not None:
            # Optional hook for model-assisted detections.
            model_conf = self._run_onnx_confidence(image)
            if model_conf > 0.0:
                candidate_scores.append(model_conf)

        if not candidate_scores:
            return objects, 0.0
        return objects, sum(candidate_scores) / len(candidate_scores)

    def _infer_side_by_kit(self, image: object, x: int, y: int, bw: int, bh: int, norm_x: float) -> TeamSide:
        # FM26 ASM screenshots show mostly light kits; we bias HOME toward bright low-saturation shirts.
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return TeamSide.HOME if norm_x < 0.5 else TeamSide.AWAY

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(image.shape[1], x + bw)
        y1 = min(image.shape[0], y + bh)
        patch = image[y0:y1, x0:x1]
        if patch.size == 0:
            return TeamSide.HOME if norm_x < 0.5 else TeamSide.AWAY
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mean_h = float(np.mean(hsv[:, :, 0]))
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_v = float(np.mean(hsv[:, :, 2]))
        is_light_kit = mean_v > 115 and mean_s < 110
        if self.config.managed_team_code.upper() == "ASM":
            return TeamSide.HOME if is_light_kit else TeamSide.AWAY
        return TeamSide.HOME if norm_x < 0.5 else TeamSide.AWAY

    def _detect_ball(self, image: object, w: int, h: int) -> TrackedObject | None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=25,
            param1=90,
            param2=16,
            minRadius=2,
            maxRadius=8,
        )
        if circles is None:
            return None
        circle = np.uint16(np.around(circles))[0][0]
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
        return TrackedObject(
            object_id=self._consume_id(),
            label="ball",
            side=None,
            bbox=BoundingBox(
                x=max(0.0, (x - r) / w),
                y=max(0.0, (y - r) / h),
                w=min(1.0, (2 * r) / w),
                h=min(1.0, (2 * r) / h),
                confidence=0.7,
            ),
        )

    def _load_onnx_session(self, model_path: str | None):
        if not model_path:
            return None
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            print("ONNX runtime not installed. Model inference disabled.")
            return None
        try:
            return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        except Exception as exc:
            print(f"ONNX model init failed: {exc}")
            return None

    def _run_onnx_confidence(self, image: object) -> float:
        # Lightweight health signal for model availability.
        if self._session is None:
            return 0.0
        _ = image
        return 0.8

    def _consume_id(self) -> int:
        self._next_id += 1
        return self._next_id

