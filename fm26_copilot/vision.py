from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Tuple

from .contracts import FrameSignals, TeamSide, TrackedObject
from .calibration import PitchCalibrator
from .detector import OpenCVOnnxDetector, OpenCVOnnxDetectorConfig
from .tracking import SimpleTracker


@dataclass
class VisionOutput:
    frame: FrameSignals
    home_players: List[TrackedObject]
    away_players: List[TrackedObject]
    ball: TrackedObject | None
    field_coordinates: Dict[int, Tuple[float, float]]
    confidence: float
    calibration_confidence: float


class VisionStack:
    def __init__(self, onnx_model_path: str | None = None, managed_team_code: str = "ASM"):
        self.detector = OpenCVOnnxDetector(
            OpenCVOnnxDetectorConfig(
                onnx_model_path=onnx_model_path,
                managed_team_code=managed_team_code,
            )
        )
        self.tracker = SimpleTracker()
        self.calibrator = PitchCalibrator()

    def process(self, frame: FrameSignals) -> VisionOutput:
        tracked_objects = frame.tracked_objects
        detector_conf = 0.0
        calibration_conf = 0.0
        calibration = None
        if frame.raw_frame is not None:
            detected, detector_conf = self.detector.detect(frame.raw_frame)
            if detected:
                tracked_objects = detected
            calibration = self.calibrator.estimate(frame.raw_frame)
            if calibration is not None:
                calibration_conf = calibration.confidence

        if tracked_objects:
            tracked_objects = self.tracker.update(tracked_objects)

        home_players = [o for o in tracked_objects if o.label == "player" and o.side == TeamSide.HOME]
        away_players = [o for o in tracked_objects if o.label == "player" and o.side == TeamSide.AWAY]
        ball = next((o for o in tracked_objects if o.label == "ball"), None)

        if calibration is not None:
            field_coordinates = self.calibrator.map_to_pitch(tracked_objects, calibration)
        else:
            # Fallback normalization when pitch calibration is unavailable.
            field_coordinates = {}
            for obj in tracked_objects:
                field_coordinates[obj.object_id] = (
                    obj.bbox.x / max(frame.camera_zoom, 0.1),
                    obj.bbox.y / max(frame.camera_zoom, 0.1),
                )

        confs = [o.bbox.confidence for o in tracked_objects]
        confidence = mean(confs) if confs else detector_conf
        return VisionOutput(
            frame=frame,
            home_players=home_players,
            away_players=away_players,
            ball=ball,
            field_coordinates=field_coordinates,
            confidence=confidence,
            calibration_confidence=calibration_conf,
        )

