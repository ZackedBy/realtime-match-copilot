from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterator

from .contracts import BoundingBox, FrameSignals, TeamSide, TrackedObject


@dataclass
class ScreenCaptureConfig:
    target_fps: int = 20
    simulation_mode: bool = True
    monitor_index: int = 0


class ScreenCapture:
    def __init__(self, config: ScreenCaptureConfig):
        self.config = config
        self._dxgi_camera = None
        self._live_mode_enabled = False
        if not self.config.simulation_mode:
            self._live_mode_enabled = self._init_dxgi()

    def frames(self) -> Iterator[FrameSignals]:
        frame_interval = 1 / max(self.config.target_fps, 1)
        scoreboard_time_s = 0
        home_score = 0
        away_score = 0
        while True:
            ts_ms = int(time.time() * 1000)
            scoreboard_time_s += 1
            if random.random() < 0.005:
                home_score += 1
            if random.random() < 0.005:
                away_score += 1

            if self._live_mode_enabled and self._dxgi_camera is not None:
                frame = self._dxgi_camera.get_latest_frame()
                if frame is not None:
                    yield FrameSignals(
                        timestamp_ms=ts_ms,
                        scoreboard_time_s=scoreboard_time_s,
                        score_home=home_score,
                        score_away=away_score,
                        possession_home_pct=random.uniform(35, 65),
                        fatigue_home=min(1.0, scoreboard_time_s / 7200),
                        fatigue_away=min(1.0, scoreboard_time_s / 7000),
                        camera_zoom=1.0,
                        raw_frame=frame,
                        tracked_objects=[],
                    )
                    time.sleep(frame_interval)
                    continue

            yield FrameSignals(
                timestamp_ms=ts_ms,
                scoreboard_time_s=scoreboard_time_s,
                score_home=home_score,
                score_away=away_score,
                possession_home_pct=random.uniform(35, 65),
                fatigue_home=min(1.0, scoreboard_time_s / 7200),
                fatigue_away=min(1.0, scoreboard_time_s / 7000),
                camera_zoom=random.uniform(0.8, 1.4),
                raw_frame=None,
                tracked_objects=self._mock_objects(),
            )
            time.sleep(frame_interval)

    def _init_dxgi(self) -> bool:
        try:
            import dxcam  # type: ignore
        except ImportError:
            print("DXGI capture disabled: install dxcam for live capture.")
            return False
        try:
            self._dxgi_camera = dxcam.create(output_idx=self.config.monitor_index)
            self._dxgi_camera.start(target_fps=self.config.target_fps, video_mode=True)
            print("DXGI Desktop Duplication capture started.")
            return True
        except Exception as exc:
            print(f"DXGI capture init failed, fallback to simulation: {exc}")
            self._dxgi_camera = None
            return False

    def _mock_objects(self) -> list[TrackedObject]:
        objects: list[TrackedObject] = []
        for idx in range(22):
            side = TeamSide.HOME if idx < 11 else TeamSide.AWAY
            objects.append(
                TrackedObject(
                    object_id=idx,
                    label="player",
                    side=side,
                    bbox=BoundingBox(
                        x=random.uniform(0.02, 0.98),
                        y=random.uniform(0.05, 0.95),
                        w=0.03,
                        h=0.06,
                        confidence=random.uniform(0.7, 0.98),
                    ),
                    velocity_x=random.uniform(-0.03, 0.03),
                    velocity_y=random.uniform(-0.03, 0.03),
                )
            )
        objects.append(
            TrackedObject(
                object_id=99,
                label="ball",
                side=None,
                bbox=BoundingBox(
                    x=random.uniform(0.05, 0.95),
                    y=random.uniform(0.06, 0.94),
                    w=0.01,
                    h=0.01,
                    confidence=random.uniform(0.6, 0.95),
                ),
            )
        )
        return objects

