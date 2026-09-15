from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque
from collections import deque

from .telemetry import TelemetryCollector


@dataclass
class DriftDetector:
    window: int = 120
    threshold_quality_drop: float = 0.2
    _quality_hist: Deque[float] = field(default_factory=deque)

    def update(self, quality_proxy: float) -> bool:
        self._quality_hist.append(quality_proxy)
        while len(self._quality_hist) > self.window:
            self._quality_hist.popleft()
        if len(self._quality_hist) < self.window:
            return False
        head = list(self._quality_hist)[: self.window // 2]
        tail = list(self._quality_hist)[self.window // 2 :]
        return (sum(head) / len(head)) - (sum(tail) / len(tail)) > self.threshold_quality_drop


@dataclass
class ReleasePolicy:
    min_avg_fps: float = 12.0
    max_fail_safe_ratio: float = 0.25
    min_quality_proxy: float = 0.58

    def is_rollout_safe(self, telemetry: TelemetryCollector) -> tuple[bool, str]:
        snap = telemetry.snapshot()
        if not snap:
            return False, "Нет данных телеметрии."
        if snap.get("avg_fps", 0.0) < self.min_avg_fps:
            return False, "FPS ниже порога."
        if snap.get("failsafe_ratio", 1.0) > self.max_fail_safe_ratio:
            return False, "Слишком частое включение fail-safe."
        if snap.get("avg_quality_proxy", 0.0) < self.min_quality_proxy:
            return False, "Качество рекомендаций ниже порога."
        return True, "Можно расширять rollout."

