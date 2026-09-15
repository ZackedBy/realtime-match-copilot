from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Deque, Dict, List

from .contracts import TelemetrySample


@dataclass
class TelemetryCollector:
    window: int = 300
    _samples: Deque[TelemetrySample] = field(default_factory=deque)

    def push(self, sample: TelemetrySample) -> None:
        self._samples.append(sample)
        while len(self._samples) > self.window:
            self._samples.popleft()

    def snapshot(self) -> Dict[str, float]:
        if not self._samples:
            return {}

        fps_values = [s.pipeline_fps for s in self._samples]
        quality_values = [s.recommendation_quality_proxy for s in self._samples]
        dropped_values = [s.dropped_frames for s in self._samples]
        failsafe_values = [1.0 if s.fail_safe_active else 0.0 for s in self._samples]

        stage_names = set()
        for sample in self._samples:
            stage_names.update(sample.stage_latencies_ms.keys())

        stage_means: Dict[str, float] = {}
        for stage in sorted(stage_names):
            vals: List[float] = []
            for sample in self._samples:
                if stage in sample.stage_latencies_ms:
                    vals.append(sample.stage_latencies_ms[stage])
            if vals:
                stage_means[stage] = mean(vals)

        summary: Dict[str, float] = {
            "avg_fps": mean(fps_values),
            "avg_quality_proxy": mean(quality_values),
            "avg_dropped_frames": mean(dropped_values),
            "failsafe_ratio": mean(failsafe_values),
        }
        for stage, value in stage_means.items():
            summary[f"latency_{stage}_ms"] = value
        return summary

