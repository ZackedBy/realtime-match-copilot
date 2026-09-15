from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .contracts import BoundingBox, TrackedObject


def _center(box: BoundingBox) -> tuple[float, float]:
    return (box.x + box.w * 0.5, box.y + box.h * 0.5)


@dataclass
class TrackedState:
    track_id: int
    label: str
    center_x: float
    center_y: float
    stale_ticks: int = 0


@dataclass
class SimpleTracker:
    max_distance: float = 0.08
    max_stale_ticks: int = 8
    _states: Dict[int, TrackedState] = field(default_factory=dict)
    _next_id: int = 10000

    def update(self, detections: List[TrackedObject]) -> List[TrackedObject]:
        updated: List[TrackedObject] = []
        used_track_ids: set[int] = set()

        for det in detections:
            cx, cy = _center(det.bbox)
            track_id = self._best_match(det.label, cx, cy, used_track_ids)
            if track_id is None:
                track_id = self._new_track_id()
            self._states[track_id] = TrackedState(track_id=track_id, label=det.label, center_x=cx, center_y=cy, stale_ticks=0)
            used_track_ids.add(track_id)
            updated.append(
                TrackedObject(
                    object_id=track_id,
                    label=det.label,
                    side=det.side,
                    bbox=det.bbox,
                    velocity_x=det.velocity_x,
                    velocity_y=det.velocity_y,
                )
            )

        self._age_unmatched(used_track_ids)
        return updated

    def _best_match(self, label: str, cx: float, cy: float, used: set[int]) -> int | None:
        candidate: Tuple[int, float] | None = None
        for track_id, state in self._states.items():
            if track_id in used or state.label != label:
                continue
            dist = ((cx - state.center_x) ** 2 + (cy - state.center_y) ** 2) ** 0.5
            if dist > self.max_distance:
                continue
            if candidate is None or dist < candidate[1]:
                candidate = (track_id, dist)
        return candidate[0] if candidate else None

    def _age_unmatched(self, used: set[int]) -> None:
        for track_id, state in list(self._states.items()):
            if track_id in used:
                continue
            state.stale_ticks += 1
            if state.stale_ticks > self.max_stale_ticks:
                del self._states[track_id]

    def _new_track_id(self) -> int:
        self._next_id += 1
        return self._next_id

