from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class MatchPhase(str, Enum):
    BUILD_UP = "build_up"
    FINAL_THIRD = "final_third"
    TRANSITION_ATTACK = "transition_attack"
    TRANSITION_DEFENSE = "transition_defense"
    SET_PIECE = "set_piece"


class TeamSide(str, Enum):
    HOME = "home"
    AWAY = "away"


@dataclass(slots=True)
class BoundingBox:
    x: float
    y: float
    w: float
    h: float
    confidence: float


@dataclass(slots=True)
class TrackedObject:
    object_id: int
    label: str
    side: Optional[TeamSide]
    bbox: BoundingBox
    velocity_x: float = 0.0
    velocity_y: float = 0.0


@dataclass(slots=True)
class FrameSignals:
    timestamp_ms: int
    scoreboard_time_s: int
    score_home: int
    score_away: int
    possession_home_pct: float
    fatigue_home: float
    fatigue_away: float
    camera_zoom: float
    raw_frame: object | None = None
    tracked_objects: List[TrackedObject] = field(default_factory=list)


@dataclass(slots=True)
class TacticalState:
    phase: MatchPhase
    compactness_home: float
    compactness_away: float
    width_home: float
    width_away: float
    defensive_line_home: float
    defensive_line_away: float
    flank_overload_side: Optional[TeamSide]
    press_resistance_home: float
    press_resistance_away: float
    risk_home: float
    risk_away: float
    pattern_tags: List[str] = field(default_factory=list)


@dataclass(slots=True)
class Recommendation:
    rec_id: str
    title: str
    body: str
    team: TeamSide
    impact: float
    confidence: float
    risk: float
    applies_if: List[str]
    evidence: List[str]


@dataclass(slots=True)
class OverlayCard:
    priority: int
    recommendation: Recommendation
    stale_after_ms: int


@dataclass(slots=True)
class TelemetrySample:
    stage_latencies_ms: Dict[str, float]
    pipeline_fps: float
    model_confidence: Dict[str, float]
    recommendation_quality_proxy: float
    dropped_frames: int
    fail_safe_active: bool

