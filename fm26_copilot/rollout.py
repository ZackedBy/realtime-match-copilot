from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RolloutWave:
    name: str
    scenarios: List[str]
    traffic_pct: int
    requires_shadow_eval: bool = True


DEFAULT_WAVES = [
    RolloutWave(
        name="wave1",
        scenarios=["flank_defense", "press_escape", "rest_defense"],
        traffic_pct=10,
    ),
    RolloutWave(
        name="wave2",
        scenarios=["build_up_variants", "set_piece_marking", "transition_shapes"],
        traffic_pct=35,
    ),
    RolloutWave(
        name="wave3",
        scenarios=["personalized_style_recs", "opponent_specific_adaptation"],
        traffic_pct=100,
    ),
]

