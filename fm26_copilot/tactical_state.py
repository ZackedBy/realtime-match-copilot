from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Deque, List

from .contracts import MatchPhase, TacticalState, TeamSide
from .vision import VisionOutput


def _spread(values: List[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


@dataclass
class TacticalStateEngine:
    history_size: int = 120
    _phases: Deque[MatchPhase] = field(default_factory=deque)

    def infer(self, vision_output: VisionOutput) -> TacticalState:
        home_x = [p.bbox.x for p in vision_output.home_players]
        home_y = [p.bbox.y for p in vision_output.home_players]
        away_x = [p.bbox.x for p in vision_output.away_players]
        away_y = [p.bbox.y for p in vision_output.away_players]

        home_width = _spread(home_x)
        away_width = _spread(away_x)
        home_compactness = 1.0 - min(1.0, _spread(home_y))
        away_compactness = 1.0 - min(1.0, _spread(away_y))

        home_line = mean(home_y) if home_y else 0.5
        away_line = mean(away_y) if away_y else 0.5

        ball_x = vision_output.ball.bbox.x if vision_output.ball else 0.5
        if ball_x > 0.75:
            phase = MatchPhase.FINAL_THIRD
        elif ball_x < 0.25:
            phase = MatchPhase.BUILD_UP
        elif abs((vision_output.frame.possession_home_pct / 100.0) - 0.5) > 0.1:
            phase = MatchPhase.TRANSITION_ATTACK
        else:
            phase = MatchPhase.TRANSITION_DEFENSE

        self._phases.append(phase)
        while len(self._phases) > self.history_size:
            self._phases.popleft()

        flank_side = None
        if vision_output.ball:
            if vision_output.ball.bbox.y < 0.2:
                flank_side = TeamSide.HOME
            elif vision_output.ball.bbox.y > 0.8:
                flank_side = TeamSide.AWAY

        press_res_home = max(0.0, min(1.0, home_compactness + 0.15 - vision_output.frame.fatigue_home * 0.4))
        press_res_away = max(0.0, min(1.0, away_compactness + 0.15 - vision_output.frame.fatigue_away * 0.4))
        risk_home = max(0.0, min(1.0, (1.0 - home_compactness) * 0.6 + vision_output.frame.fatigue_home * 0.4))
        risk_away = max(0.0, min(1.0, (1.0 - away_compactness) * 0.6 + vision_output.frame.fatigue_away * 0.4))

        pattern_tags: list[str] = []
        if flank_side == TeamSide.AWAY:
            pattern_tags.append("away_flank_overload")
        if flank_side == TeamSide.HOME:
            pattern_tags.append("home_flank_overload")
        if press_res_home < 0.45:
            pattern_tags.append("home_pressing_issues")
        if press_res_away < 0.45:
            pattern_tags.append("away_pressing_issues")
        if risk_home > 0.65:
            pattern_tags.append("home_high_risk_block")
        if risk_away > 0.65:
            pattern_tags.append("away_high_risk_block")

        return TacticalState(
            phase=phase,
            compactness_home=home_compactness,
            compactness_away=away_compactness,
            width_home=home_width,
            width_away=away_width,
            defensive_line_home=home_line,
            defensive_line_away=away_line,
            flank_overload_side=flank_side,
            press_resistance_home=press_res_home,
            press_resistance_away=press_res_away,
            risk_home=risk_home,
            risk_away=risk_away,
            pattern_tags=pattern_tags,
        )

