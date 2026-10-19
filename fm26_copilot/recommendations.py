from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .contracts import Recommendation, TacticalState, TeamSide


@dataclass
class RecommendationEngine:
    managed_team: TeamSide = TeamSide.HOME

    def generate(self, state: TacticalState) -> List[Recommendation]:
        recs: List[Recommendation] = []
        managed_risk = state.risk_home if self.managed_team == TeamSide.HOME else state.risk_away
        managed_press = state.press_resistance_home if self.managed_team == TeamSide.HOME else state.press_resistance_away
        managed_def_line = state.defensive_line_home if self.managed_team == TeamSide.HOME else state.defensive_line_away
        overload_against_managed = (
            "away_flank_overload" in state.pattern_tags if self.managed_team == TeamSide.HOME else "home_flank_overload" in state.pattern_tags
        )

        if managed_risk > 0.62:
            recs.append(
                Recommendation(
                    rec_id="lower_line_reduce_space",
                    title="Снизить линию обороны на 1 шаг",
                    body="Соперник регулярно получает пространство за спиной блока.",
                    team=self.managed_team,
                    impact=0.72,
                    confidence=min(0.95, 0.55 + managed_risk * 0.5),
                    risk=0.24,
                    applies_if=["managed_risk > 0.62", "phase in transition_defense/final_third"],
                    evidence=["managed_high_risk_block", f"def_line={managed_def_line:.2f}"],
                )
            )

        if overload_against_managed:
            recs.append(
                Recommendation(
                    rec_id="protect_wide_flank",
                    title="Укрепить перегруженный фланг в обороне",
                    body="Перегруз соперника на фланге повторяется, есть риск cutback.",
                    team=self.managed_team,
                    impact=0.67,
                    confidence=0.74,
                    risk=0.28,
                    applies_if=["flank_overload against managed team detected >= 2 windows"],
                    evidence=["pattern:flank_overload"],
                )
            )

        if managed_press < 0.45:
            recs.append(
                Recommendation(
                    rec_id="add_short_support",
                    title="Упростить выход из прессинга",
                    body="Добавьте короткую опцию через опорника/ЦЗ и снизьте риск потерь.",
                    team=self.managed_team,
                    impact=0.61,
                    confidence=0.71,
                    risk=0.22,
                    applies_if=["managed_press_resistance < 0.45"],
                    evidence=["managed_pressing_issues"],
                )
            )

        return sorted(recs, key=lambda r: (r.impact * r.confidence - r.risk), reverse=True)

