from __future__ import annotations

from fm26_copilot.contracts import MatchPhase, TacticalState, TeamSide
from fm26_copilot.recommendations import RecommendationEngine


def _state(**overrides: object) -> TacticalState:
    base = dict(
        phase=MatchPhase.TRANSITION_DEFENSE,
        compactness_home=0.5,
        compactness_away=0.5,
        width_home=0.5,
        width_away=0.5,
        defensive_line_home=0.5,
        defensive_line_away=0.5,
        flank_overload_side=None,
        press_resistance_home=0.7,
        press_resistance_away=0.7,
        risk_home=0.4,
        risk_away=0.4,
        pattern_tags=[],
    )
    base.update(overrides)
    return TacticalState(**base)  # type: ignore[arg-type]


def test_high_risk_emits_lower_line() -> None:
    eng = RecommendationEngine(managed_team=TeamSide.HOME)
    recs = eng.generate(_state(risk_home=0.8))
    ids = [r.rec_id for r in recs]
    assert "lower_line_reduce_space" in ids


def test_low_risk_skips_lower_line() -> None:
    eng = RecommendationEngine(managed_team=TeamSide.HOME)
    recs = eng.generate(_state(risk_home=0.4))
    assert all(r.rec_id != "lower_line_reduce_space" for r in recs)


def test_flank_overload_against_home() -> None:
    eng = RecommendationEngine(managed_team=TeamSide.HOME)
    recs = eng.generate(_state(pattern_tags=["away_flank_overload"]))
    assert any(r.rec_id == "protect_wide_flank" for r in recs)


def test_press_resistance_triggers_support() -> None:
    eng = RecommendationEngine(managed_team=TeamSide.AWAY)
    recs = eng.generate(_state(press_resistance_away=0.3))
    assert any(r.rec_id == "add_short_support" for r in recs)


def test_recs_sorted_by_score() -> None:
    eng = RecommendationEngine(managed_team=TeamSide.HOME)
    recs = eng.generate(
        _state(
            risk_home=0.9,
            press_resistance_home=0.2,
            pattern_tags=["away_flank_overload"],
        )
    )
    assert len(recs) >= 2
    scores = [r.impact * r.confidence - r.risk for r in recs]
    assert scores == sorted(scores, reverse=True)
