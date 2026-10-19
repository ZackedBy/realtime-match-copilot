from __future__ import annotations

from fm26_copilot.contracts import Recommendation, TeamSide
from fm26_copilot.safety import SafetyAndConfidenceGate


def _rec(confidence: float = 0.8, risk: float = 0.2, rec_id: str = "r1") -> Recommendation:
    return Recommendation(
        rec_id=rec_id,
        title="t",
        body="b",
        team=TeamSide.HOME,
        impact=0.5,
        confidence=confidence,
        risk=risk,
        applies_if=[],
        evidence=[],
    )


def test_fail_safe_on_low_vision() -> None:
    gate = SafetyAndConfidenceGate()
    cards, fail_safe = gate.filter([_rec()], vision_confidence=0.4, tracker_ok=True)
    assert fail_safe is True
    assert cards == []


def test_fail_safe_on_tracker_down() -> None:
    gate = SafetyAndConfidenceGate()
    cards, fail_safe = gate.filter([_rec()], vision_confidence=0.9, tracker_ok=False)
    assert fail_safe is True
    assert cards == []


def test_filters_low_confidence_and_high_risk() -> None:
    gate = SafetyAndConfidenceGate()
    cards, fail_safe = gate.filter(
        [_rec(confidence=0.5), _rec(confidence=0.9, risk=0.9, rec_id="r2"), _rec(confidence=0.9, risk=0.2, rec_id="r3")],
        vision_confidence=0.9,
        tracker_ok=True,
    )
    assert fail_safe is False
    assert [c.recommendation.rec_id for c in cards] == ["r3"]


def test_max_cards_per_tick() -> None:
    gate = SafetyAndConfidenceGate(max_cards_per_tick=1)
    cards, _ = gate.filter(
        [_rec(rec_id="a"), _rec(rec_id="b")],
        vision_confidence=0.9,
        tracker_ok=True,
    )
    assert len(cards) == 1
