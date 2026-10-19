from __future__ import annotations

from fm26_copilot.contracts import Recommendation, TeamSide
from fm26_copilot.feedback import FeedbackStore


def _rec(rec_id: str) -> Recommendation:
    return Recommendation(
        rec_id=rec_id,
        title="t",
        body="b",
        team=TeamSide.HOME,
        impact=0.5,
        confidence=0.8,
        risk=0.2,
        applies_if=[],
        evidence=[],
    )


def test_quality_proxy_neutral_without_feedback() -> None:
    store = FeedbackStore()
    assert store.quality_proxy() == 0.5


def test_quality_proxy_reflects_accept_dismiss() -> None:
    store = FeedbackStore()
    store.mark_accepted(_rec("a"))
    store.mark_accepted(_rec("a"))
    store.mark_dismissed(_rec("b"))
    assert abs(store.quality_proxy() - (2 / 3)) < 1e-9
