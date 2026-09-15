from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .contracts import OverlayCard, Recommendation


@dataclass
class SafetyAndConfidenceGate:
    min_confidence: float = 0.68
    max_risk: float = 0.55
    max_cards_per_tick: int = 2

    def filter(self, recs: List[Recommendation], vision_confidence: float, tracker_ok: bool) -> tuple[List[OverlayCard], bool]:
        fail_safe_active = (vision_confidence < 0.55) or (not tracker_ok)
        if fail_safe_active:
            return [], True

        accepted = [
            r
            for r in recs
            if r.confidence >= self.min_confidence and r.risk <= self.max_risk
        ]
        accepted = accepted[: self.max_cards_per_tick]
        cards = [
            OverlayCard(
                priority=idx + 1,
                recommendation=rec,
                stale_after_ms=25_000,
            )
            for idx, rec in enumerate(accepted)
        ]
        return cards, False

