from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

from .contracts import OverlayCard
from .overlay_qt import QtOverlayConfig, QtOverlaySink


@dataclass
class OverlayController:
    rate_limit_sec: float = 8.0
    use_qt_overlay: bool = False
    _last_emit_by_rec_id: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.qt_sink = QtOverlaySink(QtOverlayConfig(enabled=self.use_qt_overlay))

    def renderable_cards(self, cards: List[OverlayCard]) -> List[OverlayCard]:
        now = time.time()
        output: List[OverlayCard] = []
        for card in cards:
            rid = card.recommendation.rec_id
            last_emit = self._last_emit_by_rec_id.get(rid, 0.0)
            if now - last_emit < self.rate_limit_sec:
                continue
            self._last_emit_by_rec_id[rid] = now
            output.append(card)
        return output

    def present(self, cards: List[OverlayCard]) -> None:
        if self.qt_sink.push_cards(cards):
            return
        for card in cards:
            rec = card.recommendation
            print(
                f"[P{card.priority}] {rec.title} | impact={rec.impact:.2f} "
                f"conf={rec.confidence:.2f} risk={rec.risk:.2f}"
            )
            print(f"  why: {rec.body}")
            if rec.evidence:
                print(f"  evidence: {', '.join(rec.evidence)}")

    def present_status(self, lines: List[str]) -> None:
        if not lines:
            return
        payload = "\n".join(lines)
        if self.qt_sink.push_text(payload):
            return
        print("[DIAG]")
        for line in lines:
            print(f"  {line}")

