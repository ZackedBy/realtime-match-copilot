from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .contracts import Recommendation


@dataclass
class FeedbackStore:
    accepted: Dict[str, int] = field(default_factory=dict)
    dismissed: Dict[str, int] = field(default_factory=dict)

    def mark_accepted(self, rec: Recommendation) -> None:
        self.accepted[rec.rec_id] = self.accepted.get(rec.rec_id, 0) + 1

    def mark_dismissed(self, rec: Recommendation) -> None:
        self.dismissed[rec.rec_id] = self.dismissed.get(rec.rec_id, 0) + 1

    def quality_proxy(self) -> float:
        total_acc = sum(self.accepted.values())
        total_dis = sum(self.dismissed.values())
        total = total_acc + total_dis
        if total == 0:
            return 0.5
        return total_acc / total

