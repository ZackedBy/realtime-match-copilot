from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import Recommendation


@dataclass
class ReplayExpectation:
    scenario_tag: str
    expected_recommendation_ids: list[str]


def score_replay(recommendations: Iterable[Recommendation], expectation: ReplayExpectation) -> float:
    got = {r.rec_id for r in recommendations}
    expected = set(expectation.expected_recommendation_ids)
    if not expected:
        return 1.0
    overlap = len(got & expected)
    return overlap / len(expected)

