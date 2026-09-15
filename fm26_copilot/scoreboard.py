from __future__ import annotations

from dataclasses import dataclass

from .contracts import TeamSide
from . import fm26_scoreboard_rois as sb
from .match_intel import MatchIntelExtractor


@dataclass
class SideDetectionResult:
    side: TeamSide | None
    confidence: float
    reason: str


class ScoreboardSideResolver:
    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr
        self._team_reader = MatchIntelExtractor(use_ocr=True)

    def resolve(self, frame_bgr: object, managed_team_code: str) -> SideDetectionResult:
        if frame_bgr is None:
            return SideDetectionResult(side=None, confidence=0.0, reason="no_frame")

        if self.use_ocr:
            ocr_result = self._resolve_by_ocr(frame_bgr, managed_team_code)
            if ocr_result.side is not None:
                return ocr_result

        chip_result = self._resolve_by_team_color_chips(frame_bgr)
        if chip_result.side is not None:
            return chip_result
        return SideDetectionResult(side=None, confidence=0.0, reason="unresolved")

    def _resolve_by_ocr(self, frame_bgr: object, managed_team_code: str) -> SideDetectionResult:
        left_txt, right_txt, ocr_reason = self._team_reader.read_scoreboard_teams(frame_bgr)
        if ocr_reason == "ocr_unavailable":
            return SideDetectionResult(side=None, confidence=0.0, reason="ocr_unavailable")
        if not left_txt and not right_txt:
            return SideDetectionResult(side=None, confidence=0.0, reason=f"ocr_miss:{ocr_reason}")

        managed = managed_team_code.upper().strip()
        if managed and managed in left_txt:
            return SideDetectionResult(side=TeamSide.HOME, confidence=0.85, reason=f"ocr_left:{left_txt}")
        if managed and managed in right_txt:
            return SideDetectionResult(side=TeamSide.AWAY, confidence=0.85, reason=f"ocr_right:{right_txt}")
        return SideDetectionResult(side=None, confidence=0.0, reason=f"ocr_miss:{left_txt}|{right_txt}")

    def _resolve_by_team_color_chips(self, frame_bgr: object) -> SideDetectionResult:
        # FM scoreboard shows kit colors next to team names.
        # If both chip zones have enough chroma, we return a low-confidence guess
        # that can stabilize after a few frames in app-level smoothing.
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return SideDetectionResult(side=None, confidence=0.0, reason="chip_unavailable")

        hk = sb.crop_norm(frame_bgr, sb.HOME_KIT_X0, sb.HOME_KIT_X1, sb.TOP_Y0, sb.TOP_Y1)
        ak = sb.crop_norm(frame_bgr, sb.AWAY_KIT_X0, sb.AWAY_KIT_X1, sb.TOP_Y0, sb.TOP_Y1)
        if hk.size == 0 or ak.size == 0:
            return SideDetectionResult(side=None, confidence=0.0, reason="chip_empty")
        w_h = hk.shape[1]
        left_zone = hk[:, : max(1, w_h // 2)]
        w_a = ak.shape[1]
        right_zone = ak[:, : max(1, w_a // 2)]

        def chroma_score(zone: object) -> float:
            hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1].astype(np.float32)
            val = hsv[:, :, 2].astype(np.float32)
            return float((sat * (val / 255.0)).mean())

        l_score = chroma_score(left_zone)
        r_score = chroma_score(right_zone)
        # Not enough evidence to map team code to side only from chips.
        if max(l_score, r_score) < 25.0:
            return SideDetectionResult(side=None, confidence=0.0, reason="chip_low_signal")
        guessed = TeamSide.HOME if l_score >= r_score else TeamSide.AWAY
        return SideDetectionResult(side=guessed, confidence=0.35, reason=f"chip_score:{l_score:.1f}/{r_score:.1f}")

