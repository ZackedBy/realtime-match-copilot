from __future__ import annotations

from unittest.mock import MagicMock

from fm26_copilot.contracts import TeamSide
from fm26_copilot.scoreboard import ScoreboardSideResolver, SideDetectionResult


def test_no_frame() -> None:
    resolver = ScoreboardSideResolver(use_ocr=False)
    result = resolver.resolve(None, "ASM")
    assert result.side is None
    assert result.reason == "no_frame"


def test_ocr_left_maps_to_home() -> None:
    resolver = ScoreboardSideResolver(use_ocr=True)
    resolver._team_reader = MagicMock()
    resolver._team_reader.read_scoreboard_teams.return_value = ("ASM", "OPP", "ok")
    result = resolver.resolve(object(), "ASM")
    assert result.side == TeamSide.HOME
    assert result.confidence == 0.85
    assert result.reason.startswith("ocr_left:")


def test_ocr_right_maps_to_away() -> None:
    resolver = ScoreboardSideResolver(use_ocr=True)
    resolver._team_reader = MagicMock()
    resolver._team_reader.read_scoreboard_teams.return_value = ("OPP", "ASM", "ok")
    result = resolver.resolve(object(), "asm")
    assert result.side == TeamSide.AWAY


def test_ocr_miss_then_unresolved_without_chips() -> None:
    resolver = ScoreboardSideResolver(use_ocr=True)
    resolver._team_reader = MagicMock()
    resolver._team_reader.read_scoreboard_teams.return_value = ("AAA", "BBB", "ok")
    resolver._resolve_by_team_color_chips = MagicMock(  # type: ignore[method-assign]
        return_value=SideDetectionResult(side=None, confidence=0.0, reason="chip_low_signal")
    )
    result = resolver.resolve(object(), "ASM")
    assert result.side is None
    assert result.reason == "chip_low_signal"


def test_use_ocr_false_skips_ocr_path() -> None:
    resolver = ScoreboardSideResolver(use_ocr=False)
    assert resolver.use_ocr is False
    assert resolver._team_reader.use_ocr is False
    resolver._team_reader = MagicMock()
    resolver._resolve_by_team_color_chips = MagicMock(  # type: ignore[method-assign]
        return_value=SideDetectionResult(
            side=TeamSide.HOME, confidence=0.35, reason="chip_score:30.0/10.0"
        )
    )
    result = resolver.resolve(object(), "ASM")
    resolver._team_reader.read_scoreboard_teams.assert_not_called()
    assert result.side == TeamSide.HOME
