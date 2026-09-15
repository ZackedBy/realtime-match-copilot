from __future__ import annotations

from dataclasses import dataclass

from .contracts import TeamSide


@dataclass
class TeamContext:
    managed_team_code: str = "ASM"
    managed_team_side: TeamSide = TeamSide.HOME
    auto_side_enabled: bool = True

