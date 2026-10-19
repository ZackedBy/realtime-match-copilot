"""
FM26 match screen — top scoreboard regions (normalized 0..1).

Tuned for **1920×1080** tactical match view (Russian UI): thin central bar
``[home][score][away]`` below the clock. Wider aspect ratios may need tweaks.

**Калибровка:** ``--dump-scoreboard-crops`` пишет ``00b_*`` с линейкой **0–100**
по ширине кадра (делите отметки на 100 → доля ``x`` в JSON). См. также
``x_norm_to_ruler`` / ``ruler_unit_to_x_norm``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Thin strip: дважды по +5% высоты полоски (2 × 0.0018 от базы 0.0208/0.0568)
TOP_Y0 = 0.0172
TOP_Y1 = 0.0532

# Horizontal bands — ruler 0..100 → divide by 100 (user calibration 1920×1080)
HOME_NAME_X0 = 0.49
HOME_NAME_X1 = 0.52

SCORE_X0 = 0.52
SCORE_X1 = 0.56

# Away name (как было до последней правки — у тебя совпадало)
AWAY_NAME_X0 = 0.562
AWAY_NAME_X1 = 0.598

# Домашние цвета (линейка 48.3–48.5)
HOME_KIT_X0 = 0.483
HOME_KIT_X1 = 0.485

# Выездные цвета (линейка 60.1–60.3)
AWAY_KIT_X0 = 0.601
AWAY_KIT_X1 = 0.603

_ROI_KEYS = frozenset(
    {
        "TOP_Y0",
        "TOP_Y1",
        "HOME_NAME_X0",
        "HOME_NAME_X1",
        "SCORE_X0",
        "SCORE_X1",
        "AWAY_NAME_X0",
        "AWAY_NAME_X1",
        "HOME_KIT_X0",
        "HOME_KIT_X1",
        "AWAY_KIT_X0",
        "AWAY_KIT_X1",
    }
)


def configure_from_path(path: str | None) -> bool:
    """Load ROI fractions from JSON; mutates module globals. Returns True if file applied."""
    g = globals()
    p: Path | None = None
    if path:
        p = Path(path)
    elif os.environ.get("FM26_ROI_CONFIG"):
        p = Path(os.environ["FM26_ROI_CONFIG"])
    else:
        cand = Path.home() / ".fm26_copilot" / "scoreboard_rois.json"
        if cand.is_file():
            p = cand

    if p is None or not p.is_file():
        return False

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"ROI JSON is not valid JSON: {p}") from exc
    if not isinstance(data, dict):
        raise ValueError("ROI JSON must be an object with float keys")
    for key, val in data.items():
        if key not in _ROI_KEYS:
            continue
        try:
            parsed = float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ROI key {key!r} must be a float, got {val!r}") from exc
        if not 0.0 <= parsed <= 1.0:
            raise ValueError(f"ROI key {key!r} must be in [0, 1], got {parsed}")
        g[key] = parsed
    print(f"[fm26_copilot] ROI loaded from {p}")
    return True


def x_norm_to_ruler(u: float) -> float:
    """Map normalized horizontal coordinate ``0..1`` to ruler ticks ``0..100`` (full frame width)."""
    return float(u) * 100.0


def ruler_unit_to_x_norm(tick: float) -> float:
    """Convert ruler tick ``0..100`` to normalized ``x``."""
    return max(0.0, min(1.0, float(tick) / 100.0))


def crop_norm(frame_bgr: object, x0: float, x1: float, y0: float, y1: float) -> object:
    h, w = frame_bgr.shape[:2]
    xi0 = max(0, int(w * x0))
    xi1 = min(w, int(w * x1))
    yi0 = max(0, int(h * y0))
    yi1 = min(h, int(h * y1))
    if xi1 <= xi0 or yi1 <= yi0:
        return frame_bgr[0:0, 0:0]
    return frame_bgr[yi0:yi1, xi0:xi1]
