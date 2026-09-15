from __future__ import annotations

from pathlib import Path

import numpy as np

from . import fm26_scoreboard_rois as sb


def _draw_horizontal_ruler_ticks(canvas: object, y_edge: int, w: int) -> None:
    import cv2  # type: ignore

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(canvas, (0, y_edge), (w - 1, y_edge), (70, 200, 90), 1)
    for tick in range(0, 101, 5):
        x = int(round(w * tick / 100.0))
        cv2.line(canvas, (x, y_edge), (x, y_edge + 12), (60, 255, 120), 1)
        if tick % 10 == 0:
            tx = max(2, min(w - 28, x - 10))
            cv2.putText(
                canvas,
                str(tick),
                (tx, y_edge + 28),
                font,
                0.42,
                (235, 235, 245),
                1,
                cv2.LINE_AA,
            )


def render_scoreboard_strip_with_ruler(frame_bgr: object) -> np.ndarray:
    """Strip crop with a 0–100 ruler (tick = ``round(x_norm * 100)`` over full frame width)."""
    import cv2  # type: ignore

    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    h, w = frame_bgr.shape[:2]
    y0, y1 = int(h * sb.TOP_Y0), int(h * sb.TOP_Y1)
    strip = np.asarray(frame_bgr[y0:y1, :])
    sh, sw = strip.shape[:2]
    pad = 52
    canvas = np.zeros((sh + pad, sw, 3), dtype=np.uint8)
    canvas[:sh, :] = strip

    font = cv2.FONT_HERSHEY_SIMPLEX
    roi_edges: list[tuple[str, float, float, tuple[int, int, int]]] = [
        ("home_kit", sb.HOME_KIT_X0, sb.HOME_KIT_X1, (180, 120, 80)),
        ("home_name", sb.HOME_NAME_X0, sb.HOME_NAME_X1, (200, 160, 255)),
        ("score", sb.SCORE_X0, sb.SCORE_X1, (120, 255, 160)),
        ("away_name", sb.AWAY_NAME_X0, sb.AWAY_NAME_X1, (255, 200, 120)),
        ("away_kit", sb.AWAY_KIT_X0, sb.AWAY_KIT_X1, (180, 220, 255)),
    ]
    for label, xa, xb, col in roi_edges:
        for xn in (xa, xb):
            xi = int(round(sw * xn))
            cv2.line(canvas, (xi, 0), (xi, sh - 1), col, 1)
            t = int(round(xn * 100))
            cv2.putText(
                canvas,
                f"{t}",
                (max(2, xi - 10), max(14, sh // 3)),
                font,
                0.32,
                col,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            label,
            (int(round(sw * (xa + xb) / 2)) - 24, max(18, sh - 8)),
            font,
            0.32,
            col,
            1,
            cv2.LINE_AA,
        )

    _draw_horizontal_ruler_ticks(canvas, sh, sw)
    cv2.putText(
        canvas,
        "ruler: 0..100 == x across FULL frame (x_norm * 100)",
        (4, sh + pad - 8),
        font,
        0.34,
        (160, 170, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas


def render_full_frame_ruler_overlay(frame_bgr: object) -> np.ndarray:
    """Full screenshot: scoreboard band outlined + tick marks 0..100 on bottom edge of band."""
    import cv2  # type: ignore

    vis = np.asarray(frame_bgr).copy()
    h, w = vis.shape[:2]
    y0, y1 = int(h * sb.TOP_Y0), int(h * sb.TOP_Y1)
    cv2.rectangle(vis, (0, y0), (w - 1, y1), (40, 220, 100), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for tick in range(0, 101, 10):
        x = int(round(w * tick / 100.0))
        y_end = min(h - 1, y1 + 18)
        cv2.line(vis, (x, y1), (x, y_end), (50, 255, 120), 1)
        cv2.putText(
            vis,
            str(tick),
            (max(0, x - 14), min(h - 4, y1 + 34)),
            font,
            0.48,
            (230, 230, 240),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        vis,
        "scoreboard band (green); ticks 0-100 = horizontal position",
        (8, max(y0 - 6, 16)),
        font,
        0.45,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )
    return vis


def dump_scoreboard_crops(frame_bgr: object, out_dir: str | Path) -> list[str]:
    """Save scoreboard-related crops as PNGs so you can tune ``scoreboard_rois.json``."""
    import cv2  # type: ignore

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return written

    h, w = frame_bgr.shape[:2]
    y0, y1 = int(h * sb.TOP_Y0), int(h * sb.TOP_Y1)
    strip = frame_bgr[y0:y1, :]
    path_strip = out / "00_full_scoreboard_strip.png"
    cv2.imwrite(str(path_strip), strip)
    written.append(str(path_strip))

    path_ruler = out / "00b_scoreboard_strip_ruler_0_100.png"
    cv2.imwrite(str(path_ruler), render_scoreboard_strip_with_ruler(frame_bgr))
    written.append(str(path_ruler))

    path_full = out / "00c_full_frame_ruler_overlay.png"
    cv2.imwrite(str(path_full), render_full_frame_ruler_overlay(frame_bgr))
    written.append(str(path_full))

    regions = [
        ("01_home_name", sb.HOME_NAME_X0, sb.HOME_NAME_X1, sb.TOP_Y0, sb.TOP_Y1),
        ("02_score", sb.SCORE_X0, sb.SCORE_X1, sb.TOP_Y0, sb.TOP_Y1),
        ("03_away_name", sb.AWAY_NAME_X0, sb.AWAY_NAME_X1, sb.TOP_Y0, sb.TOP_Y1),
        ("04_home_kit", sb.HOME_KIT_X0, sb.HOME_KIT_X1, sb.TOP_Y0, sb.TOP_Y1),
        ("05_away_kit", sb.AWAY_KIT_X0, sb.AWAY_KIT_X1, sb.TOP_Y0, sb.TOP_Y1),
    ]
    for label, x0, x1, ry0, ry1 in regions:
        crop = sb.crop_norm(frame_bgr, x0, x1, ry0, ry1)
        if crop.size == 0:
            continue
        p = out / f"{label}.png"
        cv2.imwrite(str(p), crop)
        written.append(str(p))
    return written
