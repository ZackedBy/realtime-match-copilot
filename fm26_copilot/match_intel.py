from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import List

from .contracts import TeamSide
from . import fm26_scoreboard_rois as sb
from . import ocr_engine


def _cv2_const(cv_mod: object, name: str, fallback: int) -> int:
    """Some ``cv2`` wheels omit enum names on the top-level module; values are stable."""
    return int(getattr(cv_mod, name, fallback))


# Fallback if ``cv2.THRESH_BINARY`` is missing on the module object.
_CV2_THRESH_BINARY = 0


@dataclass
class PlayerInfo:
    name: str
    position: str


@dataclass
class MatchIntelSnapshot:
    opponent_team: str
    opponent_players: List[PlayerInfo]
    score_home: str
    score_away: str
    home_team_code: str
    away_team_code: str
    home_color: str
    away_color: str
    reason: str


def _crop_center_h_strip(zone_bgr: object, frac: float = 0.42) -> object:
    """Middle horizontal window of a kit band — trims purple chrome / edges."""
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return zone_bgr
    if zone_bgr is None or getattr(zone_bgr, "size", 0) == 0:
        return zone_bgr
    x = np.asarray(zone_bgr)
    _h, w = x.shape[:2]
    margin = (1.0 - frac) / 2.0
    c0 = int(w * margin)
    c1 = int(w * (1.0 - margin))
    return x[:, c0:max(c0 + 1, c1)]


def _crop_left_h_strip(zone_bgr: object, frac: float = 0.55) -> object:
    """Left fraction of width — FM away badge: two colours stacked in a left column."""
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return zone_bgr
    if zone_bgr is None or getattr(zone_bgr, "size", 0) == 0:
        return zone_bgr
    x = np.asarray(zone_bgr)
    _h, w = x.shape[:2]
    c1 = max(1, int(w * frac))
    return x[:, :c1]


def _split_kit_vertical_stack(zone_bgr: object) -> tuple[object, object]:
    """FM badge column: **primary (main) = lower half**, **secondary = upper half**."""
    if zone_bgr is None or getattr(zone_bgr, "size", 0) == 0:
        return zone_bgr, zone_bgr
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return zone_bgr, zone_bgr
    img = np.asarray(zone_bgr)
    if img.shape[0] > 6:
        img = img[int(img.shape[0] * 0.15) :, :]
    h = img.shape[0]
    mid = max(1, h // 2)
    primary = img[mid:, :]
    secondary = img[:mid, :]
    return primary, secondary


def _kit_zone_to_hsv(zone_bgr: object) -> object:
    """3-channel BGR (default) or RGB if ``FM26_KIT_INPUT_RGB=1`` — fixes red/blue swap."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None
    if zone_bgr is None or getattr(zone_bgr, "size", 0) == 0:
        return None
    z = np.asarray(zone_bgr)
    if z.ndim == 2:
        z = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
    elif z.shape[2] == 4:
        z = cv2.cvtColor(z, cv2.COLOR_BGRA2BGR)
    rgb = os.environ.get("FM26_KIT_INPUT_RGB", "").lower() in ("1", "true", "yes")
    if rgb:
        return cv2.cvtColor(z, cv2.COLOR_RGB2HSV)
    return cv2.cvtColor(z, cv2.COLOR_BGR2HSV)


def _hsv_kit_color_name(zone_bgr: object) -> str:
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return "unknown"
    if zone_bgr is None or getattr(zone_bgr, "size", 0) == 0:
        return "unknown"
    hsv = _kit_zone_to_hsv(zone_bgr)
    if hsv is None:
        return "unknown"
    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))
    # Dark blue scoreboard bars (Club Brugge) often read as low-S without an explicit blue rule
    if s_mean >= 26 and 95 <= h_mean <= 138:
        return "blue"
    # Burgundy / dark red (Monaco bar) sits at low H; classify before orange
    if s_mean >= 35:
        if h_mean <= 15 or h_mean >= 165:
            return "red"
        if h_mean < 28:
            return "orange"
        if h_mean < 38:
            return "yellow"
        if h_mean < 85:
            return "green"
        if h_mean < 130:
            return "blue"
        if h_mean < 160:
            return "purple"
    if v_mean < 45 and s_mean < 50:
        return "black"
    if s_mean < 28 and v_mean > 170:
        return "white"
    if s_mean < 35:
        return "gray"
    if h_mean < 16 or h_mean >= 165:
        return "red"
    if h_mean < 85:
        return "green"
    if h_mean < 130:
        return "blue"
    return "gray"


class MatchIntelExtractor:
    POS_RE = re.compile(r"\b(GK|CB|LB|RB|DM|CM|AM|LW|RW|ST|CF|WB|LWB|RWB|CDM|CAM)\b", re.IGNORECASE)
    TEAM_TOKEN_RE = re.compile(r"\b([A-Z]{2,5})\b")
    _TEAM_OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    _SCORE_OCR_ALLOWLIST = "0123456789-: Ii|l"
    # FM scoreboard font / EasyOCR recurring confusions (extend as needed)
    _TEAM_OCR_FIXES: dict[str, str] = {"BUIT": "CLU", "BU1T": "CLU", "CL0": "CLU", "ASIV": "ASM", "A5M": "ASM"}

    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr

    def read_scoreboard_teams(self, frame_bgr: object) -> tuple[str, str, str]:
        """Public helper: left team code, right team code, status reason."""
        return self._read_scoreboard_teams(frame_bgr)

    def extract(
        self,
        frame_bgr: object,
        managed_team_code: str,
        managed_side: TeamSide,
        *,
        include_players: bool = True,
    ) -> MatchIntelSnapshot:
        if frame_bgr is None:
            return MatchIntelSnapshot(
                opponent_team="unknown",
                opponent_players=[],
                score_home="-",
                score_away="-",
                home_team_code="",
                away_team_code="",
                home_color="unknown",
                away_color="unknown",
                reason="no_frame",
            )
        home_kit, away_kit = self._read_scoreboard_kit_pair_strings(frame_bgr)
        if not self.use_ocr:
            return MatchIntelSnapshot(
                opponent_team="unknown",
                opponent_players=[],
                score_home="-",
                score_away="-",
                home_team_code="",
                away_team_code="",
                home_color=home_kit,
                away_color=away_kit,
                reason="intel_ocr_disabled",
            )

        team_left, team_right, reason = self._read_scoreboard_teams(frame_bgr)
        score_home, score_away = self._read_scoreboard_score(frame_bgr)
        if reason in ("scoreboard_ocr", "scoreboard_partial") and (score_home == "-" or score_away == "-"):
            reason = f"{reason}|score_unread"
        opponent = "unknown"
        if managed_side == TeamSide.HOME and team_right:
            opponent = team_right
        elif managed_side == TeamSide.AWAY and team_left:
            opponent = team_left
        elif team_left and team_left.upper() != managed_team_code.upper():
            opponent = team_left
        elif team_right and team_right.upper() != managed_team_code.upper():
            opponent = team_right

        players = self._read_player_rows(frame_bgr) if include_players else []
        return MatchIntelSnapshot(
            opponent_team=opponent,
            opponent_players=players,
            score_home=score_home,
            score_away=score_away,
            home_team_code=team_left,
            away_team_code=team_right,
            home_color=home_kit,
            away_color=away_kit,
            reason=reason,
        )

    def _read_scoreboard_kit_pair_strings(self, frame_bgr: object) -> tuple[str, str]:
        """Home: tall strip → vertical stack (main bottom); wide → two columns. Away: left column, vertical stack."""
        hk = sb.crop_norm(frame_bgr, sb.HOME_KIT_X0, sb.HOME_KIT_X1, sb.TOP_Y0, sb.TOP_Y1)
        ak = sb.crop_norm(frame_bgr, sb.AWAY_KIT_X0, sb.AWAY_KIT_X1, sb.TOP_Y0, sb.TOP_Y1)
        if hk.size == 0 or ak.size == 0:
            return "unknown", "unknown"
        if hk.shape[0] > 6:
            hk = hk[int(hk.shape[0] * 0.2) :, :]
        # Узкий высокий кроп (как на скрине формы) — делим верх/низ, не лево/право
        if hk.shape[0] > max(4, int(hk.shape[1] * 1.05)):
            hk_n = _crop_center_h_strip(hk, 0.62)
            c_home_pri, c_home_sec = _split_kit_vertical_stack(hk_n)
        else:
            w_h = hk.shape[1]
            mid_h = max(1, w_h // 2)
            c_home_pri = _crop_center_h_strip(hk[:, :mid_h])
            c_home_sec = _crop_center_h_strip(hk[:, mid_h:])

        ak_col = _crop_left_h_strip(ak, 0.58)
        c_away_pri, c_away_sec = _split_kit_vertical_stack(ak_col)
        hp = _hsv_kit_color_name(c_home_pri)
        hs = _hsv_kit_color_name(c_home_sec)
        ap = _hsv_kit_color_name(c_away_pri)
        as_ = _hsv_kit_color_name(c_away_sec)
        return f"{hp}+{hs}", f"{ap}+{as_}"

    def _ocr_image_line(
        self,
        image_u8: object,
        psm: int = 7,
        *,
        scale: float = 1.0,
        allowlist: str | None = None,
        uppercase: bool = True,
        min_confidence: float = 0.1,
    ) -> str:
        """Single-line OCR on a grayscale/binary crop (``psm`` kept for API compatibility)."""
        _ = psm
        return ocr_engine.read_text_line(
            image_u8,
            scale=scale,
            uppercase=uppercase,
            allowlist=allowlist,
            min_confidence=min_confidence,
        )

    def _normalize_team_ocr(self, raw: str) -> str:
        raw = re.sub(r"[^A-Z0-9]", "", raw.upper())
        if len(raw) < 2:
            return ""
        # Shirt numbers / stats (e.g. "151") are not club codes
        if raw.isdigit():
            return ""
        if not re.search(r"[A-Z]", raw):
            return ""
        # FM match bar uses short codes (2–5 chars); long strings are almost always OCR noise
        if len(raw) > 5:
            return ""
        m = self.TEAM_TOKEN_RE.search(" " + raw + " ")
        if m:
            return self._TEAM_OCR_FIXES.get(m.group(1), m.group(1))
        fixed = self._TEAM_OCR_FIXES.get(raw, raw)
        return fixed

    @staticmethod
    def _valid_score_pair(a: str, b: str) -> tuple[str, str] | None:
        try:
            x, y = int(a), int(b)
        except ValueError:
            return None
        if not (0 <= x <= 30 and 0 <= y <= 30):
            return None
        return str(x), str(y)

    def _parse_score_text(self, txt: str) -> tuple[str, str] | None:
        if not txt:
            return None
        t = txt.strip()
        for m in re.finditer(r"\b(\d{1,2})\s*[-:Iil|]\s*(\d{1,2})\b", t):
            ok = self._valid_score_pair(m.group(1), m.group(2))
            if ok:
                return ok
        m = re.search(r"(\d{1,2})\s*[-:Iil|]\s*(\d{1,2})", t)
        if m:
            ok = self._valid_score_pair(m.group(1), m.group(2))
            if ok:
                return ok
        # Sliding 1–2 digit pairs (avoids "0261" + "1" style glitches)
        nums = re.findall(r"\d{1,2}", t)
        for i in range(len(nums) - 1):
            ok = self._valid_score_pair(nums[i], nums[i + 1])
            if ok:
                return ok
        return None

    def _ocr_score_variants(self, gray_u8: object) -> str:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return ""
        thr_bin = _cv2_const(cv2, "THRESH_BINARY", _CV2_THRESH_BINARY)
        thr_otsu = _cv2_const(cv2, "THRESH_OTSU", 8)
        morph_open = _cv2_const(cv2, "MORPH_OPEN", 1)
        best_txt = ""
        raw_g = self._ocr_image_line(
            gray_u8,
            scale=1.0,
            allowlist=self._SCORE_OCR_ALLOWLIST,
            uppercase=False,
            min_confidence=0.06,
        )
        parsed_g = self._parse_score_text(raw_g) if raw_g else None
        if parsed_g:
            return raw_g
        if raw_g:
            best_txt = raw_g
        for inv in (False, True):
            g = cv2.bitwise_not(gray_u8) if inv else gray_u8
            blur = cv2.GaussianBlur(g, (3, 3), 0)
            for src in (g, blur):
                _, binary = cv2.threshold(src, 0, 255, thr_bin + thr_otsu)
                binary = cv2.morphologyEx(binary, morph_open, np.ones((2, 2), np.uint8))
                raw = self._ocr_image_line(
                    binary,
                    scale=1.0,
                    allowlist=self._SCORE_OCR_ALLOWLIST,
                    uppercase=False,
                    min_confidence=0.06,
                )
                if self._parse_score_text(raw):
                    return raw
                if raw and len(raw) > len(best_txt):
                    best_txt = raw
        return best_txt

    def _read_scoreboard_score(self, frame_bgr: object) -> tuple[str, str]:
        score_crop = sb.crop_norm(frame_bgr, sb.SCORE_X0, sb.SCORE_X1, sb.TOP_Y0, sb.TOP_Y1)
        if score_crop.size == 0:
            return "-", "-"
        for slab in ocr_engine.scoreboard_strip_variants(score_crop):
            txt = self._ocr_score_variants(slab)
            parsed = self._parse_score_text(txt)
            if parsed:
                return parsed
        return "-", "-"

    def _ocr_team_code_from_crop(self, bgr: object) -> str:
        try:
            import cv2  # type: ignore
        except ImportError:
            return ""
        if bgr is None or bgr.size == 0:
            return ""
        best = ""
        for slab in ocr_engine.scoreboard_strip_variants(bgr):
            for mc in (0.1, 0.06):
                raw = self._ocr_image_line(
                    slab,
                    scale=1.0,
                    allowlist=self._TEAM_OCR_ALLOWLIST,
                    min_confidence=mc,
                )
                tok = self._normalize_team_ocr(raw)
                if 2 <= len(tok) <= 5:
                    return tok
                if len(tok) > len(best):
                    best = tok

        thr_bin = _cv2_const(cv2, "THRESH_BINARY", _CV2_THRESH_BINARY)
        thr_otsu = _cv2_const(cv2, "THRESH_OTSU", 8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.8, fy=2.8, interpolation=cv2.INTER_CUBIC)
        inv = cv2.bitwise_not(gray)
        candidates: list[object] = []
        _, o1 = cv2.threshold(gray, 0, 255, thr_bin + thr_otsu)
        candidates.append(o1)
        _, o2 = cv2.threshold(inv, 0, 255, thr_bin + thr_otsu)
        candidates.append(o2)
        candidates.append(ocr_engine.adaptive_threshold_gaussian_binary(gray))
        for bin_img in candidates:
            for mc in (0.1, 0.06):
                raw = self._ocr_image_line(
                    bin_img,
                    scale=1.0,
                    allowlist=self._TEAM_OCR_ALLOWLIST,
                    min_confidence=mc,
                )
                tok = self._normalize_team_ocr(raw)
                if 2 <= len(tok) <= 5:
                    return tok
                if len(tok) > len(best):
                    best = tok
        return self._normalize_team_ocr(best) if best else ""

    def _read_scoreboard_teams(self, frame_bgr: object) -> tuple[str, str, str]:
        import importlib.util

        if importlib.util.find_spec("cv2") is None:
            return "", "", "ocr_unavailable"

        left_img = sb.crop_norm(frame_bgr, sb.HOME_NAME_X0, sb.HOME_NAME_X1, sb.TOP_Y0, sb.TOP_Y1)
        right_img = sb.crop_norm(frame_bgr, sb.AWAY_NAME_X0, sb.AWAY_NAME_X1, sb.TOP_Y0, sb.TOP_Y1)
        if left_img.size == 0 or right_img.size == 0:
            return "", "", "scoreboard_empty"

        team_left = self._ocr_team_code_from_crop(left_img)
        team_right = self._ocr_team_code_from_crop(right_img)

        if not team_left and not team_right:
            return "", "", "scoreboard_ocr_error"
        if not team_left or not team_right:
            return team_left, team_right, "scoreboard_partial"
        return team_left, team_right, "scoreboard_ocr"

    def _read_player_rows(self, frame_bgr: object) -> List[PlayerInfo]:
        if frame_bgr is None or getattr(frame_bgr, "ndim", 0) < 2:
            return []
        h, w = frame_bgr.shape[:2]
        regions = [
            # Formation+table screens from your FM26 screenshots.
            (int(h * 0.20), int(h * 0.92), int(w * 0.36), int(w * 0.98)),
            (int(h * 0.56), int(h * 0.98), int(w * 0.48), int(w * 0.98)),
        ]
        text_blocks: list[str] = []
        for y0, y1, x0, x1 in regions:
            area = frame_bgr[y0:y1, x0:x1]
            if area.size == 0:
                continue
            block = ocr_engine.read_text_block(area, scale=1.85)
            if block:
                text_blocks.append(block)
        if not text_blocks:
            return []

        text = "\n".join(text_blocks)
        players: List[PlayerInfo] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if len(line) < 4:
                continue
            pos_match = self.POS_RE.search(line)
            if not pos_match:
                continue
            pos = pos_match.group(1).upper()
            # keep likely name segment after position markers/tokens
            cleaned = re.sub(r"[^A-Za-z .'-]", " ", line)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if len(cleaned) < 3:
                continue
            players.append(PlayerInfo(name=cleaned[:28], position=pos))
            if len(players) >= 11:
                break
        return players

