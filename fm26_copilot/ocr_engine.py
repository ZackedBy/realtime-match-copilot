from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

_reader_lock = threading.Lock()
_readers: dict[tuple[str, ...], Any] = {}


def default_easyocr_langs() -> list[str]:
    """Latin + Cyrillic for RU client UI; override with env ``FM26_EASYOCR_LANGS=en,ru``."""
    raw = os.environ.get("FM26_EASYOCR_LANGS", "en,ru")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _cv2_const(cv_mod: object, name: str, fallback: int) -> int:
    return int(getattr(cv_mod, name, fallback))


def preprocess_for_ocr(
    frame_bgr_or_gray: object,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """BGR or single-channel image → grayscale ``uint8``, optional upscale for small UI text."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return np.zeros((1, 1), dtype=np.uint8)

    if frame_bgr_or_gray is None or getattr(frame_bgr_or_gray, "size", 0) == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    img = np.asarray(frame_bgr_or_gray)
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.ascontiguousarray(img, dtype=np.uint8)

    if scale and abs(scale - 1.0) > 1e-6:
        inter = _cv2_const(cv2, "INTER_CUBIC", 3)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=inter)

    return gray


def scoreboard_strip_variants(bgr: object, *, scale: float = 3.2) -> list[np.ndarray]:
    """Several single-channel views of a tight scoreboard crop (names / score).

    FM26 uses bright glyphs on solid red/blue/purple bars — distance from the dominant
    background colour and V-channel Otsu usually isolate text better than raw greyscale.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        return []

    if bgr is None or getattr(bgr, "size", 0) == 0:
        return []

    b = np.ascontiguousarray(np.asarray(bgr), dtype=np.uint8)
    inter = _cv2_const(cv2, "INTER_CUBIC", 3)
    thr_bin = _cv2_const(cv2, "THRESH_BINARY", 0)
    thr_otsu = _cv2_const(cv2, "THRESH_OTSU", 8)

    if b.ndim == 2:
        big = cv2.resize(b, None, fx=scale, fy=scale, interpolation=inter)
        return [big]

    big = cv2.resize(b, None, fx=scale, fy=scale, interpolation=inter)
    out: list[np.ndarray] = []

    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    out.append(gray)

    sharp_k = np.array(
        [[-1.0, -1.0, -1.0], [-1.0, 9.0, -1.0], [-1.0, -1.0, -1.0]],
        dtype=np.float32,
    )
    cv_32f = int(getattr(cv2, "CV_32F", 5))
    sharp = cv2.filter2D(gray, cv_32f, sharp_k)
    out.append(np.clip(sharp, 0, 255).astype(np.uint8))

    flat = big.reshape(-1, 3).astype(np.float32)
    med = np.median(flat, axis=0)
    dist = np.linalg.norm(big.astype(np.float32) - med.reshape(1, 1, 3), axis=2)
    dmax = float(np.percentile(dist, 99.0)) + 1e-6
    dist_n = np.clip(dist * (255.0 / dmax), 0, 255).astype(np.uint8)
    _, d_bin = cv2.threshold(dist_n, 0, 255, thr_bin + thr_otsu)
    out.append(d_bin)
    _, d_bin_inv = cv2.threshold((255 - dist_n).astype(np.uint8), 0, 255, thr_bin + thr_otsu)
    out.append(d_bin_inv)

    hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    _, v_bin = cv2.threshold(v, 0, 255, thr_bin + thr_otsu)
    out.append(v_bin)

    k = max(3, int(round(big.shape[0] * 0.12)) | 1)
    rect = int(getattr(cv2, "MORPH_RECT", 0))
    tophat_op = int(getattr(cv2, "MORPH_TOPHAT", 5))
    kernel = cv2.getStructuringElement(rect, (k, k))
    tophat = cv2.morphologyEx(gray, tophat_op, kernel)
    _, th_bin = cv2.threshold(tophat, 0, 255, thr_bin + thr_otsu)
    out.append(th_bin)

    out.append(cv2.bitwise_not(gray))

    # Light / high-key skin: white boxes on grey — low local contrast
    gmean = float(np.mean(gray))
    if gmean > 115:
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))
        out.append(clahe.apply(gray))
        gx = cv2.Sobel(gray, cv_32f, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv_32f, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mp = float(np.percentile(mag, 98.5)) + 1e-6
        mag_u8 = np.clip(mag * (255.0 / mp), 0, 255).astype(np.uint8)
        _, e_bin = cv2.threshold(mag_u8, 0, 255, thr_bin + thr_otsu)
        out.append(e_bin)
        lab = cv2.cvtColor(big, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0]
        _, l_bin = cv2.threshold(l_ch, 0, 255, thr_bin + thr_otsu)
        out.append(l_bin)

    return out


def adaptive_threshold_gaussian_binary(
    gray_u8: object,
    *,
    block_size: int = 25,
    c: int = 7,
) -> np.ndarray:
    """Adaptive Gaussian → binary image.

    Uses numeric OpenCV enums (Gaussian ``C`` = 1, ``THRESH_BINARY`` = 0) so a stripped
    ``cv2`` module without ``ADAPTIVE_GAUSSIAN_C`` / ``THRESH_BINARY`` attributes still works.
    """
    import cv2  # type: ignore

    g = np.ascontiguousarray(np.asarray(gray_u8), dtype=np.uint8)
    return cv2.adaptiveThreshold(g, 255, 1, 0, block_size, c)


def get_easyocr_reader(langs: list[str] | None = None, gpu: bool = False) -> Any:
    if langs is None:
        langs = default_easyocr_langs()
    key = tuple(langs)
    with _reader_lock:
        if key not in _readers:
            import easyocr  # type: ignore

            _readers[key] = easyocr.Reader(list(key), gpu=gpu, verbose=False)
        return _readers[key]


def read_text_line(
    image_bgr_gray_or_binary: object,
    *,
    scale: float = 2.8,
    min_confidence: float = 0.12,
    uppercase: bool = False,
    allowlist: str | None = None,
) -> str:
    """Single-line style crop: left-to-right box order, joined with spaces."""
    import importlib.util

    if importlib.util.find_spec("easyocr") is None:
        return ""

    gray = preprocess_for_ocr(image_bgr_gray_or_binary, scale=scale)
    if gray.size < 4:
        return ""

    reader = get_easyocr_reader()
    try:
        with _reader_lock:
            kw: dict[str, Any] = {"detail": 1, "paragraph": False}
            if allowlist:
                kw["allowlist"] = allowlist
            raw = reader.readtext(gray, **kw)
    except Exception:
        return ""

    parts: list[tuple[float, str]] = []
    for item in raw:
        if len(item) < 3:
            continue
        bbox, text, conf = item[0], item[1], float(item[2])
        if conf < min_confidence or not (text and str(text).strip()):
            continue
        xs = [float(p[0]) for p in bbox]
        cx = sum(xs) / max(len(xs), 1)
        parts.append((cx, str(text).strip()))

    parts.sort(key=lambda t: t[0])
    if not parts and raw:
        best_txt, best_c = "", -1.0
        for item in raw:
            if len(item) < 3:
                continue
            t, c = str(item[1]).strip(), float(item[2])
            if t and c > best_c:
                best_txt, best_c = t, c
        if best_txt and best_c >= 0.03:
            parts.append((0.0, best_txt))

    out = " ".join(t[1] for t in parts)
    if uppercase:
        out = out.upper()
    return out.strip()


def read_text_block(
    frame_roi_bgr: object,
    *,
    scale: float = 1.85,
    min_confidence: float = 0.18,
) -> str:
    """Larger ROI (tables / squad lists): paragraph grouping, newlines between groups."""
    import importlib.util

    if importlib.util.find_spec("easyocr") is None:
        return ""

    gray = preprocess_for_ocr(frame_roi_bgr, scale=scale)
    if gray.size < 4:
        return ""

    reader = get_easyocr_reader()
    try:
        with _reader_lock:
            res = reader.readtext(gray, detail=0, paragraph=True)
    except Exception:
        return ""

    if isinstance(res, str):
        return res.strip()
    if isinstance(res, list):
        lines: list[str] = []
        for item in res:
            if isinstance(item, str):
                t = item.strip()
            elif isinstance(item, (list, tuple)) and item:
                t = str(item[0]).strip()
            else:
                t = str(item).strip()
            if t:
                lines.append(t)
        return "\n".join(lines)
    return str(res).strip()
