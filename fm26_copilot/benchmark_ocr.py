"""
Compare two OCR backends for course reporting / evaluation (Phase 1.a).

Primary (default): EasyOCR (deep learning).
Alternative: Tesseract (classical OCR) — install: pip install pytesseract + Tesseract binary.

Usage (from repo root)::

    py -3.12 -m pip install pytesseract
    py -3.12 -m fm26_copilot.benchmark_ocr --images E:\\dbg --labels E:\\dbg\\labels.csv --csv-out E:\\dbg\\ocr_metrics.csv

labels.csv: two columns without header — filename,expected_text
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path


def _normalize(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _easyocr_read(bgr: object) -> str:
    from . import ocr_engine

    return ocr_engine.read_text_line(bgr, scale=2.5, uppercase=True, min_confidence=0.06)


def _tesseract_read(bgr: object) -> str:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    try:
        import pytesseract  # type: ignore
    except ImportError:
        return ""
    gray = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    try:
        return pytesseract.image_to_string(gray, config="--psm 7").upper().strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark EasyOCR vs Tesseract on image crops")
    parser.add_argument("--images", type=Path, required=True, help="Directory with PNG/JPG crops")
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="CSV: filename,expected_text (no header) for accuracy",
    )
    parser.add_argument("--csv-out", type=Path, default=None, help="Write per-image metrics CSV")
    args = parser.parse_args()

    try:
        import cv2  # type: ignore
    except ImportError:
        print("opencv required", file=sys.stderr)
        sys.exit(1)

    labels: dict[str, str] = {}
    if args.labels and args.labels.is_file():
        with args.labels.open(encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    labels[row[0].strip()] = row[1].strip()

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in exts)
    if not files:
        print(f"No images in {args.images}", file=sys.stderr)
        sys.exit(2)

    rows_out: list[dict[str, str | float]] = []
    te_ok = te_n = 0
    ez_ok = ez_n = 0
    te_time = 0.0
    ez_time = 0.0

    for p in files:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        t0 = time.perf_counter()
        ez = _easyocr_read(bgr)
        ez_time += time.perf_counter() - t0
        t0 = time.perf_counter()
        te = _tesseract_read(bgr)
        te_time += time.perf_counter() - t0

        exp = labels.get(p.name)
        ez_match = ""
        te_match = ""
        if exp is not None:
            ez_n += 1
            te_n += 1
            if _normalize(ez) == _normalize(exp):
                ez_ok += 1
                ez_match = "1"
            else:
                ez_match = "0"
            if _normalize(te) == _normalize(exp):
                te_ok += 1
                te_match = "1"
            else:
                te_match = "0"

        rows_out.append(
            {
                "file": p.name,
                "expected": exp or "",
                "easyocr_text": ez,
                "tesseract_text": te,
                "easyocr_correct": ez_match,
                "tesseract_correct": te_match,
            }
        )

    n = len(rows_out)
    print(f"Images: {n}")
    print(f"EasyOCR  total time: {ez_time:.3f}s  mean: {ez_time / max(n,1):.4f}s/img")
    print(f"Tesseract total time: {te_time:.3f}s  mean: {te_time / max(n,1):.4f}s/img")
    if ez_n:
        print(f"EasyOCR  accuracy (exact norm match): {ez_ok}/{ez_n} = {100.0*ez_ok/ez_n:.1f}%")
    if te_n:
        print(f"Tesseract accuracy (exact norm match): {te_ok}/{te_n} = {100.0*te_ok/te_n:.1f}%")
    if te_n == 0 and not labels:
        print("(Pass --labels CSV to compute accuracy.)")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "file",
                    "expected",
                    "easyocr_text",
                    "tesseract_text",
                    "easyocr_correct",
                    "tesseract_correct",
                ],
            )
            w.writeheader()
            w.writerows(rows_out)
        print(f"Wrote {args.csv_out}")


if __name__ == "__main__":
    main()
