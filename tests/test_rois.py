from __future__ import annotations

import json
from pathlib import Path

import pytest

import fm26_copilot.fm26_scoreboard_rois as rois


def test_configure_from_missing_path(tmp_path: Path) -> None:
    assert rois.configure_from_path(str(tmp_path / "missing.json")) is False


def test_configure_from_example_json() -> None:
    example = Path(__file__).resolve().parents[1] / "fm26_copilot" / "scoreboard_rois.example.json"
    assert example.is_file()
    assert rois.configure_from_path(str(example)) is True
    assert 0.0 <= rois.TOP_Y0 <= 1.0


def test_configure_rejects_out_of_range(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"TOP_Y0": 1.5}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        rois.configure_from_path(str(bad))


def test_configure_rejects_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        rois.configure_from_path(str(bad))


def test_crop_norm_empty_on_inverted() -> None:
    import numpy as np

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = rois.crop_norm(frame, 0.8, 0.2, 0.1, 0.2)
    assert crop.shape[0] == 0


def test_ruler_roundtrip() -> None:
    assert rois.x_norm_to_ruler(0.5) == 50.0
    assert rois.ruler_unit_to_x_norm(50) == 0.5
