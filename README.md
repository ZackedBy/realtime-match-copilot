# FM26 AI Co-pilot

Desktop assistant scaffold for **Football Manager 26** match screens: capture the pitch UI, estimate tactical pressure with classical CV + scoreboard OCR, and surface a few explainable coaching tips in an overlay.

This is a **working prototype / portfolio project**, not a production ML product. Vision is OpenCV heuristics (optional ONNX path is an availability probe only). Match clock, score, and possession are simulated in `--sim` mode; in live mode those fields stay neutral until wired from OCR.

## What works today

| Layer | Status |
|-------|--------|
| DXGI screen capture (Windows) | Live via `dxcam` |
| Simulation capture | Cross-platform (`--sim`) |
| Player/ball blobs + tracking | Classical CV |
| Scoreboard side / kit OCR | EasyOCR + calibrated ROIs |
| Rule-based recommendations + safety gate | Deterministic heuristics |
| Qt / console overlay | Optional |
| Feedback / drift / rollout gates | Scaffold only (not a real release pipeline) |

## Requirements

- Python **3.11+** (3.11–3.12 recommended on Windows for prebuilt wheels)
- **Live capture:** Windows + FM26 match on screen + `dxcam`
- **Everyone else:** use `--sim --no-qt`

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

EasyOCR pulls **PyTorch** and downloads models on first OCR use.

## Run (about one minute after install)

Simulation (any OS):

```bash
python -m fm26_copilot.app --sim --no-qt --seconds 5
```

Live (Windows, match visible):

```bash
python -m fm26_copilot.app --seconds 0 --fps 20 --monitor 0
```

Tests:

```bash
python -m pytest
```

## Useful flags

```bash
# Manual side / no auto switch
python -m fm26_copilot.app --sim --no-qt --no-auto-team-side --team-side home --team-code ASM

# Diagnostics + ROI calibration dump
python -m fm26_copilot.app --diagnostics --team-code ASM
python -m fm26_copilot.app --dump-scoreboard-crops ./crops
python -m fm26_copilot.app --roi-config fm26_copilot/scoreboard_rois.example.json
```

`--seconds 0` runs until `Ctrl+C`. Missing optional deps (`dxcam`, `PySide6`, …) degrade to console / simulation.

## Project layout

```
fm26_copilot/     # capture → vision → tactics → recommendations → safety → overlay
tests/            # unit tests for rules, safety, ROI, side resolve
```

## Honest limitations

- Not a trained detector: no labeled FM dataset in-repo; ONNX does not emit boxes yet.
- Overlay feedback is not collected from the user; quality proxy stays neutral unless you call the feedback API.
- `MODEL_RELEASE.md` describes a *target* release process for the scaffold, not an active MLOps pipeline.

## License

MIT — see [LICENSE](LICENSE).
