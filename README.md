# FM26 AI Co-pilot

Production-oriented scaffold for a real-time Football Manager 26 assistant:

- Screen/frame ingestion pipeline
- Vision + tracking abstractions
- Tactical state engine
- Explainable recommendation engine
- Safety/confidence gating
- Overlay-friendly UI event stream
- Feedback, drift checks, and release policy primitives

## Quick start

```bash
python -m fm26_copilot.app
```

## Modes

- Live capture (DXGI Desktop Duplication + Qt overlay):

```bash
python -m fm26_copilot.app --seconds 0 --fps 20 --monitor 0
```

- Simulation mode:

```bash
python -m fm26_copilot.app --sim --no-qt
```

## Install dependencies

```bash
python -m pip install -e .
```

EasyOCR pulls in **PyTorch** and downloads models on first OCR use. On Windows, **Python 3.11–3.12** usually gets prebuilt wheels for all transitive deps; very new Python versions may need [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if a package falls back to a source build.

## Optional ONNX model

```bash
python -m fm26_copilot.app --onnx-model models/fm26_detector.onnx
```

## Auto side detection (ASM home/away)

- Enabled by default in live mode.
- Resolver uses top scoreboard OCR first, then kit-color chips next to team names as fallback.
- Match intel and scoreboard text use **EasyOCR** (bundled via `pip install -e .`; first run downloads recognition models).
- Disable if needed:

```bash
python -m fm26_copilot.app --no-auto-team-side --team-side home
```

## Diagnostics mode

Show side-detection internals (source, confidence, stability, current managed side):

```bash
python -m fm26_copilot.app --diagnostics --team-code ASM
```

With Qt enabled, diagnostics are shown in a separate window from tactical advice.

OCR can be explicitly disabled for side detection if needed:

```bash
python -m fm26_copilot.app --diagnostics --ocr-off --team-code ASM
```

`--seconds 0` means run continuously until `Ctrl+C`.

Note: `--ocr-off` disables OCR only for side detection.  
Opponent/team/player diagnostics keep OCR enabled by default.  
Use `--intel-ocr-off` to disable diagnostic OCR too.

If `dxcam`, `opencv`, `onnxruntime`, or `PySide6` are missing, the app degrades gracefully to console/simulation behavior.
