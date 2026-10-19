# Model / release notes (scaffold)

This file describes a **target** promotion checklist for the co-pilot scaffold.
There is **no trained model bundle or offline eval harness wired in CI yet**.

## Current reality

- Detector: OpenCV blobs (+ optional ONNX session load as a health probe).
- Recommendations: fixed rule thresholds in `recommendations.py`.
- Telemetry / drift / rollout: in-process helpers used at end of a run; not a deploy pipeline.

## Intended gates (when real models exist)

1. Offline replay on tagged scenarios passes thresholds.
2. Shadow run with cards suppressed.
3. Safety: min FPS, fail-safe ratio ceiling, quality floor from **real** accept/dismiss feedback.

## Rollback (intended)

- Revert to last known-good config when drift trips repeatedly.
- Keep a default heuristic path that does not require ONNX.
