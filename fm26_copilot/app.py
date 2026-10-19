from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass

from .capture import ScreenCapture, ScreenCaptureConfig
from .contracts import FrameSignals, TelemetrySample, TeamSide
from . import fm26_scoreboard_rois as scoreboard_rois
from .feedback import FeedbackStore
from .overlay import OverlayController
from .match_intel import MatchIntelExtractor
from .recommendations import RecommendationEngine
from .reliability import DriftDetector, ReleasePolicy
from .rollout import DEFAULT_WAVES
from .safety import SafetyAndConfidenceGate
from .scoreboard import ScoreboardSideResolver
from .tactical_state import TacticalStateEngine
from .telemetry import TelemetryCollector
from .team_context import TeamContext
from .vision import VisionStack


@dataclass
class CopilotRuntimeConfig:
    loop_seconds: int = 0
    target_fps: int = 20
    simulation_mode: bool = False
    use_qt_overlay: bool = True
    onnx_model_path: str | None = None
    monitor_index: int = 0
    managed_team_code: str = "ASM"
    managed_team_side: TeamSide = TeamSide.HOME
    auto_team_side: bool = True
    diagnostics_mode: bool = False
    use_ocr: bool = True
    side_switch_cooldown_sec: float = 18.0
    chip_min_margin: float = 4.0
    intel_use_ocr: bool = True
    intel_players: bool = False


class CopilotApp:
    def __init__(self, cfg: CopilotRuntimeConfig):
        self.cfg = cfg
        self.team_context = TeamContext(
            managed_team_code=cfg.managed_team_code,
            managed_team_side=cfg.managed_team_side,
            auto_side_enabled=cfg.auto_team_side,
        )
        self.capture = ScreenCapture(
            ScreenCaptureConfig(
                target_fps=cfg.target_fps,
                simulation_mode=cfg.simulation_mode,
                monitor_index=cfg.monitor_index,
            )
        )
        self.vision = VisionStack(
            onnx_model_path=cfg.onnx_model_path,
            managed_team_code=cfg.managed_team_code,
        )
        self.tactical = TacticalStateEngine()
        self.recommendations = RecommendationEngine(managed_team=self.team_context.managed_team_side)
        self.safety = SafetyAndConfidenceGate()
        self.overlay = OverlayController(use_qt_overlay=cfg.use_qt_overlay)
        self.feedback = FeedbackStore()
        self.telemetry = TelemetryCollector()
        self.drift_detector = DriftDetector()
        self.release_policy = ReleasePolicy()
        self.side_resolver = ScoreboardSideResolver(use_ocr=cfg.use_ocr)
        self.match_intel = MatchIntelExtractor(use_ocr=cfg.intel_use_ocr)
        self._last_side_update_ts = 0.0
        self._side_candidate: TeamSide | None = None
        self._side_candidate_hits = 0
        self._last_side_result_reason = "init"
        self._last_side_result_conf = 0.0
        self._last_diag_emit_ts = 0.0
        self._last_intel_update_ts = 0.0
        self._last_opponent_name = "unknown"
        self._last_opponent_players: list[str] = []
        self._last_intel_reason = "init"
        self._last_side_switch_ts = 0.0
        self._last_score_home = "-"
        self._last_score_away = "-"
        self._last_home_color = "unknown"
        self._last_away_color = "unknown"
        self._last_home_code = ""
        self._last_away_code = ""

    def run(self) -> None:
        start = time.time()
        try:
            for frame in self.capture.frames():
                tick_start = time.perf_counter()
                self._update_managed_side_if_needed(frame)

                vision_out = self.vision.process(frame)
                t1 = time.perf_counter()
                tactical_state = self.tactical.infer(vision_out)
                t2 = time.perf_counter()
                recs = self.recommendations.generate(tactical_state)
                t3 = time.perf_counter()
                tracker_ok = bool(vision_out.home_players or vision_out.away_players)
                cards, fail_safe = self.safety.filter(
                    recs=recs,
                    vision_confidence=vision_out.confidence,
                    tracker_ok=tracker_ok,
                )
                render_cards = self.overlay.renderable_cards(cards)
                self.overlay.present(render_cards)
                t4 = time.perf_counter()
                # After frame work so OCR time isn’t folded into vision latency / FPS proxy
                self._update_match_intel_if_needed(frame)

                # Do not auto-accept cards: that gamed the quality proxy / rollout gate.
                # Call FeedbackStore.mark_accepted / mark_dismissed from real UI hooks later.

                sample = TelemetrySample(
                    stage_latencies_ms={
                        "vision": (t1 - tick_start) * 1000,
                        "tactical": (t2 - t1) * 1000,
                        "recommendation": (t3 - t2) * 1000,
                        "overlay": (t4 - t3) * 1000,
                    },
                    pipeline_fps=max(0.1, 1.0 / max(0.0001, (t4 - tick_start))),
                    model_confidence={
                        "vision": vision_out.confidence,
                        "pitch_calibration": vision_out.calibration_confidence,
                    },
                    recommendation_quality_proxy=self.feedback.quality_proxy(),
                    dropped_frames=0,
                    fail_safe_active=fail_safe,
                )
                self.telemetry.push(sample)
                self.drift_detector.update(sample.recommendation_quality_proxy)
                self._emit_diagnostics_if_needed(sample)

                if self.cfg.loop_seconds > 0 and (time.time() - start > self.cfg.loop_seconds):
                    break
        except KeyboardInterrupt:
            print("Interrupted by user, stopping copilot...")

        safe, reason = self.release_policy.is_rollout_safe(self.telemetry)
        print(f"Rollout gate: {'PASS' if safe else 'HOLD'} | {reason}")
        print("Telemetry snapshot:", self.telemetry.snapshot())
        if safe:
            print("Rollout plan:")
            for wave in DEFAULT_WAVES:
                print(f" - {wave.name}: {wave.traffic_pct}% traffic | scenarios={','.join(wave.scenarios)}")

    def _update_managed_side_if_needed(self, frame: FrameSignals) -> None:
        if not self.team_context.auto_side_enabled:
            return
        if frame.raw_frame is None:
            return
        now = time.time()
        if now - self._last_side_update_ts < 1.5:
            return
        try:
            result = self.side_resolver.resolve(frame.raw_frame, self.team_context.managed_team_code)
        except Exception as exc:
            self._last_side_result_reason = f"resolver_error:{type(exc).__name__}"
            self._last_side_result_conf = 0.0
            self._last_side_update_ts = now
            return
        self._last_side_result_reason = result.reason
        self._last_side_result_conf = result.confidence
        if result.side is not None:
            # Ignore weak chip-based detections where colors are nearly equal.
            if result.reason.startswith("chip_score:"):
                try:
                    scores = result.reason.split(":", 1)[1]
                    left_s, right_s = scores.split("/")
                    margin = abs(float(left_s) - float(right_s))
                    if margin < self.cfg.chip_min_margin:
                        self._last_side_result_reason = f"chip_ambiguous_margin:{margin:.2f}"
                        self._last_side_update_ts = now
                        return
                except Exception:
                    pass

            if result.side == self._side_candidate:
                self._side_candidate_hits += 1
            else:
                self._side_candidate = result.side
                self._side_candidate_hits = 1

            high_conf_accept = result.confidence >= 0.55
            low_conf_stable_accept = result.confidence >= 0.30 and self._side_candidate_hits >= 3
            cooldown_ok = (now - self._last_side_switch_ts) >= self.cfg.side_switch_cooldown_sec
            if (high_conf_accept or low_conf_stable_accept) and result.side != self.team_context.managed_team_side and cooldown_ok:
                self.team_context.managed_team_side = result.side
                self.recommendations.managed_team = result.side
                self._last_side_switch_ts = now
                print(
                    f"[TeamSide] Auto-detected {result.side.value} for "
                    f"{self.team_context.managed_team_code} ({result.reason}, hits={self._side_candidate_hits})"
                )
        self._last_side_update_ts = now

    def _emit_diagnostics_if_needed(self, sample: TelemetrySample) -> None:
        if not self.cfg.diagnostics_mode:
            return
        now = time.time()
        if now - self._last_diag_emit_ts < 2.0:
            return
        candidate = self._side_candidate.value if self._side_candidate is not None else "none"
        managed = self.team_context.managed_team_side.value
        lines = [
            f"team={self.team_context.managed_team_code}",
            f"managed_side={managed}",
            f"opponent_team={self._last_opponent_name}",
            f"score={self._last_score_home}-{self._last_score_away}",
            f"scoreboard_teams={self._last_home_code or '?'} vs {self._last_away_code or '?'}",
            f"team_colors=home:{self._last_home_color} away:{self._last_away_color}",
            f"candidate_side={candidate}",
            f"candidate_hits={self._side_candidate_hits}",
            f"side_confidence={self._last_side_result_conf:.2f}",
            f"side_reason={self._last_side_result_reason}",
            f"vision_conf={sample.model_confidence.get('vision', 0.0):.2f}",
            f"calibration_conf={sample.model_confidence.get('pitch_calibration', 0.0):.2f}",
            f"fps={sample.pipeline_fps:.1f}",
            f"intel_reason={self._last_intel_reason}",
        ]
        if self._last_opponent_players:
            lines.append("opponent_players:")
            lines.extend(self._last_opponent_players[:11])
        else:
            lines.append("opponent_players: unavailable")
        self.overlay.present_status(lines)
        self._last_diag_emit_ts = now

    def _update_match_intel_if_needed(self, frame: FrameSignals) -> None:
        if not self.cfg.diagnostics_mode:
            return
        if frame.raw_frame is None:
            return
        now = time.time()
        if now - self._last_intel_update_ts < 3.0:
            return
        snap = self.match_intel.extract(
            frame_bgr=frame.raw_frame,
            managed_team_code=self.team_context.managed_team_code,
            managed_side=self.team_context.managed_team_side,
            include_players=self.cfg.intel_players,
        )
        self._last_intel_reason = snap.reason
        if snap.score_home != "-":
            self._last_score_home = snap.score_home
        if snap.score_away != "-":
            self._last_score_away = snap.score_away

        def _kit_readable(s: str) -> bool:
            return bool(s) and "unknown" not in s.lower()

        if _kit_readable(snap.home_color):
            self._last_home_color = snap.home_color
        if _kit_readable(snap.away_color):
            self._last_away_color = snap.away_color
        if snap.home_team_code:
            self._last_home_code = snap.home_team_code
        if snap.away_team_code:
            self._last_away_code = snap.away_team_code
        # Sticky cache: keep previously valid opponent/team list if current frame is unreadable.
        if snap.opponent_team and snap.opponent_team != "unknown":
            self._last_opponent_name = snap.opponent_team
        if snap.opponent_players:
            self._last_opponent_players = [f"- {p.position}: {p.name}" for p in snap.opponent_players]
        self._last_intel_update_ts = now


def main() -> None:
    if sys.version_info[:2] >= (3, 13):
        print(
            "fm26_copilot: on Python 3.13+ prefer 3.11–3.12 for prebuilt wheels:\n"
            "  python3.12 -m pip install -e .\n"
            "  python3.12 -m fm26_copilot.app ...",
            file=sys.stderr,
        )
    warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

    parser = argparse.ArgumentParser(description="FM26 AI Co-pilot")
    parser.add_argument("--sim", action="store_true", help="Use simulation capture mode")
    parser.add_argument("--no-qt", action="store_true", help="Disable PySide overlay")
    parser.add_argument("--seconds", type=int, default=0, help="Run duration in seconds (0 = run until Ctrl+C)")
    parser.add_argument("--fps", type=int, default=20, help="Capture target FPS")
    parser.add_argument("--monitor", type=int, default=0, help="Monitor index for DXGI capture")
    parser.add_argument("--onnx-model", type=str, default=None, help="Path to ONNX detector model")
    parser.add_argument("--team-code", type=str, default="ASM", help="Managed team short code")
    parser.add_argument(
        "--team-side",
        type=str,
        choices=["home", "away"],
        default="home",
        help="Managed team side on scoreboard",
    )
    parser.add_argument("--no-auto-team-side", action="store_true", help="Disable automatic team side detection")
    parser.add_argument("--diagnostics", action="store_true", help="Show runtime diagnostics in overlay/console")
    parser.add_argument("--ocr-off", action="store_true", help="Disable OCR and use color-chip side detection only")
    parser.add_argument("--side-switch-cooldown", type=float, default=18.0, help="Min seconds between side switches")
    parser.add_argument("--chip-min-margin", type=float, default=4.0, help="Min chip-score margin to trust chip side detection")
    parser.add_argument("--intel-ocr-off", action="store_true", help="Disable OCR for opponent/team/player diagnostics")
    parser.add_argument(
        "--intel-players",
        action="store_true",
        help="Run heavy EasyOCR on squad list regions (slow; off by default)",
    )
    parser.add_argument(
        "--roi-config",
        type=str,
        default=None,
        help="JSON with scoreboard ROI fractions (see fm26_copilot/scoreboard_rois.example.json)",
    )
    parser.add_argument(
        "--dump-scoreboard-crops",
        metavar="DIR",
        default=None,
        help="Grab one live frame, save scoreboard PNG crops to DIR, then exit",
    )
    args = parser.parse_args()

    if args.roi_config:
        if not scoreboard_rois.configure_from_path(args.roi_config):
            print(f"ERROR: --roi-config file not found: {args.roi_config}", file=sys.stderr)
            sys.exit(1)
    else:
        scoreboard_rois.configure_from_path(None)

    if args.dump_scoreboard_crops:
        from pathlib import Path

        from .scoreboard_debug import dump_scoreboard_crops

        cap = ScreenCapture(
            ScreenCaptureConfig(
                target_fps=max(5, args.fps),
                simulation_mode=args.sim,
                monitor_index=args.monitor,
            )
        )
        raw = None
        for _ in range(150):
            fr = next(cap.frames())
            if fr.raw_frame is not None:
                raw = fr.raw_frame
                break
        if raw is None:
            print(
                "No DXGI frame (use live mode without --sim; keep match screen visible).",
                file=sys.stderr,
            )
            sys.exit(2)
        out = Path(args.dump_scoreboard_crops)
        written = dump_scoreboard_crops(raw, out)
        print(f"Wrote {len(written)} files under {out.resolve()}:")
        for w in written:
            print(f"  {w}")
        print(
            "Use 00b_* for 0–100 ruler on strip; 00c_* full frame. "
            "Report ranges in 0..100 (divide by 100 for JSON x). "
            "Then --roi-config ..."
        )
        sys.exit(0)

    side = TeamSide.HOME if args.team_side == "home" else TeamSide.AWAY
    app = CopilotApp(
        CopilotRuntimeConfig(
            loop_seconds=args.seconds,
            target_fps=args.fps,
            simulation_mode=args.sim,
            use_qt_overlay=not args.no_qt,
            onnx_model_path=args.onnx_model,
            monitor_index=args.monitor,
            managed_team_code=args.team_code,
            managed_team_side=side,
            auto_team_side=not args.no_auto_team_side,
            diagnostics_mode=args.diagnostics,
            use_ocr=not args.ocr_off,
            side_switch_cooldown_sec=args.side_switch_cooldown,
            chip_min_margin=args.chip_min_margin,
            intel_use_ocr=not args.intel_ocr_off,
            intel_players=args.intel_players,
        )
    )
    app.run()


if __name__ == "__main__":
    main()

