from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import List

from .contracts import OverlayCard


@dataclass
class QtOverlayConfig:
    enabled: bool = True


class QtOverlaySink:
    def __init__(self, config: QtOverlayConfig):
        self.config = config
        self._rec_queue: queue.Queue[str] = queue.Queue(maxsize=8)
        self._diag_queue: queue.Queue[str] = queue.Queue(maxsize=8)
        self._started = False
        self._available = False
        if self.config.enabled:
            self._available = self._start_thread()

    def push_cards(self, cards: List[OverlayCard]) -> bool:
        if not self._available or not cards:
            return False
        text_lines = []
        for card in cards:
            rec = card.recommendation
            text_lines.append(
                f"[P{card.priority}] {rec.title}\n"
                f"impact={rec.impact:.2f} conf={rec.confidence:.2f} risk={rec.risk:.2f}\n"
                f"{rec.body}"
            )
        payload = "\n\n".join(text_lines)
        if self._rec_queue.full():
            try:
                self._rec_queue.get_nowait()
            except queue.Empty:
                pass
        self._rec_queue.put_nowait(payload)
        return True

    def push_text(self, text: str) -> bool:
        if not self._available or not text:
            return False
        if self._diag_queue.full():
            try:
                self._diag_queue.get_nowait()
            except queue.Empty:
                pass
        self._diag_queue.put_nowait(text)
        return True

    def _start_thread(self) -> bool:
        try:
            from PySide6 import QtCore, QtWidgets  # type: ignore
        except ImportError:
            print("PySide6 not installed. Falling back to console overlay.")
            return False

        if self._started:
            return True

        def run_overlay() -> None:
            app = QtWidgets.QApplication([])
            rec_window = QtWidgets.QWidget()
            rec_window.setWindowTitle("FM26 AI Copilot - Advice")
            rec_window.setWindowFlags(
                QtCore.Qt.WindowType.FramelessWindowHint
                | QtCore.Qt.WindowType.WindowStaysOnTopHint
                | QtCore.Qt.WindowType.Tool
            )
            rec_window.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
            rec_window.setStyleSheet("background: rgba(8, 12, 20, 185); border: 1px solid #4a5a7a;")
            rec_layout = QtWidgets.QVBoxLayout(rec_window)
            rec_label = QtWidgets.QLabel("FM26 Co-pilot advice")
            rec_label.setStyleSheet("color: white; font-size: 14px; padding: 10px;")
            rec_label.setWordWrap(True)
            rec_layout.addWidget(rec_label)
            rec_window.resize(560, 300)
            rec_window.move(30, 70)
            rec_window.show()

            diag_window = QtWidgets.QWidget()
            diag_window.setWindowTitle("FM26 AI Copilot - Diagnostics")
            diag_window.setWindowFlags(
                QtCore.Qt.WindowType.FramelessWindowHint
                | QtCore.Qt.WindowType.WindowStaysOnTopHint
                | QtCore.Qt.WindowType.Tool
            )
            diag_window.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
            diag_window.setStyleSheet("background: rgba(15, 18, 26, 190); border: 1px solid #5f6a82;")
            diag_layout = QtWidgets.QVBoxLayout(diag_window)
            diag_label = QtWidgets.QLabel("FM26 Co-pilot diagnostics")
            diag_label.setStyleSheet("color: #dce3f2; font-size: 13px; padding: 10px;")
            diag_label.setWordWrap(True)
            diag_layout.addWidget(diag_label)
            diag_window.resize(620, 420)
            diag_window.move(30, 390)
            diag_window.show()

            timer = QtCore.QTimer()

            def tick() -> None:
                try:
                    rec_text = self._rec_queue.get_nowait()
                except queue.Empty:
                    rec_text = None
                try:
                    diag_text = self._diag_queue.get_nowait()
                except queue.Empty:
                    diag_text = None
                if rec_text is not None:
                    rec_label.setText(rec_text)
                if diag_text is not None:
                    diag_label.setText(diag_text)

            timer.timeout.connect(tick)
            timer.start(160)
            app.exec()

        thread = threading.Thread(target=run_overlay, daemon=True)
        thread.start()
        self._started = True
        return True

