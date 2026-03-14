"""Fixed-point analysis view.

Split view showing a textual stability summary and a plot of critical
exponents from parameter continuation.
"""

from __future__ import annotations

import logging

import numpy as np

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from asymsafety.gui.state import AppState

logger = logging.getLogger(__name__)


class FPAnalysisView(QWidget):
    """Fixed-point analysis tab.

    Upper half: read-only monospace text showing
    :meth:`StabilityAnalysis.summary` for the currently selected fixed
    point.

    Lower half: matplotlib canvas plotting critical exponents when
    continuation data is available.
    """

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state

        # --- layout ---
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: stability summary text
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._text.setFont(mono)
        self._text.setPlaceholderText("Select a fixed point to see stability analysis.")
        splitter.addWidget(self._text)

        # Bottom: critical exponents plot
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self._figure = Figure(figsize=(10, 6), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, bottom_widget)
        bottom_layout.addWidget(self._toolbar)
        bottom_layout.addWidget(self._canvas, stretch=1)

        splitter.addWidget(bottom_widget)

        # Give each side roughly equal space
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # --- signals ---
        self._state.stability_changed.connect(self._on_stability_changed)
        self._state.continuation_changed.connect(self._on_continuation_changed)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_stability_changed(self, sa: object) -> None:
        """Update the text panel with the stability summary."""
        if sa is None:
            self._text.clear()
            self._text.setPlaceholderText(
                "Select a fixed point to see stability analysis."
            )
            return

        try:
            self._text.setPlainText(sa.summary())
        except Exception:
            logger.exception("Error generating stability summary")
            self._text.setPlainText("Error: could not generate summary (see log)")

    def _on_continuation_changed(self, continuation: object) -> None:
        """Redraw the critical exponents plot."""
        self._figure.clear()

        if continuation is None:
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "No continuation data",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=14, color="gray",
            )
            self._canvas.draw()
            return

        try:
            self._render_critical_exponents(continuation)
        except Exception:
            logger.exception("Error plotting critical exponents")
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "Plot error (see log)",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=14, color="red",
            )

        self._canvas.draw()

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render_critical_exponents(self, continuation: object) -> None:
        """Render critical exponents onto *self._figure*.

        Mirrors :func:`plot_critical_exponents` but draws directly onto
        the embedded figure to avoid cross-figure artist issues.
        """
        import matplotlib.pyplot as plt

        ax1, ax2 = self._figure.subplots(2, 1, sharex=True)

        thetas = continuation.critical_exponents_array
        param = continuation.parameter_values

        n_exponents = thetas.shape[1]
        colors = plt.cm.tab10(np.linspace(0, 1, n_exponents))

        for i in range(n_exponents):
            ax1.plot(
                param, thetas[:, i].real,
                "-o", color=colors[i],
                markersize=3, label=f"$\\theta_{{{i + 1}}}$",
            )
            ax2.plot(
                param, thetas[:, i].imag,
                "-o", color=colors[i],
                markersize=3,
            )

        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        ax1.set_ylabel(r"Re($\theta_i$)", fontsize=12)
        ax2.set_ylabel(r"Im($\theta_i$)", fontsize=12)
        ax2.set_xlabel(continuation.parameter_name, fontsize=12)
        ax1.set_title("Critical Exponents", fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)

        self._figure.tight_layout()
