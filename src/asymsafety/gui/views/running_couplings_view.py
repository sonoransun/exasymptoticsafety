"""Running couplings view.

Shows coupling values as a function of the RG scale parameter *t*.
Automatically redraws when trajectories change.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtWidgets import QVBoxLayout, QWidget

from asymsafety.gui.state import AppState

logger = logging.getLogger(__name__)


class RunningCouplingsView(QWidget):
    """Running-couplings tab.

    Plots all coupling constants vs. the RG scale *t = log(k/k0)* for
    every trajectory stored in :class:`AppState`.  Updates automatically
    when ``trajectories_changed`` fires.
    """

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state

        # --- layout ---
        layout = QVBoxLayout(self)

        self._figure = Figure(figsize=(10, 6), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, stretch=1)

        # --- signals ---
        self._state.trajectories_changed.connect(self._redraw)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _redraw(self, trajectories: list | None = None) -> None:
        """Clear and redraw running-couplings plot."""
        self._figure.clear()

        trajs = trajectories if trajectories is not None else self._state.trajectories
        if not trajs:
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "No trajectories computed",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=14, color="gray",
            )
            self._canvas.draw()
            return

        try:
            self._render_flow_diagram(trajs)
        except Exception:
            logger.exception("Error drawing running couplings")
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

    def _render_flow_diagram(self, trajectories: list) -> None:
        """Render directly onto *self._figure*, mirroring ``flow_diagram``."""
        ax = self._figure.add_subplot(111)

        for traj in trajectories:
            for name in traj.coupling_names:
                if name in traj.coupling_values:
                    ax.plot(
                        traj.t_values,
                        traj.coupling_values[name],
                        label=f"${name}$",
                    )

        ax.set_xlabel(r"$t = \log(k/k_0)$", fontsize=14)
        ax.set_ylabel("Coupling value", fontsize=14)
        ax.set_title("Running Couplings", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        self._figure.tight_layout()
