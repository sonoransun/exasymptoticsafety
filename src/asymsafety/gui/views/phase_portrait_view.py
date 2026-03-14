"""2D phase portrait view for RG flows.

Embeds a matplotlib canvas showing a streamplot of the RG flow in a
user-selected 2D slice of coupling space.  Combo boxes let the user
choose which two couplings to project onto.
"""

from __future__ import annotations

import logging

import numpy as np

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from asymsafety.gui.state import AppState

logger = logging.getLogger(__name__)


class PhasePortraitView(QWidget):
    """2D phase-portrait tab.

    Shows a streamplot of beta-function flow projected onto a pair of
    couplings chosen via combo boxes.  Redraws automatically when the
    system, fixed points, or trajectories change.
    """

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state

        # --- layout ---
        layout = QVBoxLayout(self)

        # Controls row
        controls = QHBoxLayout()
        controls.addWidget(QLabel("x coupling:"))
        self._x_combo = QComboBox()
        controls.addWidget(self._x_combo)
        controls.addWidget(QLabel("y coupling:"))
        self._y_combo = QComboBox()
        controls.addWidget(self._y_combo)
        controls.addStretch()
        layout.addLayout(controls)

        # Matplotlib canvas
        self._figure = Figure(figsize=(8, 6), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, stretch=1)

        # --- signals ---
        self._state.system_changed.connect(self._on_system_changed)
        self._state.fixed_points_changed.connect(self._redraw)
        self._state.trajectories_changed.connect(self._redraw)
        self._x_combo.currentIndexChanged.connect(self._redraw)
        self._y_combo.currentIndexChanged.connect(self._redraw)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_system_changed(self, system: object) -> None:
        """Repopulate combo boxes when the system changes."""
        self._x_combo.blockSignals(True)
        self._y_combo.blockSignals(True)
        self._x_combo.clear()
        self._y_combo.clear()

        if system is not None:
            names = system.coupling_names
            self._x_combo.addItems(names)
            self._y_combo.addItems(names)
            if len(names) >= 2:
                self._x_combo.setCurrentIndex(0)
                self._y_combo.setCurrentIndex(1)
            elif len(names) == 1:
                self._x_combo.setCurrentIndex(0)
                self._y_combo.setCurrentIndex(0)

        self._x_combo.blockSignals(False)
        self._y_combo.blockSignals(False)
        self._redraw()

    def _redraw(self, _unused: object = None) -> None:
        """Clear and redraw the phase portrait."""
        self._figure.clear()

        system = self._state.system
        if system is None or self._x_combo.count() == 0:
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "No system loaded",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=14, color="gray",
            )
            self._canvas.draw()
            return

        x_coupling = self._x_combo.currentText()
        y_coupling = self._y_combo.currentText()
        if not x_coupling or not y_coupling:
            self._canvas.draw()
            return

        try:
            self._render_portrait(x_coupling, y_coupling)
        except Exception:
            logger.exception("Error drawing phase portrait")
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

    def _render_portrait(self, x_coupling: str, y_coupling: str) -> None:
        """Render the phase portrait directly onto *self._figure*.

        We render inline rather than calling ``phase_portrait_2d``
        because that helper creates its own ``Figure`` via ``plt.subplots``,
        and transferring artists between figures is fragile.  The logic
        here mirrors the helper so output is identical.
        """
        system = self._state.system
        ax = self._figure.add_subplot(111)
        n_grid = 25
        x_range = (-0.5, 1.5)
        y_range = (-0.5, 0.5)

        x = np.linspace(x_range[0], x_range[1], n_grid)
        y = np.linspace(y_range[0], y_range[1], n_grid)
        X, Y = np.meshgrid(x, y)

        rhs = system.rhs_vector()
        names = system.coupling_names
        x_idx = names.index(x_coupling)
        y_idx = names.index(y_coupling)

        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(n_grid):
            for j in range(n_grid):
                point = np.zeros(len(names))
                point[x_idx] = X[i, j]
                point[y_idx] = Y[i, j]
                try:
                    beta_vals = rhs(0, point)
                    U[i, j] = beta_vals[x_idx]
                    V[i, j] = beta_vals[y_idx]
                except (ValueError, ZeroDivisionError, OverflowError):
                    U[i, j] = 0
                    V[i, j] = 0

        speed = np.sqrt(U**2 + V**2)
        speed = np.where(speed == 0, 1, speed)

        ax.streamplot(
            X, Y, U, V,
            color=np.log1p(speed), cmap="coolwarm",
            density=1.5, linewidth=0.8, arrowsize=1.2,
        )

        # Fixed points
        fps = self._state.fixed_points
        if fps:
            for fp in fps:
                x_val = fp.location.get(x_coupling, 0)
                y_val = fp.location.get(y_coupling, 0)
                if fp.is_gaussian:
                    marker, color, label = "o", "white", "GFP"
                else:
                    marker = "*"
                    color = "yellow"
                    label = f"NGFP ({fp.relevant_directions} rel.)"
                ax.plot(
                    x_val, y_val,
                    marker=marker, color=color,
                    markersize=12, markeredgecolor="black",
                    markeredgewidth=1.5, label=label, zorder=5,
                )

        # Trajectories
        trajs = self._state.trajectories
        if trajs:
            for traj in trajs:
                xv = traj.coupling_values.get(x_coupling)
                yv = traj.coupling_values.get(y_coupling)
                if xv is not None and yv is not None:
                    ax.plot(xv, yv, "k-", linewidth=1.5, alpha=0.7)

        ax.set_xlabel(f"${x_coupling}$", fontsize=14)
        ax.set_ylabel(f"${y_coupling}$", fontsize=14)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_title("RG Flow Phase Portrait", fontsize=14)
        if fps:
            ax.legend(fontsize=10)
        self._figure.tight_layout()
