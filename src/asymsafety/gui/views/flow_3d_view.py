"""3D flow visualization view.

Embeds a matplotlib 3D canvas showing RG flow trajectories or a quiver
phase portrait in three-dimensional coupling space.  Interactive mouse
rotation and zoom are provided by mplot3d.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from asymsafety.gui.state import AppState

logger = logging.getLogger(__name__)

# Slider range is integer; we map [1..20] -> [0.1..2.0]
_SLIDER_MIN = 1
_SLIDER_MAX = 20
_SLIDER_DEFAULT = 10


def _slider_to_scale(value: int) -> float:
    return value / 10.0


class Flow3DView(QWidget):
    """3D flow-visualization tab.

    Provides three combo boxes for selecting which couplings map to the
    x/y/z axes, check boxes to toggle fixed points, eigenvectors, and
    trajectories, a slider for eigenvector scale, and a mode selector
    for switching between trajectory lines and a quiver phase portrait.
    """

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state

        # --- layout ---
        layout = QVBoxLayout(self)

        # Row 1: coupling selectors
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("x:"))
        self._x_combo = QComboBox()
        row1.addWidget(self._x_combo)
        row1.addWidget(QLabel("y:"))
        self._y_combo = QComboBox()
        row1.addWidget(self._y_combo)
        row1.addWidget(QLabel("z:"))
        self._z_combo = QComboBox()
        row1.addWidget(self._z_combo)
        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: toggles, scale slider, viz mode
        row2 = QHBoxLayout()

        self._show_fps = QCheckBox("Show Fixed Points")
        self._show_fps.setChecked(True)
        row2.addWidget(self._show_fps)

        self._show_eigenvectors = QCheckBox("Show Eigenvectors")
        self._show_eigenvectors.setChecked(True)
        row2.addWidget(self._show_eigenvectors)

        self._show_trajectories = QCheckBox("Show Trajectories")
        self._show_trajectories.setChecked(True)
        row2.addWidget(self._show_trajectories)

        row2.addWidget(QLabel("Eigenvector scale:"))
        self._scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._scale_slider.setRange(_SLIDER_MIN, _SLIDER_MAX)
        self._scale_slider.setValue(_SLIDER_DEFAULT)
        self._scale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._scale_slider.setTickInterval(2)
        self._scale_label = QLabel(f"{_slider_to_scale(_SLIDER_DEFAULT):.1f}")
        row2.addWidget(self._scale_slider)
        row2.addWidget(self._scale_label)

        row2.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Trajectories", "Phase Portrait (quiver)"])
        row2.addWidget(self._mode_combo)
        row2.addStretch()
        layout.addLayout(row2)

        # Matplotlib 3D canvas
        self._figure = Figure(figsize=(8, 6), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, stretch=1)

        # --- signals: state ---
        self._state.system_changed.connect(self._on_system_changed)
        self._state.fixed_points_changed.connect(self._redraw)
        self._state.trajectories_changed.connect(self._redraw)
        self._state.stability_changed.connect(self._redraw)

        # --- signals: controls ---
        self._x_combo.currentIndexChanged.connect(self._redraw)
        self._y_combo.currentIndexChanged.connect(self._redraw)
        self._z_combo.currentIndexChanged.connect(self._redraw)
        self._show_fps.toggled.connect(self._redraw)
        self._show_eigenvectors.toggled.connect(self._redraw)
        self._show_trajectories.toggled.connect(self._redraw)
        self._scale_slider.valueChanged.connect(self._on_slider_changed)
        self._mode_combo.currentIndexChanged.connect(self._redraw)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_system_changed(self, system: object) -> None:
        """Repopulate coupling combo boxes."""
        for combo in (self._x_combo, self._y_combo, self._z_combo):
            combo.blockSignals(True)
            combo.clear()

        if system is not None:
            names = system.coupling_names
            for combo in (self._x_combo, self._y_combo, self._z_combo):
                combo.addItems(names)
            # Set sensible defaults
            for idx, combo in enumerate(
                (self._x_combo, self._y_combo, self._z_combo)
            ):
                if idx < len(names):
                    combo.setCurrentIndex(idx)
                else:
                    combo.setCurrentIndex(0)

        for combo in (self._x_combo, self._y_combo, self._z_combo):
            combo.blockSignals(False)
        self._redraw()

    def _on_slider_changed(self, value: int) -> None:
        self._scale_label.setText(f"{_slider_to_scale(value):.1f}")
        self._redraw()

    def _redraw(self, _unused: object = None) -> None:
        """Clear and redraw the 3D visualization."""
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

        x_name = self._x_combo.currentText()
        y_name = self._y_combo.currentText()
        z_name = self._z_combo.currentText()
        if not x_name or not y_name or not z_name:
            self._canvas.draw()
            return

        try:
            mode = self._mode_combo.currentText()
            if mode.startswith("Phase Portrait"):
                self._render_quiver(x_name, y_name, z_name)
            else:
                self._render_trajectories(x_name, y_name, z_name)
        except Exception:
            logger.exception("Error drawing 3D view")
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
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_trajectories(
        self, x_name: str, y_name: str, z_name: str
    ) -> None:
        """Draw 3D trajectories, fixed points, and eigenvectors."""
        from asymsafety.gui.visualization_3d import flow_trajectories_3d
        import matplotlib.pyplot as plt

        fps = self._state.fixed_points if self._show_fps.isChecked() else None
        trajs = (
            self._state.trajectories
            if self._show_trajectories.isChecked()
            else None
        )
        scale = _slider_to_scale(self._scale_slider.value())

        fig = flow_trajectories_3d(
            trajectories=trajs or [],
            x_coupling=x_name,
            y_coupling=y_name,
            z_coupling=z_name,
            fixed_points=fps,
            show_eigenvectors=self._show_eigenvectors.isChecked(),
            eigenvector_scale=scale,
        )

        # Transfer 3D axes content to our figure
        self._figure.clear()
        src_ax = fig.axes[0]
        dest_ax = self._figure.add_subplot(111, projection="3d")
        _copy_3d_axis(src_ax, dest_ax)
        plt.close(fig)

    def _render_quiver(
        self, x_name: str, y_name: str, z_name: str
    ) -> None:
        """Draw a 3D quiver phase portrait."""
        from asymsafety.gui.visualization_3d import phase_portrait_3d
        import matplotlib.pyplot as plt

        fig = phase_portrait_3d(
            system=self._state.system,
            x_coupling=x_name,
            y_coupling=y_name,
            z_coupling=z_name,
        )

        self._figure.clear()
        src_ax = fig.axes[0]
        dest_ax = self._figure.add_subplot(111, projection="3d")
        _copy_3d_axis(src_ax, dest_ax)
        plt.close(fig)


def _copy_3d_axis(src, dest) -> None:
    """Best-effort copy of labels/limits from one 3D axis to another.

    Full artist transfer is fragile, so we re-add key properties and
    let the caller re-plot if needed.  For the 3D views the helper
    functions already produce complete figures, so we transfer the
    collections and lines directly.
    """
    dest.set_xlabel(src.get_xlabel())
    dest.set_ylabel(src.get_ylabel())
    dest.set_zlabel(src.get_zlabel())
    dest.set_title(src.get_title())

    try:
        dest.set_xlim(src.get_xlim())
        dest.set_ylim(src.get_ylim())
        dest.set_zlim(src.get_zlim())
    except Exception:
        pass

    # Move line objects
    for line in list(src.get_lines()):
        x, y = line.get_data()
        z = line.get_data_3d()[2]
        dest.plot(x, y, z,
                  color=line.get_color(),
                  linewidth=line.get_linewidth(),
                  linestyle=line.get_linestyle(),
                  marker=line.get_marker(),
                  markersize=line.get_markersize(),
                  alpha=line.get_alpha(),
                  label=line.get_label())

    # Scatter collections
    for coll in list(src.collections):
        try:
            offsets = coll.get_offsets()
            if hasattr(coll, '_offsets3d'):
                dest.scatter(
                    coll._offsets3d[0],
                    coll._offsets3d[1],
                    coll._offsets3d[2],
                    c=coll.get_facecolors(),
                    s=coll.get_sizes(),
                    alpha=coll.get_alpha(),
                )
        except Exception:
            pass

    legend = src.get_legend()
    if legend is not None:
        dest.legend()
