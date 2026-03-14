"""RG flow integration panel.

Provides controls for setting initial conditions, integration range,
and launching flow integration via FlowIntegrator in a background
worker thread. Displays a list of computed trajectories.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asymsafety.gui.state import AppState
from asymsafety.gui.worker import ComputationWorker


class FlowPanel(QWidget):
    """Panel for integrating RG flow trajectories."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._worker: ComputationWorker | None = None
        self._coupling_names: list[str] = []
        self._ic_spins: dict[str, QDoubleSpinBox] = {}
        self.setMaximumWidth(350)
        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # --- Initial conditions group ---
        self._ic_group = QGroupBox("Initial Conditions")
        self._ic_layout = QFormLayout()
        self._ic_group.setLayout(self._ic_layout)
        layout.addWidget(self._ic_group)

        # --- Copy from FP button ---
        self._copy_fp_button = QPushButton("Copy from Selected FP")
        self._copy_fp_button.setEnabled(False)
        layout.addWidget(self._copy_fp_button)

        # --- Integration range ---
        range_form = QFormLayout()

        self._t_start_spin = QDoubleSpinBox()
        self._t_start_spin.setRange(-50.0, 50.0)
        self._t_start_spin.setDecimals(2)
        self._t_start_spin.setSingleStep(1.0)
        self._t_start_spin.setValue(-10.0)
        range_form.addRow("t_start:", self._t_start_spin)

        self._t_end_spin = QDoubleSpinBox()
        self._t_end_spin.setRange(-50.0, 50.0)
        self._t_end_spin.setDecimals(2)
        self._t_end_spin.setSingleStep(1.0)
        self._t_end_spin.setValue(10.0)
        range_form.addRow("t_end:", self._t_end_spin)

        layout.addLayout(range_form)

        # --- Action buttons ---
        btn_row = QHBoxLayout()

        self._integrate_button = QPushButton("Integrate")
        self._integrate_button.setEnabled(False)
        btn_row.addWidget(self._integrate_button)

        self._clear_button = QPushButton("Clear Trajectories")
        btn_row.addWidget(self._clear_button)

        layout.addLayout(btn_row)

        # --- Trajectory list ---
        self._trajectory_list = QListWidget()
        layout.addWidget(self._trajectory_list)

        layout.addStretch()

    # ------------------------------------------------------------------ #
    #  Signal wiring
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        self._state.system_changed.connect(self._on_system_changed)
        self._state.selected_fp_changed.connect(self._on_selected_fp_changed)
        self._state.trajectories_changed.connect(self._on_trajectories_changed)
        self._copy_fp_button.clicked.connect(self._on_copy_fp_clicked)
        self._integrate_button.clicked.connect(self._on_integrate_clicked)
        self._clear_button.clicked.connect(self._on_clear_clicked)

    # ------------------------------------------------------------------ #
    #  Dynamic coupling fields
    # ------------------------------------------------------------------ #

    def _rebuild_coupling_fields(self, names: list[str]) -> None:
        """Recreate initial-condition spin boxes for current couplings."""
        self._coupling_names = list(names)

        while self._ic_layout.rowCount() > 0:
            self._ic_layout.removeRow(0)
        self._ic_spins.clear()

        for name in names:
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            spin.setValue(0.0)
            self._ic_layout.addRow(f"{name}:", spin)
            self._ic_spins[name] = spin

    # ------------------------------------------------------------------ #
    #  Slots
    # ------------------------------------------------------------------ #

    def _on_system_changed(self, system: object) -> None:
        names = system.coupling_names  # type: ignore[union-attr]
        self._rebuild_coupling_fields(names)
        self._integrate_button.setEnabled(True)
        self._trajectory_list.clear()

    def _on_selected_fp_changed(self, fp: object) -> None:
        self._copy_fp_button.setEnabled(fp is not None)

    def _on_copy_fp_clicked(self) -> None:
        fp = self._state.selected_fp
        if fp is None:
            return
        for name, spin in self._ic_spins.items():
            if name in fp.location:
                spin.setValue(fp.location[name])

    def _on_integrate_clicked(self) -> None:
        system = self._state.system
        if system is None:
            return

        ic = {name: spin.value() for name, spin in self._ic_spins.items()}
        t_start = self._t_start_spin.value()
        t_end = self._t_end_spin.value()

        self._integrate_button.setEnabled(False)
        self._state.set_busy(True)

        self._worker = ComputationWorker(
            self._run_integration, system, ic, t_start, t_end
        )
        self._worker.result_ready.connect(self._on_integrate_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_integrate_result(self, traj: object) -> None:
        self._state.add_trajectory(traj)  # type: ignore[arg-type]
        self._integrate_button.setEnabled(True)
        self._state.set_busy(False)

    def _on_error(self, message: str) -> None:
        self._integrate_button.setEnabled(True)
        self._state.set_busy(False)
        self._state.status_message.emit(f"Integration error: {message}")

    def _on_clear_clicked(self) -> None:
        self._state.clear_trajectories()

    def _on_trajectories_changed(self, trajs: list) -> None:
        self._trajectory_list.clear()
        for i, traj in enumerate(trajs):
            ic_str = ", ".join(
                f"{n}={traj.coupling_values[n][0]:.4f}"
                for n in traj.coupling_names
            )
            self._trajectory_list.addItem(f"Traj {i + 1}: {ic_str}")

    # ------------------------------------------------------------------ #
    #  Worker callable (runs in background thread)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run_integration(system, ic: dict[str, float],
                         t_start: float, t_end: float):
        from asymsafety.analysis.flow import FlowIntegrator

        integrator = FlowIntegrator(system)
        return integrator.integrate(
            initial_conditions=ic,
            t_span=(t_start, t_end),
        )
