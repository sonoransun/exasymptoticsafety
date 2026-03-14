"""Parameter continuation panel.

Tracks a fixed point as an external parameter (number of matter fields)
is varied continuously, using the continuation algorithm with automatic
guess updating.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from asymsafety.gui.state import AppState
from asymsafety.gui.worker import ComputationWorker


_PARAMETER_CHOICES = ["n_scalars", "n_dirac", "n_vectors"]


class ContinuationPanel(QWidget):
    """Panel for running parameter continuation of fixed points."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._worker: ComputationWorker | None = None
        self._coupling_names: list[str] = []
        self._guess_spins: dict[str, QDoubleSpinBox] = {}
        self.setMaximumWidth(350)
        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # --- Parameter selector ---
        param_form = QFormLayout()

        self._param_combo = QComboBox()
        self._param_combo.addItems(_PARAMETER_CHOICES)
        param_form.addRow("Parameter:", self._param_combo)

        self._start_spin = QDoubleSpinBox()
        self._start_spin.setRange(0.0, 100.0)
        self._start_spin.setDecimals(1)
        self._start_spin.setSingleStep(1.0)
        self._start_spin.setValue(0.0)
        param_form.addRow("Start:", self._start_spin)

        self._end_spin = QDoubleSpinBox()
        self._end_spin.setRange(0.0, 100.0)
        self._end_spin.setDecimals(1)
        self._end_spin.setSingleStep(1.0)
        self._end_spin.setValue(20.0)
        param_form.addRow("End:", self._end_spin)

        self._n_steps_spin = QSpinBox()
        self._n_steps_spin.setRange(2, 200)
        self._n_steps_spin.setValue(20)
        param_form.addRow("Steps:", self._n_steps_spin)

        layout.addLayout(param_form)

        # --- Initial FP guess group ---
        self._guess_group = QGroupBox("Initial FP Guess")
        self._guess_layout = QFormLayout()
        self._guess_group.setLayout(self._guess_layout)
        layout.addWidget(self._guess_group)

        # --- Run button ---
        self._run_button = QPushButton("Run Continuation")
        self._run_button.setEnabled(False)
        layout.addWidget(self._run_button)

        layout.addStretch()

    # ------------------------------------------------------------------ #
    #  Signal wiring
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        self._state.system_changed.connect(self._on_system_changed)
        self._run_button.clicked.connect(self._on_run_clicked)

    # ------------------------------------------------------------------ #
    #  Dynamic coupling fields
    # ------------------------------------------------------------------ #

    def _rebuild_coupling_fields(self, names: list[str]) -> None:
        """Recreate guess spin boxes for current couplings."""
        self._coupling_names = list(names)

        while self._guess_layout.rowCount() > 0:
            self._guess_layout.removeRow(0)
        self._guess_spins.clear()

        for name in names:
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            spin.setValue(0.5)
            self._guess_layout.addRow(f"{name}:", spin)
            self._guess_spins[name] = spin

    # ------------------------------------------------------------------ #
    #  Slots
    # ------------------------------------------------------------------ #

    def _on_system_changed(self, system: object) -> None:
        names = system.coupling_names  # type: ignore[union-attr]
        self._rebuild_coupling_fields(names)
        self._run_button.setEnabled(True)

    def _on_run_clicked(self) -> None:
        if self._state.system is None:
            return

        param_name = self._param_combo.currentText()
        start = self._start_spin.value()
        end = self._end_spin.value()
        n_steps = self._n_steps_spin.value()
        guess = {
            name: spin.value() for name, spin in self._guess_spins.items()
        }

        self._run_button.setEnabled(False)
        self._state.set_busy(True)

        self._worker = ComputationWorker(
            self._run_continuation, param_name, start, end, n_steps, guess
        )
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_result(self, result: object) -> None:
        self._state.set_continuation(result)  # type: ignore[arg-type]
        self._run_button.setEnabled(True)
        self._state.set_busy(False)
        self._state.status_message.emit("Continuation complete.")

    def _on_error(self, message: str) -> None:
        self._run_button.setEnabled(True)
        self._state.set_busy(False)
        self._state.status_message.emit(f"Continuation error: {message}")

    # ------------------------------------------------------------------ #
    #  Worker callable (runs in background thread)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run_continuation(
        param_name: str,
        start: float,
        end: float,
        n_steps: int,
        initial_guess: dict[str, float],
    ):
        from asymsafety.actions.matter import MatterContent
        from asymsafety.analysis.continuation import continuation
        from asymsafety.beta.matter import build_eh_matter_beta_system

        parameter_values = np.linspace(start, end, n_steps)

        def system_builder(param_value: float):
            kwargs = {
                "n_scalars": 0,
                "n_dirac": 0,
                "n_vectors": 0,
            }
            kwargs[param_name] = int(round(param_value))
            matter = MatterContent(**kwargs)
            return build_eh_matter_beta_system(matter=matter)

        return continuation(
            system_builder=system_builder,
            parameter_name=param_name,
            parameter_values=parameter_values,
            initial_guess=initial_guess,
        )
