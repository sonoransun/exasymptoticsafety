"""System builder panel for selecting truncation and building beta functions.

Allows the user to choose a truncation type (Einstein-Hilbert, Quadratic
Gravity, Foliated EH, or EH + Matter), configure parameters, and build
the corresponding BetaFunctionSystem via a background worker thread.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from asymsafety.gui.state import AppState
from asymsafety.gui.worker import ComputationWorker


_TRUNCATION_LABELS = [
    "Einstein-Hilbert",
    "Quadratic Gravity",
    "Foliated EH",
    "EH + Matter",
]


class SystemPanel(QWidget):
    """Panel for selecting a truncation and building the beta function system."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._worker: ComputationWorker | None = None
        self.setMaximumWidth(350)
        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # --- Truncation selector ---
        form = QFormLayout()

        self._truncation_combo = QComboBox()
        self._truncation_combo.addItems(_TRUNCATION_LABELS)
        form.addRow("Truncation:", self._truncation_combo)

        self._dim_spin = QSpinBox()
        self._dim_spin.setRange(3, 6)
        self._dim_spin.setValue(4)
        form.addRow("Dimension d:", self._dim_spin)

        layout.addLayout(form)

        # --- Lorentzian checkbox (visible only for Foliated EH) ---
        self._lorentzian_check = QCheckBox("Lorentzian")
        self._lorentzian_check.setChecked(True)
        self._lorentzian_check.setVisible(False)
        layout.addWidget(self._lorentzian_check)

        # --- Matter content group (visible only for EH + Matter) ---
        self._matter_group = QGroupBox("Matter Content")
        self._matter_group.setVisible(False)
        matter_form = QFormLayout()

        self._n_scalars_spin = QSpinBox()
        self._n_scalars_spin.setRange(0, 50)
        self._n_scalars_spin.setValue(0)
        matter_form.addRow("n_scalars:", self._n_scalars_spin)

        self._n_dirac_spin = QSpinBox()
        self._n_dirac_spin.setRange(0, 50)
        self._n_dirac_spin.setValue(0)
        matter_form.addRow("n_dirac:", self._n_dirac_spin)

        self._n_vectors_spin = QSpinBox()
        self._n_vectors_spin.setRange(0, 50)
        self._n_vectors_spin.setValue(0)
        matter_form.addRow("n_vectors:", self._n_vectors_spin)

        self._xi_spin = QDoubleSpinBox()
        self._xi_spin.setRange(0.0, 1.0)
        self._xi_spin.setSingleStep(0.01)
        self._xi_spin.setDecimals(2)
        self._xi_spin.setValue(0.0)
        matter_form.addRow("xi:", self._xi_spin)

        self._matter_group.setLayout(matter_form)
        layout.addWidget(self._matter_group)

        # --- Build button ---
        self._build_button = QPushButton("Build System")
        layout.addWidget(self._build_button)

        # --- Info label ---
        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        layout.addStretch()

    # ------------------------------------------------------------------ #
    #  Signal wiring
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        self._truncation_combo.currentIndexChanged.connect(
            self._on_truncation_changed
        )
        self._build_button.clicked.connect(self._on_build_clicked)

    # ------------------------------------------------------------------ #
    #  Slots
    # ------------------------------------------------------------------ #

    def _on_truncation_changed(self, index: int) -> None:
        label = _TRUNCATION_LABELS[index]
        self._lorentzian_check.setVisible(label == "Foliated EH")
        self._matter_group.setVisible(label == "EH + Matter")

    def _on_build_clicked(self) -> None:
        truncation = self._truncation_combo.currentText()
        d = self._dim_spin.value()

        self._build_button.setEnabled(False)
        self._info_label.setText("Building system ...")
        self._state.set_busy(True)

        if truncation == "Einstein-Hilbert":
            self._worker = ComputationWorker(self._build_eh, d)
        elif truncation == "Quadratic Gravity":
            self._worker = ComputationWorker(self._build_quadratic, d)
        elif truncation == "Foliated EH":
            lorentzian = self._lorentzian_check.isChecked()
            self._worker = ComputationWorker(
                self._build_foliated, d, lorentzian
            )
        elif truncation == "EH + Matter":
            self._worker = ComputationWorker(
                self._build_matter,
                d,
                self._n_scalars_spin.value(),
                self._n_dirac_spin.value(),
                self._n_vectors_spin.value(),
                self._xi_spin.value(),
            )
        else:
            self._info_label.setText(f"Unknown truncation: {truncation}")
            self._build_button.setEnabled(True)
            self._state.set_busy(False)
            return

        self._worker.result_ready.connect(
            lambda result: self._on_result(result, truncation)
        )
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_result(self, system: object, name: str) -> None:
        self._state.set_system(system, name)  # type: ignore[arg-type]
        couplings = ", ".join(system.coupling_names)  # type: ignore[union-attr]
        self._info_label.setText(f"{name}\nCouplings: {couplings}")
        self._build_button.setEnabled(True)
        self._state.set_busy(False)

    def _on_error(self, message: str) -> None:
        self._info_label.setText(f"Error: {message}")
        self._build_button.setEnabled(True)
        self._state.set_busy(False)

    # ------------------------------------------------------------------ #
    #  Builder helpers (run inside ComputationWorker thread)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_eh(d: int):
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system

        return build_eh_beta_system(d=d)

    @staticmethod
    def _build_quadratic(d: int):
        from asymsafety.beta.quadratic import build_quadratic_beta_system

        return build_quadratic_beta_system(d=d)

    @staticmethod
    def _build_foliated(d: int, lorentzian: bool):
        from asymsafety.beta.foliated import build_foliated_eh_beta_system

        return build_foliated_eh_beta_system(d=d, lorentzian=lorentzian)

    @staticmethod
    def _build_matter(
        d: int,
        n_scalars: int,
        n_dirac: int,
        n_vectors: int,
        xi: float,
    ):
        from asymsafety.actions.matter import MatterContent
        from asymsafety.beta.matter import build_eh_matter_beta_system

        import sympy

        matter = MatterContent(
            n_scalars=n_scalars,
            n_dirac=n_dirac,
            n_vectors=n_vectors,
            xi=sympy.Rational(xi).limit_denominator(1000),
        )
        return build_eh_matter_beta_system(matter=matter, d=d)
