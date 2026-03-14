"""Fixed point search and display panel.

Provides controls for finding individual or all fixed points of the
current beta function system, displays results in a table, and allows
selecting a fixed point to trigger stability analysis.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymsafety.gui.state import AppState
from asymsafety.gui.worker import ComputationWorker


class FixedPointPanel(QWidget):
    """Panel for finding and inspecting fixed points."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._worker: ComputationWorker | None = None
        self._coupling_names: list[str] = []
        self._guess_spins: dict[str, QDoubleSpinBox] = {}
        self._found_fps: list = []
        self.setMaximumWidth(350)
        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # --- Initial guess group ---
        self._guess_group = QGroupBox("Initial Guess")
        self._guess_layout = QFormLayout()
        self._guess_group.setLayout(self._guess_layout)
        layout.addWidget(self._guess_group)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        self._find_button = QPushButton("Find Fixed Point")
        self._find_button.setEnabled(False)
        btn_row.addWidget(self._find_button)

        self._find_all_button = QPushButton("Find All Fixed Points")
        self._find_all_button.setEnabled(False)
        btn_row.addWidget(self._find_all_button)

        layout.addLayout(btn_row)

        # --- Results table ---
        self._table = QTableWidget()
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)

        layout.addStretch()

    # ------------------------------------------------------------------ #
    #  Signal wiring
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        self._state.system_changed.connect(self._on_system_changed)
        self._find_button.clicked.connect(self._on_find_clicked)
        self._find_all_button.clicked.connect(self._on_find_all_clicked)
        self._table.itemSelectionChanged.connect(self._on_row_selected)

    # ------------------------------------------------------------------ #
    #  Dynamic coupling fields
    # ------------------------------------------------------------------ #

    def _rebuild_coupling_fields(self, names: list[str]) -> None:
        """Recreate the initial-guess spin boxes for current couplings."""
        self._coupling_names = list(names)

        # Clear previous widgets
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

        # Reset table columns
        columns = ["Type"] + list(names) + ["Relevant"]
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)
        self._table.setRowCount(0)
        self._found_fps = []

    # ------------------------------------------------------------------ #
    #  Slots
    # ------------------------------------------------------------------ #

    def _on_system_changed(self, system: object) -> None:
        names = system.coupling_names  # type: ignore[union-attr]
        self._rebuild_coupling_fields(names)
        self._find_button.setEnabled(True)
        self._find_all_button.setEnabled(True)

    def _on_find_clicked(self) -> None:
        system = self._state.system
        if system is None:
            return

        guess = {
            name: spin.value() for name, spin in self._guess_spins.items()
        }

        self._find_button.setEnabled(False)
        self._find_all_button.setEnabled(False)
        self._state.set_busy(True)

        self._worker = ComputationWorker(self._search_single, system, guess)
        self._worker.result_ready.connect(self._on_find_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_find_all_clicked(self) -> None:
        system = self._state.system
        if system is None:
            return

        self._find_button.setEnabled(False)
        self._find_all_button.setEnabled(False)
        self._state.set_busy(True)

        self._worker = ComputationWorker(self._search_all, system)
        self._worker.result_ready.connect(self._on_find_all_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_find_result(self, result: object) -> None:
        self._find_button.setEnabled(True)
        self._find_all_button.setEnabled(True)
        self._state.set_busy(False)

        if result is None:
            self._state.status_message.emit("No fixed point found from guess.")
            return

        # Merge into existing list avoiding duplicates
        fps = list(self._found_fps)
        if not self._is_duplicate(result, fps):
            fps.append(result)
        self._found_fps = fps
        self._state.set_fixed_points(fps)
        self._populate_table(fps)

    def _on_find_all_result(self, fps: object) -> None:
        self._find_button.setEnabled(True)
        self._find_all_button.setEnabled(True)
        self._state.set_busy(False)

        self._found_fps = list(fps)  # type: ignore[arg-type]
        self._state.set_fixed_points(self._found_fps)
        self._populate_table(self._found_fps)

    def _on_error(self, message: str) -> None:
        self._find_button.setEnabled(True)
        self._find_all_button.setEnabled(True)
        self._state.set_busy(False)
        self._state.status_message.emit(f"Error: {message}")

    def _on_row_selected(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if 0 <= row < len(self._found_fps):
            fp = self._found_fps[row]
            self._state.set_selected_fp(fp)
            self._run_stability(fp)

    # ------------------------------------------------------------------ #
    #  Table population
    # ------------------------------------------------------------------ #

    def _populate_table(self, fps: list) -> None:
        self._table.setRowCount(len(fps))
        for row, fp in enumerate(fps):
            fp_type = "GFP" if fp.is_gaussian else "NGFP"
            self._table.setItem(row, 0, QTableWidgetItem(fp_type))

            for col, name in enumerate(self._coupling_names):
                val = fp.location.get(name, 0.0)
                item = QTableWidgetItem(f"{val:.6f}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._table.setItem(row, 1 + col, item)

            relevant = str(fp.relevant_directions)
            item = QTableWidgetItem(relevant)
            item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(
                row, 1 + len(self._coupling_names), item
            )

        self._table.resizeColumnsToContents()

    # ------------------------------------------------------------------ #
    #  Stability analysis (background)
    # ------------------------------------------------------------------ #

    def _run_stability(self, fp) -> None:
        system = self._state.system
        if system is None:
            return

        worker = ComputationWorker(self._analyze_stability, system, fp)
        worker.result_ready.connect(
            lambda sa: self._state.set_stability(sa)
        )
        worker.error_occurred.connect(
            lambda msg: self._state.status_message.emit(
                f"Stability error: {msg}"
            )
        )
        # Keep reference so it is not garbage collected
        self._stability_worker = worker
        worker.start()

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_duplicate(fp, fps: list, tol: float = 1e-4) -> bool:
        for existing in fps:
            dist = sum(
                (fp.location[k] - existing.location[k]) ** 2
                for k in fp.location
            ) ** 0.5
            if dist < tol:
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Worker callables (run in background thread)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _search_single(system, guess: dict[str, float]):
        from asymsafety.analysis.fixed_points import FixedPointFinder

        finder = FixedPointFinder(system)
        return finder.find_fixed_point(guess)

    @staticmethod
    def _search_all(system):
        from asymsafety.analysis.fixed_points import FixedPointFinder

        finder = FixedPointFinder(system)
        return finder.find_all_fixed_points()

    @staticmethod
    def _analyze_stability(system, fp):
        from asymsafety.analysis.stability import analyze_stability

        return analyze_stability(system, fp)
