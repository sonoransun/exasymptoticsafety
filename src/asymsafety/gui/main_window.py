"""Main application window assembling panels and views."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QMenuBar,
    QStatusBar,
    QTabWidget,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from asymsafety.gui.state import AppState
from asymsafety.gui.panels.system_panel import SystemPanel
from asymsafety.gui.panels.fixed_point_panel import FixedPointPanel
from asymsafety.gui.panels.flow_panel import FlowPanel
from asymsafety.gui.panels.continuation_panel import ContinuationPanel
from asymsafety.gui.views.phase_portrait_view import PhasePortraitView
from asymsafety.gui.views.flow_3d_view import Flow3DView
from asymsafety.gui.views.running_couplings_view import RunningCouplingsView
from asymsafety.gui.views.fp_analysis_view import FPAnalysisView


class MainWindow(QMainWindow):
    """Main application window for the Asymptotic Safety Explorer."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Asymptotic Safety Explorer")
        self.setMinimumSize(1200, 750)

        self.state = AppState(self)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        # --- Left dock: configuration panels ---
        dock = QDockWidget("Configuration", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        toolbox = QToolBox()
        self.system_panel = SystemPanel(self.state)
        self.fp_panel = FixedPointPanel(self.state)
        self.flow_panel = FlowPanel(self.state)
        self.continuation_panel = ContinuationPanel(self.state)

        toolbox.addItem(self.system_panel, "System Setup")
        toolbox.addItem(self.fp_panel, "Fixed Point Finder")
        toolbox.addItem(self.flow_panel, "Flow Integration")
        toolbox.addItem(self.continuation_panel, "Parameter Continuation")

        dock_content = QWidget()
        dock_layout = QVBoxLayout(dock_content)
        dock_layout.setContentsMargins(0, 0, 0, 0)
        dock_layout.addWidget(toolbox)
        dock.setWidget(dock_content)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        # --- Central: visualization tabs ---
        tabs = QTabWidget()
        self.phase_view = PhasePortraitView(self.state)
        self.flow_3d_view = Flow3DView(self.state)
        self.running_view = RunningCouplingsView(self.state)
        self.fp_view = FPAnalysisView(self.state)

        tabs.addTab(self.phase_view, "2D Phase Portrait")
        tabs.addTab(self.flow_3d_view, "3D Flow Space")
        tabs.addTab(self.running_view, "Running Couplings")
        tabs.addTab(self.fp_view, "Fixed Point Analysis")

        self.setCentralWidget(tabs)

        # --- Menu bar ---
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction("Quit", self.close)

        view_menu = menu.addMenu("View")
        view_menu.addAction("Toggle Config Panel", lambda: dock.setVisible(not dock.isVisible()))

        help_menu = menu.addMenu("Help")
        help_menu.addAction("About", self._show_about)

        # --- Status bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — select a truncation and click Build System")

    def _connect_signals(self):
        self.state.status_message.connect(self.status_bar.showMessage)
        self.state.busy_changed.connect(self._on_busy_changed)

    def _on_busy_changed(self, busy: bool):
        if busy:
            self.status_bar.showMessage("Computing...")
        self.system_panel.setEnabled(not busy)

    def _show_about(self):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "Asymptotic Safety Explorer",
            "Interactive tool for exploring Asymptotic Safety in quantum gravity.\n\n"
            "Computes FRG beta functions, locates non-Gaussian UV fixed points,\n"
            "determines critical exponents, and visualises RG flows in 2D and 3D.\n\n"
            "Built with PySide6 and matplotlib.",
        )
