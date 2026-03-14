"""Application entry point: creates QApplication and shows main window."""

from __future__ import annotations

import sys


def run_application():
    """Create and run the Qt application."""
    import matplotlib
    matplotlib.use("QtAgg")

    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("Asymptotic Safety Explorer")
    app.setStyle("Fusion")

    from asymsafety.gui.main_window import MainWindow
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run_application()
