"""Cross-platform GUI for Asymptotic Safety exploration.

Provides an interactive interface for configuring truncations,
finding fixed points, integrating RG flows, and visualizing
results in 2D and 3D.

Requires PySide6: install with `pip install asymsafety[gui]`

Usage:
    from asymsafety.gui import launch_gui
    launch_gui()

Or from the command line:
    asymsafety-gui
"""


def launch_gui():
    """Launch the Asymptotic Safety Explorer GUI."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        raise ImportError(
            "PySide6 is required for the GUI. "
            "Install it with: pip install asymsafety[gui]"
        )
    from asymsafety.gui.app import run_application
    run_application()


if __name__ == "__main__":
    launch_gui()
