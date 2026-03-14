"""Background computation worker using QThread.

Runs expensive operations (system building, fixed point search,
flow integration, parameter continuation) off the main GUI thread
to keep the interface responsive.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal


class ComputationWorker(QThread):
    """Generic worker that runs a callable in a background thread.

    Signals:
        progress(int, str): Progress percentage and message.
        result_ready(object): The computed result.
        error_occurred(str): Error message on failure.
    """

    progress = Signal(int, str)
    result_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, task_fn: Callable, *args: Any, **kwargs: Any):
        super().__init__()
        self._task_fn = task_fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._task_fn(*self._args, **self._kwargs)
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
