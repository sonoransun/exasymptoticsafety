"""Central application state with reactive Qt signals.

All panels write to AppState, all views subscribe to its signals.
This cleanly separates input (panels) from output (views) without
direct coupling between them.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from asymsafety.analysis.fixed_points import FixedPoint
from asymsafety.analysis.flow import RGTrajectory
from asymsafety.analysis.stability import StabilityAnalysis
from asymsafety.analysis.continuation import ContinuationResult
from asymsafety.beta.system import BetaFunctionSystem


class AppState(QObject):
    """Reactive application state.

    Panels modify state via set_* methods, which emit signals
    that views listen to for automatic redrawing.
    """

    system_changed = Signal(object)
    fixed_points_changed = Signal(list)
    selected_fp_changed = Signal(object)
    stability_changed = Signal(object)
    trajectories_changed = Signal(list)
    continuation_changed = Signal(object)
    status_message = Signal(str)
    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._system: BetaFunctionSystem | None = None
        self._system_name: str = ""
        self._fixed_points: list[FixedPoint] = []
        self._selected_fp: FixedPoint | None = None
        self._stability: StabilityAnalysis | None = None
        self._trajectories: list[RGTrajectory] = []
        self._continuation: ContinuationResult | None = None
        self._busy: bool = False

    # --- Properties ---

    @property
    def system(self) -> BetaFunctionSystem | None:
        return self._system

    @property
    def system_name(self) -> str:
        return self._system_name

    @property
    def fixed_points(self) -> list[FixedPoint]:
        return self._fixed_points

    @property
    def selected_fp(self) -> FixedPoint | None:
        return self._selected_fp

    @property
    def stability(self) -> StabilityAnalysis | None:
        return self._stability

    @property
    def trajectories(self) -> list[RGTrajectory]:
        return self._trajectories

    @property
    def continuation(self) -> ContinuationResult | None:
        return self._continuation

    @property
    def busy(self) -> bool:
        return self._busy

    # --- Setters (emit signals) ---

    def set_system(self, system: BetaFunctionSystem, name: str = "") -> None:
        self._system = system
        self._system_name = name
        self._fixed_points = []
        self._selected_fp = None
        self._stability = None
        self._trajectories = []
        self._continuation = None
        self.system_changed.emit(system)
        self.fixed_points_changed.emit([])
        self.trajectories_changed.emit([])
        self.status_message.emit(f"System: {name} ({', '.join(system.coupling_names)})")

    def set_fixed_points(self, fps: list[FixedPoint]) -> None:
        self._fixed_points = fps
        self._selected_fp = None
        self._stability = None
        self.fixed_points_changed.emit(fps)
        n_ngfp = sum(1 for fp in fps if not fp.is_gaussian)
        self.status_message.emit(
            f"Found {len(fps)} fixed points ({n_ngfp} non-Gaussian)"
        )

    def set_selected_fp(self, fp: FixedPoint | None) -> None:
        self._selected_fp = fp
        self.selected_fp_changed.emit(fp)

    def set_stability(self, sa: StabilityAnalysis | None) -> None:
        self._stability = sa
        self.stability_changed.emit(sa)

    def add_trajectory(self, traj: RGTrajectory) -> None:
        self._trajectories.append(traj)
        self.trajectories_changed.emit(self._trajectories)

    def set_trajectories(self, trajs: list[RGTrajectory]) -> None:
        self._trajectories = trajs
        self.trajectories_changed.emit(trajs)

    def set_continuation(self, result: ContinuationResult) -> None:
        self._continuation = result
        self.continuation_changed.emit(result)

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.busy_changed.emit(busy)

    def clear_trajectories(self) -> None:
        self._trajectories = []
        self.trajectories_changed.emit([])
