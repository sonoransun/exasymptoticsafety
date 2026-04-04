"""Valve models for the hydraulic network (analogues of FRG regulators).

Each valve type implements a flow-rate vs pressure-drop characteristic.
The :class:`LitimValve` mimics the sharp Litim (optimised) regulator
with on/off behaviour, while :class:`ExponentialValve` provides a smooth
proportional opening analogous to the exponential regulator.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class Valve(ABC):
    """Abstract valve (analogue of :class:`~asymsafety.frg.regulator.Regulator`).

    A valve controls the flow rate through a network element as a
    function of the pressure drop across it and an opening parameter.
    """

    @abstractmethod
    def flow_rate(self, delta_p: float, opening: float = 1.0) -> float:
        """Flow rate Q as a function of pressure drop and valve opening.

        Args:
            delta_p: Pressure difference across the valve (Pa).
            opening: Fractional opening in [0, 1].

        Returns:
            Volumetric flow rate (m^3/s).
        """

    @abstractmethod
    def impedance(self, delta_p: float, opening: float = 1.0) -> float:
        """Linearised flow coefficient dQ/d(DeltaP).

        Args:
            delta_p: Operating pressure drop (Pa).
            opening: Fractional opening in [0, 1].

        Returns:
            Derivative of flow rate with respect to pressure drop.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable valve name."""


class LitimValve(Valve):
    """On/off valve: sharp cutoff analogue of Litim regulator.

    Flow characteristic::

        Q = Cv * sign(DeltaP) * sqrt(|DeltaP|)   if opening > threshold
        Q = 0                                      otherwise

    This models the step-function behaviour of the Litim (optimised)
    regulator where modes below the scale are fully suppressed.

    Args:
        cv: Flow coefficient (valve capacity).
        threshold: Minimum opening fraction for flow to pass.
    """

    def __init__(self, cv: float = 1.0, threshold: float = 0.5) -> None:
        self.cv = cv
        self.threshold = threshold

    def flow_rate(self, delta_p: float, opening: float = 1.0) -> float:
        """Flow rate with sharp on/off characteristic."""
        if opening <= self.threshold:
            return 0.0
        sign = 1.0 if delta_p >= 0.0 else -1.0
        return self.cv * sign * math.sqrt(abs(delta_p))

    def impedance(self, delta_p: float, opening: float = 1.0) -> float:
        """Linearised impedance dQ/d(DeltaP).

        For Q = Cv * sign(dp) * sqrt(|dp|):
            dQ/d(dp) = Cv / (2 * sqrt(|dp|))  when |dp| > 0.
        Returns a large value when dp is near zero (low impedance).
        """
        if opening <= self.threshold:
            return 0.0
        abs_dp = abs(delta_p)
        if abs_dp < 1e-14:
            return self.cv * 1e7  # Near-zero drop -> very low impedance
        return self.cv / (2.0 * math.sqrt(abs_dp))

    @property
    def name(self) -> str:
        return "Litim"


class ExponentialValve(Valve):
    """Proportional valve: smooth analogue of exponential regulator.

    Flow characteristic::

        Q = Cv * DeltaP / (exp(DeltaP / P_scale) - 1)   for DeltaP != 0
        Q -> Cv * P_scale                                 for DeltaP -> 0

    This mirrors the Bose-Einstein-like exponential regulator which
    provides a smooth transition from suppressed to free propagation.

    Args:
        cv: Flow coefficient (valve capacity).
        p_scale: Characteristic pressure scale controlling the opening
            profile.
    """

    def __init__(self, cv: float = 1.0, p_scale: float = 1.0) -> None:
        self.cv = cv
        self.p_scale = p_scale

    def flow_rate(self, delta_p: float, opening: float = 1.0) -> float:
        """Flow rate with smooth proportional characteristic."""
        if abs(delta_p) < 1e-14:
            # L'Hopital limit: dp / (exp(dp/P) - 1) -> P as dp -> 0
            return self.cv * self.p_scale * opening
        ratio = delta_p / self.p_scale
        # Clamp to avoid overflow in exp
        if ratio > 500.0:
            return 0.0
        if ratio < -500.0:
            return self.cv * (-delta_p) * opening
        return self.cv * delta_p / (math.exp(ratio) - 1.0) * opening

    def impedance(self, delta_p: float, opening: float = 1.0) -> float:
        """Linearised impedance dQ/d(DeltaP).

        d/d(dp)[dp / (exp(dp/P) - 1)] evaluated analytically.
        """
        if abs(delta_p) < 1e-14:
            # Second-order expansion gives dQ/d(dp) -> Cv * opening / 2
            return self.cv * opening * 0.5
        ratio = delta_p / self.p_scale
        if abs(ratio) > 500.0:
            return 0.0
        exp_r = math.exp(ratio)
        denom = exp_r - 1.0
        # d/d(dp)[dp / (e^(dp/P) - 1)]
        #   = [denom - dp/P * e^(dp/P)] / denom^2
        numerator = denom - ratio * exp_r
        return self.cv * numerator / (denom * denom) * opening

    @property
    def name(self) -> str:
        return "Exponential"
