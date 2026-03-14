"""IR regulator functions for the Wetterich equation.

The regulator R_k(z) modifies the propagator to suppress IR modes below
the scale k. Key properties:
    R_k(z) → k^2     for z → 0   (IR suppression)
    R_k(z) → 0       for z → ∞   (UV modes unchanged)
    R_k → 0          for k → 0   (regulator removed)
"""

from abc import ABC, abstractmethod

import sympy
from sympy import Heaviside, Rational, Symbol, exp, oo, pi


class Regulator(ABC):
    """Abstract IR regulator R_k(z)."""

    @abstractmethod
    def R_k(self, z: sympy.Expr, k: Symbol) -> sympy.Expr:
        """The regulator function R_k(z)."""

    @abstractmethod
    def dR_k_dt(self, z: sympy.Expr, k: Symbol) -> sympy.Expr:
        """RG-time derivative: ∂_t R_k = k ∂_k R_k."""

    @abstractmethod
    def shape_function(self, y: sympy.Expr) -> sympy.Expr:
        """Shape function r(y) where R_k(z) = z · r(z/k^2)."""

    def P_k(self, z: sympy.Expr, k: Symbol) -> sympy.Expr:
        """Regularized kinetic term: P_k(z) = z + R_k(z)."""
        return z + self.R_k(z, k)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""


class LitimRegulator(Regulator):
    """Optimized (Litim) regulator: R_k(z) = (k^2 - z) θ(k^2 - z).

    Properties:
        - Minimizes scheme dependence (Litim 2001)
        - Yields rational/polynomial beta functions
        - P_k(z) = max(z, k^2) = k^2 for z ≤ k^2
        - ∂_t R_k = 2k^2 θ(k^2 - z)
    """

    def R_k(self, z: sympy.Expr, k: Symbol) -> sympy.Expr:
        return (k**2 - z) * Heaviside(k**2 - z)

    def dR_k_dt(self, z: sympy.Expr, k: Symbol) -> sympy.Expr:
        return 2 * k**2 * Heaviside(k**2 - z)

    def shape_function(self, y: sympy.Expr) -> sympy.Expr:
        return (1 / y - 1) * Heaviside(1 - y)

    def P_k(self, z: sympy.Expr, k: Symbol) -> sympy.Expr:
        # For z < k^2: P_k = k^2. For z > k^2: P_k = z.
        # In threshold function integrals, the effective value is k^2.
        return k**2

    @property
    def name(self) -> str:
        return "Litim"


class ExponentialRegulator(Regulator):
    """Exponential regulator: R_k(z) = z / (exp(z/k^2) - 1).

    Properties:
        - Smooth (C^∞)
        - Bose-Einstein-like distribution
        - Threshold functions require numerical integration
    """

    def R_k(self, z: sympy.Expr, k: Symbol) -> sympy.Expr:
        return z / (exp(z / k**2) - 1)

    def dR_k_dt(self, z: sympy.Expr, k: Symbol) -> sympy.Expr:
        y = z / k**2
        return 2 * z * y * exp(y) / (exp(y) - 1)**2

    def shape_function(self, y: sympy.Expr) -> sympy.Expr:
        return 1 / (exp(y) - 1)

    @property
    def name(self) -> str:
        return "Exponential"


class TypeIIRegulator(Regulator):
    """Type-II regulator for higher-derivative theories.

    Replaces Δ → Δ + R_k(Δ) in the full (possibly 4th-order) operator.
    Built on top of a base (typically Litim) regulator.

    For Γ^(2) = a Δ^2 + b Δ + c, the regulated operator is:
        a (Δ + R_k)^2 + b (Δ + R_k) + c
    """

    def __init__(self, base: Regulator | None = None):
        self.base = base or LitimRegulator()

    def R_k(self, z: sympy.Expr, k: Symbol) -> sympy.Expr:
        return self.base.R_k(z, k)

    def dR_k_dt(self, z: sympy.Expr, k: Symbol) -> sympy.Expr:
        return self.base.dR_k_dt(z, k)

    def shape_function(self, y: sympy.Expr) -> sympy.Expr:
        return self.base.shape_function(y)

    def regulate_operator(self, coeffs: list[sympy.Expr],
                          z: sympy.Expr, k: Symbol) -> sympy.Expr:
        """Apply regulator to polynomial operator Σ_n a_n Δ^n.

        Replaces Δ → Δ + R_k(Δ) in each power.
        """
        z_reg = z + self.R_k(z, k)
        result = sympy.S.Zero
        for n, a_n in enumerate(coeffs):
            result += a_n * z_reg**n
        return result

    @property
    def name(self) -> str:
        return f"Type-II ({self.base.name})"


class TypeIIIRegulator(Regulator):
    """Type-III regulator: separate regulator for each propagator pole.

    After partial-fraction decomposition:
        1/(a(z - m_+^2)(z - m_-^2)) = c_+/(z - m_+^2) + c_-/(z - m_-^2)

    Each pole gets its own regulator:
        c_+/(z + R_k(z) - m_+^2) + c_-/(z + R_k(z) - m_-^2)
    """

    def __init__(self, base: Regulator | None = None):
        self.base = base or LitimRegulator()

    def R_k(self, z: sympy.Expr, k: Symbol) -> sympy.Expr:
        return self.base.R_k(z, k)

    def dR_k_dt(self, z: sympy.Expr, k: Symbol) -> sympy.Expr:
        return self.base.dR_k_dt(z, k)

    def shape_function(self, y: sympy.Expr) -> sympy.Expr:
        return self.base.shape_function(y)

    def regulate_pole(self, z: sympy.Expr, mass_sq: sympy.Expr,
                      k: Symbol) -> sympy.Expr:
        """Regulated inverse propagator for a single pole: z + R_k(z) - m^2."""
        return z + self.R_k(z, k) - mass_sq

    @property
    def name(self) -> str:
        return f"Type-III ({self.base.name})"
