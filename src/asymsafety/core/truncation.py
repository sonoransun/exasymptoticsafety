"""Abstract base class for truncation ansaetze of the effective average action."""

from abc import ABC, abstractmethod

import sympy

from asymsafety.core.coupling import CouplingSet


class Truncation(ABC):
    """Abstract base for a truncation of Γ_k.

    Subclasses define the specific form of the action, its couplings,
    and the second functional derivative (Hessian) needed for the
    Wetterich equation.
    """

    @abstractmethod
    def couplings(self) -> CouplingSet:
        """Return all couplings in this truncation."""

    @abstractmethod
    def action_symbolic(self, d: int = 4) -> sympy.Expr:
        """Return the symbolic integrand of the truncated action.

        This is the Lagrangian density (without √g ∫d^dx).
        """

    @abstractmethod
    def hessian(self, sector: str, z: sympy.Symbol,
                R_bar: sympy.Symbol, d: int = 4) -> sympy.Expr | sympy.Matrix:
        """Return Γ_k^(2) in the given spin sector.

        Args:
            sector: One of "TT", "vector", "scalar", "ghost".
            z: The Laplacian eigenvalue z = -D^2.
            R_bar: Background Ricci scalar.
            d: Spacetime dimension.

        Returns:
            A SymPy expression (for scalar sectors like TT) or a
            SymPy Matrix (for coupled sectors like the scalar (σ, h) block).
        """
