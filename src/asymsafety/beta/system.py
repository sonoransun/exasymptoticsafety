"""Beta function system: collection of coupled RG flow equations.

A BetaFunctionSystem wraps the symbolic beta functions for all couplings
in a truncation and provides:
    - Symbolic expressions (SymPy)
    - Numerical evaluation via lambdify
    - Jacobian (stability matrix) for fixed point analysis
    - RHS vector for ODE integration
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sympy
from sympy import Expr, Matrix, Symbol, lambdify


@dataclass
class BetaFunction:
    """A single beta function β_i = ∂_t g_i."""

    coupling_name: str
    coupling_symbol: Symbol
    expression: Expr
    _numerical_func: Callable | None = field(default=None, repr=False)

    def symbolic(self) -> Expr:
        """Return the symbolic expression."""
        return self.expression

    def lambdify_func(self, all_symbols: list[Symbol]) -> Callable:
        """Create a fast numerical function.

        Args:
            all_symbols: Ordered list of all coupling symbols.

        Returns:
            A callable f(*coupling_values) -> float.
        """
        if self._numerical_func is None:
            self._numerical_func = lambdify(
                all_symbols, self.expression, modules="numpy"
            )
        return self._numerical_func


class BetaFunctionSystem:
    """A complete system of coupled beta functions.

    Represents ∂_t g_i = β_i(g_1, ..., g_n) for all couplings
    in a given truncation.
    """

    def __init__(self):
        self._betas: OrderedDict[str, BetaFunction] = OrderedDict()
        self._symbols: list[Symbol] = []
        self._rhs_func: Callable | None = None

    def add(self, beta: BetaFunction) -> None:
        """Add a beta function to the system."""
        self._betas[beta.coupling_name] = beta
        if beta.coupling_symbol not in self._symbols:
            self._symbols.append(beta.coupling_symbol)
        self._rhs_func = None  # Invalidate cache

    @property
    def coupling_names(self) -> list[str]:
        return list(self._betas.keys())

    @property
    def coupling_symbols(self) -> list[Symbol]:
        return list(self._symbols)

    @property
    def dimension(self) -> int:
        """Number of couplings (dimension of the system)."""
        return len(self._betas)

    def beta(self, name: str) -> BetaFunction:
        """Get the beta function for a specific coupling."""
        return self._betas[name]

    def symbolic_vector(self) -> list[Expr]:
        """Return all beta functions as a list of symbolic expressions."""
        return [b.expression for b in self._betas.values()]

    def evaluate(self, point: dict[str, float]) -> dict[str, float]:
        """Evaluate all beta functions at a point in coupling space.

        Args:
            point: Dictionary mapping coupling names to values.

        Returns:
            Dictionary mapping coupling names to β_i values.
        """
        # Build substitution dict
        subs = {}
        for beta in self._betas.values():
            if beta.coupling_name in point:
                subs[beta.coupling_symbol] = point[beta.coupling_name]

        result = {}
        for name, beta in self._betas.items():
            val = beta.expression.subs(subs)
            result[name] = float(val)
        return result

    def jacobian_symbolic(self) -> Matrix:
        """Symbolic stability matrix M_ij = ∂β_i/∂g_j."""
        exprs = self.symbolic_vector()
        syms = self.coupling_symbols
        return Matrix([
            [sympy.diff(expr, sym) for sym in syms]
            for expr in exprs
        ])

    def jacobian_numerical(self, point: dict[str, float]) -> np.ndarray:
        """Numerical stability matrix at a given point."""
        J_sym = self.jacobian_symbolic()
        subs = {}
        for beta in self._betas.values():
            if beta.coupling_name in point:
                subs[beta.coupling_symbol] = point[beta.coupling_name]
        J_eval = J_sym.subs(subs)
        return np.array(J_eval.tolist(), dtype=float)

    def rhs_vector(self) -> Callable:
        """Return f(t, y) suitable for scipy.integrate.solve_ivp.

        The RG flow is ∂_t g_i = β_i(g_1, ..., g_n).
        """
        if self._rhs_func is None:
            syms = self.coupling_symbols
            funcs = [
                b.lambdify_func(syms) for b in self._betas.values()
            ]

            def rhs(t, y):
                vals = {s: y[i] for i, s in enumerate(syms)}
                args = [y[i] for i in range(len(syms))]
                return [f(*args) for f in funcs]

            self._rhs_func = rhs
        return self._rhs_func

    def to_latex(self) -> str:
        """Generate LaTeX representation of all beta functions."""
        lines = []
        for name, beta in self._betas.items():
            lhs = rf"\beta_{{{sympy.latex(beta.coupling_symbol)}}}"
            rhs = sympy.latex(beta.expression)
            lines.append(rf"{lhs} = {rhs}")
        return r" \\ ".join(lines)

    def batch_evaluator(self, backend: str = "numpy") -> "BatchEvaluator":
        """Create a batch evaluator for this system.

        Args:
            backend: ``"numpy"`` (default) or ``"jax"`` (requires jax).

        Returns:
            A :class:`BatchEvaluator` that can evaluate beta functions
            at *N* points simultaneously.
        """
        if backend == "numpy":
            from asymsafety.compute.batch.evaluator import NumpyBatchEvaluator

            return NumpyBatchEvaluator(self)
        elif backend == "jax":
            from asymsafety.compute.batch.jax_evaluator import JaxBatchEvaluator

            return JaxBatchEvaluator(self)
        else:
            raise ValueError(f"Unknown batch evaluator backend: {backend}")

    def __repr__(self) -> str:
        names = ", ".join(self.coupling_names)
        return f"BetaFunctionSystem([{names}])"
