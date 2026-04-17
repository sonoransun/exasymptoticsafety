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

from asymsafety.utils.caching import disk_cache


@disk_cache(name="jacobian_symbolic")
def _compute_jacobian_symbolic(exprs: tuple[Expr, ...],
                                 syms: tuple[Symbol, ...]) -> Matrix:
    """Symbolic ∂β_i/∂g_j — disk-cached across sessions.

    Pure function of (expressions, symbols); identical inputs return the
    same Matrix bit-exactly, so disk caching is safe.
    """
    return Matrix([
        [sympy.diff(expr, sym) for sym in syms]
        for expr in exprs
    ])


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

    All hot paths (``evaluate``, ``jacobian_numerical``, ``rhs_vector``)
    use SymPy ``lambdify``-cached callables under the hood; symbolic
    substitution happens at most once per system, not per call.
    """

    def __init__(self):
        self._betas: OrderedDict[str, BetaFunction] = OrderedDict()
        self._symbols: list[Symbol] = []
        self._rhs_func: Callable | None = None
        self._eval_func: Callable | None = None
        self._jac_func: Callable | None = None
        self._jac_symbolic: Matrix | None = None

    def add(self, beta: BetaFunction) -> None:
        """Add a beta function to the system."""
        self._betas[beta.coupling_name] = beta
        if beta.coupling_symbol not in self._symbols:
            self._symbols.append(beta.coupling_symbol)
        # Invalidate all derived caches
        self._rhs_func = None
        self._eval_func = None
        self._jac_func = None
        self._jac_symbolic = None

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

    def _get_eval_func(self) -> Callable:
        """Lambdified vector evaluator: ``f(*values) -> list[float]``."""
        if self._eval_func is None:
            self._eval_func = lambdify(
                self._symbols, self.symbolic_vector(), modules="numpy"
            )
        return self._eval_func

    def _args_from_point(self, point: dict[str, float]) -> list[float]:
        """Pack coupling values in canonical order, defaulting missing → 0.0."""
        return [float(point.get(name, 0.0)) for name in self._betas.keys()]

    def evaluate(self, point: dict[str, float]) -> dict[str, float]:
        """Evaluate all beta functions at a point in coupling space.

        Args:
            point: Dictionary mapping coupling names to values. Missing
                couplings default to 0.0.

        Returns:
            Dictionary mapping coupling names to β_i values.
        """
        fn = self._get_eval_func()
        args = self._args_from_point(point)
        vals = fn(*args)
        # lambdify on a list returns a list; on a Matrix it returns ndarray.
        return {
            name: float(v)
            for name, v in zip(self._betas.keys(), vals)
        }

    def jacobian_symbolic(self) -> Matrix:
        """Symbolic stability matrix M_ij = ∂β_i/∂g_j.

        Memory-cached on the system instance; the underlying differentiation
        is also disk-cached when joblib is available (see ``utils.caching``).
        """
        if self._jac_symbolic is None:
            self._jac_symbolic = _compute_jacobian_symbolic(
                tuple(self.symbolic_vector()),
                tuple(self.coupling_symbols),
            )
        return self._jac_symbolic

    def _get_jac_func(self) -> Callable:
        """Lambdified Jacobian: ``f(*values) -> ndarray``."""
        if self._jac_func is None:
            self._jac_func = lambdify(
                self._symbols, self.jacobian_symbolic(), modules="numpy"
            )
        return self._jac_func

    def jacobian_numerical(self, point: dict[str, float]) -> np.ndarray:
        """Numerical stability matrix at a given point."""
        fn = self._get_jac_func()
        args = self._args_from_point(point)
        return np.asarray(fn(*args), dtype=float)

    def rhs_vector(self) -> Callable:
        """Return f(t, y) suitable for scipy.integrate.solve_ivp.

        The RG flow is ∂_t g_i = β_i(g_1, ..., g_n).
        """
        if self._rhs_func is None:
            fn = self._get_eval_func()
            n = len(self._symbols)

            def rhs(t, y):
                return list(fn(*[y[i] for i in range(n)]))

            self._rhs_func = rhs
        return self._rhs_func

    def __getstate__(self) -> dict:
        """Strip lambdified caches before pickling.

        Sympy expressions pickle cleanly; lambdified callables do not
        (their dynamic code object cannot be referenced from another
        interpreter). Workers rebuild the caches lazily.
        """
        state = self.__dict__.copy()
        state["_rhs_func"] = None
        state["_eval_func"] = None
        state["_jac_func"] = None
        # _jac_symbolic is a sympy Matrix and pickles fine — keep it.
        # Per-BetaFunction lambdify cache:
        for beta in state["_betas"].values():
            beta._numerical_func = None
        return state

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
