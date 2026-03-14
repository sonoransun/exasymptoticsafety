"""JAX-based batch evaluator for GPU acceleration.

Re-lambdifies SymPy expressions with ``modules=["jax"]``, then uses
``jax.vmap`` and ``jax.jit`` for automatic vectorisation and compilation.

Requires: ``pip install asymsafety[gpu]``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from asymsafety.beta.system import BetaFunctionSystem


class JaxBatchEvaluator:
    """GPU-accelerated batch evaluator using JAX ``vmap`` + ``jit``."""

    def __init__(self, system: BetaFunctionSystem) -> None:
        try:
            import jax
            import jax.numpy as jnp
            from sympy import lambdify
        except ImportError as exc:
            raise ImportError(
                "JAX is required. Install with: pip install asymsafety[gpu]"
            ) from exc

        self._system = system
        symbols = system.coupling_symbols
        names = system.coupling_names
        self._n_couplings = system.dimension

        # Re-lambdify each beta function with the JAX backend
        jax_funcs: list = []
        for name in names:
            beta = system.beta(name)
            fn = lambdify(symbols, beta.expression, modules=["jax"])
            jax_funcs.append(fn)

        # Create a single vectorised function that evaluates all betas
        def eval_single(point):
            args = [point[i] for i in range(len(symbols))]
            return jnp.array([fn(*args) for fn in jax_funcs])

        # vmap + jit for batch evaluation on GPU
        self._batch_fn = jax.jit(jax.vmap(eval_single))

    def evaluate_batch(self, points: np.ndarray) -> np.ndarray:
        """Evaluate beta functions at *N* points on the accelerator.

        Args:
            points: ``(N, n_couplings)`` array.

        Returns:
            ``(N, n_couplings)`` NumPy array.
        """
        import jax.numpy as jnp  # noqa: WPS433

        jax_points = jnp.array(points)
        result = self._batch_fn(jax_points)
        return np.asarray(result)

    def evaluate_norms(self, points: np.ndarray) -> np.ndarray:
        """Evaluate |beta(x)| at *N* points."""
        betas = self.evaluate_batch(points)
        return np.linalg.norm(betas, axis=1)
