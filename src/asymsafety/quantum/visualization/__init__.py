"""Plotting utilities for the quantum analogues of the RG flow.

Each sub-module exposes 2D plotters consuming the result types of the
quantum sub-domain it visualises:

- :mod:`asymsafety.quantum.visualization.koopman_plots` — Koopman
  spectrum, modes, QFT power spectra.
- :mod:`asymsafety.quantum.visualization.grover_plots` — Grover
  success-probability curve, measurement-distribution heatmap.
- :mod:`asymsafety.quantum.visualization.vqrg_plots` — VQRG cost
  landscape, optimisation trajectory, quantum Fisher spectrum.
- :mod:`asymsafety.quantum.visualization.thermal_plots` — partition
  function, Seeley-DeWitt extraction, Gibbs-state purity.

3D variants (``vqrg_cost_isosurface_3d``, ``koopman_spectrum_3d``)
live with the rest of the 3D plotters in
:mod:`asymsafety.gui.visualization_3d`.

All plotters reuse the canonical style from
:mod:`asymsafety.visualization.style`. Functions whose backend depends
on qiskit (Grover measurement-distribution, QFT power-spectrum) raise
``ImportError`` only when actually executed, mirroring the lazy-import
pattern of the underlying quantum modules.
"""

from asymsafety.quantum.visualization.grover_plots import (
    plot_grover_measurement_distribution,
    plot_grover_success_probability,
)
from asymsafety.quantum.visualization.koopman_plots import (
    plot_koopman_modes,
    plot_koopman_spectrum,
    plot_qft_power_spectrum,
)
from asymsafety.quantum.visualization.thermal_plots import (
    plot_gibbs_state_purity,
    plot_partition_function,
    plot_seeley_dewitt_extraction,
)
from asymsafety.quantum.visualization.vqrg_plots import (
    plot_quantum_fisher_eigenvalues,
    plot_vqrg_cost_landscape,
    plot_vqrg_optimization_trajectory,
)

__all__ = [
    "plot_koopman_spectrum",
    "plot_koopman_modes",
    "plot_qft_power_spectrum",
    "plot_grover_success_probability",
    "plot_grover_measurement_distribution",
    "plot_vqrg_cost_landscape",
    "plot_vqrg_optimization_trajectory",
    "plot_quantum_fisher_eigenvalues",
    "plot_partition_function",
    "plot_seeley_dewitt_extraction",
    "plot_gibbs_state_purity",
]
