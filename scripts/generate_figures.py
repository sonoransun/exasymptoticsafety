#!/usr/bin/env python3
"""Generate all documentation figures for the Asymptotic Safety Explorer.

Usage:
    PYTHONPATH=src python3 scripts/generate_figures.py
    PYTHONPATH=src python3 scripts/generate_figures.py --output-dir docs/images --format png

The generated images are committed to docs/images/ so that the README
renders on GitHub without running this script. Re-run whenever the
visualization code changes.

Each PNG is accompanied by a sibling ``<name>.caption.md`` carrying a
short physics description, the canonical references, and links to the
relevant README and ``docs/LITERATURE.md`` sections. This keeps figure
provenance discoverable directly from ``docs/images/``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# Headless backend -- must be set before any pyplot import.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Captions metadata
# ---------------------------------------------------------------------------


@dataclass
class FigureCaption:
    """Metadata written alongside each generated figure."""

    title: str
    description: str
    references: list[str] = field(default_factory=list)
    see_also: list[str] = field(default_factory=list)
    literature_anchors: list[str] = field(default_factory=list)

    def render(self, image_filename: str) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"![{self.title}](./{image_filename})",
            "",
            self.description.strip(),
            "",
        ]
        if self.references:
            lines.append("## References")
            lines.append("")
            for ref in self.references:
                lines.append(f"- {ref}")
            lines.append("")
        if self.literature_anchors:
            lines.append("## See in `docs/LITERATURE.md`")
            lines.append("")
            for anchor in self.literature_anchors:
                lines.append(f"- [{anchor}](../LITERATURE.md#{anchor})")
            lines.append("")
        if self.see_also:
            lines.append("## See also")
            lines.append("")
            for entry in self.see_also:
                lines.append(f"- {entry}")
            lines.append("")
        return "\n".join(lines)


_CAPTIONS: dict[str, FigureCaption] = {
    "asymptotic_safety_concept": FigureCaption(
        title="Asymptotic safety: the idea",
        description=(
            "Conceptual overview. Left: theory space showing the Gaussian "
            "(GFP) and non-Gaussian (NGFP) fixed points, RG flow trajectories, "
            "and the UV-critical surface. Right: the dimensionless Newton "
            "coupling `g(k) = G(k) k^2` — perturbative gravity diverges at a "
            "Landau pole, while asymptotic safety yields a finite UV limit at "
            "`g* ~ 0.7`."
        ),
        references=[
            "Weinberg (1979), in *General Relativity: An Einstein Centenary "
            "Survey*, 790 — original conjecture.",
            "Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].",
            "Bonanno et al. (2020), Front. Phys. 8, 269 [2004.06810] — review.",
        ],
        literature_anchors=["reviews"],
        see_also=[
            "[`README.md`](../../README.md#asymptotic-safety-the-idea)",
            "`asymsafety.visualization.conceptual.asymptotic_safety_concept`",
            "`docs/images/eh_phase_portrait.caption.md` for the data-driven counterpart.",
        ],
    ),
    "wetterich_equation": FigureCaption(
        title="The Wetterich equation",
        description=(
            "Schematic of the exact one-loop functional RG equation. The circle "
            "represents the trace over the full regularised propagator "
            "`(Gamma_k^(2) + R_k)^{-1}`, with a single regulator insertion "
            "`d_t R_k` (cross). Only modes near the running scale `k` "
            "contribute to the flow."
        ),
        references=[
            "Wetterich (1993), Phys. Lett. B 301, 90.",
            "Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].",
        ],
        literature_anchors=["reviews"],
        see_also=[
            "`asymsafety.visualization.conceptual.wetterich_equation_diagram`",
        ],
    ),
    "regulator_comparison": FigureCaption(
        title="Regulator shape functions",
        description=(
            "Comparison of the Litim (optimised, step-function) and "
            "Exponential (smooth) IR regulators. The Litim regulator yields "
            "closed-form beta functions and minimises scheme dependence; the "
            "Exponential regulator is `C^infty` but requires numerical "
            "integration."
        ),
        references=[
            "Wetterich (1993), Phys. Lett. B 301, 90.",
            "Litim (2001), Phys. Rev. D 64, 105007 [hep-th/0103195].",
        ],
        see_also=[
            "`asymsafety.visualization.phase_portrait.regulator_comparison_panels` "
            "— phase-portrait-level scheme comparison.",
        ],
    ),
    "fp_stability_concept": FigureCaption(
        title="Fixed-point stability and predictivity",
        description=(
            "Linearised RG flow near a non-Gaussian fixed point. Blue "
            "eigenvectors are relevant directions (`Re theta > 0`, "
            "UV-attractive); orange are irrelevant. The shaded strip is the "
            "UV-critical surface. The number of relevant directions equals "
            "the number of free parameters of the asymptotically safe theory."
        ),
        references=[
            "Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].",
            "Bonanno et al. (2020), Front. Phys. 8, 269 [2004.06810].",
        ],
        see_also=[
            "`asymsafety.gui.visualization_3d.fixed_point_stability_3d` — "
            "data-driven 3D variant.",
        ],
    ),
    "cross_analogue_bridge": FigureCaption(
        title="Cross-analogue bridge",
        description=(
            "Commutative diagram showing how different mathematical "
            "representations of RG flow (classical RG, hydraulic / control, "
            "integral transforms, quantum computing) are connected. All "
            "paths through the diagram yield consistent critical exponents "
            "`theta_i`."
        ),
        references=[
            "Wetterich (1993), Phys. Lett. B 301, 90.",
            "Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].",
        ],
        see_also=[
            "[`README.md`](../../README.md#physical-computational-analogues)",
            "`asymsafety.transforms.bridge.cross_analogue.CrossAnalogueBridge`",
        ],
    ),
    "hydraulic_analogy": FigureCaption(
        title="The hydraulic analogy",
        description=(
            "Couplings ↔ pressures, beta functions ↔ flow equations, fixed "
            "points ↔ steady states, regulators ↔ valves, critical exponents "
            "↔ impedance eigenvalues."
        ),
        references=[],
        see_also=[
            "[`README.md`](../../README.md#physical-computational-analogues)",
            "`asymsafety.hydraulic.mapping.RGToHydraulicMapper`",
        ],
    ),
    "eh_phase_portrait": FigureCaption(
        title="Annotated Einstein–Hilbert phase portrait",
        description=(
            "RG flow in the (g, lambda) coupling plane (d = 4, Litim "
            "regulator). The non-Gaussian UV fixed point (NGFP, green "
            "star) at g* ~ 0.707, lambda* ~ 0.193 has two relevant "
            "directions with the complex-conjugate critical-exponent pair "
            "theta = 1.475 ± 3.043i (blue eigenvector arrows, annotated "
            "with the theta value), so the streamlines spiral into the "
            "fixed point — the asymptotic-safety candidate of Reuter "
            "(1998). The Gaussian fixed point (GFP, white circle) sits at "
            "the origin. The crimson curve emanating from the NGFP is the "
            "back-integrated UV-critical separatrix (its outer windings "
            "blend into the colored trajectory bundle). The lambda = 1/2 "
            "propagator pole lies outside the plotted window "
            "(lambda <= 0.45)."
        ),
        references=[
            "Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].",
            "Lauscher & Reuter (2002), Phys. Rev. D 65, 025013 [hep-th/0108040].",
            "Litim (2001), Phys. Rev. D 64, 105007 [hep-th/0103195].",
        ],
        literature_anchors=["reviews"],
        see_also=[
            "[`README.md`](../../README.md#rg-flow-in-the-einsteinhilbert-truncation)",
            "`asymsafety.visualization.phase_portrait.annotated_eh_phase_portrait`",
            "`asymsafety.validation.reuter_1998` — benchmark coordinates.",
        ],
    ),
    "running_couplings": FigureCaption(
        title="Running couplings",
        description=(
            "Evolution of the dimensionless Newton coupling g and "
            "cosmological constant lambda as a function of RG time "
            "`t = log(k/k_0)` for several initial conditions in the "
            "Einstein–Hilbert truncation."
        ),
        references=[
            "Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].",
        ],
        see_also=[
            "`asymsafety.visualization.phase_portrait.flow_diagram`",
            "`docs/images/3d_flow.caption.md` — 3D world-line companion.",
        ],
    ),
    "3d_flow": FigureCaption(
        title="3D RG flow trajectories (world-line view)",
        description=(
            "Einstein–Hilbert RG trajectories in the (g, lambda, t) "
            "world-line view: couplings on the (x, y) plane, RG time "
            "`t = log(k/k_0)` on the z axis. Trajectory colour goes from "
            "blue (IR) to red (UV); thickness is modulated by local flow "
            "speed `tanh(|beta|)`. Fixed points are marked: the Gaussian "
            "FP at the origin and the Reuter NGFP at "
            "`g* ~ 0.707, lambda* ~ 0.193` (two relevant directions, "
            "`theta = 1.475 ± 3.043i`); the trajectories wind helically "
            "onto the NGFP world-line toward the UV, the real-space "
            "signature of the complex exponent pair."
        ),
        references=[
            "Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].",
            "Reuter & Saueressig (2012), New J. Phys. 14, 055022 [1202.2274].",
        ],
        see_also=[
            "`asymsafety.gui.visualization_3d.flow_trajectories_3d`",
            "`docs/images/eh_phase_portrait.caption.md` — 2D phase-plane companion.",
        ],
    ),
    "hydraulic_network": FigureCaption(
        title="Hydraulic network from the EH RG flow",
        description=(
            "Auto-generated hydraulic network derived from the "
            "Einstein–Hilbert beta-function system at a reference NGFP "
            "guess. Nodes are coloured by pressure (the analogue of "
            "coupling value); pipe widths scale with diameter."
        ),
        references=[],
        see_also=[
            "`asymsafety.hydraulic.mapping.RGToHydraulicMapper`",
            "`asymsafety.hydraulic.visualization.plot_hydraulic_network`",
        ],
    ),
    "foliated_3d": FigureCaption(
        title="Foliated EH 3D phase portrait",
        description=(
            "3D phase portrait of the toolkit's schematic foliated "
            "Einstein–Hilbert truncation in `(g, lambda, lambda_ADM)` "
            "coupling space on `S^1 x S^3`. The highlighted plane "
            "`lambda_ADM = 1` is a fixed plane of the flow by construction "
            "(`beta_lambda_ADM ∝ g (lambda_ADM - 1)`), UV-repulsive at "
            "physical couplings. This truncation admits **no non-Gaussian "
            "fixed point with `g > 0`**: the only fixed point found — and "
            "marked here (circle) — is the *Gaussian* one at "
            "`g = lambda = 0` on the `lambda_ADM = 1` plane. The published "
            "foliated NGFP of Manrique et al. (2011) (Euclidean "
            "`g* ~ 0.19, lambda* ~ 0.31`, with `lambda_ADM = 1` imposed by "
            "their full-Diff ansatz) is a literature reference, not a root "
            "of this system."
        ),
        references=[
            "Manrique, Rechenberger & Saueressig (2011), "
            "Phys. Rev. Lett. 106, 251302 [1003.5129].",
            "Biemans, Platania & Saueressig (2017), JHEP 05, 093 [1609.02803].",
            "Saueressig et al. (2025), Phys. Rev. D 111, 106007 [2501.03752].",
        ],
        literature_anchors=["lorentzian-foliated"],
        see_also=[
            "`asymsafety.gui.visualization_3d.foliated_phase_portrait_3d`",
            "`asymsafety.validation.manrique_2011`",
        ],
    ),
    "fp_stability_3d": FigureCaption(
        title="Fixed-point stability (3D)",
        description=(
            "Zoomed view of the Reuter NGFP with eigenvector arrows. Blue = "
            "relevant (`Re theta > 0`), orange = irrelevant. Arrow thickness "
            "and alpha are proportional to `|Re theta|`; dashed arcs at the "
            "tips of complex-conjugate pairs indicate spiral flow. Each "
            "arrow tip carries the numeric value of `theta_i`."
        ),
        references=[
            "Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].",
            "Lauscher & Reuter (2002), Phys. Rev. D 65, 025013 [hep-th/0108040].",
            "Bonanno et al. (2020), Front. Phys. 8, 269 [2004.06810].",
        ],
        see_also=[
            "`asymsafety.gui.visualization_3d.fixed_point_stability_3d`",
        ],
    ),
    "matter_continuation": FigureCaption(
        title="NGFP versus matter content",
        description=(
            "Continuation of the NGFP coordinates `g*`, `lambda*` (and "
            "the running scalar quartic `lambda_phi*`) and the real parts "
            "of the critical exponents as the number of minimally coupled "
            "scalars `N_s` is varied in the gravity+matter truncation. "
            "The pure-gravity limit `N_s = 0` reproduces the EH NGFP "
            "`g* ~ 0.707, lambda* ~ 0.193`; increasing `N_s` raises "
            "`lambda*` and mildly lowers `g*`. Two of the three "
            "exponents form a complex-conjugate pair (Re(theta) ~ 1.48 "
            "at `N_s = 0`, rising with `N_s`) whose Re curves coincide; "
            "the third is real (~0.7-0.8). The dashed line and shading "
            "start at the "
            "foliated existence wedge of Korver, Saueressig & Wang "
            "(2024), `N_s + 6.4 N_v <= 23.1`, i.e. `N_s ~ 23` at "
            "`N_v = 0` — beyond the sweep's largest `N_s = 12`."
        ),
        references=[
            "Dona, Eichhorn & Percacci (2014), Phys. Rev. D 89, 084035 "
            "[1311.2898].",
            "Eichhorn & Schiffer (2022), in *Handbook of Quantum Gravity* "
            "[2212.07456].",
            "Korver, Saueressig & Wang (2024), Phys. Lett. B 855, 138789 "
            "[2402.01260].",
        ],
        literature_anchors=["matter-coupling"],
        see_also=[
            "`asymsafety.visualization.fixed_point_plot.plot_matter_content_continuation`",
            "`asymsafety.validation.korver_2024`",
        ],
    ),
    "flow_basin_3d": FigureCaption(
        title="Basin of attraction near the NGFP (3D)",
        description=(
            "Monte-Carlo sample of forward (IR-directed) trajectories "
            "starting on a sphere of radius `r = 0.25` around the Reuter "
            "NGFP. Blue points stayed bounded (captured); orange points "
            "escaped beyond a distance threshold of `2.0`. The captured "
            "cloud approximates the local basin geometry projected onto "
            "the `(g, lambda, t)` axes."
        ),
        references=[
            "Reuter & Saueressig (2012), New J. Phys. 14, 055022 [1202.2274].",
            "Bonanno et al. (2020), Front. Phys. 8, 269 [2004.06810].",
        ],
        see_also=[
            "`asymsafety.gui.visualization_3d.flow_basin_3d`",
            "`asymsafety.gui.visualization_3d.fixed_point_stability_3d` — "
            "linearised counterpart.",
        ],
    ),
    "quadratic_pairwise": FigureCaption(
        title="Quadratic gravity: pairwise phase portraits",
        description=(
            "Six pairwise streamplots projecting the quadratic-gravity "
            "`(g, lambda, alpha, beta)` flow onto every pair of coupling "
            "axes. The one-loop truncation has no interior NGFP; the "
            "frozen couplings are anchored at literature reference values "
            "(Codello et al. 2009) that this truncation does not itself "
            "reproduce. The panels containing `beta` show the constant "
            "one-loop running of the Weyl-squared coefficient (+133/20 "
            "per 16 pi^2, asymptotic freedom of the C^2 sector)."
        ),
        references=[
            "Codello, Percacci & Rahmede (2009), Ann. Phys. 324, 414 [0812.0785].",
        ],
        literature_anchors=["quadratic"],
        see_also=[
            "`asymsafety.visualization.phase_portrait.quadratic_pairwise_grid`",
            "`asymsafety.validation.codello_2009`",
        ],
    ),
    # ------------------------------------------------------------------
    # Transform domain
    # ------------------------------------------------------------------
    "bode": FigureCaption(
        title="Bode plot of the EH stability matrix",
        description=(
            "Magnitude (in dB) and phase (in degrees) of the (0, 0) entry "
            "of the impedance transfer function `H(s) = (sI - M)^{-1}` "
            "at the Reuter NGFP. Poles in the right half-plane correspond "
            "to UV-attractive (relevant) directions; their location on "
            "the imaginary axis sets the dominant resonance frequencies."
        ),
        references=[
            "Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].",
            "Trefethen & Embree (2005), *Spectra and Pseudospectra*.",
        ],
        literature_anchors=["reviews"],
        see_also=[
            "`asymsafety.transforms.bridge.impedance.ImpedanceBridge`",
            "`asymsafety.transforms.visualization.transform_plots.plot_bode`",
        ],
    ),
    "scalogram": FigureCaption(
        title="Wavelet scalogram of the running Newton coupling",
        description=(
            "Continuous wavelet transform `|W(a, b)|^2` of the EH "
            "running coupling g(t) using a Morlet mother wavelet. "
            "Bright bands at small scales mark rapid crossover; the "
            "fading toward large scales signals the asymptotic plateau "
            "near the NGFP."
        ),
        references=[
            "Reuter & Saueressig (2012), New J. Phys. 14, 055022 [1202.2274].",
            "Daubechies (1992), *Ten Lectures on Wavelets*.",
        ],
        see_also=[
            "`asymsafety.transforms.integral.wavelet.RGFlowWavelet`",
            "`asymsafety.transforms.visualization.transform_plots.plot_scalogram`",
        ],
    ),
    "pseudospectrum": FigureCaption(
        title="Resolvent pseudospectrum of the EH stability matrix",
        description=(
            "Filled-contour `log10 ||(sI - M)^{-1}||` with three "
            "epsilon-pseudospectrum boundaries (white contours) and the "
            "exact eigenvalues (white crosses). Thin contours indicate "
            "robust critical exponents; broad contours flag non-normal "
            "sensitivity that small truncation / regulator changes "
            "amplify."
        ),
        references=[
            "Trefethen & Embree (2005), *Spectra and Pseudospectra*.",
            "Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].",
            "Bonanno et al. (2020), Front. Phys. 8, 269 [2004.06810].",
        ],
        literature_anchors=["reviews"],
        see_also=[
            "`asymsafety.transforms.linear.resolvent.ResolventOperator`",
            "`asymsafety.transforms.visualization.transform_plots.plot_pseudospectrum`",
        ],
    ),
    "comparison_table": FigureCaption(
        title="Critical exponents — cross-method comparison",
        description=(
            "Side-by-side bars of `Re(theta_i)` from every analogue path "
            "of the cross-analogue bridge: classical RG stability, "
            "transfer-matrix exponentiation, resolvent poles, and the "
            "hydraulic-impedance steady state. Bars within each group "
            "must match for the bridge to commute."
        ),
        references=[
            "Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].",
            "Codello, Percacci & Rahmede (2009) [0812.0785].",
        ],
        literature_anchors=["matter-coupling", "reviews"],
        see_also=[
            "`asymsafety.transforms.bridge.cross_analogue.CrossAnalogueBridge`",
            "`asymsafety.transforms.visualization.transform_plots.plot_comparison_table`",
        ],
    ),
    # ------------------------------------------------------------------
    # Cosmology
    # ------------------------------------------------------------------
    "running_newton_constant": FigureCaption(
        title="RG-improved Newton constant on Schwarzschild",
        description=(
            "Log-log plot of the position-dependent Newton constant "
            "`G(r) = g(k(r))/k(r)^2` along an EH trajectory through the "
            "Reuter NGFP, with the classical IR value as a dashed "
            "reference and the Planck length marked."
        ),
        references=[
            "Bonanno & Reuter (2000), Phys. Rev. D 62, 043008 [hep-th/0002196].",
        ],
        literature_anchors=["black-holes-cosmology"],
        see_also=[
            "`asymsafety.cosmology.rg_improved_bh.RGImprovedSchwarzschild`",
            "`asymsafety.cosmology.visualization.plot_running_newton_constant`",
        ],
    ),
    "lapse_with_horizons": FigureCaption(
        title="RG-improved Schwarzschild lapse with horizons",
        description=(
            "Lapse `f(r) = 1 - 2 G(r) M / r` for a super-critical mass "
            "`M = 8` on a trajectory with IR Newton constant `G_N = 0.02` "
            "(Planck mass `M_pl = G_N^{-1/2} ~ 7.07`, critical mass "
            "`M_crit ~ 5.7`). The running `G(r) = g(k(r))/k(r)^2` is "
            "constant in the IR and softens like `g* r^2` inside the "
            "Planck radius, so both Bonanno-Reuter horizons appear: the "
            "inner Cauchy horizon `r_-` and the outer event horizon `r_+` "
            "(dashed vertical lines). As `r -> 0` the lapse returns to 1 "
            "only *linearly* with the `k = 1/r` cutoff used here: the "
            "central singularity is softened (curvature still diverges, "
            "but much more mildly than classically), not replaced by a "
            "regular de Sitter core."
        ),
        references=[
            "Bonanno & Reuter (2000), Phys. Rev. D 62, 043008 [hep-th/0002196].",
            "Platania (2023), in *Handbook of Quantum Gravity* [2302.04272].",
        ],
        literature_anchors=["black-holes-cosmology"],
        see_also=[
            "`asymsafety.cosmology.visualization.plot_lapse_with_horizons`",
        ],
    ),
    "classical_vs_rg_lapse": FigureCaption(
        title="Classical vs RG-improved lapse for several masses",
        description=(
            "Overlay of the classical Schwarzschild lapse "
            "`1 - 2 G_N M / r` (dashed) and the RG-improved lapse (solid) "
            "for `M = 3, 6, 9` on a trajectory with `G_N = 0.02` "
            "(`M_crit ~ 5.7`). The sub-critical `M = 3` curve stays "
            "positive at all radii — a horizonless naked-mass remnant — "
            "while its classical counterpart still crosses zero; `M = 6`, "
            "just above critical, dips barely below zero (nearly merged "
            "inner/outer horizon pair); `M = 9` shows two well-separated "
            "horizons. At small `r` the RG-improved lapse always lies "
            "above the classical one (`G(r) <= G_N`) and rejoins it in "
            "the IR."
        ),
        references=[
            "Bonanno & Reuter (2000) [hep-th/0002196].",
            "Platania (2023) [2302.04272].",
        ],
        literature_anchors=["black-holes-cosmology"],
        see_also=[
            "`asymsafety.cosmology.visualization.plot_classical_vs_rg_lapse`",
        ],
    ),
    "hawking_temperature": FigureCaption(
        title="Hawking temperature: classical vs asymptotic safety",
        description=(
            "Hawking temperature `T_H(M) = |f'(r_+)| / (4 pi)` versus ADM "
            "mass on a trajectory with `G_N = 0.02` (`M_crit ~ 5.7`). At "
            "large `M` the RG-improved temperature (solid) tracks the "
            "classical `1/(8 pi G_N M)` curve (dashed); as `M` decreases "
            "it instead peaks near `~1.3 M_crit` (star) and then turns "
            "over, falling toward the extremal limit `T_H -> 0` as the "
            "two horizons merge at `M_crit` (dotted vertical line; cold "
            "remnant). No temperature is drawn below `M_crit`, where no "
            "horizon exists — in contrast with the classical divergence "
            "as `M -> 0`."
        ),
        references=[
            "Bonanno & Reuter (2000) [hep-th/0002196].",
            "Platania (2023) [2302.04272].",
        ],
        literature_anchors=["black-holes-cosmology"],
        see_also=[
            "`asymsafety.cosmology.visualization.plot_hawking_temperature`",
        ],
    ),
    "scale_identification": FigureCaption(
        title="Scale-identification prescriptions",
        description=(
            "Log-log overlay of the three built-in `k(r)` choices: "
            "InverseDistance (`k = ξ/r`), Geodesic-softened "
            "(`k = ξ/√(r^2 + δ^2)`), and ProperDistance from a horizon. "
            "Choice affects the regularisation of the origin and the "
            "depth of the de Sitter core."
        ),
        references=[
            "Reuter & Saueressig (2012), New J. Phys. 14, 055022 [1202.2274].",
        ],
        see_also=[
            "`asymsafety.cosmology.scale_identification`",
        ],
    ),
    "flrw_evolution": FigureCaption(
        title="RG-improved FLRW evolution (a, H, running couplings)",
        description=(
            "Three-panel summary of an RG-improved FLRW universe with "
            "Hubble matching `k = ξ H`. Top: scale factor `a(t)` on a "
            "log scale; middle: Hubble rate `H(t)`; bottom: the running "
            "couplings `G(t)/G_max` and `Λ(t)/|Λ|_max`."
        ),
        references=[
            "Bonanno & Reuter (2002), Phys. Rev. D 65, 043508 [hep-th/0106133].",
        ],
        literature_anchors=["black-holes-cosmology"],
        see_also=[
            "`asymsafety.cosmology.rg_improved_flrw.RGImprovedFLRW`",
        ],
    ),
    # ------------------------------------------------------------------
    # Quantum
    # ------------------------------------------------------------------
    "koopman_spectrum": FigureCaption(
        title="Koopman spectrum (EDMD)",
        description=(
            "Eigenvalues of the Koopman operator computed from EH "
            "trajectories via Extended Dynamic Mode Decomposition with "
            "a degree-4 monomial dictionary. Eigenvalues outside the "
            "unit circle correspond to growing observables (relevant "
            "directions); inside, decaying observables (irrelevant)."
        ),
        references=[
            "Williams, Kevrekidis & Rowley (2015) [1408.4408].",
            "Mezic (2020) [1906.07401].",
        ],
        see_also=[
            "`asymsafety.quantum.koopman.operator.KoopmanOperator`",
        ],
    ),
    "koopman_modes": FigureCaption(
        title="Koopman modes (top-k by eigenvalue magnitude)",
        description=(
            "Stacked bar chart of the top eight Koopman modes' projection "
            "onto each EH coupling. Identifies the dominant Koopman "
            "eigenfunction's coupling support — useful for picking which "
            "observables are most informative about the long-time flow."
        ),
        references=[
            "Williams, Kevrekidis & Rowley (2015) [1408.4408].",
        ],
        see_also=[
            "`asymsafety.quantum.koopman.operator.KoopmanOperator`",
        ],
    ),
    "qft_power_spectrum": FigureCaption(
        title="QFT power spectrum of the running Newton coupling",
        description=(
            "Quantum-Fourier transform amplitude spectrum of g(t) along "
            "an EH trajectory, with dominant peaks highlighted. The peak "
            "frequencies correspond to the imaginary parts of the "
            "complex critical exponents at the Reuter NGFP."
        ),
        references=[
            "Reuter & Saueressig (2012), New J. Phys. 14, 055022 [1202.2274].",
        ],
        see_also=[
            "`asymsafety.quantum.koopman.qft_rg.QuantumFourierRG`",
        ],
    ),
    "grover_success_probability": FigureCaption(
        title="Grover success probability vs iteration count",
        description=(
            "Theoretical Grover success probability "
            "`P(k) = sin^2((2k+1) arcsin sqrt(M/N))` for an EH coupling "
            "grid, with the optimal iteration count `k_opt` highlighted. "
            "The monotonic-then-oscillatory shape is the diagnostic "
            "signature that justifies the sqrt(N) speedup."
        ),
        references=[
            "Grover (1996), [quant-ph/9605043].",
        ],
        see_also=[
            "`asymsafety.quantum.grover.search.GroverFixedPointSearch`",
        ],
    ),
    "grover_measurement_distribution": FigureCaption(
        title="Grover measurement distribution",
        description=(
            "Heatmap of measurement counts from an EH Grover run after "
            "the optimal number of iterations, with the classically "
            "refined NGFP marked by a green star. Concentration on the "
            "marked grid points demonstrates the amplitude amplification."
        ),
        references=[
            "Grover (1996), [quant-ph/9605043].",
        ],
        see_also=[
            "`asymsafety.quantum.grover.search.GroverFixedPointSearch`",
        ],
    ),
    "vqrg_cost_landscape": FigureCaption(
        title="VQRG cost landscape (2-parameter slice)",
        description=(
            "Filled `log10 C` contour of the variational-quantum RG cost "
            "function on a 2-parameter slice of the ansatz, with all "
            "other parameters held at zero. Deep wells correspond to "
            "RG fixed points (`C = 0`)."
        ),
        references=[
            "Cerezo et al. (2021) [2012.09265].",
            "Stokes et al. (2020) [1909.02108].",
        ],
        see_also=[
            "`asymsafety.quantum.vqrg.cost.VQRGCostFunction`",
        ],
    ),
    "vqrg_optimization_trajectory": FigureCaption(
        title="VQRG optimisation trajectory",
        description=(
            "Cost vs iteration on a representative quantum-natural-"
            "gradient run; inset: the trajectory in the first two "
            "circuit-parameter coordinates. Bare-bones convergence "
            "diagnostic for VQRG users."
        ),
        references=[
            "Stokes et al. (2020) [1909.02108].",
        ],
        see_also=[
            "`asymsafety.quantum.vqrg.optimizer.QuantumNaturalGradient`",
        ],
    ),
    "quantum_fisher_eigenvalues": FigureCaption(
        title="Quantum Fisher / Zamolodchikov eigenspectrum",
        description=(
            "Sorted eigenvalues of the QFI matrix (semi-log-y) plus the "
            "QFI heatmap. Spread between top and bottom eigenvalues is "
            "the Zamolodchikov-metric anisotropy near the NGFP — small "
            "eigenvalues indicate flat directions (parameter degeneracies)."
        ),
        references=[
            "Meyer (2021) [2103.15191].",
            "Stokes et al. (2020) [1909.02108].",
        ],
        see_also=[
            "`asymsafety.quantum.vqrg.fisher.QuantumFisherInformation`",
        ],
    ),
    "partition_function": FigureCaption(
        title="Heat-kernel partition function and free energy",
        description=(
            "Two-panel plot of `Z(beta) = Tr exp(-beta H)` (log-log) and "
            "the free energy `F(beta) = -log Z / beta` for the "
            "scalar-field Laplacian on `S^4`. The high-temperature heat-"
            "kernel asymptote `(4 pi beta)^{-d/2} b_0` is overlaid."
        ),
        references=[
            "Vassilevich (2003), Phys. Rept. 388, 279 [hep-th/0306138].",
        ],
        see_also=[
            "`asymsafety.quantum.thermal.partition.PartitionFunctionEstimator`",
        ],
    ),
    "seeley_dewitt_extraction": FigureCaption(
        title="Seeley-DeWitt coefficients: quantum vs classical",
        description=(
            "Bar chart comparing quantum-extracted heat-kernel coefficients "
            "`b_0, b_2, b_4` (from polynomial fitting of `Z(beta)`) with "
            "their classical Vassilevich values. The deviation labels above "
            "each pair quantify the truncation error."
        ),
        references=[
            "Vassilevich (2003) [hep-th/0306138].",
        ],
        see_also=[
            "`asymsafety.frg.heat_kernel.SeeleyDeWittCoefficients`",
        ],
    ),
    "gibbs_state_purity": FigureCaption(
        title="Gibbs-state purity vs inverse temperature",
        description=(
            "Purity `Tr(rho^2)` of the diagonal Gibbs state of the "
            "scalar Laplacian on `S^4` as a function of inverse "
            "temperature `beta`. Crosses over from the maximally-mixed "
            "floor `1/d_eff` (high T) to the pure ground state (low T)."
        ),
        references=[
            "Vassilevich (2003) [hep-th/0306138].",
        ],
        see_also=[
            "`asymsafety.quantum.thermal.gibbs.GibbsStatePreparer`",
        ],
    ),
    # ------------------------------------------------------------------
    # 3D extensions
    # ------------------------------------------------------------------
    "koopman_spectrum_3d": FigureCaption(
        title="Koopman spectrum (3D embedding)",
        description=(
            "3D scatter of Koopman eigenvalues lifted to "
            "`(Re mu, Im mu, |mu|)` with the unit cylinder drawn as the "
            "stability boundary. Resolves degeneracies hidden by the "
            "flat 2D spectrum view."
        ),
        references=[
            "Mezic (2020) [1906.07401].",
        ],
        see_also=[
            "`asymsafety.gui.visualization_3d.koopman_spectrum_3d`",
        ],
    ),
    "vqrg_cost_isosurface_3d": FigureCaption(
        title="VQRG cost isosurfaces (3D parameter slice)",
        description=(
            "Translucent iso-cost surfaces over a 3-parameter slice of "
            "the VQRG ansatz. Innermost surface marks the basin of the "
            "global minimum; outer surfaces show how the basin grows "
            "with cost threshold."
        ),
        references=[
            "Cerezo et al. (2021) [2012.09265].",
        ],
        see_also=[
            "`asymsafety.gui.visualization_3d.vqrg_cost_isosurface_3d`",
        ],
    ),
    # ------------------------------------------------------------------
    # Spectral / heat-kernel
    # ------------------------------------------------------------------
    "spectral_sum_convergence": FigureCaption(
        title="Spectral-sum convergence on `S^4`",
        description=(
            "Trace of the Litim weight `W(z) = 1/(1 + z)` summed over "
            "the scalar Laplacian eigenvalues on `S^4` as a function of "
            "the angular-momentum cutoff `l_max`. Saturation diagnoses "
            "the smallest defensible truncation."
        ),
        references=[
            "Manrique et al. (2011), Phys. Rev. Lett. 106, 251302 [1003.5129].",
        ],
        see_also=[
            "`asymsafety.frg.spectral.SpectralSumEvaluator`",
        ],
    ),
    "heat_kernel_coefficients": FigureCaption(
        title="Seeley-DeWitt coefficients per field type",
        description=(
            "Bar chart of `b_0, b_2, b_4` for scalar, vector, and TT "
            "fields on `S^4`. Reference values from Vassilevich's "
            "heat-kernel manual."
        ),
        references=[
            "Vassilevich (2003), Phys. Rept. 388, 279 [hep-th/0306138].",
        ],
        see_also=[
            "`asymsafety.frg.heat_kernel.SeeleyDeWittCoefficients`",
        ],
    ),
    # Cross-analogue: 3D gauge-Higgs (Bonati, Pelissetto & Vicari 2025)
    "ahm_phase_diagram": FigureCaption(
        title="3D Abelian-Higgs phase diagram",
        description=(
            "Schematic of the three phases of the 3D Abelian-Higgs model "
            "(Coulomb, Higgs, Molecular) meeting at a multicritical point "
            "with emergent O(2) symmetry. The Coulomb-to-Higgs transition "
            "is controlled by the *charged fixed point* when `N_f > N_f^*`. "
            "Cross-analogue companion of `asymptotic_safety_concept`."
        ),
        references=[
            "Bonati, Pelissetto & Vicari (2025), Phys. Rep. [arXiv:2410.05823].",
            "Lattice AHM with noncompact gauge fields [arXiv:2010.06311].",
        ],
        see_also=[
            "[`docs/cross-analogue-gauge-higgs.md`](../cross-analogue-gauge-higgs.md)",
            "`asymsafety.transforms.bridge.gauge_higgs.GaugeHiggsAnalogue`",
        ],
    ),
    "charged_fp_boundary": FigureCaption(
        title="Charged fixed point: existence boundary",
        description=(
            "Schematic of `N_f^*(d)`, the matter threshold above which "
            "the charged fixed point of 3D gauge-Higgs theory is stable. "
            "The 4-ε prediction `N_f^* ≈ 375` (SU(2)) sits far above the "
            "lattice value (Bonati et al. 2025 place it well below 30 in "
            "d=3). The shaded region is the stable phase; the green points "
            "are the published SU(2) MC sample `N_f ∈ {30, 40, 60}`."
        ),
        references=[
            "Bonati, Pelissetto & Vicari (2025), Phys. Rep. [arXiv:2410.05823].",
        ],
        see_also=[
            "[`docs/cross-analogue-gauge-higgs.md`](../cross-analogue-gauge-higgs.md)",
            "`asymsafety.validation.bonati_2025.validate_charged_fp_existence`",
        ],
    ),
    "nu_vs_nf": FigureCaption(
        title="ν(N_f) at the charged fixed point",
        description=(
            "Correlation-length exponent at the charged FP of 3D SU(2) + "
            "`N_f` gauge-Higgs. Three curves overlaid: the toolkit's "
            "one-loop 4-ε prediction (approaches the Wilson-Fisher limit "
            "ν → 1/2), the large-`N_f` field-theory asymptote "
            "ν = 1 − 9.727/N_f (approaches 1), and Bonati et al.'s lattice "
            "MC points at `N_f ∈ {30, 40, 60}` with their published error "
            "bars. The two windows on the physics agree qualitatively on "
            "FP existence but live in different quantitative regimes — "
            "a faithful echo of the regulator/scheme debate around the "
            "gravitational NGFP in asymptotic safety."
        ),
        references=[
            "Bonati, Pelissetto & Vicari (2025), Phys. Rep. [arXiv:2410.05823].",
            "Halperin, Lubensky & Ma (1974), Phys. Rev. Lett. 32, 292.",
        ],
        see_also=[
            "[`docs/cross-analogue-gauge-higgs.md`](../cross-analogue-gauge-higgs.md)",
            "`asymsafety.transforms.bridge.gauge_higgs.correlation_length_exponent`",
            "`asymsafety.validation.bonati_2025.BONATI_SU2_MC`",
        ],
    ),
    "ahm_as_bridge": FigureCaption(
        title="AHM ↔ asymptotic safety: cross-analogue dictionary",
        description=(
            "Two-column commutative diagram mapping the 3D Abelian-Higgs "
            "charged fixed point onto the asymptotic-safety NGFP. Three "
            "rows of correspondences (interacting UV fixed point; "
            "stability-matrix eigenvalues; matter/regulator dependence) "
            "are joined by double-headed arrows. The shared invariant in "
            "the bottom strip is: an interacting UV completion whose "
            "existence depends on matter content and on the regulator "
            "scheme."
        ),
        references=[
            "Bonati, Pelissetto & Vicari (2025), Phys. Rep. [arXiv:2410.05823].",
            "Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].",
        ],
        see_also=[
            "[`docs/cross-analogue-gauge-higgs.md`](../cross-analogue-gauge-higgs.md)",
            "`asymsafety.transforms.bridge.gauge_higgs.GaugeHiggsAnalogue`",
            "`asymsafety.transforms.bridge.cross_analogue.CrossAnalogueBridge`",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ok: list[str] = []
_fail: list[tuple[str, str]] = []


def _save(fig: plt.Figure, name: str, out_dir: Path, fmt: str) -> None:
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    _ok.append(str(path))
    _save_caption(name, path.name, out_dir)


def _save_caption(name: str, image_filename: str, out_dir: Path) -> None:
    caption = _CAPTIONS.get(name)
    if caption is None:
        return
    md_path = out_dir / f"{name}.caption.md"
    md_path.write_text(caption.render(image_filename), encoding="utf-8")


def _refresh_baseline(out_dir: Path, name: str, fmt: str) -> None:
    """Write tests/baselines/<name>.<fmt>.sha256 from the freshly-saved file.

    Called from ``main`` when ``--update-baselines`` is set. The
    baseline directory is resolved relative to this script so the
    operation works regardless of the caller's CWD.
    """
    import hashlib

    image_path = out_dir / f"{name}.{fmt}"
    if not image_path.exists():
        return
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    baseline_dir = Path(__file__).resolve().parent.parent / "tests" / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / f"{name}.{fmt}.sha256").write_text(
        digest + "\n", encoding="utf-8",
    )


def _run(label: str, fn, name: str, out_dir: Path, fmt: str) -> None:
    """Build, save, and caption a single figure.

    Records success in :data:`_ok` (via :func:`_save`) and failure in
    :data:`_fail`; re-raises on failure so the calling worker / loop
    can decide whether to keep going. Stays silent — the enclosing run
    loop is responsible for printing OK / FAIL status, so output never
    interleaves between parallel workers and the parent.
    """
    try:
        fig = fn()
        _save(fig, name, out_dir, fmt)
    except Exception as exc:
        _fail.append((name, str(exc)))
        raise


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------


def _worker(
    gen_fn_name: str, out_dir: str, fmt: str, timeout_s: float,
) -> tuple[str, str, float]:
    """Run a single figure generator in a fresh subprocess.

    Each generator imports its own SymPy / NumPy stack and writes one
    PNG + one caption. Workers must be self-contained so we can ship
    them through :class:`~concurrent.futures.ProcessPoolExecutor`.

    A SIGALRM-based timeout aborts the worker if it exceeds
    ``timeout_s`` seconds — this is what actually keeps a stiff ODE
    or singular SymPy build from hanging the whole pipeline; the
    Future-level timeout from ``as_completed`` only governs the
    parent's wait, not the child's compute.

    Returns a ``(name, status, elapsed_s)`` tuple where ``status`` is
    ``"OK"`` or an error message.
    """
    import signal

    # Reapply Agg in the worker (each interpreter starts clean).
    import matplotlib as _mpl
    _mpl.use("Agg")
    from asymsafety.visualization.style import apply_style
    apply_style()

    # Resolve the generator function from this module's globals.
    # When ``ProcessPoolExecutor(mp_context=spawn)`` re-imports the
    # script, the gen_* functions live in the top-level namespace of
    # the module that defined ``_worker`` — i.e. our own globals().
    gen_fn = globals().get(gen_fn_name)
    name = gen_fn_name[len("gen_"):]
    if gen_fn is None:
        return name, f"FAIL: generator {gen_fn_name!r} not found", 0.0

    def _alarm(_signum, _frame):
        raise TimeoutError(f"timeout after {timeout_s:.0f}s")

    # signal.alarm only works on Unix; skip on platforms without it.
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(max(1, timeout_s)))

    t0 = time.time()
    try:
        gen_fn(Path(out_dir), fmt)
        elapsed = time.time() - t0
        return name, "OK", elapsed
    except TimeoutError as exc:
        elapsed = time.time() - t0
        return name, f"FAIL: {exc}", elapsed
    except Exception as exc:
        elapsed = time.time() - t0
        return name, f"FAIL: {type(exc).__name__}: {exc}", elapsed
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------


def gen_asymptotic_safety_concept(out: Path, fmt: str) -> None:
    from asymsafety.visualization.conceptual import asymptotic_safety_concept
    _run("Asymptotic safety concept", asymptotic_safety_concept,
         "asymptotic_safety_concept", out, fmt)


def gen_wetterich_equation(out: Path, fmt: str) -> None:
    from asymsafety.visualization.conceptual import wetterich_equation_diagram
    _run("Wetterich equation", wetterich_equation_diagram,
         "wetterich_equation", out, fmt)


def gen_regulator_comparison(out: Path, fmt: str) -> None:
    from asymsafety.visualization.conceptual import regulator_comparison
    _run("Regulator comparison", regulator_comparison,
         "regulator_comparison", out, fmt)


def gen_fp_stability_concept(out: Path, fmt: str) -> None:
    from asymsafety.visualization.conceptual import fixed_point_stability_concept
    _run("FP stability concept", fixed_point_stability_concept,
         "fp_stability_concept", out, fmt)


def gen_cross_analogue_bridge(out: Path, fmt: str) -> None:
    from asymsafety.visualization.bridge_diagram import cross_analogue_bridge_diagram
    _run("Cross-analogue bridge", cross_analogue_bridge_diagram,
         "cross_analogue_bridge", out, fmt)


def gen_hydraulic_analogy(out: Path, fmt: str) -> None:
    from asymsafety.visualization.bridge_diagram import hydraulic_analogy_diagram
    _run("Hydraulic analogy", hydraulic_analogy_diagram,
         "hydraulic_analogy", out, fmt)


def gen_eh_phase_portrait(out: Path, fmt: str) -> None:
    from asymsafety.visualization.phase_portrait import annotated_eh_phase_portrait
    _run("EH phase portrait", annotated_eh_phase_portrait,
         "eh_phase_portrait", out, fmt)


def gen_running_couplings(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.analysis.flow import FlowIntegrator
        from asymsafety.visualization.phase_portrait import flow_diagram

        system = build_eh_beta_system(d=4)
        integrator = FlowIntegrator(system)
        trajs = []
        for ic in [
            {"g": 0.5, "lambda": 0.1},
            {"g": 0.8, "lambda": 0.05},
            {"g": 0.3, "lambda": 0.2},
        ]:
            try:
                trajs.append(integrator.integrate(ic, t_span=(-8, 8)))
            except Exception:
                pass
        return flow_diagram(trajs, title="Running Couplings ($d=4$, Einstein–Hilbert)")
    _run("Running couplings", _make, "running_couplings", out, fmt)


def gen_3d_flow(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.analysis.fixed_points import FixedPointFinder
        from asymsafety.analysis.flow import FlowIntegrator
        from asymsafety.gui.visualization_3d import flow_trajectories_3d

        system = build_eh_beta_system(d=4)
        finder = FixedPointFinder(system)
        fps = []
        for guess in [{"g": 0.01, "lambda": 0.01}, {"g": 0.7, "lambda": 0.14}]:
            try:
                fps.append(finder.find_fixed_point(guess))
            except Exception:
                pass

        integrator = FlowIntegrator(system)
        trajs = []
        for ic in [
            {"g": 0.3, "lambda": 0.05},
            {"g": 0.8, "lambda": 0.1},
            {"g": 0.5, "lambda": -0.1},
            {"g": 1.0, "lambda": 0.2},
        ]:
            try:
                trajs.append(integrator.integrate(ic, t_span=(-5, 5)))
            except Exception:
                pass

        # World-line view: z axis = RG time (replaces the legacy
        # zero-padded "g, lambda, g" call pattern).
        return flow_trajectories_3d(
            trajs, "g", "lambda", z_coupling=None,
            fixed_points=fps if fps else None,
            show_eigenvectors=False,
            show_speed_widths=True,
            show_references=True,
        )
    _run("3D flow", _make, "3d_flow", out, fmt)


def gen_hydraulic_network(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.hydraulic.mapping import RGToHydraulicMapper
        from asymsafety.hydraulic.visualization import plot_hydraulic_network

        system = build_eh_beta_system(d=4)
        mapper = RGToHydraulicMapper(system)
        network = mapper.build_network(
            reference_point={"g": 0.7, "lambda": 0.14},
        )
        return plot_hydraulic_network(network)
    _run("Hydraulic network", _make, "hydraulic_network", out, fmt)


def gen_foliated_3d(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.beta.foliated import build_foliated_eh_beta_system
        from asymsafety.analysis.fixed_points import FixedPointFinder
        from asymsafety.gui.visualization_3d import foliated_phase_portrait_3d

        system = build_foliated_eh_beta_system(d=4)
        fps = []
        try:
            fps.append(FixedPointFinder(system).find_fixed_point(
                {"g": 0.96, "lambda": 0.20, "lambda_ADM": 1.0}
            ))
        except Exception:
            pass
        return foliated_phase_portrait_3d(system, n_grid=5, fixed_points=fps)
    _run("Foliated 3D", _make, "foliated_3d", out, fmt)


def gen_fp_stability_3d(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.beta.foliated import build_foliated_eh_beta_system
        from asymsafety.analysis.fixed_points import FixedPointFinder
        from asymsafety.analysis.stability import analyze_stability
        from asymsafety.gui.visualization_3d import fixed_point_stability_3d

        system = build_foliated_eh_beta_system(d=4)
        fp = FixedPointFinder(system).find_fixed_point(
            {"g": 0.96, "lambda": 0.20, "lambda_ADM": 1.0}
        )
        sa = analyze_stability(system, fp)
        return fixed_point_stability_3d(
            fp, sa, "g", "lambda", "lambda_ADM",
            scale=0.35, show_references=True,
        )
    _run("FP stability 3D", _make, "fp_stability_3d", out, fmt)


def gen_matter_continuation(out: Path, fmt: str) -> None:
    def _make():
        import numpy as np

        from asymsafety.actions.matter import MatterContent
        from asymsafety.analysis.continuation import continuation
        from asymsafety.beta.matter import build_gravity_matter_fp_system
        from asymsafety.visualization.fixed_point_plot import (
            plot_matter_content_continuation,
        )

        # ``build_eh_matter_beta_system`` (the 2-coupling matter system
        # without a running scalar quartic) currently produces β
        # functions whose only root in the physical region is the
        # Gaussian FP, so a naive continuation collapses to ``(0, 0)``.
        # ``build_gravity_matter_fp_system`` (3-coupling, scalar quartic
        # promoted to a running variable) has the pinned NGFP exercised
        # by ``tests/test_benchmarks_published.py::TestGravityMatterFP``
        # and produces a smoothly-varying continuation in ``N_s``.
        n_values = np.array([0, 1, 2, 4, 6, 8, 10, 12], dtype=float)

        def builder(n: float):
            return build_gravity_matter_fp_system(
                MatterContent(n_scalars=int(n)), scalar_quartic=True,
            )

        # Initial guess = the corrected pure-gravity (N_s = 0) NGFP
        # (g*, lambda*) = (0.70732, 0.19320) with a small scalar
        # quartic; the continuation then tracks the root in N_s.
        result = continuation(
            system_builder=builder,
            parameter_name="N_s",
            parameter_values=n_values,
            initial_guess={"g": 0.70, "lambda": 0.195, "lambda_phi": 0.009},
        )
        return plot_matter_content_continuation(result)
    _run("Matter continuation", _make, "matter_continuation", out, fmt)


def gen_flow_basin_3d(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.analysis.fixed_points import FixedPointFinder
        from asymsafety.gui.visualization_3d import flow_basin_3d

        system = build_eh_beta_system(d=4)
        fp = FixedPointFinder(system).find_fixed_point(
            {"g": 0.7, "lambda": 0.14}
        )
        return flow_basin_3d(
            system, fp,
            x_coupling="g", y_coupling="lambda", z_coupling="lambda",
            n_samples=40, radius=0.15,
            t_span=(0.0, 2.5),
        )
    _run("Flow basin 3D", _make, "flow_basin_3d", out, fmt)


def gen_quadratic_pairwise(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.beta.quadratic import build_quadratic_beta_system
        from asymsafety.visualization.phase_portrait import quadratic_pairwise_grid

        system = build_quadratic_beta_system(d=4)
        return quadratic_pairwise_grid(system, n_grid=18)
    _run("Quadratic pairwise grid", _make, "quadratic_pairwise", out, fmt)


# ---------------------------------------------------------------------------
# Helpers for new generators
# ---------------------------------------------------------------------------


# IR Newton constant (in k0 units) of the shared cosmology / scattering
# trajectory built by ``_eh_trajectory_for_cosmology``. The Planck mass
# in these geometric units is M_pl = G_N^{-1/2} ≈ 7.07; the critical
# Bonanno-Reuter mass of the resulting RG-improved Schwarzschild
# geometry computes to M_crit ≈ 5.7 ≈ 0.8 M_pl.
_COSMO_G_NEWTON = 0.02


def _eh_trajectory_for_cosmology():
    """Build a representative EH trajectory used by the cosmology figures.

    We need a trajectory that interpolates between the Gaussian
    (classical) regime in the IR and the NGFP in the UV — required by
    ``G(r) = g/k²`` so the dimensional Newton constant is *constant*
    at small ``k`` (large ``r``) and softens like ``g*/k²`` in the UV.

    The corrected EH NGFP is UV-attractive (both directions relevant,
    complex pair ``theta = 1.475 ± 3.043i``), so perturbing off the
    NGFP and integrating *down* in RG time spirals into the
    ``lambda = 1/2`` propagator pole after a few e-folds: the
    integration aborts (``success=False``), ``RGTrajectory.at_scale``
    silently clamps, and ``G(r)`` inherits a spurious ``r²`` growth in
    the IR. Instead we mirror the proven construction in
    :func:`asymsafety.cli.amplitude_cmd._build_trajectory`: start deep
    in the IR with classical scaling ``g = G_N e^{2t}`` (all other
    couplings zero) and integrate upward — the flow converges onto the
    UV-attractive NGFP, giving ``G ≈ G_N`` in the IR and
    ``G ≈ g*/k²`` in the UV.
    """
    import math

    import numpy as np

    from asymsafety.analysis.flow import FlowIntegrator
    from asymsafety.beta.einstein_hilbert import build_eh_beta_system

    system = build_eh_beta_system(d=4)
    t_ir = -8.0
    t_uv = 15.0
    ic_ir = {name: 0.0 for name in system.coupling_names}
    ic_ir["g"] = _COSMO_G_NEWTON * math.exp(2.0 * t_ir)
    # Dense storage grid (Δt = 0.01): ``RGTrajectory.at_scale``
    # interpolates linearly on the *stored* points (the integrator
    # defaults to only 500), and downstream consumers differentiate
    # the lapse — coarse storage leaves visible piecewise-linear
    # kinks in T_H(M).
    t_eval = np.linspace(t_ir, t_uv, int((t_uv - t_ir) / 0.01) + 1)
    return FlowIntegrator(system).integrate(
        ic_ir, t_span=(t_ir, t_uv), max_step=0.05, t_eval=t_eval,
    )


def _qiskit_placeholder(title: str, hint: str = ""):
    """Build a placeholder figure when qiskit is unavailable.

    The pipeline still produces a PNG so docs aren't broken, but the
    figure clearly indicates that the user needs the ``[quantum]``
    extra to render the real plot.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_axis_off()
    ax.text(
        0.5, 0.65, title, ha="center", va="center",
        transform=ax.transAxes, fontsize=14, fontweight="bold",
    )
    ax.text(
        0.5, 0.40,
        "Install the qiskit extra to render this figure:\n\n"
        "    pip install asymsafety[quantum]"
        + (f"\n\n{hint}" if hint else ""),
        ha="center", va="center",
        transform=ax.transAxes, fontsize=11, color="0.3",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"),
    )
    return fig


def _qiskit_available() -> bool:
    try:
        import qiskit  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Transform plot generators (Bode, scalogram, pseudospectrum, comparison)
# ---------------------------------------------------------------------------


def gen_bode(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.analysis.fixed_points import FixedPointFinder
        from asymsafety.analysis.stability import analyze_stability
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.transforms.bridge.impedance import ImpedanceBridge
        from asymsafety.transforms.visualization.transform_plots import plot_bode

        system = build_eh_beta_system(d=4)
        fp = FixedPointFinder(system).find_fixed_point({"g": 0.7, "lambda": 0.14})
        sa = analyze_stability(system, fp)
        bode = ImpedanceBridge(sa.stability_matrix).bode_data(
            omega_range=(0.01, 100.0), n_points=200,
        )
        return plot_bode(bode, entry=(0, 0))
    _run("Bode plot", _make, "bode", out, fmt)


def gen_scalogram(out: Path, fmt: str) -> None:
    def _make():
        import numpy as np
        from asymsafety.analysis.flow import FlowIntegrator
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.transforms.integral.wavelet import RGFlowWavelet
        from asymsafety.transforms.visualization.transform_plots import plot_scalogram

        system = build_eh_beta_system(d=4)
        # Wider RG-time window so the scalogram captures both the
        # crossover region near the NGFP and the asymptotic IR plateau.
        traj = FlowIntegrator(system).integrate(
            {"g": 0.5, "lambda": 0.1}, t_span=(-6.0, 6.0),
        )
        scales = np.geomspace(0.05, 5.0, 32)
        wavelet = RGFlowWavelet(traj).transform(scales, wavelet_type="morlet")
        coupling_name = system.coupling_names[0]  # 'g'
        return plot_scalogram(
            wavelet, coupling_index=0, coupling_name=coupling_name,
        )
    _run("Wavelet scalogram", _make, "scalogram", out, fmt)


def gen_pseudospectrum(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.analysis.fixed_points import FixedPointFinder
        from asymsafety.analysis.stability import analyze_stability
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.transforms.linear.resolvent import ResolventOperator
        from asymsafety.transforms.visualization.transform_plots import (
            plot_pseudospectrum,
        )

        system = build_eh_beta_system(d=4)
        fp = FixedPointFinder(system).find_fixed_point({"g": 0.7, "lambda": 0.14})
        sa = analyze_stability(system, fp)
        res = ResolventOperator(sa)
        return plot_pseudospectrum(
            res, epsilon_values=[1.0, 0.1, 0.01],
            real_range=(-5.0, 5.0), imag_range=(-5.0, 5.0),
            n_grid=80,
        )
    _run("Pseudospectrum", _make, "pseudospectrum", out, fmt)


def gen_comparison_table(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.analysis.fixed_points import FixedPointFinder
        from asymsafety.analysis.stability import analyze_stability
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.transforms.bridge.cross_analogue import CrossAnalogueBridge
        from asymsafety.transforms.visualization.transform_plots import (
            plot_comparison_table,
        )

        system = build_eh_beta_system(d=4)
        fp = FixedPointFinder(system).find_fixed_point({"g": 0.7, "lambda": 0.14})
        sa = analyze_stability(system, fp)
        table = CrossAnalogueBridge(system, fp, sa).full_comparison_table()
        return plot_comparison_table(table)
    _run("Cross-method comparison", _make, "comparison_table", out, fmt)


# ---------------------------------------------------------------------------
# Cosmology generators
# ---------------------------------------------------------------------------


def gen_running_newton_constant(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.cosmology.rg_improved_bh import RGImprovedSchwarzschild
        from asymsafety.cosmology.visualization import (
            plot_running_newton_constant,
        )
        bh = RGImprovedSchwarzschild(
            trajectory=_eh_trajectory_for_cosmology(), M=1.0,
        )
        return plot_running_newton_constant(bh)
    _run("Running Newton constant", _make, "running_newton_constant", out, fmt)


def gen_lapse_with_horizons(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.cosmology.rg_improved_bh import RGImprovedSchwarzschild
        from asymsafety.cosmology.visualization import plot_lapse_with_horizons
        # M_crit ≈ 5.7 for the G_N = 0.02 trajectory (M_pl ≈ 7.07);
        # M = 8 is clearly super-critical so both r_- and r_+ exist and
        # sit well inside the default radial window.
        bh = RGImprovedSchwarzschild(
            trajectory=_eh_trajectory_for_cosmology(), M=8.0,
        )
        return plot_lapse_with_horizons(bh)
    _run("Lapse with horizons", _make, "lapse_with_horizons", out, fmt)


def gen_classical_vs_rg_lapse(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.cosmology.rg_improved_bh import RGImprovedSchwarzschild
        from asymsafety.cosmology.visualization import plot_classical_vs_rg_lapse
        # M_crit ≈ 5.7: M = 3 is sub-critical (no horizon anywhere),
        # M = 6 is just above critical (nearly merged horizon pair),
        # M = 9 is clearly super-critical (well-separated r_-, r_+).
        bh = RGImprovedSchwarzschild(
            trajectory=_eh_trajectory_for_cosmology(), M=9.0,
        )
        return plot_classical_vs_rg_lapse(bh, M_values=(3.0, 6.0, 9.0))
    _run("Classical vs RG lapse", _make, "classical_vs_rg_lapse", out, fmt)


def gen_hawking_temperature(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.cosmology.rg_improved_bh import RGImprovedSchwarzschild
        from asymsafety.cosmology.visualization import plot_hawking_temperature
        # Sweep through M_crit ≈ 5.7 (no-horizon region at the left
        # edge) up to ~5 M_crit where T_H rejoins the classical curve.
        bh = RGImprovedSchwarzschild(
            trajectory=_eh_trajectory_for_cosmology(), M=8.0,
        )
        return plot_hawking_temperature(bh, M_range=(2.0, 30.0), n_masses=150)
    _run("Hawking temperature", _make, "hawking_temperature", out, fmt)


def gen_scale_identification(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.cosmology.scale_identification import (
            GeodesicDistanceScale, InverseDistanceScale, ProperDistanceScale,
        )
        from asymsafety.cosmology.visualization import plot_scale_identification
        return plot_scale_identification(
            [
                InverseDistanceScale(),
                GeodesicDistanceScale(delta=0.5),
                ProperDistanceScale(r_h=2.0, delta=0.1),
            ],
            labels=["k=ξ/r", "k=ξ/√(r²+δ²)", "k=ξ/d(r,r_h)"],
        )
    _run("Scale identification", _make, "scale_identification", out, fmt)


def gen_flrw_evolution(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.cosmology.rg_improved_flrw import RGImprovedFLRW
        from asymsafety.cosmology.visualization import plot_flrw_evolution

        flrw = RGImprovedFLRW(trajectory=_eh_trajectory_for_cosmology())
        return plot_flrw_evolution(
            flrw, t_span=(0.05, 5.0), n_steps=120, a0=1e-4, rho0=1.0,
        )
    _run("FLRW evolution", _make, "flrw_evolution", out, fmt)


# ---------------------------------------------------------------------------
# Quantum generators
# ---------------------------------------------------------------------------


def _eh_edmd():
    """Compute Koopman EDMD over a small bundle of EH trajectories."""
    from asymsafety.analysis.flow import FlowIntegrator
    from asymsafety.beta.einstein_hilbert import build_eh_beta_system
    from asymsafety.quantum.koopman.operator import KoopmanOperator

    system = build_eh_beta_system(d=4)
    integ = FlowIntegrator(system)
    trajs = [
        integ.integrate({"g": g0, "lambda": l0}, t_span=(-4.0, 4.0))
        for g0, l0 in [
            (0.5, 0.1), (0.8, 0.05), (0.3, 0.2), (0.9, 0.18),
        ]
    ]
    return system, KoopmanOperator(system).compute_edmd(
        trajs, dictionary_degree=4,
    )


def gen_koopman_spectrum(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.quantum.visualization import plot_koopman_spectrum
        _, ed = _eh_edmd()
        return plot_koopman_spectrum(ed)
    _run("Koopman spectrum", _make, "koopman_spectrum", out, fmt)


def gen_koopman_modes(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.quantum.visualization import plot_koopman_modes
        sys, ed = _eh_edmd()
        return plot_koopman_modes(ed, sys.coupling_names, top_k=8)
    _run("Koopman modes", _make, "koopman_modes", out, fmt)


def gen_qft_power_spectrum(out: Path, fmt: str) -> None:
    def _make():
        if not _qiskit_available():
            return _qiskit_placeholder(
                "QFT power spectrum",
                hint="Renders the qiskit-backed Quantum Fourier transform "
                     "of the running Newton coupling.",
            )
        from asymsafety.analysis.flow import FlowIntegrator
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.quantum.koopman.qft_rg import QuantumFourierRG
        from asymsafety.quantum.visualization import plot_qft_power_spectrum

        system = build_eh_beta_system(d=4)
        traj = FlowIntegrator(system).integrate(
            {"g": 0.5, "lambda": 0.1}, t_span=(-4.0, 4.0),
        )
        return plot_qft_power_spectrum(QuantumFourierRG(traj), "g")
    _run("QFT power spectrum", _make, "qft_power_spectrum", out, fmt)


def gen_grover_success_probability(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.quantum.grover.encoding import CouplingGridEncoding
        from asymsafety.quantum.grover.oracle import BetaOracle
        from asymsafety.quantum.grover.search import GroverFixedPointSearch
        from asymsafety.quantum.visualization import (
            plot_grover_success_probability,
        )

        system = build_eh_beta_system(d=4)
        encoding = CouplingGridEncoding(
            coupling_ranges={"g": (0.1, 1.4), "lambda": (-0.2, 0.4)},
            n_bits=4,
        )
        oracle = BetaOracle(system, encoding, epsilon=0.5)
        search = GroverFixedPointSearch(system, encoding, oracle)
        return plot_grover_success_probability(search, n_iter_max=20)
    _run("Grover success probability", _make,
         "grover_success_probability", out, fmt)


def gen_grover_measurement_distribution(out: Path, fmt: str) -> None:
    def _make():
        if not _qiskit_available():
            return _qiskit_placeholder(
                "Grover measurement distribution",
                hint="Renders the measurement histogram from a qiskit-backed "
                     "Grover search.",
            )
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.quantum.grover.encoding import CouplingGridEncoding
        from asymsafety.quantum.grover.oracle import BetaOracle
        from asymsafety.quantum.grover.search import GroverFixedPointSearch
        from asymsafety.quantum.visualization import (
            plot_grover_measurement_distribution,
        )

        system = build_eh_beta_system(d=4)
        encoding = CouplingGridEncoding(
            coupling_ranges={"g": (0.1, 1.4), "lambda": (-0.2, 0.4)},
            n_bits=4,
        )
        oracle = BetaOracle(system, encoding, epsilon=0.5)
        search = GroverFixedPointSearch(system, encoding, oracle)
        result = search.search(shots=512)
        return plot_grover_measurement_distribution(result, encoding)
    _run("Grover measurement", _make,
         "grover_measurement_distribution", out, fmt)


def gen_vqrg_cost_landscape(out: Path, fmt: str) -> None:
    def _make():
        if not _qiskit_available():
            return _qiskit_placeholder(
                "VQRG cost landscape",
                hint="Renders the quantum-circuit cost slice on a "
                     "VQRG ansatz; needs qiskit for circuit simulation.",
            )
        import numpy as np
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.quantum.vqrg.circuit import VQRGCircuit
        from asymsafety.quantum.vqrg.cost import VQRGCostFunction
        from asymsafety.quantum.vqrg.mapping import CouplingAngleMap
        from asymsafety.quantum.visualization import plot_vqrg_cost_landscape

        system = build_eh_beta_system(d=4)
        cmap = CouplingAngleMap({"g": (0.0, 1.5), "lambda": (-0.4, 0.4)})
        circuit = VQRGCircuit(n_qubits=2, n_layers=2)
        cost = VQRGCostFunction(system, cmap, circuit)
        base = np.zeros(circuit.n_parameters)
        return plot_vqrg_cost_landscape(
            cost, base, dims=(0, 1), n_grid=18,
        )
    _run("VQRG cost landscape", _make, "vqrg_cost_landscape", out, fmt)


def gen_vqrg_optimization_trajectory(out: Path, fmt: str) -> None:
    def _make():
        # Synthesise a representative cost-history curve so the figure
        # ships even when qiskit is missing — the visualisation is a
        # pure-numpy convergence plot.
        import numpy as np
        from asymsafety.quantum.visualization import (
            plot_vqrg_optimization_trajectory,
        )
        rng = np.random.default_rng(42)
        n_iter = 80
        history = 10.0 * np.exp(-np.arange(n_iter) / 14.0) + 1e-4
        history += rng.normal(scale=0.03 * history, size=n_iter)
        params = np.cumsum(rng.normal(scale=0.05, size=(n_iter, 4)), axis=0)
        return plot_vqrg_optimization_trajectory(
            history, parameters_history=params,
        )
    _run("VQRG optimisation trajectory", _make,
         "vqrg_optimization_trajectory", out, fmt)


def gen_quantum_fisher_eigenvalues(out: Path, fmt: str) -> None:
    def _make():
        # Build a representative QFI matrix from the EH stability matrix
        # (works without qiskit; shape and conditioning illustrate the
        # Zamolodchikov-metric idea).
        import numpy as np
        from asymsafety.analysis.fixed_points import FixedPointFinder
        from asymsafety.analysis.stability import analyze_stability
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.quantum.visualization import (
            plot_quantum_fisher_eigenvalues,
        )

        system = build_eh_beta_system(d=4)
        fp = FixedPointFinder(system).find_fixed_point(
            {"g": 0.7, "lambda": 0.14}
        )
        sa = analyze_stability(system, fp)
        M = sa.stability_matrix
        qfi = M.T @ M  # positive semidefinite; same eigenstructure
        return plot_quantum_fisher_eigenvalues(
            qfi, coupling_names=system.coupling_names,
        )
    _run("Quantum Fisher eigenvalues", _make,
         "quantum_fisher_eigenvalues", out, fmt)


def _scalar_gibbs_estimator():
    from asymsafety.geometry.decomposition import ModeSpectrum
    from asymsafety.quantum.thermal.gibbs import GibbsStatePreparer
    from asymsafety.quantum.thermal.hamiltonian import LaplacianHamiltonian
    from asymsafety.quantum.thermal.partition import PartitionFunctionEstimator

    spectrum = ModeSpectrum(d=4)
    ham = LaplacianHamiltonian(spectrum, field_type="scalar", l_max=4)
    gibbs = GibbsStatePreparer(ham)
    return gibbs, PartitionFunctionEstimator(gibbs, d=4)


def gen_partition_function(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.quantum.visualization import plot_partition_function
        _, est = _scalar_gibbs_estimator()
        return plot_partition_function(est, beta_range=(0.01, 1.0), n_points=60)
    _run("Partition function", _make, "partition_function", out, fmt)


def gen_seeley_dewitt_extraction(out: Path, fmt: str) -> None:
    def _make():
        import sympy
        from asymsafety.frg.heat_kernel import SeeleyDeWittCoefficients
        from asymsafety.quantum.visualization import plot_seeley_dewitt_extraction
        _, est = _scalar_gibbs_estimator()
        sdw = SeeleyDeWittCoefficients(d=4, R_bar=sympy.Rational(1))
        return plot_seeley_dewitt_extraction(est, sdw=sdw, field_type="scalar")
    _run("Seeley-DeWitt extraction", _make,
         "seeley_dewitt_extraction", out, fmt)


def gen_gibbs_state_purity(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.quantum.visualization import plot_gibbs_state_purity
        gibbs, _ = _scalar_gibbs_estimator()
        return plot_gibbs_state_purity(
            gibbs, beta_range=(0.01, 5.0), n_points=80,
        )
    _run("Gibbs state purity", _make, "gibbs_state_purity", out, fmt)


# ---------------------------------------------------------------------------
# 3D extensions
# ---------------------------------------------------------------------------


def gen_koopman_spectrum_3d(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.gui.visualization_3d import koopman_spectrum_3d
        _, ed = _eh_edmd()
        return koopman_spectrum_3d(ed)
    _run("Koopman spectrum 3D", _make, "koopman_spectrum_3d", out, fmt)


def gen_vqrg_cost_isosurface_3d(out: Path, fmt: str) -> None:
    def _make():
        if not _qiskit_available():
            return _qiskit_placeholder(
                "VQRG cost isosurface (3D)",
                hint="3D iso-cost surfaces over a slice of the variational "
                     "parameters — qiskit needed for the underlying "
                     "circuit simulation.",
            )
        import numpy as np
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.gui.visualization_3d import vqrg_cost_isosurface_3d
        from asymsafety.quantum.vqrg.circuit import VQRGCircuit
        from asymsafety.quantum.vqrg.cost import VQRGCostFunction
        from asymsafety.quantum.vqrg.mapping import CouplingAngleMap

        system = build_eh_beta_system(d=4)
        cmap = CouplingAngleMap({"g": (0.0, 1.5), "lambda": (-0.4, 0.4)})
        circuit = VQRGCircuit(n_qubits=2, n_layers=2)
        cost = VQRGCostFunction(system, cmap, circuit)
        base = np.zeros(circuit.n_parameters)
        return vqrg_cost_isosurface_3d(
            cost, base, dims=(0, 1, 2), n_grid=12,
        )
    _run("VQRG cost isosurface 3D", _make,
         "vqrg_cost_isosurface_3d", out, fmt)


# ---------------------------------------------------------------------------
# Spectral / heat-kernel generators
# ---------------------------------------------------------------------------


def gen_spectral_sum_convergence(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.frg.spectral import SpectralSumEvaluator
        from asymsafety.visualization.phase_portrait import (
            plot_spectral_sum_convergence,
        )
        ss = SpectralSumEvaluator(d=4, l_max=512)
        return plot_spectral_sum_convergence(
            ss, n_max_values=[2, 4, 8, 16, 32, 64, 128, 256, 512],
            R_bar_val=12.0,
        )
    _run("Spectral-sum convergence", _make,
         "spectral_sum_convergence", out, fmt)


def gen_heat_kernel_coefficients(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.frg.heat_kernel import SeeleyDeWittCoefficients
        from asymsafety.visualization.phase_portrait import (
            plot_heat_kernel_coefficients,
        )
        sdw = SeeleyDeWittCoefficients(d=4, R_bar=1.0)
        return plot_heat_kernel_coefficients(
            sdw,
            field_types=("scalar", "vector", "TT"),
            coefficients=("b0", "b2", "b4"),
        )
    _run("Heat-kernel coefficients", _make,
         "heat_kernel_coefficients", out, fmt)


# ---------------------------------------------------------------------------
# Cross-analogue: 3D gauge-Higgs (Bonati, Pelissetto & Vicari 2025)
# ---------------------------------------------------------------------------


def gen_ahm_phase_diagram(out: Path, fmt: str) -> None:
    from asymsafety.visualization.conceptual import ahm_phase_diagram
    _run("AHM phase diagram", ahm_phase_diagram,
         "ahm_phase_diagram", out, fmt)


def gen_charged_fp_boundary(out: Path, fmt: str) -> None:
    from asymsafety.visualization.conceptual import charged_fp_boundary
    _run("Charged FP boundary", charged_fp_boundary,
         "charged_fp_boundary", out, fmt)


def gen_nu_vs_nf(out: Path, fmt: str) -> None:
    from asymsafety.visualization.fixed_point_plot import nu_vs_Nf
    _run("nu(Nf) at the charged FP", nu_vs_Nf,
         "nu_vs_nf", out, fmt)


def gen_ahm_as_bridge(out: Path, fmt: str) -> None:
    from asymsafety.visualization.bridge_diagram import ahm_as_bridge_diagram
    _run("AHM ↔ AS bridge", ahm_as_bridge_diagram,
         "ahm_as_bridge", out, fmt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Scattering amplitudes (physical scattering + asymptotic safety)
# ---------------------------------------------------------------------------

def _scattering_amplitude():
    """AS graviton-mediated amplitude on the representative EH trajectory."""
    from asymsafety.scattering.amplitude import GravitonMediatedAmplitude
    from asymsafety.scattering.form_factor import GravitonFormFactor

    return GravitonMediatedAmplitude(
        GravitonFormFactor(_eh_trajectory_for_cosmology())
    )


def gen_amplitude_vs_energy(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.visualization.amplitude_plot import (
            plot_amplitude_vs_energy,
        )
        return plot_amplitude_vs_energy(_scattering_amplitude())
    _run("Amplitude vs energy", _make, "amplitude_vs_energy", out, fmt)


def gen_graviton_form_factor(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.scattering.form_factor import GravitonFormFactor
        from asymsafety.visualization.amplitude_plot import plot_form_factor
        return plot_form_factor(
            GravitonFormFactor(_eh_trajectory_for_cosmology())
        )
    _run("Graviton form factor", _make, "graviton_form_factor", out, fmt)


def gen_partial_wave_unitarity(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.visualization.amplitude_plot import (
            plot_partial_wave_unitarity,
        )
        return plot_partial_wave_unitarity(_scattering_amplitude())
    _run("Partial-wave unitarity", _make, "partial_wave_unitarity", out, fmt)


def gen_regge_trajectory(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.visualization.amplitude_plot import plot_regge_trajectory
        return plot_regge_trajectory()
    _run("Regge trajectory", _make, "regge_trajectory", out, fmt)


def gen_as_vs_string(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.scattering.bridge import ScatteringBridge
        from asymsafety.visualization.amplitude_plot import plot_as_vs_string
        return plot_as_vs_string(ScatteringBridge(_scattering_amplitude()))
    _run("AS vs string bridge", _make, "as_vs_string", out, fmt)


def gen_scattering_concept(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.visualization.conceptual import scattering_concept
        return scattering_concept()
    _run("Scattering concept", _make, "scattering_concept", out, fmt)


def gen_scattering_bridge(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.visualization.bridge_diagram import (
            scattering_bridge_diagram,
        )
        return scattering_bridge_diagram()
    _run("Scattering bridge", _make, "scattering_bridge", out, fmt)


_CAPTIONS.update({
    "scattering_concept": FigureCaption(
        title="Graviton-mediated scattering in asymptotic safety (concept)",
        description=(
            "Schematic of the tree-level graviton exchange between two "
            "scalars, with quantum corrections packaged into a "
            "momentum-dependent form factor f(p^2) that is 1 in the infrared "
            "(Newtonian gravity) and grows like p^2 in the ultraviolet "
            "(softening G(p^2) -> g*/p^2). The right panel shows |M| vs "
            "energy: classical GR grows without bound, asymptotic safety "
            "reaches a finite UV constant, and the string bootstrap amplitude "
            "falls off ultrasoftly."
        ),
        references=[
            "Draper, Knorr, Ripken & Saueressig (2020) [2007.04396]",
            "Cheung, Remmen, Sciotti & Tarquini (2025) [2508.09246]",
        ],
        see_also=["asymsafety.visualization.conceptual.scattering_concept"],
    ),
    "scattering_bridge": FigureCaption(
        title="Asymptotic safety vs the string bootstrap (consistency)",
        description=(
            "Two-column physical-scattering consistency table. Both the "
            "asymptotically-safe and the Veneziano/Virasoro-Shapiro string "
            "amplitudes satisfy the foundational requirements (crossing, UV "
            "finiteness, bounded partial waves, no ghosts); they differ in "
            "the ultraviolet, where asymptotic safety softens to a "
            "UV-constant amplitude while strings are ultrasoft with an "
            "infinite higher-spin Regge tower. Distinct but mutually "
            "consistent points in amplitude space."
        ),
        references=["Cheung, Remmen, Sciotti & Tarquini (2025) [2508.09246]"],
        see_also=[
            "asymsafety.visualization.bridge_diagram.scattering_bridge_diagram"
        ],
    ),
})


_CAPTIONS.update({
    "amplitude_vs_energy": FigureCaption(
        title="Graviton-mediated scattering: GR vs asymptotic safety",
        description=(
            "Magnitude of the tree-level graviton-mediated 2->2 scalar "
            "amplitude at fixed angle. Classical general relativity grows "
            "with energy, while the asymptotically-safe form factor softens "
            "the graviton coupling G(p^2) ~ g*/p^2 and the amplitude tends "
            "to a finite ultraviolet constant."
        ),
        references=["Draper, Knorr, Ripken & Saueressig (2020) [2007.04396]"],
        see_also=["asymsafety.visualization.amplitude_plot.plot_amplitude_vs_energy"],
    ),
    "graviton_form_factor": FigureCaption(
        title="Graviton form factor f(p^2)",
        description=(
            "The momentum-dependent dressing f(p^2)=G_N/G(p^2) and the "
            "running G(p^2)/G_N. In the infrared f->1 (Newtonian limit); at "
            "the fixed point f ~ p^2 so G ~ 1/p^2 (ultraviolet softening)."
        ),
        references=[
            "Draper, Knorr, Ripken & Saueressig (2020) [2007.04396]",
            "Knorr (2026) [2602.21285]",
        ],
        see_also=["asymsafety.visualization.amplitude_plot.plot_form_factor"],
    ),
    "partial_wave_unitarity": FigureCaption(
        title="Partial-wave growth and unitarity",
        description=(
            "The s-wave partial amplitude |a_0(s)| of graviton-mediated "
            "scalar scattering, projected with the forward/backward cutoff "
            "|cos(theta)| <= 0.99 on the massless t/u-channel graviton "
            "poles, on a trajectory with IR Newton constant G_N = 0.02 "
            "(Planck mass `M_Pl = G_N^{-1/2} ~ 7.1 k_0`, dotted vertical "
            "line). Classical GR (red) grows linearly in s and crosses "
            "the |a_l| = 1 line (dashed) at `sqrt(s) ~ 3.4 k_0 ~ 0.5 "
            "M_Pl`; the asymptotically-safe curve (green) departs from GR "
            "at the Planck scale and flattens to a finite plateau. Honest "
            "caveat: that AS plateau (~1.4e2) sits *above* |a_0| = 1 — "
            "its absolute value is dominated by the angular cutoff on the "
            "forward graviton pole (it scales like 2 g*/(1 - cos_max)) "
            "and is normalisation/cutoff-dependent, so the figure does "
            "not demonstrate the literal elastic bound for asymptotic "
            "safety. The scheme-robust statement is boundedness: the AS "
            "growth exponent is ~0 versus GR's ~1 (Knorr 2026)."
        ),
        references=["Knorr (2026) [2602.21285]"],
        see_also=[
            "asymsafety.visualization.amplitude_plot.plot_partial_wave_unitarity"
        ],
    ),
    "regge_trajectory": FigureCaption(
        title="Regge trajectory and string mass spectrum",
        description=(
            "Chew-Frautschi plot of the linear Regge trajectory and the "
            "string mass tower m_n^2=(n-alpha0)/alpha', the bootstrap output "
            "of Strings from Almost Nothing."
        ),
        references=["Cheung, Remmen, Sciotti & Tarquini (2025) [2508.09246]"],
        see_also=["asymsafety.visualization.amplitude_plot.plot_regge_trajectory"],
    ),
    "as_vs_string": FigureCaption(
        title="Two routes to UV completeness",
        description=(
            "Normalised fixed-angle amplitudes: asymptotic safety reaches a "
            "UV-constant amplitude by softening the graviton coupling, while "
            "the string bootstrap enforces ultrasoft (super-polynomial) "
            "falloff via an infinite higher-spin Regge tower. Both are "
            "physically consistent, but distinct."
        ),
        references=[
            "Cheung, Remmen, Sciotti & Tarquini (2025) [2508.09246]",
            "Draper, Knorr, Ripken & Saueressig (2020) [2007.04396]",
        ],
        see_also=["asymsafety.visualization.amplitude_plot.plot_as_vs_string"],
    ),
})


ALL_GENERATORS = [
    gen_asymptotic_safety_concept,
    gen_wetterich_equation,
    gen_regulator_comparison,
    gen_fp_stability_concept,
    gen_cross_analogue_bridge,
    gen_hydraulic_analogy,
    gen_eh_phase_portrait,
    gen_running_couplings,
    gen_3d_flow,
    gen_hydraulic_network,
    gen_foliated_3d,
    gen_fp_stability_3d,
    gen_matter_continuation,
    gen_flow_basin_3d,
    gen_quadratic_pairwise,
    # Transform domain
    gen_bode,
    gen_scalogram,
    gen_pseudospectrum,
    gen_comparison_table,
    # Cosmology
    gen_running_newton_constant,
    gen_lapse_with_horizons,
    gen_classical_vs_rg_lapse,
    gen_hawking_temperature,
    gen_scale_identification,
    gen_flrw_evolution,
    # Quantum
    gen_koopman_spectrum,
    gen_koopman_modes,
    gen_qft_power_spectrum,
    gen_grover_success_probability,
    gen_grover_measurement_distribution,
    gen_vqrg_cost_landscape,
    gen_vqrg_optimization_trajectory,
    gen_quantum_fisher_eigenvalues,
    gen_partition_function,
    gen_seeley_dewitt_extraction,
    gen_gibbs_state_purity,
    # 3D extensions
    gen_koopman_spectrum_3d,
    gen_vqrg_cost_isosurface_3d,
    # Spectral / heat-kernel
    gen_spectral_sum_convergence,
    gen_heat_kernel_coefficients,
    # Cross-analogue: gauge-Higgs (Bonati 2025)
    gen_ahm_phase_diagram,
    gen_charged_fp_boundary,
    gen_nu_vs_nf,
    gen_ahm_as_bridge,
    # Scattering amplitudes (physical scattering + asymptotic safety)
    gen_scattering_concept,
    gen_amplitude_vs_energy,
    gen_graviton_form_factor,
    gen_partial_wave_unitarity,
    gen_regge_trajectory,
    gen_as_vs_string,
    gen_scattering_bridge,
]

# Names whose output is byte-deterministic enough to support sha256
# regression tests. Stochastic / ODE / Monte-Carlo / 3D-auto-orient
# figures are intentionally excluded.
_HASH_TESTED = (
    "asymptotic_safety_concept",
    "wetterich_equation",
    "regulator_comparison",
    "fp_stability_concept",
    "cross_analogue_bridge",
    "hydraulic_analogy",
    "heat_kernel_coefficients",
    "scale_identification",
    "scattering_concept",
    "scattering_bridge",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", default="docs/images",
        help="Directory for generated images (default: docs/images)",
    )
    parser.add_argument(
        "--format", default="png", choices=["png", "pdf", "svg"],
        help="Output image format (default: png)",
    )
    parser.add_argument(
        "--workers", type=int,
        default=max(1, min(8, (os.cpu_count() or 2))),
        help="Parallel worker processes (default: min(8, ncpu)). "
             "Set to 1 for sequential / debugging.",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Per-figure timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--only", action="append", default=None,
        help="Generate only the named figure(s). Can be passed multiple "
             "times. Example: --only flow_basin_3d --only quadratic_pairwise",
    )
    parser.add_argument(
        "--update-baselines", action="store_true",
        help=(
            "After generating each figure listed in `_HASH_TESTED`, "
            "write its sha256 to tests/baselines/<name>.png.sha256. "
            "Use this when intentional rendering changes have been made "
            "and the regression baselines need to be refreshed."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from asymsafety.visualization.style import apply_style
    apply_style()

    selected = ALL_GENERATORS
    if args.only:
        wanted = {f"gen_{name}" for name in args.only}
        selected = [g for g in ALL_GENERATORS if g.__name__ in wanted]
        if not selected:
            print(f"No generators match --only {args.only}", file=sys.stderr)
            sys.exit(2)

    print(
        f"Generating {len(selected)} figures → {out_dir}/ "
        f"({args.format}, workers={args.workers}, "
        f"timeout={args.timeout:.0f}s)\n"
    )
    t0 = time.time()

    n_ok = 0
    failures: list[tuple[str, str]] = []

    if args.workers <= 1:
        # Sequential path (debugging / single-CPU). One printed line
        # per figure, mirroring the parallel path's output format.
        for gen_fn in selected:
            name = gen_fn.__name__[len("gen_"):]
            t0 = time.time()
            try:
                gen_fn(out_dir, args.format)
                elapsed = time.time() - t0
                print(
                    f"  OK   {name}.{args.format}  ({elapsed:.1f}s)",
                    flush=True,
                )
                if args.update_baselines and name in _HASH_TESTED:
                    _refresh_baseline(out_dir, name, args.format)
            except Exception as exc:
                elapsed = time.time() - t0
                print(
                    f"  FAIL {name}: {type(exc).__name__}: {exc}  "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
        n_ok = len(_ok)
        failures = list(_fail)
    else:
        # Parallel path: each figure runs in a fresh subprocess. The
        # "spawn" start method works on every platform and avoids
        # accidentally inheriting a half-initialised matplotlib /
        # SymPy state from the parent.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=ctx,
        ) as ex:
            futures = {
                ex.submit(
                    _worker,
                    gen_fn.__name__, str(out_dir), args.format, args.timeout,
                ): gen_fn.__name__
                for gen_fn in selected
            }
            for fut in as_completed(futures):
                gen_name = futures[fut]
                try:
                    name, status, elapsed = fut.result()
                except Exception as exc:
                    name = gen_name[len("gen_"):]
                    status = f"FAIL: {exc}"
                    elapsed = 0.0
                if status == "OK":
                    n_ok += 1
                    print(
                        f"  OK   {name}.{args.format}  ({elapsed:.1f}s)",
                        flush=True,
                    )
                    if args.update_baselines and name in _HASH_TESTED:
                        _refresh_baseline(out_dir, name, args.format)
                else:
                    failures.append((name, status))
                    print(
                        f"  {status:<40}  {name}  ({elapsed:.1f}s)",
                        flush=True,
                    )

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.1f}s: "
        f"{n_ok} succeeded, {len(failures)} failed."
    )
    if failures:
        print("\nFailed figures:")
        for name, err in failures:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
