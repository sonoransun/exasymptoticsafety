"""Smoke tests for every public plotting function.

These tests verify that each visualisation function returns a
non-empty :class:`matplotlib.figure.Figure` with the expected number
of axes and the expected text content (axis labels, legend entries,
inline citations). They use the ``Agg`` backend and tiny grids so the
suite stays fast.

The set covered:

- :mod:`asymsafety.visualization.style` helpers
  (``add_reference_box``, ``add_legend_panel``, ``theta_label``,
  ``format_arxiv``).
- 2D plotters in :mod:`asymsafety.visualization.phase_portrait`,
  :mod:`asymsafety.visualization.fixed_point_plot`,
  :mod:`asymsafety.visualization.bridge_diagram`,
  :mod:`asymsafety.visualization.conceptual`.
- 3D plotters in :mod:`asymsafety.gui.visualization_3d`.
- Cross-domain plotters in :mod:`asymsafety.hydraulic.visualization`.
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


# ---------------------------------------------------------------------------
# style.py helpers
# ---------------------------------------------------------------------------


class TestStyleHelpers:
    def test_format_arxiv(self):
        from asymsafety.visualization.style import format_arxiv
        assert format_arxiv("Reuter (1998)", "hep-th/9605030") == (
            "Reuter (1998) [hep-th/9605030]"
        )

    def test_theta_label_real(self):
        from asymsafety.visualization.style import theta_label
        out = theta_label(1.94)
        assert "1.94" in out
        assert "i" not in out

    def test_theta_label_complex(self):
        from asymsafety.visualization.style import theta_label
        out = theta_label(1.94 + 3.15j)
        assert "1.94" in out
        assert "3.15i" in out

    def test_add_reference_box_renders_text(self):
        from asymsafety.visualization.style import (
            add_reference_box, format_arxiv,
        )
        fig, ax = plt.subplots()
        add_reference_box(ax, [
            format_arxiv("Reuter (1998)", "hep-th/9605030"),
            format_arxiv("Litim (2001)", "hep-th/0103195"),
        ])
        # Reference box renders all citations in a single ax.text call;
        # search the rendered figure text.
        all_text = " ".join(t.get_text() for t in ax.texts)
        assert "hep-th/9605030" in all_text
        assert "hep-th/0103195" in all_text

    def test_add_legend_panel(self):
        from asymsafety.visualization.style import (
            add_legend_panel, COLOR_RELEVANT, COLOR_IRRELEVANT,
        )
        fig, ax = plt.subplots()
        add_legend_panel(ax, [
            (f"line|{COLOR_RELEVANT}", "relevant"),
            (f"line|{COLOR_IRRELEVANT}", "irrelevant"),
            ("marker_star", "NGFP"),
            ("patch|#06D6A0", "basin"),
        ])
        leg = ax.get_legend()
        assert leg is not None
        labels = [t.get_text() for t in leg.get_texts()]
        assert "relevant" in labels
        assert "NGFP" in labels


# ---------------------------------------------------------------------------
# 2D phase portrait
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eh_system():
    from asymsafety.beta.einstein_hilbert import build_eh_beta_system
    return build_eh_beta_system(d=4)


@pytest.fixture(scope="module")
def eh_ngfp(eh_system):
    from asymsafety.analysis.fixed_points import FixedPointFinder
    return FixedPointFinder(eh_system).find_fixed_point(
        {"g": 0.7, "lambda": 0.14}
    )


class Test2DPlotters:
    def test_phase_portrait_2d(self, eh_system, eh_ngfp):
        from asymsafety.visualization.phase_portrait import phase_portrait_2d
        fig = phase_portrait_2d(
            eh_system, "g", "lambda",
            x_range=(0.0, 1.2), y_range=(-0.2, 0.4),
            n_grid=8, fixed_points=[eh_ngfp],
        )
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert "g" in ax.get_xlabel()
        plt.close(fig)

    def test_flow_diagram(self, eh_system):
        from asymsafety.analysis.flow import FlowIntegrator
        from asymsafety.visualization.phase_portrait import flow_diagram
        integ = FlowIntegrator(eh_system)
        trajs = [integ.integrate({"g": 0.5, "lambda": 0.1},
                                 t_span=(-1, 1))]
        fig = flow_diagram(trajs)
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert "log" in ax.get_xlabel().lower() or "k_0" in ax.get_xlabel()
        plt.close(fig)

    def test_annotated_eh_phase_portrait(self):
        from asymsafety.visualization.phase_portrait import (
            annotated_eh_phase_portrait,
        )
        fig = annotated_eh_phase_portrait(show_separatrix=False)
        assert isinstance(fig, Figure)
        # Should have streamplot axes + colorbar
        assert len(fig.axes) >= 2
        # Check that a reference-box citation is present in figure text
        all_text = " ".join(
            t.get_text()
            for ax in fig.axes
            for t in ax.texts
        )
        assert "hep-th/9605030" in all_text
        plt.close(fig)

    def test_separatrix_overlay(self, eh_system, eh_ngfp):
        from asymsafety.visualization.phase_portrait import separatrix_overlay
        fig = separatrix_overlay(eh_system, eh_ngfp)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_regulator_comparison_panels(self, eh_system):
        from asymsafety.visualization.phase_portrait import (
            regulator_comparison_panels,
        )
        # Even with the same system twice the function should run cleanly
        # and emit three panels.
        fig = regulator_comparison_panels(eh_system, eh_system, n_grid=10)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_quadratic_pairwise_grid(self):
        from asymsafety.beta.quadratic import build_quadratic_beta_system
        from asymsafety.visualization.phase_portrait import (
            quadratic_pairwise_grid,
        )
        sys = build_quadratic_beta_system(d=4)
        fig = quadratic_pairwise_grid(sys, n_grid=8)
        assert isinstance(fig, Figure)
        # Six pairwise panels
        assert sum(1 for ax in fig.axes if ax.get_xlabel()) >= 4
        plt.close(fig)


# ---------------------------------------------------------------------------
# Fixed point plot
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def matter_continuation_result():
    from asymsafety.actions.matter import MatterContent
    from asymsafety.analysis.continuation import continuation
    from asymsafety.beta.matter import build_eh_matter_beta_system

    def builder(n: float):
        return build_eh_matter_beta_system(
            MatterContent(n_scalars=int(n)), d=4
        )

    return continuation(
        system_builder=builder,
        parameter_name="N_s",
        parameter_values=np.array([0.0, 2.0, 4.0]),
        initial_guess={"g": 0.7, "lambda": 0.14},
    )


class TestFixedPointPlot:
    def test_plot_critical_exponents(self, matter_continuation_result):
        from asymsafety.visualization.fixed_point_plot import (
            plot_critical_exponents,
        )
        fig = plot_critical_exponents(matter_continuation_result)
        assert isinstance(fig, Figure)
        # Two panels (Re and Im)
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_plot_fixed_point_locations(self, matter_continuation_result):
        from asymsafety.visualization.fixed_point_plot import (
            plot_fixed_point_locations,
        )
        fig = plot_fixed_point_locations(
            matter_continuation_result,
            reference_values={"g": 0.707, "lambda": 0.193},
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_matter_content_continuation(
        self, matter_continuation_result
    ):
        from asymsafety.visualization.fixed_point_plot import (
            plot_matter_content_continuation,
        )
        fig = plot_matter_content_continuation(matter_continuation_result)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Conceptual + bridge diagrams
# ---------------------------------------------------------------------------


class TestConceptualPlots:
    @pytest.mark.parametrize("fn_name", [
        "asymptotic_safety_concept",
        "regulator_comparison",
        "fixed_point_stability_concept",
        "wetterich_equation_diagram",
        "scattering_concept",
    ])
    def test_conceptual(self, fn_name):
        import asymsafety.visualization.conceptual as mod
        fn = getattr(mod, fn_name)
        fig = fn()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_cross_analogue_bridge(self):
        from asymsafety.visualization.bridge_diagram import (
            cross_analogue_bridge_diagram,
        )
        fig = cross_analogue_bridge_diagram()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_hydraulic_analogy(self):
        from asymsafety.visualization.bridge_diagram import (
            hydraulic_analogy_diagram,
        )
        fig = hydraulic_analogy_diagram()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_scattering_bridge(self):
        from asymsafety.visualization.bridge_diagram import (
            scattering_bridge_diagram,
        )
        fig = scattering_bridge_diagram()
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 3D plotters
# ---------------------------------------------------------------------------


class Test3DPlotters:
    def test_flow_trajectories_3d_world_line(self, eh_system, eh_ngfp):
        from asymsafety.analysis.flow import FlowIntegrator
        from asymsafety.gui.visualization_3d import flow_trajectories_3d
        integ = FlowIntegrator(eh_system)
        trajs = [integ.integrate({"g": 0.5, "lambda": 0.1},
                                 t_span=(-1, 1))]
        # World-line view: z = RG time
        fig = flow_trajectories_3d(
            trajs, "g", "lambda", z_coupling=None,
            fixed_points=[eh_ngfp], show_eigenvectors=False,
            show_speed_widths=False,
        )
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert "log" in ax.get_zlabel().lower()
        plt.close(fig)

    def test_flow_trajectories_3d_warns_on_duplicate_z(
        self, eh_system, eh_ngfp,
    ):
        from asymsafety.analysis.flow import FlowIntegrator
        from asymsafety.gui.visualization_3d import flow_trajectories_3d
        integ = FlowIntegrator(eh_system)
        trajs = [integ.integrate({"g": 0.5, "lambda": 0.1},
                                 t_span=(-1, 1))]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = flow_trajectories_3d(
                trajs, "g", "lambda", "g",
                fixed_points=[eh_ngfp], show_eigenvectors=False,
                show_speed_widths=False,
            )
            assert any(
                "duplicates" in str(w.message) for w in caught
            ), "expected a deprecation/duplicate warning"
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_phase_portrait_3d_rejects_duplicate_z(self, eh_system):
        from asymsafety.gui.visualization_3d import phase_portrait_3d
        with pytest.raises(ValueError):
            phase_portrait_3d(eh_system, "g", "lambda", "g")

    def test_phase_portrait_3d_quadratic(self):
        from asymsafety.beta.quadratic import build_quadratic_beta_system
        from asymsafety.gui.visualization_3d import phase_portrait_3d
        sys = build_quadratic_beta_system(d=4)
        fig = phase_portrait_3d(
            sys, "g", "lambda", "alpha",
            x_range=(0.0, 1.2), y_range=(-0.2, 0.4),
            z_range=(-0.05, 0.05), n_grid=4,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_fixed_point_stability_3d(self):
        from asymsafety.beta.foliated import build_foliated_eh_beta_system
        from asymsafety.analysis.fixed_points import FixedPointFinder
        from asymsafety.analysis.stability import analyze_stability
        from asymsafety.gui.visualization_3d import fixed_point_stability_3d
        sys = build_foliated_eh_beta_system(d=4)
        fp = FixedPointFinder(sys).find_fixed_point(
            {"g": 0.96, "lambda": 0.20, "lambda_ADM": 1.0}
        )
        sa = analyze_stability(sys, fp)
        fig = fixed_point_stability_3d(
            fp, sa, "g", "lambda", "lambda_ADM",
            scale=0.3, show_theta_labels=True,
        )
        assert isinstance(fig, Figure)
        # Legend should be present (proxy-artist legend panel)
        leg = fig.axes[0].get_legend()
        assert leg is not None
        plt.close(fig)

    def test_fixed_point_stability_3d_rejects_duplicate(self):
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.analysis.fixed_points import FixedPointFinder
        from asymsafety.analysis.stability import analyze_stability
        from asymsafety.gui.visualization_3d import fixed_point_stability_3d
        sys = build_eh_beta_system(d=4)
        fp = FixedPointFinder(sys).find_fixed_point(
            {"g": 0.7, "lambda": 0.14}
        )
        sa = analyze_stability(sys, fp)
        with pytest.raises(ValueError):
            fixed_point_stability_3d(fp, sa, "g", "lambda", "g")

    def test_foliated_phase_portrait_3d(self):
        from asymsafety.beta.foliated import build_foliated_eh_beta_system
        from asymsafety.gui.visualization_3d import foliated_phase_portrait_3d
        sys = build_foliated_eh_beta_system(d=4)
        fig = foliated_phase_portrait_3d(
            sys, n_grid=4, show_references=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_flow_basin_3d_smoke(self, eh_system, eh_ngfp):
        from asymsafety.gui.visualization_3d import flow_basin_3d
        # Tiny sample for speed
        fig = flow_basin_3d(
            eh_system, eh_ngfp,
            x_coupling="g", y_coupling="lambda", z_coupling="lambda",
            n_samples=12, radius=0.1,
            t_span=(0.0, 1.0),
            show_references=False,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Hydraulic plotters
# ---------------------------------------------------------------------------


class TestHydraulicPlotters:
    def test_plot_hydraulic_network(self, eh_system):
        from asymsafety.hydraulic.mapping import RGToHydraulicMapper
        from asymsafety.hydraulic.visualization import plot_hydraulic_network
        mapper = RGToHydraulicMapper(eh_system)
        net = mapper.build_network(
            reference_point={"g": 0.7, "lambda": 0.14}
        )
        fig = plot_hydraulic_network(net)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Transform plotters
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eh_stability(eh_system, eh_ngfp):
    from asymsafety.analysis.stability import analyze_stability
    return analyze_stability(eh_system, eh_ngfp)


@pytest.fixture(scope="module")
def eh_resolvent(eh_stability):
    from asymsafety.transforms.linear.resolvent import ResolventOperator
    return ResolventOperator(eh_stability)


@pytest.fixture(scope="module")
def eh_impedance_bode(eh_stability):
    from asymsafety.transforms.bridge.impedance import ImpedanceBridge
    return ImpedanceBridge(eh_stability.stability_matrix).bode_data(
        omega_range=(0.1, 10.0), n_points=24,
    )


@pytest.fixture(scope="module")
def eh_wavelet_result(eh_system):
    from asymsafety.analysis.flow import FlowIntegrator
    from asymsafety.transforms.integral.wavelet import RGFlowWavelet
    traj = FlowIntegrator(eh_system).integrate(
        {"g": 0.5, "lambda": 0.1}, t_span=(-1.0, 1.0),
    )
    return RGFlowWavelet(traj).transform(
        np.array([0.2, 0.5, 1.0]), wavelet_type="morlet",
    )


@pytest.fixture(scope="module")
def eh_comparison_table(eh_system, eh_ngfp, eh_stability):
    from asymsafety.transforms.bridge.cross_analogue import CrossAnalogueBridge
    return CrossAnalogueBridge(
        eh_system, eh_ngfp, eh_stability,
    ).full_comparison_table()


class TestTransformPlotters:
    def test_plot_bode(self, eh_impedance_bode):
        from asymsafety.transforms.visualization.transform_plots import plot_bode
        fig = plot_bode(eh_impedance_bode, entry=(0, 0))
        assert isinstance(fig, Figure)
        assert any("Bode" in (ax.get_title() or "") for ax in fig.axes)
        plt.close(fig)

    def test_plot_scalogram(self, eh_wavelet_result):
        from asymsafety.transforms.visualization.transform_plots import plot_scalogram
        fig = plot_scalogram(eh_wavelet_result, coupling_index=0)
        assert isinstance(fig, Figure)
        assert "Scale" in fig.axes[0].get_ylabel()
        plt.close(fig)

    def test_plot_pseudospectrum(self, eh_resolvent):
        from asymsafety.transforms.visualization.transform_plots import (
            plot_pseudospectrum,
        )
        fig = plot_pseudospectrum(
            eh_resolvent, epsilon_values=[1.0, 0.1],
            real_range=(-3, 3), imag_range=(-3, 3), n_grid=10,
        )
        assert isinstance(fig, Figure)
        assert "pseudospectrum" in fig.axes[0].get_title().lower()
        plt.close(fig)

    def test_plot_comparison_table(self, eh_comparison_table):
        from asymsafety.transforms.visualization.transform_plots import (
            plot_comparison_table,
        )
        fig = plot_comparison_table(eh_comparison_table)
        assert isinstance(fig, Figure)
        leg = fig.axes[0].get_legend()
        assert leg is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# Cosmology plotters
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eh_trajectory_for_cosmology(eh_system):
    from asymsafety.analysis.flow import FlowIntegrator
    return FlowIntegrator(eh_system).integrate(
        {"g": 0.5, "lambda": 0.1}, t_span=(-3.0, 3.0),
    )


@pytest.fixture(scope="module")
def supercritical_bh(eh_trajectory_for_cosmology):
    from asymsafety.cosmology.rg_improved_bh import RGImprovedSchwarzschild
    return RGImprovedSchwarzschild(
        trajectory=eh_trajectory_for_cosmology, M=2.0,
    )


class TestCosmologyPlotters:
    def test_plot_running_newton_constant(self, supercritical_bh):
        from asymsafety.cosmology.visualization import (
            plot_running_newton_constant,
        )
        fig = plot_running_newton_constant(
            supercritical_bh, r_range=(0.1, 10.0), n_points=40,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_lapse_with_horizons(self, supercritical_bh):
        from asymsafety.cosmology.visualization import plot_lapse_with_horizons
        fig = plot_lapse_with_horizons(
            supercritical_bh, r_range=(0.1, 10.0), n_points=80,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_classical_vs_rg_lapse(self, supercritical_bh):
        from asymsafety.cosmology.visualization import (
            plot_classical_vs_rg_lapse,
        )
        fig = plot_classical_vs_rg_lapse(
            supercritical_bh, M_values=(0.5, 1.0, 2.0),
            r_range=(0.1, 10.0), n_points=60,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_hawking_temperature(self, supercritical_bh):
        from asymsafety.cosmology.visualization import plot_hawking_temperature
        fig = plot_hawking_temperature(
            supercritical_bh, M_range=(0.5, 3.0), n_masses=8,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_scale_identification(self):
        from asymsafety.cosmology.scale_identification import (
            GeodesicDistanceScale, InverseDistanceScale,
        )
        from asymsafety.cosmology.visualization import (
            plot_scale_identification,
        )
        fig = plot_scale_identification(
            [InverseDistanceScale(), GeodesicDistanceScale(delta=0.5)],
            r_range=(0.1, 10.0), n_points=40,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_flrw_evolution(self, eh_trajectory_for_cosmology):
        from asymsafety.cosmology.rg_improved_flrw import RGImprovedFLRW
        from asymsafety.cosmology.visualization import plot_flrw_evolution
        flrw = RGImprovedFLRW(trajectory=eh_trajectory_for_cosmology)
        fig = plot_flrw_evolution(
            flrw, t_span=(0.05, 1.5), n_steps=20,
        )
        assert isinstance(fig, Figure)
        # Three stacked panels
        assert len(fig.axes) == 3
        plt.close(fig)


# ---------------------------------------------------------------------------
# Quantum plotters (qiskit-dependent ones use importorskip)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eh_edmd_result(eh_system):
    from asymsafety.analysis.flow import FlowIntegrator
    from asymsafety.quantum.koopman.operator import KoopmanOperator
    integ = FlowIntegrator(eh_system)
    trajs = [
        integ.integrate({"g": g0, "lambda": l0}, t_span=(-2.0, 2.0))
        for g0, l0 in [(0.5, 0.1), (0.3, 0.2)]
    ]
    return KoopmanOperator(eh_system).compute_edmd(
        trajs, dictionary_degree=3,
    )


@pytest.fixture(scope="module")
def scalar_partition_estimator():
    from asymsafety.geometry.decomposition import ModeSpectrum
    from asymsafety.quantum.thermal.gibbs import GibbsStatePreparer
    from asymsafety.quantum.thermal.hamiltonian import LaplacianHamiltonian
    from asymsafety.quantum.thermal.partition import PartitionFunctionEstimator
    spectrum = ModeSpectrum(d=4)
    ham = LaplacianHamiltonian(spectrum, field_type="scalar", l_max=3)
    gibbs = GibbsStatePreparer(ham)
    return gibbs, PartitionFunctionEstimator(gibbs, d=4)


class TestQuantumPlotters:
    def test_plot_koopman_spectrum(self, eh_edmd_result):
        from asymsafety.quantum.visualization import plot_koopman_spectrum
        fig = plot_koopman_spectrum(eh_edmd_result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_koopman_modes(self, eh_system, eh_edmd_result):
        from asymsafety.quantum.visualization import plot_koopman_modes
        fig = plot_koopman_modes(
            eh_edmd_result, eh_system.coupling_names, top_k=4,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_qft_power_spectrum(self, eh_system):
        pytest.importorskip("qiskit")
        from asymsafety.analysis.flow import FlowIntegrator
        from asymsafety.quantum.koopman.qft_rg import QuantumFourierRG
        from asymsafety.quantum.visualization import plot_qft_power_spectrum
        traj = FlowIntegrator(eh_system).integrate(
            {"g": 0.5, "lambda": 0.1}, t_span=(-1.0, 1.0),
        )
        fig = plot_qft_power_spectrum(QuantumFourierRG(traj), "g")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_grover_success_probability(self, eh_system):
        from asymsafety.quantum.grover.encoding import CouplingGridEncoding
        from asymsafety.quantum.grover.oracle import BetaOracle
        from asymsafety.quantum.grover.search import GroverFixedPointSearch
        from asymsafety.quantum.visualization import (
            plot_grover_success_probability,
        )
        encoding = CouplingGridEncoding(
            coupling_ranges={"g": (0.1, 1.4), "lambda": (-0.2, 0.4)},
            n_bits=3,
        )
        oracle = BetaOracle(eh_system, encoding, epsilon=0.5)
        search = GroverFixedPointSearch(eh_system, encoding, oracle)
        fig = plot_grover_success_probability(search, n_iter_max=8)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_grover_measurement_distribution(self, eh_system):
        pytest.importorskip("qiskit")
        from asymsafety.quantum.grover.encoding import CouplingGridEncoding
        from asymsafety.quantum.grover.oracle import BetaOracle
        from asymsafety.quantum.grover.search import GroverFixedPointSearch
        from asymsafety.quantum.visualization import (
            plot_grover_measurement_distribution,
        )
        encoding = CouplingGridEncoding(
            coupling_ranges={"g": (0.1, 1.4), "lambda": (-0.2, 0.4)},
            n_bits=3,
        )
        oracle = BetaOracle(eh_system, encoding, epsilon=0.5)
        search = GroverFixedPointSearch(eh_system, encoding, oracle)
        result = search.search(shots=128)
        fig = plot_grover_measurement_distribution(result, encoding)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_vqrg_cost_landscape(self, eh_system):
        pytest.importorskip("qiskit")
        from asymsafety.quantum.vqrg.circuit import VQRGCircuit
        from asymsafety.quantum.vqrg.cost import VQRGCostFunction
        from asymsafety.quantum.vqrg.mapping import CouplingAngleMap
        from asymsafety.quantum.visualization import plot_vqrg_cost_landscape
        cmap = CouplingAngleMap({"g": (0.0, 1.5), "lambda": (-0.4, 0.4)})
        circuit = VQRGCircuit(n_qubits=2, n_layers=1)
        cost = VQRGCostFunction(eh_system, cmap, circuit)
        base = np.zeros(circuit.n_parameters)
        fig = plot_vqrg_cost_landscape(cost, base, dims=(0, 1), n_grid=6)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_vqrg_optimization_trajectory(self):
        from asymsafety.quantum.visualization import (
            plot_vqrg_optimization_trajectory,
        )
        history = list(np.linspace(2.0, 0.001, 20))
        params = np.cumsum(np.random.RandomState(1).randn(20, 3) * 0.1, axis=0)
        fig = plot_vqrg_optimization_trajectory(
            history, parameters_history=params,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_quantum_fisher_eigenvalues(self):
        from asymsafety.quantum.visualization import (
            plot_quantum_fisher_eigenvalues,
        )
        rng = np.random.RandomState(0)
        A = rng.randn(4, 4)
        qfi = A.T @ A
        fig = plot_quantum_fisher_eigenvalues(
            qfi, coupling_names=("g", "lambda", "alpha", "beta"),
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_partition_function(self, scalar_partition_estimator):
        from asymsafety.quantum.visualization import plot_partition_function
        _, est = scalar_partition_estimator
        fig = plot_partition_function(est, beta_range=(0.05, 1.0), n_points=10)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_seeley_dewitt_extraction(self, scalar_partition_estimator):
        from asymsafety.quantum.visualization import plot_seeley_dewitt_extraction
        _, est = scalar_partition_estimator
        fig = plot_seeley_dewitt_extraction(est)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_gibbs_state_purity(self, scalar_partition_estimator):
        from asymsafety.quantum.visualization import plot_gibbs_state_purity
        gibbs, _ = scalar_partition_estimator
        fig = plot_gibbs_state_purity(
            gibbs, beta_range=(0.05, 2.0), n_points=10,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 3D extensions
# ---------------------------------------------------------------------------


class Test3DExtensions:
    def test_koopman_spectrum_3d(self, eh_edmd_result):
        from asymsafety.gui.visualization_3d import koopman_spectrum_3d
        fig = koopman_spectrum_3d(eh_edmd_result, show_references=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_vqrg_cost_isosurface_3d(self, eh_system):
        pytest.importorskip("qiskit")
        from asymsafety.gui.visualization_3d import vqrg_cost_isosurface_3d
        from asymsafety.quantum.vqrg.circuit import VQRGCircuit
        from asymsafety.quantum.vqrg.cost import VQRGCostFunction
        from asymsafety.quantum.vqrg.mapping import CouplingAngleMap
        cmap = CouplingAngleMap({"g": (0.0, 1.5), "lambda": (-0.4, 0.4)})
        circuit = VQRGCircuit(n_qubits=2, n_layers=1)
        cost = VQRGCostFunction(eh_system, cmap, circuit)
        base = np.zeros(circuit.n_parameters)
        fig = vqrg_cost_isosurface_3d(
            cost, base, dims=(0, 1, 2), n_grid=6,
            show_references=False,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Spectral / heat-kernel helpers
# ---------------------------------------------------------------------------


class TestSpectralAndHeatKernel:
    def test_plot_spectral_sum_convergence(self):
        from asymsafety.frg.spectral import SpectralSumEvaluator
        from asymsafety.visualization.phase_portrait import (
            plot_spectral_sum_convergence,
        )
        ss = SpectralSumEvaluator(d=4, l_max=32)
        fig = plot_spectral_sum_convergence(
            ss, n_max_values=[2, 4, 8, 16],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_heat_kernel_coefficients(self):
        from asymsafety.frg.heat_kernel import SeeleyDeWittCoefficients
        from asymsafety.visualization.phase_portrait import (
            plot_heat_kernel_coefficients,
        )
        sdw = SeeleyDeWittCoefficients(d=4, R_bar=1.0)
        fig = plot_heat_kernel_coefficients(
            sdw, field_types=("scalar", "vector"),
            coefficients=("b0", "b2"),
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_beta_norm_grid(self):
        from asymsafety.visualization.phase_portrait import plot_beta_norm_grid
        nx, ny = 6, 5
        arrays = {
            "axis_g": np.linspace(0.0, 1.0, nx),
            "axis_lambda": np.linspace(-0.2, 0.4, ny),
            "betas": np.random.RandomState(0).randn(nx * ny, 2),
        }
        fig = plot_beta_norm_grid(arrays)
        assert isinstance(fig, Figure)
        plt.close(fig)
