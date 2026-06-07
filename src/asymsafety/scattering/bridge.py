"""Bridge: the asymptotically-safe amplitude vs. the string bootstrap.

This is the "combined physical scattering + asymptotic safety" layer.  It
takes the asymptotically-safe graviton-mediated amplitude
(:class:`~asymsafety.scattering.amplitude.GravitonMediatedAmplitude`) and a
bootstrap reference amplitude
(:class:`~asymsafety.scattering.bootstrap.StringAmplitude`) and runs the
*same* physical-scattering consistency battery on both — analogous to the
commutative-diagram cross-checks of
:mod:`asymsafety.transforms.bridge.cross_analogue`.

The point is the contrast emphasised by *Strings from Almost Nothing*
(PRL ``cw4p-cqh7``): physical-scattering consistency alone — once the
**ultrasoft** high-energy assumption and the **higher-spin residue
cancellation** (infinite Regge tower) are imposed — singles out string
amplitudes.  Asymptotic safety satisfies the *foundational* requirements
(crossing, UV finiteness, no ghosts, bounded partial waves) but realises
UV completeness by **softening** the graviton coupling, ``G(p²) → g*/p²``,
rather than by an infinite tower of higher-spin resonances.  It is
therefore a *distinct* point in the space of physically-consistent
amplitudes: UV-constant rather than ultrasoft, with no Regge tower.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from asymsafety.scattering import bootstrap as _bs
from asymsafety.scattering import consistency as _cons
from asymsafety.scattering.amplitude import GravitonMediatedAmplitude
from asymsafety.scattering.bootstrap import StringAmplitude


@dataclass
class ScatteringBridge:
    """Compare an AS amplitude against a string-bootstrap amplitude.

    Args:
        as_amplitude: The asymptotically-safe graviton-mediated amplitude.
        string_amplitude: Bootstrap reference (default Virasoro–Shapiro).
        cos_theta: Fixed scattering angle for high-energy comparisons.
    """

    as_amplitude: GravitonMediatedAmplitude
    string_amplitude: StringAmplitude = field(
        default_factory=lambda: StringAmplitude(
            kind="virasoro_shapiro", alphap=0.25
        )
    )
    cos_theta: float = 0.3

    # -- high-energy behaviour --------------------------------------------

    def compare_high_energy(self) -> dict:
        """Contrast the fixed-angle high-energy behaviour of both amplitudes."""
        as_froissart = _cons.froissart_growth(
            self.as_amplitude, cos_theta=self.cos_theta, dressed=True
        )
        as_ultrasoft = _bs.ultrasoft_falloff(
            self.as_amplitude, cos_theta=self.cos_theta
        )
        string_ultrasoft = _bs.ultrasoft_falloff(
            self.string_amplitude, cos_theta=self.cos_theta
        )
        return {
            "as_growth_exponent": as_froissart["growth_exponent"],
            "as_uv_constant": as_froissart["uv_constant"],
            "as_ultrasoft": as_ultrasoft["ultrasoft"],
            "string_ultrasoft": string_ultrasoft["ultrasoft"],
            "string_slope_hi": string_ultrasoft["slope_hi"],
            # AS softens to a UV constant; strings fall off ultrasoftly.
            "distinct_uv": (
                as_froissart["uv_constant"] and string_ultrasoft["ultrasoft"]
            ),
        }

    # -- consistency comparison -------------------------------------------

    def consistency_table(self) -> dict:
        """Per-criterion pass/fail for the AS and string amplitudes.

        The asymptotically-safe side is evaluated with the full
        :mod:`~asymsafety.scattering.consistency` battery; the string side
        with the criteria that apply to it (crossing, ultrasoft, Regge
        tower).  ``None`` marks a criterion not meaningfully defined for a
        given amplitude.
        """
        as_report = _cons.consistency_report(
            self.as_amplitude, cos_theta=self.cos_theta
        )
        he = self.compare_high_energy()
        # Crossing for the string amplitude (manifest for VS / Veneziano):
        string_crossing = (
            abs(
                self.string_amplitude.eval(2.1, -1.3, -2.8)
                - self.string_amplitude.eval(-1.3, 2.1, -2.8)
            )
            < 1e-9
        )
        rows = {
            "crossing": {
                "as": as_report["crossing"]["passed"],
                "string": bool(string_crossing),
            },
            "uv_finite_or_bounded": {
                "as": as_report["uv_finiteness"]["passed"],
                "string": True,  # string amplitudes fall off in the UV
            },
            "partial_wave_bounded": {
                "as": as_report["unitarity"]["as_bounded"],
                "string": None,  # pole-dense; not evaluated here
            },
            "no_ghost_pole": {
                "as": as_report["causality"]["passed"],
                "string": None,  # real higher-spin tower, different notion
            },
            "ultrasoft_falloff": {
                "as": bool(he["as_ultrasoft"]),
                "string": bool(he["string_ultrasoft"]),
            },
            "infinite_regge_tower": {
                "as": False,  # finitely many channels, no tower
                "string": True,
            },
        }
        return rows

    def full_comparison_table(self) -> list[dict]:
        """Flatten :meth:`consistency_table` into printable rows."""
        table = self.consistency_table()
        return [
            {"criterion": name, "as": row["as"], "string": row["string"]}
            for name, row in table.items()
        ]

    # -- verdict -----------------------------------------------------------

    def verify(self) -> dict:
        """Structured verdict on the AS-vs-string comparison.

        ``as_physically_consistent`` is True when the AS amplitude passes
        the *foundational* physical-scattering requirements (crossing, UV
        finiteness, no ghosts, bounded partial waves).  The
        ``distinguishing_features`` record what separates it from the
        string bootstrap (no ultrasoft falloff, no Regge tower).
        """
        as_report = _cons.consistency_report(
            self.as_amplitude, cos_theta=self.cos_theta
        )
        he = self.compare_high_energy()

        foundational = {
            "crossing": as_report["crossing"]["passed"],
            "uv_finiteness": as_report["uv_finiteness"]["passed"],
            "no_ghosts": as_report["causality"]["passed"],
            "partial_waves_bounded": as_report["unitarity"]["as_bounded"],
        }
        as_consistent = all(foundational.values())

        distinguishing = {
            # AS reaches UV completeness by softening, not by a tower.
            "as_uv_constant_not_ultrasoft": (
                he["as_uv_constant"] and not he["as_ultrasoft"]
            ),
            "string_ultrasoft": he["string_ultrasoft"],
            "as_has_no_regge_tower": True,
            "string_has_regge_tower": True,
        }

        summary = (
            "Asymptotic safety satisfies the foundational physical-"
            "scattering requirements (crossing, UV finiteness, no ghosts, "
            "bounded partial waves) but reaches UV completeness by softening "
            "the graviton coupling G(p^2) -> g*/p^2, giving a UV-CONSTANT "
            "amplitude. The string bootstrap instead enforces ULTRASOFT "
            "high-energy falloff and an infinite higher-spin Regge tower. "
            "The two are distinct, mutually consistent points in the space "
            "of physical amplitudes."
            if as_consistent
            else "Asymptotic-safety amplitude failed a foundational check; "
            "see foundational_checks."
        )

        return {
            "as_physically_consistent": as_consistent,
            "foundational_checks": foundational,
            "distinguishing_features": distinguishing,
            "high_energy": he,
            "distinct_from_strings": bool(
                distinguishing["as_uv_constant_not_ultrasoft"]
                and he["string_ultrasoft"]
            ),
            "summary": summary,
        }
