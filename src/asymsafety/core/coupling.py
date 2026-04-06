"""Coupling constants and their properties."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import sympy
from sympy import Rational, Symbol


@dataclass
class Coupling:
    """A single coupling constant in the gravitational action.

    Attributes:
        name: Human-readable name (e.g., "Newton", "cosmological").
        symbol: SymPy symbol for the dimensionful coupling.
        canonical_dim: Mass dimension of the coupling.
        dimensionless_symbol: SymPy symbol for the dimensionless version.
    """

    name: str
    symbol: Symbol
    canonical_dim: Rational | int | sympy.Expr
    dimensionless_symbol: Symbol | None = None

    def __post_init__(self):
        if self.dimensionless_symbol is None:
            self.dimensionless_symbol = Symbol(
                self.symbol.name.lower(), real=True
            )

    def make_dimensionless(self, k: Symbol) -> sympy.Expr:
        """Convert to dimensionless coupling: g_dimless = k^{-d_can} * G."""
        return k ** (-self.canonical_dim) * self.symbol


@dataclass
class CouplingSet:
    """An ordered collection of couplings defining a truncation.

    Provides dict-like access by name and vector operations for
    fixed point analysis.
    """

    _couplings: OrderedDict[str, Coupling] = field(
        default_factory=OrderedDict
    )

    def add(self, coupling: Coupling) -> None:
        """Add a coupling to the set."""
        self._couplings[coupling.name] = coupling

    def __getitem__(self, name: str) -> Coupling:
        return self._couplings[name]

    def __contains__(self, name: str) -> bool:
        return name in self._couplings

    def __iter__(self):
        return iter(self._couplings.values())

    def __len__(self) -> int:
        return len(self._couplings)

    @property
    def names(self) -> list[str]:
        return list(self._couplings.keys())

    @property
    def symbols(self) -> list[Symbol]:
        return [c.symbol for c in self._couplings.values()]

    @property
    def dimensionless_symbols(self) -> list[Symbol]:
        return [c.dimensionless_symbol for c in self._couplings.values()]

    def dimensionless_vector(self) -> list[Symbol]:
        """Return all dimensionless coupling symbols as an ordered list."""
        return self.dimensionless_symbols


def make_eh_couplings(d: int = 4) -> CouplingSet:
    """Create the Einstein-Hilbert coupling set: (G, Λ)."""
    cs = CouplingSet()
    cs.add(Coupling(
        name="Newton",
        symbol=Symbol("G", positive=True),
        canonical_dim=2 - d,
        dimensionless_symbol=Symbol("g", positive=True),
    ))
    cs.add(Coupling(
        name="cosmological",
        symbol=Symbol("Lambda", real=True),
        canonical_dim=2,
        dimensionless_symbol=Symbol("lambda", real=True),
    ))
    return cs


def make_quadratic_couplings(d: int = 4) -> CouplingSet:
    """Create the quadratic gravity coupling set: (G, Λ, α, β)."""
    cs = make_eh_couplings(d)
    cs.add(Coupling(
        name="R_squared",
        symbol=Symbol("alpha_dim", real=True),
        canonical_dim=d - 4,
        dimensionless_symbol=Symbol("alpha", real=True),
    ))
    cs.add(Coupling(
        name="Weyl_squared",
        symbol=Symbol("beta_dim", real=True),
        canonical_dim=d - 4,
        dimensionless_symbol=Symbol("beta", real=True),
    ))
    return cs


def make_foliated_eh_couplings(d: int = 4) -> CouplingSet:
    """Create foliated EH couplings: (G, Λ, λ_ADM)."""
    cs = make_eh_couplings(d)
    cs.add(Coupling(
        name="lambda_ADM",
        symbol=Symbol("lambda_ADM", real=True),
        canonical_dim=0,
        dimensionless_symbol=Symbol("lambda_ADM", real=True),
    ))
    return cs


def make_gravity_matter_couplings(
    d: int = 4,
    scalar_quartic: bool = False,
    yukawa: bool = False,
    running_xi: bool = False,
) -> CouplingSet:
    """Create gravity-matter coupling set.

    Always includes (G, Λ). Optionally adds λ_φ, y, ξ.
    """
    cs = make_eh_couplings(d)
    if scalar_quartic:
        cs.add(Coupling(
            name="scalar_quartic",
            symbol=Symbol("lambda_phi_dim", real=True),
            canonical_dim=4 - d,
            dimensionless_symbol=Symbol("lambda_phi", real=True),
        ))
    if yukawa:
        cs.add(Coupling(
            name="Yukawa",
            symbol=Symbol("y_dim", real=True),
            canonical_dim=Rational(4 - d, 2),
            dimensionless_symbol=Symbol("y", positive=True),
        ))
    if running_xi:
        cs.add(Coupling(
            name="non_minimal",
            symbol=Symbol("xi_dim", real=True),
            canonical_dim=0,
            dimensionless_symbol=Symbol("xi", real=True),
        ))
    return cs
