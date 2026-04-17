"""Map a CLI ``--truncation`` string to a :class:`BetaFunctionSystem`.

Centralizes the dispatch so both ``eval`` and ``scan`` subcommands call
the same builder. Optional matter content and extension flags are passed
through ``params`` (the parsed --param dictionary).
"""

from __future__ import annotations

from asymsafety.actions.matter import MatterContent
from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.beta.foliated import build_foliated_eh_beta_system
from asymsafety.beta.matter import (
    build_eh_matter_beta_system,
    build_gravity_matter_fp_system,
)
from asymsafety.beta.quadratic import build_quadratic_beta_system
from asymsafety.beta.system import BetaFunctionSystem


SUPPORTED_TRUNCATIONS = (
    "eh",
    "quadratic",
    "foliated",
    "eh_matter",
    "gravity_matter",
)


def build_truncation(
    name: str, params: dict[str, str | int | float | bool]
) -> BetaFunctionSystem:
    """Construct a beta system by name.

    Args:
        name: One of :data:`SUPPORTED_TRUNCATIONS`.
        params: Free-form key/value pairs from ``--param``. Recognized keys
            depend on the truncation:

                * ``d`` (int): spacetime dimension (default 4)
                * ``n_scalars`` / ``n_dirac`` / ``n_vectors`` (int): matter
                  content for ``eh_matter`` / ``gravity_matter``
                * ``scalar_quartic`` / ``yukawa`` / ``running_xi`` (bool):
                  extensions for ``gravity_matter``
                * ``lorentzian`` (bool): for ``foliated``

    Returns:
        A populated :class:`BetaFunctionSystem`.

    Raises:
        ValueError: If ``name`` is not supported or required params missing.
    """
    d = int(params.get("d", 4))

    if name == "eh":
        return build_eh_beta_system(d=d)

    if name == "quadratic":
        return build_quadratic_beta_system(d=d)

    if name == "foliated":
        lorentzian = _as_bool(params.get("lorentzian", True))
        return build_foliated_eh_beta_system(d=d, lorentzian=lorentzian)

    if name == "eh_matter":
        matter = _matter_from_params(params)
        return build_eh_matter_beta_system(matter, d=d)

    if name == "gravity_matter":
        matter = _matter_from_params(params)
        return build_gravity_matter_fp_system(
            matter,
            d=d,
            scalar_quartic=_as_bool(params.get("scalar_quartic", True)),
            yukawa=_as_bool(params.get("yukawa", False)),
            running_xi=_as_bool(params.get("running_xi", False)),
        )

    raise ValueError(
        f"Unknown truncation {name!r}. "
        f"Supported: {', '.join(SUPPORTED_TRUNCATIONS)}"
    )


def _matter_from_params(params: dict) -> MatterContent:
    return MatterContent(
        n_scalars=int(params.get("n_scalars", 0)),
        n_dirac=int(params.get("n_dirac", 0)),
        n_vectors=int(params.get("n_vectors", 0)),
    )


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def parse_kv_pairs(s: str | None) -> dict[str, str]:
    """Parse ``key=value,key=value`` into a dict (strings only).

    Empty input → empty dict. Whitespace tolerated. Values are not coerced
    here — :func:`build_truncation` does its own coercion based on key.
    """
    if not s:
        return {}
    out: dict[str, str] = {}
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--param token {token!r} missing '='")
        k, v = token.split("=", 1)
        out[k.strip()] = v.strip()
    return out
