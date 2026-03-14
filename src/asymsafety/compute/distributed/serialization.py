"""Serialization of BetaFunctionSystem for distributed execution.

SymPy expressions are serialized via sympy.srepr() (deterministic string
representation). Lambdified callables cannot be pickled, so workers
re-lambdify from the expression strings.

For BOINC/crowd workers that may lack SymPy, standalone Python code
can be generated via sympy.pycode().
"""

from __future__ import annotations
import json

import sympy
from sympy import Symbol

from asymsafety.beta.system import BetaFunction, BetaFunctionSystem


def serialize_beta_system(system: BetaFunctionSystem) -> dict:
    """Serialize a BetaFunctionSystem to a JSON-compatible dict.

    The dict contains SymPy expression strings that can be reconstructed
    on any machine with SymPy installed.
    """
    data = {
        "coupling_names": system.coupling_names,
        "couplings": [],
        "expressions": [],
    }
    for name in system.coupling_names:
        beta = system.beta(name)
        data["couplings"].append({
            "name": name,
            "symbol": str(beta.coupling_symbol),
            "assumptions": {k: v for k, v in beta.coupling_symbol.assumptions0.items()},
        })
        data["expressions"].append(sympy.srepr(beta.expression))
    return data


def deserialize_beta_system(data: dict) -> BetaFunctionSystem:
    """Reconstruct a BetaFunctionSystem from serialized data."""
    system = BetaFunctionSystem()
    for coupling_info, expr_str in zip(data["couplings"], data["expressions"]):
        sym = Symbol(coupling_info["symbol"], **coupling_info["assumptions"])
        expr = sympy.sympify(expr_str)
        system.add(BetaFunction(
            coupling_name=coupling_info["name"],
            coupling_symbol=sym,
            expression=expr,
        ))
    return system


def serialize_to_json(system: BetaFunctionSystem) -> str:
    """Serialize to a JSON string."""
    return json.dumps(serialize_beta_system(system), indent=2)


def deserialize_from_json(json_str: str) -> BetaFunctionSystem:
    """Deserialize from a JSON string."""
    return deserialize_beta_system(json.loads(json_str))


def generate_standalone_code(system: BetaFunctionSystem) -> str:
    """Generate standalone Python code for beta functions (no SymPy needed).

    Useful for BOINC/crowd computing workers that may not have SymPy.
    The generated code only requires numpy.
    """
    lines = [
        '"""Auto-generated beta functions. Requires only numpy."""',
        "import numpy as np",
        "",
    ]

    symbols = system.coupling_symbols
    # Sanitize symbol names: Python keywords like 'lambda' become 'lambda_'
    import keyword
    safe_names = []
    rename_map = {}
    for s in symbols:
        name = str(s)
        if keyword.iskeyword(name):
            safe = name + "_"
            rename_map[name] = safe
            safe_names.append(safe)
        else:
            safe_names.append(name)
    args = ", ".join(safe_names)

    for name in system.coupling_names:
        beta = system.beta(name)
        code = sympy.pycode(beta.expression)
        # Replace keyword symbol names in generated code
        for orig, safe in rename_map.items():
            code = code.replace(orig, safe)
        fn_name = name if not keyword.iskeyword(name) else name + "_"
        lines.append(f"def beta_{fn_name}({args}):")
        lines.append(f"    return {code}")
        lines.append("")

    # Add a combined evaluator
    fn_names = []
    for n in system.coupling_names:
        fn_name = n if not keyword.iskeyword(n) else n + "_"
        fn_names.append(f"beta_{fn_name}({args})")
    lines.append(f"def evaluate_all({args}):")
    lines.append(f"    return [{', '.join(fn_names)}]")
    lines.append("")

    return "\n".join(lines)
