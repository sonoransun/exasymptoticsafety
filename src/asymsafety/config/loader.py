"""Runtime configuration loader.

Reads defaults from (in increasing priority):

    1. Hardcoded defaults in this file.
    2. ``~/.asymsafety.toml`` (user-global).
    3. ``./asymsafety.toml`` (project-local; whatever ``Path.cwd()`` is).
    4. ``ASYMSAFETY_<SECTION>_<KEY>`` environment variables.

Each layer overrides keys per-field; missing keys fall through. Use
``load_config()`` once at application entry; subsequent calls refresh
from disk.

The file format mirrors the dataclass structure:

.. code-block:: toml

    [solver]
    fixed_point_tol = 1.0e-10
    flow_max_step   = 0.1

    [cache]
    dir          = "~/.cache/asymsafety"
    disk_enabled = true

    [compute]
    batch_backend    = "auto"
    parallel_default = false
    max_workers      = 0          # 0 → OS default

    [plot]
    style = "default"

Environment override examples::

    ASYMSAFETY_SOLVER_FIXED_POINT_TOL=1e-6
    ASYMSAFETY_COMPUTE_BATCH_BACKEND=numpy
    ASYMSAFETY_CACHE_DISK_ENABLED=false
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SolverConfig:
    fixed_point_tol: float = 1.0e-10
    flow_max_step: float = 0.1
    flow_rtol: float = 1.0e-8
    flow_atol: float = 1.0e-10


@dataclass
class CacheConfig:
    dir: str = "~/.cache/asymsafety"
    disk_enabled: bool = True


@dataclass
class ComputeRuntimeConfig:
    batch_backend: str = "auto"          # numpy | jax | auto
    parallel_default: bool = False
    max_workers: int = 0                  # 0 → OS default


@dataclass
class PlotConfig:
    style: str = "default"


@dataclass
class Config:
    solver: SolverConfig = field(default_factory=SolverConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    compute: ComputeRuntimeConfig = field(default_factory=ComputeRuntimeConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

USER_CONFIG_PATH = Path("~/.asymsafety.toml").expanduser()
PROJECT_CONFIG_FILENAME = "asymsafety.toml"


def load_config(
    user_path: Path | str | None = None,
    project_path: Path | str | None = None,
    env: dict[str, str] | None = None,
) -> Config:
    """Construct a :class:`Config` by merging defaults, files, and env.

    Args:
        user_path: Override for the user-global config path. Defaults to
            ``~/.asymsafety.toml``. Pass ``False``-y values to skip.
        project_path: Override for the project-local config path. Defaults
            to ``./asymsafety.toml`` (resolved against ``Path.cwd()``).
        env: Mapping for environment variable lookup. Defaults to
            :data:`os.environ`.

    Returns:
        A fully-populated :class:`Config`.
    """
    if env is None:
        env = os.environ  # type: ignore[assignment]

    user_path = Path(user_path) if user_path else USER_CONFIG_PATH
    project_path = (
        Path(project_path) if project_path
        else Path.cwd() / PROJECT_CONFIG_FILENAME
    )

    cfg = Config()
    _merge_file(cfg, user_path)
    _merge_file(cfg, project_path)
    _merge_env(cfg, env)

    # Side effect: propagate cache.dir to the env so utils.caching picks
    # it up without an explicit argument. The env var (if pre-set by the
    # user) keeps priority, since _merge_env already ran.
    if "ASYMSAFETY_CACHE_DIR" not in env:
        os.environ["ASYMSAFETY_CACHE_DIR"] = str(
            Path(cfg.cache.dir).expanduser()
        )

    return cfg


# ---------------------------------------------------------------------------
# Internal merging
# ---------------------------------------------------------------------------

def _merge_file(cfg: Config, path: Path) -> None:
    """If ``path`` exists and parses, overwrite matching fields on ``cfg``."""
    if not path.is_file():
        return
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return
    for section_name, section_data in data.items():
        if not isinstance(section_data, dict):
            continue
        section = getattr(cfg, section_name, None)
        if section is None or not is_dataclass(section):
            continue
        _apply_section(section, section_data)


def _apply_section(section: Any, values: dict[str, Any]) -> None:
    """Set fields on a dataclass instance, coercing types when needed."""
    type_map = {f.name: f.type for f in fields(section)}
    for key, value in values.items():
        if key not in type_map:
            continue  # silently ignore unknown keys
        setattr(section, key, _coerce(value, type_map[key]))


def _merge_env(cfg: Config, env: dict[str, str]) -> None:
    """Apply ``ASYMSAFETY_<SECTION>_<KEY>`` overrides from env."""
    prefix = "ASYMSAFETY_"
    for env_key, raw in env.items():
        if not env_key.startswith(prefix):
            continue
        rest = env_key[len(prefix):].lower()
        # Find the longest section name that prefixes `rest`. Section names
        # are class attribute names on Config (no underscores in any of
        # them currently), so a single split on "_" suffices.
        for section_name in (f.name for f in fields(cfg)):
            head = section_name + "_"
            if rest.startswith(head):
                key = rest[len(head):]
                section = getattr(cfg, section_name)
                type_map = {f.name: f.type for f in fields(section)}
                if key in type_map:
                    setattr(section, key, _coerce(raw, type_map[key]))
                break


def _coerce(value: Any, type_hint: Any) -> Any:
    """Best-effort coercion of a TOML/env value to the dataclass field type.

    ``type_hint`` may be a string (PEP 563 deferred evaluation) or an
    actual type. Strings are interpreted as their leading word ('int',
    'float', 'bool', 'str').
    """
    target = _normalize_type(type_hint)
    if target is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if target is int:
        return int(value)
    if target is float:
        return float(value)
    if target is str:
        return str(value)
    return value


def _normalize_type(type_hint: Any) -> type | None:
    if isinstance(type_hint, type):
        return type_hint
    name = str(type_hint).strip().split("|")[0].strip().split("[")[0].lower()
    return {
        "int": int,
        "float": float,
        "bool": bool,
        "str": str,
    }.get(name)
