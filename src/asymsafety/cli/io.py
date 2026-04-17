"""Output writer dispatch by file extension.

Supported formats:
    .npz  → numpy.savez_compressed (always available)
    .h5   → h5py (optional dep — error message if missing)
    .json → JSON (always available)

The dispatcher chooses by the output filename's suffix; an unknown
suffix falls back to ``.npz``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_results(
    output_path: str | Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a result bundle to ``output_path``.

    Args:
        output_path: Where to save. Suffix decides the format.
        arrays: Dict of array data (e.g. ``"betas"``, ``"grid"``).
        metadata: Scalar/string fields embedded as attrs (HDF5) or alongside
            the arrays (NPZ/JSON).

    Returns:
        The :class:`Path` actually written.
    """
    path = Path(output_path)
    suffix = path.suffix.lower()
    metadata = metadata or {}

    if suffix == ".h5":
        return _write_hdf5(path, arrays, metadata)
    if suffix == ".json":
        return _write_json(path, arrays, metadata)
    # default = npz
    return _write_npz(path, arrays, metadata)


# ---------------------------------------------------------------------------
# NPZ
# ---------------------------------------------------------------------------

def _write_npz(path: Path, arrays: dict[str, np.ndarray],
                metadata: dict) -> Path:
    payload: dict[str, np.ndarray] = dict(arrays)
    # Embed metadata as a 0-d JSON-encoded string array
    payload["__metadata__"] = np.array(json.dumps(metadata, default=str))
    np.savez_compressed(path, **payload)
    return path


def read_npz(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Inverse of :func:`_write_npz`. Returns (arrays, metadata)."""
    data = np.load(path, allow_pickle=False)
    arrays = {k: data[k] for k in data.files if k != "__metadata__"}
    meta_raw = data["__metadata__"].item() if "__metadata__" in data.files else "{}"
    metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
    return arrays, metadata


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------

def _write_hdf5(path: Path, arrays: dict[str, np.ndarray],
                 metadata: dict) -> Path:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "HDF5 output requires h5py. Install with: pip install asymsafety[hdf5]"
        ) from exc
    with h5py.File(path, "w") as fh:
        for name, arr in arrays.items():
            fh.create_dataset(name, data=arr, compression="gzip")
        for k, v in metadata.items():
            fh.attrs[k] = _hdf5_safe(v)
    return path


def _hdf5_safe(value: Any) -> Any:
    """Coerce a metadata value into something h5py can store as an attr."""
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), default=str)
    return json.dumps(value, default=str)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def _write_json(path: Path, arrays: dict[str, np.ndarray],
                 metadata: dict) -> Path:
    payload = {
        "metadata": metadata,
        "arrays": {k: arr.tolist() for k, arr in arrays.items()},
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def read_json(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    raw = json.loads(Path(path).read_text())
    arrays = {k: np.asarray(v) for k, v in raw.get("arrays", {}).items()}
    metadata = raw.get("metadata", {})
    return arrays, metadata
