"""GPU and distributed computing acceleration for asymsafety.

All acceleration is optional — the core library works without
any additional dependencies. Install extras for acceleration:

    pip install asymsafety[gpu]          # JAX for GPU acceleration
    pip install asymsafety[distributed]  # Ray for distributed computing
    pip install asymsafety[dask]         # Dask as alternative distributed backend
"""

from asymsafety.compute.config import get_backend, available_backends, ComputeConfig
