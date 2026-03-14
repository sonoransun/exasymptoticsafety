"""Dockerfile generation for cloud deployment."""

from __future__ import annotations


def generate_dockerfile(
    gpu: bool = False,
    ray: bool = False,
    extras: list[str] | None = None,
) -> str:
    """Generate a Dockerfile for asymptoticsafety deployment.

    Args:
        gpu: Include NVIDIA CUDA base image and JAX GPU support.
        ray: Include Ray for distributed computing.
        extras: Additional pip extras to install.
    """
    if gpu:
        base = "nvidia/cuda:12.6.0-runtime-ubuntu24.04"
        python_install = (
            "RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && "
            "rm -rf /var/lib/apt/lists/*"
        )
        jax_extra = '"jax[cuda12]>=0.4"'
    else:
        base = "python:3.14-slim"
        python_install = ""
        jax_extra = ""

    pip_extras = list(extras or [])
    if gpu:
        pip_extras.append("gpu")
    if ray:
        pip_extras.append("distributed")

    extras_str = ",".join(pip_extras)
    install_cmd = f"pip install asymsafety[{extras_str}]" if extras_str else "pip install asymsafety"
    if jax_extra:
        install_cmd += f" {jax_extra}"

    lines = [
        f"FROM {base}",
        "",
    ]
    if python_install:
        lines.append(python_install)
        lines.append("")

    lines.extend([
        "WORKDIR /app",
        "",
        "COPY . /app",
        f"RUN {install_cmd}",
        "",
        'ENTRYPOINT ["python", "-m"]',
        "",
    ])

    return "\n".join(lines)


def generate_compose(n_workers: int = 4, gpu: bool = False) -> str:
    """Generate a docker-compose.yml for local multi-container testing."""
    gpu_section = ""
    if gpu:
        gpu_section = """
      deploy:
        resources:
          reservations:
            devices:
              - capabilities: [gpu]"""

    workers = "\n".join(
        f"""
  worker-{i}:
    build: .
    command: asymsafety.compute.distributed.task_server
    depends_on:
      - head{gpu_section}"""
        for i in range(n_workers)
    )

    return f"""version: "3.8"
services:
  head:
    build: .
    ports:
      - "8765:8765"
    command: python -m asymsafety.compute.distributed.task_server --port 8765{gpu_section}
{workers}
"""
