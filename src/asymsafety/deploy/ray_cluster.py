"""Ray cluster configuration templates for cloud deployment."""

from __future__ import annotations


def generate_ray_cluster_yaml(
    provider: str = "aws",
    head_instance: str = "m5.xlarge",
    worker_instance: str = "m5.xlarge",
    gpu_worker_instance: str = "g4dn.xlarge",
    min_workers: int = 0,
    max_workers: int = 10,
    region: str = "us-east-1",
) -> str:
    """Generate a Ray cluster YAML config for `ray up`.

    Args:
        provider: Cloud provider ("aws", "gcp", "azure").
        head_instance: Instance type for the head node.
        worker_instance: Instance type for CPU workers.
        gpu_worker_instance: Instance type for GPU workers.
        min_workers: Minimum number of worker nodes.
        max_workers: Maximum number of worker nodes (autoscaling).
        region: Cloud region.
    """
    return f"""# Ray cluster configuration for Asymptotic Safety Explorer
# Launch with: ray up cluster.yaml
# Connect with: ray.init(address="ray://<head-ip>:10001")

cluster_name: asymsafety-cluster

provider:
    type: {provider}
    region: {region}

auth:
    ssh_user: ubuntu

head_node_type:
    node_config:
        InstanceType: {head_instance}

worker_node_types:
    - name: cpu-workers
      node_config:
          InstanceType: {worker_instance}
      min_workers: {min_workers}
      max_workers: {max_workers}

    - name: gpu-workers
      node_config:
          InstanceType: {gpu_worker_instance}
      min_workers: 0
      max_workers: {max_workers // 2}

setup_commands:
    - pip install asymsafety[gpu,distributed]

head_setup_commands:
    - pip install asymsafety[gpu,distributed]
"""


def generate_kuberay_spec(
    replicas: int = 4,
    gpu_per_worker: int = 0,
    image: str = "asymsafety:latest",
) -> str:
    """Generate a KubeRay RayCluster spec for Kubernetes deployment."""
    gpu_resources = ""
    if gpu_per_worker:
        gpu_resources = f"""
              limits:
                nvidia.com/gpu: {gpu_per_worker}"""

    return f"""apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: asymsafety-cluster
spec:
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
    template:
      spec:
        containers:
          - name: ray-head
            image: {image}
            ports:
              - containerPort: 6379
              - containerPort: 10001
              - containerPort: 8265

  workerGroupSpecs:
    - replicas: {replicas}
      groupName: workers
      rayStartParams: {{}}
      template:
        spec:
          containers:
            - name: ray-worker
              image: {image}
              resources:{gpu_resources}
"""
