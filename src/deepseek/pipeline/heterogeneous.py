"""
Heterogeneous Ray Scheduling Utilities

This module provides resource auto-detection, custom resource registration,
and scheduling utilities for heterogeneous clusters (Apple Silicon, NVIDIA GPUs, etc.)

Features:
- Resource auto-detection for each node
- Custom resource type registration with Ray
- Placement group creation for pipeline parallelism
- Node health monitoring for heterogeneous clusters
"""

import logging
import os
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import ray

LOGGER = logging.getLogger(__name__)


class NodeArchitecture(Enum):
    """Node architecture type."""
    UNKNOWN = "unknown"
    APPLE_SILICON = "apple_silicon"
    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"
    CPU_ONLY = "cpu_only"


@dataclass
class DetectedResources:
    """Auto-detected resources on a node."""
    # Node identification
    node_id: str = ""
    hostname: str = ""
    architecture: NodeArchitecture = NodeArchitecture.UNKNOWN
    
    # Core resources
    cpu_cores: int = 0
    memory_gb: float = 0.0
    
    # GPU resources
    has_metal: bool = False
    has_cuda: bool = False
    cuda_device_count: int = 0
    cuda_compute_cap: float = 0.0
    gpu_memory_gb: float = 0.0
    
    # Network
    network_bandwidth_gbps: float = 0.0
    
    # Labels
    labels: list[str] = field(default_factory=list)
    
    def to_ray_resources(self) -> dict[str, float]:
        """Convert to Ray custom resource specification."""
        resources: dict[str, float] = {
            "cpu_cores": float(self.cpu_cores),
            "memory_gb": self.memory_gb,
        }
        
        if self.has_metal:
            resources["metal"] = 1.0
            # Apple Silicon uses unified memory
            resources["gpu_memory_gb"] = self.memory_gb * 0.75
        
        if self.has_cuda:
            resources["cuda"] = float(self.cuda_device_count)
            resources["cuda_compute_cap"] = self.cuda_compute_cap
            resources["gpu_memory_gb"] = self.gpu_memory_gb
            
            # Add specific GPU type resource
            if self.cuda_compute_cap >= 9.0:
                resources["H100"] = float(self.cuda_device_count)
                resources["has_hbm"] = 1.0
            elif self.cuda_compute_cap >= 8.0:
                resources["A100"] = float(self.cuda_device_count)
                resources["has_hbm"] = 1.0
        
        return resources


def detect_cpu_cores() -> int:
    """Detect number of CPU cores."""
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except Exception:
        return os.cpu_count() or 1


def detect_memory_gb() -> float:
    """Detect system memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # Fallback: try to read from /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024**2)
        except Exception:
            pass
        return 16.0  # Default fallback


def detect_metal_support() -> bool:
    """Detect Apple Metal GPU support (macOS only)."""
    if platform.system() != "Darwin":
        return False
    
    try:
        # Check if we're on Apple Silicon
        machine = platform.machine()
        if machine.startswith("arm"):
            return True
        
        # For Intel Macs, check for discrete GPU
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "Metal" in result.stdout
    except Exception:
        return False


def detect_cuda_gpus() -> tuple[int, float, float]:
    """Detect NVIDIA CUDA GPUs.
    
    Returns:
        Tuple of (device_count, compute_capability, total_memory_gb)
    """
    # Check for CUDA environment
    if not (os.path.exists("/dev/nvidia0") or os.environ.get("CUDA_VISIBLE_DEVICES")):
        return (0, 0.0, 0.0)
    
    try:
        # Try nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count,compute_cap,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split(",")
                count = len(lines)
                compute_cap = float(parts[1].strip()) if len(parts) > 1 else 8.0
                memory_mb = float(parts[2].strip()) if len(parts) > 2 else 40000
                return (count, compute_cap, memory_mb / 1024)
    except Exception:
        pass
    
    # Fallback: try PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                props = torch.cuda.get_device_properties(0)
                compute_cap = float(f"{props.major}.{props.minor}")
                memory_gb = props.total_memory / (1024**3)
                return (count, compute_cap, memory_gb)
    except Exception:
        pass
    
    return (0, 0.0, 0.0)


def detect_resources() -> DetectedResources:
    """Auto-detect all resources on the current node."""
    import socket
    import uuid
    
    hostname = socket.gethostname()
    node_id = f"{hostname}-{str(uuid.uuid4())[:8]}"
    
    cpu_cores = detect_cpu_cores()
    memory_gb = detect_memory_gb()
    has_metal = detect_metal_support()
    cuda_count, cuda_compute_cap, gpu_memory_gb = detect_cuda_gpus()
    has_cuda = cuda_count > 0
    
    # Determine architecture
    if has_metal:
        architecture = NodeArchitecture.APPLE_SILICON
        labels = ["apple_silicon"]
        if memory_gb >= 96:
            labels.append("mac_studio_ultra")
        elif memory_gb >= 36:
            labels.append("mac_studio")
        elif memory_gb >= 16:
            labels.append("macbook_pro")
    elif has_cuda:
        architecture = NodeArchitecture.NVIDIA_GPU
        labels = ["nvidia_gpu"]
        if cuda_compute_cap >= 9.0:
            labels.append("h100")
        elif cuda_compute_cap >= 8.0:
            labels.append("a100")
        elif cuda_compute_cap >= 7.5:
            labels.append("rtx")
    else:
        architecture = NodeArchitecture.CPU_ONLY
        labels = ["cpu_only"]
    
    return DetectedResources(
        node_id=node_id,
        hostname=hostname,
        architecture=architecture,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        has_metal=has_metal,
        has_cuda=has_cuda,
        cuda_device_count=cuda_count,
        cuda_compute_cap=cuda_compute_cap,
        gpu_memory_gb=gpu_memory_gb,
        labels=labels,
    )


def register_custom_resources_with_ray(resources: DetectedResources | None = None) -> dict[str, float]:
    """Register custom resources with Ray.
    
    Call this before ray.init() to register detected resources.
    
    Args:
        resources: Pre-detected resources, or None to auto-detect
        
    Returns:
        Dictionary of registered resources
    """
    if resources is None:
        resources = detect_resources()
    
    return resources.to_ray_resources()


def init_ray_with_resources(
    address: str | None = None,
    resources: DetectedResources | None = None,
    **kwargs: Any,
) -> None:
    """Initialize Ray with auto-detected custom resources.
    
    This is a wrapper around ray.init() that:
    1. Auto-detects hardware resources
    2. Registers custom resource types (metal, cuda, memory_gb, etc.)
    3. Initializes Ray with these resources
    
    Args:
        address: Ray cluster address (None for local)
        resources: Pre-detected resources, or None to auto-detect
        **kwargs: Additional arguments passed to ray.init()
    """
    if ray.is_initialized():
        LOGGER.warning("Ray already initialized, skipping resource registration")
        return
    
    custom_resources = register_custom_resources_with_ray(resources)
    
    LOGGER.info("Detected resources: %s", custom_resources)
    
    # Build ray.init arguments
    init_kwargs: dict[str, Any] = {
        "ignore_reinit_error": True,
        "resources": custom_resources,
        **kwargs,
    }
    
    if address:
        init_kwargs["address"] = address
    
    ray.init(**init_kwargs)
    LOGGER.info("Ray initialized with custom resources")


@dataclass
class PlacementBundle:
    """Resource bundle for a placement group slot."""
    resources: dict[str, float]
    name: str = ""


@dataclass 
class PlacementGroupConfig:
    """Configuration for a Ray placement group."""
    name: str
    bundles: list[PlacementBundle]
    strategy: str = "SPREAD"  # SPREAD, PACK, STRICT_SPREAD, STRICT_PACK


def create_pipeline_parallel_placement_group(
    name: str,
    pp_size: int,
    backend_resources: dict[str, float],
    strategy: str = "SPREAD",
) -> "ray.util.placement_group.PlacementGroup":
    """Create a placement group for pipeline parallelism.
    
    Args:
        name: Name for the placement group
        pp_size: Pipeline parallel size (number of stages)
        backend_resources: Resource requirements per stage
        strategy: Placement strategy (SPREAD, PACK, etc.)
        
    Returns:
        Ray PlacementGroup object
    """
    bundles = [backend_resources.copy() for _ in range(pp_size)]
    
    pg = ray.util.placement_group(
        bundles=bundles,
        strategy=strategy,
        name=name,
    )
    
    LOGGER.info(
        "Created placement group '%s' with %d bundles, strategy=%s",
        name, pp_size, strategy
    )
    
    return pg


def create_heterogeneous_placement_group(
    name: str,
    stage_resources: list[dict[str, float]],
    strategy: str = "STRICT_SPREAD",
) -> "ray.util.placement_group.PlacementGroup":
    """Create a placement group for heterogeneous pipeline parallelism.
    
    Allows different resource requirements for each pipeline stage.
    For example: Stage 0 on Apple Silicon, Stage 1 on H100.
    
    Args:
        name: Name for the placement group
        stage_resources: List of resource requirements per stage
        strategy: Placement strategy
        
    Returns:
        Ray PlacementGroup object
    """
    pg = ray.util.placement_group(
        bundles=stage_resources,
        strategy=strategy,
        name=name,
    )
    
    LOGGER.info(
        "Created heterogeneous placement group '%s' with %d stages",
        name, len(stage_resources)
    )
    
    return pg


@dataclass
class NodeHealth:
    """Health status for a node."""
    node_id: str
    is_healthy: bool
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float | None = None
    gpu_memory_utilization: float | None = None
    last_heartbeat: float = 0.0
    error_message: str | None = None


class ClusterHealthMonitor:
    """Monitor health of nodes in a heterogeneous cluster.
    
    Tracks node health via heartbeats and resource utilization.
    Can trigger failover when nodes become unhealthy.
    """
    
    def __init__(
        self,
        check_interval_secs: float = 30.0,
        failure_threshold: int = 3,
        on_node_failure: Callable[[str], None] | None = None,
    ):
        """Initialize health monitor.
        
        Args:
            check_interval_secs: Interval between health checks
            failure_threshold: Consecutive failures before marking unhealthy
            on_node_failure: Callback when a node fails
        """
        self.check_interval_secs = check_interval_secs
        self.failure_threshold = failure_threshold
        self.on_node_failure = on_node_failure
        
        self._node_health: dict[str, NodeHealth] = {}
        self._failure_counts: dict[str, int] = {}
        self._running = False
    
    def register_node(self, node_id: str) -> None:
        """Register a node for health monitoring."""
        self._node_health[node_id] = NodeHealth(
            node_id=node_id,
            is_healthy=True,
            last_heartbeat=time.time(),
        )
        self._failure_counts[node_id] = 0
        LOGGER.info("Registered node %s for health monitoring", node_id)
    
    def update_health(self, health: NodeHealth) -> None:
        """Update health status for a node."""
        node_id = health.node_id
        
        if health.is_healthy:
            self._failure_counts[node_id] = 0
        else:
            self._failure_counts[node_id] = self._failure_counts.get(node_id, 0) + 1
            
            if self._failure_counts[node_id] >= self.failure_threshold:
                health.is_healthy = False
                LOGGER.warning(
                    "Node %s marked unhealthy after %d consecutive failures",
                    node_id, self.failure_threshold
                )
                if self.on_node_failure:
                    self.on_node_failure(node_id)
        
        health.last_heartbeat = time.time()
        self._node_health[node_id] = health
    
    def get_healthy_nodes(self) -> list[str]:
        """Get list of healthy node IDs."""
        return [
            node_id for node_id, health in self._node_health.items()
            if health.is_healthy
        ]
    
    def get_node_health(self, node_id: str) -> NodeHealth | None:
        """Get health status for a specific node."""
        return self._node_health.get(node_id)
    
    def get_cluster_summary(self) -> dict[str, Any]:
        """Get summary of cluster health."""
        total = len(self._node_health)
        healthy = sum(1 for h in self._node_health.values() if h.is_healthy)
        
        return {
            "total_nodes": total,
            "healthy_nodes": healthy,
            "unhealthy_nodes": total - healthy,
            "nodes": {
                node_id: {
                    "is_healthy": h.is_healthy,
                    "cpu_utilization": h.cpu_utilization,
                    "memory_utilization": h.memory_utilization,
                    "gpu_utilization": h.gpu_utilization,
                    "last_heartbeat": h.last_heartbeat,
                }
                for node_id, h in self._node_health.items()
            },
        }


def get_ray_cluster_resources() -> dict[str, float]:
    """Get total available resources across the Ray cluster."""
    if not ray.is_initialized():
        return {}
    
    return dict(ray.cluster_resources())


def get_ray_available_resources() -> dict[str, float]:
    """Get currently available (unused) resources in the Ray cluster."""
    if not ray.is_initialized():
        return {}
    
    return dict(ray.available_resources())


__all__ = [
    # Enums
    "NodeArchitecture",
    # Data classes
    "DetectedResources",
    "PlacementBundle",
    "PlacementGroupConfig",
    "NodeHealth",
    # Detection functions
    "detect_resources",
    "detect_cpu_cores",
    "detect_memory_gb",
    "detect_metal_support",
    "detect_cuda_gpus",
    # Ray integration
    "register_custom_resources_with_ray",
    "init_ray_with_resources",
    "create_pipeline_parallel_placement_group",
    "create_heterogeneous_placement_group",
    "get_ray_cluster_resources",
    "get_ray_available_resources",
    # Health monitoring
    "ClusterHealthMonitor",
]
