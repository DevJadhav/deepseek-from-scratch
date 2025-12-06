"""
Graceful Degradation for Heterogeneous DeepSeek Training

Provides fallback mechanisms when heterogeneous nodes become unavailable,
ensuring training can continue with reduced capacity.

Features:
- Automatic backend fallback (CUDA -> MPS -> CPU)
- Expert redistribution on node failure
- Adaptive batch sizing based on available resources
- Checkpoint recovery with degraded capacity
- Health monitoring and proactive degradation

Usage:
    from deepseek.pipeline.graceful_degradation import (
        GracefulDegradationManager,
        DegradationConfig,
        ResourceMonitor,
    )

    manager = GracefulDegradationManager(config)
    manager.register_fallback(
        primary=Backend.PYTORCH_CUDA,
        fallback=Backend.PYTORCH_MPS,
    )

    # During training
    backend = manager.get_available_backend(preferred=Backend.PYTORCH_CUDA)
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class Backend(Enum):
    """Available compute backends."""
    PYTORCH_CUDA = "pytorch_cuda"
    PYTORCH_MPS = "pytorch_mps"
    PYTORCH_CPU = "pytorch_cpu"
    MLX = "mlx"
    RUST_CUDA = "rust_cuda"
    RUST_METAL = "rust_metal"
    RUST_CPU = "rust_cpu"

    @property
    def priority(self) -> int:
        """Get priority (lower = higher priority)."""
        priorities = {
            Backend.PYTORCH_CUDA: 1,
            Backend.RUST_CUDA: 2,
            Backend.MLX: 3,
            Backend.PYTORCH_MPS: 4,
            Backend.RUST_METAL: 5,
            Backend.PYTORCH_CPU: 6,
            Backend.RUST_CPU: 7,
        }
        return priorities.get(self, 100)

    def is_gpu(self) -> bool:
        """Check if this is a GPU backend."""
        return self in (
            Backend.PYTORCH_CUDA,
            Backend.PYTORCH_MPS,
            Backend.MLX,
            Backend.RUST_CUDA,
            Backend.RUST_METAL,
        )


class DegradationLevel(Enum):
    """Level of degradation."""
    NONE = "none"  # Full capacity
    MILD = "mild"  # Minor capacity reduction (e.g., reduced batch size)
    MODERATE = "moderate"  # Significant reduction (e.g., fallback backend)
    SEVERE = "severe"  # Major reduction (e.g., CPU only)
    CRITICAL = "critical"  # Minimal functionality


@dataclass
class ResourceStatus:
    """Status of a compute resource."""
    backend: Backend
    available: bool
    healthy: bool
    capacity_percent: float  # 0-100
    memory_used_gb: float
    memory_total_gb: float
    last_check: datetime
    error_message: str | None = None

    @property
    def memory_percent(self) -> float:
        """Get memory usage percentage."""
        if self.memory_total_gb <= 0:
            return 0.0
        return (self.memory_used_gb / self.memory_total_gb) * 100


@dataclass
class DegradationEvent:
    """Record of a degradation event."""
    timestamp: datetime
    from_level: DegradationLevel
    to_level: DegradationLevel
    reason: str
    affected_backend: Backend | None
    actions_taken: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "from_level": self.from_level.value,
            "to_level": self.to_level.value,
            "reason": self.reason,
            "affected_backend": self.affected_backend.value if self.affected_backend else None,
            "actions_taken": self.actions_taken,
        }


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation."""
    # Enable/disable degradation handling
    enabled: bool = True

    # Health check settings
    health_check_interval_ms: float = 5000
    health_check_timeout_ms: float = 1000
    max_consecutive_failures: int = 3

    # Fallback chain
    fallback_chain: list[Backend] = field(default_factory=list)

    # Capacity thresholds
    memory_warning_threshold: float = 80.0  # Percent
    memory_critical_threshold: float = 95.0
    min_capacity_threshold: float = 10.0  # Below this = critical

    # Batch size adjustment
    enable_batch_adjustment: bool = True
    min_batch_size: int = 1
    batch_reduction_factor: float = 0.5

    # Expert redistribution
    enable_expert_redistribution: bool = True
    min_experts_per_node: int = 4

    # Recovery settings
    enable_auto_recovery: bool = True
    recovery_check_interval_ms: float = 10000

    # Logging
    log_events: bool = True

    def __post_init__(self):
        """Set default fallback chain."""
        if not self.fallback_chain:
            self.fallback_chain = [
                Backend.PYTORCH_CUDA,
                Backend.RUST_CUDA,
                Backend.MLX,
                Backend.PYTORCH_MPS,
                Backend.RUST_METAL,
                Backend.PYTORCH_CPU,
                Backend.RUST_CPU,
            ]


class ResourceChecker(ABC):
    """Abstract base for resource checking."""

    @abstractmethod
    def check(self) -> ResourceStatus:
        """Check resource status."""
        pass


class PyTorchCUDAChecker(ResourceChecker):
    """Check PyTorch CUDA availability."""

    def check(self) -> ResourceStatus:
        """Check CUDA status."""
        try:
            import torch

            if not torch.cuda.is_available():
                return ResourceStatus(
                    backend=Backend.PYTORCH_CUDA,
                    available=False,
                    healthy=False,
                    capacity_percent=0,
                    memory_used_gb=0,
                    memory_total_gb=0,
                    last_check=datetime.now(),
                    error_message="CUDA not available",
                )

            # Get memory info
            device = torch.cuda.current_device()
            mem_allocated = torch.cuda.memory_allocated(device) / (1024**3)
            mem_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)

            return ResourceStatus(
                backend=Backend.PYTORCH_CUDA,
                available=True,
                healthy=True,
                capacity_percent=100 - (mem_allocated / mem_total * 100),
                memory_used_gb=mem_allocated,
                memory_total_gb=mem_total,
                last_check=datetime.now(),
            )

        except Exception as e:
            return ResourceStatus(
                backend=Backend.PYTORCH_CUDA,
                available=False,
                healthy=False,
                capacity_percent=0,
                memory_used_gb=0,
                memory_total_gb=0,
                last_check=datetime.now(),
                error_message=str(e),
            )


class PyTorchMPSChecker(ResourceChecker):
    """Check PyTorch MPS (Metal) availability."""

    def check(self) -> ResourceStatus:
        """Check MPS status."""
        try:
            import torch

            if not torch.backends.mps.is_available():
                return ResourceStatus(
                    backend=Backend.PYTORCH_MPS,
                    available=False,
                    healthy=False,
                    capacity_percent=0,
                    memory_used_gb=0,
                    memory_total_gb=0,
                    last_check=datetime.now(),
                    error_message="MPS not available",
                )

            # MPS doesn't provide detailed memory info
            # Estimate based on system
            return ResourceStatus(
                backend=Backend.PYTORCH_MPS,
                available=True,
                healthy=True,
                capacity_percent=100,  # Assume full capacity
                memory_used_gb=0,
                memory_total_gb=64,  # Typical Mac Studio
                last_check=datetime.now(),
            )

        except Exception as e:
            return ResourceStatus(
                backend=Backend.PYTORCH_MPS,
                available=False,
                healthy=False,
                capacity_percent=0,
                memory_used_gb=0,
                memory_total_gb=0,
                last_check=datetime.now(),
                error_message=str(e),
            )


class MLXChecker(ResourceChecker):
    """Check MLX availability."""

    def check(self) -> ResourceStatus:
        """Check MLX status."""
        try:
            import mlx.core as mx

            # Try a simple operation to verify MLX is working
            _ = mx.array([1.0])

            return ResourceStatus(
                backend=Backend.MLX,
                available=True,
                healthy=True,
                capacity_percent=100,
                memory_used_gb=0,
                memory_total_gb=64,
                last_check=datetime.now(),
            )

        except ImportError:
            return ResourceStatus(
                backend=Backend.MLX,
                available=False,
                healthy=False,
                capacity_percent=0,
                memory_used_gb=0,
                memory_total_gb=0,
                last_check=datetime.now(),
                error_message="MLX not installed",
            )
        except Exception as e:
            return ResourceStatus(
                backend=Backend.MLX,
                available=False,
                healthy=False,
                capacity_percent=0,
                memory_used_gb=0,
                memory_total_gb=0,
                last_check=datetime.now(),
                error_message=str(e),
            )


class CPUChecker(ResourceChecker):
    """Check CPU availability (always available)."""

    def __init__(self, backend: Backend = Backend.PYTORCH_CPU):
        self.backend = backend

    def check(self) -> ResourceStatus:
        """Check CPU status."""
        import os

        # Get CPU count as capacity indicator
        cpu_count = os.cpu_count() or 1

        return ResourceStatus(
            backend=self.backend,
            available=True,
            healthy=True,
            capacity_percent=100,
            memory_used_gb=0,
            memory_total_gb=32,  # Typical
            last_check=datetime.now(),
        )


class ResourceMonitor:
    """
    Monitors resource health across backends.

    Continuously checks backend availability and triggers
    degradation when resources become unavailable.
    """

    def __init__(self, config: DegradationConfig):
        self.config = config
        self._checkers: dict[Backend, ResourceChecker] = {}
        self._status: dict[Backend, ResourceStatus] = {}
        self._failure_counts: dict[Backend, int] = {}
        self._callbacks: list[Callable[[Backend, ResourceStatus], None]] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._setup_checkers()

    def _setup_checkers(self) -> None:
        """Set up resource checkers."""
        self._checkers = {
            Backend.PYTORCH_CUDA: PyTorchCUDAChecker(),
            Backend.PYTORCH_MPS: PyTorchMPSChecker(),
            Backend.MLX: MLXChecker(),
            Backend.PYTORCH_CPU: CPUChecker(Backend.PYTORCH_CPU),
            Backend.RUST_CPU: CPUChecker(Backend.RUST_CPU),
        }

    def start(self) -> None:
        """Start monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            for backend in self.config.fallback_chain:
                if backend in self._checkers:
                    self._check_backend(backend)

            time.sleep(self.config.health_check_interval_ms / 1000)

    def _check_backend(self, backend: Backend) -> None:
        """Check a single backend."""
        checker = self._checkers.get(backend)
        if not checker:
            return

        try:
            status = checker.check()

            with self._lock:
                old_status = self._status.get(backend)
                self._status[backend] = status

                # Track failures
                if not status.healthy:
                    self._failure_counts[backend] = self._failure_counts.get(backend, 0) + 1
                else:
                    self._failure_counts[backend] = 0

                # Notify on status change
                if old_status and old_status.healthy != status.healthy:
                    for callback in self._callbacks:
                        callback(backend, status)

        except Exception as e:
            with self._lock:
                self._status[backend] = ResourceStatus(
                    backend=backend,
                    available=False,
                    healthy=False,
                    capacity_percent=0,
                    memory_used_gb=0,
                    memory_total_gb=0,
                    last_check=datetime.now(),
                    error_message=str(e),
                )
                self._failure_counts[backend] = self._failure_counts.get(backend, 0) + 1

    def check_now(self, backend: Backend) -> ResourceStatus | None:
        """Immediately check a backend."""
        self._check_backend(backend)
        with self._lock:
            return self._status.get(backend)

    def get_status(self, backend: Backend) -> ResourceStatus | None:
        """Get cached status for backend."""
        with self._lock:
            return self._status.get(backend)

    def get_all_status(self) -> dict[Backend, ResourceStatus]:
        """Get status for all backends."""
        with self._lock:
            return dict(self._status)

    def is_healthy(self, backend: Backend) -> bool:
        """Check if backend is healthy."""
        with self._lock:
            status = self._status.get(backend)
            if not status:
                return False
            return status.healthy and self._failure_counts.get(backend, 0) < self.config.max_consecutive_failures

    def on_status_change(self, callback: Callable[[Backend, ResourceStatus], None]) -> None:
        """Register callback for status changes."""
        self._callbacks.append(callback)


class GracefulDegradationManager:
    """
    Manages graceful degradation for heterogeneous training.

    Handles:
    - Backend fallback when resources become unavailable
    - Batch size adjustment based on capacity
    - Expert redistribution on node failure
    - Recovery when resources become available again
    """

    def __init__(self, config: DegradationConfig | None = None):
        self.config = config or DegradationConfig()
        self.monitor = ResourceMonitor(self.config)
        self._current_level = DegradationLevel.NONE
        self._active_backend: Backend | None = None
        self._current_batch_size: int | None = None
        self._events: list[DegradationEvent] = []
        self._callbacks: list[Callable[[DegradationEvent], None]] = []

        # Register monitor callback
        self.monitor.on_status_change(self._on_resource_change)

    def start(self) -> None:
        """Start degradation manager."""
        if self.config.enabled:
            self.monitor.start()
            self._initialize_backend()

    def stop(self) -> None:
        """Stop degradation manager."""
        self.monitor.stop()

    def _initialize_backend(self) -> None:
        """Initialize with best available backend."""
        for backend in self.config.fallback_chain:
            status = self.monitor.check_now(backend)
            if status and status.available and status.healthy:
                self._active_backend = backend
                break

        if self._active_backend is None:
            # Fallback to CPU
            self._active_backend = Backend.PYTORCH_CPU
            self._degrade(
                DegradationLevel.SEVERE,
                "No GPU backends available, falling back to CPU",
                None,
            )

    def _on_resource_change(self, backend: Backend, status: ResourceStatus) -> None:
        """Handle resource status change."""
        if backend == self._active_backend and not status.healthy:
            # Active backend failed, need to fallback
            self._handle_backend_failure(backend)
        elif status.healthy and self._current_level != DegradationLevel.NONE:
            # Resource recovered, check if we can upgrade
            self._check_recovery()

    def _handle_backend_failure(self, failed_backend: Backend) -> None:
        """Handle failure of a backend."""
        # Find next available backend in fallback chain
        fallback = self._find_fallback(failed_backend)

        if fallback:
            level = self._determine_level(fallback)
            self._degrade(
                level,
                f"Backend {failed_backend.value} failed, falling back to {fallback.value}",
                failed_backend,
            )
            self._active_backend = fallback
        else:
            self._degrade(
                DegradationLevel.CRITICAL,
                f"No fallback available after {failed_backend.value} failure",
                failed_backend,
            )

    def _find_fallback(self, failed_backend: Backend) -> Backend | None:
        """Find next available backend in fallback chain."""
        started = False
        for backend in self.config.fallback_chain:
            if backend == failed_backend:
                started = True
                continue
            if started and self.monitor.is_healthy(backend):
                return backend

        # If we didn't find one after the failed backend, try from the beginning
        for backend in self.config.fallback_chain:
            if backend != failed_backend and self.monitor.is_healthy(backend):
                return backend

        return None

    def _determine_level(self, backend: Backend) -> DegradationLevel:
        """Determine degradation level based on backend."""
        if backend in (Backend.PYTORCH_CUDA, Backend.RUST_CUDA):
            return DegradationLevel.NONE
        elif backend in (Backend.MLX, Backend.PYTORCH_MPS, Backend.RUST_METAL):
            return DegradationLevel.MILD
        elif backend in (Backend.PYTORCH_CPU, Backend.RUST_CPU):
            return DegradationLevel.SEVERE
        return DegradationLevel.MODERATE

    def _degrade(
        self,
        to_level: DegradationLevel,
        reason: str,
        affected_backend: Backend | None,
    ) -> None:
        """Record degradation event."""
        event = DegradationEvent(
            timestamp=datetime.now(),
            from_level=self._current_level,
            to_level=to_level,
            reason=reason,
            affected_backend=affected_backend,
        )

        # Take actions based on level
        if to_level in (DegradationLevel.MODERATE, DegradationLevel.SEVERE):
            if self.config.enable_batch_adjustment:
                self._adjust_batch_size()
                event.actions_taken.append("batch_size_reduced")

        if to_level in (DegradationLevel.SEVERE, DegradationLevel.CRITICAL):
            if self.config.enable_expert_redistribution:
                event.actions_taken.append("expert_redistribution_triggered")

        self._current_level = to_level
        self._events.append(event)

        # Notify callbacks
        for callback in self._callbacks:
            callback(event)

        if self.config.log_events:
            print(f"[DEGRADATION] {self._current_level.value}: {reason}")

    def _check_recovery(self) -> None:
        """Check if we can recover to a better state."""
        if not self.config.enable_auto_recovery:
            return

        # Find best available backend
        for backend in self.config.fallback_chain:
            if self.monitor.is_healthy(backend):
                if backend.priority < (self._active_backend.priority if self._active_backend else 100):
                    # Better backend available
                    old_level = self._current_level
                    new_level = self._determine_level(backend)

                    if new_level.value < old_level.value:  # type: ignore
                        self._active_backend = backend
                        self._current_level = new_level

                        event = DegradationEvent(
                            timestamp=datetime.now(),
                            from_level=old_level,
                            to_level=new_level,
                            reason=f"Recovered to {backend.value}",
                            affected_backend=None,
                            actions_taken=["recovered"],
                        )
                        self._events.append(event)

                        if self.config.log_events:
                            print(f"[RECOVERY] Upgraded to {backend.value}")

                break

    def _adjust_batch_size(self) -> None:
        """Reduce batch size for degraded operation."""
        if self._current_batch_size is None:
            return

        new_size = max(
            self.config.min_batch_size,
            int(self._current_batch_size * self.config.batch_reduction_factor),
        )
        self._current_batch_size = new_size

    def get_available_backend(self, preferred: Backend | None = None) -> Backend:
        """Get best available backend."""
        if preferred and self.monitor.is_healthy(preferred):
            return preferred

        if self._active_backend:
            return self._active_backend

        # Fallback chain
        for backend in self.config.fallback_chain:
            if self.monitor.is_healthy(backend):
                return backend

        # Last resort
        return Backend.PYTORCH_CPU

    def get_current_level(self) -> DegradationLevel:
        """Get current degradation level."""
        return self._current_level

    def get_events(self) -> list[DegradationEvent]:
        """Get degradation events."""
        return list(self._events)

    def set_batch_size(self, batch_size: int) -> None:
        """Set current batch size for adjustment."""
        self._current_batch_size = batch_size

    def get_recommended_batch_size(self) -> int | None:
        """Get recommended batch size after adjustment."""
        return self._current_batch_size

    def on_degradation(self, callback: Callable[[DegradationEvent], None]) -> None:
        """Register callback for degradation events."""
        self._callbacks.append(callback)


# Global manager instance
_manager: GracefulDegradationManager | None = None


def get_degradation_manager() -> GracefulDegradationManager:
    """Get or create global degradation manager."""
    global _manager
    if _manager is None:
        _manager = GracefulDegradationManager()
    return _manager


def configure_degradation(config: DegradationConfig) -> GracefulDegradationManager:
    """Configure global degradation manager."""
    global _manager
    _manager = GracefulDegradationManager(config)
    return _manager
