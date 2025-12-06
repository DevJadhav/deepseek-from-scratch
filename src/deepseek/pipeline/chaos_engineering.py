"""
Chaos Engineering for DeepSeek Training Pipeline

Provides controlled failure injection and recovery testing for
heterogeneous distributed training.

Features:
- Random node failure injection
- Network partition simulation
- GPU memory pressure simulation
- Checkpoint corruption testing
- Expert routing failures
- Graceful degradation validation

Usage:
    from deepseek.pipeline.chaos_engineering import ChaosEngine, ChaosConfig

    chaos = ChaosEngine(ChaosConfig(
        failure_rate=0.1,
        enable_network_partition=True,
    ))

    with chaos.inject_failures():
        # Run training step
        pass
"""

from __future__ import annotations

import contextlib
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


class FailureType(Enum):
    """Types of failures that can be injected."""
    NODE_CRASH = "node_crash"
    NETWORK_PARTITION = "network_partition"
    GPU_OOM = "gpu_oom"
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"
    EXPERT_ROUTING_FAILURE = "expert_routing_failure"
    SLOW_NODE = "slow_node"
    DATA_CORRUPTION = "data_corruption"
    TIMEOUT = "timeout"


class NodeType(Enum):
    """Types of nodes in heterogeneous cluster."""
    H100_CUDA = "h100_cuda"
    APPLE_SILICON = "apple_silicon"
    CPU_ONLY = "cpu_only"
    COORDINATOR = "coordinator"


@dataclass
class FailureEvent:
    """Record of an injected failure."""
    failure_type: FailureType
    node_type: NodeType | None
    node_id: str | None
    timestamp: datetime
    duration_ms: float | None
    recovered: bool = False
    recovery_time_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "failure_type": self.failure_type.value,
            "node_type": self.node_type.value if self.node_type else None,
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "recovered": self.recovered,
            "recovery_time_ms": self.recovery_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class ChaosConfig:
    """Configuration for chaos engineering."""
    # Enable/disable chaos injection
    enabled: bool = True

    # Global failure rate (probability of any failure per operation)
    failure_rate: float = 0.01

    # Per-failure type rates
    node_crash_rate: float = 0.005
    network_partition_rate: float = 0.01
    gpu_oom_rate: float = 0.02
    checkpoint_corruption_rate: float = 0.001
    expert_routing_failure_rate: float = 0.01
    slow_node_rate: float = 0.05
    data_corruption_rate: float = 0.001
    timeout_rate: float = 0.02

    # Failure parameters
    slow_node_delay_ms: tuple[float, float] = (100, 5000)  # Min, max delay
    network_partition_duration_ms: tuple[float, float] = (1000, 10000)
    oom_memory_threshold: float = 0.95  # 95% memory usage triggers OOM

    # Node targeting
    target_node_types: list[NodeType] = field(default_factory=list)
    exclude_nodes: list[str] = field(default_factory=list)

    # Recovery settings
    enable_auto_recovery: bool = True
    recovery_timeout_ms: float = 30000
    max_retry_attempts: int = 3

    # Logging and reporting
    log_failures: bool = True
    collect_metrics: bool = True

    def __post_init__(self):
        """Set defaults for target node types."""
        if not self.target_node_types:
            self.target_node_types = list(NodeType)


class FailureInjector(ABC):
    """Base class for failure injectors."""

    @abstractmethod
    def inject(self, node_id: str | None = None) -> FailureEvent:
        """Inject a failure."""
        pass

    @abstractmethod
    def recover(self, event: FailureEvent) -> bool:
        """Attempt recovery from failure."""
        pass


class NodeCrashInjector(FailureInjector):
    """Simulates node crashes."""

    def __init__(self, config: ChaosConfig):
        self.config = config
        self._crashed_nodes: set[str] = set()

    def inject(self, node_id: str | None = None) -> FailureEvent:
        """Inject node crash failure."""
        if node_id:
            self._crashed_nodes.add(node_id)

        return FailureEvent(
            failure_type=FailureType.NODE_CRASH,
            node_type=None,
            node_id=node_id,
            timestamp=datetime.now(),
            duration_ms=None,  # Until recovery
        )

    def recover(self, event: FailureEvent) -> bool:
        """Recover from node crash."""
        if event.node_id and event.node_id in self._crashed_nodes:
            self._crashed_nodes.discard(event.node_id)
            event.recovered = True
            return True
        return False

    def is_node_crashed(self, node_id: str) -> bool:
        """Check if node is crashed."""
        return node_id in self._crashed_nodes


class NetworkPartitionInjector(FailureInjector):
    """Simulates network partitions."""

    def __init__(self, config: ChaosConfig):
        self.config = config
        self._partitions: dict[str, float] = {}  # node_id -> end_time

    def inject(self, node_id: str | None = None) -> FailureEvent:
        """Inject network partition."""
        min_dur, max_dur = self.config.network_partition_duration_ms
        duration_ms = random.uniform(min_dur, max_dur)

        if node_id:
            self._partitions[node_id] = time.time() + duration_ms / 1000

        return FailureEvent(
            failure_type=FailureType.NETWORK_PARTITION,
            node_type=None,
            node_id=node_id,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
        )

    def recover(self, event: FailureEvent) -> bool:
        """Recover from partition."""
        if event.node_id and event.node_id in self._partitions:
            del self._partitions[event.node_id]
            event.recovered = True
            return True
        return False

    def is_partitioned(self, node_id: str) -> bool:
        """Check if node is partitioned."""
        if node_id not in self._partitions:
            return False
        if time.time() > self._partitions[node_id]:
            del self._partitions[node_id]
            return False
        return True


class GPUOOMInjector(FailureInjector):
    """Simulates GPU out-of-memory errors."""

    def __init__(self, config: ChaosConfig):
        self.config = config

    def inject(self, node_id: str | None = None) -> FailureEvent:
        """Inject OOM error."""
        return FailureEvent(
            failure_type=FailureType.GPU_OOM,
            node_type=NodeType.H100_CUDA,
            node_id=node_id,
            timestamp=datetime.now(),
            duration_ms=0,  # Instant failure
            metadata={"memory_threshold": self.config.oom_memory_threshold},
        )

    def recover(self, event: FailureEvent) -> bool:
        """Recover from OOM (clear cache, reduce batch size)."""
        event.recovered = True
        event.metadata["recovery_action"] = "clear_cache_reduce_batch"
        return True


class SlowNodeInjector(FailureInjector):
    """Simulates slow nodes."""

    def __init__(self, config: ChaosConfig):
        self.config = config
        self._delays: dict[str, float] = {}

    def inject(self, node_id: str | None = None) -> FailureEvent:
        """Inject slowdown."""
        min_delay, max_delay = self.config.slow_node_delay_ms
        delay_ms = random.uniform(min_delay, max_delay)

        if node_id:
            self._delays[node_id] = delay_ms

        return FailureEvent(
            failure_type=FailureType.SLOW_NODE,
            node_type=None,
            node_id=node_id,
            timestamp=datetime.now(),
            duration_ms=delay_ms,
            metadata={"delay_ms": delay_ms},
        )

    def recover(self, event: FailureEvent) -> bool:
        """Clear slowdown."""
        if event.node_id and event.node_id in self._delays:
            del self._delays[event.node_id]
            event.recovered = True
            return True
        return False

    def get_delay(self, node_id: str) -> float:
        """Get current delay for node (ms)."""
        return self._delays.get(node_id, 0)


class ExpertRoutingFailureInjector(FailureInjector):
    """Simulates expert routing failures in MoE."""

    def __init__(self, config: ChaosConfig):
        self.config = config
        self._failed_experts: set[int] = set()

    def inject(self, node_id: str | None = None) -> FailureEvent:
        """Inject expert routing failure."""
        # Randomly select an expert to fail
        expert_id = random.randint(0, 255)  # Up to 256 experts
        self._failed_experts.add(expert_id)

        return FailureEvent(
            failure_type=FailureType.EXPERT_ROUTING_FAILURE,
            node_type=None,
            node_id=node_id,
            timestamp=datetime.now(),
            duration_ms=None,
            metadata={"failed_expert_id": expert_id},
        )

    def recover(self, event: FailureEvent) -> bool:
        """Recover expert."""
        expert_id = event.metadata.get("failed_expert_id")
        if expert_id is not None and expert_id in self._failed_experts:
            self._failed_experts.discard(expert_id)
            event.recovered = True
            return True
        return False

    def is_expert_failed(self, expert_id: int) -> bool:
        """Check if expert is failed."""
        return expert_id in self._failed_experts


class ChaosMetrics:
    """Tracks chaos engineering metrics."""

    def __init__(self):
        self.total_injections = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self.events: list[FailureEvent] = []
        self._lock = threading.Lock()

    def record_injection(self, event: FailureEvent) -> None:
        """Record an injection event."""
        with self._lock:
            self.total_injections += 1
            self.events.append(event)

    def record_recovery(self, success: bool) -> None:
        """Record recovery attempt."""
        with self._lock:
            if success:
                self.successful_recoveries += 1
            else:
                self.failed_recoveries += 1

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        with self._lock:
            return {
                "total_injections": self.total_injections,
                "successful_recoveries": self.successful_recoveries,
                "failed_recoveries": self.failed_recoveries,
                "recovery_rate": (
                    self.successful_recoveries / self.total_injections
                    if self.total_injections > 0 else 0
                ),
                "events_by_type": self._count_by_type(),
            }

    def _count_by_type(self) -> dict[str, int]:
        """Count events by failure type."""
        counts: dict[str, int] = {}
        for event in self.events:
            key = event.failure_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts


class ChaosEngine:
    """
    Main chaos engineering controller.

    Coordinates failure injection across the distributed training pipeline.

    Example:
        chaos = ChaosEngine(ChaosConfig(failure_rate=0.1))

        # Inject failures during training
        with chaos.inject_failures():
            train_step()

        # Or manually
        if chaos.should_inject():
            event = chaos.inject_failure(FailureType.NODE_CRASH)
    """

    def __init__(self, config: ChaosConfig | None = None):
        """Initialize chaos engine."""
        self.config = config or ChaosConfig()
        self.metrics = ChaosMetrics()
        self._running = False

        # Initialize injectors
        self._injectors: dict[FailureType, FailureInjector] = {
            FailureType.NODE_CRASH: NodeCrashInjector(self.config),
            FailureType.NETWORK_PARTITION: NetworkPartitionInjector(self.config),
            FailureType.GPU_OOM: GPUOOMInjector(self.config),
            FailureType.SLOW_NODE: SlowNodeInjector(self.config),
            FailureType.EXPERT_ROUTING_FAILURE: ExpertRoutingFailureInjector(self.config),
        }

        # Callbacks for custom failure handling
        self._failure_callbacks: list[Callable[[FailureEvent], None]] = []
        self._recovery_callbacks: list[Callable[[FailureEvent, bool], None]] = []

    def start(self) -> None:
        """Start chaos engine."""
        self._running = True

    def stop(self) -> None:
        """Stop chaos engine."""
        self._running = False

    def should_inject(self, failure_type: FailureType | None = None) -> bool:
        """Check if we should inject a failure."""
        if not self.config.enabled or not self._running:
            return False

        if failure_type:
            rate = self._get_rate_for_type(failure_type)
        else:
            rate = self.config.failure_rate

        return random.random() < rate

    def _get_rate_for_type(self, failure_type: FailureType) -> float:
        """Get injection rate for failure type."""
        rate_map = {
            FailureType.NODE_CRASH: self.config.node_crash_rate,
            FailureType.NETWORK_PARTITION: self.config.network_partition_rate,
            FailureType.GPU_OOM: self.config.gpu_oom_rate,
            FailureType.CHECKPOINT_CORRUPTION: self.config.checkpoint_corruption_rate,
            FailureType.EXPERT_ROUTING_FAILURE: self.config.expert_routing_failure_rate,
            FailureType.SLOW_NODE: self.config.slow_node_rate,
            FailureType.DATA_CORRUPTION: self.config.data_corruption_rate,
            FailureType.TIMEOUT: self.config.timeout_rate,
        }
        return rate_map.get(failure_type, self.config.failure_rate)

    def inject_failure(
        self,
        failure_type: FailureType,
        node_id: str | None = None,
    ) -> FailureEvent | None:
        """Inject a specific failure."""
        if not self.config.enabled:
            return None

        injector = self._injectors.get(failure_type)
        if not injector:
            return None

        event = injector.inject(node_id)
        self.metrics.record_injection(event)

        # Notify callbacks
        for callback in self._failure_callbacks:
            callback(event)

        if self.config.log_failures:
            print(f"[CHAOS] Injected {failure_type.value} on node {node_id}")

        return event

    def recover(self, event: FailureEvent) -> bool:
        """Attempt recovery from failure."""
        if not self.config.enable_auto_recovery:
            return False

        injector = self._injectors.get(event.failure_type)
        if not injector:
            return False

        start_time = time.time()
        success = False

        for attempt in range(self.config.max_retry_attempts):
            success = injector.recover(event)
            if success:
                break
            time.sleep(0.1)  # Brief wait between attempts

        recovery_time_ms = (time.time() - start_time) * 1000
        event.recovery_time_ms = recovery_time_ms

        self.metrics.record_recovery(success)

        # Notify callbacks
        for callback in self._recovery_callbacks:
            callback(event, success)

        if self.config.log_failures:
            status = "SUCCESS" if success else "FAILED"
            print(f"[CHAOS] Recovery {status} for {event.failure_type.value}")

        return success

    @contextlib.contextmanager
    def inject_failures(self) -> Generator[None, None, None]:
        """Context manager for failure injection during operations."""
        self.start()
        try:
            yield
        finally:
            self.stop()

    def on_failure(self, callback: Callable[[FailureEvent], None]) -> None:
        """Register callback for failure events."""
        self._failure_callbacks.append(callback)

    def on_recovery(self, callback: Callable[[FailureEvent, bool], None]) -> None:
        """Register callback for recovery events."""
        self._recovery_callbacks.append(callback)

    def get_metrics(self) -> dict[str, Any]:
        """Get chaos metrics."""
        return self.metrics.get_summary()

    # Convenience methods for checking node status
    def is_node_crashed(self, node_id: str) -> bool:
        """Check if node is crashed."""
        injector = self._injectors.get(FailureType.NODE_CRASH)
        if isinstance(injector, NodeCrashInjector):
            return injector.is_node_crashed(node_id)
        return False

    def is_node_partitioned(self, node_id: str) -> bool:
        """Check if node is network partitioned."""
        injector = self._injectors.get(FailureType.NETWORK_PARTITION)
        if isinstance(injector, NetworkPartitionInjector):
            return injector.is_partitioned(node_id)
        return False

    def get_node_delay(self, node_id: str) -> float:
        """Get slowdown delay for node (ms)."""
        injector = self._injectors.get(FailureType.SLOW_NODE)
        if isinstance(injector, SlowNodeInjector):
            return injector.get_delay(node_id)
        return 0.0

    def is_expert_failed(self, expert_id: int) -> bool:
        """Check if expert is failed."""
        injector = self._injectors.get(FailureType.EXPERT_ROUTING_FAILURE)
        if isinstance(injector, ExpertRoutingFailureInjector):
            return injector.is_expert_failed(expert_id)
        return False


# Global chaos engine instance
_chaos_engine: ChaosEngine | None = None


def get_chaos_engine() -> ChaosEngine:
    """Get or create global chaos engine."""
    global _chaos_engine
    if _chaos_engine is None:
        _chaos_engine = ChaosEngine()
    return _chaos_engine


def configure_chaos(config: ChaosConfig) -> ChaosEngine:
    """Configure global chaos engine."""
    global _chaos_engine
    _chaos_engine = ChaosEngine(config)
    return _chaos_engine
