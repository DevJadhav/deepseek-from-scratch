#!/usr/bin/env python3
"""
Cost Tracker for DeepSeek Training Pipeline.

Tracks GPU usage, calculates costs, and manages budget alerts for
training runs on Modal H100 GPUs.

Features:
- GPU-hour accumulation per stage
- Real-time cost calculation ($3.95/hr H100)
- Budget threshold alerts (50%, 75%, 90%, 95%)
- Persist to JSON for recovery
- Historical cost tracking
- Real-time $/token tracking (Section 4.3 paper experiments)
- Energy efficiency metrics (Joules/token)
- Cluster cost analysis for heterogeneous setups

Usage:
    from monitoring.cost_tracker import CostTracker

    tracker = CostTracker(budget_limit=1000.0)
    tracker.start_session("pretrain")
    # ... training runs ...
    tracker.end_session()
    print(tracker.get_summary())
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class GPUType(Enum):
    """Supported GPU types with hourly rates."""

    H100 = 3.95
    A100_80GB = 2.78
    A100_40GB = 2.21
    A10G = 1.10
    T4 = 0.76
    # Apple Silicon (estimated cloud rates)
    M2_ULTRA = 1.50
    M3_MAX = 0.85

    @property
    def hourly_rate(self) -> float:
        """Get hourly rate for this GPU type."""
        return self.value
    
    @property
    def tdp_watts(self) -> float:
        """Get TDP in watts for energy calculations."""
        tdp_map = {
            "H100": 700,
            "A100_80GB": 400,
            "A100_40GB": 400,
            "A10G": 150,
            "T4": 70,
            "M2_ULTRA": 60,
            "M3_MAX": 40,
        }
        return tdp_map.get(self.name, 300)


class AlertLevel(Enum):
    """Budget alert levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"


@dataclass
class CostAlert:
    """Represents a cost alert."""

    level: AlertLevel
    threshold_percent: float
    current_cost: float
    budget_limit: float
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""

    def __post_init__(self):
        if not self.message:
            remaining = self.budget_limit - self.current_cost
            self.message = (
                f"{self.level.value.upper()}: Budget {self.threshold_percent:.0f}% used. "
                f"Spent ${self.current_cost:.2f} of ${self.budget_limit:.2f}. "
                f"Remaining: ${remaining:.2f}"
            )

    def to_dict(self) -> dict:
        """Convert alert to dictionary."""
        return {
            "level": self.level.value,
            "threshold_percent": self.threshold_percent,
            "current_cost": self.current_cost,
            "budget_limit": self.budget_limit,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CostAlert:
        """Create alert from dictionary."""
        return cls(
            level=AlertLevel(data["level"]),
            threshold_percent=data["threshold_percent"],
            current_cost=data["current_cost"],
            budget_limit=data["budget_limit"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message=data["message"],
        )


@dataclass
class CostRecord:
    """Record of a single GPU usage session."""

    stage: str
    gpu_type: GPUType
    start_time: datetime
    end_time: datetime | None = None
    gpu_hours: float = 0.0
    cost: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.end_time and self.gpu_hours == 0.0:
            duration = (self.end_time - self.start_time).total_seconds() / 3600
            self.gpu_hours = duration
            self.cost = duration * self.gpu_type.hourly_rate

    def finalize(self, end_time: datetime | None = None) -> None:
        """Finalize the record with end time and calculate cost."""
        self.end_time = end_time or datetime.now()
        duration = (self.end_time - self.start_time).total_seconds() / 3600
        self.gpu_hours = duration
        self.cost = duration * self.gpu_type.hourly_rate

    def to_dict(self) -> dict:
        """Convert record to dictionary."""
        return {
            "stage": self.stage,
            "gpu_type": self.gpu_type.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "gpu_hours": self.gpu_hours,
            "cost": self.cost,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CostRecord:
        """Create record from dictionary."""
        return cls(
            stage=data["stage"],
            gpu_type=GPUType[data["gpu_type"]],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=(datetime.fromisoformat(data["end_time"]) if data["end_time"] else None),
            gpu_hours=data["gpu_hours"],
            cost=data["cost"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class CostTracker:
    """
    Track GPU costs for training pipeline.

    Features:
    - Track multiple stages (pretrain, sft, grpo, etc.)
    - Real-time cost calculation
    - Budget alerts at configurable thresholds
    - Persistence to JSON
    - Historical tracking

    Example:
        tracker = CostTracker(budget_limit=1000.0)
        tracker.start_session("pretrain")
        # ... training ...
        tracker.end_session()
        print(tracker.total_cost)
    """

    budget_limit: float = 1000.0
    gpu_type: GPUType = GPUType.H100
    alert_thresholds: list[float] = field(default_factory=lambda: [50.0, 75.0, 90.0, 95.0])
    persist_path: Path | None = None
    alert_callback: Callable[[CostAlert], None] | None = None

    # Internal state
    records: list[CostRecord] = field(default_factory=list)
    alerts: list[CostAlert] = field(default_factory=list)
    _current_record: CostRecord | None = field(default=None, repr=False)
    _triggered_thresholds: set[float] = field(default_factory=set, repr=False)

    def __post_init__(self):
        """Initialize tracker and load persisted state if available."""
        if self.persist_path and self.persist_path.exists():
            self.load()

    @property
    def total_gpu_hours(self) -> float:
        """Total GPU hours used across all sessions."""
        total = sum(r.gpu_hours for r in self.records)
        if self._current_record:
            # Add current session's elapsed time
            elapsed = (datetime.now() - self._current_record.start_time).total_seconds() / 3600
            total += elapsed
        return total

    @property
    def total_cost(self) -> float:
        """Total cost across all sessions."""
        total = sum(r.cost for r in self.records)
        if self._current_record:
            # Add current session's cost
            elapsed = (datetime.now() - self._current_record.start_time).total_seconds() / 3600
            total += elapsed * self._current_record.gpu_type.hourly_rate
        return total

    @property
    def remaining_budget(self) -> float:
        """Remaining budget."""
        return max(0.0, self.budget_limit - self.total_cost)

    @property
    def budget_percent_used(self) -> float:
        """Percentage of budget used."""
        if self.budget_limit <= 0:
            return 100.0
        return (self.total_cost / self.budget_limit) * 100

    @property
    def is_over_budget(self) -> bool:
        """Check if over budget."""
        return self.total_cost >= self.budget_limit

    @property
    def estimated_hours_remaining(self) -> float:
        """Estimate GPU hours remaining based on budget."""
        return self.remaining_budget / self.gpu_type.hourly_rate

    def start_session(
        self,
        stage: str,
        gpu_type: GPUType | None = None,
        metadata: dict | None = None,
    ) -> CostRecord:
        """
        Start a new GPU usage session.

        Args:
            stage: Training stage name (pretrain, sft, grpo, etc.)
            gpu_type: GPU type (defaults to tracker's default)
            metadata: Additional metadata for the session

        Returns:
            The created CostRecord
        """
        if self._current_record:
            # End previous session if still active
            self.end_session()

        self._current_record = CostRecord(
            stage=stage,
            gpu_type=gpu_type or self.gpu_type,
            start_time=datetime.now(),
            metadata=metadata or {},
        )
        return self._current_record

    def end_session(self, metadata: dict | None = None) -> CostRecord | None:
        """
        End the current GPU usage session.

        Args:
            metadata: Additional metadata to add to the record

        Returns:
            The finalized CostRecord or None if no active session
        """
        if not self._current_record:
            return None

        self._current_record.finalize()
        if metadata:
            self._current_record.metadata.update(metadata)

        self.records.append(self._current_record)
        record = self._current_record
        self._current_record = None

        # Check for alerts
        self._check_alerts()

        # Persist state
        if self.persist_path:
            self.save()

        return record

    def add_gpu_time(
        self,
        hours: float,
        stage: str = "manual",
        gpu_type: GPUType | None = None,
        metadata: dict | None = None,
    ) -> CostRecord:
        """
        Add GPU time directly without session tracking.

        Args:
            hours: Number of GPU hours
            stage: Training stage name
            gpu_type: GPU type (defaults to tracker's default)
            metadata: Additional metadata

        Returns:
            The created CostRecord
        """
        gpu = gpu_type or self.gpu_type
        now = datetime.now()
        record = CostRecord(
            stage=stage,
            gpu_type=gpu,
            start_time=now,
            end_time=now,
            gpu_hours=hours,
            cost=hours * gpu.hourly_rate,
            metadata=metadata or {},
        )
        self.records.append(record)

        # Check for alerts
        self._check_alerts()

        # Persist state
        if self.persist_path:
            self.save()

        return record

    def _check_alerts(self) -> None:
        """Check and trigger budget alerts."""
        percent_used = self.budget_percent_used

        for threshold in self.alert_thresholds:
            if percent_used >= threshold and threshold not in self._triggered_thresholds:
                self._triggered_thresholds.add(threshold)

                # Determine alert level
                if percent_used >= 100:
                    level = AlertLevel.EXCEEDED
                elif percent_used >= 95:
                    level = AlertLevel.CRITICAL
                elif percent_used >= 75:
                    level = AlertLevel.WARNING
                else:
                    level = AlertLevel.INFO

                alert = CostAlert(
                    level=level,
                    threshold_percent=threshold,
                    current_cost=self.total_cost,
                    budget_limit=self.budget_limit,
                )
                self.alerts.append(alert)

                # Call alert callback if provided
                if self.alert_callback:
                    self.alert_callback(alert)

    def get_stage_summary(self) -> dict[str, dict]:
        """Get cost summary by stage."""
        summary: dict[str, dict] = {}
        for record in self.records:
            if record.stage not in summary:
                summary[record.stage] = {
                    "gpu_hours": 0.0,
                    "cost": 0.0,
                    "sessions": 0,
                }
            summary[record.stage]["gpu_hours"] += record.gpu_hours
            summary[record.stage]["cost"] += record.cost
            summary[record.stage]["sessions"] += 1
        return summary

    def get_summary(self) -> dict:
        """Get full cost summary."""
        return {
            "total_gpu_hours": self.total_gpu_hours,
            "total_cost": self.total_cost,
            "budget_limit": self.budget_limit,
            "remaining_budget": self.remaining_budget,
            "budget_percent_used": self.budget_percent_used,
            "is_over_budget": self.is_over_budget,
            "estimated_hours_remaining": self.estimated_hours_remaining,
            "gpu_type": self.gpu_type.name,
            "hourly_rate": self.gpu_type.hourly_rate,
            "total_sessions": len(self.records),
            "stages": self.get_stage_summary(),
            "alerts": [a.to_dict() for a in self.alerts],
        }

    def save(self, path: Path | None = None) -> None:
        """Save tracker state to JSON file."""
        save_path = path or self.persist_path
        if not save_path:
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "budget_limit": self.budget_limit,
            "gpu_type": self.gpu_type.name,
            "alert_thresholds": self.alert_thresholds,
            "records": [r.to_dict() for r in self.records],
            "alerts": [a.to_dict() for a in self.alerts],
            "triggered_thresholds": list(self._triggered_thresholds),
            "saved_at": datetime.now().isoformat(),
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path | None = None) -> None:
        """Load tracker state from JSON file."""
        load_path = path or self.persist_path
        if not load_path or not load_path.exists():
            return

        with open(load_path) as f:
            data = json.load(f)

        self.budget_limit = data.get("budget_limit", self.budget_limit)
        self.gpu_type = GPUType[data.get("gpu_type", "H100")]
        self.alert_thresholds = data.get("alert_thresholds", self.alert_thresholds)
        self.records = [CostRecord.from_dict(r) for r in data.get("records", [])]
        self.alerts = [CostAlert.from_dict(a) for a in data.get("alerts", [])]
        self._triggered_thresholds = set(data.get("triggered_thresholds", []))

    def reset(self) -> None:
        """Reset tracker state."""
        self.records = []
        self.alerts = []
        self._current_record = None
        self._triggered_thresholds = set()

        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()

    def __str__(self) -> str:
        """String representation of tracker state."""
        return (
            f"CostTracker("
            f"total_cost=${self.total_cost:.2f}, "
            f"budget=${self.budget_limit:.2f}, "
            f"used={self.budget_percent_used:.1f}%, "
            f"sessions={len(self.records)})"
        )


def create_tracker(
    budget: float = 1000.0,
    gpu_type: str = "H100",
    persist_path: str | Path | None = None,
) -> CostTracker:
    """
    Factory function to create a CostTracker.

    Args:
        budget: Budget limit in USD
        gpu_type: GPU type name (H100, A100_80GB, etc.)
        persist_path: Path to persist tracker state

    Returns:
        Configured CostTracker instance
    """
    return CostTracker(
        budget_limit=budget,
        gpu_type=GPUType[gpu_type],
        persist_path=Path(persist_path) if persist_path else None,
    )


# ============================================================================
# Section 4.3 Paper Experiments: Real-Time Cost Analysis
# ============================================================================


@dataclass
class TokenMetrics:
    """Real-time token processing metrics for $/token analysis."""
    
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    cost_per_token: float = 0.0
    cost_per_million_tokens: float = 0.0
    energy_per_token_joules: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
            "cost_per_token": self.cost_per_token,
            "cost_per_million_tokens": self.cost_per_million_tokens,
            "energy_per_token_joules": self.energy_per_token_joules,
        }


@dataclass
class ClusterCostMetrics:
    """Cost metrics for heterogeneous cluster analysis (A4 experiment)."""
    
    # Cluster composition
    h100_count: int = 0
    a100_count: int = 0
    a10g_count: int = 0
    metal_count: int = 0
    
    # Cost breakdown
    total_hourly_cost: float = 0.0
    cost_per_device: dict[str, float] = field(default_factory=dict)
    
    # Efficiency metrics
    throughput_per_dollar: float = 0.0
    tokens_per_dollar: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "h100_count": self.h100_count,
            "a100_count": self.a100_count,
            "a10g_count": self.a10g_count,
            "metal_count": self.metal_count,
            "total_hourly_cost": self.total_hourly_cost,
            "cost_per_device": self.cost_per_device,
            "throughput_per_dollar": self.throughput_per_dollar,
            "tokens_per_dollar": self.tokens_per_dollar,
        }


@dataclass
class EnergyMetrics:
    """Energy consumption metrics for Figure 1 (Throughput vs Energy)."""
    
    total_energy_joules: float = 0.0
    total_energy_kwh: float = 0.0
    avg_power_watts: float = 0.0
    peak_power_watts: float = 0.0
    energy_efficiency_tokens_per_joule: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_energy_joules": self.total_energy_joules,
            "total_energy_kwh": self.total_energy_kwh,
            "avg_power_watts": self.avg_power_watts,
            "peak_power_watts": self.peak_power_watts,
            "energy_efficiency_tokens_per_joule": self.energy_efficiency_tokens_per_joule,
        }


class RealTimeCostAnalyzer:
    """
    Real-time cost analyzer for paper experiments (Section 4.3).
    
    Tracks:
    - $/token in real-time
    - Energy efficiency (Joules/token)
    - Cluster cost analysis for heterogeneous setups
    - Throughput vs cost trade-offs
    """
    
    def __init__(
        self,
        tracker: CostTracker,
        update_interval_seconds: float = 1.0,
    ):
        self.tracker = tracker
        self.update_interval = update_interval_seconds
        
        # Token tracking
        self._total_tokens = 0
        self._token_timestamps: list[tuple[float, int]] = []  # (timestamp, token_count)
        
        # Energy tracking
        self._energy_samples: list[tuple[float, float]] = []  # (timestamp, power_watts)
        self._total_energy_joules = 0.0
        
        # Cluster composition
        self._cluster_devices: dict[str, int] = {}
        
        # Real-time metrics
        self._current_metrics: TokenMetrics = TokenMetrics()
        self._cluster_metrics: ClusterCostMetrics = ClusterCostMetrics()
        self._energy_metrics: EnergyMetrics = EnergyMetrics()
        
        # Background update thread
        self._running = False
        self._update_thread: threading.Thread | None = None
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start real-time monitoring."""
        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
    
    def stop(self) -> None:
        """Stop real-time monitoring."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
    
    def record_tokens(self, token_count: int) -> None:
        """Record tokens processed."""
        with self._lock:
            now = time.time()
            self._total_tokens += token_count
            self._token_timestamps.append((now, token_count))
            
            # Keep only last 60 seconds of timestamps
            cutoff = now - 60.0
            self._token_timestamps = [
                (ts, tc) for ts, tc in self._token_timestamps if ts > cutoff
            ]
    
    def record_power_sample(self, power_watts: float) -> None:
        """Record a power consumption sample."""
        with self._lock:
            now = time.time()
            self._energy_samples.append((now, power_watts))
            
            # Calculate incremental energy
            if len(self._energy_samples) >= 2:
                prev_ts, prev_power = self._energy_samples[-2]
                dt = now - prev_ts
                avg_power = (prev_power + power_watts) / 2
                self._total_energy_joules += avg_power * dt
            
            # Keep only last 60 seconds
            cutoff = now - 60.0
            self._energy_samples = [
                (ts, pw) for ts, pw in self._energy_samples if ts > cutoff
            ]
    
    def set_cluster_composition(
        self,
        h100_count: int = 0,
        a100_count: int = 0,
        a10g_count: int = 0,
        metal_count: int = 0,
    ) -> None:
        """Set the cluster device composition for cost analysis."""
        self._cluster_devices = {
            "H100": h100_count,
            "A100_80GB": a100_count,
            "A10G": a10g_count,
            "M2_ULTRA": metal_count,
        }
        self._update_cluster_metrics()
    
    def _update_loop(self) -> None:
        """Background update loop."""
        while self._running:
            self._update_metrics()
            time.sleep(self.update_interval)
    
    def _update_metrics(self) -> None:
        """Update all real-time metrics."""
        with self._lock:
            now = time.time()
            
            # Calculate tokens per second (last 10 seconds)
            recent_cutoff = now - 10.0
            recent_tokens = sum(
                tc for ts, tc in self._token_timestamps if ts > recent_cutoff
            )
            tokens_per_second = recent_tokens / 10.0 if recent_tokens > 0 else 0.0
            
            # Calculate cost per token
            total_cost = self.tracker.total_cost
            cost_per_token = total_cost / max(1, self._total_tokens)
            cost_per_million = cost_per_token * 1_000_000
            
            # Calculate energy per token
            energy_per_token = self._total_energy_joules / max(1, self._total_tokens)
            
            # Update current metrics
            self._current_metrics = TokenMetrics(
                total_tokens=self._total_tokens,
                tokens_per_second=tokens_per_second,
                cost_per_token=cost_per_token,
                cost_per_million_tokens=cost_per_million,
                energy_per_token_joules=energy_per_token,
            )
            
            # Update energy metrics
            if self._energy_samples:
                powers = [pw for _, pw in self._energy_samples]
                self._energy_metrics = EnergyMetrics(
                    total_energy_joules=self._total_energy_joules,
                    total_energy_kwh=self._total_energy_joules / 3_600_000,
                    avg_power_watts=sum(powers) / len(powers),
                    peak_power_watts=max(powers),
                    energy_efficiency_tokens_per_joule=(
                        self._total_tokens / max(1, self._total_energy_joules)
                    ),
                )
    
    def _update_cluster_metrics(self) -> None:
        """Update cluster cost metrics."""
        total_hourly = 0.0
        cost_per_device: dict[str, float] = {}
        
        for device_type, count in self._cluster_devices.items():
            if count > 0:
                try:
                    gpu = GPUType[device_type]
                    device_cost = gpu.hourly_rate * count
                    total_hourly += device_cost
                    cost_per_device[device_type] = device_cost
                except KeyError:
                    pass
        
        # Calculate efficiency metrics
        throughput = self._current_metrics.tokens_per_second * 3600  # tokens/hour
        tokens_per_dollar = throughput / max(0.01, total_hourly)
        throughput_per_dollar = tokens_per_dollar
        
        self._cluster_metrics = ClusterCostMetrics(
            h100_count=self._cluster_devices.get("H100", 0),
            a100_count=self._cluster_devices.get("A100_80GB", 0),
            a10g_count=self._cluster_devices.get("A10G", 0),
            metal_count=self._cluster_devices.get("M2_ULTRA", 0),
            total_hourly_cost=total_hourly,
            cost_per_device=cost_per_device,
            throughput_per_dollar=throughput_per_dollar,
            tokens_per_dollar=tokens_per_dollar,
        )
    
    def get_current_metrics(self) -> TokenMetrics:
        """Get current token metrics."""
        return self._current_metrics
    
    def get_cluster_metrics(self) -> ClusterCostMetrics:
        """Get cluster cost metrics."""
        return self._cluster_metrics
    
    def get_energy_metrics(self) -> EnergyMetrics:
        """Get energy metrics."""
        return self._energy_metrics
    
    def get_full_analysis(self) -> dict[str, Any]:
        """Get complete cost analysis for paper experiments."""
        return {
            "token_metrics": self._current_metrics.to_dict(),
            "cluster_metrics": self._cluster_metrics.to_dict(),
            "energy_metrics": self._energy_metrics.to_dict(),
            "tracker_summary": self.tracker.get_summary(),
        }
    
    def export_paper_figures_data(self) -> dict[str, Any]:
        """
        Export data suitable for paper figures (Section 4.3).
        
        Returns data for:
        - Figure 1: Throughput vs Energy
        - Figure 2: Mixed Cluster Efficiency  
        - A4: Cluster cost analysis
        """
        return {
            "figure_1_throughput_energy": {
                "throughput_tokens_per_sec": self._current_metrics.tokens_per_second,
                "energy_joules_per_token": self._current_metrics.energy_per_token_joules,
                "cost_per_million_tokens": self._current_metrics.cost_per_million_tokens,
            },
            "figure_2_cluster_efficiency": {
                "device_composition": self._cluster_devices,
                "total_hourly_cost": self._cluster_metrics.total_hourly_cost,
                "tokens_per_dollar": self._cluster_metrics.tokens_per_dollar,
            },
            "a4_cluster_cost": {
                "cost_breakdown": self._cluster_metrics.cost_per_device,
                "efficiency_per_device_type": {
                    device: self._cluster_metrics.tokens_per_dollar / max(1, count)
                    for device, count in self._cluster_devices.items()
                    if count > 0
                },
            },
        }
