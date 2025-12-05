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
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class GPUType(Enum):
    """Supported GPU types with hourly rates."""

    H100 = 3.95
    A100_80GB = 2.78
    A100_40GB = 2.21
    A10G = 1.10
    T4 = 0.76

    @property
    def hourly_rate(self) -> float:
        """Get hourly rate for this GPU type."""
        return self.value


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
