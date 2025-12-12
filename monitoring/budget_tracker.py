#!/usr/bin/env python3
"""
Budget Tracker Setup for DeepSeek Training (L5-L10)
====================================================

Implements budget tracking with alerts for training runs.

Features:
- L5: $500 budget tracking per backend
- L6: 50% ($250) INFO alert
- L7: 75% ($375) WARNING alert
- L8: 90% ($450) CRITICAL alert + checkpoint
- L9: 95% ($475) checkpoint + notification
- L10: 99% ($495) auto-stop

Usage:
    from monitoring.budget_tracker import setup_budget_tracker

    tracker = setup_budget_tracker(
        backend="pytorch",
        budget_limit=500.0,
        log_dir="./logs",
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from monitoring.cost_tracker import AlertLevel, CostAlert, CostTracker, GPUType

LOGGER = logging.getLogger(__name__)


@dataclass
class BudgetAlertConfig:
    """Configuration for budget alerts."""
    threshold_percent: float
    threshold_amount: float
    level: AlertLevel
    action: str  # "log", "checkpoint", "notify", "stop"
    message: str


class BudgetTracker:
    """
    Budget tracker with configurable alerts (L5-L10).
    
    Wraps CostTracker with specific alert thresholds:
    - 50% ($250): INFO alert
    - 75% ($375): WARNING alert
    - 90% ($450): CRITICAL alert + checkpoint
    - 95% ($475): checkpoint + notification
    - 99% ($495): auto-stop
    """
    
    DEFAULT_BUDGET = 500.0
    DEFAULT_ALERTS = [
        BudgetAlertConfig(50.0, 250.0, AlertLevel.INFO, "log", "Budget 50% used"),
        BudgetAlertConfig(75.0, 375.0, AlertLevel.WARNING, "log", "Budget 75% used"),
        BudgetAlertConfig(90.0, 450.0, AlertLevel.CRITICAL, "checkpoint", "Budget 90% - saving checkpoint"),
        BudgetAlertConfig(95.0, 475.0, AlertLevel.CRITICAL, "notify", "Budget 95% - emergency checkpoint"),
        BudgetAlertConfig(99.0, 495.0, AlertLevel.EXCEEDED, "stop", "Budget 99% - stopping training"),
    ]
    
    def __init__(
        self,
        backend: str,
        budget_limit: float = DEFAULT_BUDGET,
        gpu_type: GPUType = GPUType.A100_80GB,
        log_dir: str = "./logs",
        checkpoint_callback: Callable[[], None] | None = None,
        stop_callback: Callable[[], None] | None = None,
        notify_callback: Callable[[str], None] | None = None,
    ):
        """
        Initialize BudgetTracker.
        
        Args:
            backend: Training backend name
            budget_limit: Maximum budget in dollars
            gpu_type: GPU type for cost calculation
            log_dir: Directory for logs and persistence
            checkpoint_callback: Function to save emergency checkpoint
            stop_callback: Function to stop training
            notify_callback: Function to send notifications
        """
        self.backend = backend
        self.budget_limit = budget_limit
        self.gpu_type = gpu_type
        self.log_dir = Path(log_dir)
        
        # Callbacks
        self.checkpoint_callback = checkpoint_callback
        self.stop_callback = stop_callback
        self.notify_callback = notify_callback
        
        # Calculate alert thresholds based on budget
        self.alert_configs = [
            BudgetAlertConfig(
                threshold_percent=cfg.threshold_percent,
                threshold_amount=budget_limit * (cfg.threshold_percent / 100),
                level=cfg.level,
                action=cfg.action,
                message=cfg.message,
            )
            for cfg in self.DEFAULT_ALERTS
        ]
        
        # Setup persist path
        persist_dir = self.log_dir / "json" / backend
        persist_dir.mkdir(parents=True, exist_ok=True)
        persist_path = persist_dir / "budget_tracker.json"
        
        # Create underlying cost tracker
        self.cost_tracker = CostTracker(
            budget_limit=budget_limit,
            gpu_type=gpu_type,
            alert_thresholds=[cfg.threshold_percent for cfg in self.alert_configs],
            persist_path=persist_path,
            alert_callback=self._handle_alert,
        )
        
        # Track triggered alerts
        self._triggered_alerts: set[float] = set()
        self._alert_log: list[dict] = []
        
        LOGGER.info(
            f"BudgetTracker initialized: backend={backend}, "
            f"budget=${budget_limit:.2f}, gpu={gpu_type.name}"
        )
    
    def _handle_alert(self, alert: CostAlert) -> None:
        """Handle a cost alert."""
        # Find matching config
        config = None
        for cfg in self.alert_configs:
            if abs(cfg.threshold_percent - alert.threshold_percent) < 0.1:
                config = cfg
                break
        
        if config is None:
            LOGGER.warning(f"No config found for alert at {alert.threshold_percent}%")
            return
        
        # Skip if already triggered
        if alert.threshold_percent in self._triggered_alerts:
            return
        
        self._triggered_alerts.add(alert.threshold_percent)
        
        # Log the alert
        alert_record = {
            "timestamp": datetime.now().isoformat(),
            "backend": self.backend,
            "threshold_percent": alert.threshold_percent,
            "threshold_amount": config.threshold_amount,
            "current_cost": alert.current_cost,
            "budget_limit": alert.budget_limit,
            "level": alert.level.value,
            "action": config.action,
            "message": config.message,
        }
        self._alert_log.append(alert_record)
        
        # Log to console
        log_level = getattr(logging, alert.level.value.upper(), logging.INFO)
        LOGGER.log(
            log_level,
            f"[{self.backend}] {config.message}: "
            f"${alert.current_cost:.2f}/${alert.budget_limit:.2f} "
            f"({alert.threshold_percent:.0f}%)"
        )
        
        # Execute action
        if config.action == "checkpoint":
            self._do_checkpoint(alert_record)
        elif config.action == "notify":
            self._do_checkpoint(alert_record)
            self._do_notify(alert_record)
        elif config.action == "stop":
            self._do_checkpoint(alert_record)
            self._do_notify(alert_record)
            self._do_stop(alert_record)
        
        # Save alert log
        self._save_alert_log()
    
    def _do_checkpoint(self, alert_record: dict) -> None:
        """Save emergency checkpoint."""
        LOGGER.info(f"[{self.backend}] Saving emergency checkpoint...")
        if self.checkpoint_callback:
            try:
                self.checkpoint_callback()
                alert_record["checkpoint_saved"] = True
            except Exception as e:
                LOGGER.error(f"Checkpoint failed: {e}")
                alert_record["checkpoint_saved"] = False
                alert_record["checkpoint_error"] = str(e)
        else:
            LOGGER.warning("No checkpoint callback configured")
            alert_record["checkpoint_saved"] = False
    
    def _do_notify(self, alert_record: dict) -> None:
        """Send notification."""
        message = (
            f"[{self.backend}] Budget Alert: "
            f"{alert_record['threshold_percent']:.0f}% used "
            f"(${alert_record['current_cost']:.2f}/${alert_record['budget_limit']:.2f})"
        )
        LOGGER.warning(f"Notification: {message}")
        
        if self.notify_callback:
            try:
                self.notify_callback(message)
                alert_record["notification_sent"] = True
            except Exception as e:
                LOGGER.error(f"Notification failed: {e}")
                alert_record["notification_sent"] = False
                alert_record["notification_error"] = str(e)
    
    def _do_stop(self, alert_record: dict) -> None:
        """Stop training."""
        LOGGER.critical(f"[{self.backend}] STOPPING TRAINING - Budget exhausted!")
        alert_record["training_stopped"] = True
        
        if self.stop_callback:
            try:
                self.stop_callback()
            except Exception as e:
                LOGGER.error(f"Stop callback failed: {e}")
                alert_record["stop_error"] = str(e)
        else:
            # Raise exception to stop training
            raise BudgetExhaustedException(
                f"Budget exhausted: ${alert_record['current_cost']:.2f}/"
                f"${alert_record['budget_limit']:.2f}"
            )
    
    def _save_alert_log(self) -> None:
        """Save alert log to file."""
        try:
            alert_log_path = self.log_dir / "json" / self.backend / "budget_alerts.json"
            with open(alert_log_path, "w") as f:
                json.dump(self._alert_log, f, indent=2)
        except Exception as e:
            LOGGER.warning(f"Failed to save alert log: {e}")
    
    def start_session(self, stage: str, metadata: dict | None = None):
        """Start a training session."""
        return self.cost_tracker.start_session(stage, metadata=metadata)
    
    def end_session(self, metadata: dict | None = None):
        """End the current session."""
        return self.cost_tracker.end_session(metadata=metadata)
    
    def add_gpu_time(self, hours: float, stage: str = "training"):
        """Add GPU time directly."""
        return self.cost_tracker.add_gpu_time(hours, stage)
    
    def check_budget(self) -> bool:
        """Check if still within budget."""
        return not self.cost_tracker.is_over_budget
    
    @property
    def total_cost(self) -> float:
        """Total cost so far."""
        return self.cost_tracker.total_cost
    
    @property
    def remaining_budget(self) -> float:
        """Remaining budget."""
        return self.cost_tracker.remaining_budget
    
    @property
    def budget_percent_used(self) -> float:
        """Percentage of budget used."""
        return self.cost_tracker.budget_percent_used
    
    def get_summary(self) -> dict:
        """Get budget summary."""
        return {
            "backend": self.backend,
            "budget_limit": self.budget_limit,
            "total_cost": self.total_cost,
            "remaining_budget": self.remaining_budget,
            "percent_used": self.budget_percent_used,
            "gpu_type": self.gpu_type.name,
            "gpu_hours": self.cost_tracker.total_gpu_hours,
            "alerts_triggered": len(self._triggered_alerts),
            "is_over_budget": self.cost_tracker.is_over_budget,
        }


class BudgetExhaustedException(Exception):
    """Raised when training budget is exhausted."""
    pass


def setup_budget_tracker(
    backend: str,
    budget_limit: float = 500.0,
    gpu_type: GPUType = GPUType.A100_80GB,
    log_dir: str = "./logs",
    checkpoint_callback: Callable[[], None] | None = None,
    stop_callback: Callable[[], None] | None = None,
    notify_callback: Callable[[str], None] | None = None,
) -> BudgetTracker:
    """
    Factory function to create a BudgetTracker.
    
    Args:
        backend: Training backend (pytorch, mlx, jax)
        budget_limit: Maximum budget (default: $500)
        gpu_type: GPU type (default: A100-80GB)
        log_dir: Directory for logs
        checkpoint_callback: Function to save checkpoint
        stop_callback: Function to stop training
        notify_callback: Function to send notifications
    
    Returns:
        Configured BudgetTracker
    """
    return BudgetTracker(
        backend=backend,
        budget_limit=budget_limit,
        gpu_type=gpu_type,
        log_dir=log_dir,
        checkpoint_callback=checkpoint_callback,
        stop_callback=stop_callback,
        notify_callback=notify_callback,
    )


if __name__ == "__main__":
    # Test the BudgetTracker
    import time
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    print("Testing BudgetTracker...")
    print("=" * 60)
    
    # Create tracker
    tracker = setup_budget_tracker(
        backend="test",
        budget_limit=100.0,  # $100 for testing
        gpu_type=GPUType.A100_80GB,
        log_dir="./logs",
        checkpoint_callback=lambda: print("  [CALLBACK] Saving checkpoint..."),
        notify_callback=lambda msg: print(f"  [CALLBACK] Notification: {msg}"),
    )
    
    print(f"Budget: ${tracker.budget_limit:.2f}")
    print(f"GPU: {tracker.gpu_type.name} @ ${tracker.gpu_type.hourly_rate:.2f}/hr")
    print()
    
    # Simulate training
    hourly_rate = tracker.gpu_type.hourly_rate
    
    try:
        # Add GPU time incrementally to trigger alerts
        stages = [
            (10.0, "warmup"),      # ~$27.80
            (10.0, "phase1"),      # ~$55.60 (>50%)
            (10.0, "phase2"),      # ~$83.40 (>75%)
            (5.0, "phase3"),       # ~$97.30 (>90%)
            (1.0, "phase4"),       # ~$100.08 (>95%, >99%)
        ]
        
        for hours, stage in stages:
            print(f"Adding {hours} GPU hours for {stage}...")
            tracker.add_gpu_time(hours, stage)
            summary = tracker.get_summary()
            print(f"  Cost: ${summary['total_cost']:.2f} ({summary['percent_used']:.1f}%)")
            print()
            
    except BudgetExhaustedException as e:
        print(f"\n[STOPPED] {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    summary = tracker.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
