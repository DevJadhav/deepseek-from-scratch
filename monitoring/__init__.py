"""
Monitoring Module for DeepSeek Training Pipeline.

This module provides cost tracking, logging, and dashboard visualization for
training runs on Modal A100 GPUs.

Components:
- cost_tracker: GPU-hour accumulation, cost calculation, budget alerts
- budget_tracker: Budget management with alert callbacks (L5-L10)
- dual_logger: W&B + TensorBoard + JSON logging (L1-L4)
- dashboard: Rich terminal UI for live training progress
"""

from monitoring.cost_tracker import (
    AlertLevel,
    CostAlert,
    CostRecord,
    CostTracker,
    GPUType,
)
from monitoring.budget_tracker import (
    BudgetAlertConfig,
    BudgetExhaustedException,
    BudgetTracker,
    setup_budget_tracker,
)
from monitoring.dual_logger import (
    DualLogger,
    DualLoggerConfig,
    create_dual_logger,
)
from monitoring.dashboard import TrainingDashboard

__all__ = [
    # Cost tracking
    "CostTracker",
    "CostAlert",
    "CostRecord",
    "AlertLevel",
    "GPUType",
    # Budget tracking (L5-L10)
    "BudgetTracker",
    "BudgetAlertConfig",
    "BudgetExhaustedException",
    "setup_budget_tracker",
    # Dual logging (L1-L4)
    "DualLogger",
    "DualLoggerConfig",
    "create_dual_logger",
    # Dashboard
    "TrainingDashboard",
]
