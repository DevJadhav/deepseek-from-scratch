"""
Monitoring Module for DeepSeek Training Pipeline.

This module provides cost tracking and dashboard visualization for
training runs on Modal H100 GPUs.

Components:
- cost_tracker: GPU-hour accumulation, cost calculation, budget alerts
- dashboard: Rich terminal UI for live training progress
"""

from monitoring.cost_tracker import CostAlert, CostRecord, CostTracker
from monitoring.dashboard import TrainingDashboard

__all__ = [
    "CostTracker",
    "CostAlert",
    "CostRecord",
    "TrainingDashboard",
]
