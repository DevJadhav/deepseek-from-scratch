#!/usr/bin/env python3
"""
Dual Logger for DeepSeek Training (L1-L4)
==========================================

Implements dual logging to W&B (offline) and TensorBoard with JSON fallback.

Features:
- L1: W&B offline mode for disconnected training
- L2: TensorBoard for real-time visualization
- L3: DualLogger wrapper for unified API
- L4: JSON fallback for reliable persistence

Usage:
    from monitoring.dual_logger import DualLogger

    logger = DualLogger(backend="pytorch", run_name="tiny-v1")
    logger.log({"loss": 0.5, "lr": 1e-4}, step=100)
    logger.close()
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Setup module logger
LOGGER = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


@dataclass
class DualLoggerConfig:
    """Configuration for DualLogger."""
    # Backend identification
    backend: str = "pytorch"
    run_name: Optional[str] = None
    
    # W&B settings (L1)
    wandb_enabled: bool = True
    wandb_mode: str = "offline"  # offline, online, disabled
    wandb_project: str = "deepseek-training"
    wandb_entity: Optional[str] = None
    wandb_dir: str = "./logs/wandb"
    
    # TensorBoard settings (L2)
    tensorboard_enabled: bool = True
    tensorboard_dir: str = "./logs/tensorboard"
    
    # JSON fallback settings (L4)
    json_enabled: bool = True
    json_dir: str = "./logs/json"
    
    # Budget settings
    budget_limit: float = 500.0
    platform: str = "modal"
    gpu_config: str = "A100-80GB x 8"
    
    # Logging intervals
    log_interval: int = 1
    flush_interval: int = 100


class DualLogger:
    """
    Dual logging to W&B (offline) and TensorBoard with JSON fallback.
    
    Implements:
    - L1: W&B offline mode
    - L2: TensorBoard logging
    - L3: Unified DualLogger wrapper
    - L4: JSON fallback
    """
    
    def __init__(
        self,
        backend: str = "pytorch",
        run_name: Optional[str] = None,
        config: Optional[DualLoggerConfig] = None,
        **kwargs,
    ):
        """
        Initialize DualLogger.
        
        Args:
            backend: Training backend (pytorch, mlx, jax)
            run_name: Name for this training run
            config: Full configuration (or use kwargs)
            **kwargs: Override config values
        """
        if config is None:
            config = DualLoggerConfig(backend=backend, run_name=run_name, **kwargs)
        else:
            # Override with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.config = config
        self.backend = config.backend
        self.run_name = config.run_name or f"{backend}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # State
        self._initialized = False
        self._step = 0
        self._start_time = time.time()
        self._metrics_buffer: List[Dict[str, Any]] = []
        
        # Loggers
        self._wandb_run = None
        self._tb_writer = None
        self._json_path: Optional[Path] = None
        
        # Initialize loggers
        self._init_wandb()
        self._init_tensorboard()
        self._init_json()
        
        self._initialized = True
        LOGGER.info(f"DualLogger initialized: backend={backend}, run={self.run_name}")
    
    def _init_wandb(self) -> None:
        """Initialize W&B in offline mode (L1)."""
        if not self.config.wandb_enabled:
            LOGGER.info("W&B disabled by config")
            return
        
        if not WANDB_AVAILABLE:
            LOGGER.warning("wandb not installed, skipping W&B initialization")
            return
        
        try:
            # Create log directory
            wandb_dir = Path(self.config.wandb_dir) / self.backend
            wandb_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize W&B in offline mode
            self._wandb_run = wandb.init(
                mode=self.config.wandb_mode,
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=f"{self.backend}-{self.run_name}",
                dir=str(wandb_dir),
                config={
                    "backend": self.backend,
                    "run_name": self.run_name,
                    "budget_limit": self.config.budget_limit,
                    "platform": self.config.platform,
                    "gpu_config": self.config.gpu_config,
                },
                reinit=True,
            )
            LOGGER.info(f"W&B initialized in {self.config.wandb_mode} mode: {wandb_dir}")
        except Exception as e:
            LOGGER.error(f"Failed to initialize W&B: {e}")
            self._wandb_run = None
    
    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard logging (L2)."""
        if not self.config.tensorboard_enabled:
            LOGGER.info("TensorBoard disabled by config")
            return
        
        if not TENSORBOARD_AVAILABLE:
            LOGGER.warning("tensorboard not installed, skipping TensorBoard initialization")
            return
        
        try:
            tb_dir = Path(self.config.tensorboard_dir) / self.backend / self.run_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
            LOGGER.info(f"TensorBoard initialized: {tb_dir}")
        except Exception as e:
            LOGGER.error(f"Failed to initialize TensorBoard: {e}")
            self._tb_writer = None
    
    def _init_json(self) -> None:
        """Initialize JSON fallback logging (L4)."""
        if not self.config.json_enabled:
            LOGGER.info("JSON logging disabled by config")
            return
        
        try:
            json_dir = Path(self.config.json_dir) / self.backend
            json_dir.mkdir(parents=True, exist_ok=True)
            
            self._json_path = json_dir / f"{self.run_name}.jsonl"
            
            # Write header record
            header = {
                "type": "header",
                "timestamp": datetime.now().isoformat(),
                "backend": self.backend,
                "run_name": self.run_name,
                "config": {
                    "budget_limit": self.config.budget_limit,
                    "platform": self.config.platform,
                    "gpu_config": self.config.gpu_config,
                },
            }
            with open(self._json_path, "w") as f:
                f.write(json.dumps(header) + "\n")
            
            LOGGER.info(f"JSON logging initialized: {self._json_path}")
        except Exception as e:
            LOGGER.error(f"Failed to initialize JSON logging: {e}")
            self._json_path = None
    
    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """
        Log metrics to all backends.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step (auto-increments if not provided)
            commit: Whether to commit immediately (W&B)
        """
        if step is None:
            step = self._step
        self._step = step + 1
        
        # Log to W&B
        if self._wandb_run is not None:
            try:
                wandb.log(metrics, step=step, commit=commit)
            except Exception as e:
                LOGGER.warning(f"W&B log failed: {e}")
        
        # Log to TensorBoard
        if self._tb_writer is not None:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self._tb_writer.add_scalar(key, value, step)
                    elif isinstance(value, dict):
                        # Log nested dicts as grouped scalars
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                self._tb_writer.add_scalar(f"{key}/{sub_key}", sub_value, step)
            except Exception as e:
                LOGGER.warning(f"TensorBoard log failed: {e}")
        
        # Log to JSON (always works)
        if self._json_path is not None:
            try:
                record = {
                    "type": "metrics",
                    "timestamp": datetime.now().isoformat(),
                    "step": step,
                    "elapsed_seconds": time.time() - self._start_time,
                    **metrics,
                }
                with open(self._json_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                LOGGER.warning(f"JSON log failed: {e}")
        
        # Periodic flush
        if step % self.config.flush_interval == 0:
            self.flush()
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self._wandb_run is not None:
            try:
                wandb.config.update(params, allow_val_change=True)
            except Exception as e:
                LOGGER.warning(f"W&B hyperparams log failed: {e}")
        
        if self._tb_writer is not None:
            try:
                # TensorBoard hparams
                self._tb_writer.add_hparams(
                    {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))},
                    {},
                )
            except Exception as e:
                LOGGER.warning(f"TensorBoard hyperparams log failed: {e}")
        
        if self._json_path is not None:
            try:
                record = {
                    "type": "hyperparams",
                    "timestamp": datetime.now().isoformat(),
                    "params": params,
                }
                with open(self._json_path, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except Exception as e:
                LOGGER.warning(f"JSON hyperparams log failed: {e}")
    
    def log_alert(
        self,
        message: str,
        level: str = "INFO",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an alert/warning.
        
        Args:
            message: Alert message
            level: Alert level (INFO, WARNING, CRITICAL)
            data: Additional data
        """
        alert_record = {
            "type": "alert",
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "data": data or {},
        }
        
        # Log to W&B
        if self._wandb_run is not None:
            try:
                wandb.alert(
                    title=f"[{level}] {self.backend}",
                    text=message,
                    level=getattr(wandb.AlertLevel, level, wandb.AlertLevel.INFO),
                )
            except Exception as e:
                LOGGER.warning(f"W&B alert failed: {e}")
        
        # Log to JSON
        if self._json_path is not None:
            try:
                with open(self._json_path, "a") as f:
                    f.write(json.dumps(alert_record) + "\n")
            except Exception as e:
                LOGGER.warning(f"JSON alert log failed: {e}")
        
        # Log to console
        log_func = getattr(LOGGER, level.lower(), LOGGER.info)
        log_func(f"ALERT [{level}]: {message}")
    
    def flush(self) -> None:
        """Flush all loggers."""
        if self._tb_writer is not None:
            try:
                self._tb_writer.flush()
            except Exception:
                pass
    
    def close(self) -> None:
        """Close all loggers."""
        LOGGER.info(f"Closing DualLogger: {self.run_name}")
        
        # Close W&B
        if self._wandb_run is not None:
            try:
                wandb.finish()
            except Exception as e:
                LOGGER.warning(f"W&B close failed: {e}")
        
        # Close TensorBoard
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception as e:
                LOGGER.warning(f"TensorBoard close failed: {e}")
        
        # Write footer to JSON
        if self._json_path is not None:
            try:
                footer = {
                    "type": "footer",
                    "timestamp": datetime.now().isoformat(),
                    "total_steps": self._step,
                    "total_seconds": time.time() - self._start_time,
                }
                with open(self._json_path, "a") as f:
                    f.write(json.dumps(footer) + "\n")
            except Exception as e:
                LOGGER.warning(f"JSON footer write failed: {e}")
        
        self._initialized = False
    
    def __enter__(self) -> "DualLogger":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    @property
    def step(self) -> int:
        """Current step."""
        return self._step
    
    @property
    def elapsed_time(self) -> float:
        """Elapsed time in seconds."""
        return time.time() - self._start_time


def create_dual_logger(
    backend: str,
    run_name: Optional[str] = None,
    budget_limit: float = 500.0,
    log_dir: str = "./logs",
    **kwargs,
) -> DualLogger:
    """
    Factory function to create a DualLogger.
    
    Args:
        backend: Training backend (pytorch, mlx, jax)
        run_name: Name for this training run
        budget_limit: Budget limit for cost tracking
        log_dir: Base directory for logs
        **kwargs: Additional config overrides
    
    Returns:
        Configured DualLogger instance
    """
    config = DualLoggerConfig(
        backend=backend,
        run_name=run_name,
        budget_limit=budget_limit,
        wandb_dir=f"{log_dir}/wandb",
        tensorboard_dir=f"{log_dir}/tensorboard",
        json_dir=f"{log_dir}/json",
        **kwargs,
    )
    return DualLogger(config=config)


if __name__ == "__main__":
    # Test the DualLogger
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DualLogger")
    parser.add_argument("--backend", default="pytorch", help="Backend name")
    parser.add_argument("--steps", type=int, default=100, help="Number of test steps")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing DualLogger...")
    
    with create_dual_logger(
        backend=args.backend,
        run_name="test-run",
        budget_limit=500.0,
    ) as logger:
        # Log hyperparameters
        logger.log_hyperparams({
            "learning_rate": 1e-4,
            "batch_size": 32,
            "model_size": "tiny",
        })
        
        # Log metrics
        import random
        for step in range(args.steps):
            loss = 5.0 * (0.99 ** step) + random.uniform(-0.1, 0.1)
            logger.log({
                "train/loss": loss,
                "train/lr": 1e-4 * (0.99 ** step),
                "train/throughput": 1000 + random.randint(-100, 100),
            }, step=step)
        
        # Log alert
        logger.log_alert("Test alert", level="INFO", data={"test": True})
    
    print("DualLogger test complete!")
