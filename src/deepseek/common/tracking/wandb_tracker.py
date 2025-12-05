"""
Weights & Biases Integration for DeepSeek Training
===================================================

Comprehensive experiment tracking with W&B including:
- Training metrics (loss, perplexity, learning rate, gradients)
- MoE-specific metrics (expert utilization, load balance)
- Memory and throughput metrics
- Artifact versioning and model checkpoints
- Hyperparameter logging from Hydra config
- Alert configuration for training anomalies

Usage:
    from deepseek.common.tracking.wandb_tracker import WandbTracker
    
    tracker = WandbTracker(config)
    tracker.init()
    
    for step in range(max_steps):
        # ... training loop ...
        tracker.log_training_step(metrics)
        tracker.log_expert_stats(expert_stats)
        
    tracker.finish()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

try:
    import wandb
    from wandb import AlertLevel
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


@dataclass
class WandbConfig:
    """Configuration for W&B tracking."""
    enabled: bool = True
    project: str = "deepseek-training"
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "training"
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    mode: str = "online"  # online, offline, disabled
    resume: str = "allow"
    id: Optional[str] = None
    log_model: bool = True
    log_code: bool = True
    log_config: bool = True
    dir: str = "./wandb"
    
    # Alert thresholds
    loss_spike_threshold: float = 2.0
    loss_spike_window: int = 100
    gpu_util_threshold: float = 50.0


class WandbTracker:
    """
    Comprehensive W&B tracking for DeepSeek training.
    
    Handles:
    - Initialization and configuration
    - Training metrics logging
    - Expert/MoE statistics
    - Memory and throughput metrics
    - Model checkpointing as artifacts
    - Alert management
    """
    
    def __init__(
        self,
        config: Union[WandbConfig, Dict[str, Any]],
        hydra_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize W&B tracker.
        
        Args:
            config: W&B configuration
            hydra_config: Full Hydra configuration for logging
        """
        if isinstance(config, dict):
            config = WandbConfig(**config)
        self.config = config
        self.hydra_config = hydra_config
        
        self.run = None
        self.step = 0
        self.loss_history: List[float] = []
        self.start_time = time.time()
        self._initialized = False
        
    def init(self) -> Optional[Any]:
        """Initialize W&B run."""
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Tracking disabled.")
            return None
            
        if not self.config.enabled:
            return None
            
        # Set mode
        if self.config.mode == "disabled":
            os.environ["WANDB_MODE"] = "disabled"
            return None
            
        # Initialize run
        self.run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=self.config.name,
            group=self.config.group,
            job_type=self.config.job_type,
            tags=self.config.tags,
            notes=self.config.notes,
            mode=self.config.mode,
            resume=self.config.resume,
            id=self.config.id,
            dir=self.config.dir,
            config=self.hydra_config,
            save_code=self.config.log_code,
        )
        
        self._initialized = True
        return self.run
        
    def log_training_step(
        self,
        loss: float,
        learning_rate: float,
        gradient_norm: Optional[float] = None,
        epoch: Optional[int] = None,
        tokens_seen: Optional[int] = None,
        step: Optional[int] = None,
        **extra_metrics,
    ) -> None:
        """
        Log training step metrics.
        
        Args:
            loss: Training loss
            learning_rate: Current learning rate
            gradient_norm: Gradient norm (if computed)
            epoch: Current epoch
            tokens_seen: Total tokens processed
            step: Current step (uses internal counter if None)
            **extra_metrics: Additional metrics to log
        """
        if not self._initialized or self.run is None:
            return
            
        if step is not None:
            self.step = step
        else:
            self.step += 1
            
        # Update loss history for spike detection
        self.loss_history.append(loss)
        if len(self.loss_history) > self.config.loss_spike_window:
            self.loss_history.pop(0)
            
        # Build metrics dict
        metrics = {
            "train/loss": loss,
            "train/learning_rate": learning_rate,
            "train/step": self.step,
        }
        
        if gradient_norm is not None:
            metrics["train/gradient_norm"] = gradient_norm
        if epoch is not None:
            metrics["train/epoch"] = epoch
        if tokens_seen is not None:
            metrics["train/tokens_seen"] = tokens_seen
            
        # Add perplexity
        try:
            metrics["train/perplexity"] = torch.exp(torch.tensor(loss)).item()
        except (ValueError, OverflowError):
            pass
            
        # Add extra metrics
        for key, value in extra_metrics.items():
            if not key.startswith("train/"):
                key = f"train/{key}"
            metrics[key] = value
            
        # Log to W&B
        self.run.log(metrics, step=self.step)
        
        # Check for loss spike
        self._check_loss_spike(loss)
        
        # Check for NaN
        self._check_nan(loss)
        
    def log_expert_stats(
        self,
        expert_utilization: Optional[Dict[int, float]] = None,
        load_balance_cv: Optional[float] = None,
        per_expert_tokens: Optional[Dict[int, int]] = None,
        router_entropy: Optional[float] = None,
        dropped_tokens: Optional[int] = None,
        dropped_token_ratio: Optional[float] = None,
    ) -> None:
        """
        Log MoE expert statistics.
        
        Args:
            expert_utilization: Utilization per expert (0-1)
            load_balance_cv: Coefficient of variation for load balance
            per_expert_tokens: Token count per expert
            router_entropy: Router probability entropy
            dropped_tokens: Number of dropped tokens
            dropped_token_ratio: Ratio of dropped tokens
        """
        if not self._initialized or self.run is None:
            return
            
        metrics = {}
        
        if expert_utilization is not None:
            for expert_id, util in expert_utilization.items():
                metrics[f"moe/expert_{expert_id}_utilization"] = util
            metrics["moe/mean_utilization"] = sum(expert_utilization.values()) / len(expert_utilization)
            
        if load_balance_cv is not None:
            metrics["moe/load_balance_cv"] = load_balance_cv
            
        if per_expert_tokens is not None:
            for expert_id, count in per_expert_tokens.items():
                metrics[f"moe/expert_{expert_id}_tokens"] = count
                
        if router_entropy is not None:
            metrics["moe/router_entropy"] = router_entropy
            
        if dropped_tokens is not None:
            metrics["moe/dropped_tokens"] = dropped_tokens
            
        if dropped_token_ratio is not None:
            metrics["moe/dropped_token_ratio"] = dropped_token_ratio
            
        if metrics:
            self.run.log(metrics, step=self.step)
            
    def log_mtp_metrics(
        self,
        accuracies: Dict[int, float],
        mtp_loss: Optional[float] = None,
    ) -> None:
        """
        Log Multi-Token Prediction metrics.
        
        Args:
            accuracies: Accuracy at each prediction depth
            mtp_loss: Total MTP loss
        """
        if not self._initialized or self.run is None:
            return
            
        metrics = {}
        
        for depth, accuracy in accuracies.items():
            metrics[f"mtp/accuracy_depth_{depth}"] = accuracy
            
        if mtp_loss is not None:
            metrics["mtp/loss"] = mtp_loss
            
        if metrics:
            self.run.log(metrics, step=self.step)
            
    def log_memory_stats(
        self,
        allocated_gb: Optional[float] = None,
        reserved_gb: Optional[float] = None,
        peak_gb: Optional[float] = None,
        activation_memory_gb: Optional[float] = None,
    ) -> None:
        """
        Log GPU memory statistics.
        
        Args:
            allocated_gb: Currently allocated memory in GB
            reserved_gb: Reserved memory in GB
            peak_gb: Peak memory usage in GB
            activation_memory_gb: Estimated activation memory
        """
        if not self._initialized or self.run is None:
            return
            
        metrics = {}
        
        if allocated_gb is not None:
            metrics["memory/allocated_gb"] = allocated_gb
        if reserved_gb is not None:
            metrics["memory/reserved_gb"] = reserved_gb
        if peak_gb is not None:
            metrics["memory/peak_gb"] = peak_gb
        if activation_memory_gb is not None:
            metrics["memory/activation_gb"] = activation_memory_gb
            
        if metrics:
            self.run.log(metrics, step=self.step)
            
    def log_throughput(
        self,
        tokens_per_sec: Optional[float] = None,
        samples_per_sec: Optional[float] = None,
        mbu: Optional[float] = None,
        step_time: Optional[float] = None,
    ) -> None:
        """
        Log throughput metrics.
        
        Args:
            tokens_per_sec: Training throughput in tokens/second
            samples_per_sec: Training throughput in samples/second
            mbu: Model Bandwidth Utilization
            step_time: Time per training step
        """
        if not self._initialized or self.run is None:
            return
            
        metrics = {}
        
        if tokens_per_sec is not None:
            metrics["throughput/tokens_per_sec"] = tokens_per_sec
        if samples_per_sec is not None:
            metrics["throughput/samples_per_sec"] = samples_per_sec
        if mbu is not None:
            metrics["throughput/mbu"] = mbu
        if step_time is not None:
            metrics["throughput/step_time"] = step_time
            
        if metrics:
            self.run.log(metrics, step=self.step)
            
    def log_validation(
        self,
        val_loss: float,
        val_perplexity: Optional[float] = None,
        **extra_metrics,
    ) -> None:
        """
        Log validation metrics.
        
        Args:
            val_loss: Validation loss
            val_perplexity: Validation perplexity
            **extra_metrics: Additional validation metrics
        """
        if not self._initialized or self.run is None:
            return
            
        metrics = {
            "val/loss": val_loss,
        }
        
        if val_perplexity is not None:
            metrics["val/perplexity"] = val_perplexity
        else:
            try:
                metrics["val/perplexity"] = torch.exp(torch.tensor(val_loss)).item()
            except (ValueError, OverflowError):
                pass
                
        for key, value in extra_metrics.items():
            if not key.startswith("val/"):
                key = f"val/{key}"
            metrics[key] = value
            
        self.run.log(metrics, step=self.step)
        
    def log_model_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log model checkpoint as W&B artifact.
        
        Args:
            checkpoint_path: Path to checkpoint file or directory
            metadata: Additional metadata for the artifact
        """
        if not self._initialized or self.run is None or not self.config.log_model:
            return
            
        checkpoint_path = Path(checkpoint_path)
        
        artifact = wandb.Artifact(
            name=f"model-{self.run.id}",
            type="model",
            metadata=metadata or {},
        )
        
        if checkpoint_path.is_dir():
            artifact.add_dir(str(checkpoint_path))
        else:
            artifact.add_file(str(checkpoint_path))
            
        self.run.log_artifact(artifact)
        
    def log_table(
        self,
        table_name: str,
        columns: List[str],
        data: List[List[Any]],
    ) -> None:
        """
        Log a table to W&B.
        
        Args:
            table_name: Name of the table
            columns: Column names
            data: Table data as list of rows
        """
        if not self._initialized or self.run is None:
            return
            
        table = wandb.Table(columns=columns, data=data)
        self.run.log({table_name: table}, step=self.step)
        
    def log_histogram(
        self,
        name: str,
        values: Union[torch.Tensor, List[float]],
    ) -> None:
        """
        Log a histogram to W&B.
        
        Args:
            name: Name of the histogram
            values: Values to histogram
        """
        if not self._initialized or self.run is None:
            return
            
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
            
        self.run.log({name: wandb.Histogram(values)}, step=self.step)
        
    def _check_loss_spike(self, loss: float) -> None:
        """Check for loss spike and send alert if detected."""
        if len(self.loss_history) < self.config.loss_spike_window:
            return
            
        mean_loss = sum(self.loss_history[:-1]) / (len(self.loss_history) - 1)
        
        if loss > mean_loss * self.config.loss_spike_threshold:
            if WANDB_AVAILABLE and self.run is not None:
                wandb.alert(
                    title="Loss Spike Detected",
                    text=f"Loss spiked to {loss:.4f} (mean: {mean_loss:.4f}) at step {self.step}",
                    level=AlertLevel.WARN,
                )
                
    def _check_nan(self, loss: float) -> None:
        """Check for NaN loss and send alert."""
        if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
            if WANDB_AVAILABLE and self.run is not None:
                wandb.alert(
                    title="NaN/Inf Loss Detected",
                    text=f"Loss is {loss} at step {self.step}",
                    level=AlertLevel.ERROR,
                )
                
    def watch_model(
        self,
        model: nn.Module,
        log: str = "gradients",
        log_freq: int = 100,
    ) -> None:
        """
        Watch model for gradient/parameter logging.
        
        Args:
            model: PyTorch model to watch
            log: What to log ("gradients", "parameters", "all")
            log_freq: Logging frequency
        """
        if not self._initialized or self.run is None:
            return
            
        wandb.watch(model, log=log, log_freq=log_freq)

    def log_model_architecture(
        self,
        model: nn.Module,
        input_shape: tuple = (1, 128),
    ) -> None:
        """
        Log model architecture visualization to W&B.
        
        Args:
            model: PyTorch model
            input_shape: Shape of input tensor (batch_size, seq_len)
        """
        if not self._initialized or self.run is None:
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Build architecture summary
        arch_summary = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_size_gb": total_params * 4 / 1e9,  # Assuming float32
        }
        
        # Get layer info
        layer_info = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    layer_info.append({
                        "name": name,
                        "type": type(module).__name__,
                        "parameters": params,
                    })
        
        # Log architecture table
        if layer_info:
            columns = ["name", "type", "parameters"]
            data = [[l["name"], l["type"], l["parameters"]] for l in layer_info]
            table = wandb.Table(columns=columns, data=data)
            self.run.log({"model/architecture": table})
        
        # Log summary config
        self.run.summary.update(arch_summary)

    def finish(self) -> None:
        """Finish W&B run."""
        if self._initialized and self.run is not None:
            # Log final summary
            total_time = time.time() - self.start_time
            self.run.summary["total_training_time"] = total_time
            self.run.summary["total_steps"] = self.step
            
            self.run.finish()
            self._initialized = False


# =============================================================================
# W&B Sweeps Support
# =============================================================================

def create_sweep_config(
    method: str = "bayes",
    metric_name: str = "val/loss",
    metric_goal: str = "minimize",
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a W&B sweep configuration.
    
    Args:
        method: Search method ("grid", "random", "bayes")
        metric_name: Metric to optimize
        metric_goal: Goal ("minimize" or "maximize")
        parameters: Parameter search space
        
    Returns:
        Sweep configuration dictionary
        
    Example:
        sweep_config = create_sweep_config(
            method="bayes",
            parameters={
                "learning_rate": {"min": 1e-5, "max": 1e-3, "distribution": "log_uniform"},
                "batch_size": {"values": [8, 16, 32]},
                "dropout": {"min": 0.0, "max": 0.5},
            }
        )
        sweep_id = wandb.sweep(sweep_config, project="my-project")
        wandb.agent(sweep_id, train_function)
    """
    if parameters is None:
        parameters = {
            "learning_rate": {
                "min": 1e-5,
                "max": 1e-3,
                "distribution": "log_uniform_values",
            },
            "batch_size": {"values": [4, 8, 16, 32]},
            "weight_decay": {"min": 0.0, "max": 0.1},
            "warmup_steps": {"min": 100, "max": 2000},
        }
    
    return {
        "method": method,
        "metric": {
            "name": metric_name,
            "goal": metric_goal,
        },
        "parameters": parameters,
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 100,
            "eta": 3,
        },
    }


def run_sweep(
    sweep_config: Dict[str, Any],
    train_fn,
    project: str,
    count: int = 10,
    entity: Optional[str] = None,
) -> str:
    """
    Run a W&B sweep.
    
    Args:
        sweep_config: Sweep configuration
        train_fn: Training function that reads config from wandb.config
        project: W&B project name
        count: Number of sweep runs
        entity: W&B entity (optional)
        
    Returns:
        Sweep ID
    """
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is required for sweeps")
    
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    wandb.agent(sweep_id, train_fn, count=count)
    return sweep_id


def get_wandb_tracker(config: Dict[str, Any]) -> WandbTracker:
    """
    Create W&B tracker from config dictionary.
    
    Args:
        config: Configuration dictionary with 'wandb' key
        
    Returns:
        Initialized WandbTracker
    """
    wandb_config = config.get("wandb", {})
    return WandbTracker(
        config=wandb_config,
        hydra_config=config,
    )


# Utility functions for getting GPU memory stats
def get_gpu_memory_stats() -> Dict[str, float]:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {}
        
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def compute_mbu(
    model_size_bytes: int,
    step_time: float,
    gradient_accumulation_steps: int = 1,
) -> float:
    """
    Compute Model Bandwidth Utilization.
    
    MBU = (model_size * 2 * gradient_accumulation_steps) / (step_time * memory_bandwidth)
    
    Args:
        model_size_bytes: Model size in bytes
        step_time: Time per step in seconds
        gradient_accumulation_steps: Number of gradient accumulation steps
        
    Returns:
        MBU as a fraction (0-1)
    """
    # Assume A100 bandwidth: 2039 GB/s
    memory_bandwidth = 2039 * 1e9  # bytes/sec
    
    # 2x for forward and backward pass
    bytes_moved = model_size_bytes * 2 * gradient_accumulation_steps
    
    mbu = bytes_moved / (step_time * memory_bandwidth)
    return min(mbu, 1.0)  # Cap at 1.0
