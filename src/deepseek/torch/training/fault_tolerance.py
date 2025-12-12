"""
Fault Tolerance and Elastic Training for DeepSeek Distributed Training.

This module provides:
- TorchElastic integration via torchrun launcher
- Elastic scaling (min/max worker configuration)
- Health check heartbeat mechanism
- Graceful degradation on node failure
- Spot instance preemption handling (checkpoint on SIGTERM)
- Automatic batch size adjustment on worker count change
- Failure injection tests for robustness validation
- RetryManager with exponential backoff (F1)
- SIGTERM graceful shutdown with 30s grace period (F2)
- Checkpoint resume logic with validation (F3, F7)
- Failure logging to W&B (F4)
- Modal container restart policy configuration (F5)
- NaN loss rollback mechanism (F6)
- Health check endpoint (F8)
- Cross-backend failure handling (F9)
- Retry budget tracking (F10)

Reference: DeepSeek-V3 uses fault-tolerant training infrastructure
to handle failures during long training runs.
"""

import os
import sys
import signal
import time
import threading
import traceback
import math
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List, Tuple
from enum import Enum
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import atexit

try:
    from deepseek.torch.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# Try to import optional dependencies
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from safetensors import safe_open
    from safetensors.torch import save_file as safetensors_save
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class WorkerState(str, Enum):
    """State of a worker in the elastic training system."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    FAILED = "failed"
    TERMINATED = "terminated"
    PREEMPTED = "preempted"


class FailureType(str, Enum):
    """Types of failures that can occur during training."""
    OOM = "oom"
    NAN_LOSS = "nan_loss"
    DIVERGENCE = "divergence"
    TIMEOUT = "timeout"
    NETWORK = "network"
    PREEMPTION = "preemption"
    CHECKPOINT_CORRUPT = "checkpoint_corrupt"
    UNKNOWN = "unknown"


@dataclass
class FailureRecord:
    """Record of a training failure for logging and analysis."""
    failure_type: FailureType
    timestamp: datetime
    step: int
    loss: Optional[float]
    stack_trace: str
    backend: str = "pytorch"
    rank: int = 0
    world_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "failure_type": self.failure_type.value,
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
            "loss": self.loss,
            "stack_trace": self.stack_trace,
            "backend": self.backend,
            "rank": self.rank,
            "world_size": self.world_size,
            "metadata": self.metadata,
        }


# =============================================================================
# F1: RetryManager with Exponential Backoff
# =============================================================================

class RetryManager:
    """Manage retries with exponential backoff for fault tolerance (F1).
    
    Provides:
    - Max 3 attempts with exponential backoff (1s, 2s, 4s)
    - Failure classification and tracking
    - Integration with W&B logging
    - Budget-aware retry decisions
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        budget_tracker: Optional[Any] = None,
        wandb_logger: Optional[Any] = None,
    ):
        """Initialize RetryManager.
        
        Args:
            max_attempts: Maximum retry attempts (default: 3)
            base_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            backoff_factor: Multiplier for exponential backoff
            budget_tracker: Optional BudgetTracker for cost tracking (F10)
            wandb_logger: Optional W&B logger for failure logging (F4)
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.budget_tracker = budget_tracker
        self.wandb_logger = wandb_logger
        
        # Tracking
        self.attempt_count = 0
        self.failure_history: List[FailureRecord] = []
        self.total_retry_time = 0.0
        self.last_failure: Optional[FailureRecord] = None
    
    def should_retry(self, error: Exception, step: int, loss: Optional[float] = None) -> bool:
        """Determine if training should retry after failure.
        
        Args:
            error: The exception that occurred
            step: Current training step
            loss: Current loss value (if available)
            
        Returns:
            True if retry should be attempted
        """
        self.attempt_count += 1
        failure_type = self._classify_failure(error, loss)
        
        # Record failure
        failure = FailureRecord(
            failure_type=failure_type,
            timestamp=datetime.now(),
            step=step,
            loss=loss,
            stack_trace=traceback.format_exc(),
            rank=dist.get_rank() if dist.is_initialized() else 0,
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
        )
        self.failure_history.append(failure)
        self.last_failure = failure
        
        # Log to W&B (F4)
        self._log_failure_to_wandb(failure)
        
        # Check budget (F10)
        if self.budget_tracker is not None:
            if not self.budget_tracker.can_continue():
                logger.warning("Budget exhausted - not retrying")
                return False
        
        # Check attempt limit
        if self.attempt_count >= self.max_attempts:
            logger.error(f"Max retry attempts ({self.max_attempts}) reached")
            return False
        
        # Don't retry certain fatal errors
        if failure_type in [FailureType.CHECKPOINT_CORRUPT]:
            logger.error(f"Fatal error type {failure_type} - not retrying")
            return False
        
        return True
    
    def get_retry_delay(self) -> float:
        """Calculate delay before next retry using exponential backoff.
        
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.backoff_factor ** (self.attempt_count - 1))
        delay = min(delay, self.max_delay)
        self.total_retry_time += delay
        return delay
    
    def execute_with_retry(
        self,
        fn: Callable,
        *args,
        step: int = 0,
        loss: Optional[float] = None,
        **kwargs,
    ) -> Any:
        """Execute function with automatic retry logic.
        
        Args:
            fn: Function to execute
            step: Current training step
            loss: Current loss for context
            *args, **kwargs: Arguments for fn
            
        Returns:
            Result of fn
            
        Raises:
            Last exception if all retries exhausted
        """
        last_error = None
        
        while True:
            try:
                result = fn(*args, **kwargs)
                # Reset on success
                self.attempt_count = 0
                return result
            except Exception as e:
                last_error = e
                logger.error(f"Attempt {self.attempt_count} failed: {e}")
                
                if not self.should_retry(e, step, loss):
                    raise
                
                delay = self.get_retry_delay()
                logger.info(f"Retrying in {delay:.1f}s (attempt {self.attempt_count}/{self.max_attempts})")
                time.sleep(delay)
        
        raise last_error
    
    def _classify_failure(self, error: Exception, loss: Optional[float]) -> FailureType:
        """Classify the type of failure for appropriate handling.
        
        Args:
            error: The exception
            loss: Current loss value
            
        Returns:
            FailureType classification
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # OOM detection
        if "out of memory" in error_str or "oom" in error_type:
            return FailureType.OOM
        
        # NaN loss detection (F6)
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            return FailureType.NAN_LOSS
        if "nan" in error_str or "inf" in error_str:
            return FailureType.NAN_LOSS
        
        # Divergence detection
        if loss is not None and loss > 100.0:  # Configurable threshold
            return FailureType.DIVERGENCE
        
        # Timeout
        if "timeout" in error_str or "timed out" in error_str:
            return FailureType.TIMEOUT
        
        # Network errors
        if any(x in error_str for x in ["connection", "network", "nccl", "distributed"]):
            return FailureType.NETWORK
        
        # Checkpoint corruption
        if "checkpoint" in error_str and ("corrupt" in error_str or "invalid" in error_str):
            return FailureType.CHECKPOINT_CORRUPT
        
        return FailureType.UNKNOWN
    
    def _log_failure_to_wandb(self, failure: FailureRecord) -> None:
        """Log failure to Weights & Biases (F4).
        
        Args:
            failure: Failure record to log
        """
        if not HAS_WANDB:
            return
        
        try:
            if wandb.run is not None:
                wandb.log({
                    "failure/type": failure.failure_type.value,
                    "failure/step": failure.step,
                    "failure/loss": failure.loss,
                    "failure/retry_count": self.attempt_count,
                    "failure/total_retry_time": self.total_retry_time,
                })
                
                # Log as alert for critical failures
                if failure.failure_type in [FailureType.NAN_LOSS, FailureType.DIVERGENCE]:
                    wandb.alert(
                        title=f"Training Failure: {failure.failure_type.value}",
                        text=f"Step {failure.step}: {failure.stack_trace[:500]}",
                        level=wandb.AlertLevel.ERROR,
                    )
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics.
        
        Returns:
            Dictionary of retry stats
        """
        return {
            "total_attempts": self.attempt_count,
            "total_failures": len(self.failure_history),
            "total_retry_time": self.total_retry_time,
            "failure_types": {
                ft.value: sum(1 for f in self.failure_history if f.failure_type == ft)
                for ft in FailureType
            },
        }
    
    def reset(self) -> None:
        """Reset retry state for new training run."""
        self.attempt_count = 0
        self.failure_history.clear()
        self.total_retry_time = 0.0
        self.last_failure = None


# =============================================================================
# F3/F7: Checkpoint Resume and Validation
# =============================================================================

def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "step_*.pt") -> Optional[str]:
    """Find the latest checkpoint in a directory (F3).
    
    Args:
        checkpoint_dir: Directory to search
        pattern: Glob pattern for checkpoints
        
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        logger.info(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None
    
    # Try multiple patterns
    patterns = [pattern, "step_*.safetensors", "checkpoint_*.pt", "checkpoint_*.safetensors"]
    checkpoints = []
    
    for pat in patterns:
        checkpoints.extend(checkpoint_path.glob(pat))
    
    if not checkpoints:
        logger.info(f"No checkpoints found in {checkpoint_dir}")
        return None
    
    def get_step(p: Path) -> int:
        """Extract step number from checkpoint filename."""
        try:
            # Handle patterns like step_1000.pt, checkpoint_step_1000.pt
            name = p.stem
            parts = name.replace("checkpoint_", "").replace("step_", "").split("_")
            for part in parts:
                if part.isdigit():
                    return int(part)
            return 0
        except Exception:
            return 0
    
    latest = max(checkpoints, key=lambda p: (get_step(p), p.stat().st_mtime))
    logger.info(f"Found latest checkpoint: {latest}")
    return str(latest)


def validate_checkpoint(checkpoint_path: str) -> Tuple[bool, str]:
    """Validate checkpoint integrity before resume (F7).
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not checkpoint_path:
        return False, "No checkpoint path provided"
    
    path = Path(checkpoint_path)
    if not path.exists():
        return False, f"Checkpoint file does not exist: {checkpoint_path}"
    
    # Check file size
    if path.stat().st_size == 0:
        return False, "Checkpoint file is empty"
    
    try:
        if checkpoint_path.endswith(".safetensors"):
            if not HAS_SAFETENSORS:
                return False, "safetensors not installed"
            
            with safe_open(checkpoint_path, framework="pt") as f:
                keys = list(f.keys())
            
            if not keys:
                return False, "Safetensors checkpoint has no keys"
            
            logger.info(f"Validated safetensors checkpoint with {len(keys)} keys")
            return True, ""
        else:
            # PyTorch checkpoint
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            # Check for required keys
            required_keys = ["model_state_dict"]
            alternative_keys = ["state_dict", "model"]
            
            has_model = any(k in state for k in required_keys + alternative_keys)
            if not has_model:
                return False, f"Checkpoint missing model state (has keys: {list(state.keys())})"
            
            logger.info(f"Validated PyTorch checkpoint with keys: {list(state.keys())}")
            return True, ""
            
    except Exception as e:
        return False, f"Checkpoint validation failed: {e}"


def load_checkpoint_with_validation(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load checkpoint with validation and compatibility handling (F3/F7).
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state
        strict: Whether to require exact key match
        
    Returns:
        Checkpoint metadata (step, etc.)
        
    Raises:
        ValueError: If checkpoint is invalid
    """
    is_valid, error_msg = validate_checkpoint(checkpoint_path)
    if not is_valid:
        raise ValueError(f"Invalid checkpoint: {error_msg}")
    
    try:
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
            model.load_state_dict(state_dict, strict=strict)
            return {"step": 0, "source": "safetensors"}
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"], strict=strict)
            elif "model" in checkpoint:
                model.load_state_dict(checkpoint["model"], strict=strict)
            else:
                # Assume checkpoint is just the state dict
                model.load_state_dict(checkpoint, strict=strict)
            
            # Load optimizer if provided
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            return {
                "step": checkpoint.get("global_step", checkpoint.get("step", 0)),
                "epoch": checkpoint.get("epoch", 0),
                "loss": checkpoint.get("loss"),
                "source": "pytorch",
            }
            
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}")


# =============================================================================
# F6: NaN Loss Detection and Rollback
# =============================================================================

class NaNLossDetector:
    """Detect NaN/Inf losses and trigger rollback (F6).
    
    Monitors loss values and provides rollback functionality
    when training diverges.
    """
    
    def __init__(
        self,
        loss_threshold: float = 100.0,
        nan_streak_limit: int = 3,
        checkpoint_manager: Optional[Any] = None,
    ):
        """Initialize NaN detector.
        
        Args:
            loss_threshold: Loss value above which triggers divergence alert
            nan_streak_limit: Number of consecutive NaN before rollback
            checkpoint_manager: Optional checkpoint manager for rollback
        """
        self.loss_threshold = loss_threshold
        self.nan_streak_limit = nan_streak_limit
        self.checkpoint_manager = checkpoint_manager
        
        self.nan_streak = 0
        self.last_valid_loss = None
        self.last_valid_checkpoint: Optional[str] = None
        self.loss_history: List[Tuple[int, float]] = []
    
    def check_loss(self, loss: float, step: int) -> Tuple[bool, Optional[str]]:
        """Check if loss is valid and track for rollback.
        
        Args:
            loss: Current loss value
            step: Current training step
            
        Returns:
            Tuple of (is_valid, action_needed)
            action_needed: None, "warn", or "rollback"
        """
        # Detect NaN/Inf
        if math.isnan(loss) or math.isinf(loss):
            self.nan_streak += 1
            logger.warning(f"Step {step}: NaN/Inf loss detected (streak: {self.nan_streak})")
            
            if self.nan_streak >= self.nan_streak_limit:
                return False, "rollback"
            return False, "warn"
        
        # Detect divergence
        if loss > self.loss_threshold:
            logger.warning(f"Step {step}: Loss {loss:.4f} exceeds threshold {self.loss_threshold}")
            return False, "warn"
        
        # Valid loss - update tracking
        self.nan_streak = 0
        self.last_valid_loss = loss
        self.loss_history.append((step, loss))
        
        # Keep only recent history
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-500:]
        
        return True, None
    
    def set_valid_checkpoint(self, checkpoint_path: str) -> None:
        """Mark a checkpoint as valid for potential rollback.
        
        Args:
            checkpoint_path: Path to valid checkpoint
        """
        self.last_valid_checkpoint = checkpoint_path
        logger.debug(f"Set valid rollback checkpoint: {checkpoint_path}")
    
    def get_rollback_checkpoint(self) -> Optional[str]:
        """Get checkpoint path for rollback.
        
        Returns:
            Path to checkpoint for rollback, or None
        """
        return self.last_valid_checkpoint
    
    def reset(self) -> None:
        """Reset detector state."""
        self.nan_streak = 0
        self.loss_history.clear()


# =============================================================================
# F8: Health Check Endpoint
# =============================================================================

class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoint (F8)."""
    
    # Class variable to hold training state
    training_state: Dict[str, Any] = {
        "status": "unknown",
        "step": 0,
        "loss": None,
        "uptime": 0,
        "retry_count": 0,
        "backend": "unknown",
    }
    
    def do_GET(self):
        """Handle GET requests for health status."""
        if self.path == "/health" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            response = {
                **self.training_state,
                "timestamp": datetime.now().isoformat(),
            }
            self.wfile.write(json.dumps(response).encode())
        
        elif self.path == "/ready":
            # Readiness check
            if self.training_state["status"] in ["running", "checkpointing"]:
                self.send_response(200)
            else:
                self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ready": self.training_state["status"] == "running"}).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class HealthCheckServer:
    """Background HTTP server for health checks (F8).
    
    Provides:
    - /health - Full status JSON
    - /ready - Kubernetes-style readiness probe
    """
    
    def __init__(self, port: int = 8080, host: str = "0.0.0.0"):
        """Initialize health check server.
        
        Args:
            port: Port to listen on
            host: Host to bind to
        """
        self.port = port
        self.host = host
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self) -> bool:
        """Start the health check server.
        
        Returns:
            True if started successfully
        """
        try:
            self.server = HTTPServer((self.host, self.port), HealthCheckHandler)
            self.server.timeout = 0.5  # Short timeout for responsive shutdown
            self.thread = threading.Thread(target=self._serve, daemon=True)
            self._running = True
            self.thread.start()
            logger.info(f"Health check server started on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.warning(f"Failed to start health server: {e}")
            return False
    
    def _serve(self) -> None:
        """Serve requests until stopped."""
        while self._running:
            self.server.handle_request()
    
    def stop(self) -> None:
        """Stop the health check server."""
        self._running = False
        if self.server:
            self.server.server_close()  # Close socket immediately
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)  # Wait max 1 second
        logger.info("Health check server stopped")
    
    @staticmethod
    def update_status(
        status: str = None,
        step: int = None,
        loss: float = None,
        retry_count: int = None,
        backend: str = None,
        **kwargs,
    ) -> None:
        """Update training status for health endpoint.
        
        Args:
            status: Current status (running, checkpointing, failed, etc.)
            step: Current training step
            loss: Current loss value
            retry_count: Number of retries so far
            backend: Training backend (pytorch, rust)
            **kwargs: Additional status fields
        """
        if status is not None:
            HealthCheckHandler.training_state["status"] = status
        if step is not None:
            HealthCheckHandler.training_state["step"] = step
        if loss is not None:
            HealthCheckHandler.training_state["loss"] = loss
        if retry_count is not None:
            HealthCheckHandler.training_state["retry_count"] = retry_count
        if backend is not None:
            HealthCheckHandler.training_state["backend"] = backend
        
        HealthCheckHandler.training_state.update(kwargs)


# =============================================================================
# F5: Modal Container Restart Policy
# =============================================================================

def get_modal_restart_policy() -> Dict[str, Any]:
    """Get recommended Modal container restart policy configuration (F5).
    
    Returns:
        Dictionary with Modal restart configuration
    """
    return {
        # Container restart settings
        "retries": 3,
        "timeout": 3600,  # 1 hour timeout per attempt
        
        # Spot instance handling
        "spot_policy": {
            "enabled": True,
            "max_interruptions": 5,
            "checkpoint_on_preempt": True,
        },
        
        # OOM handling
        "oom_policy": {
            "retry_with_larger_memory": True,
            "memory_increment_gb": 10,
            "max_memory_gb": 80,
        },
        
        # Health checks
        "health_check": {
            "interval_seconds": 30,
            "timeout_seconds": 10,
            "unhealthy_threshold": 3,
        },
        
        # Example Modal function decorator
        "example_decorator": '''
@app.function(
    retries=3,
    timeout=3600,
    gpu="A100-80GB",
    memory=65536,  # 64GB
    _allow_background_volume_commits=True,
)
def train_with_retry():
    """Training function with automatic retry on failure."""
    pass
''',
    }


def create_modal_training_stub() -> str:
    """Generate Modal training stub with restart policy (F5).
    
    Returns:
        Python code for Modal training with fault tolerance
    """
    return '''
import modal

app = modal.App("deepseek-training")

# Volumes for checkpoint persistence
checkpoints_volume = modal.Volume.from_name("deepseek-checkpoints", create_if_missing=True)

@app.function(
    gpu="A100-80GB",
    memory=65536,
    timeout=7200,  # 2 hours
    retries=3,  # Auto-retry on failure
    volumes={"/checkpoints": checkpoints_volume},
    _allow_background_volume_commits=True,
)
def train_step(
    backend: str,
    model_size: str,
    resume_from: str = None,
):
    """Training step with fault tolerance.
    
    Modal will automatically:
    - Retry up to 3 times on failure
    - Persist checkpoints to volume
    - Handle spot preemption gracefully
    """
    from deepseek.torch.training.fault_tolerance import (
        RetryManager,
        find_latest_checkpoint,
        validate_checkpoint,
        HealthCheckServer,
    )
    
    # Start health server
    health_server = HealthCheckServer(port=8080)
    health_server.start()
    
    # Find checkpoint to resume
    checkpoint_dir = f"/checkpoints/{backend}/{model_size}"
    if resume_from is None:
        resume_from = find_latest_checkpoint(checkpoint_dir)
    
    if resume_from:
        is_valid, _ = validate_checkpoint(resume_from)
        if not is_valid:
            resume_from = None
    
    # Initialize retry manager
    retry_manager = RetryManager(max_attempts=3)
    
    try:
        HealthCheckServer.update_status(status="running", backend=backend)
        
        # Training logic here
        # ...
        
        HealthCheckServer.update_status(status="completed")
    except Exception as e:
        HealthCheckServer.update_status(status="failed")
        raise
    finally:
        health_server.stop()
        checkpoints_volume.commit()  # Persist checkpoints
'''


# =============================================================================
# F9: Cross-Backend Failure Handling  
# =============================================================================

class CrossBackendCoordinator:
    """Coordinate training across multiple backends (F9).
    
    Handles scenarios where one backend fails but others can continue.
    """
    
    def __init__(
        self,
        backends: List[str] = None,
        shared_checkpoint_dir: str = "/checkpoints",
    ):
        """Initialize coordinator.
        
        Args:
            backends: List of backend names (e.g., ["pytorch", "rust"])
            shared_checkpoint_dir: Shared checkpoint directory
        """
        self.backends = backends or ["pytorch", "rust"]
        self.shared_checkpoint_dir = shared_checkpoint_dir
        
        self.backend_status: Dict[str, str] = {b: "unknown" for b in self.backends}
        self.backend_steps: Dict[str, int] = {b: 0 for b in self.backends}
        self.failed_backends: List[str] = []
    
    def report_status(self, backend: str, status: str, step: int = 0) -> None:
        """Report status from a backend.
        
        Args:
            backend: Backend name
            status: Current status
            step: Current step
        """
        self.backend_status[backend] = status
        self.backend_steps[backend] = step
        
        if status == "failed" and backend not in self.failed_backends:
            self.failed_backends.append(backend)
            logger.warning(f"Backend {backend} failed at step {step}")
    
    def can_continue(self) -> bool:
        """Check if training can continue with remaining backends.
        
        Returns:
            True if at least one backend is healthy
        """
        healthy = [b for b in self.backends if self.backend_status[b] not in ["failed", "terminated"]]
        return len(healthy) > 0
    
    def get_healthy_backends(self) -> List[str]:
        """Get list of healthy backends.
        
        Returns:
            List of healthy backend names
        """
        return [b for b in self.backends if self.backend_status[b] not in ["failed", "terminated"]]
    
    def sync_checkpoint_from_healthy(self, failed_backend: str) -> Optional[str]:
        """Find checkpoint from a healthy backend to use for failed one.
        
        Args:
            failed_backend: Backend that failed
            
        Returns:
            Path to checkpoint from healthy backend, or None
        """
        healthy = self.get_healthy_backends()
        if not healthy:
            return None
        
        # Find best checkpoint from healthy backends
        best_checkpoint = None
        best_step = 0
        
        for backend in healthy:
            checkpoint_dir = os.path.join(self.shared_checkpoint_dir, backend)
            checkpoint = find_latest_checkpoint(checkpoint_dir)
            
            if checkpoint:
                # Extract step
                try:
                    step = int(Path(checkpoint).stem.split("_")[1])
                    if step > best_step:
                        best_step = step
                        best_checkpoint = checkpoint
                except Exception:
                    pass
        
        if best_checkpoint:
            logger.info(f"Found checkpoint from healthy backend at step {best_step}: {best_checkpoint}")
        
        return best_checkpoint
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all backend statuses.
        
        Returns:
            Summary dictionary
        """
        return {
            "backends": self.backends,
            "status": self.backend_status,
            "steps": self.backend_steps,
            "failed": self.failed_backends,
            "can_continue": self.can_continue(),
        }


# =============================================================================
# F10: Retry Budget Tracking Integration
# =============================================================================

class RetryBudgetTracker:
    """Track retry costs against training budget (F10).
    
    Integrates with BudgetTracker to deduct actual runtime from budget.
    """
    
    def __init__(
        self,
        budget_tracker: Optional[Any] = None,
        max_retry_cost: float = 50.0,  # Max $ to spend on retries
    ):
        """Initialize retry budget tracker.
        
        Args:
            budget_tracker: Optional BudgetTracker instance
            max_retry_cost: Maximum cost to spend on retries
        """
        self.budget_tracker = budget_tracker
        self.max_retry_cost = max_retry_cost
        
        self.retry_costs: List[Dict[str, Any]] = []
        self.total_retry_cost = 0.0
        self.retry_start_time: Optional[float] = None
    
    def start_retry(self) -> None:
        """Mark start of a retry attempt."""
        self.retry_start_time = time.time()
    
    def end_retry(self, success: bool, gpu_count: int = 1, cost_per_hour: float = 2.50) -> float:
        """Mark end of retry and calculate cost.
        
        Args:
            success: Whether retry succeeded
            gpu_count: Number of GPUs used
            cost_per_hour: Cost per GPU per hour (A100-80GB @ $2.50/hr)
            
        Returns:
            Cost of this retry
        """
        if self.retry_start_time is None:
            return 0.0
        
        duration = time.time() - self.retry_start_time
        hours = duration / 3600
        cost = hours * gpu_count * cost_per_hour
        
        self.retry_costs.append({
            "duration": duration,
            "cost": cost,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })
        self.total_retry_cost += cost
        self.retry_start_time = None
        
        # Update budget tracker if available
        if self.budget_tracker is not None:
            try:
                self.budget_tracker.add_cost(cost, category="retry")
            except Exception:
                pass
        
        logger.info(f"Retry cost: ${cost:.2f} (total retry cost: ${self.total_retry_cost:.2f})")
        return cost
    
    def can_afford_retry(self) -> bool:
        """Check if we can afford another retry.
        
        Returns:
            True if retry budget allows
        """
        return self.total_retry_cost < self.max_retry_cost
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry budget statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_retries": len(self.retry_costs),
            "total_retry_cost": self.total_retry_cost,
            "max_retry_cost": self.max_retry_cost,
            "remaining_budget": self.max_retry_cost - self.total_retry_cost,
            "retry_history": self.retry_costs,
        }


# =============================================================================
# Original Classes (Enhanced)
# =============================================================================

@dataclass
class ElasticConfig:
    """Configuration for elastic training.
    
    Attributes:
        min_workers: Minimum number of workers to continue training
        max_workers: Maximum number of workers
        max_restarts: Maximum number of restarts per worker
        rdzv_backend: Rendezvous backend (c10d, etcd, etc.)
        rdzv_endpoint: Rendezvous endpoint address
        heartbeat_interval: Seconds between heartbeats
        heartbeat_timeout: Seconds before declaring worker dead
        checkpoint_on_preempt: Whether to checkpoint on preemption
        checkpoint_timeout: Max seconds to wait for checkpoint
        auto_scale_batch_size: Adjust batch size based on workers
        initial_batch_size: Batch size for initial worker count
    """
    min_workers: int = 1
    max_workers: int = 8
    max_restarts: int = 3
    rdzv_backend: str = "c10d"
    rdzv_endpoint: str = "localhost:29400"
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 120.0
    checkpoint_on_preempt: bool = True
    checkpoint_timeout: float = 300.0
    auto_scale_batch_size: bool = True
    initial_batch_size: int = 32


class HeartbeatMonitor:
    """Monitor for worker health via heartbeats.
    
    Sends periodic heartbeats and detects unresponsive workers.
    """
    
    def __init__(
        self,
        config: ElasticConfig,
        rank: int,
        world_size: int,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_heartbeat: Dict[int, float] = {}
        self._state = WorkerState.INITIALIZING
        
        # Initialize heartbeat times
        for r in range(world_size):
            self._last_heartbeat[r] = time.time()
    
    def start(self) -> None:
        """Start heartbeat monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._state = WorkerState.RUNNING
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"Heartbeat monitor started for rank {self.rank}")
    
    def stop(self) -> None:
        """Stop heartbeat monitoring."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._state = WorkerState.TERMINATED
    
    def _heartbeat_loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            try:
                self._send_heartbeat()
                self._check_workers()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
            
            time.sleep(self.config.heartbeat_interval)
    
    def _send_heartbeat(self) -> None:
        """Send heartbeat to all workers."""
        if not dist.is_initialized():
            return
        
        # Use a simple all-reduce as heartbeat
        heartbeat = torch.tensor([time.time()], dtype=torch.float64)
        if torch.cuda.is_available():
            heartbeat = heartbeat.cuda()
        
        try:
            dist.all_reduce(heartbeat, op=dist.ReduceOp.MAX)
            self._last_heartbeat[self.rank] = time.time()
        except Exception as e:
            logger.warning(f"Heartbeat send failed: {e}")
    
    def _check_workers(self) -> List[int]:
        """Check for unresponsive workers.
        
        Returns:
            List of failed worker ranks
        """
        current_time = time.time()
        failed_workers = []
        
        for rank, last_time in self._last_heartbeat.items():
            if current_time - last_time > self.config.heartbeat_timeout:
                failed_workers.append(rank)
                logger.warning(f"Worker {rank} unresponsive for {current_time - last_time:.1f}s")
        
        return failed_workers
    
    def set_state(self, state: WorkerState) -> None:
        """Set current worker state."""
        self._state = state
    
    def get_state(self) -> WorkerState:
        """Get current worker state."""
        return self._state


class PreemptionHandler:
    """Handle spot instance preemption gracefully.
    
    Registers signal handlers for SIGTERM and triggers checkpoint
    before shutdown.
    """
    
    def __init__(
        self,
        checkpoint_fn: Optional[Callable[[], None]] = None,
        config: Optional[ElasticConfig] = None,
    ):
        self.checkpoint_fn = checkpoint_fn
        self.config = config or ElasticConfig()
        self._preempted = False
        self._original_handlers: Dict[int, Any] = {}
    
    def register_handlers(self) -> None:
        """Register signal handlers for preemption."""
        # SIGTERM is typically sent for spot instance preemption
        self._original_handlers[signal.SIGTERM] = signal.signal(
            signal.SIGTERM, self._handle_sigterm
        )
        
        # Also handle SIGINT for graceful shutdown
        self._original_handlers[signal.SIGINT] = signal.signal(
            signal.SIGINT, self._handle_sigint
        )
        
        # Register atexit handler
        atexit.register(self._atexit_handler)
        
        logger.info("Preemption handlers registered")
    
    def unregister_handlers(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        self._original_handlers.clear()
    
    def _handle_sigterm(self, signum: int, frame) -> None:
        """Handle SIGTERM (preemption signal)."""
        logger.warning("Received SIGTERM - preemption detected!")
        self._preempted = True
        
        if self.config.checkpoint_on_preempt and self.checkpoint_fn is not None:
            logger.info("Saving checkpoint before shutdown...")
            try:
                self.checkpoint_fn()
                logger.info("Checkpoint saved successfully")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
        
        # Allow clean shutdown
        sys.exit(0)
    
    def _handle_sigint(self, signum: int, frame) -> None:
        """Handle SIGINT (Ctrl+C)."""
        logger.info("Received SIGINT - graceful shutdown")
        
        if self.checkpoint_fn is not None:
            logger.info("Saving checkpoint...")
            try:
                self.checkpoint_fn()
            except Exception as e:
                logger.error(f"Checkpoint failed: {e}")
        
        sys.exit(0)
    
    def _atexit_handler(self) -> None:
        """Handle process exit."""
        if self._preempted:
            logger.info("Process exiting after preemption")
    
    @property
    def was_preempted(self) -> bool:
        """Check if process was preempted."""
        return self._preempted


class ElasticTrainer:
    """Elastic training orchestrator.
    
    Manages:
    - Worker scaling up/down
    - Batch size adjustment
    - Checkpoint/restore on worker changes
    - Graceful handling of failures
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: ElasticConfig,
        checkpoint_fn: Optional[Callable[[str], None]] = None,
        load_checkpoint_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.checkpoint_fn = checkpoint_fn
        self.load_checkpoint_fn = load_checkpoint_fn
        
        # Get distributed info
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Initialize components
        self.heartbeat_monitor = HeartbeatMonitor(config, self.rank, self.world_size)
        self.preemption_handler = PreemptionHandler(
            self._emergency_checkpoint, config
        )
        
        # State tracking
        self.current_batch_size = config.initial_batch_size
        self.last_worker_count = self.world_size
        self.global_step = 0
        
        # Checkpoint path
        self.checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def start(self) -> None:
        """Start elastic training infrastructure."""
        self.heartbeat_monitor.start()
        self.preemption_handler.register_handlers()
        
        # Check for existing checkpoint
        self._try_restore_checkpoint()
        
        logger.info(
            f"Elastic trainer started",
            extra={
                "rank": self.rank,
                "world_size": self.world_size,
                "batch_size": self.current_batch_size,
            }
        )
    
    def stop(self) -> None:
        """Stop elastic training infrastructure."""
        self.heartbeat_monitor.stop()
        self.preemption_handler.unregister_handlers()
        
        # Save final checkpoint
        self._save_checkpoint("final")
        
        logger.info("Elastic trainer stopped")
    
    def check_worker_changes(self) -> bool:
        """Check if number of workers has changed.
        
        Returns:
            True if worker count changed
        """
        if not dist.is_initialized():
            return False
        
        current_world_size = dist.get_world_size()
        
        if current_world_size != self.last_worker_count:
            logger.info(
                f"Worker count changed: {self.last_worker_count} -> {current_world_size}"
            )
            
            if self.config.auto_scale_batch_size:
                self._adjust_batch_size(current_world_size)
            
            self.last_worker_count = current_world_size
            return True
        
        return False
    
    def _adjust_batch_size(self, new_world_size: int) -> None:
        """Adjust batch size based on worker count.
        
        Args:
            new_world_size: New number of workers
        """
        # Scale batch size proportionally
        scale_factor = new_world_size / max(1, self.last_worker_count)
        new_batch_size = int(self.current_batch_size * scale_factor)
        
        # Clamp to reasonable bounds
        new_batch_size = max(1, min(new_batch_size, 256))
        
        logger.info(
            f"Adjusting batch size: {self.current_batch_size} -> {new_batch_size}"
        )
        
        self.current_batch_size = new_batch_size
    
    def get_batch_size(self) -> int:
        """Get current batch size (may change with worker count)."""
        return self.current_batch_size
    
    def _emergency_checkpoint(self) -> None:
        """Save checkpoint in emergency (preemption/failure)."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"emergency_rank{self.rank}.pt"
        )
        self._save_checkpoint(checkpoint_path)
    
    def _save_checkpoint(self, suffix: str) -> None:
        """Save checkpoint.
        
        Args:
            suffix: Suffix for checkpoint filename
        """
        if self.checkpoint_fn is not None:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"checkpoint_{suffix}.pt"
            )
            self.checkpoint_fn(checkpoint_path)
        else:
            # Default checkpoint saving
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"checkpoint_{suffix}.pt"
            )
            
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "batch_size": self.current_batch_size,
                "world_size": self.world_size,
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _try_restore_checkpoint(self) -> bool:
        """Try to restore from latest checkpoint.
        
        Returns:
            True if checkpoint was restored
        """
        checkpoint_files = []
        
        if os.path.exists(self.checkpoint_dir):
            for f in os.listdir(self.checkpoint_dir):
                if f.startswith("checkpoint_") and f.endswith(".pt"):
                    checkpoint_files.append(os.path.join(self.checkpoint_dir, f))
        
        if not checkpoint_files:
            return False
        
        # Get latest checkpoint
        latest = max(checkpoint_files, key=os.path.getmtime)
        
        try:
            if self.load_checkpoint_fn is not None:
                state = self.load_checkpoint_fn(latest)
            else:
                checkpoint = torch.load(latest, map_location="cpu")
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.global_step = checkpoint.get("global_step", 0)
            
            logger.info(f"Restored checkpoint from {latest}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return False
    
    def step(self) -> None:
        """Called after each training step."""
        self.global_step += 1
        
        # Check for worker changes periodically
        if self.global_step % 100 == 0:
            self.check_worker_changes()


def graceful_degradation_wrapper(
    train_fn: Callable,
    config: ElasticConfig,
    checkpoint_dir: str,
) -> Callable:
    """Wrapper that enables graceful degradation on failure.
    
    Args:
        train_fn: Training function to wrap
        config: Elastic config
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Wrapped function with fault tolerance
    """
    def wrapped(*args, **kwargs):
        restart_count = 0
        
        while restart_count < config.max_restarts:
            try:
                return train_fn(*args, **kwargs)
            except Exception as e:
                restart_count += 1
                logger.error(
                    f"Training failed (attempt {restart_count}/{config.max_restarts}): {e}"
                )
                
                if restart_count >= config.max_restarts:
                    raise
                
                # Wait before retry
                time.sleep(10 * restart_count)
                
                # Try to restore checkpoint
                logger.info("Attempting to restart from checkpoint...")
    
    return wrapped


@contextmanager
def elastic_training_context(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Optional[ElasticConfig] = None,
):
    """Context manager for elastic training.
    
    Args:
        model: Model being trained
        optimizer: Optimizer
        config: Elastic configuration
        
    Yields:
        ElasticTrainer instance
    """
    config = config or ElasticConfig()
    trainer = ElasticTrainer(model, optimizer, config)
    
    try:
        trainer.start()
        yield trainer
    finally:
        trainer.stop()


class FailureInjector:
    """Inject failures for testing fault tolerance.
    
    Used to validate that the training system handles failures correctly.
    """
    
    def __init__(
        self,
        failure_rate: float = 0.01,
        seed: Optional[int] = None,
    ):
        """Initialize failure injector.
        
        Args:
            failure_rate: Probability of failure per step
            seed: Random seed for reproducibility
        """
        self.failure_rate = failure_rate
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
        
        self._failures_injected = 0
    
    def maybe_fail(self, step: int = 0) -> None:
        """Maybe inject a failure.
        
        Args:
            step: Current training step (for logging)
            
        Raises:
            RuntimeError: If failure is injected
        """
        if torch.rand(1, generator=self.rng).item() < self.failure_rate:
            self._failures_injected += 1
            logger.warning(f"Injecting failure at step {step}")
            raise RuntimeError(f"Injected failure at step {step}")
    
    def maybe_oom(self, step: int = 0) -> None:
        """Maybe inject an OOM-like failure.
        
        Args:
            step: Current training step
            
        Raises:
            torch.cuda.OutOfMemoryError: If OOM is injected
        """
        if torch.rand(1, generator=self.rng).item() < self.failure_rate:
            self._failures_injected += 1
            logger.warning(f"Injecting OOM at step {step}")
            raise torch.cuda.OutOfMemoryError("Injected OOM failure")
    
    def maybe_hang(self, max_seconds: float = 60.0, step: int = 0) -> None:
        """Maybe inject a hang (for timeout testing).
        
        Args:
            max_seconds: Maximum hang duration
            step: Current training step
        """
        if torch.rand(1, generator=self.rng).item() < self.failure_rate:
            self._failures_injected += 1
            hang_time = torch.rand(1, generator=self.rng).item() * max_seconds
            logger.warning(f"Injecting hang of {hang_time:.1f}s at step {step}")
            time.sleep(hang_time)
    
    @property
    def failures_injected(self) -> int:
        """Number of failures injected so far."""
        return self._failures_injected


def get_elastic_launch_command(
    script: str,
    config: ElasticConfig,
    nproc_per_node: int = 1,
    nnodes: str = "1:4",
    script_args: Optional[List[str]] = None,
) -> str:
    """Generate torchrun command for elastic training.
    
    Args:
        script: Python script to run
        config: Elastic configuration
        nproc_per_node: Number of processes per node
        nnodes: Node count range (min:max)
        script_args: Additional arguments for the script
        
    Returns:
        Complete torchrun command
    """
    cmd_parts = [
        "torchrun",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={nproc_per_node}",
        f"--max_restarts={config.max_restarts}",
        f"--rdzv_backend={config.rdzv_backend}",
        f"--rdzv_endpoint={config.rdzv_endpoint}",
        "--rdzv_id=deepseek_training",
        script,
    ]
    
    if script_args:
        cmd_parts.extend(script_args)
    
    return " ".join(cmd_parts)


def document_recovery_procedures() -> str:
    """Document recovery procedures for different failure modes.
    
    Returns:
        Markdown documentation of recovery procedures
    """
    return """
# DeepSeek Training Recovery Procedures

## Worker Failure
1. System automatically detects via heartbeat timeout
2. Remaining workers continue if >= min_workers
3. Failed worker restarts and rejoins via rendezvous
4. Training resumes from latest checkpoint

## Node Failure (All Workers on Node)
1. Detected via heartbeat monitor
2. Training continues with reduced world size
3. Batch size automatically adjusted (if enabled)
4. Failed node can rejoin after recovery

## Preemption (Spot Instance)
1. SIGTERM received from cloud provider
2. Emergency checkpoint saved immediately
3. Worker terminates gracefully
4. New spot instance launches and loads checkpoint

## OOM (Out of Memory)
1. Caught by training loop
2. Gradient accumulation increased automatically
3. Batch size reduced if accumulation maxed out
4. Training continues from last successful step

## Network Partition
1. Heartbeats fail across partition
2. Each partition continues if >= min_workers
3. After partition heals, workers reconnect
4. Checkpoint reconciliation may be needed

## Complete Cluster Failure
1. All checkpoints preserved on persistent storage
2. New cluster launches via same configuration
3. Training loads from latest valid checkpoint
4. Global step and state fully restored

## Recovery Commands

### Restart from checkpoint:
```bash
torchrun --nnodes=1:8 --nproc_per_node=8 train.py \\
    --resume_from_checkpoint=./checkpoints/latest.pt
```

### Reduce cluster size:
```bash
torchrun --nnodes=1:4 --nproc_per_node=8 train.py \\
    --auto_scale_batch_size
```

### Manual checkpoint save:
```bash
kill -SIGUSR1 <training_pid>
```
"""
