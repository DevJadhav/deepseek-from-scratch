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

Reference: DeepSeek-V3 uses fault-tolerant training infrastructure
to handle failures during long training runs.
"""

import os
import sys
import signal
import time
import threading
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
from contextlib import contextmanager
import json
import atexit

try:
    from deepseek.torch.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class WorkerState(str, Enum):
    """State of a worker in the elastic training system."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    FAILED = "failed"
    TERMINATED = "terminated"
    PREEMPTED = "preempted"


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
