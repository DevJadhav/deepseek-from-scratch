#!/usr/bin/env python3
"""
Run Ray Pipeline on Modal with 5D Parallelism
==============================================

This script deploys and runs the DeepSeek Ray pipeline on Modal A100-80GB × 8 GPUs
with 5D parallelism and DualPipe bidirectional pipeline scheduling.

Prerequisites:
1. Modal CLI installed: uv add modal
2. Modal credentials configured: modal token set

Usage:
    # Run initial verification (8 GPUs, ~$45 for 2 hours)
    uv run python scripts/run_modal_pipeline.py --scale initial --backend pytorch --max-steps 100
    
    # Run sequential pipeline (PyTorch first, then Rust)
    uv run python scripts/run_modal_pipeline.py --sequential --max-steps 1000
    
    # Run with auto-retry (max 3 attempts)
    uv run python scripts/run_modal_pipeline.py --scale initial --backend pytorch --max-retries 3
    
    # Verify volumes and setup
    uv run python scripts/run_modal_pipeline.py --verify-setup

Cost Estimates (A100-80GB @ $2.50/hr per GPU):
- Initial (8 GPUs): $20.00/hour
- Sequential (8 GPUs, both backends): ~$40/hour total
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Retry Configuration and Manager
# =============================================================================

class BackendType(Enum):
    """Backend execution types."""
    PYTORCH = "pytorch"
    RUST = "rust"


class RunStatus(Enum):
    """Status of a training run."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay_seconds: float = 60.0
    exponential_backoff: bool = True
    checkpoint_resume: bool = True
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed)."""
        if self.exponential_backoff:
            return self.base_delay_seconds * (2 ** attempt)
        return self.base_delay_seconds


@dataclass
class RunAttempt:
    """Record of a single run attempt."""
    attempt_number: int
    started_at: str
    ended_at: Optional[str] = None
    status: RunStatus = RunStatus.NOT_STARTED
    error_message: Optional[str] = None
    checkpoint_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass
class BackendRunState:
    """State for a backend's training run."""
    backend: BackendType
    status: RunStatus = RunStatus.NOT_STARTED
    attempts: List[RunAttempt] = field(default_factory=list)
    final_checkpoint: Optional[str] = None
    total_cost_usd: float = 0.0
    
    @property
    def attempt_count(self) -> int:
        return len(self.attempts)
    
    @property
    def last_attempt(self) -> Optional[RunAttempt]:
        return self.attempts[-1] if self.attempts else None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.value,
            "status": self.status.value,
            "attempts": [a.to_dict() for a in self.attempts],
            "final_checkpoint": self.final_checkpoint,
            "total_cost_usd": self.total_cost_usd,
            "attempt_count": self.attempt_count,
        }


class RetryManager:
    """
    Manages retry logic for Modal training runs.
    
    Features:
    - Max 3 retry attempts per backend
    - Exponential backoff (60s × 2^attempt)
    - Checkpoint-based resume
    - State persistence to JSON
    """
    
    def __init__(
        self,
        retry_config: RetryConfig,
        state_file: Optional[Path] = None,
    ):
        self.retry_config = retry_config
        self.state_file = state_file or Path("retry_state.json")
        self.backend_states: Dict[BackendType, BackendRunState] = {}
        self._load_state()
    
    def _load_state(self) -> None:
        """Load state from file if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                for backend_name, state_data in data.get("backends", {}).items():
                    backend = BackendType(backend_name)
                    state = BackendRunState(backend=backend)
                    state.status = RunStatus(state_data.get("status", "not_started"))
                    state.final_checkpoint = state_data.get("final_checkpoint")
                    state.total_cost_usd = state_data.get("total_cost_usd", 0.0)
                    self.backend_states[backend] = state
            except Exception as e:
                print(f"Warning: Could not load retry state: {e}")
    
    def _save_state(self) -> None:
        """Save current state to file."""
        data = {
            "updated_at": datetime.utcnow().isoformat(),
            "backends": {
                k.value: v.to_dict() for k, v in self.backend_states.items()
            },
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_or_create_state(self, backend: BackendType) -> BackendRunState:
        """Get or create state for a backend."""
        if backend not in self.backend_states:
            self.backend_states[backend] = BackendRunState(backend=backend)
        return self.backend_states[backend]
    
    def should_retry(self, backend: BackendType) -> bool:
        """Check if backend should be retried."""
        state = self.get_or_create_state(backend)
        if state.status == RunStatus.COMPLETED:
            return False
        return state.attempt_count < self.retry_config.max_attempts
    
    def start_attempt(self, backend: BackendType) -> RunAttempt:
        """Start a new attempt for a backend."""
        state = self.get_or_create_state(backend)
        attempt = RunAttempt(
            attempt_number=state.attempt_count + 1,
            started_at=datetime.utcnow().isoformat(),
            status=RunStatus.IN_PROGRESS,
        )
        state.attempts.append(attempt)
        state.status = RunStatus.IN_PROGRESS
        self._save_state()
        return attempt
    
    def complete_attempt(
        self,
        backend: BackendType,
        success: bool,
        error_message: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        cost_usd: float = 0.0,
    ) -> None:
        """Complete the current attempt."""
        state = self.get_or_create_state(backend)
        if not state.attempts:
            return
        
        attempt = state.attempts[-1]
        attempt.ended_at = datetime.utcnow().isoformat()
        attempt.status = RunStatus.COMPLETED if success else RunStatus.FAILED
        attempt.error_message = error_message
        attempt.checkpoint_path = checkpoint_path
        attempt.metrics = metrics or {}
        
        state.total_cost_usd += cost_usd
        
        if success:
            state.status = RunStatus.COMPLETED
            state.final_checkpoint = checkpoint_path
        elif self.should_retry(backend):
            state.status = RunStatus.RETRYING
        else:
            state.status = RunStatus.FAILED
        
        self._save_state()
    
    def get_checkpoint_for_resume(self, backend: BackendType) -> Optional[str]:
        """Get the checkpoint path for resuming training."""
        state = self.get_or_create_state(backend)
        for attempt in reversed(state.attempts):
            if attempt.checkpoint_path:
                return attempt.checkpoint_path
        return None
    
    def run_with_retry(
        self,
        backend: BackendType,
        run_fn: Callable[..., Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run a function with automatic retry on failure.
        
        Args:
            backend: Backend type being run
            run_fn: Function to execute (should return dict with 'success' key)
            **kwargs: Arguments to pass to run_fn
            
        Returns:
            Result dict from the last attempt
        """
        state = self.get_or_create_state(backend)
        
        # Check if already completed
        if state.status == RunStatus.COMPLETED:
            print(f"[{backend.value}] Already completed, skipping")
            return {"success": True, "skipped": True, "checkpoint": state.final_checkpoint}
        
        last_result = None
        
        while self.should_retry(backend):
            attempt = self.start_attempt(backend)
            attempt_num = attempt.attempt_number
            max_attempts = self.retry_config.max_attempts
            
            print(f"\n[{backend.value}] Attempt {attempt_num}/{max_attempts}")
            
            # Add checkpoint resume if available
            if self.retry_config.checkpoint_resume:
                checkpoint = self.get_checkpoint_for_resume(backend)
                if checkpoint:
                    print(f"[{backend.value}] Resuming from checkpoint: {checkpoint}")
                    kwargs["resume_from"] = checkpoint
            
            try:
                result = run_fn(**kwargs)
                success = result.get("success", False)
                
                self.complete_attempt(
                    backend=backend,
                    success=success,
                    checkpoint_path=result.get("checkpoint_path"),
                    metrics=result.get("metrics", {}),
                    cost_usd=result.get("cost_usd", 0.0),
                    error_message=result.get("error") if not success else None,
                )
                
                last_result = result
                
                if success:
                    print(f"[{backend.value}] ✅ Completed successfully!")
                    return result
                else:
                    print(f"[{backend.value}] ❌ Failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                self.complete_attempt(
                    backend=backend,
                    success=False,
                    error_message=str(e),
                )
                last_result = {"success": False, "error": str(e)}
                print(f"[{backend.value}] ❌ Exception: {e}")
            
            # Check if we should retry
            if self.should_retry(backend):
                delay = self.retry_config.get_delay(attempt_num - 1)
                print(f"[{backend.value}] Waiting {delay}s before retry...")
                time.sleep(delay)
        
        print(f"[{backend.value}] Max retries ({self.retry_config.max_attempts}) exhausted")
        return last_result or {"success": False, "error": "Max retries exhausted"}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all backend states."""
        return {
            "backends": {k.value: v.to_dict() for k, v in self.backend_states.items()},
            "total_cost_usd": sum(s.total_cost_usd for s in self.backend_states.values()),
            "all_completed": all(
                s.status == RunStatus.COMPLETED for s in self.backend_states.values()
            ) if self.backend_states else False,
        }


def check_modal_credentials() -> bool:
    """Check if Modal credentials are configured."""
    token_id = os.environ.get("MODAL_TOKEN_ID", "")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
    
    if not token_id or not token_secret:
        # Try loading from .env
        from dotenv import load_dotenv
        load_dotenv()
        token_id = os.environ.get("MODAL_TOKEN_ID", "")
        token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
    
    return bool(token_id and token_secret)


def print_cost_estimate(scale: str, max_hours: float = 2.0, sequential: bool = False):
    """Print cost estimate for the run using A100-80GB pricing."""
    from deepseek.cloud.modal.ray_cluster import (
        Parallelism5DConfig, GPU_HOURLY_RATE, GPU_COUNT, TOTAL_HOURLY_RATE
    )
    
    if scale == "initial":
        config = Parallelism5DConfig.initial_config()
    else:
        config = Parallelism5DConfig.scaled_config()
    
    cost_per_hour = config.total_gpus * GPU_HOURLY_RATE
    estimated_cost = cost_per_hour * max_hours
    
    # Double if running sequential (both backends)
    if sequential:
        estimated_cost *= 2
    
    print(f"\n{'='*60}")
    print(f"COST ESTIMATE (A100-80GB @ ${GPU_HOURLY_RATE}/GPU/hr)")
    print(f"{'='*60}")
    print(f"Scale: {scale}")
    print(f"GPUs: {config.total_gpus}")
    print(f"  - Tensor Parallel (TP): {config.tensor_parallel_size}")
    print(f"  - Pipeline Parallel (PP): {config.pipeline_parallel_size}")
    print(f"  - Data Parallel (DP): {config.data_parallel_size}")
    print(f"  - Expert Parallel (EP): {config.expert_parallel_size}")
    print(f"Cost per hour: ${cost_per_hour:.2f}")
    if sequential:
        print(f"Mode: Sequential (PyTorch → Rust)")
        print(f"Estimated total ({max_hours}h × 2 backends): ${estimated_cost:.2f}")
    else:
        print(f"Estimated total ({max_hours} hours): ${estimated_cost:.2f}")
    print(f"{'='*60}\n")
    
    return estimated_cost


def run_pytorch_verification(scale: str, max_steps: int, resume_from: Optional[str] = None):
    """Run PyTorch backend verification on Modal."""
    import modal
    from deepseek.cloud.modal.ray_cluster import run_pytorch_verification, Parallelism5DConfig
    
    if scale == "initial":
        config = Parallelism5DConfig.initial_config()
    else:
        config = Parallelism5DConfig.scaled_config()
    
    print(f"Running PyTorch verification on Modal...")
    print(f"Config: {config.to_dict()}")
    print(f"Max steps: {max_steps}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    
    # Run on Modal
    result = run_pytorch_verification.remote(
        parallelism_config=config.to_dict(),
        max_steps=max_steps,
        resume_from=resume_from,
    )
    
    print(f"\nResult: {result}")
    return result


def run_rust_verification(scale: str, max_steps: int, resume_from: Optional[str] = None):
    """Run Rust backend verification on Modal."""
    import modal
    from deepseek.cloud.modal.ray_cluster import run_rust_verification, Parallelism5DConfig
    
    if scale == "initial":
        config = Parallelism5DConfig.initial_config()
    else:
        config = Parallelism5DConfig.scaled_config()
    
    print(f"Running Rust verification on Modal...")
    print(f"Config: {config.to_dict()}")
    print(f"Max steps: {max_steps}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    
    # Run on Modal
    result = run_rust_verification.remote(
        parallelism_config=config.to_dict(),
        max_steps=max_steps,
        resume_from=resume_from,
    )
    
    print(f"\nResult: {result}")
    return result


def run_full_cluster(scale: str, backend: str, max_steps: int, resume_from: Optional[str] = None):
    """Deploy and run full Ray cluster on Modal."""
    import modal
    from deepseek.cloud.modal.ray_cluster import deploy_ray_cluster
    
    print(f"Deploying Ray cluster on Modal...")
    print(f"Scale: {scale}")
    print(f"Backend: {backend}")
    print(f"Max steps: {max_steps}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    
    # This will deploy and run the cluster
    # Note: This requires the Modal app to be deployed first
    result = deploy_ray_cluster(
        scale=scale,
        backend=backend,
        max_steps=max_steps,
        resume_from=resume_from,
    )
    
    print(f"\nResult: {result}")
    return result


def verify_setup() -> Dict[str, Any]:
    """Verify Modal volumes and setup."""
    from deepseek.cloud.modal.ray_cluster import (
        setup_directories, verify_volumes, verify_pytorch_setup
    )
    
    print("🔧 Setting up directories...")
    setup_result = setup_directories.remote()
    print(f"   Created {setup_result.get('total_created', 0)} directories")
    
    print("🔍 Verifying volumes...")
    volume_result = verify_volumes.remote()
    print(f"   Status: {volume_result.get('status', 'unknown')}")
    
    print("🐍 Verifying PyTorch setup...")
    pytorch_result = verify_pytorch_setup.remote()
    print(f"   PyTorch: {pytorch_result.get('pytorch_version', 'unknown')}")
    print(f"   CUDA: {pytorch_result.get('cuda_available', False)}")
    if pytorch_result.get('cuda_available'):
        print(f"   GPU: {pytorch_result.get('gpu_name', 'unknown')}")
        print(f"   Memory: {pytorch_result.get('gpu_memory_gb', 0):.1f} GB")
    
    return {
        "setup": setup_result,
        "volumes": volume_result,
        "pytorch": pytorch_result,
    }


def verify_rust_setup() -> Dict[str, Any]:
    """Verify Rust binary build on Modal."""
    from deepseek.cloud.modal.ray_cluster import verify_rust_binary
    
    print("🦀 Verifying Rust binary...")
    result = verify_rust_binary.remote()
    print(f"   Binary exists: {result.get('binary_exists', False)}")
    print(f"   CUDA enabled: {result.get('cuda_enabled', False)}")
    print(f"   PyO3 enabled: {result.get('pyo3_enabled', False)}")
    print(f"   GPU available: {result.get('gpu_available', False)}")
    if result.get('gpu_available'):
        print(f"   GPU: {result.get('gpu_name', 'unknown')}")
    
    return result


def run_sequential_pipeline(
    scale: str,
    max_steps: int,
    retry_config: RetryConfig,
    verify_only: bool = False,
) -> Dict[str, Any]:
    """
    Run sequential pipeline: PyTorch first, then Rust.
    
    Both backends share the same volumes:
    - /checkpoints/{pytorch,rust}/{model_size}/
    - /logs/{wandb,tensorboard,json}/{pytorch,rust}/
    
    Auto-retry with max 3 attempts per backend.
    """
    print("\n" + "="*60)
    print("SEQUENTIAL PIPELINE EXECUTION")
    print("="*60)
    print(f"Order: PyTorch → Rust")
    print(f"Max retries per backend: {retry_config.max_attempts}")
    print(f"Checkpoint resume: {retry_config.checkpoint_resume}")
    print("="*60 + "\n")
    
    retry_manager = RetryManager(
        retry_config=retry_config,
        state_file=Path("sequential_pipeline_state.json"),
    )
    
    results = {}
    
    # Step 1: PyTorch backend
    print("\n" + "-"*40)
    print("PHASE 1: PyTorch Backend")
    print("-"*40)
    
    def run_pytorch(**kwargs):
        if verify_only:
            return run_pytorch_verification(scale, max_steps, **kwargs)
        return run_full_cluster(scale, "pytorch", max_steps, **kwargs)
    
    pytorch_result = retry_manager.run_with_retry(
        backend=BackendType.PYTORCH,
        run_fn=run_pytorch,
    )
    results["pytorch"] = pytorch_result
    
    # Check if PyTorch succeeded before running Rust
    if not pytorch_result.get("success", False) and not pytorch_result.get("skipped", False):
        print("\n⚠️  PyTorch backend failed, skipping Rust backend")
        results["rust"] = {"success": False, "error": "Skipped due to PyTorch failure"}
        results["summary"] = retry_manager.get_summary()
        return results
    
    # Step 2: Rust backend
    print("\n" + "-"*40)
    print("PHASE 2: Rust Backend")
    print("-"*40)
    
    def run_rust(**kwargs):
        if verify_only:
            return run_rust_verification(scale, max_steps, **kwargs)
        return run_full_cluster(scale, "rust", max_steps, **kwargs)
    
    rust_result = retry_manager.run_with_retry(
        backend=BackendType.RUST,
        run_fn=run_rust,
    )
    results["rust"] = rust_result
    
    # Summary
    summary = retry_manager.get_summary()
    results["summary"] = summary
    
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"PyTorch: {results['pytorch'].get('success', False)}")
    print(f"Rust: {results['rust'].get('success', False)}")
    print(f"Total cost: ${summary.get('total_cost_usd', 0):.2f}")
    print("="*60 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run DeepSeek Ray Pipeline on Modal with 5D Parallelism (A100-80GB × 8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scale",
        choices=["initial", "scaled"],
        default="initial",
        help="Scale of deployment: initial (8 GPUs) or scaled (64 GPUs)",
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "rust"],
        default="pytorch",
        help="Training backend to use",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run single-GPU verification instead of full cluster",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cost estimate without running",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=2.0,
        help="Maximum hours for cost estimate",
    )
    
    # New arguments for Steps 1-3
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequential pipeline: PyTorch first, then Rust",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per backend (default: 3)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=60.0,
        help="Base delay between retries in seconds (default: 60)",
    )
    parser.add_argument(
        "--no-checkpoint-resume",
        action="store_true",
        help="Disable checkpoint-based resume on retry",
    )
    parser.add_argument(
        "--verify-setup",
        action="store_true",
        help="Verify Modal volumes and setup (runs setup_directories, verify_volumes)",
    )
    parser.add_argument(
        "--verify-rust",
        action="store_true",
        help="Verify Rust binary build on Modal",
    )
    
    args = parser.parse_args()
    
    # Handle verify-setup
    if args.verify_setup:
        print("\n🔧 Verifying Modal setup...")
        try:
            result = verify_setup()
            print("\n✅ Setup verification complete!")
            return 0
        except Exception as e:
            print(f"\n❌ Setup verification failed: {e}")
            return 1
    
    # Handle verify-rust
    if args.verify_rust:
        print("\n🦀 Verifying Rust binary build...")
        try:
            result = verify_rust_setup()
            if result.get("binary_exists") and result.get("gpu_available"):
                print("\n✅ Rust verification complete!")
                return 0
            else:
                print("\n⚠️  Rust verification incomplete")
                return 1
        except Exception as e:
            print(f"\n❌ Rust verification failed: {e}")
            return 1
    
    # Print cost estimate
    estimated_cost = print_cost_estimate(args.scale, args.max_hours, args.sequential)
    
    if args.dry_run:
        print("Dry run mode - not deploying to Modal")
        return 0
    
    # Check credentials
    if not check_modal_credentials():
        print("\n⚠️  Modal credentials not configured!")
        print("Please set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in your environment")
        print("or in .env file. Get tokens from: https://modal.com/settings")
        print("\nTo configure:")
        print("  1. Go to https://modal.com/settings")
        print("  2. Create a new token")
        print("  3. Add to .env file:")
        print("     MODAL_TOKEN_ID=your-token-id")
        print("     MODAL_TOKEN_SECRET=your-token-secret")
        return 1
    
    # Confirm cost
    print(f"Estimated cost: ${estimated_cost:.2f}")
    response = input("Proceed? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted")
        return 0
    
    # Create retry config
    retry_config = RetryConfig(
        max_attempts=args.max_retries,
        base_delay_seconds=args.retry_delay,
        exponential_backoff=True,
        checkpoint_resume=not args.no_checkpoint_resume,
    )
    
    # Run
    try:
        if args.sequential:
            # Sequential pipeline: PyTorch first, then Rust
            result = run_sequential_pipeline(
                scale=args.scale,
                max_steps=args.max_steps,
                retry_config=retry_config,
                verify_only=args.verify_only,
            )
            success = result.get("summary", {}).get("all_completed", False)
        elif args.verify_only:
            # Single backend verification with retry
            retry_manager = RetryManager(retry_config)
            backend = BackendType(args.backend)
            
            def run_verification(**kwargs):
                if args.backend == "pytorch":
                    return run_pytorch_verification(args.scale, args.max_steps, **kwargs)
                else:
                    return run_rust_verification(args.scale, args.max_steps, **kwargs)
            
            result = retry_manager.run_with_retry(backend, run_verification)
            success = result.get("success", False)
        else:
            # Full cluster with retry
            retry_manager = RetryManager(retry_config)
            backend = BackendType(args.backend)
            
            def run_cluster(**kwargs):
                return run_full_cluster(args.scale, args.backend, args.max_steps, **kwargs)
            
            result = retry_manager.run_with_retry(backend, run_cluster)
            success = result.get("success", False)
        
        if success:
            print("\n✅ Completed successfully!")
            return 0
        else:
            print("\n❌ Pipeline failed")
            return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
