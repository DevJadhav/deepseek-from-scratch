#!/usr/bin/env python3
"""
Run Ray Pipeline on Modal with 5D Parallelism
==============================================

This script deploys and runs the DeepSeek Ray pipeline on Modal A100-40GB GPUs
with 5D parallelism and DualPipe bidirectional pipeline scheduling.

Prerequisites:
1. Modal CLI installed: uv pip install modal
2. Modal credentials configured: modal token set

Usage:
    # Run initial verification (8 GPUs, ~$34 for 2 hours)
    uv run python scripts/run_modal_pipeline.py --scale initial --backend pytorch --max-steps 100
    
    # Run scaled training (64 GPUs, ~$270 for 2 hours)  
    uv run python scripts/run_modal_pipeline.py --scale scaled --backend pytorch --max-steps 1000
    
    # Run Rust backend
    uv run python scripts/run_modal_pipeline.py --scale initial --backend rust --max-steps 100

Cost Estimates (A100-40GB @ $0.000583/sec):
- Initial (8 GPUs): $16.79/hour
- Scaled (64 GPUs): $134.32/hour
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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


def print_cost_estimate(scale: str, max_hours: float = 2.0):
    """Print cost estimate for the run."""
    from deepseek.cloud.modal.ray_cluster import Parallelism5DConfig
    
    if scale == "initial":
        config = Parallelism5DConfig.initial_config()
    else:
        config = Parallelism5DConfig.scaled_config()
    
    cost_per_hour = config.total_gpus * 0.000583 * 3600
    estimated_cost = cost_per_hour * max_hours
    
    print(f"\n{'='*60}")
    print(f"COST ESTIMATE")
    print(f"{'='*60}")
    print(f"Scale: {scale}")
    print(f"GPUs: {config.total_gpus}")
    print(f"  - Tensor Parallel (TP): {config.tensor_parallel_size}")
    print(f"  - Pipeline Parallel (PP): {config.pipeline_parallel_size}")
    print(f"  - Data Parallel (DP): {config.data_parallel_size}")
    print(f"  - Expert Parallel (EP): {config.expert_parallel_size}")
    print(f"Cost per hour: ${cost_per_hour:.2f}")
    print(f"Estimated total ({max_hours} hours): ${estimated_cost:.2f}")
    print(f"{'='*60}\n")
    
    return estimated_cost


def run_pytorch_verification(scale: str, max_steps: int):
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
    
    # Run on Modal
    result = run_pytorch_verification.remote(
        parallelism_config=config.to_dict(),
        max_steps=max_steps,
    )
    
    print(f"\nResult: {result}")
    return result


def run_rust_verification(scale: str, max_steps: int):
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
    
    # Run on Modal
    result = run_rust_verification.remote(
        parallelism_config=config.to_dict(),
        max_steps=max_steps,
    )
    
    print(f"\nResult: {result}")
    return result


def run_full_cluster(scale: str, backend: str, max_steps: int):
    """Deploy and run full Ray cluster on Modal."""
    import modal
    from deepseek.cloud.modal.ray_cluster import deploy_ray_cluster
    
    print(f"Deploying Ray cluster on Modal...")
    print(f"Scale: {scale}")
    print(f"Backend: {backend}")
    print(f"Max steps: {max_steps}")
    
    # This will deploy and run the cluster
    # Note: This requires the Modal app to be deployed first
    result = deploy_ray_cluster(
        scale=scale,
        backend=backend,
        max_steps=max_steps,
    )
    
    print(f"\nResult: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run DeepSeek Ray Pipeline on Modal with 5D Parallelism",
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
    
    args = parser.parse_args()
    
    # Print cost estimate
    estimated_cost = print_cost_estimate(args.scale, args.max_hours)
    
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
    
    # Run
    try:
        if args.verify_only:
            if args.backend == "pytorch":
                result = run_pytorch_verification(args.scale, args.max_steps)
            else:
                result = run_rust_verification(args.scale, args.max_steps)
        else:
            result = run_full_cluster(args.scale, args.backend, args.max_steps)
        
        print("\n✅ Completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
