#!/usr/bin/env python3
"""
Hydra-Based Training Entry Point
================================

This script provides a Hydra-based entry point for DeepSeek training,
enabling full configuration management through YAML files and command-line
overrides.

Usage:
    # Basic training with default config
    uv run python scripts/train_hydra.py

    # Override specific values
    uv run python scripts/train_hydra.py training.batch_size=16 model.d_model=512

    # Use a specific experiment config
    uv run python scripts/train_hydra.py +experiment=tiny_test

    # Enable FP8 mixed precision
    uv run python scripts/train_hydra.py model.fp8.enabled=true

    # Force FP8 via flag
    uv run python scripts/train_hydra.py --force-fp8

Configuration Structure:
    config/hydra/
    ├── config.yaml          # Main config (defaults)
    ├── model/               # Model architecture configs
    ├── training/            # Training hyperparameters
    ├── data/                # Dataset configs
    ├── environment/         # Environment settings
    └── experiment/          # Experiment presets

Environment Variables:
    DEEPSEEK_PROJECT_ROOT: Override project root detection
    HYDRA_FULL_ERROR: Set to 1 for full stack traces
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import Hydra schema for validation (noqa to allow dynamic path)
from config.hydra.schema import validate_config  # noqa: E402


def convert_hydra_to_pipeline_config(cfg: DictConfig):
    """
    Convert Hydra config to PipelineConfig.

    This bridges the gap between Hydra's structured config and
    the existing pipeline infrastructure.
    """
    from deepseek.pipeline.config import (
        Backend,
        DataConfig,
        DistributedConfig,
        ModelConfig,
        PipelineConfig,
        TrainingConfig,
    )

    # Build model config
    model = ModelConfig(
        d_model=cfg.model.d_model,
        num_heads=cfg.model.attention.num_heads,
        num_layers=cfg.model.num_layers,
        vocab_size=cfg.model.vocab_size,
        d_hidden=cfg.model.d_hidden,
        d_latent=cfg.model.attention.d_latent,
        d_rope=cfg.model.attention.qk_rope_head_dim,
        num_experts=cfg.model.moe.num_experts if cfg.model.moe.use_moe else 1,
        num_shared_experts=cfg.model.moe.num_shared_experts if cfg.model.moe.use_moe else 0,
        top_k=cfg.model.moe.top_k if cfg.model.moe.use_moe else 1,
        mtp_k=cfg.model.mtp.depth if cfg.model.mtp.enabled else 0,
        max_seq_len=cfg.model.attention.max_position_embeddings,
        use_moe=cfg.model.moe.use_moe,
        use_mtp=cfg.model.mtp.enabled,
        use_fp8=cfg.model.fp8.enabled,
        dropout=cfg.model.dropout,
    )

    # Build training config
    training = TrainingConfig(
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        warmup_steps=cfg.training.scheduler.warmup_steps,
        max_steps=cfg.training.max_steps,
        gradient_accumulation_steps=cfg.training.gradient.accumulation_steps,
        max_grad_norm=cfg.training.gradient.max_grad_norm,
        gradient_checkpointing=cfg.training.gradient.use_gradient_checkpointing,
        save_every_n_steps=cfg.training.save_interval,
        log_every_n_steps=cfg.training.log_interval,
        eval_every_n_steps=cfg.training.eval_interval,
        checkpoint_dir=cfg.training.checkpoint_dir,
    )

    # Build data config
    data = DataConfig(
        max_seq_len=cfg.data.max_seq_len,
        num_workers=cfg.data.num_workers,
    )

    # Build distributed config
    distributed = DistributedConfig(
        num_workers=cfg.training.parallel.world_size,
        pipeline_parallel_size=cfg.training.parallel.pipeline_parallel_size,
        tensor_parallel_size=cfg.training.parallel.tensor_parallel_size,
    )

    # Detect backend
    backend = Backend.AUTO

    # Create pipeline config
    pipeline_config = PipelineConfig(
        run_name=cfg.logging.wandb.name or "hydra-training",
        backend=backend,
        model=model,
        training=training,
        data=data,
        distributed=distributed,
        use_wandb=cfg.logging.wandb.enabled,
        wandb_project=cfg.logging.wandb.project,
        wandb_entity=cfg.logging.wandb.entity,
        output_dir=cfg.logging.log_dir,
    )

    return pipeline_config


def setup_environment(cfg: DictConfig):
    """Set up environment based on config."""
    import random

    import numpy as np

    seed = cfg.environment.seed
    random.seed(seed)
    np.random.seed(seed)

    # Set CUDA visible devices if specified
    if cfg.environment.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.environment.cuda_visible_devices

    # Set project root
    os.environ["DEEPSEEK_PROJECT_ROOT"] = str(project_root)


@hydra.main(
    version_base=None,
    config_path="../config/hydra",
    config_name="config",
)
def main(cfg: DictConfig) -> float | None:
    """
    Main training entry point with Hydra configuration.

    Args:
        cfg: Hydra configuration (automatically injected)

    Returns:
        Final validation loss (for hyperparameter optimization)
    """
    # Print resolved config
    print("=" * 60)
    print("DeepSeek Training - Hydra Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Validate config
    errors = validate_config(cfg)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    # Setup environment
    setup_environment(cfg)

    # Convert to pipeline config
    pipeline_config = convert_hydra_to_pipeline_config(cfg)

    print("\nPipeline Configuration Summary:")
    print(pipeline_config.summary())

    # Detect backend and run appropriate training
    detected_backend = pipeline_config.detect_backend()
    print(f"\nDetected backend: {detected_backend.value}")

    # Import and run the appropriate workflow
    from deepseek.pipeline.workflow import DeepSeekWorkflow

    workflow = DeepSeekWorkflow(pipeline_config)

    # Run training
    try:
        result = workflow.run()
        print("\nTraining completed successfully!")

        # Return final loss for hyperparameter optimization
        if result and "final_loss" in result:
            return result["final_loss"]
        return None

    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


def cli_main():
    """CLI entry point with --force-fp8 flag support."""
    import argparse

    # Pre-parse for custom flags
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--force-fp8", action="store_true", help="Force FP8 mixed precision")
    args, remaining = parser.parse_known_args()

    # If --force-fp8 is set, add it to Hydra overrides
    if args.force_fp8:
        remaining.append("model.fp8.enabled=true")

    # Update sys.argv for Hydra
    sys.argv = [sys.argv[0]] + remaining

    # Run Hydra main
    main()


if __name__ == "__main__":
    cli_main()
