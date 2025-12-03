"""
Tiny MLX Configuration for ~10M Parameter Model
================================================

Configuration for training a DeepSeek-V3.2-like model with:
- ~10M parameters (TINY preset)
- MLX backend (Apple Silicon optimized)
- Multi-Token Prediction (MTP k=2)
- TinyStories dataset

Usage:
    # As module
    from configs.tiny_mlx_config import create_tiny_config
    config = create_tiny_config()
    
    # Save to JSON
    python -c "from configs.tiny_mlx_config import create_tiny_config; create_tiny_config().save('configs/tiny_mlx_config.json')"
"""

import sys
from pathlib import Path

# Add ray_pipeline to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ray_pipeline.config import (
    PipelineConfig, 
    ModelConfig, 
    DataConfig, 
    TrainingConfig,
    SFTConfig,
    GRPOConfig,
    DistillationConfig,
    DistributedConfig,
    ModelSize, 
    Backend, 
    Stage
)


def create_tiny_config() -> PipelineConfig:
    """
    Create configuration for ~10M parameter model with MLX backend.
    
    Architecture:
        - d_model: 256 (hidden dimension)
        - num_layers: 4 (transformer layers)
        - num_heads: 4 (attention heads)
        - vocab_size: 8000 (vocabulary size)
        - num_experts: 4 (MoE routed experts)
        - mtp_k: 2 (multi-token prediction)
    
    Returns:
        PipelineConfig: Complete pipeline configuration
    """
    # Start with TINY preset
    config = PipelineConfig.from_size(ModelSize.TINY)
    
    # ================================================================
    # Backend Configuration - MLX (Apple Silicon)
    # ================================================================
    config.backend = Backend.MLX
    
    # ================================================================
    # Model Architecture - Enable MTP
    # ================================================================
    config.model.d_model = 256
    config.model.num_heads = 4
    config.model.num_layers = 4
    config.model.vocab_size = 8000
    config.model.max_seq_len = 512
    config.model.d_latent = 64
    config.model.d_rope = 32
    config.model.num_experts = 4
    config.model.num_shared_experts = 1
    config.model.top_k = 1
    config.model.num_expert_groups = 2
    
    # Enable Multi-Token Prediction (k=2)
    config.model.mtp_k = 2
    
    # ================================================================
    # Data Configuration - TinyStories
    # ================================================================
    config.data.data_dir = "./data/stories"
    config.data.cache_dir = "./cache"
    config.data.tokenizer_name = "gpt2"  # Simple tokenizer for stories
    config.data.domain_weights = {"stories": 1.0}
    
    # Curriculum learning
    config.data.use_curriculum = True
    config.data.curriculum_start_seq_len = 128
    config.data.curriculum_end_seq_len = 512
    config.data.curriculum_warmup_steps = 2000
    config.data.curriculum_total_steps = 1000
    
    # Processing
    config.data.num_workers = 2
    config.data.prefetch_batches = 2
    config.data.shuffle_buffer_size = 5000
    
    # ================================================================
    # Training Hyperparameters
    # ================================================================
    config.training.learning_rate = 3e-4
    config.training.min_learning_rate = 1e-5
    config.training.weight_decay = 0.01
    config.training.max_grad_norm = 1.0
    
    # Schedule
    config.training.warmup_steps = 500
    config.training.max_steps = 1000
    config.training.scheduler = "cosine"
    
    # Batch configuration
    config.training.batch_size = 16
    config.training.gradient_accumulation_steps = 4
    
    # Mixed precision
    config.training.use_amp = True
    config.training.amp_dtype = "float16"
    
    # Checkpointing
    config.training.checkpoint_dir = "./checkpoints/tiny-mlx"
    config.training.save_every_n_steps = 1000
    config.training.keep_last_n_checkpoints = 3
    
    # Logging
    config.training.log_every_n_steps = 50
    config.training.eval_every_n_steps = 500
    
    # Gradient checkpointing (save memory)
    config.training.gradient_checkpointing = True
    
    # ================================================================
    # SFT (Supervised Fine-Tuning) Configuration
    # ================================================================
    config.sft.use_lora = True
    config.sft.lora_r = 8
    config.sft.lora_alpha = 16
    config.sft.lora_dropout = 0.05
    
    # NEFTune for better generalization
    config.sft.use_neftune = True
    config.sft.neftune_alpha = 5.0
    
    # Training
    config.sft.learning_rate = 2e-5
    config.sft.num_epochs = 3
    config.sft.warmup_ratio = 0.03
    
    # ================================================================
    # GRPO (Group Relative Policy Optimization) Configuration
    # ================================================================
    config.grpo.beta = 0.01  # KL penalty
    config.grpo.group_size = 4  # Samples per prompt
    config.grpo.learning_rate = 1e-6
    config.grpo.num_iterations = 1000
    config.grpo.kl_target = 0.1
    config.grpo.use_rule_based_reward = True
    
    # ================================================================
    # Pipeline Stages
    # ================================================================
    config.stages_to_run = [
        Stage.DATA_PREP,
        Stage.PRETRAIN,
        Stage.SFT,
        Stage.GRPO,
        Stage.EXPORT,
    ]
    
    # ================================================================
    # Distributed Configuration (single GPU for tiny model)
    # ================================================================
    config.distributed.num_workers = 1
    config.distributed.use_ray = False  # Local execution for tiny model
    
    # Run name for checkpoints/logs
    config.run_name = "tiny-mlx-tinystories"
    
    return config


def create_pretrain_only_config() -> PipelineConfig:
    """
    Create configuration for pre-training only (faster iteration).
    """
    config = create_tiny_config()
    config.stages_to_run = [Stage.DATA_PREP, Stage.PRETRAIN]
    config.training.max_steps = 5000  # Shorter for testing
    config.run_name = "tiny-mlx-pretrain-only"
    return config


def create_quick_test_config() -> PipelineConfig:
    """
    Create configuration for quick testing (minimal steps).
    """
    config = create_tiny_config()
    config.stages_to_run = [Stage.DATA_PREP, Stage.PRETRAIN]
    config.training.max_steps = 100
    config.training.eval_every_n_steps = 50
    config.training.save_every_n_steps = 50
    config.training.log_every_n_steps = 10
    config.data.curriculum_warmup_steps = 50
    config.data.curriculum_total_steps = 100
    config.run_name = "tiny-mlx-quick-test"
    return config


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Generate tiny MLX config")
    parser.add_argument(
        "--mode",
        choices=["full", "pretrain", "test"],
        default="full",
        help="Configuration mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print configuration summary"
    )
    
    args = parser.parse_args()
    
    # Create config based on mode
    if args.mode == "full":
        config = create_tiny_config()
    elif args.mode == "pretrain":
        config = create_pretrain_only_config()
    else:
        config = create_quick_test_config()
    
    # Show summary
    if args.show:
        print(config.summary())
    
    # Save to file
    output_path = args.output or f"configs/tiny_mlx_{args.mode}.json"
    config.save(output_path)
    print(f"Configuration saved to: {output_path}")
    
    # Print parameter estimate
    params = config.model.estimate_params()
    print(f"Estimated parameters: {params:,} (~{params/1e6:.1f}M)")
