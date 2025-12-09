"""
OmegaConf Structured Config Schema
==================================

Defines dataclass-based schemas for configuration validation.
These schemas ensure type safety and provide IDE autocompletion.

Usage:
    from configs.hydra.schema import DeepSeekConfig
    
    # Load and validate config
    cfg = OmegaConf.structured(DeepSeekConfig())
    cfg = OmegaConf.merge(cfg, OmegaConf.load("config.yaml"))
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from omegaconf import MISSING, OmegaConf


# =============================================================================
# Enums for type-safe configuration
# =============================================================================

class OptimizerType(str, Enum):
    """Supported optimizer types."""
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"


class SchedulerType(str, Enum):
    """Supported learning rate scheduler types."""
    WSD = "wsd"  # Warmup-Stable-Decay
    COSINE = "cosine"
    LINEAR = "linear"
    CONSTANT = "constant"


class ParallelismMode(str, Enum):
    """Supported parallelism modes."""
    NONE = "none"
    DDP = "ddp"
    FSDP = "fsdp"
    PIPELINE = "pipeline"
    DUALPIPE = "dualpipe"


class FP8Format(str, Enum):
    """FP8 precision formats."""
    E4M3 = "e4m3"
    E5M2 = "e5m2"


# =============================================================================
# Model Configuration Schema
# =============================================================================

@dataclass
class AttentionConfig:
    """Attention layer configuration."""
    num_heads: int = 32
    d_latent: int = 512
    q_latent: int = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    decoupled_rope: bool = True
    rope_base: float = 10000.0
    rope_scaling: Optional[float] = None
    max_position_embeddings: int = 4096


@dataclass
class HierarchicalRoutingConfig:
    """Hierarchical routing configuration for memory-constrained scenarios."""
    enabled: bool = False
    num_groups: int = 8
    experts_per_group: int = 32
    group_top_k: int = 2
    expert_top_k_per_group: int = 4
    group_selection: str = "softmax"  # "softmax" or "sigmoid"
    group_load_balance: bool = True
    group_bias_lr: float = 0.001
    group_bias_clamp: float = 2.0


@dataclass
class ConsolidationConfig:
    """Expert consolidation configuration for very memory-constrained setups."""
    enabled: bool = False
    factor: int = 4  # Consolidate every N experts


@dataclass
class MoEConfig:
    """Mixture of Experts configuration."""
    use_moe: bool = True
    num_experts: int = 256
    num_shared_experts: int = 2
    top_k: int = 8
    expert_intermediate_dim: int = 2048
    use_auxiliary_loss: bool = False
    aux_loss_coef: float = 0.001
    load_balance_coef: float = 0.01
    capacity_factor: float = 1.25
    use_hierarchical_routing: bool = False  # Enable for A100/consumer GPUs
    hierarchical_routing: HierarchicalRoutingConfig = field(default_factory=HierarchicalRoutingConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    routing_temperature: float = 0.1
    bias_update_rate: float = 0.001
    load_balance_ema_decay: float = 0.99
    bias_clamp: float = 2.0
    min_capacity: int = 4
    enable_token_dropping: bool = True
    enable_megablocks: bool = True
    block_size: int = 64


@dataclass
class MTPConfig:
    """Multi-Token Prediction configuration."""
    enabled: bool = True
    depth: int = 1
    loss_weights: list[float] = field(default_factory=lambda: [1.0])
    share_embeddings: bool = True
    use_speculative_decoding: bool = True


@dataclass
class FP8Config:
    """FP8 mixed precision configuration."""
    enabled: bool = False
    format: str = "e4m3"  # e4m3 or e5m2
    tile_size: int = 128
    amax_history_len: int = 1024
    delayed_scaling: bool = True


@dataclass
class ModelConfig:
    """Complete model configuration."""
    # Core architecture
    name: str = "deepseek_v3"
    vocab_size: int = 128256
    num_layers: int = 61
    d_model: int = 7168
    d_hidden: int = 18432
    dropout: float = 0.0
    
    # Sub-configs
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    fp8: FP8Config = field(default_factory=FP8Config)
    
    # Initialization
    init_std: float = 0.02
    use_scaled_init: bool = True


# =============================================================================
# Training Configuration Schema
# =============================================================================

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str = "adamw"
    lr: float = 2.2e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.1
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "wsd"
    warmup_steps: int = 2000
    stable_steps: int = 0  # Auto-calculated if 0
    decay_steps: int = 0  # Auto-calculated if 0
    min_lr_ratio: float = 0.1
    decay_style: str = "cosine"


@dataclass
class GradientConfig:
    """Gradient handling configuration."""
    max_grad_norm: float = 1.0
    accumulation_steps: int = 1
    use_gradient_checkpointing: bool = True
    checkpointing_policy: str = "every_n"
    checkpointing_n: int = 2


@dataclass
class ParallelConfig:
    """Parallelism configuration."""
    mode: str = "fsdp"
    world_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = False


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    # Basic training params
    batch_size: int = 8
    max_steps: int = 100000
    eval_interval: int = 1000
    save_interval: int = 5000
    log_interval: int = 100
    
    # Precision
    mixed_precision: str = "bf16"
    compile_model: bool = False
    
    # Sub-configs
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    gradient: GradientConfig = field(default_factory=GradientConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None


# =============================================================================
# Data Configuration Schema
# =============================================================================

@dataclass
class DataConfig:
    """Data loading configuration."""
    dataset: str = "fineweb-edu"
    data_dir: str = "./data"
    train_split: str = "train"
    val_split: str = "validation"
    max_seq_len: int = 4096
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    streaming: bool = True


# =============================================================================
# Logging Configuration Schema
# =============================================================================

@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    enabled: bool = True
    project: str = "deepseek-from-scratch"
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    save_code: bool = True
    log_model: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "./logs"
    wandb: WandbConfig = field(default_factory=WandbConfig)
    tensorboard: bool = False
    tensorboard_dir: str = "./tensorboard"


# =============================================================================
# Environment Configuration Schema
# =============================================================================

@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    name: str = "local"
    seed: int = 42
    deterministic: bool = False
    cuda_visible_devices: Optional[str] = None
    num_gpus: int = 1
    nodes: int = 1


# =============================================================================
# Ray Tune Configuration Schema
# =============================================================================

class TuneSchedulerType(str, Enum):
    """Supported Ray Tune scheduler types."""
    ASHA = "asha"  # Async Successive Halving Algorithm (default, fast early stopping)
    PBT = "pbt"    # Population Based Training (adaptive mutation)


@dataclass
class TuneSearchSpaceConfig:
    """
    Search space configuration for hyperparameter tuning.
    
    For each hyperparameter, specify either:
    - choices: List of discrete values to sample from (tune.choice)
    - min/max: Range for continuous sampling (tune.loguniform for lr, tune.uniform otherwise)
    """
    # Learning rate search space
    learning_rate_min: float = 1e-5
    learning_rate_max: float = 1e-3
    learning_rate_choices: Optional[list[float]] = None  # If set, overrides min/max
    
    # Batch size search space
    batch_size_choices: list[int] = field(default_factory=lambda: [8, 16, 32, 64])
    
    # Warmup steps search space
    warmup_steps_min: int = 100
    warmup_steps_max: int = 2000
    
    # Weight decay search space
    weight_decay_min: float = 0.001
    weight_decay_max: float = 0.1
    
    # MoE capacity factor search space
    moe_capacity_factor_min: float = 1.0
    moe_capacity_factor_max: float = 2.0
    
    # GRPO beta search space (for alignment stage)
    grpo_beta_min: float = 0.01
    grpo_beta_max: float = 0.5


@dataclass
class TuneConfig:
    """
    Ray Tune hyperparameter optimization configuration.
    
    Integrates with existing Hydra config system and checkpoint structure.
    Uses Hydra for config composition only; Ray Tune handles trial parallelization.
    """
    # Enable/disable tuning
    enabled: bool = False
    
    # Scheduler configuration
    scheduler_type: str = "asha"  # "asha" (default) or "pbt"
    
    # Trial configuration
    num_samples: int = 10  # Number of hyperparameter configurations to try
    max_concurrent_trials: int = 4  # Max parallel trials
    
    # ASHA-specific settings (Async Successive Halving)
    grace_period: int = 100  # Min steps before early stopping
    reduction_factor: int = 4  # Halving factor for ASHA
    max_t: int = 10000  # Max training steps per trial
    
    # PBT-specific settings (Population Based Training)
    perturbation_interval: int = 100  # Steps between perturbations (checkpoint frequency)
    hyperparam_mutations: Optional[dict] = None  # Custom mutation ranges
    
    # Backend-specific metric names for optimization
    torch_metric: str = "torch_val_loss"
    mlx_metric: str = "mlx_val_loss"
    rust_metric: str = "rust_val_loss"
    
    # Optimization direction
    mode: str = "min"  # "min" for loss, "max" for accuracy
    
    # Checkpoint integration (uses Hydra checkpoint.output_dir)
    checkpoint_subdir: str = "tune"  # Subdirectory under checkpoint.output_dir
    
    # Search space configuration
    search_space: TuneSearchSpaceConfig = field(default_factory=TuneSearchSpaceConfig)
    
    # Trial naming
    trial_name_template: str = "{experiment_name}_trial_{trial_id}"


# =============================================================================
# Master Configuration Schema
# =============================================================================

@dataclass
class DeepSeekConfig:
    """
    Master configuration schema for DeepSeek training.
    
    This provides full type checking and validation for all config fields.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_config(cfg: Any) -> list[str]:
    """
    Validate configuration and return list of errors.
    
    Args:
        cfg: Configuration to validate (OmegaConf or dict)
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Convert to OmegaConf if needed
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    
    # Model validation
    if hasattr(cfg, 'model'):
        if cfg.model.vocab_size <= 0:
            errors.append("model.vocab_size must be positive")
        if cfg.model.num_layers <= 0:
            errors.append("model.num_layers must be positive")
        if cfg.model.d_model <= 0:
            errors.append("model.d_model must be positive")
        if cfg.model.d_model % cfg.model.attention.num_heads != 0:
            errors.append("model.d_model must be divisible by attention.num_heads")
    
    # Training validation
    if hasattr(cfg, 'training'):
        if cfg.training.batch_size <= 0:
            errors.append("training.batch_size must be positive")
        if cfg.training.optimizer.lr <= 0:
            errors.append("training.optimizer.lr must be positive")
        if cfg.training.gradient.max_grad_norm <= 0:
            errors.append("training.gradient.max_grad_norm must be positive")
    
    # MoE validation
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'moe'):
        moe = cfg.model.moe
        if moe.use_moe:
            if moe.num_experts <= 0:
                errors.append("model.moe.num_experts must be positive")
            if moe.top_k > moe.num_experts:
                errors.append("model.moe.top_k cannot exceed num_experts")
    
    # Tune validation
    if hasattr(cfg, 'tune') and cfg.tune.enabled:
        tune = cfg.tune
        if tune.scheduler_type not in ("asha", "pbt"):
            errors.append("tune.scheduler_type must be 'asha' or 'pbt'")
        if tune.num_samples <= 0:
            errors.append("tune.num_samples must be positive")
        if tune.max_concurrent_trials <= 0:
            errors.append("tune.max_concurrent_trials must be positive")
        if tune.grace_period <= 0:
            errors.append("tune.grace_period must be positive")
        if tune.perturbation_interval <= 0:
            errors.append("tune.perturbation_interval must be positive")
        if tune.mode not in ("min", "max"):
            errors.append("tune.mode must be 'min' or 'max'")
        # Search space validation
        ss = tune.search_space
        if ss.learning_rate_min >= ss.learning_rate_max:
            errors.append("tune.search_space.learning_rate_min must be < learning_rate_max")
        if ss.warmup_steps_min >= ss.warmup_steps_max:
            errors.append("tune.search_space.warmup_steps_min must be < warmup_steps_max")
    
    return errors


def load_and_validate(config_path: str) -> Any:
    """
    Load configuration from file and validate.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Validated OmegaConf configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # Merge with schema for defaults
    schema = OmegaConf.structured(DeepSeekConfig)
    cfg = OmegaConf.merge(schema, cfg)
    
    # Validate
    errors = validate_config(cfg)
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return cfg


# Register resolvers for dynamic config values
OmegaConf.register_new_resolver("now", lambda fmt: __import__("datetime").datetime.now().strftime(fmt), replace=True)
OmegaConf.register_new_resolver("env", lambda key, default="": __import__("os").environ.get(key, default), replace=True)
OmegaConf.register_new_resolver("eval", lambda expr: eval(expr), replace=True)
