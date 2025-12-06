"""
Configuration Validation with Pydantic Strict Mode

Provides strict configuration validation for all DeepSeek training configs
using Pydantic v2 with strict mode enabled.

Features:
- Strict type validation (no coercion)
- Nested model validation
- Custom validators for domain-specific rules
- Comprehensive error messages
- JSON schema generation for documentation

Usage:
    from deepseek.config.validation import (
        ModelConfig,
        TrainingConfig,
        PipelineConfig,
        validate_config,
    )

    # Will raise ValidationError if invalid
    config = ModelConfig(
        d_model=4096,
        n_layers=32,
        n_heads=32,
    )
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================================
# Enums
# ============================================================================


class DeviceType(str, Enum):
    """Compute device types."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"
    MLX = "mlx"


class PrecisionType(str, Enum):
    """Training precision types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"


class RoutingStrategy(str, Enum):
    """MoE routing strategies."""
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
    TOPK_SOFTMAX = "topk_softmax"


class OptimizerType(str, Enum):
    """Optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    LAMB = "lamb"


class SchedulerType(str, Enum):
    """Learning rate scheduler types."""
    COSINE = "cosine"
    LINEAR = "linear"
    CONSTANT = "constant"
    WARMUP_COSINE = "warmup_cosine"


class LoadBalanceMethod(str, Enum):
    """MoE load balancing methods."""
    AUXILIARY_LOSS = "auxiliary_loss"
    BIAS_UPDATE = "bias_update"
    COMBINED = "combined"


# ============================================================================
# Base Configuration
# ============================================================================


class StrictConfig(BaseModel):
    """Base config with strict validation."""

    model_config = ConfigDict(
        strict=True,  # No type coercion
        frozen=False,  # Allow modification
        validate_default=True,  # Validate default values
        extra="forbid",  # No extra fields allowed
        use_enum_values=True,  # Convert enums to values
    )


# ============================================================================
# Model Configuration
# ============================================================================


class AttentionConfig(StrictConfig):
    """Configuration for attention mechanism."""

    num_heads: Annotated[int, Field(ge=1, le=256, description="Number of attention heads")]
    head_dim: Annotated[int, Field(ge=8, le=256, description="Dimension per head")]
    num_kv_heads: Annotated[int, Field(ge=1, description="Number of key-value heads for GQA")] | None = None
    dropout: Annotated[float, Field(ge=0.0, le=1.0, description="Attention dropout")] = 0.0
    use_flash_attention: bool = True
    use_sliding_window: bool = False
    sliding_window_size: Annotated[int, Field(ge=64)] = 4096

    @field_validator("num_kv_heads")
    @classmethod
    def validate_kv_heads(cls, v: int | None, info) -> int | None:
        """Validate KV heads divides num_heads."""
        if v is not None:
            num_heads = info.data.get("num_heads")
            if num_heads and num_heads % v != 0:
                raise ValueError(f"num_kv_heads ({v}) must divide num_heads ({num_heads})")
        return v


class MLAConfig(StrictConfig):
    """Configuration for Multi-Head Latent Attention."""

    d_latent: Annotated[int, Field(ge=64, le=4096, description="Latent dimension")]
    compression_ratio: Annotated[float, Field(ge=1.0, le=32.0, description="KV compression ratio")] = 8.0
    use_decoupled_rope: bool = True
    rope_base: Annotated[float, Field(ge=100, le=100000000)] = 10000.0
    rope_scaling: Literal["linear", "ntk", "yarn", "dynamic_ntk"] | None = None
    rope_scaling_factor: Annotated[float, Field(ge=1.0, le=100.0)] = 1.0


class MoEConfig(StrictConfig):
    """Configuration for Mixture of Experts."""

    num_experts: Annotated[int, Field(ge=1, le=512, description="Total number of experts")]
    num_shared_experts: Annotated[int, Field(ge=0, le=16, description="Number of shared experts")] = 1
    top_k: Annotated[int, Field(ge=1, le=64, description="Experts active per token")] = 8
    num_expert_groups: Annotated[int, Field(ge=1, le=64, description="Expert groups for hierarchical routing")] = 8
    expert_intermediate_size: Annotated[int, Field(ge=64, description="Expert FFN intermediate size")] = 2048
    routing_strategy: RoutingStrategy = RoutingStrategy.SIGMOID
    load_balance_method: LoadBalanceMethod = LoadBalanceMethod.BIAS_UPDATE
    capacity_factor: Annotated[float, Field(ge=1.0, le=4.0, description="Expert capacity factor")] = 1.25
    enable_token_dropping: bool = True
    expert_dropout: Annotated[float, Field(ge=0.0, le=0.5)] = 0.0

    # Bias update parameters
    bias_update_alpha: Annotated[float, Field(ge=0.0001, le=0.1)] = 0.001
    bias_ema_decay: Annotated[float, Field(ge=0.9, le=0.9999)] = 0.99
    bias_clamp: Annotated[float, Field(ge=0.5, le=10.0)] = 2.0

    @model_validator(mode="after")
    def validate_expert_groups(self) -> "MoEConfig":
        """Validate expert groups divides num_experts."""
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"num_expert_groups ({self.num_expert_groups})"
            )
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})")
        return self


class ModelConfig(StrictConfig):
    """Configuration for the complete model."""

    # Core dimensions
    d_model: Annotated[int, Field(ge=64, le=65536, description="Model hidden dimension")] = 4096
    n_layers: Annotated[int, Field(ge=1, le=200, description="Number of transformer layers")] = 32
    vocab_size: Annotated[int, Field(ge=1000, le=500000, description="Vocabulary size")] = 102400

    # FFN configuration
    ffn_hidden_mult: Annotated[float, Field(ge=1.0, le=16.0)] = 4.0
    ffn_activation: Literal["gelu", "silu", "swiglu", "relu"] = "swiglu"

    # Attention
    attention: AttentionConfig = Field(
        default_factory=lambda: AttentionConfig(num_heads=32, head_dim=128)
    )

    # MLA (optional)
    use_mla: bool = False
    mla: MLAConfig | None = None

    # MoE (optional)
    use_moe: bool = False
    moe: MoEConfig | None = None

    # Normalization
    norm_eps: Annotated[float, Field(ge=1e-12, le=1e-3)] = 1e-5
    use_rms_norm: bool = True

    # Precision
    precision: PrecisionType = PrecisionType.BF16

    # Context length
    max_seq_len: Annotated[int, Field(ge=128, le=262144)] = 4096

    @model_validator(mode="after")
    def validate_mla_config(self) -> "ModelConfig":
        """Validate MLA config presence."""
        if self.use_mla and self.mla is None:
            raise ValueError("MLA config required when use_mla=True")
        return self

    @model_validator(mode="after")
    def validate_moe_config(self) -> "ModelConfig":
        """Validate MoE config presence."""
        if self.use_moe and self.moe is None:
            raise ValueError("MoE config required when use_moe=True")
        return self


# ============================================================================
# Training Configuration
# ============================================================================


class OptimizerConfig(StrictConfig):
    """Configuration for optimizer."""

    type: OptimizerType = OptimizerType.ADAMW
    learning_rate: Annotated[float, Field(ge=1e-8, le=1.0)] = 3e-4
    weight_decay: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    beta1: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9
    beta2: Annotated[float, Field(ge=0.0, le=1.0)] = 0.95
    eps: Annotated[float, Field(ge=1e-12, le=1e-3)] = 1e-8
    grad_clip: Annotated[float, Field(ge=0.0, le=100.0)] = 1.0


class SchedulerConfig(StrictConfig):
    """Configuration for learning rate scheduler."""

    type: SchedulerType = SchedulerType.WARMUP_COSINE
    warmup_steps: Annotated[int, Field(ge=0, le=100000)] = 2000
    min_lr_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    decay_steps: Annotated[int, Field(ge=0)] | None = None


class DataConfig(StrictConfig):
    """Configuration for training data."""

    data_dir: str = "./data"
    batch_size: Annotated[int, Field(ge=1, le=10000)] = 8
    micro_batch_size: Annotated[int, Field(ge=1, le=1000)] = 1
    num_workers: Annotated[int, Field(ge=0, le=128)] = 4
    shuffle_buffer_size: Annotated[int, Field(ge=100, le=1000000)] = 10000
    prefetch_batches: Annotated[int, Field(ge=0, le=100)] = 2

    # Domain weights for pre-training
    domain_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "web": 0.60,
            "code": 0.20,
            "math": 0.10,
            "books": 0.05,
            "scientific": 0.05,
        }
    )

    @field_validator("domain_weights")
    @classmethod
    def validate_domain_weights(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate domain weights sum to approximately 1.0."""
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Domain weights must sum to 1.0, got {total}")
        return v

    @model_validator(mode="after")
    def validate_batch_sizes(self) -> "DataConfig":
        """Validate batch size divisibility."""
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by "
                f"micro_batch_size ({self.micro_batch_size})"
            )
        return self


class CheckpointConfig(StrictConfig):
    """Configuration for checkpointing."""

    checkpoint_dir: str = "./checkpoints"
    save_steps: Annotated[int, Field(ge=1, le=100000)] = 1000
    keep_last_n: Annotated[int, Field(ge=1, le=100)] = 5
    save_optimizer: bool = True
    save_scheduler: bool = True
    use_safetensors: bool = True


class TrainingConfig(StrictConfig):
    """Complete training configuration."""

    # Training steps
    total_steps: Annotated[int, Field(ge=1, le=10000000)] = 100000
    gradient_accumulation_steps: Annotated[int, Field(ge=1, le=1000)] = 1
    eval_steps: Annotated[int, Field(ge=1, le=100000)] = 500
    log_steps: Annotated[int, Field(ge=1, le=10000)] = 10

    # Components
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    # Device
    device: DeviceType = DeviceType.CUDA

    # Precision
    mixed_precision: bool = True
    precision: PrecisionType = PrecisionType.BF16

    # Reproducibility
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 42

    # Debugging
    debug: bool = False
    profile: bool = False


# ============================================================================
# Distributed Configuration
# ============================================================================


class DistributedConfig(StrictConfig):
    """Configuration for distributed training."""

    # Parallelism
    data_parallel_size: Annotated[int, Field(ge=1, le=1024)] = 1
    tensor_parallel_size: Annotated[int, Field(ge=1, le=16)] = 1
    pipeline_parallel_size: Annotated[int, Field(ge=1, le=64)] = 1
    expert_parallel_size: Annotated[int, Field(ge=1, le=128)] = 1
    sequence_parallel_size: Annotated[int, Field(ge=1, le=64)] = 1

    # Backend
    backend: Literal["nccl", "gloo", "mps"] = "nccl"
    init_method: str = "env://"

    # ZeRO optimization
    zero_stage: Annotated[int, Field(ge=0, le=3)] = 0
    offload_optimizer: bool = False
    offload_params: bool = False

    @model_validator(mode="after")
    def validate_parallelism(self) -> "DistributedConfig":
        """Validate parallelism settings."""
        total_gpus = (
            self.data_parallel_size *
            self.tensor_parallel_size *
            self.pipeline_parallel_size
        )
        if total_gpus > 1024:
            raise ValueError(f"Total GPU count ({total_gpus}) exceeds reasonable limit")
        return self


# ============================================================================
# Pipeline Configuration
# ============================================================================


class PipelineConfig(StrictConfig):
    """Complete pipeline configuration."""

    # Model
    model: ModelConfig = Field(default_factory=ModelConfig)

    # Training
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # Distributed
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)

    # Experiment tracking
    experiment_name: str = "deepseek-training"
    run_name: str | None = None

    # Output
    output_dir: str = "./outputs"
    log_dir: str = "./logs"

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from JSON file."""
        path = Path(path)
        return cls.model_validate_json(path.read_text())


# ============================================================================
# Validation Utilities
# ============================================================================


def validate_config(config_dict: dict[str, Any], config_class: type[StrictConfig]) -> StrictConfig:
    """
    Validate a configuration dictionary against a config class.

    Args:
        config_dict: Dictionary with configuration values
        config_class: Pydantic model class to validate against

    Returns:
        Validated configuration object

    Raises:
        ValidationError: If validation fails
    """
    return config_class.model_validate(config_dict)


def generate_schema(config_class: type[StrictConfig]) -> dict[str, Any]:
    """
    Generate JSON schema for a configuration class.

    Args:
        config_class: Pydantic model class

    Returns:
        JSON schema dictionary
    """
    return config_class.model_json_schema()


def load_and_validate(path: str | Path, config_class: type[StrictConfig]) -> StrictConfig:
    """
    Load and validate configuration from JSON file.

    Args:
        path: Path to JSON configuration file
        config_class: Pydantic model class to validate against

    Returns:
        Validated configuration object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If validation fails
    """
    import json

    path = Path(path)
    with open(path) as f:
        config_dict = json.load(f)

    return validate_config(config_dict, config_class)


# ============================================================================
# Default Configurations
# ============================================================================


def get_tiny_config() -> PipelineConfig:
    """Get tiny configuration for testing."""
    return PipelineConfig(
        model=ModelConfig(
            d_model=256,
            n_layers=4,
            vocab_size=1000,
            attention=AttentionConfig(num_heads=4, head_dim=64),
        ),
        training=TrainingConfig(
            total_steps=100,
            data=DataConfig(batch_size=2, micro_batch_size=1),
        ),
    )


def get_small_moe_config() -> PipelineConfig:
    """Get small MoE configuration for testing."""
    return PipelineConfig(
        model=ModelConfig(
            d_model=512,
            n_layers=8,
            vocab_size=32000,
            attention=AttentionConfig(num_heads=8, head_dim=64),
            use_moe=True,
            moe=MoEConfig(
                num_experts=16,
                num_shared_experts=1,
                top_k=2,
                num_expert_groups=4,
                expert_intermediate_size=1024,
            ),
        ),
        training=TrainingConfig(
            total_steps=1000,
            data=DataConfig(batch_size=8, micro_batch_size=2),
        ),
    )


def get_v3_config() -> PipelineConfig:
    """Get DeepSeek-V3 style configuration."""
    return PipelineConfig(
        model=ModelConfig(
            d_model=4096,
            n_layers=32,
            vocab_size=102400,
            attention=AttentionConfig(num_heads=32, head_dim=128, num_kv_heads=8),
            use_mla=True,
            mla=MLAConfig(
                d_latent=512,
                compression_ratio=8.0,
                use_decoupled_rope=True,
            ),
            use_moe=True,
            moe=MoEConfig(
                num_experts=256,
                num_shared_experts=1,
                top_k=8,
                num_expert_groups=8,
                expert_intermediate_size=2048,
                load_balance_method=LoadBalanceMethod.BIAS_UPDATE,
            ),
        ),
        training=TrainingConfig(
            total_steps=100000,
            optimizer=OptimizerConfig(
                learning_rate=3e-4,
                weight_decay=0.1,
            ),
            scheduler=SchedulerConfig(
                warmup_steps=2000,
            ),
        ),
        distributed=DistributedConfig(
            data_parallel_size=4,
            expert_parallel_size=8,
        ),
    )
