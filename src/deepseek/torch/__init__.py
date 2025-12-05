"""
DeepSeek PyTorch Implementation

PyTorch/CUDA implementations with support for:
- Flash Attention integration
- Distributed training (FSDP, DDP)
- Mixed precision (FP16, BF16, FP8)
- Triton kernels for custom operations

Import as: from deepseek.torch import MultiQueryAttention, GRPOTrainer, etc.
"""

from .model import (
    # Attention mechanisms
    AttentionBackend,
    MultiQueryAttention,
    GroupedQueryAttention,
    MultiHeadLatentAttention,
    DeepSeekAttention,
    # Transformer components
    RMSNorm,
    DeepSeekLayer,
    DeepSeekModel,
    # MoE
    Expert,
    DeepSeekMoE,
    StandardMoE,
    # KV Cache
    KVCache,
    # RoPE
    RotaryPositionalEncoding,
    ExtendedRoPEConfig,
    ExtendedRotaryPositionalEncoding,
)
from .training import (
    # GRPO Training
    GRPOTrainer,
    GroupSampler,
    # Agent Training
    ToolType,
    ToolStatus,
    ToolCall,
    ToolResponse,
    AgentActionType,
    AgentAction,
    AgentStep,
    AgentTrajectory,
    TaskTier,
    RewardWeights,
    RewardBreakdown,
    AgentRewardComputer,
    TaskTemplate,
    TaskGenerator,
    AgentEnvironment,
    AgentGRPOConfig,
    AgentGRPOTrainer,
    # Optimization
    CompileMode,
    CompileConfig,
    compile_model,
    PrecisionMode,
    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    # Memory profiling
    MemoryStats,
    MemoryProfiler,
)

__all__ = [
    # Attention
    "AttentionBackend",
    "MultiQueryAttention",
    "GroupedQueryAttention",
    "MultiHeadLatentAttention",
    "DeepSeekAttention",
    # Transformer
    "RMSNorm",
    "DeepSeekLayer",
    "DeepSeekModel",
    # MoE
    "Expert",
    "DeepSeekMoE",
    "StandardMoE",
    # KV Cache
    "KVCache",
    # RoPE
    "RotaryPositionalEncoding",
    "ExtendedRoPEConfig",
    "ExtendedRotaryPositionalEncoding",
    # Training
    "GRPOTrainer",
    "GroupSampler",
    # Agent
    "ToolType",
    "ToolStatus",
    "ToolCall",
    "ToolResponse",
    "AgentActionType",
    "AgentAction",
    "AgentStep",
    "AgentTrajectory",
    "TaskTier",
    "RewardWeights",
    "RewardBreakdown",
    "AgentRewardComputer",
    "TaskTemplate",
    "TaskGenerator",
    "AgentEnvironment",
    "AgentGRPOConfig",
    "AgentGRPOTrainer",
    # Optimization
    "CompileMode",
    "CompileConfig",
    "compile_model",
    "PrecisionMode",
    "MixedPrecisionConfig",
    "MixedPrecisionTrainer",
    # Memory
    "MemoryStats",
    "MemoryProfiler",
]
