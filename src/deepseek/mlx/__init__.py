"""
DeepSeek MLX Implementation

MLX-native implementations optimized for Apple Silicon.
Provides automatic Neural Engine utilization where beneficial.

Note: This module is named mlx_impl to avoid conflict with the mlx package.
Import as: from mlx_impl import MultiQueryAttention, etc.
"""

from .attention import (
    GroupedQueryAttention,
    MultiHeadLatentAttention,
    MultiQueryAttention,
)
from .moe import DeepSeekMoE
from .mtp import MTPModel
from .grpo import GRPOTrainer
from .agent import (
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
    ToolCallParser,
    AgentGroupSampler,
    create_agent_trainer,
    compute_agent_reward,
)
from .kernel_fusions import (
    MLX_AVAILABLE,
    AppleSiliconGen,
    MetalFeatures,
    get_metal_features,
    TileConfig,
    softmax_fused,
    softmax_fused_safe,
    RMSNormFused,
    rms_norm_fused,
    attention_tiled,
    attention_fused,
    matmul_fp16_tiled,
    matmul_bf16_tiled,
    swiglu_fused,
    MLXKernelStats,
    MLXKernelDispatcher,
    get_dispatcher as get_mlx_dispatcher,
)

__all__ = [
    # Attention
    "MultiQueryAttention",
    "GroupedQueryAttention",
    "MultiHeadLatentAttention",
    # MoE
    "DeepSeekMoE",
    # MTP
    "MTPModel",
    # GRPO
    "GRPOTrainer",
    # Agent Training
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
    "ToolCallParser",
    "AgentGroupSampler",
    "create_agent_trainer",
    "compute_agent_reward",
    # Kernel Fusions
    "MLX_AVAILABLE",
    "AppleSiliconGen",
    "MetalFeatures",
    "get_metal_features",
    "TileConfig",
    "softmax_fused",
    "softmax_fused_safe",
    "RMSNormFused",
    "rms_norm_fused",
    "attention_tiled",
    "attention_fused",
    "matmul_fp16_tiled",
    "matmul_bf16_tiled",
    "swiglu_fused",
    "MLXKernelStats",
    "MLXKernelDispatcher",
    "get_mlx_dispatcher",
]
