"""
DeepSeek MLX Implementation

MLX-native implementations optimized for Apple Silicon.
Provides automatic Neural Engine utilization where beneficial.

Note: This module is named mlx_impl to avoid conflict with the mlx package.
Import as: from mlx_impl import MultiQueryAttention, etc.
"""

from .attention import MultiQueryAttention, GroupedQueryAttention, MultiHeadLatentAttention
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
]
