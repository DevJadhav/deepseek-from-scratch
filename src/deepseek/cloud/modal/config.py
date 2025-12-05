"""
Modal Configuration for DeepSeek Training
==========================================

Configuration management for Modal cloud GPU training with 5D parallelism.
Loads environment variables from environment and provides structured configuration.

Security Note:
- Secrets (MODAL_TOKEN_ID, MODAL_TOKEN_SECRET) must be set via environment variables
- Never store secrets in code or dataclass defaults
- Use `modal secret create` or environment variables for credential management

5D Parallelism for 10 GPUs
--------------------------
- TP (Tensor Parallel) = 2: Split weights across GPU pairs
- PP (Pipeline Parallel) = 1: No pipeline stages for small models
- DP (Data Parallel) = 5: 5 replicas processing different batches
- EP (Expert Parallel) = 1: All experts on same GPU
- SP (Sequence Parallel) = 1: No sequence splitting

Total: TP × PP × DP × EP = 2 × 1 × 5 × 1 = 10 GPUs
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from deepseek.cloud.modal.logging_utils import get_logger

# Module logger
_logger = get_logger(__name__)

# Load environment variables from .env file if present (development only)
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
    _logger.debug("loaded_dotenv", path=str(_env_path))


@dataclass
class Parallelism5DConfig:
    """
    5D Parallelism configuration for distributed training.
    
    For 10 A100-80GB GPUs, we use:
    - TP=2: Column-parallel for attention QKV projections
    - PP=1: Single pipeline stage (no inter-layer split)
    - DP=5: 5 data parallel groups
    - EP=1: Experts colocated (small MoE)
    - SP=1: No sequence parallelism (short sequences)
    
    Communication patterns:
    - TP: NVLink within GPU pairs (high bandwidth)
    - DP: NCCL all-reduce across replicas
    - EP: All-to-all for expert routing (when EP > 1)
    """
    
    # Core parallelism dimensions
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 5
    expert_parallel_size: int = 1
    sequence_parallel_size: int = 1
    
    # Derived values
    @property
    def total_gpus(self) -> int:
        """Total number of GPUs required."""
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.data_parallel_size
            * self.expert_parallel_size
        )
    
    @property
    def tp_group_size(self) -> int:
        """Size of tensor parallel groups."""
        return self.tensor_parallel_size
    
    @property
    def dp_group_size(self) -> int:
        """Size of data parallel groups."""
        return self.data_parallel_size
    
    def get_rank_mapping(self, global_rank: int) -> Dict[str, int]:
        """
        Map global GPU rank to parallelism group ranks.
        
        Args:
            global_rank: Global GPU rank (0-9 for 10 GPUs)
            
        Returns:
            Dict with tp_rank, pp_rank, dp_rank, ep_rank
        """
        # For TP=2, DP=5: GPUs 0-1 are TP group 0, GPUs 2-3 are TP group 1, etc.
        tp_rank = global_rank % self.tensor_parallel_size
        dp_rank = global_rank // self.tensor_parallel_size
        
        return {
            "global_rank": global_rank,
            "tp_rank": tp_rank,
            "pp_rank": 0,  # Single pipeline stage
            "dp_rank": dp_rank,
            "ep_rank": 0,  # Single expert group
            "sp_rank": 0,  # No sequence parallel
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "sequence_parallel_size": self.sequence_parallel_size,
            "total_gpus": self.total_gpus,
        }


@dataclass
class ModalConfig:
    """
    Modal cloud configuration for DeepSeek training.
    
    Manages:
    - Modal authentication (from environment)
    - GPU configuration (A100-80GB preemptible)
    - Checkpoint volume mounting
    - NCCL communication settings
    
    Security:
    - Secrets are loaded from environment variables only
    - Never store secrets in dataclass defaults or code
    - Use Modal's built-in secret management for production
    """
    
    # Modal authentication - MUST be set via environment variables
    # Do not provide defaults to prevent accidental credential exposure
    token_id: str = field(default="", repr=False)  # repr=False to hide in logs
    token_secret: str = field(default="", repr=False)  # repr=False to hide in logs
    
    # GPU configuration
    gpu_type: str = "H100"
    gpu_memory: str = "80GB"
    gpu_count: int = 1  # Per container (we spawn 10 containers)
    num_containers: int = 10
    use_preemptible: bool = True
    
    # Retry configuration for preemption
    max_retries: int = 3
    retry_delay_seconds: int = 60
    
    # Checkpoint configuration
    checkpoint_dir: str = "./checkpoints"
    checkpoint_remote_path: str = "/checkpoints"
    sync_on_save: bool = True  # Bi-directional sync
    
    # Volume configuration
    volume_name: str = "deepseek-checkpoints"
    
    # NCCL configuration
    nccl_debug: str = field(default_factory=lambda: os.getenv("NCCL_DEBUG", "WARN"))
    nccl_ib_disable: bool = field(
        default_factory=lambda: os.getenv("NCCL_IB_DISABLE", "0") == "1"
    )
    nccl_socket_ifname: str = field(
        default_factory=lambda: os.getenv("NCCL_SOCKET_IFNAME", "")
    )
    
    # Timeout configuration
    container_timeout_seconds: int = 86400  # 24 hours max
    
    # 5D Parallelism
    parallelism: Parallelism5DConfig = field(default_factory=Parallelism5DConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Load secrets from environment if not explicitly provided
        if not self.token_id:
            self.token_id = os.getenv("MODAL_TOKEN_ID", "")
        if not self.token_secret:
            self.token_secret = os.getenv("MODAL_TOKEN_SECRET", "")
        
        # Validate required credentials
        missing_secrets = []
        if not self.token_id:
            missing_secrets.append("MODAL_TOKEN_ID")
        if not self.token_secret:
            missing_secrets.append("MODAL_TOKEN_SECRET")
        
        if missing_secrets:
            _logger.error(
                "missing_credentials",
                missing=missing_secrets,
                hint="Set via environment variables or use 'modal token new'",
            )
            raise ValueError(
                f"Missing required Modal credentials: {', '.join(missing_secrets)}. "
                "Set via environment variables or use 'modal token new' to authenticate."
            )
        
        # Ensure checkpoint directory exists
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        _logger.debug(
            "config_initialized",
            gpu_type=self.gpu_type,
            num_containers=self.num_containers,
            checkpoint_dir=self.checkpoint_dir,
        )
    
    @property
    def gpu_spec(self) -> str:
        """GPU specification string for Modal."""
        return f"{self.gpu_type}-{self.gpu_memory}"
    
    def get_nccl_env(self) -> Dict[str, str]:
        """Get NCCL environment variables for multi-GPU communication."""
        env = {
            "NCCL_DEBUG": self.nccl_debug,
            "NCCL_IB_DISABLE": "1" if self.nccl_ib_disable else "0",
        }
        if self.nccl_socket_ifname:
            env["NCCL_SOCKET_IFNAME"] = self.nccl_socket_ifname
        return env
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gpu_type": self.gpu_type,
            "gpu_memory": self.gpu_memory,
            "gpu_count": self.gpu_count,
            "num_containers": self.num_containers,
            "use_preemptible": self.use_preemptible,
            "max_retries": self.max_retries,
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_remote_path": self.checkpoint_remote_path,
            "sync_on_save": self.sync_on_save,
            "parallelism": self.parallelism.to_dict(),
            "nccl_env": self.get_nccl_env(),
        }


def get_modal_config() -> ModalConfig:
    """
    Get Modal configuration from environment.
    
    Returns:
        ModalConfig: Configured Modal settings
        
    Raises:
        ValueError: If required environment variables are missing
    """
    return ModalConfig()


def get_5d_config(
    tp: int = 2,
    pp: int = 1,
    dp: int = 5,
    ep: int = 1,
    sp: int = 1,
) -> Parallelism5DConfig:
    """
    Create a 5D parallelism configuration.
    
    Args:
        tp: Tensor parallel size (default: 2)
        pp: Pipeline parallel size (default: 1)
        dp: Data parallel size (default: 5)
        ep: Expert parallel size (default: 1)
        sp: Sequence parallel size (default: 1)
        
    Returns:
        Parallelism5DConfig: Configured parallelism settings
    """
    return Parallelism5DConfig(
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        data_parallel_size=dp,
        expert_parallel_size=ep,
        sequence_parallel_size=sp,
    )


def validate_gpu_count(config: ModalConfig) -> bool:
    """
    Validate that GPU count matches parallelism configuration.
    
    Args:
        config: Modal configuration
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required = config.parallelism.total_gpus
    available = config.num_containers * config.gpu_count
    
    if available < required:
        raise ValueError(
            f"Insufficient GPUs: {available} available, {required} required "
            f"(TP={config.parallelism.tensor_parallel_size} × "
            f"PP={config.parallelism.pipeline_parallel_size} × "
            f"DP={config.parallelism.data_parallel_size} × "
            f"EP={config.parallelism.expert_parallel_size})"
        )
    
    return True
