"""
Training Infrastructure for DeepSeek From Scratch

This module provides:
- Device-aware training loop with MPS/CUDA/CPU support
- AdamW optimizer with weight decay
- Cosine annealing LR scheduler with warmup
- Gradient accumulation for effective larger batch sizes
- Mixed precision training (FP16/BF16)
- Checkpoint saving and loading
- Simple DataLoader abstraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import GradScaler, autocast
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Callable
import os
import json
import time
from deepseek.utils.logging import get_logger, TRAINING_LOSS, TOKENS_PROCESSED
from deepseek.utils.errors import retry, OOMError, TrainingError
from deepseek.utils.checkpoint import save_checkpoint_with_checksum, load_checkpoint_with_checksum
from deepseek.utils.validation import validate_inputs, validate_gradients
from deepseek.utils.distributed import (
    is_dist_avail_and_initialized, 
    get_rank, 
    get_world_size, 
    get_local_rank, 
    setup_distributed, 
    cleanup_distributed,
    reduce_dict
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

logger = get_logger(__name__)


def get_device() -> torch.device:
    """Get the best available device for Apple Silicon, CUDA, or CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    
    # Scheduler
    warmup_steps: int = 100
    max_steps: int = 10000
    min_lr_ratio: float = 0.1  # min_lr = learning_rate * min_lr_ratio
    
    # Batch
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Mixed Precision (only for CUDA)
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"
    
    # Checkpointing
    save_every: int = 1000
    checkpoint_dir: str = "./checkpoints"
    
    # Logging
    log_every: int = 10
    
    # Distributed
    backend: str = "nccl"  # 'nccl' or 'gloo'
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD


# =============================================================================
# DSA Alignment Loss (DeepSeek-V3.2)
# =============================================================================

class DSAAlignmentLoss(nn.Module):
    """
    Alignment loss for training DeepSeek Sparse Attention (DSA).
    
    During training, DSA outputs should match full attention outputs.
    This loss helps sparse attention learn to approximate full attention
    behavior while maintaining efficiency benefits at inference time.
    
    Components:
    - MSE loss: Ensures numerical similarity
    - Cosine similarity: Ensures directional alignment
    
    For efficiency, can sample a subset of positions rather than all.
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        cosine_weight: float = 0.5,
        sample_ratio: float = 0.1,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.sample_ratio = sample_ratio
    
    def forward(
        self,
        sparse_output: torch.Tensor,
        full_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute alignment loss between sparse and full attention outputs.
        
        Args:
            sparse_output: Output from sparse attention [B, T, D]
            full_output: Output from full attention [B, T, D] (teacher/reference)
            
        Returns:
            Combined alignment loss
        """
        # Optionally sample positions for efficiency
        if self.sample_ratio < 1.0:
            B, T, D = sparse_output.shape
            num_samples = max(1, int(T * self.sample_ratio))
            indices = torch.randperm(T, device=sparse_output.device)[:num_samples]
            sparse_output = sparse_output[:, indices, :]
            full_output = full_output[:, indices, :]
        
        loss = torch.tensor(0.0, device=sparse_output.device, dtype=sparse_output.dtype)
        
        # MSE loss
        if self.mse_weight > 0:
            mse = F.mse_loss(sparse_output, full_output)
            loss = loss + self.mse_weight * mse
        
        # Cosine similarity loss (1 - similarity)
        if self.cosine_weight > 0:
            sparse_flat = sparse_output.reshape(-1, sparse_output.shape[-1])
            full_flat = full_output.reshape(-1, full_output.shape[-1])
            cosine_sim = F.cosine_similarity(sparse_flat, full_flat, dim=-1)
            cosine_loss = 1.0 - cosine_sim.mean()
            loss = loss + self.cosine_weight * cosine_loss
        
        return loss


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Create a schedule with linear warmup and cosine decay.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359)).item())
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Training loop with support for:
    - Apple Silicon MPS / CUDA / CPU
    - Gradient accumulation
    - Mixed precision (CUDA only)
    - Checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        
        # Initialize distributed
        self.is_distributed = setup_distributed(config.backend)
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.local_rank = get_local_rank()
        
        # Device setup
        if device:
            self.device = device
        elif self.is_distributed:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = get_device()
            
        self.model = model.to(self.device)
        
        # Wrap model (DDP/FSDP)
        if self.is_distributed:
            if config.use_fsdp:
                # Simple FSDP wrapping policy
                auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy, min_num_params=1e5
                )
                self.model = FSDP(
                    self.model,
                    auto_wrap_policy=auto_wrap_policy,
                    device_id=self.device if self.device.type == "cuda" else None
                )
                logger.info("Wrapped model with FSDP")
            else:
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank] if self.device.type == "cuda" else None
                )
                logger.info("Wrapped model with DDP")
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
        
        # Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
            min_lr_ratio=config.min_lr_ratio,
        )
        
        # Mixed precision (only for CUDA)
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        self.amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
        
        # State
        self.global_step = 0
        self.accumulated_loss = 0.0
        
        # Create checkpoint dir
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        logger.info("Trainer initialized", 
                    rank=self.rank,
                    world_size=self.world_size,
                    device=str(self.device),
                    mixed_precision=self.use_amp,
                    amp_dtype=str(config.amp_dtype) if self.use_amp else 'N/A',
                    grad_accum_steps=config.gradient_accumulation_steps,
                    effective_batch_size=config.batch_size * config.gradient_accumulation_steps * self.world_size)
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute language modeling loss.
        Override this for custom loss computation.
        """
        # Forward pass
        logits = self.model(input_ids)
        
        # Shift for causal LM (predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Cross entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        return loss
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step with gradient accumulation support.
        Returns metrics dict.
        """
        try:
            return self._train_step_impl(input_ids, labels)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise OOMError("CUDA Out of Memory") from e
            raise e

    @retry(exceptions=(OOMError,), tries=3, delay=1.0, logger=logger)
    @validate_inputs
    def _train_step_impl(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        
        # Mixed precision context
        if self.use_amp:
            with autocast(dtype=self.amp_dtype):
                loss = self.compute_loss(input_ids, labels)
                loss = loss / self.config.gradient_accumulation_steps
        else:
            loss = self.compute_loss(input_ids, labels)
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.accumulated_loss += loss.item()
        
        # Record metrics (only on rank 0 or aggregated?)
        # For simplicity, we log local loss, but we should probably aggregate for logging
        
        return {"loss": loss.item() * self.config.gradient_accumulation_steps}
    
    def optimizer_step(self) -> Dict[str, float]:
        """
        Perform optimizer step after gradient accumulation.
        Returns metrics including gradient norm.
        """
        # Unscale for gradient clipping
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        
        # Validate gradients
        validate_gradients(self.model)
        
        # Optimizer step
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Scheduler step
        self.scheduler.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Update state
        self.global_step += 1
        avg_loss = self.accumulated_loss
        self.accumulated_loss = 0.0
        
        return {
            "avg_loss": avg_loss,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.scheduler.get_last_lr()[0],
            "step": self.global_step,
        }
    
    def train(
        self,
        data_iterator,  # Iterator yielding (input_ids, labels) batches
        num_steps: Optional[int] = None,
        callback: Optional[Callable[[Dict[str, float]], None]] = None,
    ):
        """
        Main training loop.
        
        Args:
            data_iterator: Iterator yielding (input_ids, labels) tuples
            num_steps: Number of optimizer steps (default: config.max_steps)
            callback: Optional callback function called after each optimizer step
        """
        num_steps = num_steps or self.config.max_steps
        self.model.train()
        
        accum_step = 0
        start_time = time.time()
        
        for batch_idx, (input_ids, labels) in enumerate(data_iterator):
            # Train step
            step_metrics = self.train_step(input_ids, labels)
            accum_step += 1
            
            # Optimizer step after accumulation
            if accum_step >= self.config.gradient_accumulation_steps:
                metrics = self.optimizer_step()
                accum_step = 0
                
                # Logging
                if self.global_step % self.config.log_every == 0 and self.rank == 0:
                    elapsed = time.time() - start_time
                    
                    # Reduce loss for logging if distributed
                    if self.is_distributed:
                        metrics = reduce_dict(metrics)
                        
                    logger.info("Training step",
                                step=metrics['step'],
                                loss=metrics['avg_loss'],
                                lr=metrics['lr'],
                                grad_norm=metrics['grad_norm'],
                                elapsed_s=elapsed)
                    
                    # Record metrics (only rank 0)
                    TRAINING_LOSS.labels(model_name="deepseek", phase="train").observe(metrics['avg_loss'])
                    # Note: TOKENS_PROCESSED needs to be carefully handled in DDP (sum across ranks)
                    # For now, we skip or just log local * world_size approximation
                
                # Checkpointing
                if self.global_step % self.config.save_every == 0 and self.rank == 0:
                    self.save_checkpoint()
                
                # Callback
                if callback:
                    callback(metrics)
                
                # Check if done
                if self.global_step >= num_steps:
                    if self.rank == 0:
                        logger.info("Training complete", step=self.global_step)
                    break
        
        # Final checkpoint
        if self.rank == 0:
            self.save_checkpoint(final=True)
            
        # Cleanup
        cleanup_distributed()
    
    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        if self.config.use_fsdp:
            # FSDP saving is more complex, need to set state dict type
            # For now, we assume FULL_STATE_DICT for simplicity on small models
            # In production, use SHARDED_STATE_DICT
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                model_state = self.model.state_dict()
        else:
            model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()

        suffix = "final" if final else f"step_{self.global_step}"
        path = os.path.join(self.config.checkpoint_dir, f"checkpoint_{suffix}.pt")
        
        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": self.config.__dict__,
        }
        
        if self.use_amp and self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        save_checkpoint_with_checksum(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = load_checkpoint_with_checksum(path, self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        
        if self.use_amp and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info("Checkpoint loaded", path=path, step=self.global_step)


class SimpleDataLoader:
    """
    Simple data loader for language modeling.
    Generates random batches for demonstration.
    Replace with actual data loading logic.
    """
    
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        batch_size: int,
        num_batches: int = 10000,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_batches = num_batches
        self._idx = 0
    
    def __iter__(self):
        self._idx = 0
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._idx >= self.num_batches:
            raise StopIteration
        
        self._idx += 1
        
        # Random data for demonstration
        # In practice, load actual tokenized text
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        labels = input_ids.clone()  # For causal LM, labels = input_ids
        
        return input_ids, labels



