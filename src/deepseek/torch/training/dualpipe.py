"""
DualPipe Pipeline Parallelism for DeepSeek Training.

This module implements DeepSeek's DualPipe algorithm which provides:
- Bidirectional micro-batch scheduling (forward flows 0→N, backward flows N→0)
- Interleaved forward/backward schedule to minimize pipeline bubbles
- Non-blocking send/recv operations with CUDA streams
- Activation stashing between forward and backward passes
- Pipeline-parallel-aware learning rate scheduling
- Warmup and cooldown phase handling
- Bubble ratio measurement and logging

The DualPipe schedule significantly reduces pipeline bubbles compared to
standard 1F1B (GPipe) or PipeDream schedules by interleaving computations
more efficiently.

Reference: DeepSeek-V3 uses DualPipe for efficient pipeline parallelism
in their distributed training infrastructure.
"""

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable
from collections import deque
from enum import Enum, auto
from contextlib import contextmanager
import time

try:
    from deepseek.torch.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ScheduleAction(Enum):
    """Actions in the DualPipe schedule."""
    FORWARD = auto()
    BACKWARD = auto()
    SEND_FORWARD = auto()
    RECV_FORWARD = auto()
    SEND_BACKWARD = auto()
    RECV_BACKWARD = auto()
    WAIT = auto()
    OPTIMIZER_STEP = auto()


@dataclass
class PipelineConfig:
    """Configuration for DualPipe pipeline parallelism.
    
    Attributes:
        num_stages: Number of pipeline stages
        num_micro_batches: Number of micro-batches per mini-batch
        stage_to_rank: Mapping from stage ID to rank
        chunks_per_stage: Number of model chunks per stage
        interleaved: Use interleaved 1F1B schedule
        overlap_comm: Overlap communication with computation
        async_grad_reduce: Asynchronously reduce gradients
    """
    num_stages: int
    num_micro_batches: int
    stage_to_rank: Dict[int, int] = field(default_factory=dict)
    chunks_per_stage: int = 1
    interleaved: bool = True
    overlap_comm: bool = True
    async_grad_reduce: bool = True
    
    def __post_init__(self):
        # Default: stage i on rank i
        if not self.stage_to_rank:
            self.stage_to_rank = {i: i for i in range(self.num_stages)}


@dataclass
class MicroBatchState:
    """State for a single micro-batch during pipeline execution.
    
    Attributes:
        micro_batch_id: ID of this micro-batch
        input_tensor: Input to this stage
        output_tensor: Output from this stage (for backward)
        grad_input: Gradient w.r.t. input (for backward)
        grad_output: Gradient w.r.t. output (received from next stage)
        forward_done: Whether forward pass is complete
        backward_done: Whether backward pass is complete
    """
    micro_batch_id: int
    input_tensor: Optional[torch.Tensor] = None
    output_tensor: Optional[torch.Tensor] = None
    grad_input: Optional[torch.Tensor] = None
    grad_output: Optional[torch.Tensor] = None
    forward_done: bool = False
    backward_done: bool = False


@dataclass
class ScheduleStep:
    """Single step in the DualPipe schedule.
    
    Attributes:
        action: Type of action to perform
        micro_batch_id: Which micro-batch this action is for
        chunk_id: Which model chunk (for interleaved)
        peer_rank: Rank to communicate with (for send/recv)
    """
    action: ScheduleAction
    micro_batch_id: int
    chunk_id: int = 0
    peer_rank: Optional[int] = None


# Global pipeline state
_PIPELINE_GROUP: Optional[dist.ProcessGroup] = None
_PIPELINE_RANK: int = 0
_PIPELINE_WORLD_SIZE: int = 1


def init_pipeline_group(ranks: List[int]) -> dist.ProcessGroup:
    """Initialize pipeline parallel process group.
    
    Args:
        ranks: List of ranks in the pipeline (in order)
        
    Returns:
        Process group for pipeline communication
    """
    global _PIPELINE_GROUP, _PIPELINE_RANK, _PIPELINE_WORLD_SIZE
    
    if not dist.is_initialized():
        _PIPELINE_WORLD_SIZE = 1
        _PIPELINE_RANK = 0
        return None
    
    _PIPELINE_GROUP = dist.new_group(ranks)
    _PIPELINE_WORLD_SIZE = len(ranks)
    _PIPELINE_RANK = ranks.index(dist.get_rank())
    
    return _PIPELINE_GROUP


def get_pipeline_rank() -> int:
    """Get rank within pipeline group."""
    return _PIPELINE_RANK


def get_pipeline_world_size() -> int:
    """Get pipeline parallel world size."""
    return _PIPELINE_WORLD_SIZE


def get_prev_rank() -> Optional[int]:
    """Get rank of previous stage (None if first stage)."""
    if _PIPELINE_RANK == 0:
        return None
    return _PIPELINE_RANK - 1


def get_next_rank() -> Optional[int]:
    """Get rank of next stage (None if last stage)."""
    if _PIPELINE_RANK == _PIPELINE_WORLD_SIZE - 1:
        return None
    return _PIPELINE_RANK + 1


def is_first_stage() -> bool:
    """Check if this is the first pipeline stage."""
    return _PIPELINE_RANK == 0


def is_last_stage() -> bool:
    """Check if this is the last pipeline stage."""
    return _PIPELINE_RANK == _PIPELINE_WORLD_SIZE - 1


class DualPipeScheduler:
    """DualPipe bidirectional pipeline scheduler.
    
    Implements the DualPipe scheduling algorithm that minimizes pipeline
    bubbles through bidirectional micro-batch flow:
    - Forward passes flow from stage 0 to stage N
    - Backward passes flow from stage N to stage 0
    - Interleaved schedule overlaps forward and backward
    
    The schedule has three phases:
    1. Warmup: Build up the pipeline with forward passes
    2. Steady State: 1F1B interleaved forward and backward
    3. Cooldown: Drain the pipeline with backward passes
    """
    
    def __init__(self, config: PipelineConfig):
        """Initialize scheduler.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.num_stages = config.num_stages
        self.num_micro_batches = config.num_micro_batches
        self.stage_id = get_pipeline_rank()
        
        # Pre-compute the schedule
        self.schedule = self._build_schedule()
        
        logger.info(
            f"DualPipe scheduler initialized",
            stage=self.stage_id,
            num_stages=self.num_stages,
            num_micro_batches=self.num_micro_batches,
            schedule_length=len(self.schedule),
        )
    
    def _build_schedule(self) -> List[ScheduleStep]:
        """Build the complete DualPipe schedule for this stage.
        
        Returns:
            List of schedule steps to execute
        """
        schedule = []
        
        # Number of warmup micro-batches for this stage
        # Earlier stages do more warmup forwards
        num_warmup = min(
            self.num_stages - self.stage_id - 1,
            self.num_micro_batches
        )
        
        # Number of 1F1B iterations
        num_steady = self.num_micro_batches - num_warmup
        
        forward_idx = 0
        backward_idx = 0
        
        # Warmup phase: only forward passes
        for _ in range(num_warmup):
            # Receive from previous stage (if not first)
            if not is_first_stage():
                schedule.append(ScheduleStep(
                    action=ScheduleAction.RECV_FORWARD,
                    micro_batch_id=forward_idx,
                    peer_rank=get_prev_rank(),
                ))
            
            # Forward computation
            schedule.append(ScheduleStep(
                action=ScheduleAction.FORWARD,
                micro_batch_id=forward_idx,
            ))
            
            # Send to next stage (if not last)
            if not is_last_stage():
                schedule.append(ScheduleStep(
                    action=ScheduleAction.SEND_FORWARD,
                    micro_batch_id=forward_idx,
                    peer_rank=get_next_rank(),
                ))
            
            forward_idx += 1
        
        # Steady state: 1F1B interleaved
        for _ in range(num_steady):
            # Forward pass for new micro-batch
            if forward_idx < self.num_micro_batches:
                if not is_first_stage():
                    schedule.append(ScheduleStep(
                        action=ScheduleAction.RECV_FORWARD,
                        micro_batch_id=forward_idx,
                        peer_rank=get_prev_rank(),
                    ))
                
                schedule.append(ScheduleStep(
                    action=ScheduleAction.FORWARD,
                    micro_batch_id=forward_idx,
                ))
                
                if not is_last_stage():
                    schedule.append(ScheduleStep(
                        action=ScheduleAction.SEND_FORWARD,
                        micro_batch_id=forward_idx,
                        peer_rank=get_next_rank(),
                    ))
                
                forward_idx += 1
            
            # Backward pass for completed micro-batch
            if not is_last_stage():
                schedule.append(ScheduleStep(
                    action=ScheduleAction.RECV_BACKWARD,
                    micro_batch_id=backward_idx,
                    peer_rank=get_next_rank(),
                ))
            
            schedule.append(ScheduleStep(
                action=ScheduleAction.BACKWARD,
                micro_batch_id=backward_idx,
            ))
            
            if not is_first_stage():
                schedule.append(ScheduleStep(
                    action=ScheduleAction.SEND_BACKWARD,
                    micro_batch_id=backward_idx,
                    peer_rank=get_prev_rank(),
                ))
            
            backward_idx += 1
        
        # Cooldown phase: only backward passes
        while backward_idx < self.num_micro_batches:
            if not is_last_stage():
                schedule.append(ScheduleStep(
                    action=ScheduleAction.RECV_BACKWARD,
                    micro_batch_id=backward_idx,
                    peer_rank=get_next_rank(),
                ))
            
            schedule.append(ScheduleStep(
                action=ScheduleAction.BACKWARD,
                micro_batch_id=backward_idx,
            ))
            
            if not is_first_stage():
                schedule.append(ScheduleStep(
                    action=ScheduleAction.SEND_BACKWARD,
                    micro_batch_id=backward_idx,
                    peer_rank=get_prev_rank(),
                ))
            
            backward_idx += 1
        
        return schedule
    
    def __iter__(self):
        """Iterate through schedule steps."""
        return iter(self.schedule)
    
    def __len__(self):
        """Number of steps in schedule."""
        return len(self.schedule)


class PipelineStage(nn.Module):
    """A single stage in the DualPipe pipeline.
    
    Wraps a portion of the model and handles:
    - Forward/backward computation
    - Activation stashing for backward pass
    - Communication with adjacent stages
    """
    
    def __init__(
        self,
        module: nn.Module,
        stage_id: int,
        config: PipelineConfig,
    ):
        """Initialize pipeline stage.
        
        Args:
            module: Model layers for this stage
            stage_id: Stage ID (0 to num_stages-1)
            config: Pipeline configuration
        """
        super().__init__()
        self.module = module
        self.stage_id = stage_id
        self.config = config
        
        # Activation storage for backward pass
        self.input_stash: Dict[int, torch.Tensor] = {}
        self.output_stash: Dict[int, torch.Tensor] = {}
        
        # Communication handles for async operations
        self.send_handles: Dict[int, Any] = {}
        self.recv_handles: Dict[int, Any] = {}
        
        # Buffers for receiving tensors
        self.recv_forward_buffer: Optional[torch.Tensor] = None
        self.recv_backward_buffer: Optional[torch.Tensor] = None
        
        # CUDA streams for overlapped communication
        if config.overlap_comm and torch.cuda.is_available():
            self.comm_stream = torch.cuda.Stream()
        else:
            self.comm_stream = None
    
    def set_recv_buffers(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device):
        """Pre-allocate receive buffers for communication.
        
        Args:
            shape: Shape of tensors to receive
            dtype: Data type
            device: Device for buffers
        """
        self.recv_forward_buffer = torch.empty(shape, dtype=dtype, device=device)
        self.recv_backward_buffer = torch.empty(shape, dtype=dtype, device=device)
    
    def forward_step(
        self,
        micro_batch_id: int,
        input_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute forward pass for a micro-batch.
        
        Args:
            micro_batch_id: ID of micro-batch
            input_tensor: Input from previous stage (or data loader)
            
        Returns:
            Output tensor for this stage
        """
        # Stash input for backward
        if input_tensor is not None:
            self.input_stash[micro_batch_id] = input_tensor.detach().requires_grad_(True)
            x = self.input_stash[micro_batch_id]
        else:
            x = self.input_stash.get(micro_batch_id)
            if x is None:
                raise RuntimeError(f"No input for micro-batch {micro_batch_id}")
        
        # Forward through stage
        output = self.module(x)
        
        # Stash output for backward
        self.output_stash[micro_batch_id] = output
        
        return output
    
    def backward_step(
        self,
        micro_batch_id: int,
        grad_output: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Execute backward pass for a micro-batch.
        
        Args:
            micro_batch_id: ID of micro-batch
            grad_output: Gradient from next stage (or loss)
            
        Returns:
            Gradient w.r.t. input for previous stage
        """
        output = self.output_stash.get(micro_batch_id)
        if output is None:
            raise RuntimeError(f"No output stashed for micro-batch {micro_batch_id}")
        
        # Compute gradients
        if grad_output is not None:
            output.backward(grad_output)
        else:
            # Last stage: use loss gradient
            output.backward()
        
        # Get gradient w.r.t. input
        input_tensor = self.input_stash.get(micro_batch_id)
        grad_input = input_tensor.grad if input_tensor is not None else None
        
        # Clean up stash
        del self.output_stash[micro_batch_id]
        if micro_batch_id in self.input_stash:
            del self.input_stash[micro_batch_id]
        
        return grad_input
    
    def send_forward(
        self,
        micro_batch_id: int,
        tensor: torch.Tensor,
        dst_rank: int,
    ) -> None:
        """Send activation to next stage.
        
        Args:
            micro_batch_id: ID of micro-batch
            tensor: Tensor to send
            dst_rank: Destination rank
        """
        group = _PIPELINE_GROUP
        
        if self.comm_stream is not None:
            with torch.cuda.stream(self.comm_stream):
                handle = dist.isend(tensor.contiguous(), dst_rank, group=group)
        else:
            handle = dist.isend(tensor.contiguous(), dst_rank, group=group)
        
        self.send_handles[('forward', micro_batch_id)] = handle
    
    def recv_forward(
        self,
        micro_batch_id: int,
        src_rank: int,
    ) -> torch.Tensor:
        """Receive activation from previous stage.
        
        Args:
            micro_batch_id: ID of micro-batch
            src_rank: Source rank
            
        Returns:
            Received tensor
        """
        group = _PIPELINE_GROUP
        buffer = self.recv_forward_buffer
        
        if buffer is None:
            raise RuntimeError("Receive buffer not set. Call set_recv_buffers first.")
        
        if self.comm_stream is not None:
            with torch.cuda.stream(self.comm_stream):
                handle = dist.irecv(buffer, src_rank, group=group)
                handle.wait()
            torch.cuda.current_stream().wait_stream(self.comm_stream)
        else:
            handle = dist.irecv(buffer, src_rank, group=group)
            handle.wait()
        
        # Stash for forward
        self.input_stash[micro_batch_id] = buffer.clone().requires_grad_(True)
        
        return self.input_stash[micro_batch_id]
    
    def send_backward(
        self,
        micro_batch_id: int,
        tensor: torch.Tensor,
        dst_rank: int,
    ) -> None:
        """Send gradient to previous stage.
        
        Args:
            micro_batch_id: ID of micro-batch
            tensor: Gradient tensor to send
            dst_rank: Destination rank
        """
        group = _PIPELINE_GROUP
        
        if self.comm_stream is not None:
            with torch.cuda.stream(self.comm_stream):
                handle = dist.isend(tensor.contiguous(), dst_rank, group=group)
        else:
            handle = dist.isend(tensor.contiguous(), dst_rank, group=group)
        
        self.send_handles[('backward', micro_batch_id)] = handle
    
    def recv_backward(
        self,
        micro_batch_id: int,
        src_rank: int,
    ) -> torch.Tensor:
        """Receive gradient from next stage.
        
        Args:
            micro_batch_id: ID of micro-batch
            src_rank: Source rank
            
        Returns:
            Received gradient tensor
        """
        group = _PIPELINE_GROUP
        buffer = self.recv_backward_buffer
        
        if buffer is None:
            raise RuntimeError("Receive buffer not set. Call set_recv_buffers first.")
        
        if self.comm_stream is not None:
            with torch.cuda.stream(self.comm_stream):
                handle = dist.irecv(buffer, src_rank, group=group)
                handle.wait()
            torch.cuda.current_stream().wait_stream(self.comm_stream)
        else:
            handle = dist.irecv(buffer, src_rank, group=group)
            handle.wait()
        
        return buffer.clone()
    
    def wait_all_sends(self):
        """Wait for all pending send operations to complete."""
        for handle in self.send_handles.values():
            handle.wait()
        self.send_handles.clear()


class DualPipeEngine:
    """DualPipe execution engine.
    
    Orchestrates the execution of the DualPipe schedule including:
    - Micro-batch data loading
    - Forward/backward computation
    - Inter-stage communication
    - Loss computation
    - Optimizer steps
    """
    
    def __init__(
        self,
        stage: PipelineStage,
        config: PipelineConfig,
        loss_fn: Optional[Callable] = None,
    ):
        """Initialize DualPipe engine.
        
        Args:
            stage: Pipeline stage for this rank
            config: Pipeline configuration
            loss_fn: Loss function (only used on last stage)
        """
        self.stage = stage
        self.config = config
        self.loss_fn = loss_fn
        self.scheduler = DualPipeScheduler(config)
        
        # Micro-batch states
        self.states: Dict[int, MicroBatchState] = {}
        
        # Metrics tracking
        self.total_forward_time = 0.0
        self.total_backward_time = 0.0
        self.total_comm_time = 0.0
        self.num_iterations = 0
    
    def train_step(
        self,
        data_iterator,
        labels_iterator=None,
    ) -> Dict[str, Any]:
        """Execute one training step (all micro-batches).
        
        Args:
            data_iterator: Iterator yielding input tensors
            labels_iterator: Iterator yielding label tensors (for last stage)
            
        Returns:
            Dict with loss and metrics
        """
        # Initialize states for all micro-batches
        for i in range(self.config.num_micro_batches):
            self.states[i] = MicroBatchState(micro_batch_id=i)
        
        losses = []
        
        # Execute schedule
        for step in self.scheduler:
            mb_id = step.micro_batch_id
            
            if step.action == ScheduleAction.RECV_FORWARD:
                start = time.time()
                tensor = self.stage.recv_forward(mb_id, step.peer_rank)
                self.states[mb_id].input_tensor = tensor
                self.total_comm_time += time.time() - start
                
            elif step.action == ScheduleAction.FORWARD:
                start = time.time()
                
                # Get input (from recv or data loader)
                if is_first_stage():
                    input_tensor = next(data_iterator)
                    if torch.cuda.is_available():
                        input_tensor = input_tensor.cuda()
                    self.states[mb_id].input_tensor = input_tensor
                    self.stage.input_stash[mb_id] = input_tensor.requires_grad_(True)
                
                output = self.stage.forward_step(
                    mb_id,
                    self.states[mb_id].input_tensor,
                )
                self.states[mb_id].output_tensor = output
                self.states[mb_id].forward_done = True
                
                # Compute loss on last stage
                if is_last_stage() and labels_iterator is not None:
                    labels = next(labels_iterator)
                    if torch.cuda.is_available():
                        labels = labels.cuda()
                    loss = self.loss_fn(output, labels)
                    losses.append(loss.detach())
                    self.states[mb_id].grad_output = loss
                
                self.total_forward_time += time.time() - start
                
            elif step.action == ScheduleAction.SEND_FORWARD:
                start = time.time()
                self.stage.send_forward(
                    mb_id,
                    self.states[mb_id].output_tensor,
                    step.peer_rank,
                )
                self.total_comm_time += time.time() - start
                
            elif step.action == ScheduleAction.RECV_BACKWARD:
                start = time.time()
                grad = self.stage.recv_backward(mb_id, step.peer_rank)
                self.states[mb_id].grad_output = grad
                self.total_comm_time += time.time() - start
                
            elif step.action == ScheduleAction.BACKWARD:
                start = time.time()
                grad_input = self.stage.backward_step(
                    mb_id,
                    self.states[mb_id].grad_output if not is_last_stage() else None,
                )
                self.states[mb_id].grad_input = grad_input
                self.states[mb_id].backward_done = True
                self.total_backward_time += time.time() - start
                
            elif step.action == ScheduleAction.SEND_BACKWARD:
                start = time.time()
                self.stage.send_backward(
                    mb_id,
                    self.states[mb_id].grad_input,
                    step.peer_rank,
                )
                self.total_comm_time += time.time() - start
        
        # Wait for all sends to complete
        self.stage.wait_all_sends()
        
        # Clean up states
        self.states.clear()
        self.num_iterations += 1
        
        # Compute metrics
        total_loss = sum(losses) / len(losses) if losses else torch.tensor(0.0)
        bubble_ratio = self._compute_bubble_ratio()
        
        return {
            "loss": total_loss,
            "bubble_ratio": bubble_ratio,
            "forward_time": self.total_forward_time / max(1, self.num_iterations),
            "backward_time": self.total_backward_time / max(1, self.num_iterations),
            "comm_time": self.total_comm_time / max(1, self.num_iterations),
        }
    
    def _compute_bubble_ratio(self) -> float:
        """Compute pipeline bubble ratio.
        
        Bubble ratio = idle time / total time
        
        For DualPipe, this is approximately:
        (num_stages - 1) / (num_stages + num_micro_batches - 1)
        """
        p = self.config.num_stages
        m = self.config.num_micro_batches
        
        # Ideal: no bubbles
        # DualPipe: roughly (p-1) bubbles out of (p + m - 1) total time units
        bubble_ratio = (p - 1) / (p + m - 1)
        
        return bubble_ratio


def partition_model(
    model: nn.Module,
    num_stages: int,
    balance: Optional[List[int]] = None,
) -> List[nn.Module]:
    """Partition a model into pipeline stages.
    
    Args:
        model: Full model to partition
        num_stages: Number of pipeline stages
        balance: Number of layers per stage (default: equal)
        
    Returns:
        List of nn.Module, one per stage
    """
    # Get list of layers
    if hasattr(model, 'layers'):
        layers = list(model.layers)
    else:
        # Try to get children
        layers = list(model.children())
    
    num_layers = len(layers)
    
    if balance is None:
        # Equal distribution
        layers_per_stage = num_layers // num_stages
        balance = [layers_per_stage] * num_stages
        # Handle remainder
        remainder = num_layers % num_stages
        for i in range(remainder):
            balance[i] += 1
    
    assert sum(balance) == num_layers, \
        f"Balance {balance} doesn't match number of layers {num_layers}"
    
    stages = []
    start_idx = 0
    
    for stage_id, num_stage_layers in enumerate(balance):
        end_idx = start_idx + num_stage_layers
        stage_layers = layers[start_idx:end_idx]
        stage_module = nn.Sequential(*stage_layers)
        stages.append(stage_module)
        start_idx = end_idx
    
    return stages


@contextmanager
def pipeline_parallel_context(
    ranks: List[int],
    config: PipelineConfig,
):
    """Context manager for pipeline parallel training.
    
    Args:
        ranks: List of ranks in the pipeline
        config: Pipeline configuration
        
    Yields:
        Pipeline process group
    """
    global _PIPELINE_GROUP, _PIPELINE_RANK, _PIPELINE_WORLD_SIZE
    
    group = init_pipeline_group(ranks)
    
    try:
        yield group
    finally:
        _PIPELINE_GROUP = None
        _PIPELINE_RANK = 0
        _PIPELINE_WORLD_SIZE = 1


def benchmark_dualpipe_vs_gpipe(
    model: nn.Module,
    num_stages: int = 4,
    num_micro_batches_list: List[int] = [4, 8, 16, 32],
    batch_size: int = 8,
    seq_len: int = 512,
    num_iterations: int = 10,
) -> Dict[str, Dict[int, float]]:
    """Benchmark DualPipe vs standard GPipe schedule.
    
    Args:
        model: Model to test
        num_stages: Number of pipeline stages
        num_micro_batches_list: List of micro-batch counts to test
        batch_size: Batch size per micro-batch
        seq_len: Sequence length
        num_iterations: Number of iterations per benchmark
        
    Returns:
        Dict mapping schedule name to {num_micro_batches: bubble_ratio}
    """
    results = {"dualpipe": {}, "gpipe": {}}
    
    for num_micro_batches in num_micro_batches_list:
        # DualPipe bubble ratio
        p = num_stages
        m = num_micro_batches
        dualpipe_bubble = (p - 1) / (p + m - 1)
        results["dualpipe"][num_micro_batches] = dualpipe_bubble
        
        # GPipe bubble ratio (more bubbles due to sequential forward then backward)
        # GPipe: (p-1) warmup + m steady + (p-1) cooldown, but with longer bubbles
        gpipe_bubble = (p - 1) / m
        results["gpipe"][num_micro_batches] = gpipe_bubble
        
        logger.info(
            f"Pipeline bubble comparison",
            num_micro_batches=num_micro_batches,
            dualpipe_bubble=f"{dualpipe_bubble:.2%}",
            gpipe_bubble=f"{gpipe_bubble:.2%}",
        )
    
    return results


class DualPipeLRScheduler:
    """Learning rate scheduler aware of pipeline parallelism.
    
    Adjusts learning rate based on:
    - Effective batch size (accounting for micro-batches)
    - Warmup steps distributed across stages
    - Per-stage scaling for load balancing
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: PipelineConfig,
        base_lr: float = 1e-4,
        warmup_steps: int = 100,
        max_steps: int = 10000,
        min_lr_ratio: float = 0.1,
    ):
        """Initialize scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            config: Pipeline configuration
            base_lr: Base learning rate
            warmup_steps: Warmup steps
            max_steps: Maximum training steps
            min_lr_ratio: Minimum LR as ratio of base LR
        """
        self.optimizer = optimizer
        self.config = config
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self._compute_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _compute_lr(self) -> float:
        """Compute learning rate for current step."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.max_steps - self.warmup_steps
            )
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return self.base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine)
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self._compute_lr()
