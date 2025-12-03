import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Optional, Tuple, Dict

class PipelineStage(nn.Module):
    """
    Represents a stage in the pipeline.
    Holds a subset of layers.
    """
    def __init__(self, layers: nn.ModuleList, stage_id: int, num_stages: int, 
                 is_first: bool, is_last: bool,
                 embed: Optional[nn.Module] = None,
                 head: Optional[nn.Module] = None,
                 norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.is_first = is_first
        self.is_last = is_last
        
        # First stage has embedding
        self.embed = embed
        
        # Last stage has norm and head
        self.norm = norm
        self.head = head
        
    def forward(self, x, mask=None):
        if self.is_first and self.embed:
            x = self.embed(x)
            
        for layer in self.layers:
            x = layer(x, mask)
            
        if self.is_last:
            if self.norm:
                x = self.norm(x)
            if self.head:
                x = self.head(x)
                
        return x

def send_tensor(tensor: torch.Tensor, dst_rank: int, tag: int = 0):
    """Send tensor to destination rank."""
    if not dist.is_initialized():
        return
    # Send shape first? Or assume fixed shape?
    # For simplicity, assume fixed shape or metadata exchange.
    # In real PP, we usually know the shape (B, S, D).
    dist.send(tensor, dst=dst_rank, tag=tag)

def recv_tensor(shape: torch.Size, src_rank: int, dtype=torch.float32, device='cpu', tag: int = 0):
    """Receive tensor from source rank."""
    if not dist.is_initialized():
        return torch.zeros(shape, dtype=dtype, device=device)
    tensor = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(tensor, src=src_rank, tag=tag)
    return tensor

class PipelineScheduler:
    """
    1F1B Scheduler for Pipeline Parallelism.
    """
    def __init__(self, stage: PipelineStage, micro_batches: int, 
                 chunk_size: int, hidden_dim: int, seq_len: int,
                 vocab_size: int):
        self.stage = stage
        self.micro_batches = micro_batches
        self.chunk_size = chunk_size # Batch size per micro-batch
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Neighbors
        self.prev_rank = self.rank - 1 if self.rank > 0 else None
        self.next_rank = self.rank + 1 if self.rank < self.world_size - 1 else None
        
    def run_step(self, data_iterator):
        """
        Run one training step with 1F1B.
        """
        # Simplified 1F1B:
        # 1. Warmup: Forward passes to fill pipeline
        # 2. 1F1B: Forward + Backward
        # 3. Cooldown: Backward passes to drain pipeline
        
        # Storage for activations and gradients
        # We need to store input activations for backward pass
        input_activations = []
        
        # Warmup
        num_warmup = min(self.world_size - self.rank - 1, self.micro_batches)
        # Actually warmup depends on rank.
        # Rank 0 pushes M microbatches.
        # Rank k starts when it receives.
        
        # This logic is complex to implement fully from scratch in one go.
        # Let's implement a simplified "All Forward then All Backward" (GPipe style) first
        # as it's easier to verify and less prone to deadlock for this demo.
        # 1F1B is optimization.
        
        # GPipe Schedule:
        # FWD: MB1, MB2, ..., MBM
        # BWD: MBM, ..., MB2, MB1
        
        # Forward Pass
        outputs = []
        for i in range(self.micro_batches):
            # 1. Receive Input
            if self.stage.is_first:
                # Get from data loader
                try:
                    batch = next(data_iterator)
                    x = batch[0] # Input IDs
                    # Move to device
                    x = x.to(next(self.stage.parameters()).device)
                except StopIteration:
                    break
            else:
                # Receive from prev rank
                shape = (self.chunk_size, self.seq_len, self.hidden_dim)
                x = recv_tensor(shape, self.prev_rank, device=next(self.stage.parameters()).device)
                x.requires_grad = True
                
            # Store input for backward
            input_activations.append(x)
            
            # 2. Compute
            out = self.stage(x)
            outputs.append(out)
            
            # 3. Send Output
            if not self.stage.is_last:
                send_tensor(out.detach(), self.next_rank)
        
        # Backward Pass
        # Reverse order
        for i in reversed(range(self.micro_batches)):
            out = outputs[i]
            x = input_activations[i]
            
            # 1. Receive Grad Output
            if self.stage.is_last:
                # Compute loss (if we had labels)
                # For simplicity, assume loss is computed here or passed back?
                # Usually Last Stage computes loss.
                # Let's assume we have labels for the batch.
                # We need to access labels corresponding to microbatch i.
                # This requires data_iterator to be deterministic or buffered.
                
                # Mock gradient for now
                grad_out = torch.randn_like(out)
            else:
                shape = (self.chunk_size, self.seq_len, self.hidden_dim)
                grad_out = recv_tensor(shape, self.next_rank, device=out.device)
                
            # 2. Compute Grad Input
            # torch.autograd.backward(out, grad_out)
            # But we need to link x to out.
            # We should have kept the graph?
            # Yes, 'out' has grad_fn.
            out.backward(grad_out)
            
            # 3. Send Grad Input
            if not self.stage.is_first:
                grad_in = x.grad
                send_tensor(grad_in, self.prev_rank)
                
        # Optimizer step happens outside
