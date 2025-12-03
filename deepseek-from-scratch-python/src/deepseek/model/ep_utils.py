import torch
import torch.distributed as dist
from typing import Optional, Tuple
from deepseek.utils.distributed import get_expert_model_parallel_group, get_expert_model_parallel_world_size

class _AllToAll(torch.autograd.Function):
    """
    All-to-All communication.
    Forward: Scatter input to all ranks.
    Backward: Gather gradients from all ranks.
    """
    
    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_, output_split_sizes=None, input_split_sizes=None):
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        
        if dist.is_initialized():
            group = get_expert_model_parallel_group()
            world_size = get_expert_model_parallel_world_size()
            if world_size > 1:
                # If split sizes are provided, use all_to_all_single (variable size)
                # Otherwise use all_to_all (equal size)
                # For simplicity in this implementation, we assume equal size or handle variable size if needed.
                # DeepSeek MoE usually has variable tokens per expert -> variable size.
                
                if output_split_sizes is not None and input_split_sizes is not None:
                    # Variable size all-to-all
                    # We need to compute output shape first?
                    # Usually we exchange shapes first or assume max capacity.
                    # PyTorch all_to_all_single takes: output, input, output_split_sizes, input_split_sizes
                    
                    # We need to allocate output tensor.
                    # Total output elements = sum(output_split_sizes) * hidden_dim
                    # Input shape: (Total_Input_Tokens, Hidden)
                    
                    hidden_dim = input_.size(-1)
                    total_output_tokens = sum(output_split_sizes)
                    output = torch.empty(
                        (total_output_tokens, hidden_dim),
                        device=input_.device,
                        dtype=input_.dtype
                    )
                    
                    dist.all_to_all_single(
                        output,
                        input_,
                        output_split_sizes=output_split_sizes,
                        input_split_sizes=input_split_sizes,
                        group=group
                    )
                    return output
                else:
                    # Equal size
                    # input: (World, TokensPerRank, Hidden) -> output: (World, TokensPerRank, Hidden)
                    output = torch.empty_like(input_)
                    dist.all_to_all_single(output, input_, group=group)
                    return output
                    
        return input_
    
    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_initialized():
            group = get_expert_model_parallel_group()
            world_size = get_expert_model_parallel_world_size()
            if world_size > 1:
                input_split_sizes = ctx.input_split_sizes
                output_split_sizes = ctx.output_split_sizes
                
                if input_split_sizes is not None and output_split_sizes is not None:
                    # Reverse direction: Output becomes Input for backward
                    # We send back gradients.
                    # Forward: Input (Split A) -> Output (Split B)
                    # Backward: GradOutput (Split B) -> GradInput (Split A)
                    
                    hidden_dim = grad_output.size(-1)
                    total_input_tokens = sum(input_split_sizes)
                    grad_input = torch.empty(
                        (total_input_tokens, hidden_dim),
                        device=grad_output.device,
                        dtype=grad_output.dtype
                    )
                    
                    dist.all_to_all_single(
                        grad_input,
                        grad_output,
                        output_split_sizes=input_split_sizes, # Recipient expects input_split_sizes
                        input_split_sizes=output_split_sizes, # Sender sends output_split_sizes
                        group=group
                    )
                    return grad_input, None, None
                else:
                    grad_input = torch.empty_like(grad_output)
                    dist.all_to_all_single(grad_input, grad_output, group=group)
                    return grad_input, None, None
                    
        return grad_output, None, None

def all_to_all(input_, output_split_sizes=None, input_split_sizes=None):
    return _AllToAll.apply(input_, output_split_sizes, input_split_sizes)
