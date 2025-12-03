import pytest
import torch
import torch.nn as nn
from deepseek.model.parallel import ColumnParallelLinear, RowParallelLinear

def test_parallel_linear_shapes():
    """Test shapes of parallel linear layers (single device mock)."""
    # Note: On single device (world_size=1), these should behave like standard Linear
    
    batch_size = 2
    in_features = 16
    out_features = 32
    
    # Column Parallel
    col_linear = ColumnParallelLinear(in_features, out_features, bias=True)
    x = torch.randn(batch_size, in_features)
    y = col_linear(x)
    assert y.shape == (batch_size, out_features)
    
    # Row Parallel
    row_linear = RowParallelLinear(out_features, in_features, bias=True)
    x = torch.randn(batch_size, out_features)
    y = row_linear(x)
    assert y.shape == (batch_size, in_features)

def test_parallel_linear_gradients():
    """Test backward pass."""
    in_features = 4
    out_features = 4
    model = ColumnParallelLinear(in_features, out_features)
    x = torch.randn(2, in_features, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert model.weight.grad is not None
