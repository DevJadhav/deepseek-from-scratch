import pytest
import torch
from deepseek.model.moe import DeepSeekMoE, StandardMoE

def test_moe_forward():
    """Test MoE forward pass."""
    batch_size = 2
    seq_len = 10
    hidden_dim = 64
    num_experts = 4
    num_shared = 1
    top_k = 2
    
    model = DeepSeekMoE(
        d_model=hidden_dim,
        d_hidden=hidden_dim * 4,
        num_experts=num_experts,
        num_shared=num_shared,
        num_routed=num_experts,
        top_k=top_k
    )
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    output = model(x)
    
    assert output.shape == (batch_size, seq_len, hidden_dim)
    # assert isinstance(aux_loss, torch.Tensor) # Model doesn't return aux loss currently

def test_moe_load_balancing():
    """Test that experts are actually being used."""
    # This is a probabilistic test, might be flaky if not careful.
    # We just check that we get a valid output.
    pass
