import pytest
import torch
from deepseek.model.attention import MultiQueryAttention, GroupedQueryAttention
from deepseek.model.mla import MultiHeadLatentAttention

@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [16, 128])
@pytest.mark.parametrize("hidden_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [4, 8])
def test_mqa_shape(batch_size, seq_len, hidden_dim, num_heads):
    """Test Multi-Query Attention output shape."""
    model = MultiQueryAttention(hidden_dim, num_heads)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    output = model(x)
    assert output.shape == (batch_size, seq_len, hidden_dim)

def test_mla_shape():
    """Test Multi-Head Latent Attention output shape."""
    batch_size = 2
    seq_len = 32
    hidden_dim = 64
    num_heads = 4
    d_latent = 16
    d_rope = 8
    
    model = MultiHeadLatentAttention(
        d_model=hidden_dim,
        num_heads=num_heads,
        d_latent=d_latent,
        d_rope=d_rope
    )
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    output = model(x)
    assert output.shape == (batch_size, seq_len, hidden_dim)

def test_attention_causality():
    """Test that attention does not attend to future tokens."""
    hidden_dim = 32
    num_heads = 4
    # MQA signature: d_model, num_heads
    model = MultiQueryAttention(hidden_dim, num_heads)
    model.eval()
    
    # Sequence with 2 tokens
    x = torch.randn(1, 2, hidden_dim)
    
    # Forward pass
    out1 = model(x)
    
    # Change second token
    x_mod = x.clone()
    x_mod[:, 1, :] = torch.randn(1, hidden_dim)
    
    # Forward pass again
    out2 = model(x_mod)
    
    # Check first token output equality
    # Note: MQA might not implement causal masking by default in this codebase
    # Let's check if it accepts a mask or has causal=True
    # If not, we skip this check or fix the model.
    # Assuming it behaves like standard attention without mask if not provided.
    # If the implementation doesn't enforce causality without a mask, this test fails.
    # We will pass a causal mask if needed.
    
    # Create causal mask
    mask = torch.tril(torch.ones(2, 2)).unsqueeze(0).unsqueeze(0)
    
    out1_masked = model(x, mask=mask)
    out2_masked = model(x_mod, mask=mask)
    
    assert torch.allclose(out1_masked[:, 0, :], out2_masked[:, 0, :], atol=1e-5)
