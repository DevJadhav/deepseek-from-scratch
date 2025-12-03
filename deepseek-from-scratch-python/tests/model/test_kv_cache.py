import pytest
import torch
from deepseek.model.mla import MultiHeadLatentAttention, KVCache

def test_kv_cache_correctness():
    """Test that KV Cache produces same output as full sequence."""
    batch_size = 1
    seq_len = 10
    d_model = 32
    num_heads = 4
    d_latent = 16
    d_rope = 8
    
    model = MultiHeadLatentAttention(d_model, num_heads, d_latent, d_rope)
    model.eval()
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    # 1. Full Forward
    with torch.no_grad():
        out_full = model(x, mask=mask)
        
    # 2. Cached Forward (Step by Step)
    cache = KVCache(batch_size, seq_len, num_heads, d_model // num_heads)
    out_cached = []
    
    with torch.no_grad():
        for i in range(seq_len):
            token = x[:, i:i+1, :] # (B, 1, D)
            # We need to handle mask if used, but here no mask
            out_step = model(token, kv_cache=cache)
            out_cached.append(out_step)
            
    out_cached = torch.cat(out_cached, dim=1)
    
    # Compare
    # Note: Numerical differences might exist due to RoPE implementation details or float precision
    # But should be close.
    # Compare
    diff = (out_full - out_cached).abs().max()
    print(f"Max difference: {diff}")
    assert torch.allclose(out_full, out_cached, atol=1e-4)

def test_kv_cache_update():
    """Test KV Cache update logic."""
    cache = KVCache(1, 10, 2, 4)
    k = torch.randn(1, 2, 1, 4)
    v = torch.randn(1, 2, 1, 4)
    
    k_out, v_out = cache.update(k, v)
    
    assert k_out.shape == (1, 2, 1, 4)
    assert cache.current_seq_len == 1
    assert torch.allclose(cache.k_cache[:, :, 0:1, :], k)
