import pytest
import torch
from deepseek.torch.model.mla import MultiHeadLatentAttention, KVCache, LatentKVCache

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
    
    # 1. Full Forward with causal attention (no explicit mask, is_causal=True)
    with torch.no_grad():
        out_full = model(x, mask=None, is_causal=True)
        
    # 2. Cached Forward (Step by Step)
    cache = KVCache(batch_size, seq_len, num_heads, d_model // num_heads)
    out_cached = []
    
    with torch.no_grad():
        for i in range(seq_len):
            token = x[:, i:i+1, :] # (B, 1, D)
            out_step = model(token, kv_cache=cache, is_causal=False)  # is_causal=False when using cache
            out_cached.append(out_step)
            
    out_cached = torch.cat(out_cached, dim=1)
    
    # Compare
    # Note: Numerical differences might exist due to RoPE implementation details or float precision
    # But should be close. Using a slightly relaxed tolerance for numerical stability.
    diff = (out_full - out_cached).abs().max()
    print(f"Max difference: {diff}")
    assert torch.allclose(out_full, out_cached, atol=1e-3), f"Max difference: {diff}"

def test_kv_cache_update():
    """Test KV Cache update logic."""
    cache = KVCache(1, 10, 2, 4)
    k = torch.randn(1, 2, 1, 4)
    v = torch.randn(1, 2, 1, 4)
    
    k_out, v_out = cache.update(k, v)
    
    assert k_out.shape == (1, 2, 1, 4)
    assert cache.current_seq_len == 1
    assert torch.allclose(cache.k_cache[:, :, 0:1, :], k)


# ============================================================================
# LatentKVCache Tests (MLA Compression)
# ============================================================================

def test_latent_kv_cache_basic():
    """Test basic LatentKVCache functionality."""
    batch_size = 2
    max_seq_len = 64
    d_latent = 128
    
    cache = LatentKVCache(batch_size, max_seq_len, d_latent)
    
    # First update
    c_kv = torch.randn(batch_size, 10, d_latent)
    result = cache.update(c_kv)
    
    assert result.shape == (batch_size, 10, d_latent)
    assert cache.current_seq_len == 10
    
    # Second update
    c_kv2 = torch.randn(batch_size, 5, d_latent)
    result2 = cache.update(c_kv2)
    
    assert result2.shape == (batch_size, 15, d_latent)
    assert cache.current_seq_len == 15
    
    # Verify first part is unchanged
    assert torch.allclose(result2[:, :10, :], result)


def test_latent_kv_cache_reset():
    """Test LatentKVCache reset."""
    cache = LatentKVCache(1, 32, 64)
    
    c_kv = torch.randn(1, 10, 64)
    cache.update(c_kv)
    
    assert cache.current_seq_len == 10
    
    cache.reset()
    
    assert cache.current_seq_len == 0
    assert (cache.latent_cache == 0).all()


def test_latent_kv_cache_overflow():
    """Test LatentKVCache raises error on overflow."""
    cache = LatentKVCache(1, 10, 64)
    
    c_kv = torch.randn(1, 8, 64)
    cache.update(c_kv)
    
    # This should exceed max_seq_len
    c_kv2 = torch.randn(1, 5, 64)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        cache.update(c_kv2)


def test_latent_kv_cache_memory_savings():
    """Test that LatentKVCache achieves expected memory savings."""
    batch_size = 4
    seq_len = 1024
    n_heads = 32
    head_dim = 128
    d_latent = 512  # Typical DeepSeek-V3 latent dimension
    
    comparison = LatentKVCache.compare_memory_savings(
        batch_size=batch_size,
        seq_len=seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
        d_latent=d_latent,
    )
    
    # Standard KV cache: 2 × batch × heads × seq × head_dim × 4 bytes
    expected_standard = 2 * batch_size * n_heads * seq_len * head_dim * 4
    assert comparison["standard_kv_bytes"] == expected_standard
    
    # Latent cache: batch × seq × d_latent × 4 bytes
    expected_latent = batch_size * seq_len * d_latent * 4
    assert comparison["latent_cache_bytes"] == expected_latent
    
    # Memory reduction should be ~16× for these dimensions
    # (2 × 32 × 128) / 512 = 16
    expected_ratio = (2 * n_heads * head_dim) / d_latent
    assert abs(comparison["memory_reduction_ratio"] - expected_ratio) < 0.01
    
    print(f"Memory reduction: {comparison['memory_reduction_ratio']:.1f}×")
    print(f"Savings: {comparison['savings_percent']:.1f}%")


def test_latent_kv_cache_with_mla():
    """Test LatentKVCache integration with MultiHeadLatentAttention."""
    batch_size = 2
    seq_len = 16
    d_model = 64
    num_heads = 4
    d_latent = 32
    d_rope = 16
    max_seq_len = 64
    
    model = MultiHeadLatentAttention(d_model, num_heads, d_latent, d_rope, max_seq_len)
    model.eval()
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Full forward without cache
    with torch.no_grad():
        out_full = model(x, is_causal=True)
    
    # Forward with latent cache (step by step)
    latent_cache = LatentKVCache(batch_size, max_seq_len, d_latent)
    out_cached = []
    
    with torch.no_grad():
        for i in range(seq_len):
            token = x[:, i:i+1, :]
            out_step = model.forward_with_latent_cache(
                token, 
                latent_cache=latent_cache, 
                is_causal=False
            )
            out_cached.append(out_step)
    
    out_cached = torch.cat(out_cached, dim=1)
    
    # Compare outputs
    diff = (out_full - out_cached).abs().max()
    print(f"Max difference with latent cache: {diff}")
    # Note: There may be some numerical differences due to different computation order
    # but results should be close
    assert torch.allclose(out_full, out_cached, atol=1e-2), f"Max difference: {diff}"
