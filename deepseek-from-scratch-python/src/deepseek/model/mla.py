import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from deepseek.model.parallel import ColumnParallelLinear, RowParallelLinear

class KVCache:
    """
    Key-Value Cache for efficient generation.
    """
    def __init__(self, max_batch_size, max_seq_len, n_heads, head_dim, dtype=torch.float32, device='cpu'):
        self.k_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        self.v_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        self.current_seq_len = 0
        
    def update(self, k, v):
        """
        Update cache with new k, v.
        k, v: (B, H, S_new, D)
        """
        batch_size, n_heads, seq_len, head_dim = k.shape
        start_pos = self.current_seq_len
        end_pos = start_pos + seq_len
        
        self.k_cache[:batch_size, :, start_pos:end_pos, :] = k
        self.v_cache[:batch_size, :, start_pos:end_pos, :] = v
        
        self.current_seq_len = end_pos
        
        return self.k_cache[:batch_size, :, :end_pos, :], self.v_cache[:batch_size, :, :end_pos, :]


# ============================================================================
# RoPE Implementations
# ============================================================================

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        # Create inverse frequency bands
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len=None):
        # x: (B, H, Seq, D)
        if seq_len is None:
            seq_len = x.shape[2]
            
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq) # (Seq, D/2)
        emb = torch.cat((freqs, freqs), dim=-1) # (Seq, D)
        
        # Reshape for broadcasting: (1, 1, Seq, D)
        return emb.unsqueeze(0).unsqueeze(0)


class ExtendedRoPEConfig:
    """Configuration for extended RoPE with NTK/YaRN scaling."""
    
    def __init__(
        self,
        d_head: int = 64,
        max_seq_len: int = 131072,
        base: float = 10000.0,
        scaling_type: str = "yarn",  # "none", "ntk", "yarn"
        ntk_alpha: float = 8.0,
        yarn_scale: float = 1.0,
        yarn_original_max_position: int = 4096,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
    ):
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.ntk_alpha = ntk_alpha
        self.yarn_scale = yarn_scale
        self.yarn_original_max_position = yarn_original_max_position
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow


class ExtendedRotaryPositionalEncoding(nn.Module):
    """
    Extended RoPE with NTK-aware scaling and YaRN interpolation.
    
    For 128K context support in DeepSeek-V3.2.
    
    Supports:
    - Standard RoPE
    - NTK-aware scaling (increases base frequency)
    - YaRN interpolation (blends frequencies for long context)
    """
    
    def __init__(self, config: ExtendedRoPEConfig):
        super().__init__()
        self.config = config
        self.dim = config.d_head
        
        # Compute base frequencies with scaling
        if config.scaling_type == "ntk":
            scaled_base = config.base * (config.ntk_alpha ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (scaled_base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        elif config.scaling_type == "yarn":
            inv_freq = self._compute_yarn_inv_freq()
        else:
            inv_freq = 1.0 / (config.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        self.register_buffer('inv_freq', inv_freq)
        
        # YaRN attention scaling
        if config.scaling_type == "yarn":
            self.yarn_attn_factor = 0.1 * math.log(
                config.max_seq_len / config.yarn_original_max_position
            ) + 1.0
        else:
            self.yarn_attn_factor = 1.0
        
        # Cache
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None
        self._cache_seq_len = 0
    
    def _compute_yarn_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies with YaRN interpolation."""
        config = self.config
        base_inv_freq = 1.0 / (config.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        scale = config.max_seq_len / config.yarn_original_max_position
        
        freqs = torch.arange(0, self.dim // 2).float()
        low = torch.clamp((freqs - config.yarn_beta_slow) / (config.yarn_beta_fast - config.yarn_beta_slow), 0.0, 1.0)
        
        inv_freq = base_inv_freq * (1 - low) + (base_inv_freq / scale) * low
        return inv_freq
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cos/sin cache."""
        if seq_len <= self._cache_seq_len and self._cos_cache is not None:
            return
        
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self._cos_cache = emb.cos().to(dtype)
        self._sin_cache = emb.sin().to(dtype)
        self._cache_seq_len = seq_len
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply extended rotary positional encoding.
        
        Args:
            x: Input tensor [batch, heads, seq_len, dim]
            offset: Position offset for KV cache
            
        Returns:
            Position-encoded tensor
        """
        seq_len = x.shape[2]
        self._update_cache(offset + seq_len, x.device, x.dtype)
        
        cos = self._cos_cache[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cache[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        
        result = x * cos + rotated * sin
        
        if self.config.scaling_type == "yarn":
            result = result * self.yarn_attn_factor
        
        return result

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, pos_emb):
    # x: (B, H, Seq, D)
    # pos_emb: (1, 1, Seq, D)
    return (x * pos_emb.cos()) + (rotate_half(x) * pos_emb.sin())

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, d_rope, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent
        self.d_rope = d_rope
        self.head_dim = d_model // num_heads
        
        # Down-projection (Compression)
        self.kv_down_proj = ColumnParallelLinear(d_model, d_latent, bias=False)
        
        # Up-projection (Decompression)
        # We project to (num_heads * (head_dim - d_rope)) for content
        # and (num_heads * d_rope) is handled separately or part of it?
        # DeepSeek paper: KV are compressed. 
        # Let's follow the Rust implementation logic:
        # q_proj -> q_pe, q_unpe
        # kv_down -> latent
        # latent -> k_pe, k_unpe, v
        
        self.q_proj = ColumnParallelLinear(d_model, d_model, bias=False)
        
        # KV Up-projections
        # KV Up-projections
        # These operate on latent vector. 
        # If latent is sharded (ColumnParallel output), we need to handle that.
        # But wait, ColumnParallelLinear output is sharded along output dim usually?
        # Let's assume standard Megatron style:
        # Layer 1: ColumnParallel (splits output) -> Output is sharded [H1, H2]
        # Layer 2: RowParallel (splits input) -> Input is sharded [H1, H2]
        # So if kv_down_proj is ColumnParallel, latent is sharded.
        # Then kv_up_proj should be RowParallel? No, that would sum reduce.
        # We want to produce Heads.
        
        # Actually, for Attention:
        # QKV Proj: ColumnParallel (splits heads)
        # Output Proj: RowParallel (reduces heads)
        
        # Here we have compression.
        # kv_down_proj (d_model -> d_latent). If we split d_latent, then latent is sharded.
        # kv_up_proj (d_latent -> heads). If latent is sharded, we need RowParallel?
        # Or we can replicate d_latent (ColumnParallel with gather_output=True).
        
        # Let's keep it simple:
        # q_proj: ColumnParallel (split heads)
        # kv_down: ColumnParallel (split latent?) Or replicate?
        # If we split latent, then up_proj needs to take sharded input.
        
        # Let's assume we split heads.
        # So q_proj splits d_model (output).
        
        self.kv_up_proj_v = ColumnParallelLinear(d_latent, num_heads * self.head_dim, bias=False)
        self.kv_up_proj_k = ColumnParallelLinear(d_latent, num_heads * (self.head_dim - d_rope), bias=False)
        self.kv_up_proj_k_pe = ColumnParallelLinear(d_latent, d_rope, bias=False) # This one is tricky if shared.
        
        # If k_pe is shared, it shouldn't be split across heads?
        # Or we replicate it?
        # For now, let's use standard Linear for small shared parts or replicate.
        
        self.o_proj = RowParallelLinear(d_model, d_model, bias=False, input_is_parallel=True)
        self.rope = RotaryPositionalEncoding(d_rope, max_seq_len)

    def forward(self, x, mask=None, kv_cache: Optional[KVCache] = None):
        batch_size, seq_len, _ = x.shape
        
        # 1. Query Processing
        q = self.q_proj(x) # (B, Seq, D)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, Seq, D_head)
        
        # Split Q into PE and Content parts
        q_pe = q[..., :self.d_rope]
        q_unpe = q[..., self.d_rope:]
        
        # Apply RoPE to Q_pe
        # If using cache, we need correct position indices
        if kv_cache:
            start_pos = kv_cache.current_seq_len
            pos_emb = self.rope(q_pe, seq_len + start_pos)
            # We only need pos_emb for the new tokens
            pos_emb = pos_emb[:, :, start_pos:, :]
        else:
            pos_emb = self.rope(q_pe)
            
        q_pe = apply_rotary_pos_emb(q_pe, pos_emb)
        
        # Reassemble Q
        q = torch.cat([q_pe, q_unpe], dim=-1)
        
        # 2. KV Processing (Compressed)
        latent = self.kv_down_proj(x) # (B, Seq, D_latent)
        
        # Decompress V
        v = self.kv_up_proj_v(latent) # (B, Seq, H * D_head)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Decompress K Content
        k_unpe = self.kv_up_proj_k(latent) # (B, Seq, H * (D_head - D_rope))
        k_unpe = k_unpe.view(batch_size, seq_len, self.num_heads, self.head_dim - self.d_rope).transpose(1, 2)
        
        # Decompress K RoPE (Shared)
        k_pe = self.kv_up_proj_k_pe(latent) # (B, Seq, D_rope)
        k_pe = k_pe.unsqueeze(1) # (B, 1, Seq, D_rope)
        k_pe = apply_rotary_pos_emb(k_pe, pos_emb)
        # Broadcast K_pe to all heads
        k_pe = k_pe.expand(-1, self.num_heads, -1, -1)
        
        # Concatenate K
        k = torch.cat([k_pe, k_unpe], dim=-1)
        
        # KV Cache Update
        if kv_cache:
            k, v = kv_cache.update(k, v)
        
        # 3. Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Adjust mask for cache
            if kv_cache:
                # Mask should match (Seq_new, Seq_total)
                pass
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output)

class DeepSeekAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, d_rope, max_seq_len=2048):
        super().__init__()
        self.mla = MultiHeadLatentAttention(d_model, num_heads, d_latent, d_rope, max_seq_len)
        
    def forward(self, x, mask=None, kv_cache: Optional[KVCache] = None):
        return self.mla(x, mask, kv_cache)
