# DeepSeek Attention

## Overview

DeepSeek Attention combines **Multi-Head Latent Attention (MLA)** with **Decoupled Rotary Position Embedding (RoPE)**. The key innovation is separating content information (which can be compressed) from positional information (which needs dedicated handling).

**Paper:** [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)

---

## When to Use DeepSeek Attention

### ✅ Use When:
- **Building production LLMs** - State-of-the-art efficiency
- **Long-context models** - 100k+ token support
- **Need both efficiency and quality** - Best of both worlds
- **Using RoPE for positions** - Decoupled design preserves RoPE benefits

### ❌ Don't Use When:
- **Simple projects** - MQA/GQA are simpler
- **Short contexts only** - Overhead not worth it
- **Not using RoPE** - Design assumes rotary embeddings

---

## The Problem: Why Decouple?

### Standard RoPE + MLA Issue

In standard RoPE, positional information is baked into K and Q:

$$Q_{rope} = Q \cdot R_{\theta}(pos)$$
$$K_{rope} = K \cdot R_{\theta}(pos)$$

Where $R_{\theta}(pos)$ is the rotation matrix for position $pos$.

**Problem with MLA compression:**
- If we compress $K_{rope}$, positional information is also compressed
- RoPE relies on precise rotations; compression corrupts them
- Result: Degraded position-awareness, especially for long sequences

### Solution: Decoupled Paths

Separate content and position:

```
Content Path (can compress):
X → W_DKV → C_KV → [W_UK → K_c, W_UV → V_c]

Position Path (must preserve):
X → W_KR → K_r → RoPE(K_r)
X → W_QR → Q_r → RoPE(Q_r)
```

---

## Mathematical Foundation

### Decoupled Projections

**Content Projections (Compressed):**

$$C_{KV} = XW^{DKV} \in \mathbb{R}^{B \times L \times d_c}$$
$$K^C = C_{KV}W^{UK} \in \mathbb{R}^{B \times L \times D}$$
$$V = C_{KV}W^{UV} \in \mathbb{R}^{B \times L \times D}$$

**Position Projections (Not Compressed):**

$$K^R = XW^{KR} \in \mathbb{R}^{B \times L \times d_r}$$
$$Q^R = XW^{QR} \in \mathbb{R}^{B \times L \times H \cdot d_r}$$

Where $d_r$ is a small dimension for RoPE (typically 64-128).

### Applying RoPE

RoPE is applied **only to positional components**:

$$\hat{K}^R = \text{RoPE}(K^R)$$
$$\hat{Q}^R = \text{RoPE}(Q^R)$$

### Combining Content and Position

**Keys** are concatenated:
$$K = [K^C \| \hat{K}^R]$$

**Queries** are concatenated per head:
$$Q_h = [Q_h^C \| \hat{Q}_h^R]$$

The attention dimension becomes $d_h + d_r$ per head.

### Final Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_h + d_r}}\right)V$$

Note: The scaling factor includes both content and position dimensions.

---

## RoPE Review

### What is RoPE?

Rotary Position Embedding encodes position through rotation in 2D subspaces:

For position $m$ and dimension pairs $(i, i+1)$:

$$R_\theta(m) = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

Where $\theta_i = 10000^{-2i/d}$.

### RoPE Properties

1. **Relative Position**: $Q_m^T K_n$ depends only on $m - n$
2. **Decays with Distance**: Attention naturally decays for far positions
3. **Efficient**: Applied element-wise, no position embeddings stored

### RoPE Implementation

```python
def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding.
    
    Args:
        x: (B, H, L, d_r) - queries or keys
        freqs: (L, d_r/2) - precomputed frequencies
    """
    # Split into pairs
    x1, x2 = x[..., ::2], x[..., 1::2]  # Even, odd dims
    
    cos = freqs.cos()
    sin = freqs.sin()
    
    # Rotate pairs
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    
    # Interleave back
    return torch.stack([out1, out2], dim=-1).flatten(-2)
```

---

## Implementation Details

### Rust Implementation

```rust
pub struct DeepSeekAttention {
    n_heads: usize,
    d_head: usize,
    d_latent: usize,
    d_rope: usize,
    
    // Content path (compressed)
    w_dkv: Linear,    // D → d_latent
    w_dq: Linear,     // D → d_latent
    w_uk: Linear,     // d_latent → H * d_h
    w_uv: Linear,     // d_latent → H * d_h
    w_uq: Linear,     // d_latent → H * d_h
    
    // Position path (not compressed)
    w_kr: Linear,     // D → d_rope (shared across heads)
    w_qr: Linear,     // D → H * d_rope
    
    // Output
    w_o: Linear,
    
    // Precomputed RoPE frequencies
    rope_freqs: Tensor,
}

impl DeepSeekAttention {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        
        // === Content Path (Compressed) ===
        let c_kv = self.w_dkv.forward(x)?;
        let c_q = self.w_dq.forward(x)?;
        
        let k_c = self.w_uk.forward(&c_kv)?
            .reshape((b, t, self.n_heads, self.d_head))?
            .transpose(1, 2)?;
        let v = self.w_uv.forward(&c_kv)?
            .reshape((b, t, self.n_heads, self.d_head))?
            .transpose(1, 2)?;
        let q_c = self.w_uq.forward(&c_q)?
            .reshape((b, t, self.n_heads, self.d_head))?
            .transpose(1, 2)?;
        
        // === Position Path (RoPE) ===
        let k_r = self.w_kr.forward(x)?
            .reshape((b, t, 1, self.d_rope))?
            .transpose(1, 2)?;  // (B, 1, T, d_rope) - shared
        let q_r = self.w_qr.forward(x)?
            .reshape((b, t, self.n_heads, self.d_rope))?
            .transpose(1, 2)?;  // (B, H, T, d_rope)
        
        // Apply RoPE
        let k_r = apply_rope(&k_r, &self.rope_freqs)?;
        let q_r = apply_rope(&q_r, &self.rope_freqs)?;
        
        // Broadcast k_r to all heads
        let k_r = k_r.broadcast_to((b, self.n_heads, t, self.d_rope))?;
        
        // === Combine Content + Position ===
        let k = Tensor::cat(&[&k_c, &k_r], D::Minus1)?;  // (B, H, T, d_h + d_rope)
        let q = Tensor::cat(&[&q_c, &q_r], D::Minus1)?;  // (B, H, T, d_h + d_rope)
        
        // === Attention ===
        let total_dim = (self.d_head + self.d_rope) as f64;
        let scale = 1.0 / total_dim.sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?)? * scale;
        
        // Causal mask
        let mask = create_causal_mask(t, x.device())?;
        let scores = scores.broadcast_add(&mask)?;
        
        let attn = softmax(&scores, D::Minus1)?;
        
        // V only uses content path
        let out = attn.matmul(&v)?;
        
        // Output projection
        let out = out.transpose(1, 2)?
            .reshape((b, t, self.n_heads * self.d_head))?;
        self.w_o.forward(&out)
    }
}

fn apply_rope(x: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let (b, h, t, d) = x.dims4()?;
    let half_d = d / 2;
    
    // Split into pairs
    let x1 = x.narrow(D::Minus1, 0, half_d)?;
    let x2 = x.narrow(D::Minus1, half_d, half_d)?;
    
    // Get cos/sin for this sequence length
    let freqs = freqs.narrow(0, 0, t)?;
    let cos = freqs.cos()?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = freqs.sin()?.unsqueeze(0)?.unsqueeze(0)?;
    
    // Rotate
    let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let out2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
    
    Tensor::cat(&[&out1, &out2], D::Minus1)
}
```

### Python Implementation

```python
class DeepSeekAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_latent: int,
        d_rope: int,
        max_seq_len: int = 8192
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_latent = d_latent
        self.d_rope = d_rope
        
        # Content path (compressed)
        self.W_DKV = nn.Linear(d_model, d_latent, bias=False)
        self.W_DQ = nn.Linear(d_model, d_latent, bias=False)
        self.W_UK = nn.Linear(d_latent, d_model, bias=False)
        self.W_UV = nn.Linear(d_latent, d_model, bias=False)
        self.W_UQ = nn.Linear(d_latent, d_model, bias=False)
        
        # Position path (not compressed)
        self.W_KR = nn.Linear(d_model, d_rope, bias=False)  # Shared
        self.W_QR = nn.Linear(d_model, num_heads * d_rope, bias=False)
        
        # Output
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        # Precompute RoPE frequencies
        self.register_buffer(
            'rope_freqs', 
            self._compute_freqs(max_seq_len, d_rope)
        )
    
    def _compute_freqs(self, max_len: int, dim: int) -> torch.Tensor:
        theta = 10000.0 ** (-torch.arange(0, dim, 2).float() / dim)
        positions = torch.arange(max_len).float()
        freqs = positions.unsqueeze(1) * theta.unsqueeze(0)
        return freqs  # (max_len, dim/2)
    
    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to x of shape (B, H, L, d_rope)."""
        L = x.size(2)
        freqs = self.rope_freqs[:L]  # (L, d_rope/2)
        
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos = freqs.cos().view(1, 1, L, -1)
        sin = freqs.sin().view(1, 1, L, -1)
        
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        
        return torch.stack([out1, out2], dim=-1).flatten(-2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # === Content Path ===
        c_kv = self.W_DKV(x)
        c_q = self.W_DQ(x)
        
        k_c = self.W_UK(c_kv).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_UV(c_kv).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        q_c = self.W_UQ(c_q).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        
        # === Position Path ===
        k_r = self.W_KR(x).view(B, L, 1, self.d_rope).transpose(1, 2)  # Shared
        q_r = self.W_QR(x).view(B, L, self.num_heads, self.d_rope).transpose(1, 2)
        
        # Apply RoPE
        k_r = self._apply_rope(k_r)
        q_r = self._apply_rope(q_r)
        
        # Broadcast k_r to all heads
        k_r = k_r.expand(-1, self.num_heads, -1, -1)
        
        # === Combine ===
        k = torch.cat([k_c, k_r], dim=-1)  # (B, H, L, d_h + d_rope)
        q = torch.cat([q_c, q_r], dim=-1)
        
        # === Attention ===
        scale = 1.0 / math.sqrt(self.d_head + self.d_rope)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, L, d_h)
        
        # Output
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.W_O(out)
```

---

## KV Cache Strategy

### What to Cache

In DeepSeek Attention, we have options:

**Option 1: Cache Compressed Latent Only**
```
Cache: C_KV (size: L × d_c)
Pros: Maximum memory savings
Cons: Must reconstruct K, V on every forward
```

**Option 2: Cache Compressed Latent + Position**
```
Cache: [C_KV, K_R] (size: L × (d_c + d_r))
Pros: Balanced approach, position preserved
Cons: Slightly more memory
```

**Option 3: Cache Full K, V (No MLA Benefit)**
```
Cache: [K, V] (size: L × 2D)
Pros: No reconstruction overhead
Cons: Loses MLA memory benefit
```

### Recommended Strategy

Cache `[C_KV, K_R_with_rope]`:
- Latent for content (compressed)
- Position keys with RoPE already applied (no recomputation)

```python
class DeepSeekAttentionWithCache(nn.Module):
    def forward_with_cache(
        self, 
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        # ... (projections as before) ...
        
        # Cache: (C_KV_past, K_R_past_with_rope)
        if cache is not None:
            c_kv_past, k_r_past = cache
            c_kv_full = torch.cat([c_kv_past, c_kv], dim=1)
            k_r_full = torch.cat([k_r_past, k_r_with_rope], dim=2)
        else:
            c_kv_full = c_kv
            k_r_full = k_r_with_rope
        
        # Reconstruct K, V from full latent
        k_c = self.W_UK(c_kv_full)
        v = self.W_UV(c_kv_full)
        
        # Combine with position
        k = torch.cat([k_c, k_r_full], dim=-1)
        
        # ... attention ...
        
        new_cache = (c_kv_full, k_r_full)
        return output, new_cache
```

---

## Performance Analysis

### Memory Comparison

For model with $D=4096$, $H=32$, $d_c=512$, $d_r=64$:

| Method | KV Cache per Token | Ratio |
|--------|-------------------|-------|
| MHA | $2 \times 4096 = 8192$ | 1.0x |
| MQA | $2 \times 128 = 256$ | 0.03x |
| MLA | $512$ | 0.06x |
| **DeepSeek** | $512 + 64 = 576$ | 0.07x |

### Compute Overhead

Additional compute for DeepSeek vs MHA:
1. Content compression/reconstruction: ~$1.25BD^2$
2. Position projections: ~$0.1BD^2$
3. RoPE application: ~$O(BLd_r)$ (negligible)

**Total overhead**: ~15-20% more compute, 93% less KV memory.

---

## Practical Tips

1. **Choose $d_r$ wisely**: 64-128 is typical; larger for longer contexts
2. **Precompute RoPE freqs**: Avoid recomputing on every forward
3. **Fuse projections**: Combine $W^{UK}$ and $W^{KR}$ output into single kernel
4. **Flash Attention compatibility**: Concatenated K works with FlashAttention
5. **Position extrapolation**: RoPE enables length extrapolation if trained correctly

---

## References

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)
