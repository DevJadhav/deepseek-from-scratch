# Multi-Head Latent Attention (MLA)

## Overview

Multi-Head Latent Attention (MLA) is DeepSeek's innovation that compresses Key-Value representations into a **low-dimensional latent space** before storing in the KV cache. This achieves even greater memory savings than MQA/GQA while maintaining model expressivity through learned up-projections.

**Paper:** [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)

---

## When to Use MLA

### ✅ Use MLA When:
- **Very long contexts** - 100k+ token sequences
- **Large-scale models** - 20B+ parameters
- **Production inference** - Need maximum memory efficiency
- **KV cache is the bottleneck** - Attention compute is not limiting

### ❌ Don't Use MLA When:
- **Small models** - Compression overhead not worth it
- **Short sequences** - MQA/GQA sufficient
- **Training from scratch** - More complex to implement correctly

---

## Mathematical Foundation

### Key Insight

Instead of caching full Key-Value representations:
- **Standard**: Cache $K \in \mathbb{R}^{L \times D}$, $V \in \mathbb{R}^{L \times D}$
- **MLA**: Cache $C_{KV} \in \mathbb{R}^{L \times d_c}$ where $d_c \ll D$

### Compression Pipeline

**Step 1: Down-Projection (Compression)**

Input $X \in \mathbb{R}^{B \times L \times D}$ is compressed to a latent representation:

$$C_{KV} = XW^{DKV}$$

Where $W^{DKV} \in \mathbb{R}^{D \times d_c}$ and $d_c$ is the compressed dimension (typically $D/4$ to $D/8$).

**Step 2: Up-Projection (Reconstruction)**

Keys and Values are reconstructed from the latent:

$$K^C = C_{KV} W^{UK}, \quad V^C = C_{KV} W^{UV}$$

Where $W^{UK}, W^{UV} \in \mathbb{R}^{d_c \times D}$.

**Step 3: Query Compression (Optional)**

Queries can also be compressed for additional efficiency:

$$C_Q = XW^{DQ}$$
$$Q = C_Q W^{UQ}$$

### Complete MLA Equations

For input $X$:

$$C_{KV} = XW^{DKV} \quad \text{(compress KV)}$$
$$C_Q = XW^{DQ} \quad \text{(compress Q)}$$

$$K = C_{KV} W^{UK} \quad \text{(reconstruct K)}$$
$$V = C_{KV} W^{UV} \quad \text{(reconstruct V)}$$
$$Q = C_Q W^{UQ} \quad \text{(reconstruct Q)}$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right)V$$

### Memory Analysis

| Method | KV Cache per Token | Relative Size |
|--------|-------------------|---------------|
| MHA | $2D$ | 1.0x |
| MQA | $2d_h$ | $1/H$ |
| GQA-G | $2Gd_h$ | $G/H$ |
| **MLA** | $d_c$ | $d_c/(2D)$ |

For $D = 4096$, $d_c = 512$:
- **MLA saves 16x** compared to MHA
- **MLA saves 4x** compared to MQA (with $H=32$)

---

## Visual Representation

```
Standard MHA KV Cache:
┌──────────────────────────────────────┐
│ K: [L × H × d_h] = [L × D]           │  ← Full dimension
│ V: [L × H × d_h] = [L × D]           │
│ Total: 2 × L × D                     │
└──────────────────────────────────────┘

MLA KV Cache:
┌──────────────────────────────────────┐
│ C_KV: [L × d_c]                      │  ← Compressed!
│ Total: L × d_c  (d_c << D)           │
└──────────────────────────────────────┘

Reconstruction (on-the-fly):
C_KV ──┬── W^UK ──► K [L × D]
       │
       └── W^UV ──► V [L × D]
```

---

## Implementation Details

### Rust Implementation

```rust
pub struct MultiHeadLatentAttention {
    n_heads: usize,
    d_head: usize,
    d_latent: usize,  // Compressed dimension
    
    // Down-projection (compression)
    w_dkv: Linear,    // D → d_latent
    w_dq: Linear,     // D → d_latent
    
    // Up-projection (reconstruction)  
    w_uk: Linear,     // d_latent → H * d_h
    w_uv: Linear,     // d_latent → H * d_h
    w_uq: Linear,     // d_latent → H * d_h
    
    // Output
    w_o: Linear,      // H * d_h → D
}

impl MultiHeadLatentAttention {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        
        // Compress to latent space
        let c_kv = self.w_dkv.forward(x)?;  // (B, T, d_latent)
        let c_q = self.w_dq.forward(x)?;    // (B, T, d_latent)
        
        // Reconstruct Q, K, V from latent
        let q = self.w_uq.forward(&c_q)?
            .reshape((b, t, self.n_heads, self.d_head))?
            .transpose(1, 2)?;
        let k = self.w_uk.forward(&c_kv)?
            .reshape((b, t, self.n_heads, self.d_head))?
            .transpose(1, 2)?;
        let v = self.w_uv.forward(&c_kv)?
            .reshape((b, t, self.n_heads, self.d_head))?
            .transpose(1, 2)?;
        
        // Standard attention
        let scale = 1.0 / (self.d_head as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?)? * scale;
        
        // Causal mask
        let mask = create_causal_mask(t, x.device())?;
        let scores = scores.broadcast_add(&mask)?;
        
        let attn = softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?;
        
        // Output projection
        let out = out.transpose(1, 2)?
            .reshape((b, t, self.n_heads * self.d_head))?;
        self.w_o.forward(&out)
    }
    
    // For inference: cache only the compressed latent
    pub fn forward_with_cache(
        &self, 
        x: &Tensor, 
        cache: Option<&Tensor>
    ) -> Result<(Tensor, Tensor)> {
        let c_kv = self.w_dkv.forward(x)?;
        
        // Append to cache (only store compressed representation!)
        let c_kv_full = match cache {
            Some(prev) => Tensor::cat(&[prev, &c_kv], 1)?,
            None => c_kv.clone(),
        };
        
        // Reconstruct K, V from full cache
        let k = self.w_uk.forward(&c_kv_full)?;
        let v = self.w_uv.forward(&c_kv_full)?;
        
        // ... rest of attention computation
        
        Ok((output, c_kv_full))  // Return new cache
    }
}
```

### Python Implementation

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_latent: int,
        d_rope: int = None  # Optional positional dim
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_latent = d_latent
        
        # Down-projection (compression)
        self.W_DKV = nn.Linear(d_model, d_latent, bias=False)
        self.W_DQ = nn.Linear(d_model, d_latent, bias=False)
        
        # Up-projection (reconstruction)
        self.W_UK = nn.Linear(d_latent, d_model, bias=False)
        self.W_UV = nn.Linear(d_latent, d_model, bias=False)
        self.W_UQ = nn.Linear(d_latent, d_model, bias=False)
        
        # Output
        self.W_O = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # Compress to latent
        c_kv = self.W_DKV(x)  # (B, L, d_latent)
        c_q = self.W_DQ(x)    # (B, L, d_latent)
        
        # Reconstruct Q, K, V
        q = self.W_UQ(c_q).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_UK(c_kv).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_UV(c_kv).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # Output
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.W_O(out)
```

---

## Compression Ratio Selection

### Guidelines for Choosing $d_c$

| Model Size | Recommended $d_c$ | Compression Ratio |
|------------|-------------------|-------------------|
| 7B | $D/4$ (1024 for D=4096) | 8x |
| 13B | $D/4$ to $D/6$ | 8-12x |
| 70B+ | $D/8$ | 16x |

### Trade-off Analysis

Higher compression ($d_c$ smaller):
- ✅ More memory savings
- ❌ More information loss
- ❌ Higher reconstruction compute

Lower compression ($d_c$ larger):
- ✅ Better quality
- ❌ Less memory savings
- ✅ Lower reconstruction compute

---

## Comparison with Other Methods

### Feature Comparison

| Feature | MHA | MQA | GQA | MLA |
|---------|-----|-----|-----|-----|
| KV Cache Size | $2LD$ | $2Ld_h$ | $2LGd_h$ | $Ld_c$ |
| Per-Head KV | Yes | No | Per-Group | Shared Latent |
| Reconstruction | No | No | No | Yes |
| Quality | Best | Worst | Good | Very Good |
| Complexity | Low | Low | Low | Medium |

### When MLA Wins

1. **Very long sequences**: At 100k tokens, MLA's 16x savings is transformative
2. **Serving at scale**: Memory = batch size = throughput
3. **Large models**: Compression overhead amortized over more compute

---

## Advanced: MLA with Decoupled RoPE

DeepSeek extends MLA with separate positional encoding paths (see [DeepSeek Attention](./04-deepseek-attention.md)):

```
Content Path:  X → Compress → Cache → Reconstruct → K_c, V_c
Position Path: X → RoPE projection → K_r (not cached or small cache)

Final: K = [K_c; K_r]  (concatenate)
```

This separates content (can be compressed) from position (needs full precision).

---

## Performance Characteristics

### Compute Breakdown

| Operation | FLOPs |
|-----------|-------|
| Compression ($W^{DKV}, W^{DQ}$) | $2BD \cdot d_c$ |
| Reconstruction ($W^{UK}, W^{UV}, W^{UQ}$) | $3Bd_c \cdot D$ |
| Attention | $O(BHL^2d_h)$ |
| **Additional vs MHA** | $2BD \cdot d_c + 3Bd_c \cdot D$ |

For $d_c = D/4$: Additional compute is ~$1.25BD^2$ (small vs total)

### Memory Bandwidth

| Phase | MHA | MLA |
|-------|-----|-----|
| KV Cache Load | $2LD$ per token | $d_c$ per token + reconstruction |
| For L=100k, D=4096 | 800MB | 50MB + compute |

**Conclusion**: MLA trades memory bandwidth for compute, beneficial when memory-bound.

---

## Practical Tips

1. **Start with $d_c = D/4$**: Good balance for most cases
2. **Profile your workload**: MLA benefits memory-bound scenarios
3. **Combine with FlashAttention**: Reconstruction fits in SRAM
4. **Quantize the latent**: INT8 latent cache for 2x more savings
5. **Batch reconstruction**: Amortize overhead across batch

---

## References

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Reducing Transformer Key-Value Cache Size with Cross-Layer Attention](https://arxiv.org/abs/2405.12981)
