# Multi-Query Attention (MQA)

## Overview

Multi-Query Attention (MQA) is an attention variant that uses a **single Key-Value head** shared across all Query heads. This dramatically reduces the KV-cache size during inference, making it ideal for memory-constrained deployments.

**Paper:** [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (Shazeer, 2019)

---

## When to Use MQA

### ✅ Use MQA When:
- **Memory is the bottleneck** - Inference on edge devices, mobile, or limited GPU memory
- **Long sequences** - KV cache grows linearly with sequence length
- **High-throughput inference** - Serving many requests with limited memory
- **Batch inference** - Larger batches fit in memory with smaller KV cache

### ❌ Don't Use MQA When:
- **Training quality is critical** - MQA can slightly degrade model quality
- **Small models** - Overhead of reduced expressivity not worth the memory savings
- **Short sequences** - Memory savings are negligible

---

## Mathematical Foundation

### Standard Multi-Head Attention (MHA)

In standard MHA with $H$ heads, each head has its own Query, Key, and Value projections:

$$Q_h = XW_Q^h, \quad K_h = XW_K^h, \quad V_h = XW_V^h$$

Where:
- $X \in \mathbb{R}^{B \times L \times D}$ is the input (batch, sequence, dimension)
- $W_Q^h, W_K^h, W_V^h \in \mathbb{R}^{D \times d_h}$ are per-head projection matrices
- $d_h = D / H$ is the head dimension

The attention output for each head:

$$\text{Attention}_h(Q_h, K_h, V_h) = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_h}}\right) V_h$$

### Multi-Query Attention

MQA modifies this by using a **single shared** Key and Value projection:

$$Q_h = XW_Q^h \quad \text{(per-head)}$$
$$K = XW_K \quad \text{(shared)}$$
$$V = XW_V \quad \text{(shared)}$$

Where:
- $W_Q^h \in \mathbb{R}^{D \times d_h}$ - Still per-head for queries
- $W_K, W_V \in \mathbb{R}^{D \times d_h}$ - **Single shared** projection

The attention computation broadcasts K and V across all heads:

$$\text{Attention}_h(Q_h, K, V) = \text{softmax}\left(\frac{Q_h K^T}{\sqrt{d_h}}\right) V$$

### Memory Analysis

**KV Cache Size Comparison:**

| Component | MHA | MQA |
|-----------|-----|-----|
| Keys | $L \times H \times d_h = L \times D$ | $L \times d_h = L \times D/H$ |
| Values | $L \times H \times d_h = L \times D$ | $L \times d_h = L \times D/H$ |
| **Total** | $2LD$ | $2L \times D/H$ |
| **Reduction** | 1x | $H$x |

For a model with $H=32$ heads, MQA reduces KV cache by **32x**.

---

## Implementation Details

### Rust Implementation

```rust
pub struct MultiQueryAttention {
    n_heads: usize,
    d_head: usize,
    w_q: Linear,   // Projects to all heads: D → H*d_h
    w_k: Linear,   // Projects to single head: D → d_h
    w_v: Linear,   // Projects to single head: D → d_h
    w_o: Linear,   // Output projection: H*d_h → D
}

impl MultiQueryAttention {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        
        // Query: (B, T, H*d_h) → (B, H, T, d_h)
        let q = self.w_q.forward(x)?
            .reshape((b, t, self.n_heads, self.d_head))?
            .transpose(1, 2)?;
        
        // Key/Value: (B, T, d_h) → (B, 1, T, d_h) for broadcasting
        let k = self.w_k.forward(x)?
            .reshape((b, t, 1, self.d_head))?
            .transpose(1, 2)?;
        let v = self.w_v.forward(x)?
            .reshape((b, t, 1, self.d_head))?
            .transpose(1, 2)?;
        
        // Attention: K, V broadcast across H heads
        let scale = 1.0 / (self.d_head as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?)? * scale;
        let attn = softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?;  // V broadcasts
        
        // Reshape back: (B, H, T, d_h) → (B, T, D)
        out.transpose(1, 2)?.reshape((b, t, self.n_heads * self.d_head))
    }
}
```

### Python Implementation

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Query: full projection for all heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # Key/Value: single head projection
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # Q: (B, L, H, d_h) → (B, H, L, d_h)
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K, V: (B, L, d_h) → (B, 1, L, d_h) for broadcasting
        k = self.k_proj(x).view(B, L, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, 1, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with broadcasting
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, L, d_h)
        
        # Reshape: (B, H, L, d_h) → (B, L, D)
        return self.o_proj(out.transpose(1, 2).reshape(B, L, D))
```

---

## Performance Characteristics

### Computational Complexity

| Operation | MHA | MQA |
|-----------|-----|-----|
| Q Projection | $O(BD \cdot D)$ | $O(BD \cdot D)$ |
| K Projection | $O(BD \cdot D)$ | $O(BD \cdot d_h)$ |
| V Projection | $O(BD \cdot D)$ | $O(BD \cdot d_h)$ |
| Attention | $O(BHL^2d_h)$ | $O(BHL^2d_h)$ |
| **Total Projection** | $O(3BD^2)$ | $O(BD^2 + 2BDd_h)$ |

### Memory Bandwidth

During autoregressive decoding, MQA significantly reduces memory bandwidth:

- **MHA**: Load $2 \times H \times d_h$ per token per layer
- **MQA**: Load $2 \times d_h$ per token per layer
- **Speedup**: Up to $H$x for memory-bound inference

---

## Quality Considerations

### Trade-offs

1. **Reduced Expressivity**: Sharing KV across heads limits the diversity of attention patterns
2. **Training Stability**: Generally trains as stably as MHA
3. **Quality Gap**: Typically 0.5-2% degradation on benchmarks, depending on model size

### Mitigation Strategies

1. **Larger Models**: Quality gap shrinks with model scale
2. **GQA Hybrid**: Use groups of heads sharing KV (see GQA doc)
3. **Distillation**: Train MQA model from MHA teacher

---

## Practical Tips

1. **Inference Serving**: MQA enables 2-4x larger batch sizes in production
2. **Long Context**: Essential for 100k+ token context windows
3. **Quantization**: Combine with INT8/FP8 for even more memory savings
4. **KV Cache Reuse**: Shared KV enables efficient prefix caching

---

## References

- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) (Uses MQA)
- [Falcon LLM](https://huggingface.co/tiiuae/falcon-40b) (Uses MQA)
