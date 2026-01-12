# Grouped-Query Attention (GQA)

## Overview

Grouped-Query Attention (GQA) is a middle ground between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). It groups query heads and assigns a **shared Key-Value head per group**, balancing memory efficiency with model expressivity.

**Paper:** [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023)

---

## When to Use GQA

### ✅ Use GQA When:
- **Need balance** - Want memory savings without MQA's quality loss
- **Large models** - 7B+ parameter models benefit most
- **Moderate sequence lengths** - 4k-32k context windows
- **Converting from MHA** - Can upconvert MHA checkpoints to GQA

### ❌ Don't Use GQA When:
- **Extreme memory constraints** - MQA saves more memory
- **Small models** - Overhead not worth it for < 1B params
- **Very short sequences** - KV cache is already small

---

## Mathematical Foundation

### GQA Configuration

Let:
- $H$ = Total number of query heads
- $G$ = Number of KV groups (where $H \mod G = 0$)
- $h_g = H / G$ = Query heads per group
- $d_h = D / H$ = Dimension per head

### Projection Matrices

**Query Projection** (per-head, same as MHA):
$$Q_h = XW_Q^h, \quad W_Q^h \in \mathbb{R}^{D \times d_h}, \quad h \in [1, H]$$

**Key/Value Projection** (per-group):
$$K_g = XW_K^g, \quad V_g = XW_V^g, \quad W_K^g, W_V^g \in \mathbb{R}^{D \times d_h}, \quad g \in [1, G]$$

### Attention Computation

For query head $h$ in group $g = \lceil h / h_g \rceil$:

$$\text{Attention}_h = \text{softmax}\left(\frac{Q_h K_g^T}{\sqrt{d_h}}\right) V_g$$

The key insight: Multiple query heads share the same $K_g$ and $V_g$.

### Memory Analysis

| Configuration | KV Cache Size | Example (32 heads) |
|---------------|---------------|---------------------|
| MHA ($G = H$) | $2 \times L \times D$ | $2LD$ |
| GQA-8 ($G = 8$) | $2 \times L \times G \times d_h$ | $0.25 \times 2LD$ |
| GQA-4 ($G = 4$) | $2 \times L \times G \times d_h$ | $0.125 \times 2LD$ |
| MQA ($G = 1$) | $2 \times L \times d_h$ | $0.03125 \times 2LD$ |

---

## Visual Representation

```
MHA (H=8 heads, G=8 groups):
Q1→K1,V1  Q2→K2,V2  Q3→K3,V3  Q4→K4,V4  Q5→K5,V5  Q6→K6,V6  Q7→K7,V7  Q8→K8,V8

GQA (H=8 heads, G=4 groups):
Q1→K1,V1  Q2→K1,V1  Q3→K2,V2  Q4→K2,V2  Q5→K3,V3  Q6→K3,V3  Q7→K4,V4  Q8→K4,V4
└──Group 1──┘        └──Group 2──┘        └──Group 3──┘        └──Group 4──┘

GQA (H=8 heads, G=2 groups):
Q1→K1,V1  Q2→K1,V1  Q3→K1,V1  Q4→K1,V1  Q5→K2,V2  Q6→K2,V2  Q7→K2,V2  Q8→K2,V2
└─────────Group 1─────────────┘        └─────────Group 2─────────────┘

MQA (H=8 heads, G=1 group):
Q1→K1,V1  Q2→K1,V1  Q3→K1,V1  Q4→K1,V1  Q5→K1,V1  Q6→K1,V1  Q7→K1,V1  Q8→K1,V1
└──────────────────────────Group 1──────────────────────────────────┘
```

---

## Implementation Details

### Rust Implementation

```rust
pub struct GroupedQueryAttention {
    n_heads: usize,      // Total query heads (H)
    n_kv_heads: usize,   // Number of KV groups (G)
    d_head: usize,       // Dimension per head
    heads_per_group: usize,  // H / G
    
    w_q: Linear,  // D → H * d_h
    w_k: Linear,  // D → G * d_h
    w_v: Linear,  // D → G * d_h
    w_o: Linear,  // H * d_h → D
}

impl GroupedQueryAttention {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        
        // Query: (B, T, H*d_h) → (B, H, T, d_h)
        let q = self.w_q.forward(x)?
            .reshape((b, t, self.n_heads, self.d_head))?
            .transpose(1, 2)?;
        
        // Key/Value: (B, T, G*d_h) → (B, G, T, d_h)
        let k = self.w_k.forward(x)?
            .reshape((b, t, self.n_kv_heads, self.d_head))?
            .transpose(1, 2)?;
        let v = self.w_v.forward(x)?
            .reshape((b, t, self.n_kv_heads, self.d_head))?
            .transpose(1, 2)?;
        
        // Repeat K, V for each head in the group
        // (B, G, T, d_h) → (B, G, 1, T, d_h) → (B, G, h_g, T, d_h) → (B, H, T, d_h)
        let k = k.unsqueeze(2)?
            .expand((b, self.n_kv_heads, self.heads_per_group, t, self.d_head))?
            .reshape((b, self.n_heads, t, self.d_head))?;
        let v = v.unsqueeze(2)?
            .expand((b, self.n_kv_heads, self.heads_per_group, t, self.d_head))?
            .reshape((b, self.n_heads, t, self.d_head))?;
        
        // Standard attention
        let scale = 1.0 / (self.d_head as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?)? * scale;
        let attn = softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?;
        
        out.transpose(1, 2)?.reshape((b, t, self.n_heads * self.d_head))
    }
}
```

### Python Implementation

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_groups: int):
        super().__init__()
        assert num_heads % num_groups == 0, "Heads must be divisible by groups"
        
        self.num_heads = num_heads      # H
        self.num_groups = num_groups    # G
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // num_groups
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)  # D → H*d_h
        self.k_proj = nn.Linear(d_model, num_groups * self.head_dim, bias=False)  # D → G*d_h
        self.v_proj = nn.Linear(d_model, num_groups * self.head_dim, bias=False)  # D → G*d_h
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # Q: (B, L, H, d_h) → (B, H, L, d_h)
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K, V: (B, L, G, d_h) → (B, G, L, d_h)
        k = self.k_proj(x).view(B, L, self.num_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_groups, self.head_dim).transpose(1, 2)
        
        # Repeat K, V for each head in group
        # (B, G, L, d_h) → (B, G, 1, L, d_h) → (B, G, h_g, L, d_h) → (B, H, L, d_h)
        k = k.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        k = k.reshape(B, self.num_heads, L, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        v = v.reshape(B, self.num_heads, L, self.head_dim)
        
        # Standard attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        return self.o_proj(out.transpose(1, 2).reshape(B, L, D))
```

---

## Uptraining from MHA to GQA

One key advantage of GQA: You can convert an existing MHA model to GQA with minimal quality loss.

### Mean Pooling Conversion

For each group $g$, average the KV weights from the corresponding heads:

$$W_K^g = \frac{1}{h_g} \sum_{h \in \text{group}_g} W_K^h$$
$$W_V^g = \frac{1}{h_g} \sum_{h \in \text{group}_g} W_V^h$$

### Uptraining Recipe

1. **Initialize** GQA with mean-pooled weights from MHA checkpoint
2. **Short finetune** (5-10% of original training) to recover quality
3. **Result**: Near-MHA quality with GQA efficiency

---

## Performance Comparison

### Inference Speed (relative to MHA)

| Config | Memory | Throughput | Quality |
|--------|--------|------------|---------|
| MHA | 1.0x | 1.0x | Baseline |
| GQA-8 | 0.25x | 1.5-2x | -0.2% |
| GQA-4 | 0.125x | 2-3x | -0.5% |
| GQA-2 | 0.0625x | 3-4x | -1.0% |
| MQA | 0.03x | 4-5x | -1.5% |

### Common Configurations

| Model | Heads | Groups | Ratio |
|-------|-------|--------|-------|
| Llama 2 7B | 32 | 32 | MHA |
| Llama 2 70B | 64 | 8 | GQA-8 |
| Mistral 7B | 32 | 8 | GQA-4 |
| Falcon 40B | 64 | 1 | MQA |

---

## When to Choose Which?

### Decision Framework

```
                    ┌─────────────────┐
                    │ Memory Critical? │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │ Yes          │ No           │
              ▼              │              ▼
    ┌─────────────────┐      │    ┌─────────────────┐
    │ Extreme savings │      │    │ Use MHA         │
    │ needed?         │      │    │ (full quality)  │
    └────────┬────────┘      │    └─────────────────┘
             │               │
    ┌────────┼────────┐      │
    │ Yes    │ No     │      │
    ▼        ▼        │      │
 ┌─────┐  ┌─────┐     │      │
 │ MQA │  │ GQA │     │      │
 └─────┘  └──┬──┘     │      │
             │        │      │
       Choose G based │      │
       on quality/    │      │
       memory tradeoff│      │
```

---

## Practical Tips

1. **Start with GQA-8**: Good default for most large models
2. **Benchmark your use case**: Memory savings vs quality loss varies
3. **Consider sequence length**: Longer sequences benefit more from GQA
4. **KV cache quantization**: Combine GQA with INT8 KV cache for 2x more savings
5. **Batch size tuning**: GQA enables larger batches = higher throughput

---

## References

- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Mistral 7B](https://arxiv.org/abs/2310.06825)
