# Standard Mixture of Experts (MoE)

## Overview

Mixture of Experts (MoE) is a neural network architecture that uses a **sparse gating mechanism** to selectively activate a subset of "expert" networks for each input. This enables scaling model capacity without proportionally increasing compute.

**Key Papers:**
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (Shazeer et al., 2017)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

---

## When to Use Standard MoE

### ✅ Use MoE When:
- **Scaling model capacity** - Want larger models without proportional compute increase
- **Diverse task requirements** - Different experts can specialize for different inputs
- **Training large models** - MoE enables training of much larger models
- **Have sufficient batch size** - Experts need enough tokens to be efficient

### ❌ Don't Use MoE When:
- **Small batch sizes** - Expert utilization becomes inefficient
- **Simple tasks** - Dense models may be sufficient and simpler
- **Memory constrained** - All expert weights must be in memory
- **Latency critical** - Routing adds overhead

---

## Mathematical Foundation

### MoE Layer Architecture

A standard MoE layer consists of:
1. **Router** $G$: Determines which experts process each token
2. **Experts** $\{E_1, E_2, ..., E_N\}$: Specialized sub-networks
3. **Combiner**: Aggregates expert outputs

### Router (Gating) Function

For input token $x$:

$$G(x) = \text{softmax}(xW_g)$$

Where $W_g \in \mathbb{R}^{D \times N}$ maps to $N$ expert probabilities.

### Top-K Selection

Select top-$k$ experts (typically $k=1$ or $k=2$):

$$\text{TopK}(G(x)) = \{(i_1, g_1), (i_2, g_2), ..., (i_k, g_k)\}$$

Where $i_j$ are expert indices and $g_j$ are their gating weights.

### Normalized Gating

Renormalize selected expert weights:

$$\hat{g}_j = \frac{g_j}{\sum_{l=1}^{k} g_l}$$

### Expert Computation

Each expert $E_i$ is typically a feed-forward network:

$$E_i(x) = W_2^i \cdot \text{ReLU}(W_1^i \cdot x)$$

Where $W_1^i \in \mathbb{R}^{D \times D_{ff}}$ and $W_2^i \in \mathbb{R}^{D_{ff} \times D}$.

### Final Output

Weighted combination of selected experts:

$$y = \sum_{j=1}^{k} \hat{g}_j \cdot E_{i_j}(x)$$

---

## Load Balancing

### The Problem

Without constraints, routing often degenerates:
- A few experts receive most tokens ("expert collapse")
- Other experts rarely used ("dead experts")

### Auxiliary Loss

Add a load-balancing loss to encourage uniform expert usage:

$$\mathcal{L}_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

Where:
- $f_i = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}[i \in \text{TopK}(G(x_t))]$ = fraction of tokens routed to expert $i$
- $P_i = \frac{1}{T}\sum_{t=1}^{T} G(x_t)_i$ = average probability for expert $i$
- $\alpha$ = balancing coefficient (typically 0.01)
- $T$ = total tokens in batch

### Intuition

$f_i \cdot P_i$ is high when:
- Expert $i$ receives many tokens AND
- Router assigns high probability to expert $i$

Minimizing this encourages spreading tokens across all experts.

---

## Implementation Details

### Rust Implementation

```rust
pub struct StandardMoE {
    n_experts: usize,
    top_k: usize,
    d_model: usize,
    d_hidden: usize,
    
    router: Linear,           // D → N
    experts: Vec<Expert>,     // N experts
}

pub struct Expert {
    w1: Linear,  // D → D_ff
    w2: Linear,  // D_ff → D
}

impl Expert {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.w1.forward(x)?;
        let h = h.relu()?;
        self.w2.forward(&h)
    }
}

impl StandardMoE {
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t, d) = x.dims3()?;
        let flat_x = x.reshape((b * t, d))?;  // (B*T, D)
        
        // === Routing ===
        let router_logits = self.router.forward(&flat_x)?;  // (B*T, N)
        let router_probs = softmax(&router_logits, D::Minus1)?;
        
        // Top-K selection
        let (top_k_probs, top_k_indices) = router_probs.topk(self.top_k)?;
        
        // Normalize top-k probabilities
        let top_k_probs = (&top_k_probs / top_k_probs.sum(D::Minus1)?)?;
        
        // === Expert Computation ===
        let mut output = Tensor::zeros((b * t, d), DType::F32, x.device())?;
        
        // For each selected expert position
        for k in 0..self.top_k {
            let indices = top_k_indices.i((.., k))?;  // (B*T,)
            let probs = top_k_probs.i((.., k))?.unsqueeze(1)?;  // (B*T, 1)
            
            // Process tokens per expert
            for expert_idx in 0..self.n_experts {
                // Find tokens routed to this expert
                let mask = indices.eq(expert_idx as u32)?;
                
                if mask.any()?.to_scalar::<u8>()? > 0 {
                    // Gather tokens for this expert
                    let expert_input = flat_x.index_select(&mask.nonzero()?, 0)?;
                    
                    // Expert computation
                    let expert_output = self.experts[expert_idx].forward(&expert_input)?;
                    
                    // Weight by routing probability
                    let expert_probs = probs.index_select(&mask.nonzero()?, 0)?;
                    let weighted_output = (expert_output * expert_probs)?;
                    
                    // Scatter back
                    output = output.index_add(&mask.nonzero()?, &weighted_output, 0)?;
                }
            }
        }
        
        // === Load Balancing Loss ===
        let aux_loss = self.compute_aux_loss(&router_probs, &top_k_indices)?;
        
        Ok((output.reshape((b, t, d))?, aux_loss))
    }
    
    fn compute_aux_loss(
        &self, 
        router_probs: &Tensor,  // (B*T, N)
        top_k_indices: &Tensor, // (B*T, k)
    ) -> Result<Tensor> {
        let n_tokens = router_probs.dim(0)?;
        
        // f_i: fraction of tokens routed to each expert
        let mut f = vec![0f32; self.n_experts];
        for i in 0..self.n_experts {
            let count = top_k_indices.eq(i as u32)?.sum_all()?.to_scalar::<u32>()?;
            f[i] = count as f32 / (n_tokens * self.top_k) as f32;
        }
        let f = Tensor::new(&f, router_probs.device())?;
        
        // P_i: mean probability per expert
        let p = router_probs.mean(0)?;  // (N,)
        
        // aux_loss = N * sum(f_i * P_i)
        let aux_loss = (f * p)?.sum_all()? * (self.n_experts as f64);
        
        Ok(aux_loss)
    }
}
```

### Python Implementation

```python
class Expert(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


class StandardMoE(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_hidden: int, 
        n_experts: int, 
        top_k: int = 2,
        aux_loss_coef: float = 0.01
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(d_model, d_hidden) for _ in range(n_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        flat_x = x.view(-1, D)  # (B*L, D)
        
        # === Routing ===
        router_logits = self.router(flat_x)  # (B*L, N)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # === Expert Computation ===
        output = torch.zeros_like(flat_x)
        
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # (B*L,)
            expert_probs = top_k_probs[:, k:k+1]  # (B*L, 1)
            
            for i, expert in enumerate(self.experts):
                # Mask for tokens routed to this expert
                mask = (expert_indices == i)
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_output = expert(expert_input)
                    output[mask] += expert_output * expert_probs[mask]
        
        # === Auxiliary Loss ===
        aux_loss = self._compute_aux_loss(router_probs, top_k_indices)
        
        return output.view(B, L, D), aux_loss
    
    def _compute_aux_loss(
        self, 
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        n_tokens = router_probs.size(0)
        
        # f_i: fraction routed to each expert
        # One-hot encode indices and sum
        one_hot = F.one_hot(top_k_indices, self.n_experts).float()  # (B*L, k, N)
        f = one_hot.sum(dim=[0, 1]) / (n_tokens * self.top_k)  # (N,)
        
        # P_i: mean probability
        p = router_probs.mean(dim=0)  # (N,)
        
        # aux_loss = α * N * Σ(f_i * P_i)
        aux_loss = self.aux_loss_coef * self.n_experts * (f * p).sum()
        
        return aux_loss
```

---

## Capacity Factor and Token Dropping

### The Problem

With fixed expert capacity, some experts may receive more tokens than they can handle.

### Capacity Factor

Set maximum tokens per expert:

$$\text{capacity} = \frac{\text{batch\_size} \times \text{seq\_len} \times k}{N} \times C$$

Where $C$ is the capacity factor (typically 1.0-2.0).

### Token Dropping

Tokens exceeding capacity are:
1. **Dropped**: Passed through a residual connection
2. **Overflow routing**: Sent to next-best expert

```python
def route_with_capacity(self, x, router_probs, capacity_factor=1.25):
    B_L, N = router_probs.shape
    capacity = int(B_L * self.top_k / N * capacity_factor)
    
    top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
    
    # Track expert usage
    expert_counts = torch.zeros(N, device=x.device)
    
    # Mask for dropped tokens
    dropped_mask = torch.zeros(B_L, dtype=torch.bool, device=x.device)
    
    for token_idx in range(B_L):
        for k in range(self.top_k):
            expert_idx = top_k_indices[token_idx, k].item()
            if expert_counts[expert_idx] < capacity:
                expert_counts[expert_idx] += 1
            else:
                # Drop this token-expert assignment
                top_k_probs[token_idx, k] = 0
                dropped_mask[token_idx] = True
    
    return top_k_probs, top_k_indices, dropped_mask
```

---

## Expert Parallelism

### Data Layout

For efficient GPU execution, rearrange tokens by expert:

```
Before (token order):
Token 0 → Expert 2
Token 1 → Expert 0  
Token 2 → Expert 2
Token 3 → Expert 1

After (expert order):
Expert 0: [Token 1]
Expert 1: [Token 3]
Expert 2: [Token 0, Token 2]
```

### All-to-All Communication

In distributed training, tokens must be routed across GPUs:

```
GPU 0: Experts 0-3    GPU 1: Experts 4-7
      ↓ all2all ↓           ↓ all2all ↓
Tokens for E0-3 → GPU 0   Tokens for E4-7 → GPU 1
      ↓ compute ↓           ↓ compute ↓
      ↓ all2all ↓           ↓ all2all ↓
Outputs back to original GPUs
```

---

## Performance Characteristics

### Compute Analysis

| Component | Dense FFN | MoE (top-2 of 16) |
|-----------|-----------|-------------------|
| Parameters | $2D \cdot D_{ff}$ | $16 \times 2D \cdot D_{ff}$ |
| FLOPs per token | $4D \cdot D_{ff}$ | $2 \times 4D \cdot D_{ff}$ |
| **Parameter/FLOP ratio** | 1x | **8x** |

MoE achieves 8x more parameters for only 2x compute!

### Memory vs Compute Trade-off

```
Dense:     Low params, high utilization
           ████████████████ (all params used)

MoE:       High params, sparse utilization
           ██░░░░░░░░░░░░░░ (only top-k used)
           ████████████████ (but all in memory)
```

---

## Common Configurations

| Model | Experts | Top-K | Capacity | Notes |
|-------|---------|-------|----------|-------|
| Switch | 128-2048 | 1 | 1.0 | Maximum sparsity |
| Mixtral | 8 | 2 | - | Balanced |
| DeepSeek | 64 | 6 | 1.5 | High top-k |
| Grok | 8 | 2 | - | Similar to Mixtral |

---

## Practical Tips

1. **Start with top-k=2**: Good balance of quality and efficiency
2. **Use aux loss carefully**: Too high destroys specialization, too low causes collapse
3. **Monitor expert utilization**: Track per-expert token counts during training
4. **Batch size matters**: Larger batches = better expert utilization
5. **Expert initialization**: Initialize all experts identically for stable training
6. **Gradient noise**: MoE gradients are noisier; may need lower learning rate

---

## References

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
