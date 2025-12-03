# DeepSeek Mixture of Experts (MoE)

## Overview

DeepSeek's MoE architecture introduces several innovations over standard MoE:
1. **Shared Experts**: Always-activated experts providing common knowledge
2. **Routed Experts**: Sparsely-activated specialized experts
3. **Fine-grained Expert Segmentation**: More, smaller experts for better routing
4. **Device-limited Routing**: Constraint for efficient distributed execution

**Key Paper:** [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)

---

## When to Use DeepSeek MoE

### ✅ Use DeepSeek MoE When:
- **Training very large models** - Superior scaling properties
- **Distributed training** - Device-limited routing reduces communication
- **Need both generalist and specialist capabilities** - Shared + routed experts
- **Want fine-grained specialization** - Smaller experts = better routing decisions

### ❌ Don't Use When:
- **Single GPU inference** - Simpler MoE architectures sufficient
- **Memory constrained** - More experts = more parameters
- **Small models** - Overhead not justified

### When to Prefer Over Standard MoE:
| Standard MoE | DeepSeek MoE |
|--------------|--------------|
| Simpler to implement | Better specialization |
| Fewer experts | Many fine-grained experts |
| All experts equal | Shared + routed split |
| Any-to-any routing | Device-aware routing |

---

## Mathematical Foundation

### Architecture Overview

DeepSeek MoE layer consists of:
1. **Shared Experts** $\{S_1, ..., S_{N_s}\}$: Always activated
2. **Routed Experts** $\{R_1, ..., R_{N_r}\}$: Top-K activated
3. **Router** $G$: Routes tokens to routed experts

### Forward Computation

For input token $x$:

$$y = \sum_{i=1}^{N_s} S_i(x) + \sum_{j \in \text{TopK}} g_j \cdot R_j(x)$$

Where:
- First term: Sum of all shared expert outputs
- Second term: Weighted sum of selected routed expert outputs

### Router Function

The router computes gating scores:

$$G(x) = \text{softmax}\left(\text{TopK}(x \cdot W_g)\right)$$

Where $W_g \in \mathbb{R}^{D \times N_r}$ is the routing weight matrix.

### Normalized Gating Weights

After top-K selection, normalize the weights:

$$g_j = \frac{\exp(s_j)}{\sum_{l \in \text{TopK}} \exp(s_l)}$$

Where $s_j$ is the raw routing score for expert $j$.

---

## Fine-Grained Expert Segmentation

### The Innovation

Instead of few large experts, use many smaller experts:

```
Standard MoE:         DeepSeek MoE:
8 experts            64 experts (8x segmentation)
D_ff = 4096          D_ff = 512 (each)
Same total params    Same total params

Top-2 → 2 experts    Top-6 → 6 experts
Coarse routing       Fine-grained routing
```

### Mathematical Formulation

For segmentation factor $M$:

| Parameter | Standard | DeepSeek |
|-----------|----------|----------|
| Num Experts | $N$ | $N \times M$ |
| Expert Hidden Dim | $D_{ff}$ | $D_{ff} / M$ |
| Top-K | $k$ | $k \times M$ |
| Activated Params | Same | Same |

### Why It Works

Fine-grained segmentation enables:
1. **More precise routing**: 64 choices vs 8
2. **Better specialization**: Each expert handles narrower domain
3. **Combinatorial flexibility**: $\binom{64}{6} \gg \binom{8}{2}$

---

## Shared vs Routed Experts

### Design Rationale

**Shared experts** capture knowledge needed by ALL tokens:
- Common language patterns
- Frequent transformations
- General reasoning

**Routed experts** capture specialized knowledge:
- Domain-specific processing
- Rare patterns
- Task-specific features

### Architecture Diagram

```
Input Token x
      │
      ├──────────────────────────────────────┐
      │                                      │
      ▼                                      ▼
┌─────────────────┐              ┌─────────────────────┐
│ Shared Experts  │              │      Router         │
│  S₁, S₂, ..., Sₛ│              │   softmax(xWg)      │
│  (always active)│              │   → top-K indices   │
└────────┬────────┘              └──────────┬──────────┘
         │                                  │
         │ sum all                          │ top-K select
         ▼                                  ▼
    ┌────────┐              ┌─────────────────────────────┐
    │Σ Sᵢ(x) │              │  Routed Experts Rⱼ          │
    └────┬───┘              │  (only selected activated)  │
         │                  │  Σ gⱼ · Rⱼ(x)               │
         │                  └──────────────┬──────────────┘
         │                                 │
         └──────────────┬──────────────────┘
                        │
                        ▼
                  y = shared + routed
```

### Parameter Allocation

DeepSeek typically uses:
- **2 shared experts**: ~25% of FFN parameters
- **64 routed experts**: ~75% of FFN parameters (only 6 active)
- **Effective activation**: ~50% of parameters

---

## Device-Limited Routing

### The Problem

In distributed training, all-to-all communication for MoE is expensive:
- N experts across M devices
- Each token may need to visit any device
- Full all-to-all shuffle required

### The Solution

Limit each token to experts on a subset of devices:

$$\text{TopK}(G(x)) \subseteq \mathcal{E}_{\text{device}[x]}$$

Where $\mathcal{E}_{\text{device}[x]}$ is the set of experts on nearby devices.

### Implementation

```python
def device_limited_routing(
    router_logits: torch.Tensor,  # (B*L, N_routed)
    top_k: int,
    device_limit: int,  # Max devices to communicate with
    n_experts_per_device: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    n_devices = router_logits.size(1) // n_experts_per_device
    
    # Reshape to (B*L, n_devices, experts_per_device)
    logits_by_device = router_logits.view(-1, n_devices, n_experts_per_device)
    
    # Score each device by max expert score
    device_scores = logits_by_device.max(dim=-1).values  # (B*L, n_devices)
    
    # Select top devices
    _, top_devices = torch.topk(device_scores, device_limit, dim=-1)
    
    # Mask experts on non-selected devices
    device_mask = torch.zeros_like(router_logits)
    for d in range(device_limit):
        device_idx = top_devices[:, d]
        start_idx = device_idx * n_experts_per_device
        for e in range(n_experts_per_device):
            device_mask.scatter_(1, (start_idx + e).unsqueeze(1), 1.0)
    
    # Apply mask and route
    masked_logits = router_logits * device_mask + (1 - device_mask) * float('-inf')
    top_k_probs, top_k_indices = torch.topk(
        F.softmax(masked_logits, dim=-1), top_k, dim=-1
    )
    
    return top_k_probs, top_k_indices
```

---

## Complete Implementation

### Rust Implementation

```rust
pub struct DeepSeekMoE {
    // Architecture params
    n_shared: usize,
    n_routed: usize,
    top_k: usize,
    d_model: usize,
    d_hidden: usize,  // Per-expert hidden dim
    
    // Experts
    shared_experts: Vec<Expert>,
    routed_experts: Vec<Expert>,
    
    // Router
    router: Linear,
}

impl DeepSeekMoE {
    pub fn new(
        d_model: usize,
        d_hidden_total: usize,  // Total FFN hidden dim
        n_shared: usize,
        n_routed: usize,
        top_k: usize,
        device: &Device,
    ) -> Result<Self> {
        // Fine-grained: each expert has smaller hidden dim
        let d_hidden = d_hidden_total / (n_shared + top_k);
        
        let shared_experts = (0..n_shared)
            .map(|_| Expert::new(d_model, d_hidden, device))
            .collect::<Result<Vec<_>>>()?;
        
        let routed_experts = (0..n_routed)
            .map(|_| Expert::new(d_model, d_hidden, device))
            .collect::<Result<Vec<_>>>()?;
        
        let router = Linear::new(d_model, n_routed, device)?;
        
        Ok(Self {
            n_shared,
            n_routed,
            top_k,
            d_model,
            d_hidden,
            shared_experts,
            routed_experts,
            router,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t, d) = x.dims3()?;
        let flat_x = x.reshape((b * t, d))?;
        
        // === Shared Expert Computation (always active) ===
        let mut shared_output = Tensor::zeros((b * t, d), DType::F32, x.device())?;
        for expert in &self.shared_experts {
            shared_output = (shared_output + expert.forward(&flat_x)?)?;
        }
        
        // === Routing for Routed Experts ===
        let router_logits = self.router.forward(&flat_x)?;  // (B*T, N_routed)
        let router_probs = softmax(&router_logits, D::Minus1)?;
        
        // Top-K selection
        let (top_k_probs, top_k_indices) = router_probs.topk(self.top_k)?;
        let top_k_probs = (&top_k_probs / top_k_probs.sum_keepdim(D::Minus1)?)?;
        
        // === Routed Expert Computation ===
        let mut routed_output = Tensor::zeros((b * t, d), DType::F32, x.device())?;
        
        for k in 0..self.top_k {
            let indices = top_k_indices.i((.., k))?;
            let probs = top_k_probs.i((.., k))?.unsqueeze(1)?;
            
            for expert_idx in 0..self.n_routed {
                let mask = indices.eq(expert_idx as u32)?;
                
                if mask.any()?.to_scalar::<u8>()? > 0 {
                    let expert_input = flat_x.index_select(&mask.nonzero()?, 0)?;
                    let expert_output = self.routed_experts[expert_idx].forward(&expert_input)?;
                    let expert_probs = probs.index_select(&mask.nonzero()?, 0)?;
                    let weighted = (expert_output * expert_probs)?;
                    
                    routed_output = routed_output.index_add(&mask.nonzero()?, &weighted, 0)?;
                }
            }
        }
        
        // === Combine ===
        let output = (shared_output + routed_output)?.reshape((b, t, d))?;
        
        // === Aux Loss ===
        let aux_loss = self.compute_aux_loss(&router_probs, &top_k_indices)?;
        
        Ok((output, aux_loss))
    }
    
    fn compute_aux_loss(
        &self,
        router_probs: &Tensor,
        top_k_indices: &Tensor,
    ) -> Result<Tensor> {
        let n_tokens = router_probs.dim(0)?;
        
        // Load balancing over routed experts only
        let mut f = vec![0f32; self.n_routed];
        for i in 0..self.n_routed {
            let count = top_k_indices.eq(i as u32)?.sum_all()?.to_scalar::<u32>()?;
            f[i] = count as f32 / (n_tokens * self.top_k) as f32;
        }
        let f = Tensor::new(&f, router_probs.device())?;
        
        let p = router_probs.mean(0)?;
        let aux_loss = (f * p)?.sum_all()? * (self.n_routed as f64) * 0.01;
        
        Ok(aux_loss)
    }
}
```

### Python Implementation

```python
class DeepSeekMoE(nn.Module):
    """
    DeepSeek Mixture of Experts with shared and routed experts.
    
    Key innovations:
    1. Shared experts (always active) for common knowledge
    2. Routed experts (sparse) for specialized knowledge
    3. Fine-grained segmentation (many small experts)
    """
    
    def __init__(
        self,
        d_model: int,
        d_hidden_total: int,  # Total FFN hidden dimension
        n_shared: int = 2,    # Number of shared experts
        n_routed: int = 64,   # Number of routed experts
        top_k: int = 6,       # Experts to activate per token
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.n_shared = n_shared
        self.n_routed = n_routed
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        
        # Fine-grained: each expert smaller
        # Total activated = n_shared + top_k
        self.d_hidden = d_hidden_total // (n_shared + top_k)
        
        # Shared experts (always activated)
        self.shared_experts = nn.ModuleList([
            Expert(d_model, self.d_hidden) for _ in range(n_shared)
        ])
        
        # Routed experts (sparsely activated)
        self.routed_experts = nn.ModuleList([
            Expert(d_model, self.d_hidden) for _ in range(n_routed)
        ])
        
        # Router (only for routed experts)
        self.router = nn.Linear(d_model, n_routed, bias=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        flat_x = x.view(-1, D)  # (B*L, D)
        
        # === Shared Experts (always active) ===
        shared_output = sum(expert(flat_x) for expert in self.shared_experts)
        
        # === Routing ===
        router_logits = self.router(flat_x)  # (B*L, n_routed)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # === Routed Experts ===
        routed_output = torch.zeros_like(flat_x)
        
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            expert_probs = top_k_probs[:, k:k+1]
            
            for i, expert in enumerate(self.routed_experts):
                mask = (expert_indices == i)
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_output = expert(expert_input)
                    routed_output[mask] += expert_output * expert_probs[mask]
        
        # === Combine ===
        output = shared_output + routed_output
        
        # === Auxiliary Loss ===
        aux_loss = self._compute_aux_loss(router_probs, top_k_indices)
        
        return output.view(B, L, D), aux_loss
    
    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        n_tokens = router_probs.size(0)
        
        # Load balancing loss (routed experts only)
        one_hot = F.one_hot(top_k_indices, self.n_routed).float()
        f = one_hot.sum(dim=[0, 1]) / (n_tokens * self.top_k)
        p = router_probs.mean(dim=0)
        
        aux_loss = self.aux_loss_coef * self.n_routed * (f * p).sum()
        return aux_loss
    
    def get_expert_utilization(self) -> Dict[str, torch.Tensor]:
        """For monitoring expert usage during training."""
        return {
            'shared': torch.ones(self.n_shared),  # Always 1.0
            'routed': self._routed_usage,  # Track during forward
        }
```

---

## Expert Selection Strategies

### Comparison of Routing Strategies

| Strategy | Formula | Pros | Cons |
|----------|---------|------|------|
| **Softmax + TopK** | $\text{TopK}(\text{softmax}(xW))$ | Simple, stable | Soft assignment |
| **Noisy TopK** | $\text{TopK}(\text{softmax}(xW + \epsilon))$ | Exploration | Noise during eval |
| **Expert Choice** | Experts pick tokens | Perfect balance | Asymmetric |
| **DeepSeek** | TopK + aux loss | Balanced, stable | Hyperparameter tuning |

### DeepSeek's Approach

DeepSeek combines:
1. **Standard softmax routing**: Clean, differentiable
2. **Auxiliary loss**: Encourages balance
3. **No noise at inference**: Deterministic evaluation

---

## Load Balancing in DeepSeek

### Balance Loss Formula

$$\mathcal{L}_{bal} = \alpha \cdot N_r \cdot \sum_{i=1}^{N_r} f_i \cdot P_i$$

Where (for routed experts only):
- $f_i$: Fraction of tokens routed to expert $i$
- $P_i$: Mean probability assigned to expert $i$
- $\alpha$: Balance coefficient (typically 0.01-0.1)
- $N_r$: Number of routed experts

### Why Routed Only?

Shared experts don't need balancing because:
- They're always activated (100% utilization)
- No routing decision needed
- Load is inherently balanced

---

## Performance Analysis

### Memory Usage

```
Standard MoE (8 experts, top-2):
- Stored: 8 × D × D_ff parameters
- Active: 2 × D × D_ff per token

DeepSeek MoE (2 shared + 64 routed, top-6):
- Stored: (2 + 64) × D × (D_ff/8) = 8.25 × D × D_ff
- Active: (2 + 6) × D × (D_ff/8) = D × D_ff per token
```

### Computation Cost

Both use similar FLOPs per token, but DeepSeek:
- More routing overhead (64 vs 8 experts)
- Better GPU utilization (more parallelism in smaller experts)
- Shared experts always hit in cache

---

## Practical Guidelines

### Hyperparameter Selection

| Parameter | Typical Range | DeepSeek Default |
|-----------|---------------|------------------|
| `n_shared` | 1-4 | 2 |
| `n_routed` | 16-128 | 64 |
| `top_k` | 2-8 | 6 |
| `aux_loss_coef` | 0.001-0.1 | 0.01 |
| `segmentation` | 2-16x | 8x |

### Training Tips

1. **Monitor expert utilization** - All routed experts should be ~equally used
2. **Adjust aux_loss_coef** - Increase if load imbalanced, decrease if experts too similar
3. **Shared experts first** - They stabilize training early
4. **Gradual capacity** - Start with higher capacity, reduce over training

### Debugging

Common issues:
- **Expert collapse**: Some experts unused → increase aux_loss_coef
- **Poor specialization**: All experts similar → decrease aux_loss_coef
- **Training instability**: Large gradients → add gradient clipping

---

## Comparison: Standard MoE vs DeepSeek MoE

| Aspect | Standard MoE | DeepSeek MoE |
|--------|--------------|--------------|
| Expert Count | 8-16 large | 64+ small |
| Shared Knowledge | None | Dedicated shared experts |
| Routing Granularity | Coarse | Fine-grained |
| Routing Constraint | Any expert | Device-limited optional |
| Load Balancing | Per-expert | Per-routed-expert |
| Activation Pattern | Pure sparse | Shared + sparse |

---

## References

- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [Mixture-of-Experts Meets Instruction Tuning](https://arxiv.org/abs/2305.14705)
