# 256-Expert MoE with Hierarchical Routing

## Overview

DeepSeek V3 uses a **256-expert Mixture-of-Experts** architecture with hierarchical two-stage routing. This design enables the model to scale to 671B total parameters while only activating 37B per token, achieving a 18x efficiency improvement.

**Key Papers:**
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066) (DeepSeek-AI, 2024)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek-AI, 2024)
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (Shazeer et al., 2017)
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (Mistral AI, 2024)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      256-EXPERT MOE ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input Token ─────────┬───────────────────────────────────────────────────► │
│                       │                                                     │
│                       ▼                                                     │
│               ┌──────────────┐                                              │
│               │   ROUTER     │                                              │
│               │ (Gating Net) │                                              │
│               └──────┬───────┘                                              │
│                      │                                                      │
│          ┌───────────┴───────────┐                                          │
│          │     STAGE 1: GROUPS   │                                          │
│          │  Select top-G groups  │                                          │
│          └───────────┬───────────┘                                          │
│                      │                                                      │
│    ┌─────────────────┼─────────────────┐                                    │
│    │                 │                 │                                    │
│    ▼                 ▼                 ▼                                    │
│ ┌──────────┐   ┌──────────┐      ┌──────────┐                               │
│ │ Group 0  │   │ Group 1  │ ...  │ Group 7  │  (8 groups)                   │
│ │ 32 exp.  │   │ 32 exp.  │      │ 32 exp.  │                               │
│ └────┬─────┘   └────┬─────┘      └────┬─────┘                               │
│      │              │                 │                                     │
│      │              │                 │                                     │
│ ┌────┴─────┐   ┌────┴─────┐      ┌────┴─────┐                               │
│ │ STAGE 2  │   │ STAGE 2  │      │ STAGE 2  │                               │
│ │ Top-K in │   │ Top-K in │      │ Top-K in │                               │
│ │  group   │   │  group   │      │  group   │                               │
│ └────┬─────┘   └────┬─────┘      └────┬─────┘                               │
│      │              │                 │                                     │
│      ▼              ▼                 ▼                                     │
│ ┌─────────┐   ┌─────────┐       ┌─────────┐                                 │
│ │Expert 0 │   │Expert 32│  ...  │Expert 224│                                │
│ │Expert 1 │   │Expert 33│       │Expert 225│                                │
│ │  ...    │   │  ...    │       │  ...     │                                │
│ │Expert 3 │   │Expert 35│       │Expert 227│                                │
│ └────┬────┘   └────┬────┘       └────┬─────┘                                │
│      │              │                 │                                     │
│      └──────────────┴────────┬────────┘                                     │
│                              │                                              │
│                              ▼                                              │
│                    ┌──────────────────┐                                     │
│                    │  WEIGHTED SUM    │                                     │
│                    │  + Shared Expert │                                     │
│                    └────────┬─────────┘                                     │
│                             │                                               │
│                             ▼                                               │
│                        Output Token                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Configuration

```python
@dataclass
class DeepSeekMoEV3Config:
    """Configuration for 256-Expert MoE."""
    
    # Model dimensions
    d_model: int = 7168
    
    # Expert configuration
    num_experts: int = 256
    num_experts_per_group: int = 32
    num_groups: int = 8              # 256 / 32 = 8 groups
    
    # Routing configuration
    top_k_experts: int = 8           # Experts per token
    top_k_groups: int = 4            # Groups per token
    experts_per_group: int = 2       # Experts selected per group
    
    # Expert FFN dimensions
    routed_hidden_mult: float = 1.5  # routed_hidden = d_model * mult
    shared_hidden_mult: float = 1.5  # shared_hidden = d_model * mult
    
    # Shared expert (always active)
    num_shared_experts: int = 1
    
    # Load balancing
    aux_loss_coef: float = 0.001
    z_loss_coef: float = 0.001
    
    # Advanced options
    normalize_expert_weights: bool = True
    capacity_factor: float = 1.25
    
    @property
    def routed_hidden(self) -> int:
        return int(self.d_model * self.routed_hidden_mult)
    
    @property
    def shared_hidden(self) -> int:
        return int(self.d_model * self.shared_hidden_mult)
```

## Hierarchical Routing Implementation

### Two-Stage Router

```python
class HierarchicalRouter(nn.Module):
    """Two-stage hierarchical router for 256 experts."""
    
    def __init__(self, config: DeepSeekMoEV3Config):
        super().__init__()
        self.config = config
        
        # Group-level router
        self.group_router = nn.Linear(config.d_model, config.num_groups, bias=False)
        
        # Expert-level routers (one per group)
        self.expert_routers = nn.ModuleList([
            nn.Linear(config.d_model, config.num_experts_per_group, bias=False)
            for _ in range(config.num_groups)
        ])
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
            expert_indices: (batch, seq, top_k) - selected expert indices
            expert_weights: (batch, seq, top_k) - routing weights
            aux_data: auxiliary data for load balancing
        """
        batch, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)  # (B*S, D)
        
        # Stage 1: Select top-G groups
        group_logits = self.group_router(x_flat)  # (B*S, num_groups)
        group_probs = F.softmax(group_logits / self.temperature, dim=-1)
        
        top_g_probs, top_g_indices = torch.topk(
            group_probs, 
            self.config.top_k_groups, 
            dim=-1
        )  # (B*S, top_k_groups)
        
        # Stage 2: Select top-K experts within selected groups
        all_expert_indices = []
        all_expert_weights = []
        
        for i, group_idx in enumerate(range(self.config.num_groups)):
            # Get expert logits for this group
            expert_logits = self.expert_routers[group_idx](x_flat)  # (B*S, experts_per_group)
            expert_probs = F.softmax(expert_logits / self.temperature, dim=-1)
            
            # Select top experts in group
            top_e_probs, top_e_local = torch.topk(
                expert_probs,
                self.config.experts_per_group,
                dim=-1
            )
            
            # Convert to global expert indices
            global_expert_idx = group_idx * self.config.num_experts_per_group + top_e_local
            
            # Mask by group selection
            group_selected = (top_g_indices == group_idx).any(dim=-1, keepdim=True)
            group_weight = group_probs[:, group_idx:group_idx+1]
            
            # Combined weight: group_prob * expert_prob
            combined_weights = group_weight * top_e_probs * group_selected.float()
            
            all_expert_indices.append(global_expert_idx)
            all_expert_weights.append(combined_weights)
        
        # Concatenate and select final top-K
        all_indices = torch.cat(all_expert_indices, dim=-1)  # (B*S, num_groups * experts_per_group)
        all_weights = torch.cat(all_expert_weights, dim=-1)
        
        # Final top-K selection
        top_k_weights, top_k_positions = torch.topk(
            all_weights,
            self.config.top_k_experts,
            dim=-1
        )
        top_k_indices = torch.gather(all_indices, -1, top_k_positions)
        
        # Normalize weights
        if self.config.normalize_expert_weights:
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Reshape back
        expert_indices = top_k_indices.reshape(batch, seq_len, -1)
        expert_weights = top_k_weights.reshape(batch, seq_len, -1)
        
        aux_data = {
            'group_probs': group_probs.reshape(batch, seq_len, -1),
            'group_logits': group_logits.reshape(batch, seq_len, -1),
        }
        
        return expert_indices, expert_weights, aux_data
```

### Full MoE Module

```python
class DeepSeekMoEV3(nn.Module):
    """256-Expert MoE with hierarchical routing."""
    
    def __init__(self, config: DeepSeekMoEV3Config):
        super().__init__()
        self.config = config
        
        # Hierarchical router
        self.router = HierarchicalRouter(config)
        
        # 256 routed experts (grouped for efficiency)
        self.experts = nn.ModuleList([
            self._create_expert(config.d_model, config.routed_hidden)
            for _ in range(config.num_experts)
        ])
        
        # Shared expert (always active)
        self.shared_expert = self._create_expert(config.d_model, config.shared_hidden)
    
    def _create_expert(self, d_model: int, hidden: int) -> nn.Module:
        """Create a single FFN expert."""
        return nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE."""
        batch, seq_len, d_model = x.shape
        
        # Get routing decisions
        expert_indices, expert_weights, aux_data = self.router(x)
        
        # Shared expert (always computed)
        shared_out = self.shared_expert(x)
        
        # Routed experts
        routed_out = self._compute_routed_experts(
            x, expert_indices, expert_weights
        )
        
        # Combine shared + routed
        output = shared_out + routed_out
        
        # Store aux data for loss computation
        self._aux_data = aux_data
        
        return output
    
    def _compute_routed_experts(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted sum of routed expert outputs."""
        batch, seq_len, d_model = x.shape
        top_k = indices.shape[-1]
        
        x_flat = x.reshape(-1, d_model)  # (B*S, D)
        indices_flat = indices.reshape(-1, top_k)  # (B*S, K)
        weights_flat = weights.reshape(-1, top_k)  # (B*S, K)
        
        output = torch.zeros_like(x_flat)
        
        # Group tokens by expert for efficient batched computation
        for expert_idx in range(self.config.num_experts):
            # Find all (token, position) pairs routed to this expert
            mask = (indices_flat == expert_idx)  # (B*S, K)
            
            if not mask.any():
                continue
            
            # Get token indices that use this expert
            token_indices = mask.any(dim=-1).nonzero(as_tuple=True)[0]
            
            if len(token_indices) == 0:
                continue
            
            # Get tokens and weights for this expert
            tokens = x_flat[token_indices]  # (num_tokens, D)
            
            # Get weights for this expert
            expert_weights = (weights_flat[token_indices] * mask[token_indices].float()).sum(dim=-1)
            
            # Compute expert output
            expert_out = self.experts[expert_idx](tokens)  # (num_tokens, D)
            
            # Weighted accumulation
            output.index_add_(0, token_indices, expert_out * expert_weights.unsqueeze(-1))
        
        return output.reshape(batch, seq_len, d_model)
    
    def get_load_balance_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return load balancing statistics."""
        # Expert utilization across batch
        # Returns (expert_counts, group_counts)
        return self._aux_data.get('expert_counts', torch.zeros(self.config.num_experts)), \
               self._aux_data.get('group_counts', torch.zeros(self.config.num_groups))
```

## Load Balancing

### Auxiliary Loss Functions

```python
def compute_aux_loss(
    router_logits: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    config: DeepSeekMoEV3Config
) -> Dict[str, torch.Tensor]:
    """Compute auxiliary losses for load balancing."""
    
    batch_size, seq_len, _ = router_logits.shape
    num_tokens = batch_size * seq_len
    
    # 1. Load Balancing Loss (encourages uniform expert usage)
    # Count tokens per expert
    expert_counts = torch.zeros(num_experts, device=router_logits.device)
    for k in range(expert_indices.shape[-1]):
        expert_counts.scatter_add_(
            0,
            expert_indices[:, :, k].flatten(),
            torch.ones(num_tokens, device=router_logits.device)
        )
    
    # Normalize
    expert_probs = expert_counts / num_tokens
    
    # Router probability mass per expert
    router_probs = F.softmax(router_logits, dim=-1).mean(dim=(0, 1))
    
    # Balance loss: minimize correlation between routing and expert load
    load_balance_loss = (expert_probs * router_probs).sum() * num_experts
    
    # 2. Z-Loss (prevents router logits from growing too large)
    z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
    
    # 3. Group Diversity Loss (encourage using multiple groups)
    if hasattr(config, 'num_groups'):
        group_indices = expert_indices // config.num_experts_per_group
        unique_groups = group_indices.unique(dim=-1).float().mean()
        group_diversity_loss = -unique_groups  # Maximize diversity
    else:
        group_diversity_loss = torch.tensor(0.0)
    
    return {
        'load_balance_loss': config.aux_loss_coef * load_balance_loss,
        'z_loss': config.z_loss_coef * z_loss,
        'group_diversity_loss': 0.0001 * group_diversity_loss,
        'expert_utilization': expert_probs,
    }
```

### Expert Capacity Management

```python
class CapacityManagedMoE(nn.Module):
    """MoE with expert capacity limits for training stability."""
    
    def __init__(self, config: DeepSeekMoEV3Config):
        super().__init__()
        self.config = config
        self.capacity_factor = config.capacity_factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        
        # Calculate capacity per expert
        tokens_per_expert = (batch * seq_len * self.config.top_k_experts) / self.config.num_experts
        capacity = int(tokens_per_expert * self.capacity_factor)
        
        # Get routing with capacity limits
        indices, weights = self._route_with_capacity(x, capacity)
        
        return self._compute_with_capacity(x, indices, weights, capacity)
    
    def _route_with_capacity(
        self,
        x: torch.Tensor,
        capacity: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens respecting capacity limits."""
        # Get initial routing
        indices, weights, _ = self.router(x)
        
        batch, seq_len, top_k = indices.shape
        
        # Track expert usage
        expert_counts = torch.zeros(self.config.num_experts, device=x.device)
        
        # Iterate and drop tokens exceeding capacity
        valid_mask = torch.ones_like(indices, dtype=torch.bool)
        
        for b in range(batch):
            for s in range(seq_len):
                for k in range(top_k):
                    expert_idx = indices[b, s, k].item()
                    if expert_counts[expert_idx] >= capacity:
                        valid_mask[b, s, k] = False
                    else:
                        expert_counts[expert_idx] += 1
        
        # Zero out dropped token weights
        weights = weights * valid_mask.float()
        
        # Renormalize
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return indices, weights
```

## Rust Implementation

```rust
pub struct DeepSeekMoEV3 {
    config: DeepSeekMoEV3Config,
    group_router: Linear,
    expert_routers: Vec<Linear>,
    experts: Vec<FFNExpert>,
    shared_expert: FFNExpert,
}

impl DeepSeekMoEV3 {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (batch, seq_len, d_model) = x.dims3();
        
        // Stage 1: Group routing
        let group_logits = self.group_router.forward(&x.flatten(0, 1));
        let group_probs = softmax(&group_logits, -1);
        let (top_g_probs, top_g_indices) = top_k(&group_probs, self.config.top_k_groups);
        
        // Stage 2: Expert routing within groups
        let mut all_expert_outputs = Vec::new();
        let mut all_weights = Vec::new();
        
        for (group_idx, expert_router) in self.expert_routers.iter().enumerate() {
            let expert_logits = expert_router.forward(&x.flatten(0, 1));
            let expert_probs = softmax(&expert_logits, -1);
            
            // Get group weight
            let group_mask = top_g_indices.eq(group_idx as i64);
            let group_weight = group_probs.select(-1, group_idx);
            
            // Select experts in this group
            let (top_e_probs, top_e_local) = top_k(&expert_probs, self.config.experts_per_group);
            
            // Compute expert outputs
            let global_idx = group_idx * self.config.num_experts_per_group;
            for i in 0..self.config.experts_per_group {
                let expert_idx = global_idx + top_e_local.select(-1, i as i64);
                let expert_out = self.experts[expert_idx].forward(&x.flatten(0, 1));
                
                let weight = &group_weight * &top_e_probs.select(-1, i as i64) * &group_mask;
                all_expert_outputs.push(expert_out);
                all_weights.push(weight);
            }
        }
        
        // Combine expert outputs
        let routed_out = self.weighted_sum(&all_expert_outputs, &all_weights);
        
        // Add shared expert
        let shared_out = self.shared_expert.forward(&x);
        
        shared_out + routed_out.reshape(&[batch, seq_len, d_model])
    }
}
```

## Parameter Efficiency

### Scaling Analysis

| Component | Parameters | Active per Token |
|-----------|-----------|------------------|
| 256 Routed Experts | 614B | 28B (8 experts) |
| Shared Expert | 2.4B | 2.4B (always) |
| Router | 14M | 14M |
| Attention + Embedding | 56B | 7B |
| **Total** | **671B** | **37B** |

### Memory Distribution

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY DISTRIBUTION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Experts (256): ████████████████████████████████████████  91.4% │
│  Attention:     ████                                       8.1% │
│  Embeddings:    █                                          0.4% │
│  Router:        |                                          0.01%│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Benefits

### Training Efficiency
- **8 experts/token**: Maintains quality while reducing compute
- **Hierarchical routing**: O(log n) routing vs O(n) for flat
- **Grouped structure**: Enables efficient parallelization

### Inference Optimization
- **Expert caching**: Keep hot experts in memory
- **Predictive loading**: Pre-fetch likely experts
- **Speculative routing**: Route while computing previous layer

## Summary

The 256-Expert MoE with hierarchical routing achieves:
- **18x parameter efficiency**: 37B active from 671B total
- **Scalable routing**: Two-stage hierarchy for fast decisions
- **Balanced load**: Auxiliary losses prevent collapse
- **Production ready**: Capacity management for stability

This architecture enables DeepSeek V3 to match or exceed dense models at a fraction of the compute cost.
