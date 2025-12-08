# Auxiliary-Loss-Free Load Balancing

> **DeepSeek's Clean Gradient Training for MoE**

Load balancing is critical for MoE training: without it, a few experts dominate while others remain dormant. DeepSeek introduces **auxiliary-loss-free load balancing** using learnable biases that affect routing but not gradients.

---

## Table of Contents

1. [The Load Balancing Problem](#1-the-load-balancing-problem)
2. [Traditional Auxiliary Loss Approach](#2-traditional-auxiliary-loss-approach)
3. [DeepSeek's Bias-Based Solution](#3-deepseeks-bias-based-solution)
4. [Implementation Across Backends](#4-implementation-across-backends)
5. [Practical Tuning Guide](#5-practical-tuning-guide)

---

## 1. The Load Balancing Problem

### Expert Collapse

Without intervention, MoE routing tends to collapse:

```
Initial State (Step 0):
Expert 0: ████████████ 12.5% (balanced)
Expert 1: ████████████ 12.5%
Expert 2: ████████████ 12.5%
Expert 3: ████████████ 12.5%
Expert 4: ████████████ 12.5%
Expert 5: ████████████ 12.5%
Expert 6: ████████████ 12.5%
Expert 7: ████████████ 12.5%

After 1000 Steps (No Balancing):
Expert 0: ██████████████████████████████████████ 45%
Expert 1: ██████████████████████ 25%
Expert 2: ██████████████ 15%
Expert 3: ████████ 8%
Expert 4: ████ 4%
Expert 5: ██ 2%
Expert 6: █ 0.8%
Expert 7:   0.2% (dead)
```

### Why Collapse Happens

1. **Rich-get-richer**: Popular experts get more gradients, improve faster
2. **Gradient magnitude**: Small probabilities → small gradients
3. **Initialization bias**: Random init creates initial preferences
4. **Token specialization**: Some patterns are more common

---

## 2. Traditional Auxiliary Loss Approach

### The Standard Solution

Add a loss term that penalizes imbalanced load:

```python
def auxiliary_load_loss(router_probs: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """
    Standard auxiliary loss for load balancing.

    L_aux = α × n_experts × Σ(f_i × P_i)

    Where:
    - f_i = fraction of tokens routed to expert i
    - P_i = mean routing probability to expert i
    - α = balancing coefficient
    """
    n_experts = router_probs.shape[-1]
    n_tokens = router_probs.shape[0]

    # Fraction of tokens per expert
    top_expert = router_probs.argmax(dim=-1)
    expert_counts = torch.bincount(top_expert, minlength=n_experts).float()
    f = expert_counts / n_tokens  # [n_experts]

    # Mean probability per expert
    P = router_probs.mean(dim=0)  # [n_experts]

    # Auxiliary loss (product of fraction and probability)
    L_aux = alpha * n_experts * (f * P).sum()

    return L_aux
```

### Problems with Auxiliary Loss

| Issue | Impact |
|-------|--------|
| **Gradient interference** | Aux loss gradients can conflict with task gradients |
| **Hyperparameter sensitivity** | α too high → quality degradation; too low → collapse |
| **Training instability** | Sudden load shifts cause loss spikes |
| **Suboptimal trade-off** | Must balance quality vs balance |

```
Loss Landscape with Auxiliary Loss:

     ▲ Loss
     │
     │  ╱╲   ╱╲
     │ ╱  ╲ ╱  ╲ ← Auxiliary loss creates
     │╱    ╳    ╲   conflicting gradients
     │    ╱ ╲
     │───╱───╲───► Expert Load
        low  high
```

---

## 3. DeepSeek's Bias-Based Solution

### Core Insight

Separate routing decisions from gradient flow:

```
Traditional Approach:
    Router logits → Softmax → Top-K → Weights (used in forward AND backward)
                                         ↓
                              Auxiliary loss added to gradients

DeepSeek Approach:
    Router logits → Softmax → Top-K → Weights (clean gradients)
         ↓
    + Bias ────────────────────────→ Affects routing only
         ↑                           (NOT in gradient computation)
    Updated AFTER batch based on load
```

### The Algorithm

```python
class RouterBiasController:
    """
    Auxiliary-loss-free load balancing via learnable biases.

    Key principle: Update biases AFTER each batch, not during backward pass.
    This keeps gradients clean while still balancing load.

    Algorithm:
    1. Track exponential moving average of expert loads
    2. After each forward pass:
       - Compute load deviation from mean
       - Update biases: overloaded experts get negative bias
    3. Biases affect softmax routing but NOT gating weights
    """

    def __init__(
        self,
        n_experts: int,
        update_alpha: float = 0.001,
        ema_decay: float = 0.99,
        target_load: Optional[float] = None,
    ):
        self.n_experts = n_experts
        self.update_alpha = update_alpha
        self.ema_decay = ema_decay
        self.target_load = target_load or (1.0 / n_experts)

        # EMA of expert loads
        self.load_ema = np.zeros(n_experts)
        self.steps = 0

    def update(
        self,
        expert_counts: np.ndarray,
        biases: torch.Tensor,
    ) -> None:
        """
        Update biases based on observed load.

        Called AFTER forward pass, not during backward.

        Args:
            expert_counts: Number of tokens routed to each expert
            biases: The bias tensor to update (in-place)
        """
        # Normalize counts to fractions
        total = expert_counts.sum()
        if total == 0:
            return

        load_fractions = expert_counts / total

        # Update EMA
        if self.steps == 0:
            self.load_ema = load_fractions
        else:
            self.load_ema = (
                self.ema_decay * self.load_ema +
                (1 - self.ema_decay) * load_fractions
            )
        self.steps += 1

        # Compute bias update
        # Overloaded experts (high load) get NEGATIVE bias
        # Underloaded experts (low load) get POSITIVE bias
        load_deviation = self.load_ema - self.target_load
        bias_update = self.update_alpha * load_deviation

        # Apply update (detached from gradient graph)
        with torch.no_grad():
            biases.data -= torch.from_numpy(bias_update).to(biases.device).float()

    def get_load_stats(self) -> Dict[str, float]:
        """Return current load statistics."""
        return {
            'load_mean': self.load_ema.mean(),
            'load_std': self.load_ema.std(),
            'load_max': self.load_ema.max(),
            'load_min': self.load_ema.min(),
            'load_imbalance': self.load_ema.max() / (self.load_ema.min() + 1e-8),
        }
```

### How It Works

```
Step 1: Forward pass with current biases
┌──────────────────────────────────────────────────────────┐
│  logits = router(x)                # Raw router output   │
│  logits_biased = logits + bias     # Add bias            │
│  probs = softmax(logits_biased)    # Routing probs       │
│  top_k = topk(probs)               # Select experts      │
│                                                          │
│  # IMPORTANT: Gating weights use original logits!       │
│  gate_weights = softmax(logits)[top_k.indices]          │
└──────────────────────────────────────────────────────────┘

Step 2: Backward pass (clean gradients)
┌──────────────────────────────────────────────────────────┐
│  loss.backward()                                        │
│                                                          │
│  # Gradients flow through gate_weights (original logits)│
│  # NOT through biased routing - biases are detached     │
└──────────────────────────────────────────────────────────┘

Step 3: Update biases AFTER backward
┌──────────────────────────────────────────────────────────┐
│  # Count expert usage from this batch                    │
│  expert_counts = count_expert_usage(top_k.indices)       │
│                                                          │
│  # Update biases (separate from gradient computation)   │
│  bias_controller.update(expert_counts, self.bias)       │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Across Backends

### PyTorch Implementation

From `src/deepseek/torch/model/moe.py`:

```python
class DeepSeekMoEV3(nn.Module):
    def __init__(self, config: DeepSeekMoEV3Config):
        super().__init__()
        # ... other initialization ...

        # Router
        self.expert_router = nn.Linear(config.d_model, config.n_routed_experts, bias=False)

        # Learnable biases (separate from router weights)
        # These are Parameters but we update them manually, not via gradient
        self.expert_biases = nn.Parameter(
            torch.zeros(config.n_routed_experts),
            requires_grad=False  # No gradients!
        )

        # Bias controller
        self.bias_controller = RouterBiasController(
            n_experts=config.n_routed_experts,
            update_alpha=config.bias_update_alpha,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        # Router logits
        router_logits = self.expert_router(x_flat)

        # === Key: Separate routing from gating ===
        # Routing uses biased logits
        routing_logits = router_logits + self.expert_biases
        routing_probs = F.softmax(routing_logits, dim=-1)
        top_k = torch.topk(routing_probs, k=self.config.top_k_experts, dim=-1)

        # Gating uses ORIGINAL logits (clean gradients)
        gating_probs = F.softmax(router_logits, dim=-1)
        gate_weights = torch.gather(gating_probs, 1, top_k.indices)
        gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Dispatch to experts using gate_weights
        output = self._dispatch_with_weights(x_flat, top_k.indices, gate_weights)

        # Track load for bias update (after forward, before backward)
        expert_counts = self._count_expert_usage(top_k.indices)

        # Add shared experts
        for shared in self.shared_experts:
            output = output + shared(x_flat)

        return output.view(batch, seq_len, d_model), {
            'expert_indices': top_k.indices,
            'expert_counts': expert_counts,
            'load_stats': self.bias_controller.get_load_stats(),
        }

    def update_biases(self, expert_counts: np.ndarray):
        """Called after optimizer.step(), not during backward."""
        self.bias_controller.update(expert_counts, self.expert_biases)

    def _count_expert_usage(self, indices: torch.Tensor) -> np.ndarray:
        """Count how many tokens went to each expert."""
        counts = torch.bincount(
            indices.view(-1),
            minlength=self.config.n_routed_experts
        )
        return counts.cpu().numpy()
```

### Rust Implementation

```rust
pub struct RouterBiasController {
    n_experts: usize,
    update_alpha: f32,
    ema_decay: f32,
    load_ema: Vec<f32>,
    target_load: f32,
    steps: usize,
}

impl RouterBiasController {
    pub fn new(n_experts: usize, update_alpha: f32) -> Self {
        Self {
            n_experts,
            update_alpha,
            ema_decay: 0.99,
            load_ema: vec![0.0; n_experts],
            target_load: 1.0 / n_experts as f32,
            steps: 0,
        }
    }

    pub fn update(&mut self, expert_counts: &[usize], biases: &mut Tensor) -> Result<()> {
        let total: usize = expert_counts.iter().sum();
        if total == 0 {
            return Ok(());
        }

        // Normalize to fractions
        let load_fractions: Vec<f32> = expert_counts
            .iter()
            .map(|&c| c as f32 / total as f32)
            .collect();

        // Update EMA
        if self.steps == 0 {
            self.load_ema = load_fractions;
        } else {
            for (ema, load) in self.load_ema.iter_mut().zip(load_fractions.iter()) {
                *ema = self.ema_decay * *ema + (1.0 - self.ema_decay) * load;
            }
        }
        self.steps += 1;

        // Compute bias updates
        let bias_updates: Vec<f32> = self.load_ema
            .iter()
            .map(|&load| self.update_alpha * (load - self.target_load))
            .collect();

        // Apply updates (subtract: overloaded gets negative bias)
        let update_tensor = Tensor::from_vec(bias_updates, (self.n_experts,), biases.device())?;
        *biases = (biases.clone() - update_tensor)?;

        Ok(())
    }

    pub fn load_imbalance(&self) -> f32 {
        let max = self.load_ema.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = self.load_ema.iter().cloned().fold(f32::INFINITY, f32::min);
        max / (min + 1e-8)
    }
}
```

### MLX Implementation

From `src/deepseek/mlx/moe.py`:

```python
class LoadBalancingState:
    """Track load statistics for bias-based balancing in MLX."""

    def __init__(self, n_experts: int, ema_decay: float = 0.99):
        self.n_experts = n_experts
        self.ema_decay = ema_decay
        self.load_ema = mx.zeros((n_experts,))
        self.steps = 0

    def update(self, expert_indices: mx.array):
        """Update load EMA from routing decisions."""
        # Count expert usage
        flat_indices = expert_indices.reshape(-1)
        counts = mx.zeros((self.n_experts,))

        # MLX doesn't have bincount, use scatter_add equivalent
        for idx in range(self.n_experts):
            counts = counts.at[idx].set((flat_indices == idx).sum())

        # Normalize
        total = counts.sum()
        if total > 0:
            load_fractions = counts / total

            # Update EMA
            if self.steps == 0:
                self.load_ema = load_fractions
            else:
                self.load_ema = (
                    self.ema_decay * self.load_ema +
                    (1 - self.ema_decay) * load_fractions
                )
            self.steps += 1

    def get_bias_update(self, alpha: float = 0.001) -> mx.array:
        """Compute bias updates based on current load."""
        target = 1.0 / self.n_experts
        deviation = self.load_ema - target
        return alpha * deviation


class DeepSeekMoEV3(nn.Module):
    def __init__(self, config: DeepSeekMoEV3Config):
        super().__init__()
        self.config = config

        # Expert biases (not a parameter - updated manually)
        self.expert_biases = mx.zeros((config.n_routed_experts,))

        # Load tracking
        self.load_state = LoadBalancingState(config.n_routed_experts)

        # ... other initialization ...

    def __call__(self, x: mx.array) -> Tuple[mx.array, Dict]:
        batch, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)

        # Router with bias
        router_logits = self.expert_gate(x_flat)
        routing_logits = router_logits + self.expert_biases
        routing_probs = mx.softmax(routing_logits, axis=-1)

        # Top-k selection
        top_k_indices = mx.argpartition(-routing_probs, kth=self.config.top_k_experts, axis=-1)
        top_k_indices = top_k_indices[:, :self.config.top_k_experts]

        # Gating weights from ORIGINAL logits
        gating_probs = mx.softmax(router_logits, axis=-1)
        gate_weights = mx.take_along_axis(gating_probs, top_k_indices, axis=-1)
        gate_weights = gate_weights / (gate_weights.sum(axis=-1, keepdims=True) + 1e-6)

        # Dispatch with clean gating weights
        output = self._dispatch(x_flat, top_k_indices, gate_weights)

        # Update load statistics
        self.load_state.update(top_k_indices)

        return output.reshape(batch, seq_len, d_model), {'load_state': self.load_state}

    def update_biases(self):
        """Call after each training step."""
        bias_update = self.load_state.get_bias_update(self.config.bias_update_alpha)
        self.expert_biases = self.expert_biases - bias_update
```

---

## 5. Practical Tuning Guide

### Hyperparameter Selection

| Parameter | Recommended | Range | Notes |
|-----------|-------------|-------|-------|
| `update_alpha` | 0.001 | 0.0001 - 0.01 | Higher = faster balancing, more noise |
| `ema_decay` | 0.99 | 0.95 - 0.999 | Higher = smoother, slower response |
| `target_load` | 1/n_experts | - | Uniform by default |

### Monitoring Load Balance

```python
def log_load_balance(moe_module, step: int, writer: SummaryWriter):
    """Log load balancing metrics to TensorBoard."""
    stats = moe_module.bias_controller.get_load_stats()

    writer.add_scalar('moe/load_mean', stats['load_mean'], step)
    writer.add_scalar('moe/load_std', stats['load_std'], step)
    writer.add_scalar('moe/load_imbalance', stats['load_imbalance'], step)

    # Histogram of expert loads
    writer.add_histogram(
        'moe/expert_loads',
        moe_module.bias_controller.load_ema,
        step
    )

    # Histogram of biases
    writer.add_histogram(
        'moe/expert_biases',
        moe_module.expert_biases.detach().cpu().numpy(),
        step
    )
```

### Training Loop Integration

```python
# Training loop with bias-based load balancing
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()

    # Forward pass
    output, moe_info = model(batch['input_ids'])
    loss = criterion(output, batch['labels'])

    # Backward pass (clean gradients!)
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step
    optimizer.step()

    # === Bias update AFTER optimizer step ===
    for module in model.modules():
        if isinstance(module, DeepSeekMoEV3):
            module.update_biases(moe_info['expert_counts'])

    # Logging
    if step % 100 == 0:
        log_load_balance(module, step, writer)
```

### Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| Load still imbalanced | `update_alpha` too low | Increase to 0.005-0.01 |
| Training unstable | `update_alpha` too high | Decrease to 0.0001-0.0005 |
| Oscillating load | `ema_decay` too low | Increase to 0.995-0.999 |
| Slow convergence | `ema_decay` too high | Decrease to 0.95-0.98 |

---

## Summary

DeepSeek's auxiliary-loss-free load balancing:

1. **Separates routing from gating**: Biased softmax for selection, original softmax for weights
2. **Post-hoc updates**: Modify biases after forward pass, not during backward
3. **EMA smoothing**: Track load over time to avoid overreaction
4. **Clean gradients**: Task loss gradients flow without interference

This approach achieves balanced load without sacrificing model quality.

---

## Next Steps

- [Mixture of Experts Architecture](./mixture_of_experts.md)
- [Latent Attention Mechanics](./latent_attention.md)
- [Production Scaling Guide](../03_production_scaling_guide.md)
