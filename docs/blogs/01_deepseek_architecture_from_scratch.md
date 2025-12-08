# DeepSeek Architecture from Scratch: A Three-Backend Deep Dive

> **A Principal Developer Advocate's Guide to Understanding DeepSeek-V3 and R1**

This comprehensive guide explores the DeepSeek-V3 and DeepSeek-R1 architectures through the lens of three distinct implementations: **Rust/Candle** (for production-grade Metal/CUDA), **PyTorch** (for research flexibility), and **MLX** (for Apple Silicon optimization). By examining how the same concepts materialize across different frameworks, you'll gain deeper architectural intuition than any single implementation could provide.

---

## Table of Contents

1. [The DeepSeek Philosophy](#1-the-deepseek-philosophy)
2. [Architecture Overview](#2-architecture-overview)
3. [Multi-Head Latent Attention (MLA)](#3-multi-head-latent-attention-mla)
4. [Mixture of Experts (MoE)](#4-mixture-of-experts-moe)
5. [Rotary Position Encoding (RoPE)](#5-rotary-position-encoding-rope)
6. [The Complete Transformer Block](#6-the-complete-transformer-block)
7. [Backend Comparison Matrix](#7-backend-comparison-matrix)
8. [When to Use Each Backend](#8-when-to-use-each-backend)

---

## 1. The DeepSeek Philosophy

DeepSeek-V3 represents a paradigm shift in large language model design. Instead of brute-forcing scale, DeepSeek achieves state-of-the-art performance through **architectural efficiency**:

| Design Principle | Traditional Approach | DeepSeek Approach |
|------------------|---------------------|-------------------|
| **KV Cache** | Store full K,V tensors | Compress to 14× smaller latent |
| **Expert Routing** | Auxiliary loss penalties | Bias-based load balancing |
| **Attention** | Standard Multi-Head | Decoupled content + position paths |
| **Scaling** | More parameters | Sparse activation via MoE |

### Why Three Backends?

This repository implements DeepSeek across three backends, each optimized for different use cases:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DeepSeek-From-Scratch                          │
├─────────────────┬─────────────────────┬─────────────────────────────┤
│   Rust/Candle   │      PyTorch        │           MLX               │
│   (Production)  │     (Research)      │    (Apple Silicon)          │
├─────────────────┼─────────────────────┼─────────────────────────────┤
│ • Metal/CUDA    │ • CUDA/MPS/CPU      │ • Metal + Neural Engine     │
│ • 13.5 steps/s  │ • 10.2 steps/sec    │ • 3.3 steps/sec (local)     │
│ • Type-safe     │ • Dynamic graphs    │ • Unified memory            │
│ • Zero-copy     │ • torch.compile     │ • Lazy evaluation           │
└─────────────────┴─────────────────────┴─────────────────────────────┘
```

---

## 2. Architecture Overview

The DeepSeek transformer follows a modified decoder-only architecture:

```
Input Tokens
     │
     ▼
┌─────────────┐
│  Embedding  │
└─────────────┘
     │
     ▼
┌─────────────────────────────────┐
│     Transformer Block × N       │
│  ┌───────────────────────────┐  │
│  │   RMSNorm                 │  │
│  │      │                    │  │
│  │   MLA Attention           │◄─── Multi-Head Latent Attention
│  │      │                    │     with Decoupled RoPE
│  │   + Residual              │  │
│  │      │                    │  │
│  │   RMSNorm                 │  │
│  │      │                    │  │
│  │   MoE / MLP               │◄─── 256 Experts with
│  │      │                    │     Hierarchical Routing
│  │   + Residual              │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
     │
     ▼
┌─────────────┐
│  RMSNorm    │
└─────────────┘
     │
     ▼
┌─────────────┐
│   LM Head   │
└─────────────┘
     │
     ▼
Output Logits
```

### Model Configuration Across Backends

**PyTorch** (`src/deepseek/torch/model/transformer.py:141-169`):

```python
class DeepSeekModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_latent: int = 128,        # MLA compression dimension
        d_rope: int = 32,           # RoPE dimension
        d_hidden: int = 2048,       # FFN hidden size
        num_experts: int = 8,       # Number of MoE experts
        num_shared: int = 1,        # Shared experts (always active)
        num_routed: int = 8,        # Routed experts
        top_k: int = 2,             # Top-k expert selection
        use_moe: bool = True,
        dropout: float = 0.0,
        attention_config: Optional[FlashAttentionConfig] = None,
        checkpoint_config: Optional[GradientCheckpointConfig] = None,
    ):
        # Weight tying: embed and head share weights
        self.head.weight = self.embed.weight
```

**Rust/Candle** (`rust-src/src/model/mla.rs:347-370`):

```rust
pub struct MultiHeadLatentAttention {
    d_model: usize,
    num_heads: usize,
    d_head: usize,
    d_latent: usize,         // Latent compression dimension
    w_q: Linear,             // Query projection
    w_dkv: Linear,           // Down-projection for KV
    w_uk: Linear,            // Up-projection for K
    w_uv: Linear,            // Up-projection for V
    w_o: Linear,             // Output projection
}

impl MultiHeadLatentAttention {
    pub fn new(d_model: usize, num_heads: usize, d_latent: usize, vb: VarBuilder) -> Result<Self> {
        let d_head = d_model / num_heads;
        // ...projections initialized via VarBuilder
    }
}
```

**MLX** (`src/deepseek/mlx/attention.py:413-434`):

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, d_rope=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent
        self.d_head = d_model // num_heads
        self.d_rope = d_rope

        # Compression path
        self.W_DK = nn.Linear(d_model, d_latent, bias=False)
        self.W_UV = nn.Linear(d_latent, num_heads * self.d_head, bias=False)

        # Query compression
        self.W_DQ = nn.Linear(d_model, d_latent, bias=False)
        self.W_UQ = nn.Linear(d_latent, num_heads * self.d_head, bias=False)
```

---

## 3. Multi-Head Latent Attention (MLA)

MLA is DeepSeek's signature innovation: **compress KV cache by 14× while maintaining attention quality**.

### The Core Insight

Standard attention stores `K` and `V` tensors of shape `[batch, heads, seq_len, head_dim]`. For long sequences, this becomes prohibitively expensive. MLA introduces a **latent bottleneck**:

```
Standard Attention Memory:    2 × num_heads × head_dim × seq_len
MLA Memory:                   d_latent × seq_len
Compression Ratio:            (2 × num_heads × head_dim) / d_latent ≈ 14×
```

### Three-Backend Implementation

#### PyTorch: Full-Featured MLA with Flash Attention

From `src/deepseek/torch/model/mla.py`:

```python
class DeepSeekAttention(nn.Module):
    """
    DeepSeek's decoupled attention with:
    - Content path: compressed KV via latent space
    - Position path: decoupled RoPE for positional awareness
    - Combined scores BEFORE softmax (key for correctness)
    """

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # === Content Path ===
        # Query: full projection
        q_c = self.w_q_c(x).view(batch, seq_len, self.n_heads, self.d_head)

        # Key-Value: compress to latent, then expand
        c_kv = self.w_down_kv(x)                    # [B, T, d_latent]
        k_c = self.w_up_k(c_kv)                     # [B, T, n_heads * d_head]
        v = self.w_up_v(c_kv)                       # [B, T, n_heads * d_head]

        # === Position Path (Decoupled RoPE) ===
        q_r = self.w_q_r(x).view(batch, seq_len, self.n_heads, self.d_rope)
        k_r = self.w_k_r(x).view(batch, seq_len, 1, self.d_rope)  # Shared across heads

        q_r, k_r = self.rope(q_r, k_r, seq_len, offset=kv_cache.seq_len if kv_cache else 0)

        # === CRITICAL: Combine scores BEFORE softmax ===
        # This is what makes decoupled RoPE work correctly
        content_scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / math.sqrt(self.d_head)
        position_scores = torch.matmul(q_r, k_r.transpose(-2, -1)) / math.sqrt(self.d_rope)

        attn_scores = content_scores + position_scores  # Combined pre-softmax!

        # Apply causal mask and softmax
        attn_weights = F.softmax(attn_scores + mask, dim=-1)
        output = torch.matmul(attn_weights, v)

        return self.w_o(output.view(batch, seq_len, -1))
```

#### Rust: Memory-Efficient Latent Cache

From `rust-src/src/model/mla.rs:454-515`:

```rust
impl MultiHeadLatentAttention {
    /// Forward pass with latent KV cache (memory-efficient).
    ///
    /// Stores compressed latent C_KV instead of full K/V tensors,
    /// achieving ~14× memory reduction.
    pub fn forward_with_latent_cache(
        &self,
        x: &Tensor,
        latent_cache: Option<&mut LatentKVCache>
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // 1. Query Path (full dimension)
        let q = self.w_q.forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;

        // 2. Key/Value Path - Compute compressed latent
        let c_kv = self.w_dkv.forward(x)?;  // [batch, seq_len, d_latent]

        // 3. Update latent cache (stores only d_latent, not full K/V)
        let c_kv_full = if let Some(cache) = latent_cache {
            cache.update(&c_kv)?  // ~14× smaller than standard cache
        } else {
            c_kv
        };

        // 4. Up-project cached latent to full K and V ON-DEMAND
        let total_seq_len = c_kv_full.dim(1)?;

        let k = self.w_uk.forward(&c_kv_full)?
            .reshape((batch_size, total_seq_len, self.num_heads, self.d_head))?;

        let v = self.w_uv.forward(&c_kv_full)?
            .reshape((batch_size, total_seq_len, self.num_heads, self.d_head))?;

        // 5. Standard scaled dot-product attention
        let scale = 1.0 / (self.d_head as f64).sqrt();
        let attn_scores = q.matmul(&k.transpose(2, 3)?)? * scale;
        // ... masking and softmax
    }
}
```

#### MLX: Unified Memory Optimization

From `src/deepseek/mlx/attention.py:78-152`:

```python
class LatentKVCache:
    """
    Latent KV Cache for MLA on MLX - stores compressed latent representations.

    Instead of storing full K/V tensors (batch, heads, seq_len, head_dim),
    this cache stores the compressed latent C_KV (batch, seq_len, d_latent),
    achieving approximately 14× memory reduction.
    """

    def __init__(self, batch_size: int, max_seq_len: int, d_latent: int):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.d_latent = d_latent
        self.current_seq_len = 0
        self.latent_cache = None  # Lazy initialization

    def update(self, c_kv: mx.array) -> mx.array:
        """Update cache with new compressed latent."""
        new_seq_len = c_kv.shape[1]

        # MLX's lazy evaluation means concatenation is efficient
        if self.current_seq_len == 0:
            self.latent_cache = c_kv
        else:
            self.latent_cache = mx.concatenate([self.latent_cache, c_kv], axis=1)

        self.current_seq_len += new_seq_len
        return self.latent_cache

    @staticmethod
    def memory_reduction_ratio(n_heads: int, head_dim: int, d_latent: int) -> float:
        """Calculate memory reduction vs standard KV cache."""
        standard_kv_size = 2 * n_heads * head_dim  # K and V
        return standard_kv_size / d_latent  # Typically ~14×
```

### Memory Comparison

| Configuration | Standard KV Cache | MLA Latent Cache | Reduction |
|---------------|-------------------|------------------|-----------|
| 7B model, 32K context | 16 GB | 1.1 GB | 14.5× |
| 70B model, 128K context | 256 GB | 18 GB | 14.2× |
| 671B model (V3), 128K | 2.4 TB | 170 GB | 14.1× |

---

## 4. Mixture of Experts (MoE)

DeepSeek-V3 uses **256 routed experts** with **hierarchical 2-stage routing** and **auxiliary-loss-free load balancing**.

### Hierarchical Routing Architecture

```
Input Hidden States [batch, seq_len, d_model]
                    │
                    ▼
        ┌───────────────────────┐
        │   Router (Stage 1)    │
        │   Group Selection     │
        │   256 → 8 groups      │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Router (Stage 2)    │
        │   Expert Selection    │
        │   32 experts/group    │
        │   Top-K = 8           │
        └───────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │Expert 0│  │Expert 1│  │  ...   │
   │(Shared)│  │(Routed)│  │        │
   └────────┘  └────────┘  └────────┘
        │           │           │
        └───────────┼───────────┘
                    ▼
           Weighted Sum Output
```

### Three-Backend Implementation

#### PyTorch: Full 256-Expert Implementation

From `src/deepseek/torch/model/moe.py`:

```python
@dataclass
class DeepSeekMoEV3Config:
    """Configuration for DeepSeek-V3 256-expert MoE."""
    d_model: int = 7168
    d_expert: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_expert_groups: int = 8          # Hierarchical routing
    n_experts_per_group: int = 32     # 256 / 8 = 32
    top_k_groups: int = 4
    top_k_experts: int = 8            # Total experts per token

    # Auxiliary-loss-free load balancing
    use_bias_update: bool = True
    bias_update_alpha: float = 0.001  # EMA update rate


class DeepSeekMoEV3(nn.Module):
    """256-Expert MoE with hierarchical routing and aux-loss-free balancing."""

    def __init__(self, config: DeepSeekMoEV3Config):
        super().__init__()
        self.config = config

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            ExpertV3(config.d_model, config.d_expert)
            for _ in range(config.n_shared_experts)
        ])

        # Routed experts (sparse activation)
        self.routed_experts = nn.ModuleList([
            ExpertV3(config.d_model, config.d_expert)
            for _ in range(config.n_routed_experts)
        ])

        # Two-stage router
        self.group_router = nn.Linear(config.d_model, config.n_expert_groups, bias=False)
        self.expert_router = nn.Linear(config.d_model, config.n_routed_experts, bias=False)

        # Learnable biases for load balancing (NOT in router weights)
        self.expert_biases = nn.Parameter(torch.zeros(config.n_routed_experts))

        # Load balancing controller
        self.bias_controller = RouterBiasController(
            n_experts=config.n_routed_experts,
            update_alpha=config.bias_update_alpha
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        # Stage 1: Group selection
        group_logits = self.group_router(x_flat)
        group_probs = F.softmax(group_logits, dim=-1)
        top_groups = torch.topk(group_probs, k=self.config.top_k_groups, dim=-1)

        # Stage 2: Expert selection within selected groups
        expert_logits = self.expert_router(x_flat)

        # Apply bias for load balancing (affects routing, NOT gradients)
        expert_logits_biased = expert_logits + self.expert_biases

        # Mask experts not in selected groups
        expert_mask = self._create_group_mask(top_groups.indices)
        expert_logits_masked = expert_logits_biased.masked_fill(~expert_mask, float('-inf'))

        # Select top-k experts
        expert_probs = F.softmax(expert_logits_masked, dim=-1)
        top_experts = torch.topk(expert_probs, k=self.config.top_k_experts, dim=-1)

        # Compute expert outputs
        output = self._compute_expert_outputs(x_flat, top_experts)

        # Add shared expert contributions
        for shared_expert in self.shared_experts:
            output = output + shared_expert(x_flat)

        return output.view(batch, seq_len, d_model), {'expert_usage': top_experts.indices}
```

#### Rust: MegaBlocks-Style Block-Sparse Dispatch

From `rust-src/src/model/moe.rs`:

```rust
/// 256-Expert MoE with hierarchical routing
pub struct DeepSeekMoEV3 {
    config: MoEConfig,
    shared_experts: Vec<Expert>,
    routed_experts: Vec<Expert>,
    group_router: Linear,
    expert_router: Linear,
    expert_biases: Tensor,
}

impl DeepSeekMoEV3 {
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, MoEMetrics)> {
        let (batch, seq_len, d_model) = x.dims3()?;
        let x_flat = x.reshape((batch * seq_len, d_model))?;

        // Stage 1: Group routing
        let group_logits = self.group_router.forward(&x_flat)?;
        let group_probs = ops::softmax(&group_logits, 1)?;
        let (top_group_vals, top_group_idx) = group_probs.topk(self.config.top_k_groups)?;

        // Stage 2: Expert routing within groups
        let expert_logits = self.expert_router.forward(&x_flat)?;
        let expert_logits_biased = (expert_logits + &self.expert_biases)?;

        // Block-sparse dispatch for efficiency
        let dispatch_result = self.block_sparse_dispatch(&x_flat, &expert_logits_biased, &top_group_idx)?;

        // Combine expert outputs
        let routed_output = self.combine_expert_outputs(&dispatch_result)?;

        // Add shared experts (always computed)
        let mut output = routed_output;
        for expert in &self.shared_experts {
            output = (output + expert.forward(&x_flat)?)?;
        }

        Ok((output.reshape((batch, seq_len, d_model))?, dispatch_result.metrics))
    }
}
```

#### MLX: Optimized Sparse MoE for Apple Silicon

From `src/deepseek/mlx/moe.py`:

```python
class DeepSeekMoEV3(nn.Module):
    """
    DeepSeek-V3 style MoE with 256 experts.

    Key features:
    - Hierarchical 2-stage routing (groups → experts)
    - Auxiliary-loss-free load balancing via bias updates
    - Shared experts for common knowledge
    """

    def __init__(self, config: DeepSeekMoEV3Config):
        super().__init__()
        self.config = config

        # Experts with SwiGLU activation
        self.shared_experts = [Expert(config) for _ in range(config.n_shared_experts)]
        self.routed_experts = [Expert(config) for _ in range(config.n_routed_experts)]

        # Hierarchical router
        self.group_gate = nn.Linear(config.d_model, config.n_expert_groups, bias=False)
        self.expert_gate = nn.Linear(config.d_model, config.n_routed_experts, bias=False)

        # Bias-based load balancing (no aux loss needed)
        self.expert_biases = mx.zeros((config.n_routed_experts,))

        # Load balancing state
        self.load_state = LoadBalancingState(config.n_routed_experts)

    def __call__(self, x: mx.array) -> Tuple[mx.array, Dict]:
        batch, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)
        n_tokens = x_flat.shape[0]

        # === Hierarchical Routing ===
        # Stage 1: Select top-k groups
        group_logits = self.group_gate(x_flat)
        group_probs = mx.softmax(group_logits, axis=-1)
        top_k_groups = mx.argpartition(-group_probs, kth=self.config.top_k_groups, axis=-1)
        top_k_groups = top_k_groups[:, :self.config.top_k_groups]

        # Stage 2: Select experts within groups
        expert_logits = self.expert_gate(x_flat)
        expert_logits_biased = expert_logits + self.expert_biases

        # Mask experts outside selected groups
        group_mask = self._create_group_mask(top_k_groups, n_tokens)
        expert_logits_masked = mx.where(group_mask, expert_logits_biased, -1e9)

        # Select top-k experts
        expert_probs = mx.softmax(expert_logits_masked, axis=-1)
        top_k_experts = mx.argpartition(-expert_probs, kth=self.config.top_k_experts, axis=-1)
        top_k_experts = top_k_experts[:, :self.config.top_k_experts]
        top_k_weights = mx.take_along_axis(expert_probs, top_k_experts, axis=-1)

        # Normalize weights
        top_k_weights = top_k_weights / (top_k_weights.sum(axis=-1, keepdims=True) + 1e-6)

        # === Compute Expert Outputs ===
        output = mx.zeros_like(x_flat)

        # Routed experts (sparse)
        for expert_idx in range(self.config.n_routed_experts):
            # Find tokens routed to this expert
            mask = (top_k_experts == expert_idx).any(axis=-1)
            if not mask.any():
                continue

            expert_input = x_flat[mask]
            expert_output = self.routed_experts[expert_idx](expert_input)

            # Weight by routing probability
            weight_idx = (top_k_experts[mask] == expert_idx).astype(mx.float32)
            weights = (top_k_weights[mask] * weight_idx).sum(axis=-1, keepdims=True)

            output = mx.where(mask.reshape(-1, 1), output + expert_output * weights, output)

        # Shared experts (always active)
        for shared_expert in self.shared_experts:
            output = output + shared_expert(x_flat)

        # Update load balancing statistics
        self.load_state.update(top_k_experts)

        return output.reshape(batch, seq_len, d_model), {'load_state': self.load_state}
```

### Auxiliary-Loss-Free Load Balancing

The key innovation is updating router biases **after** each batch without affecting gradients:

```python
class RouterBiasController:
    """
    Updates expert biases based on load without adding auxiliary loss.

    Algorithm (from DeepSeek-V3 paper):
    1. Track exponential moving average of expert loads
    2. After each batch, compute bias update:
       bias_update = alpha * (load - mean_load)
    3. Subtract from biases (overloaded experts get negative bias)
    """

    def __init__(self, n_experts: int, update_alpha: float = 0.001):
        self.n_experts = n_experts
        self.update_alpha = update_alpha
        self.load_ema = np.zeros(n_experts)
        self.ema_decay = 0.99

    def update_biases(self, expert_counts: np.ndarray, biases: torch.Tensor) -> torch.Tensor:
        """Update biases after batch (NOT during backward pass)."""
        # Update load EMA
        self.load_ema = self.ema_decay * self.load_ema + (1 - self.ema_decay) * expert_counts

        # Compute bias adjustment
        mean_load = self.load_ema.mean()
        load_deviation = self.load_ema - mean_load

        # Overloaded experts get negative bias → fewer tokens routed
        bias_update = self.update_alpha * load_deviation

        # Apply update (detached from computation graph)
        with torch.no_grad():
            biases.data -= torch.from_numpy(bias_update).to(biases.device)

        return biases
```

---

## 5. Rotary Position Encoding (RoPE)

DeepSeek uses **extended RoPE** with multiple scaling strategies for 128K+ context.

### Scaling Strategies Across Backends

#### Rust: Comprehensive RoPE with All Scaling Types

From `rust-src/src/model/mla.rs:63-82`:

```rust
/// RoPE scaling types for long context
#[derive(Clone, Debug)]
pub enum RoPEScalingType {
    /// No scaling (original RoPE)
    None,
    /// Linear interpolation
    Linear { scale: f32 },
    /// NTK-aware scaling (modifies base frequency)
    NTKAware { alpha: f32 },
    /// YaRN: Yet another RoPE extensioN
    YaRN {
        scale: f32,
        original_max_seq_len: usize,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
    },
    /// Dynamic NTK (compute alpha based on sequence length)
    DynamicNTK { max_position_embeddings: usize },
}
```

From `rust-src/src/model/mla.rs:177-265`:

```rust
impl ExtendedRotaryPositionalEncoding {
    fn compute_inv_freq(config: &RoPEConfig, device: &Device) -> Result<Tensor> {
        let d_head = config.d_head;

        match &config.scaling_type {
            RoPEScalingType::None => {
                // Standard RoPE: θ_i = base^(-2i/d)
                let inv_freq: Vec<f32> = (0..d_head)
                    .step_by(2)
                    .map(|i| 1.0 / config.base.powf(i as f32 / d_head as f32))
                    .collect();
                Tensor::from_vec(inv_freq, (d_head / 2,), device)
            }

            RoPEScalingType::NTKAware { alpha } => {
                // NTK-aware: scale base frequency for extrapolation
                // new_base = base * alpha^(d/(d-2))
                let new_base = config.base * alpha.powf(d_head as f32 / (d_head as f32 - 2.0));
                let inv_freq: Vec<f32> = (0..d_head)
                    .step_by(2)
                    .map(|i| 1.0 / new_base.powf(i as f32 / d_head as f32))
                    .collect();
                Tensor::from_vec(inv_freq, (d_head / 2,), device)
            }

            RoPEScalingType::YaRN { scale, original_max_seq_len, beta_fast, beta_slow, .. } => {
                // YaRN: interpolate between scaled and unscaled based on frequency
                let half_dim = d_head / 2;
                let mut inv_freq = Vec::with_capacity(half_dim);

                for i in (0..d_head).step_by(2) {
                    let base_freq = 1.0 / config.base.powf(i as f32 / d_head as f32);
                    let wavelength = 2.0 * std::f32::consts::PI / base_freq;

                    // Compute interpolation factor
                    let low_freq_wavelen = (*original_max_seq_len as f32) / *beta_slow;
                    let high_freq_wavelen = (*original_max_seq_len as f32) / *beta_fast;

                    let gamma = if wavelength < high_freq_wavelen {
                        0.0  // High frequency: no interpolation
                    } else if wavelength > low_freq_wavelen {
                        1.0  // Low frequency: full interpolation
                    } else {
                        (wavelength - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
                    };

                    let scaled_freq = base_freq / scale;
                    let final_freq = (1.0 - gamma) * base_freq + gamma * scaled_freq;
                    inv_freq.push(final_freq);
                }

                Tensor::from_vec(inv_freq, (half_dim,), device)
            }
            // ... other scaling types
        }
    }
}
```

#### MLX: Extended RoPE for Apple Silicon

From `src/deepseek/mlx/attention.py:159-203`:

```python
class RoPEScalingType(Enum):
    """RoPE scaling methods for extended context."""
    NONE = "none"
    LINEAR = "linear"
    NTK_AWARE = "ntk_aware"
    DYNAMIC_NTK = "dynamic_ntk"
    YARN = "yarn"


@dataclass
class ExtendedRoPEConfig:
    """Configuration for Extended RoPE."""
    d_head: int = 64
    max_seq_len: int = 131072  # 128K context
    base: float = 10000.0
    scaling_type: RoPEScalingType = RoPEScalingType.NTK_AWARE
    ntk_alpha: float = 8.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 0.707
    original_max_seq_len: int = 4096

    @classmethod
    def for_128k(cls, d_head: int = 64) -> "ExtendedRoPEConfig":
        """Create config for 128K context with NTK-aware scaling."""
        return cls(
            d_head=d_head,
            max_seq_len=131072,
            scaling_type=RoPEScalingType.NTK_AWARE,
            ntk_alpha=8.0,
        )
```

### RoPE Mathematical Foundation

The rotation applied to query/key vectors:

```
For position m and dimension 2i:
    RoPE(x, m, 2i)   = x_{2i}   cos(mθ_i) - x_{2i+1} sin(mθ_i)
    RoPE(x, m, 2i+1) = x_{2i}   sin(mθ_i) + x_{2i+1} cos(mθ_i)

Where θ_i = base^(-2i/d)

For NTK-aware scaling with factor α:
    θ_i = (base × α^(d/(d-2)))^(-2i/d)
```

---

## 6. The Complete Transformer Block

### PyTorch: Full-Featured with Gradient Checkpointing

From `src/deepseek/torch/model/transformer.py:48-138`:

```python
class DeepSeekLayer(nn.Module):
    """
    Single transformer layer with MLA attention and optional MoE.
    Supports gradient checkpointing for memory-efficient training.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_latent: int,
        d_rope: int,
        d_hidden: int,
        num_experts: int,
        num_shared: int,
        num_routed: int,
        top_k: int,
        use_moe: bool = True,
        dropout: float = 0.0,
        attention_config: Optional[FlashAttentionConfig] = None,
        checkpoint_config: Optional[GradientCheckpointConfig] = None,
    ):
        super().__init__()

        # Attention with Flash Attention support
        self.attn = DeepSeekAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_latent=d_latent,
            d_rope=d_rope,
            dropout=dropout,
            attention_config=attention_config,
        )
        self.attn_norm = RMSNorm(d_model)

        # Feed Forward (MoE or Standard MLP)
        if use_moe:
            self.mlp = DeepSeekMoE(d_model, d_hidden, num_experts, num_shared, num_routed, top_k)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.SiLU(),
                nn.Linear(d_hidden, d_model)
            )
        self.mlp_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention block with residual (pre-norm)
        if self.checkpoint_config.enabled and self.training:
            attn_out = checkpoint(
                self._attention_forward, x, mask,
                use_reentrant=self.checkpoint_config.use_reentrant,
            )
        else:
            attn_out = self._attention_forward(x, mask)
        x = x + attn_out

        # MLP/MoE block with residual (pre-norm)
        if self.checkpoint_config.enabled and self.checkpoint_config.checkpoint_moe and self.training:
            mlp_out = checkpoint(self._mlp_forward, x, use_reentrant=False)
        else:
            mlp_out = self._mlp_forward(x)
        x = x + mlp_out

        return x
```

### RMSNorm: Simpler Than LayerNorm

All three backends implement RMSNorm for efficiency:

```python
# PyTorch
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms) * self.weight
```

```rust
// Rust - inline for performance
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (variance + eps as f64)?.rsqrt()?;
    (x * rms)? * weight
}
```

```python
# MLX - uses nn.RMSNorm built-in
self.norm = nn.RMSNorm(d_model, eps=1e-6)
```

---

## 7. Backend Comparison Matrix

| Feature | PyTorch | Rust/Candle | MLX |
|---------|---------|-------------|-----|
| **Primary Hardware** | CUDA/MPS/CPU | Metal/CUDA | Metal + ANE |
| **Training Speed** | 10.2 steps/s | 13.5 steps/s | 3.3 steps/s |
| **Flash Attention** | Yes (via torch) | Manual impl | Optimized |
| **Gradient Checkpointing** | Full support | Partial | Via mx.checkpoint |
| **Dynamic Shapes** | Full | Limited | Full |
| **Compilation** | torch.compile | Native | Lazy eval |
| **Distributed** | FSDP, DDP | Ring Attention | Single-device |
| **Quantization** | FP8 simulation | Native FP8 | Quantize API |
| **KV Cache** | Latent + Standard | Latent-first | Both |

### Performance Characteristics

```
┌─────────────────────────────────────────────────────────────┐
│                 Inference Latency (ms)                      │
│                 batch=4, seq=64, d=512                      │
├─────────────────┬───────────┬───────────┬───────────────────┤
│   Component     │   Rust    │  PyTorch  │       MLX         │
├─────────────────┼───────────┼───────────┼───────────────────┤
│   MLA Attention │   10.74   │    0.96   │      0.97         │
│   MoE Forward   │    4.97   │   49.85   │      2.53         │
│   Full Layer    │   15.71   │   50.81   │      3.50         │
└─────────────────┴───────────┴───────────┴───────────────────┘

Note: Rust numbers on Metal show kernel launch overhead;
      PyTorch MoE includes dynamic dispatch overhead;
      MLX benefits from unified memory on Apple Silicon.
```

---

## 8. When to Use Each Backend

### Choose PyTorch When:
- **Research & Experimentation**: Dynamic graphs, easy debugging
- **CUDA Hardware**: Best support for NVIDIA GPUs
- **Integration**: HuggingFace, existing PyTorch ecosystems
- **Training**: FSDP, DDP, gradient checkpointing
- **Flexibility**: Custom loss functions, novel architectures

### Choose Rust/Candle When:
- **Production Deployment**: Type safety, no runtime overhead
- **Metal Performance**: Best throughput on Apple Silicon
- **CUDA Production**: Native performance without Python GIL
- **Memory Efficiency**: Zero-copy operations, precise control
- **Embedding**: Edge devices, serverless functions

### Choose MLX When:
- **Apple Silicon Development**: Optimized for M1/M2/M3/M4
- **Unified Memory**: Large models without CPU-GPU transfers
- **Rapid Prototyping**: NumPy-like API, lazy evaluation
- **Local Inference**: Best local experience on Mac
- **Neural Engine**: Can leverage ANE for certain ops

---

## Next Steps

This architectural overview sets the foundation. Continue with:

1. **[Mixture of Experts Deep Dive](./techniques/mixture_of_experts.md)** - 256-expert routing details
2. **[Latent Attention Mechanics](./techniques/latent_attention.md)** - MLA implementation details
3. **[Production Scaling Guide](./03_production_scaling_guide.md)** - Deploying at scale
4. **[Build It Yourself](./04_step_by_step_creation.md)** - Step-by-step tutorial

---

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948)
- [Multi-Head Latent Attention Paper](https://arxiv.org/abs/2405.04434)
- [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071)
- [MegaBlocks: Efficient Sparse Training](https://arxiv.org/abs/2211.15841)
