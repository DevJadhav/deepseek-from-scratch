# Build It Yourself: DeepSeek Implementation Handbook

> **A Principal Developer Advocate's Tutorial for Building DeepSeek from Scratch**

This hands-on guide walks you through implementing DeepSeek's key components step-by-step across all three backends. By the end, you'll have working implementations of MLA, MoE, and the complete transformer.

---

## Table of Contents

1. [Prerequisites and Setup](#1-prerequisites-and-setup)
2. [Step 1: RMSNorm](#2-step-1-rmsnorm)
3. [Step 2: Rotary Position Encoding](#3-step-2-rotary-position-encoding)
4. [Step 3: Multi-Head Latent Attention](#4-step-3-multi-head-latent-attention)
5. [Step 4: SwiGLU Expert Network](#5-step-4-swiglu-expert-network)
6. [Step 5: Mixture of Experts Layer](#6-step-5-mixture-of-experts-layer)
7. [Step 6: Complete Transformer Block](#7-step-6-complete-transformer-block)
8. [Step 7: Full Model Assembly](#8-step-7-full-model-assembly)
9. [Step 8: Training Loop](#9-step-8-training-loop)
10. [Step 9: Inference and Generation](#10-step-9-inference-and-generation)

---

## 1. Prerequisites and Setup

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/DevJadhav/deepseek-from-scratch.git
cd DeepSeek-From-Scratch

# Python environment (PyTorch + MLX)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras

# Rust environment
cd rust-src
cargo build --release
cd ..

# Verify installations
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
uv run python -c "import mlx.core as mx; print(f'MLX: {mx.__version__}')"
cargo --version
```

### Project Structure for This Tutorial

```
tutorial/
├── pytorch/
│   ├── rmsnorm.py
│   ├── rope.py
│   ├── mla.py
│   ├── expert.py
│   ├── moe.py
│   ├── transformer.py
│   └── model.py
├── rust/
│   └── src/
│       ├── rmsnorm.rs
│       ├── rope.rs
│       └── ...
└── mlx/
    ├── rmsnorm.py
    ├── rope.py
    └── ...
```

---

## 2. Step 1: RMSNorm

RMSNorm is simpler and faster than LayerNorm:

### Mathematical Definition

```
RMSNorm(x) = x / RMS(x) × γ

where RMS(x) = √(mean(x²) + ε)
```

### PyTorch Implementation

```python
# tutorial/pytorch/rmsnorm.py
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Compared to LayerNorm:
    - No mean subtraction (faster)
    - Only scale parameter (no bias)
    - Similar quality in practice
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x * rms) * self.weight


# Test
if __name__ == "__main__":
    norm = RMSNorm(512)
    x = torch.randn(2, 10, 512)
    y = norm(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    print(f"Output mean: {y.mean():.4f}, std: {y.std():.4f}")
```

### Rust Implementation

```rust
// tutorial/rust/src/rmsnorm.rs
use candle_core::{Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((dim,), "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute RMS: sqrt(mean(x^2) + eps)
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let rms = (variance + self.eps)?.sqrt()?;

        // Normalize and scale
        let normalized = x.broadcast_div(&rms)?;
        normalized.broadcast_mul(&self.weight)
    }
}
```

### MLX Implementation

```python
# tutorial/mlx/rmsnorm.py
import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm for MLX."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # MLX operations are lazy - evaluated on demand
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x * rms) * self.weight
```

---

## 3. Step 2: Rotary Position Encoding

RoPE encodes position through rotation:

### PyTorch Implementation

```python
# tutorial/pytorch/rope.py
import torch
import torch.nn as nn
import math


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Encoding (RoPE).

    Key insight: encode position by rotating query/key vectors.
    The inner product Q·K then naturally encodes relative position.
    """

    def __init__(self, d_head: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        self.base = base

        # Compute inverse frequencies: θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin for all positions
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, max_seq_len: int):
        """Precompute cos and sin for efficiency."""
        positions = torch.arange(max_seq_len)
        # freqs: [max_seq_len, d_head/2]
        freqs = torch.outer(positions, self.inv_freq)
        # cos/sin: [max_seq_len, d_head/2]
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer('cos_cache', cos, persistent=False)
        self.register_buffer('sin_cache', sin, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x: [batch, heads, seq_len, d_head]
            offset: position offset (for KV cache)

        Returns:
            Rotated tensor of same shape
        """
        batch, heads, seq_len, d_head = x.shape

        # Get cos/sin for this sequence
        cos = self.cos_cache[offset:offset + seq_len]  # [seq_len, d_head/2]
        sin = self.sin_cache[offset:offset + seq_len]

        # Reshape for broadcasting: [1, 1, seq_len, d_head/2]
        cos = cos.view(1, 1, seq_len, -1)
        sin = sin.view(1, 1, seq_len, -1)

        # Split into even and odd dimensions
        x_even = x[..., 0::2]  # [batch, heads, seq_len, d_head/2]
        x_odd = x[..., 1::2]

        # Apply rotation:
        # x'_even = x_even * cos - x_odd * sin
        # x'_odd  = x_even * sin + x_odd * cos
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Interleave back
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        return x_rotated.flatten(-2)  # [batch, heads, seq_len, d_head]


# Test
if __name__ == "__main__":
    rope = RotaryPositionalEncoding(d_head=64)
    x = torch.randn(2, 8, 32, 64)  # [batch, heads, seq, d_head]
    y = rope(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

    # Verify position encoding works
    y_offset = rope(x[:, :, :16], offset=16)
    print(f"With offset: {y_offset.shape}")
```

### Rust Implementation

```rust
// tutorial/rust/src/rope.rs
use candle_core::{Device, Result, Tensor, IndexOp};

pub struct RotaryPositionalEncoding {
    inv_freq: Tensor,
    cos_cache: Tensor,
    sin_cache: Tensor,
}

impl RotaryPositionalEncoding {
    pub fn new(d_head: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        // Compute inverse frequencies
        let inv_freq: Vec<f32> = (0..d_head)
            .step_by(2)
            .map(|i| 1.0 / 10000f32.powf(i as f32 / d_head as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (d_head / 2,), device)?;

        // Precompute cos/sin
        let positions = Tensor::arange(0f32, max_seq_len as f32, device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;

        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;

        Ok(Self { inv_freq, cos_cache, sin_cache })
    }

    pub fn forward(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (batch, num_heads, seq_len, d_head) = x.dims4()?;

        // Get cached cos/sin
        let cos = self.cos_cache.narrow(0, offset, seq_len)?
            .reshape((1, 1, seq_len, d_head / 2))?;
        let sin = self.sin_cache.narrow(0, offset, seq_len)?
            .reshape((1, 1, seq_len, d_head / 2))?;

        // Reshape x to [batch, heads, seq, d_head/2, 2]
        let x_reshaped = x.reshape((batch, num_heads, seq_len, d_head / 2, 2))?;
        let x_even = x_reshaped.i((.., .., .., .., 0))?;
        let x_odd = x_reshaped.i((.., .., .., .., 1))?;

        // Broadcast cos/sin
        let cos = cos.broadcast_as((batch, num_heads, seq_len, d_head / 2))?;
        let sin = sin.broadcast_as((batch, num_heads, seq_len, d_head / 2))?;

        // Apply rotation
        let out_even = (x_even.mul(&cos)? - x_odd.mul(&sin)?)?;
        let out_odd = (x_even.mul(&sin)? + x_odd.mul(&cos)?)?;

        // Stack and flatten
        let out = Tensor::stack(&[&out_even, &out_odd], 4)?;
        out.flatten_from(3)
    }
}
```

---

## 4. Step 3: Multi-Head Latent Attention

MLA compresses KV cache through a latent bottleneck:

### PyTorch Implementation

```python
# tutorial/pytorch/mla.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from rope import RotaryPositionalEncoding


class KVCache:
    """Simple KV cache for generation."""

    def __init__(self, batch_size: int, max_seq: int, n_heads: int, d_head: int, device):
        self.k = torch.zeros(batch_size, n_heads, max_seq, d_head, device=device)
        self.v = torch.zeros(batch_size, n_heads, max_seq, d_head, device=device)
        self.seq_len = 0

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        new_seq = k.shape[2]
        end = self.seq_len + new_seq
        self.k[:, :, self.seq_len:end] = k
        self.v[:, :, self.seq_len:end] = v
        self.seq_len = end
        return self.k[:, :, :self.seq_len], self.v[:, :, :self.seq_len]


class LatentKVCache:
    """Memory-efficient latent cache for MLA (~14× smaller)."""

    def __init__(self, batch_size: int, max_seq: int, d_latent: int, device):
        self.cache = torch.zeros(batch_size, max_seq, d_latent, device=device)
        self.seq_len = 0

    def update(self, c_kv: torch.Tensor) -> torch.Tensor:
        new_seq = c_kv.shape[1]
        end = self.seq_len + new_seq
        self.cache[:, self.seq_len:end] = c_kv
        self.seq_len = end
        return self.cache[:, :self.seq_len]


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA).

    Key innovation: compress K, V to low-dimensional latent space.

    Memory comparison:
    - Standard: 2 × n_heads × d_head × seq_len (K and V)
    - MLA: d_latent × seq_len

    Compression ratio: (2 × n_heads × d_head) / d_latent ≈ 14×
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_latent: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_latent = d_latent
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Query projection (full dimension)
        self.w_q = nn.Linear(d_model, d_model, bias=False)

        # Key-Value compression path
        self.w_down_kv = nn.Linear(d_model, d_latent, bias=False)  # Compress
        self.w_up_k = nn.Linear(d_latent, d_model, bias=False)     # Expand K
        self.w_up_v = nn.Linear(d_latent, d_model, bias=False)     # Expand V

        # Output
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[LatentKVCache] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional latent cache.

        Args:
            x: [batch, seq_len, d_model]
            mask: attention mask
            cache: latent KV cache for generation
        """
        batch, seq_len, _ = x.shape

        # Query: full projection
        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.d_head)
        q = q.transpose(1, 2)  # [batch, heads, seq, d_head]

        # Key-Value: compress to latent
        c_kv = self.w_down_kv(x)  # [batch, seq, d_latent]

        # Update cache if provided
        if cache is not None:
            c_kv_full = cache.update(c_kv)
            total_seq = c_kv_full.shape[1]
        else:
            c_kv_full = c_kv
            total_seq = seq_len

        # Expand from latent
        k = self.w_up_k(c_kv_full).view(batch, total_seq, self.n_heads, self.d_head)
        v = self.w_up_v(c_kv_full).view(batch, total_seq, self.n_heads, self.d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        if mask is None:
            mask = torch.triu(
                torch.ones(seq_len, total_seq, device=x.device) * float('-inf'),
                diagonal=total_seq - seq_len + 1
            )
        scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply to values
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(out)


# Test
if __name__ == "__main__":
    # Compare memory usage
    batch, seq, d_model, n_heads, d_latent = 4, 1024, 512, 8, 64
    d_head = d_model // n_heads

    standard_kv_memory = 2 * batch * n_heads * seq * d_head * 4  # FP32
    latent_memory = batch * seq * d_latent * 4

    print(f"Standard KV cache: {standard_kv_memory / 1e6:.2f} MB")
    print(f"Latent cache: {latent_memory / 1e6:.2f} MB")
    print(f"Compression ratio: {standard_kv_memory / latent_memory:.1f}×")

    # Test forward pass
    mla = MultiHeadLatentAttention(d_model, n_heads, d_latent)
    x = torch.randn(batch, seq, d_model)
    y = mla(x)
    print(f"\nInput: {x.shape}, Output: {y.shape}")
```

---

## 5. Step 4: SwiGLU Expert Network

Each MoE expert uses SwiGLU activation:

### PyTorch Implementation

```python
# tutorial/pytorch/expert.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """
    Expert network with SwiGLU activation.

    SwiGLU(x) = (x @ W_gate) × SiLU(x @ W_up) @ W_down

    Benefits over ReLU:
    - Smoother gradients (no dead neurons)
    - Gating provides learned selectivity
    - Better training stability at scale
    """

    def __init__(self, d_model: int, d_expert: int):
        super().__init__()
        # Gate and up projections (can be fused)
        self.w_gate = nn.Linear(d_model, d_expert, bias=False)
        self.w_up = nn.Linear(d_model, d_expert, bias=False)
        # Down projection
        self.w_down = nn.Linear(d_expert, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate * silu(up)
        gate = self.w_gate(x)
        up = self.w_up(x)
        hidden = F.silu(gate) * up
        return self.w_down(hidden)


class FusedSwiGLUExpert(nn.Module):
    """
    Memory-optimized SwiGLU with fused gate+up projection.

    Reduces memory by computing gate and up in single matmul.
    """

    def __init__(self, d_model: int, d_expert: int):
        super().__init__()
        # Fused gate+up projection
        self.w_gate_up = nn.Linear(d_model, 2 * d_expert, bias=False)
        self.w_down = nn.Linear(d_expert, d_model, bias=False)
        self.d_expert = d_expert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.w_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        return self.w_down(hidden)


# Test
if __name__ == "__main__":
    expert = SwiGLUExpert(512, 2048)
    x = torch.randn(4, 32, 512)
    y = expert(x)
    print(f"Expert output shape: {y.shape}")

    # Parameter count
    params = sum(p.numel() for p in expert.parameters())
    print(f"Parameters: {params:,}")
```

---

## 6. Step 5: Mixture of Experts Layer

The MoE layer with top-k routing:

### PyTorch Implementation

```python
# tutorial/pytorch/moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from expert import SwiGLUExpert
import numpy as np


class RouterBiasController:
    """Auxiliary-loss-free load balancing via bias updates."""

    def __init__(self, n_experts: int, alpha: float = 0.001):
        self.n_experts = n_experts
        self.alpha = alpha
        self.load_ema = np.zeros(n_experts)
        self.steps = 0

    def update(self, counts: np.ndarray, biases: torch.Tensor):
        """Update biases based on expert load (call AFTER backward)."""
        if counts.sum() == 0:
            return

        load = counts / counts.sum()

        # EMA update
        decay = 0.99
        if self.steps == 0:
            self.load_ema = load
        else:
            self.load_ema = decay * self.load_ema + (1 - decay) * load
        self.steps += 1

        # Compute bias adjustment
        target = 1.0 / self.n_experts
        deviation = self.load_ema - target

        # Apply update (overloaded experts get negative bias)
        with torch.no_grad():
            biases.data -= torch.from_numpy(self.alpha * deviation).to(biases.device).float()


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer with:
    - Top-k routing
    - Shared experts (always active)
    - Auxiliary-loss-free load balancing
    """

    def __init__(
        self,
        d_model: int,
        d_expert: int,
        n_routed_experts: int = 8,
        n_shared_experts: int = 1,
        top_k: int = 2,
        bias_alpha: float = 0.001,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.top_k = top_k

        # Router
        self.router = nn.Linear(d_model, n_routed_experts, bias=False)

        # Learnable biases for load balancing (NOT gradient-updated)
        self.expert_biases = nn.Parameter(
            torch.zeros(n_routed_experts), requires_grad=False
        )

        # Routed experts
        self.routed_experts = nn.ModuleList([
            SwiGLUExpert(d_model, d_expert)
            for _ in range(n_routed_experts)
        ])

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            SwiGLUExpert(d_model, d_expert)
            for _ in range(n_shared_experts)
        ])

        # Bias controller
        self.bias_controller = RouterBiasController(n_routed_experts, bias_alpha)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with separate routing and gating.

        Key: Biased softmax for routing, original softmax for gating.
        This keeps gradients clean while balancing load.
        """
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        n_tokens = x_flat.shape[0]

        # Router logits
        router_logits = self.router(x_flat)

        # Routing: use biased logits for expert selection
        routing_logits = router_logits + self.expert_biases
        routing_probs = F.softmax(routing_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)

        # Gating: use ORIGINAL logits for weights (clean gradients)
        gating_probs = F.softmax(router_logits, dim=-1)
        gate_weights = torch.gather(gating_probs, 1, top_k_indices)
        gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Compute expert outputs
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            indices = top_k_indices[:, k]
            weights = gate_weights[:, k:k+1]

            for e in range(self.n_routed_experts):
                mask = (indices == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.routed_experts[e](expert_input)
                    output[mask] += weights[mask] * expert_output

        # Add shared expert contributions
        for shared in self.shared_experts:
            output = output + shared(x_flat)

        # Track expert usage for bias updates
        expert_counts = torch.bincount(
            top_k_indices.view(-1),
            minlength=self.n_routed_experts
        ).cpu().numpy()

        return output.view(batch, seq_len, d_model), {
            'expert_indices': top_k_indices,
            'expert_counts': expert_counts,
        }

    def update_biases(self, expert_counts: np.ndarray):
        """Call after optimizer.step()."""
        self.bias_controller.update(expert_counts, self.expert_biases)


# Test
if __name__ == "__main__":
    moe = MixtureOfExperts(
        d_model=512,
        d_expert=1024,
        n_routed_experts=8,
        n_shared_experts=1,
        top_k=2,
    )

    x = torch.randn(2, 32, 512)
    y, info = moe(x)
    print(f"Output shape: {y.shape}")
    print(f"Expert counts: {info['expert_counts']}")

    # Simulate training step
    moe.update_biases(info['expert_counts'])
    print(f"Updated biases: {moe.expert_biases}")
```

---

## 7. Step 6: Complete Transformer Block

Combine attention and MoE into a full block:

### PyTorch Implementation

```python
# tutorial/pytorch/transformer.py
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from rmsnorm import RMSNorm
from mla import MultiHeadLatentAttention, LatentKVCache
from moe import MixtureOfExperts


class DeepSeekBlock(nn.Module):
    """
    Single DeepSeek transformer block.

    Architecture (Pre-norm):
    x → RMSNorm → MLA → + → RMSNorm → MoE → +
    └─────────────────────┘ └────────────────┘
         Residual 1              Residual 2
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_latent: int,
        d_expert: int,
        n_routed_experts: int = 8,
        n_shared_experts: int = 1,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Attention
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadLatentAttention(d_model, n_heads, d_latent, dropout)

        # MoE
        self.moe_norm = RMSNorm(d_model)
        self.moe = MixtureOfExperts(
            d_model, d_expert, n_routed_experts, n_shared_experts, top_k
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[LatentKVCache] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        # Attention block with residual
        h = x + self.attn(self.attn_norm(x), mask, cache)

        # MoE block with residual
        moe_out, moe_info = self.moe(self.moe_norm(h))
        h = h + moe_out

        return h, moe_info


# Test
if __name__ == "__main__":
    block = DeepSeekBlock(
        d_model=512,
        n_heads=8,
        d_latent=64,
        d_expert=1024,
        n_routed_experts=8,
    )

    x = torch.randn(2, 32, 512)
    y, info = block(x)
    print(f"Block output: {y.shape}")
```

---

## 8. Step 7: Full Model Assembly

Put everything together:

### PyTorch Implementation

```python
# tutorial/pytorch/model.py
import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass
from rmsnorm import RMSNorm
from transformer import DeepSeekBlock
from mla import LatentKVCache


@dataclass
class DeepSeekConfig:
    """Model configuration."""
    vocab_size: int = 32000
    n_layers: int = 12
    d_model: int = 512
    n_heads: int = 8
    d_latent: int = 64      # MLA latent dimension
    d_expert: int = 1024    # Expert hidden size
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    top_k: int = 2
    dropout: float = 0.0
    max_seq_len: int = 4096


class DeepSeekModel(nn.Module):
    """
    Complete DeepSeek model.

    Architecture:
    - Token embedding
    - N × DeepSeek blocks (MLA + MoE)
    - Final RMSNorm
    - LM head (tied with embedding)
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            DeepSeekBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_latent=config.d_latent,
                d_expert=config.d_expert,
                n_routed_experts=config.n_routed_experts,
                n_shared_experts=config.n_shared_experts,
                top_k=config.top_k,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # Output
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embed.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        caches: Optional[List[LatentKVCache]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            mask: attention mask
            caches: list of LatentKVCache per layer

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embed
        x = self.embed(input_ids)

        # Process through layers
        all_expert_counts = []
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches else None
            x, info = layer(x, mask, cache)
            all_expert_counts.append(info['expert_counts'])

        # Output
        x = self.norm(x)
        logits = self.head(x)

        return logits, all_expert_counts

    def update_all_biases(self, all_expert_counts: List):
        """Update biases for all MoE layers."""
        for layer, counts in zip(self.layers, all_expert_counts):
            layer.moe.update_biases(counts)


# Test
if __name__ == "__main__":
    config = DeepSeekConfig(
        vocab_size=32000,
        n_layers=4,
        d_model=256,
        n_heads=4,
        d_latent=32,
        d_expert=512,
        n_routed_experts=4,
    )

    model = DeepSeekModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    logits, counts = model(input_ids)
    print(f"Logits shape: {logits.shape}")

    # Update biases
    model.update_all_biases(counts)
```

---

## 9. Step 8: Training Loop

Complete training with loss and optimization:

### PyTorch Implementation

```python
# tutorial/pytorch/train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import DeepSeekModel, DeepSeekConfig
from tqdm import tqdm


def train_step(
    model: DeepSeekModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    """Single training step with mixed precision."""
    model.train()

    input_ids = batch['input_ids']
    labels = batch['labels']

    # Forward with mixed precision
    with torch.cuda.amp.autocast():
        logits, expert_counts = model(input_ids)

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

    # Backward
    optimizer.zero_grad()
    scaler.scale(loss).backward()

    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()

    # Update MoE biases AFTER optimizer step
    model.update_all_biases(expert_counts)

    return loss.item()


def train_epoch(
    model: DeepSeekModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        loss = train_step(model, batch, optimizer, scaler)
        total_loss += loss
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss:.4f}'})

    return total_loss / num_batches


def main():
    # Config
    config = DeepSeekConfig(
        vocab_size=32000,
        n_layers=6,
        d_model=512,
        n_heads=8,
        d_latent=64,
        d_expert=1024,
        n_routed_experts=8,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    model = DeepSeekModel(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Dummy data (replace with real dataloader)
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000, seq_len=64, vocab_size=32000):
            self.size = size
            self.seq_len = seq_len
            self.vocab_size = vocab_size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
            labels = torch.randint(0, self.vocab_size, (self.seq_len,))
            return {'input_ids': input_ids, 'labels': labels}

    dataloader = DataLoader(DummyDataset(), batch_size=8, shuffle=True)

    # Training loop
    for epoch in range(3):
        avg_loss = train_epoch(model, dataloader, optimizer, scaler, device)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
```

---

## 10. Step 9: Inference and Generation

Text generation with KV caching:

### PyTorch Implementation

```python
# tutorial/pytorch/generate.py
import torch
import torch.nn.functional as F
from model import DeepSeekModel, DeepSeekConfig
from mla import LatentKVCache
from typing import List


def generate(
    model: DeepSeekModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    Generate text with KV caching for efficiency.

    Uses MLA's latent cache (~14× smaller than standard KV cache).
    """
    model.eval()
    device = next(model.parameters()).device
    batch_size = prompt_ids.shape[0]

    # Initialize latent caches for each layer
    caches = [
        LatentKVCache(
            batch_size=batch_size,
            max_seq=model.config.max_seq_len,
            d_latent=model.config.d_latent,
            device=device,
        )
        for _ in range(model.config.n_layers)
    ]

    # Process prompt (prefill)
    with torch.no_grad():
        logits, _ = model(prompt_ids, caches=caches)
        next_token_logits = logits[:, -1, :]

    generated = prompt_ids.clone()

    # Generate tokens one at a time
    for _ in range(max_new_tokens):
        # Sample next token
        if temperature > 0:
            probs = F.softmax(next_token_logits / temperature, dim=-1)

            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

            # Sample
            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token = torch.gather(sorted_indices, 1, next_token)
        else:
            # Greedy
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        # Append to generated sequence
        generated = torch.cat([generated, next_token], dim=1)

        # Forward pass with cache (only new token)
        with torch.no_grad():
            logits, _ = model(next_token, caches=caches)
            next_token_logits = logits[:, -1, :]

    return generated


# Test
if __name__ == "__main__":
    config = DeepSeekConfig(
        vocab_size=32000,
        n_layers=4,
        d_model=256,
        n_heads=4,
        d_latent=32,
        d_expert=512,
        n_routed_experts=4,
    )

    model = DeepSeekModel(config)
    model.eval()

    # Dummy prompt
    prompt = torch.randint(0, config.vocab_size, (1, 10))

    print("Generating...")
    output = generate(model, prompt, max_new_tokens=50, temperature=0.8)
    print(f"Generated shape: {output.shape}")
    print(f"Generated tokens: {output[0].tolist()[:20]}...")
```

---

## Summary

Congratulations! You've implemented DeepSeek from scratch:

1. **RMSNorm**: Efficient normalization
2. **RoPE**: Position encoding through rotation
3. **MLA**: 14× KV cache compression via latent space
4. **SwiGLU Expert**: Smooth gating in FFN
5. **MoE**: Top-k routing with bias-based load balancing
6. **Transformer Block**: Attention + MoE with residuals
7. **Full Model**: Embeddings + blocks + head
8. **Training**: Mixed precision with bias updates
9. **Generation**: Efficient inference with latent cache

### Next Steps

- Add Flash Attention integration
- Implement distributed training (FSDP)
- Add speculative decoding for faster generation
- Scale to 256 experts with hierarchical routing

---

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Architecture Overview](./01_deepseek_architecture_from_scratch.md)
- [MoE Deep Dive](./techniques/mixture_of_experts.md)
- [Production Guide](./03_production_scaling_guide.md)
