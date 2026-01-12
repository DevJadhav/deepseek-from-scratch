# FP8 Quantization

## Overview

FP8 (8-bit floating point) quantization reduces model size and increases throughput by representing weights and activations with only 8 bits instead of 16 or 32 bits. DeepSeek uses a **tile-based quantization** approach for FP8 Linear layers.

**Key Features:**
- **2-4x memory reduction** compared to FP16/BF16
- **Faster matrix multiplication** on supported hardware
- **Minimal accuracy loss** with proper scaling

**Relevant Standards:**
- IEEE FP8: E4M3 (range-focused) and E5M2 (precision-focused)
- Nvidia H100/H200 native FP8 support
- Apple M3+ supports FP8 operations

**Key Papers:**
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) (Micikevicius et al., 2022)
- [8-bit Numerical Formats for Deep Neural Networks](https://arxiv.org/abs/2206.02915) (Noune et al., 2022)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) (Dettmers et al., 2022)

---

## When to Use FP8

### ✅ Use FP8 When:
- **Inference at scale** - Reduces memory bandwidth
- **Large batch sizes** - Better hardware utilization
- **Memory constrained** - 2x model size reduction vs FP16
- **Hardware support available** - H100, M3+, TPU v5

### ❌ Don't Use FP8 When:
- **Training from scratch** - Use higher precision
- **Small models** - Overhead not justified
- **High precision required** - Scientific computing
- **Legacy hardware** - No FP8 acceleration

### When to Prefer Over Other Formats:
| Format | Bits | Use Case |
|--------|------|----------|
| FP32 | 32 | Reference, training |
| FP16/BF16 | 16 | Standard training/inference |
| **FP8** | **8** | **High-throughput inference** |
| INT8 | 8 | CPU inference, PTQ |
| INT4 | 4 | Maximum compression |

---

## FP8 Number Formats

### E4M3 Format (Range-Focused)

```
┌───┬────────────┬─────────┐
│ S │  Exponent  │ Mantissa│
│ 1 │     4      │    3    │
└───┴────────────┴─────────┘
```

- **Range:** ±448 (larger than E5M2)
- **Precision:** 3 mantissa bits
- **Best for:** Weights (need larger range)

### E5M2 Format (Precision-Focused)

```
┌───┬────────────┬─────────┐
│ S │  Exponent  │ Mantissa│
│ 1 │     5      │    2    │
└───┴────────────┴─────────┘
```

- **Range:** ±57344 (larger exponent)
- **Precision:** 2 mantissa bits
- **Best for:** Activations (gradients need precision)

### Value Encoding

For E4M3:
$$\text{value} = (-1)^S \times 2^{E-7} \times (1 + M/8)$$

Where:
- $S$: Sign bit (0 or 1)
- $E$: Exponent value (0-15, bias of 7)
- $M$: Mantissa value (0-7)

### Special Values

| Type | E4M3 | E5M2 |
|------|------|------|
| Zero | S.0000.000 | S.00000.00 |
| Max | 0.1111.110 = 448 | 0.11111.11 = 57344 |
| NaN | S.1111.111 | S.11111.11 |

---

## Tile-Based Quantization

### The Problem

Direct FP8 quantization loses too much precision because:
- Single scale factor for entire tensor
- Outliers dominate scaling
- Fine-grained features lost

### The Solution: Tile-Based Scaling

Divide tensor into tiles, compute per-tile scale factors:

```
Weight Matrix W: (M, K)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
    ┌────────┐  ┌────────┐  ┌────────┐
    │ Tile 1 │  │ Tile 2 │  │ Tile 3 │
    │ scale₁ │  │ scale₂ │  │ scale₃ │
    └────────┘  └────────┘  └────────┘
        │           │           │
        ▼           ▼           ▼
    ┌────────┐  ┌────────┐  ┌────────┐
    │ Tile 4 │  │ Tile 5 │  │ Tile 6 │
    │ scale₄ │  │ scale₅ │  │ scale₆ │
    └────────┘  └────────┘  └────────┘
```

### Tile Dimensions

Typical tile sizes:
- **Row tiles:** $(1, K)$ - One scale per row
- **Column tiles:** $(M, 1)$ - One scale per column
- **Block tiles:** $(T_M, T_K)$ - 2D blocking (e.g., 128×128)

DeepSeek uses **1D row-wise tiling** for simplicity and efficiency.

---

## Mathematical Foundation

### Quantization Formula

For each tile with values $\mathbf{x}$:

$$\text{scale} = \frac{\max(|\mathbf{x}|)}{\text{FP8\_MAX}}$$

$$\mathbf{x}_{\text{quant}} = \text{round}\left(\frac{\mathbf{x}}{\text{scale}}\right)$$

Where $\text{FP8\_MAX}$ is 448 for E4M3 or 57344 for E5M2.

### Dequantization

$$\mathbf{x}_{\text{dequant}} = \mathbf{x}_{\text{quant}} \times \text{scale}$$

### Matrix Multiplication with Scales

For $Y = XW^T$ with tiled FP8:

$$Y_{ij} = \sum_k X_{\text{quant},ik} \times W_{\text{quant},jk} \times s_X^{(i)} \times s_W^{(j)}$$

Where $s_X^{(i)}$ and $s_W^{(j)}$ are per-tile scale factors.

---

## Implementation

### Rust Implementation

```rust
use half::f16;

pub struct FP8Linear {
    // FP8 quantized weights: (out_features, in_features)
    weights_fp8: Vec<u8>,  // E4M3 format
    
    // Per-row scale factors: (out_features,)
    weight_scales: Vec<f32>,
    
    // Bias in FP32
    bias: Option<Vec<f32>>,
    
    // Dimensions
    in_features: usize,
    out_features: usize,
    tile_size: usize,
}

impl FP8Linear {
    pub fn from_fp32(
        weights: &Tensor,  // (out_features, in_features)
        bias: Option<&Tensor>,
        tile_size: usize,
    ) -> Result<Self> {
        let (out_features, in_features) = weights.dims2()?;
        let weights_f32 = weights.to_vec2::<f32>()?;
        
        let mut weights_fp8 = Vec::with_capacity(out_features * in_features);
        let mut weight_scales = Vec::with_capacity(out_features);
        
        // Quantize each row (tile)
        for row in &weights_f32 {
            let (scale, quantized) = quantize_to_fp8_e4m3(row);
            weight_scales.push(scale);
            weights_fp8.extend(quantized);
        }
        
        let bias = bias.map(|b| b.to_vec1::<f32>()).transpose()?;
        
        Ok(Self {
            weights_fp8,
            weight_scales,
            bias,
            in_features,
            out_features,
            tile_size,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (batch, in_features)
        let (batch, _) = x.dims2()?;
        let x_f32 = x.to_vec2::<f32>()?;
        
        // Quantize activations per-row
        let mut x_fp8 = Vec::new();
        let mut x_scales = Vec::new();
        for row in &x_f32 {
            let (scale, quantized) = quantize_to_fp8_e4m3(row);
            x_scales.push(scale);
            x_fp8.push(quantized);
        }
        
        // Matrix multiplication with dequantization
        let mut output = vec![vec![0f32; self.out_features]; batch];
        
        for b in 0..batch {
            for o in 0..self.out_features {
                let mut sum = 0f32;
                
                for i in 0..self.in_features {
                    let x_val = fp8_e4m3_to_f32(x_fp8[b][i]);
                    let w_val = fp8_e4m3_to_f32(
                        self.weights_fp8[o * self.in_features + i]
                    );
                    sum += x_val * w_val;
                }
                
                // Apply scales
                output[b][o] = sum * x_scales[b] * self.weight_scales[o];
                
                // Add bias
                if let Some(ref bias) = self.bias {
                    output[b][o] += bias[o];
                }
            }
        }
        
        Tensor::new(output, x.device())
    }
}

// === FP8 E4M3 Conversion ===

fn quantize_to_fp8_e4m3(values: &[f32]) -> (f32, Vec<u8>) {
    const FP8_E4M3_MAX: f32 = 448.0;
    
    // Find max absolute value
    let max_val = values.iter()
        .map(|v| v.abs())
        .fold(0f32, f32::max);
    
    // Compute scale
    let scale = if max_val == 0.0 { 1.0 } else { max_val / FP8_E4M3_MAX };
    
    // Quantize
    let quantized = values.iter()
        .map(|v| f32_to_fp8_e4m3(v / scale))
        .collect();
    
    (scale, quantized)
}

fn f32_to_fp8_e4m3(x: f32) -> u8 {
    if x == 0.0 {
        return 0;
    }
    
    let sign = if x < 0.0 { 1u8 } else { 0u8 };
    let abs_x = x.abs();
    
    // Clamp to FP8 range
    let clamped = abs_x.min(448.0).max(2f32.powi(-9));
    
    // Extract exponent and mantissa
    let log2_val = clamped.log2();
    let exponent = (log2_val.floor() as i32 + 7).clamp(0, 15) as u8;
    let mantissa = ((clamped / 2f32.powi(exponent as i32 - 7) - 1.0) * 8.0)
        .round()
        .clamp(0.0, 7.0) as u8;
    
    (sign << 7) | (exponent << 3) | mantissa
}

fn fp8_e4m3_to_f32(fp8: u8) -> f32 {
    let sign = (fp8 >> 7) & 1;
    let exponent = (fp8 >> 3) & 0xF;
    let mantissa = fp8 & 0x7;
    
    if exponent == 0 && mantissa == 0 {
        return 0.0;
    }
    
    let value = 2f32.powi(exponent as i32 - 7) * (1.0 + mantissa as f32 / 8.0);
    if sign == 1 { -value } else { value }
}
```

### Python Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FP8Linear(nn.Module):
    """
    FP8 Linear layer with tile-based quantization.
    
    Uses E4M3 format for weights (better range) and E5M2 for activations
    (better for gradients during training).
    """
    
    FP8_E4M3_MAX = 448.0
    FP8_E5M2_MAX = 57344.0
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tile_size: int = 1,  # Row-wise by default
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tile_size = tile_size
        
        # Store quantized weights as uint8
        self.register_buffer(
            'weights_fp8', 
            torch.zeros(out_features, in_features, dtype=torch.uint8)
        )
        
        # Per-row scale factors
        self.register_buffer(
            'weight_scales',
            torch.ones(out_features, dtype=torch.float32)
        )
        
        # Bias in full precision
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # For STE (Straight-Through Estimator) during training
        self.register_buffer('weights_fp32', None)
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, tile_size: int = 1) -> 'FP8Linear':
        """Convert a regular Linear layer to FP8Linear."""
        fp8_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            tile_size=tile_size,
        )
        
        # Quantize weights
        fp8_linear.quantize_weights(linear.weight.data)
        
        if linear.bias is not None:
            fp8_linear.bias.data.copy_(linear.bias.data)
        
        return fp8_linear
    
    def quantize_weights(self, weights: torch.Tensor):
        """Quantize weights to FP8 E4M3 format with per-row scaling."""
        # weights: (out_features, in_features)
        
        # Keep FP32 copy for STE
        self.weights_fp32 = weights.clone()
        
        # Compute per-row scales
        row_max = weights.abs().max(dim=1).values  # (out_features,)
        self.weight_scales = row_max / self.FP8_E4M3_MAX
        self.weight_scales = torch.where(
            self.weight_scales == 0,
            torch.ones_like(self.weight_scales),
            self.weight_scales
        )
        
        # Scale and quantize
        scaled_weights = weights / self.weight_scales.unsqueeze(1)
        self.weights_fp8 = self._float_to_fp8_e4m3(scaled_weights)
    
    def _float_to_fp8_e4m3(self, x: torch.Tensor) -> torch.Tensor:
        """Convert float tensor to FP8 E4M3 representation."""
        # Clamp to valid range
        x = x.clamp(-self.FP8_E4M3_MAX, self.FP8_E4M3_MAX)
        
        # Get sign
        sign = (x < 0).to(torch.uint8)
        abs_x = x.abs()
        
        # Handle zeros
        is_zero = abs_x == 0
        
        # Compute exponent (biased)
        log2_x = torch.log2(abs_x.clamp(min=2**-9))
        exponent = (log2_x.floor() + 7).clamp(0, 15).to(torch.uint8)
        
        # Compute mantissa
        mantissa = ((abs_x / (2.0 ** (exponent.float() - 7)) - 1.0) * 8)
        mantissa = mantissa.round().clamp(0, 7).to(torch.uint8)
        
        # Combine
        fp8 = (sign << 7) | (exponent << 3) | mantissa
        fp8 = torch.where(is_zero, torch.zeros_like(fp8), fp8)
        
        return fp8
    
    def _fp8_e4m3_to_float(self, fp8: torch.Tensor) -> torch.Tensor:
        """Convert FP8 E4M3 representation back to float."""
        sign = ((fp8 >> 7) & 1).float()
        exponent = ((fp8 >> 3) & 0xF).float()
        mantissa = (fp8 & 0x7).float()
        
        # Compute value
        value = (2.0 ** (exponent - 7)) * (1.0 + mantissa / 8.0)
        value = torch.where(
            (exponent == 0) & (mantissa == 0),
            torch.zeros_like(value),
            value
        )
        
        # Apply sign
        value = torch.where(sign == 1, -value, value)
        
        return value
    
    def dequantize_weights(self) -> torch.Tensor:
        """Dequantize weights back to FP32."""
        weights_f32 = self._fp8_e4m3_to_float(self.weights_fp8)
        weights_f32 = weights_f32 * self.weight_scales.unsqueeze(1)
        return weights_f32
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FP8 computation.
        
        Args:
            x: Input tensor of shape (batch, ..., in_features)
        
        Returns:
            Output tensor of shape (batch, ..., out_features)
        """
        # Get input shape
        orig_shape = x.shape
        x = x.view(-1, self.in_features)  # (N, in_features)
        
        # === Quantize activations (per-row) ===
        x_max = x.abs().max(dim=1, keepdim=True).values
        x_scales = x_max / self.FP8_E4M3_MAX
        x_scales = torch.where(x_scales == 0, torch.ones_like(x_scales), x_scales)
        
        x_scaled = x / x_scales
        x_fp8 = self._float_to_fp8_e4m3(x_scaled)
        x_dequant = self._fp8_e4m3_to_float(x_fp8)
        
        # === Dequantize weights ===
        weights_dequant = self.dequantize_weights()
        
        # === Matrix multiplication ===
        # y = x @ W^T
        # With scales: y = (x_q * x_scale) @ (W_q * w_scale)^T
        #            = (x_q @ W_q^T) * (x_scale @ w_scale^T)
        
        # Compute in FP32 (simulated FP8)
        output = F.linear(x_dequant * x_scales, weights_dequant, self.bias)
        
        # Reshape back
        output = output.view(*orig_shape[:-1], self.out_features)
        
        return output
    
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, '
                f'tile_size={self.tile_size}')


# === Quantization Utilities ===

def quantize_model_to_fp8(
    model: nn.Module,
    tile_size: int = 1,
    skip_patterns: list = None,
) -> nn.Module:
    """
    Quantize all Linear layers in a model to FP8.
    
    Args:
        model: The model to quantize
        tile_size: Tile size for per-tile scaling
        skip_patterns: Layer name patterns to skip
    
    Returns:
        Model with FP8Linear layers
    """
    skip_patterns = skip_patterns or ['lm_head', 'embed']
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check skip patterns
            if any(p in name for p in skip_patterns):
                continue
            
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            
            # Replace with FP8Linear
            fp8_linear = FP8Linear.from_linear(module, tile_size)
            setattr(parent, child_name, fp8_linear)
    
    return model


def compute_quantization_error(
    fp32_linear: nn.Linear,
    fp8_linear: FP8Linear,
    n_samples: int = 1000,
) -> dict:
    """Compute quantization error statistics."""
    # Weight error
    weight_dequant = fp8_linear.dequantize_weights()
    weight_error = (fp32_linear.weight - weight_dequant).abs()
    
    # Generate random inputs
    x = torch.randn(n_samples, fp32_linear.in_features)
    
    # Forward pass comparison
    with torch.no_grad():
        y_fp32 = fp32_linear(x)
        y_fp8 = fp8_linear(x)
    
    output_error = (y_fp32 - y_fp8).abs()
    
    return {
        'weight_mae': weight_error.mean().item(),
        'weight_max_error': weight_error.max().item(),
        'output_mae': output_error.mean().item(),
        'output_max_error': output_error.max().item(),
        'output_relative_error': (output_error / y_fp32.abs().clamp(min=1e-6)).mean().item(),
    }
```

---

## Hardware Considerations

### Nvidia H100/H200

Native FP8 tensor cores:
- **E4M3**: For weights and forward activations
- **E5M2**: For gradients
- **Throughput**: 2x FP16, 4x TF32

```python
# PyTorch 2.0+ FP8 on H100
import torch.cuda.amp as amp

with amp.autocast(dtype=torch.float8_e4m3fn):
    output = model(input)
```

### Apple Silicon (M3+)

ANE and GPU support FP8:
```python
# MLX FP8 support
import mlx.core as mx

x_fp8 = mx.quantize(x, bits=8, group_size=128)
```

### CPU Inference

For CPUs without FP8, use simulation:
```python
# Simulate FP8 with INT8
quantized = (x / scale).round().clamp(-127, 127).to(torch.int8)
dequantized = quantized.float() * scale
```

---

## Performance Analysis

### Memory Savings

| Precision | Bytes/Param | Reduction |
|-----------|-------------|-----------|
| FP32 | 4 | 1x |
| FP16/BF16 | 2 | 2x |
| **FP8** | **1** | **4x** |
| INT8 | 1 | 4x |
| INT4 | 0.5 | 8x |

### Accuracy Impact

Typical accuracy loss vs FP16:
- **Perplexity**: +0.1-0.3%
- **Downstream tasks**: <1% degradation
- **With per-tile scaling**: Negligible loss

### Throughput Gains

On H100:
| Operation | FP16 TFLOPs | FP8 TFLOPs | Speedup |
|-----------|-------------|------------|---------|
| MatMul | 989 | 1,979 | 2x |
| Memory BW | 3.35 TB/s | 3.35 TB/s | 1x (same) |

**Effective speedup** = min(2x compute, 1x memory) ≈ 1.5-2x for memory-bound ops

---

## Best Practices

### Quantization Strategy

1. **Keep embeddings FP16**: First/last layers most sensitive
2. **Use per-row scaling**: Better accuracy than per-tensor
3. **Quantize after training**: PTQ (Post-Training Quantization)
4. **Calibration data**: Use representative samples for scale computation

### Layer-Specific Recommendations

| Layer | Recommendation |
|-------|----------------|
| Embedding | Keep FP16 |
| Attention Q/K/V | FP8 safe |
| Attention Output | FP8 safe |
| FFN | FP8 safe |
| LM Head | Keep FP16 |
| LayerNorm | Keep FP32 |

### Debugging Tips

1. **Monitor activation ranges**: Should not exceed FP8 max
2. **Check for NaN/Inf**: Common with bad scaling
3. **Compare output distribution**: FP8 vs FP16 should match
4. **Use higher precision for critical paths**: Residual connections

---

## References

- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [NVIDIA H100 FP8 Training Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
