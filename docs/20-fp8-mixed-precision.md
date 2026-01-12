# FP8 Mixed-Precision Training

## Overview

**FP8 (8-bit Floating Point)** mixed-precision training is a memory and compute optimization technique that uses 8-bit floating point representations for forward/backward passes while maintaining higher precision for critical operations. DeepSeek V3 uses FP8 to achieve:
- **50% memory reduction** vs FP16
- **2x compute throughput** on modern hardware (H100, MI300X)
- **Minimal accuracy loss** (<0.1% perplexity increase)

**Key Papers:**
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) (Micikevicius et al., 2022)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (Micikevicius et al., 2017)
- [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861) (Dettmers et al., 2021)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek-AI, 2024)

## FP8 Format Specifications

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FP8 FORMAT VARIANTS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  E4M3 (Exponent 4, Mantissa 3) - Higher precision, smaller range            │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐                                          │
│  │ S │ E │ E │ E │ E │ M │ M │ M │                                          │
│  └───┴───┴───┴───┴───┴───┴───┴───┘                                          │
│  Range: ±448, Precision: ~0.125                                             │
│  Best for: Weights, Activations                                             │
│                                                                             │
│  E5M2 (Exponent 5, Mantissa 2) - Lower precision, larger range              │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐                                          │
│  │ S │ E │ E │ E │ E │ E │ M │ M │                                          │
│  └───┴───┴───┴───┴───┴───┴───┴───┘                                          │
│  Range: ±57344, Precision: ~0.25                                            │
│  Best for: Gradients, Loss scaling                                          │
│                                                                             │
│  Comparison with other formats:                                             │
│  ┌────────────┬───────────────┬─────────────┬────────────┐                  │
│  │ Format     │ Range         │ Precision   │ Memory     │                  │
│  ├────────────┼───────────────┼─────────────┼────────────┤                  │
│  │ FP32       │ ±3.4e38       │ 7 digits    │ 4 bytes    │                  │
│  │ BF16       │ ±3.4e38       │ 3 digits    │ 2 bytes    │                  │
│  │ FP16       │ ±65504        │ 4 digits    │ 2 bytes    │                  │
│  │ FP8 E4M3   │ ±448          │ 2 digits    │ 1 byte     │                  │
│  │ FP8 E5M2   │ ±57344        │ 1-2 digits  │ 1 byte     │                  │
│  └────────────┴───────────────┴─────────────┴────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Configuration

```python
@dataclass
class FP8Config:
    """Configuration for FP8 mixed-precision training."""
    
    # FP8 variants
    forward_dtype: str = "e4m3"      # E4M3 for forward pass
    backward_dtype: str = "e5m2"     # E5M2 for gradients
    
    # Scaling factors
    use_dynamic_scaling: bool = True
    initial_scale: float = 1.0
    scale_growth_factor: float = 2.0
    scale_backoff_factor: float = 0.5
    scale_growth_interval: int = 1000
    
    # Quantization settings
    per_tensor_scaling: bool = False  # Per-tensor vs per-channel
    symmetric: bool = True            # Symmetric quantization
    
    # Components to quantize
    quantize_weights: bool = True
    quantize_activations: bool = True
    quantize_gradients: bool = True
    
    # Exceptions (keep in higher precision)
    fp32_layers: List[str] = field(default_factory=lambda: [
        "embedding",
        "final_layernorm",
        "lm_head",
    ])
    
    # Numerical stability
    eps: float = 1e-12
    max_abs_val_e4m3: float = 448.0
    max_abs_val_e5m2: float = 57344.0
```

## Implementation

### FP8 Linear Layer

```python
class FP8Linear(nn.Module):
    """Linear layer with FP8 quantization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: FP8Config,
        bias: bool = True
    ):
        super().__init__()
        self.config = config
        
        # Master weights in FP32
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Scaling factors (learnable)
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('input_scale', torch.ones(1))
        self.register_buffer('grad_scale', torch.ones(1))
        
        # History for dynamic scaling
        self.register_buffer('amax_history', torch.zeros(config.scale_growth_interval))
        self.history_idx = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input to E4M3
        x_fp8, input_scale = self._quantize_e4m3(x)
        
        # Quantize weights to E4M3
        w_fp8, weight_scale = self._quantize_e4m3(self.weight)
        
        # Matrix multiplication in FP8 (simulated or native)
        if hasattr(torch, 'float8_e4m3fn'):
            # Native FP8 support (H100+)
            out = torch._scaled_mm(
                x_fp8.view(-1, x_fp8.shape[-1]),
                w_fp8.t(),
                scale_a=input_scale,
                scale_b=weight_scale,
                out_dtype=torch.bfloat16
            )
        else:
            # Simulated FP8
            out = F.linear(
                x_fp8.float() * input_scale,
                w_fp8.float() * weight_scale
            )
        
        out = out.view(*x.shape[:-1], -1)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def _quantize_e4m3(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to FP8 E4M3 format."""
        max_val = self.config.max_abs_val_e4m3
        
        # Compute optimal scale
        amax = x.abs().max()
        scale = max_val / (amax + self.config.eps)
        
        # Scale and clamp
        x_scaled = x * scale
        x_clamped = x_scaled.clamp(-max_val, max_val)
        
        # Quantize (round to representable values)
        if hasattr(torch, 'float8_e4m3fn'):
            x_fp8 = x_clamped.to(torch.float8_e4m3fn)
        else:
            # Simulate FP8 by reducing precision
            x_fp8 = self._simulate_fp8_e4m3(x_clamped)
        
        return x_fp8, 1.0 / scale
    
    def _simulate_fp8_e4m3(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate FP8 E4M3 quantization."""
        # Round to 3-bit mantissa precision
        # This loses precision equivalent to E4M3
        mantissa_bits = 3
        scale = 2 ** mantissa_bits
        return torch.round(x * scale) / scale
```

### Dynamic Loss Scaling

```python
class DynamicLossScaler:
    """Dynamic loss scaling for FP8 training stability."""
    
    def __init__(self, config: FP8Config):
        self.config = config
        self.scale = config.initial_scale
        self.growth_factor = config.scale_growth_factor
        self.backoff_factor = config.scale_backoff_factor
        self.growth_interval = config.scale_growth_interval
        self.steps_since_growth = 0
        self.overflow_count = 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient computation."""
        return loss * self.scale
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> bool:
        """Unscale gradients and check for overflow."""
        found_inf = False
        
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    # Unscale
                    param.grad.data.div_(self.scale)
                    
                    # Check for overflow
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        found_inf = True
                        param.grad.data.zero_()
        
        return found_inf
    
    def update(self, overflow: bool):
        """Update scale based on overflow status."""
        if overflow:
            # Reduce scale on overflow
            self.scale *= self.backoff_factor
            self.steps_since_growth = 0
            self.overflow_count += 1
        else:
            self.steps_since_growth += 1
            
            # Grow scale periodically
            if self.steps_since_growth >= self.growth_interval:
                self.scale *= self.growth_factor
                self.steps_since_growth = 0
    
    def state_dict(self) -> dict:
        return {
            'scale': self.scale,
            'steps_since_growth': self.steps_since_growth,
            'overflow_count': self.overflow_count,
        }
    
    def load_state_dict(self, state_dict: dict):
        self.scale = state_dict['scale']
        self.steps_since_growth = state_dict['steps_since_growth']
        self.overflow_count = state_dict['overflow_count']
```

### FP8 Training Loop

```python
class FP8Trainer:
    """Training loop with FP8 mixed precision."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: FP8Config
    ):
        self.model = self._convert_to_fp8(model, config)
        self.optimizer = optimizer
        self.config = config
        self.scaler = DynamicLossScaler(config)
    
    def _convert_to_fp8(self, model: nn.Module, config: FP8Config) -> nn.Module:
        """Convert model layers to FP8."""
        for name, module in model.named_modules():
            # Skip specified layers
            if any(skip in name for skip in config.fp32_layers):
                continue
            
            # Replace Linear with FP8Linear
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                
                fp8_linear = FP8Linear(
                    module.in_features,
                    module.out_features,
                    config,
                    bias=module.bias is not None
                )
                fp8_linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    fp8_linear.bias.data = module.bias.data.clone()
                
                setattr(parent, module_name, fp8_linear)
        
        return model
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Single training step with FP8."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass (FP8 for compute-heavy layers)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.model(**batch)
            loss = outputs.loss
        
        # Scale loss and backward
        scaled_loss = self.scaler.scale_loss(loss)
        scaled_loss.backward()
        
        # Unscale and check for overflow
        overflow = self.scaler.unscale_gradients(self.optimizer)
        
        if not overflow:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
        
        # Update scaler
        self.scaler.update(overflow)
        
        return {
            'loss': loss.item(),
            'scale': self.scaler.scale,
            'overflow': overflow,
        }
```

### Quantization-Aware Matmul

```python
class FP8MatMul(torch.autograd.Function):
    """Custom FP8 matrix multiplication with proper gradient handling."""
    
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        config: FP8Config
    ) -> torch.Tensor:
        # Save for backward
        ctx.save_for_backward(input, weight, input_scale, weight_scale)
        ctx.config = config
        
        # Quantize inputs
        input_fp8 = FP8MatMul._to_fp8_e4m3(input, input_scale, config)
        weight_fp8 = FP8MatMul._to_fp8_e4m3(weight, weight_scale, config)
        
        # Compute in higher precision, simulating FP8 tensor cores
        output = torch.matmul(
            input_fp8.float() * input_scale,
            weight_fp8.float().t() * weight_scale
        )
        
        return output.to(input.dtype)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, input_scale, weight_scale = ctx.saved_tensors
        config = ctx.config
        
        # Quantize gradients to E5M2
        grad_fp8 = FP8MatMul._to_fp8_e5m2(grad_output, config)
        
        # Gradient w.r.t. input
        grad_input = torch.matmul(
            grad_fp8.float(),
            weight.float() * weight_scale
        )
        
        # Gradient w.r.t. weight
        grad_weight = torch.matmul(
            grad_fp8.float().t(),
            input.float() * input_scale
        )
        
        return grad_input, grad_weight.t(), None, None, None
    
    @staticmethod
    def _to_fp8_e4m3(x: torch.Tensor, scale: torch.Tensor, config: FP8Config):
        scaled = x / scale
        clamped = scaled.clamp(-config.max_abs_val_e4m3, config.max_abs_val_e4m3)
        # Simulate reduced precision
        return torch.round(clamped * 8) / 8  # 3-bit mantissa
    
    @staticmethod
    def _to_fp8_e5m2(x: torch.Tensor, config: FP8Config):
        max_val = config.max_abs_val_e5m2
        scale = max_val / (x.abs().max() + config.eps)
        scaled = x * scale
        clamped = scaled.clamp(-max_val, max_val)
        # Simulate 2-bit mantissa
        return torch.round(clamped * 4) / 4 / scale
```

## Rust Implementation

```rust
pub struct FP8Config {
    pub forward_dtype: FP8Format,
    pub backward_dtype: FP8Format,
    pub use_dynamic_scaling: bool,
    pub initial_scale: f32,
    pub scale_growth_factor: f32,
    pub scale_backoff_factor: f32,
    pub max_abs_val_e4m3: f32,
    pub max_abs_val_e5m2: f32,
}

#[derive(Clone, Copy)]
pub enum FP8Format {
    E4M3,
    E5M2,
}

pub struct FP8Linear {
    config: FP8Config,
    weight: Tensor,
    bias: Option<Tensor>,
    weight_scale: f32,
}

impl FP8Linear {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Quantize input
        let (x_fp8, input_scale) = self.quantize_e4m3(x);
        
        // Quantize weights
        let (w_fp8, weight_scale) = self.quantize_e4m3(&self.weight);
        
        // Matrix multiplication with dequantization
        let output = x_fp8.matmul(&w_fp8.t())
            .mul_scalar(input_scale * weight_scale);
        
        // Add bias
        match &self.bias {
            Some(b) => output.add(b),
            None => output,
        }
    }
    
    fn quantize_e4m3(&self, x: &Tensor) -> (Tensor, f32) {
        let max_val = self.config.max_abs_val_e4m3;
        let amax = x.abs().max().item::<f32>();
        let scale = max_val / (amax + 1e-12);
        
        // Scale and clamp
        let x_scaled = x.mul_scalar(scale);
        let x_clamped = x_scaled.clamp(-max_val, max_val);
        
        // Simulate 3-bit mantissa
        let x_quantized = (x_clamped.mul_scalar(8.0)).round().div_scalar(8.0);
        
        (x_quantized, 1.0 / scale)
    }
}

pub struct DynamicLossScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    steps_since_growth: usize,
}

impl DynamicLossScaler {
    pub fn scale_loss(&self, loss: &Tensor) -> Tensor {
        loss.mul_scalar(self.scale)
    }
    
    pub fn unscale_gradients(&self, grads: &mut [Tensor]) -> bool {
        let mut found_inf = false;
        
        for grad in grads.iter_mut() {
            *grad = grad.div_scalar(self.scale);
            
            if grad.is_inf().any().item::<bool>() || grad.is_nan().any().item::<bool>() {
                found_inf = true;
                *grad = Tensor::zeros_like(grad);
            }
        }
        
        found_inf
    }
    
    pub fn update(&mut self, overflow: bool) {
        if overflow {
            self.scale *= self.backoff_factor;
            self.steps_since_growth = 0;
        } else {
            self.steps_since_growth += 1;
            if self.steps_since_growth >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_growth = 0;
            }
        }
    }
}
```

## Calibration and Quantization Ranges

### Finding Optimal Scales

```python
class FP8Calibrator:
    """Calibrate quantization scales before training."""
    
    def __init__(self, model: nn.Module, config: FP8Config):
        self.model = model
        self.config = config
        self.hooks = []
        self.activation_stats = {}
    
    def calibrate(self, dataloader: DataLoader, num_batches: int = 100):
        """Run calibration to find optimal scales."""
        self._register_hooks()
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                self.model(**batch)
        
        self._remove_hooks()
        
        # Compute optimal scales
        scales = {}
        for name, stats in self.activation_stats.items():
            amax = stats['amax']
            scales[name] = self.config.max_abs_val_e4m3 / (amax + 1e-12)
        
        return scales
    
    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                if name not in self.activation_stats:
                    self.activation_stats[name] = {'amax': 0}
                
                if isinstance(output, torch.Tensor):
                    amax = output.abs().max().item()
                    self.activation_stats[name]['amax'] = max(
                        self.activation_stats[name]['amax'],
                        amax
                    )
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, FP8Linear)):
                self.hooks.append(module.register_forward_hook(hook_fn(name)))
```

## Memory and Compute Benefits

### Memory Comparison

| Model Size | FP32 | FP16 | FP8 | Savings vs FP16 |
|------------|------|------|-----|-----------------|
| 7B | 28 GB | 14 GB | 7 GB | 50% |
| 70B | 280 GB | 140 GB | 70 GB | 50% |
| 671B | 2.7 TB | 1.35 TB | 675 GB | 50% |

### Throughput (H100 SXM)

| Operation | FP16 TFLOPS | FP8 TFLOPS | Speedup |
|-----------|-------------|------------|---------|
| Dense MatMul | 989 | 1979 | 2.0x |
| Sparse MatMul | 1978 | 3958 | 2.0x |
| Memory BW | 3.35 TB/s | 3.35 TB/s | 1.0x |

## Best Practices

### Training Stability
1. **Keep certain layers in FP32**: Embeddings, final LayerNorm, output head
2. **Dynamic scaling**: Adjust scale based on gradient overflow
3. **Gradient clipping**: Essential with reduced precision

### Accuracy Preservation
1. **Calibration**: Run calibration pass to find optimal scales
2. **Per-channel scaling**: More accurate than per-tensor for weights
3. **Warmup**: Start with FP16 for first 1% of training

### Hardware Considerations
1. **H100/MI300X**: Native FP8 tensor cores
2. **Older GPUs**: Simulation mode (reduced speedup)
3. **Batch size**: Larger batches amortize quantization overhead

## Summary

FP8 mixed-precision training achieves:
- **50% memory reduction** enabling larger models/batches
- **2x compute throughput** on modern GPUs
- **<0.1% quality loss** with proper implementation
- **Seamless integration** with existing training pipelines

This is essential for efficient large-scale model training on current hardware.
