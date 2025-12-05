import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum

# ============================================================================
# FP8 Format Constants (DeepSeek-V3.2 Mixed-Precision Training)
# ============================================================================

# E4M3 Format: 4-bit exponent, 3-bit mantissa
# Range: +/- 448.0, used for forward activations and weights
E4M3_MAX = 448.0
E4M3_MIN = -448.0
E4M3_SMALLEST_NORMAL = 2 ** -6

# E5M2 Format: 5-bit exponent, 2-bit mantissa
# Range: +/- 57344.0, used for gradients (larger range for stability)
E5M2_MAX = 57344.0
E5M2_MIN = -57344.0
E5M2_SMALLEST_NORMAL = 2 ** -14

# Tile size for per-tile scaling (128x128 as per DeepSeek-V3)
FP8_TILE_SIZE = 128


class FP8Format(Enum):
    """FP8 format selection for different use cases."""
    E4M3 = "e4m3"  # Higher precision, lower range - for forward pass
    E5M2 = "e5m2"  # Lower precision, higher range - for gradients
    
    @property
    def max_value(self) -> float:
        return E4M3_MAX if self == FP8Format.E4M3 else E5M2_MAX
    
    @property
    def min_value(self) -> float:
        return E4M3_MIN if self == FP8Format.E4M3 else E5M2_MIN
    
    @property
    def smallest_normal(self) -> float:
        return E4M3_SMALLEST_NORMAL if self == FP8Format.E4M3 else E5M2_SMALLEST_NORMAL


# ============================================================================
# Per-Tile Scaling Infrastructure (128x128 tiles)
# ============================================================================

@dataclass
class TileScalingConfig:
    """Per-tile scaling configuration for FP8 quantization."""
    tile_rows: int = FP8_TILE_SIZE
    tile_cols: int = FP8_TILE_SIZE
    forward_format: FP8Format = field(default_factory=lambda: FP8Format.E4M3)
    backward_format: FP8Format = field(default_factory=lambda: FP8Format.E5M2)
    dynamic_scaling: bool = True
    amax_history_len: int = 16


class TileScalingState:
    """Tile-based scaling state for FP8 training."""
    
    def __init__(self, config: TileScalingConfig):
        self.config = config
        self.scales: Optional[mx.array] = None
        self.amax_history: List[mx.array] = []
        self.history_idx = 0
    
    def compute_tile_scales(
        self, 
        tensor: mx.array, 
        fp8_format: FP8Format
    ) -> mx.array:
        """Compute per-tile scales for a tensor."""
        original_shape = tensor.shape
        
        # Flatten to 2D
        if len(tensor.shape) == 2:
            rows, cols = tensor.shape
        else:
            rows = int(mx.prod(mx.array(tensor.shape[:-1])))
            cols = tensor.shape[-1]
        
        tensor_2d = tensor.reshape(rows, cols)
        
        tile_r = self.config.tile_rows
        tile_c = self.config.tile_cols
        
        # Pad to tile boundaries
        pad_r = (tile_r - (rows % tile_r)) % tile_r
        pad_c = (tile_c - (cols % tile_c)) % tile_c
        
        if pad_r > 0 or pad_c > 0:
            # MLX pad: pad along last axis first, then second-to-last
            if pad_c > 0:
                tensor_2d = mx.pad(tensor_2d, [(0, 0), (0, pad_c)])
            if pad_r > 0:
                tensor_2d = mx.pad(tensor_2d, [(0, pad_r), (0, 0)])
        
        n_tiles_r = (rows + pad_r) // tile_r
        n_tiles_c = (cols + pad_c) // tile_c
        
        # Reshape to tiles: (n_tiles_r, tile_r, n_tiles_c, tile_c)
        tiled = tensor_2d.reshape(n_tiles_r, tile_r, n_tiles_c, tile_c)
        tiled = mx.transpose(tiled, (0, 2, 1, 3))  # (n_tiles_r, n_tiles_c, tile_r, tile_c)
        
        # Compute amax per tile
        amax = mx.max(mx.abs(tiled), axis=(-1, -2))  # (n_tiles_r, n_tiles_c)
        
        # Update history for dynamic scaling
        if self.config.dynamic_scaling:
            self._update_amax_history(amax)
        
        # Compute scale = amax / format_max
        scales = mx.maximum(amax, mx.array(1e-12)) / fp8_format.max_value
        
        self.scales = scales
        return scales
    
    def _update_amax_history(self, amax: mx.array):
        """Update amax history for dynamic scaling."""
        if len(self.amax_history) < self.config.amax_history_len:
            self.amax_history.append(amax)
        else:
            self.amax_history[self.history_idx] = amax
        self.history_idx = (self.history_idx + 1) % self.config.amax_history_len
    
    def get_smoothed_amax(self) -> Optional[mx.array]:
        """Get smoothed amax from history."""
        if not self.amax_history:
            return None
        stacked = mx.stack(self.amax_history, axis=0)
        return mx.max(stacked, axis=0)


def quantize_fp8_tiled(
    tensor: mx.array,
    scales: mx.array,
    fp8_format: FP8Format,
    tile_rows: int = FP8_TILE_SIZE,
    tile_cols: int = FP8_TILE_SIZE,
) -> mx.array:
    """Quantize tensor to FP8 with per-tile scaling."""
    original_shape = tensor.shape
    
    # Flatten to 2D
    if len(tensor.shape) == 2:
        rows, cols = tensor.shape
    else:
        rows = int(mx.prod(mx.array(tensor.shape[:-1])))
        cols = tensor.shape[-1]
    
    tensor_2d = tensor.reshape(rows, cols)
    
    # Pad to tile boundaries
    pad_r = (tile_rows - (rows % tile_rows)) % tile_rows
    pad_c = (tile_cols - (cols % tile_cols)) % tile_cols
    
    if pad_r > 0 or pad_c > 0:
        if pad_c > 0:
            tensor_2d = mx.pad(tensor_2d, [(0, 0), (0, pad_c)])
        if pad_r > 0:
            tensor_2d = mx.pad(tensor_2d, [(0, pad_r), (0, 0)])
    
    n_tiles_r = (rows + pad_r) // tile_rows
    n_tiles_c = (cols + pad_c) // tile_cols
    
    # Reshape to tiles
    tiled = tensor_2d.reshape(n_tiles_r, tile_rows, n_tiles_c, tile_cols)
    tiled = mx.transpose(tiled, (0, 2, 1, 3))
    
    # Expand scales: (n_tiles_r, n_tiles_c) -> (n_tiles_r, n_tiles_c, 1, 1)
    scales_expanded = scales.reshape(n_tiles_r, n_tiles_c, 1, 1)
    
    # Quantize
    quantized = mx.round(tiled / scales_expanded)
    quantized = mx.clip(quantized, fp8_format.min_value, fp8_format.max_value)
    
    # Reshape back
    quantized = mx.transpose(quantized, (0, 2, 1, 3))
    quantized = quantized.reshape((rows + pad_r), (cols + pad_c))
    
    # Remove padding
    if pad_r > 0 or pad_c > 0:
        quantized = quantized[:rows, :cols]
    
    return quantized.reshape(original_shape)


def dequantize_fp8_tiled(
    quantized: mx.array,
    scales: mx.array,
    tile_rows: int = FP8_TILE_SIZE,
    tile_cols: int = FP8_TILE_SIZE,
) -> mx.array:
    """Dequantize FP8 tensor with per-tile scaling."""
    original_shape = quantized.shape
    
    # Flatten to 2D
    if len(quantized.shape) == 2:
        rows, cols = quantized.shape
    else:
        rows = int(mx.prod(mx.array(quantized.shape[:-1])))
        cols = quantized.shape[-1]
    
    tensor_2d = quantized.reshape(rows, cols)
    
    # Pad to tile boundaries
    pad_r = (tile_rows - (rows % tile_rows)) % tile_rows
    pad_c = (tile_cols - (cols % tile_cols)) % tile_cols
    
    if pad_r > 0 or pad_c > 0:
        if pad_c > 0:
            tensor_2d = mx.pad(tensor_2d, [(0, 0), (0, pad_c)])
        if pad_r > 0:
            tensor_2d = mx.pad(tensor_2d, [(0, pad_r), (0, 0)])
    
    n_tiles_r = (rows + pad_r) // tile_rows
    n_tiles_c = (cols + pad_c) // tile_cols
    
    # Reshape to tiles
    tiled = tensor_2d.reshape(n_tiles_r, tile_rows, n_tiles_c, tile_cols)
    tiled = mx.transpose(tiled, (0, 2, 1, 3))
    
    # Expand scales
    scales_expanded = scales.reshape(n_tiles_r, n_tiles_c, 1, 1)
    
    # Dequantize
    dequantized = tiled * scales_expanded
    
    # Reshape back
    dequantized = mx.transpose(dequantized, (0, 2, 1, 3))
    dequantized = dequantized.reshape((rows + pad_r), (cols + pad_c))
    
    # Remove padding
    if pad_r > 0 or pad_c > 0:
        dequantized = dequantized[:rows, :cols]
    
    return dequantized.reshape(original_shape)


# ============================================================================
# FP8 Optimizer State Quantization
# ============================================================================

@dataclass
class FP8OptimizerConfig:
    """Configuration for FP8 optimizer."""
    block_wise_scaling: bool = True
    block_size: int = 128
    scale_update_interval: int = 1
    moment_format: FP8Format = field(default_factory=lambda: FP8Format.E4M3)


class FP8OptimizerState:
    """FP8 quantized optimizer state for memory efficiency."""
    
    def __init__(
        self,
        m: mx.array,
        v: mx.array,
        config: FP8OptimizerConfig,
    ):
        self.config = config
        self.step = 0
        
        # Quantize moments
        self.m_quantized, self.m_scale = self._quantize_moment(m)
        self.v_quantized, self.v_scale = self._quantize_moment(v)
    
    def _quantize_moment(self, moment: mx.array) -> Tuple[mx.array, mx.array]:
        """Quantize a moment tensor to FP8."""
        fmt = self.config.moment_format
        amax = mx.max(mx.abs(moment))
        scale = mx.maximum(amax, mx.array(1e-12)) / fmt.max_value
        
        quantized = mx.round(moment / scale)
        quantized = mx.clip(quantized, fmt.min_value, fmt.max_value)
        return quantized, scale
    
    def to_fp32(self) -> Tuple[mx.array, mx.array]:
        """Dequantize moments to FP32."""
        m = self.m_quantized * self.m_scale
        v = self.v_quantized * self.v_scale
        return m, v
    
    def update(
        self,
        grad: mx.array,
        beta1: float,
        beta2: float,
    ) -> Tuple[mx.array, mx.array]:
        """Update moments (Adam-style) and requantize."""
        # Dequantize
        m, v = self.to_fp32()
        
        # Adam update
        m_new = beta1 * m + (1 - beta1) * grad
        v_new = beta2 * v + (1 - beta2) * mx.square(grad)
        
        self.step += 1
        
        # Re-quantize
        if self.step % self.config.scale_update_interval == 0:
            self.m_quantized, self.m_scale = self._quantize_moment(m_new)
            self.v_quantized, self.v_scale = self._quantize_moment(v_new)
        else:
            # Quantize with existing scale
            fmt = self.config.moment_format
            self.m_quantized = mx.clip(
                mx.round(m_new / self.m_scale), 
                fmt.min_value, fmt.max_value
            )
            self.v_quantized = mx.clip(
                mx.round(v_new / self.v_scale),
                fmt.min_value, fmt.max_value
            )
        
        return m_new, v_new
    
    def get_bias_corrected(self, beta1: float, beta2: float) -> Tuple[mx.array, mx.array]:
        """Get bias-corrected moments."""
        m, v = self.to_fp32()
        
        bias_correction1 = 1 - beta1 ** self.step
        bias_correction2 = 1 - beta2 ** self.step
        
        m_hat = m / bias_correction1
        v_hat = v / bias_correction2
        
        return m_hat, v_hat
    
    @property
    def memory_savings_ratio(self) -> float:
        """Memory savings compared to FP32."""
        return 0.25 * 1.01


# ============================================================================
# FP8 Mixed-Precision Training Manager
# ============================================================================

@dataclass
class FP8TrainingConfig:
    """Configuration for FP8 mixed-precision training."""
    initial_loss_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    max_loss_scale: float = 2 ** 24
    min_loss_scale: float = 1.0
    fp8_forward: bool = True
    fp8_backward: bool = True
    fp8_optimizer: bool = True
    tile_config: TileScalingConfig = field(default_factory=TileScalingConfig)
    optimizer_config: FP8OptimizerConfig = field(default_factory=FP8OptimizerConfig)


class FP8MixedPrecisionManager:
    """Manages FP8 mixed-precision training with dynamic loss scaling."""
    
    def __init__(self, config: FP8TrainingConfig):
        self.config = config
        self.loss_scale = config.initial_loss_scale
        self.growth_tracker = 0
        self.optimizer_states: Dict[str, FP8OptimizerState] = {}
        self.activation_scales: Dict[str, TileScalingState] = {}
        self.weight_scales: Dict[str, TileScalingState] = {}
    
    def scale_loss(self, loss: mx.array) -> mx.array:
        """Scale loss for FP8 training."""
        return loss * self.loss_scale
    
    def unscale_gradients(self, gradients: List[mx.array]) -> bool:
        """Unscale gradients after backward pass. Returns False if overflow detected."""
        inv_scale = 1.0 / self.loss_scale
        
        # Check for overflow (MLX doesn't have in-place ops, so we check)
        for grad in gradients:
            if grad is not None:
                if mx.any(mx.isinf(grad)) or mx.any(mx.isnan(grad)):
                    return False
        
        # Unscale (returns new tensors in MLX)
        for i, grad in enumerate(gradients):
            if grad is not None:
                gradients[i] = grad * inv_scale
        
        return True
    
    def update_loss_scale(self, overflow: bool):
        """Update loss scale based on overflow status."""
        if overflow:
            self.loss_scale = max(
                self.loss_scale * self.config.backoff_factor,
                self.config.min_loss_scale
            )
            self.growth_tracker = 0
            
            if self.loss_scale <= self.config.min_loss_scale:
                print(f"Warning: Loss scale at minimum ({self.config.min_loss_scale})")
        else:
            self.growth_tracker += 1
            if self.growth_tracker >= self.config.growth_interval:
                self.loss_scale = min(
                    self.loss_scale * self.config.growth_factor,
                    self.config.max_loss_scale
                )
                self.growth_tracker = 0
    
    def quantize_activation(
        self, 
        name: str, 
        tensor: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Quantize activations for forward pass."""
        if not self.config.fp8_forward:
            return tensor, mx.ones((1,))
        
        if name not in self.activation_scales:
            self.activation_scales[name] = TileScalingState(self.config.tile_config)
        
        state = self.activation_scales[name]
        scales = state.compute_tile_scales(tensor, FP8Format.E4M3)
        quantized = quantize_fp8_tiled(
            tensor, scales, FP8Format.E4M3,
            self.config.tile_config.tile_rows,
            self.config.tile_config.tile_cols,
        )
        
        return quantized, scales
    
    def quantize_gradient(
        self,
        name: str,
        tensor: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Quantize gradients for backward pass."""
        if not self.config.fp8_backward:
            return tensor, mx.ones((1,))
        
        grad_name = f"{name}_grad"
        if grad_name not in self.activation_scales:
            self.activation_scales[grad_name] = TileScalingState(self.config.tile_config)
        
        state = self.activation_scales[grad_name]
        scales = state.compute_tile_scales(tensor, FP8Format.E5M2)
        quantized = quantize_fp8_tiled(
            tensor, scales, FP8Format.E5M2,
            self.config.tile_config.tile_rows,
            self.config.tile_config.tile_cols,
        )
        
        return quantized, scales
    
    def init_optimizer_state(self, name: str, param: mx.array):
        """Initialize FP8 optimizer state for a parameter."""
        if not self.config.fp8_optimizer:
            return
        
        m = mx.zeros_like(param)
        v = mx.zeros_like(param)
        
        self.optimizer_states[name] = FP8OptimizerState(
            m, v, self.config.optimizer_config
        )
    
    def get_optimizer_state(self, name: str) -> Optional[FP8OptimizerState]:
        """Get optimizer state for a parameter."""
        return self.optimizer_states.get(name)
    
    def memory_savings_report(self) -> dict:
        """Report memory savings from FP8."""
        total_fp32_bytes = 0
        total_fp8_bytes = 0
        
        for state in self.optimizer_states.values():
            param_size = state.m_quantized.size
            total_fp32_bytes += param_size * 8
            total_fp8_bytes += param_size * 2 + 64
        
        savings_ratio = 1.0 - (total_fp8_bytes / total_fp32_bytes) if total_fp32_bytes > 0 else 0.0
        
        return {
            "fp32_bytes": total_fp32_bytes,
            "fp8_bytes": total_fp8_bytes,
            "savings_ratio": savings_ratio,
            "optimizer_states_count": len(self.optimizer_states),
        }


# ============================================================================
# Original FP8Linear (Enhanced)
# ============================================================================

class FP8Linear(nn.Module):
    """
    Simulated FP8 Linear Layer for MLX with per-tile scaling.
    """
    def __init__(self, input_dim: int, output_dim: int, tile_size: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tile_size = tile_size
        self.weight = mx.random.normal((output_dim, input_dim)) * 0.02
        self.bias = mx.zeros((output_dim,))
        
        # Scaling state
        self.weight_scale_state = TileScalingState(TileScalingConfig(
            tile_rows=tile_size, tile_cols=tile_size
        ))
        self.input_scale_state = TileScalingState(TileScalingConfig(
            tile_rows=tile_size, tile_cols=tile_size
        ))
        
    def _quantize(self, x: mx.array, fp8_format: FP8Format = FP8Format.E4M3) -> mx.array:
        """Simulate FP8 quantization with per-tensor scaling."""
        scale = fp8_format.max_value / mx.max(mx.abs(x))
        x_scaled = x * scale
        x_int = mx.round(x_scaled)
        x_int = mx.clip(x_int, fp8_format.min_value, fp8_format.max_value)
        return x_int / scale
        
    def __call__(self, x: mx.array) -> mx.array:
        # Quantize input and weight
        x_q = self._quantize(x, FP8Format.E4M3)
        w_q = self._quantize(self.weight, FP8Format.E4M3)
        
        # Matrix multiply
        out = x_q @ w_q.T
        
        # Add bias
        return out + self.bias
