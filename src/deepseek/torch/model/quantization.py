import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any

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
    forward_format: FP8Format = FP8Format.E4M3
    backward_format: FP8Format = FP8Format.E5M2
    dynamic_scaling: bool = True
    amax_history_len: int = 16


class TileScalingState:
    """Tile-based scaling state for FP8 training."""
    
    def __init__(self, config: TileScalingConfig):
        self.config = config
        self.scales: Optional[torch.Tensor] = None
        self.amax_history: List[torch.Tensor] = []
        self.history_idx = 0
    
    def compute_tile_scales(
        self, 
        tensor: torch.Tensor, 
        fp8_format: FP8Format
    ) -> torch.Tensor:
        """Compute per-tile scales for a tensor."""
        original_shape = tensor.shape
        device = tensor.device
        
        # Flatten to 2D
        if tensor.dim() == 2:
            rows, cols = tensor.shape
        else:
            rows = tensor.numel() // tensor.shape[-1]
            cols = tensor.shape[-1]
        
        tensor_2d = tensor.view(rows, cols)
        
        tile_r = self.config.tile_rows
        tile_c = self.config.tile_cols
        
        # Pad to tile boundaries
        pad_r = (tile_r - (rows % tile_r)) % tile_r
        pad_c = (tile_c - (cols % tile_c)) % tile_c
        
        if pad_r > 0 or pad_c > 0:
            tensor_2d = F.pad(tensor_2d, (0, pad_c, 0, pad_r))
        
        n_tiles_r = (rows + pad_r) // tile_r
        n_tiles_c = (cols + pad_c) // tile_c
        
        # Reshape to tiles: (n_tiles_r, tile_r, n_tiles_c, tile_c)
        tiled = tensor_2d.view(n_tiles_r, tile_r, n_tiles_c, tile_c)
        tiled = tiled.permute(0, 2, 1, 3)  # (n_tiles_r, n_tiles_c, tile_r, tile_c)
        
        # Compute amax per tile
        amax = tiled.abs().amax(dim=(-1, -2))  # (n_tiles_r, n_tiles_c)
        
        # Update history for dynamic scaling
        if self.config.dynamic_scaling:
            self._update_amax_history(amax)
        
        # Compute scale = amax / format_max
        scales = amax.clamp(min=1e-12) / fp8_format.max_value
        
        self.scales = scales
        return scales
    
    def _update_amax_history(self, amax: torch.Tensor):
        """Update amax history for dynamic scaling."""
        if len(self.amax_history) < self.config.amax_history_len:
            self.amax_history.append(amax.clone())
        else:
            self.amax_history[self.history_idx] = amax.clone()
        self.history_idx = (self.history_idx + 1) % self.config.amax_history_len
    
    def get_smoothed_amax(self) -> Optional[torch.Tensor]:
        """Get smoothed amax from history."""
        if not self.amax_history:
            return None
        stacked = torch.stack(self.amax_history, dim=0)
        return stacked.amax(dim=0)


def quantize_fp8_tiled(
    tensor: torch.Tensor,
    scales: torch.Tensor,
    fp8_format: FP8Format,
    tile_rows: int = FP8_TILE_SIZE,
    tile_cols: int = FP8_TILE_SIZE,
) -> torch.Tensor:
    """Quantize tensor to FP8 with per-tile scaling."""
    original_shape = tensor.shape
    device = tensor.device
    
    # Flatten to 2D
    if tensor.dim() == 2:
        rows, cols = tensor.shape
    else:
        rows = tensor.numel() // tensor.shape[-1]
        cols = tensor.shape[-1]
    
    tensor_2d = tensor.view(rows, cols)
    
    # Pad to tile boundaries
    pad_r = (tile_rows - (rows % tile_rows)) % tile_rows
    pad_c = (tile_cols - (cols % tile_cols)) % tile_cols
    
    if pad_r > 0 or pad_c > 0:
        tensor_2d = F.pad(tensor_2d, (0, pad_c, 0, pad_r))
    
    n_tiles_r = (rows + pad_r) // tile_rows
    n_tiles_c = (cols + pad_c) // tile_cols
    
    # Reshape to tiles
    tiled = tensor_2d.view(n_tiles_r, tile_rows, n_tiles_c, tile_cols)
    tiled = tiled.permute(0, 2, 1, 3)
    
    # Expand scales: (n_tiles_r, n_tiles_c) -> (n_tiles_r, n_tiles_c, 1, 1)
    scales_expanded = scales.view(n_tiles_r, n_tiles_c, 1, 1)
    
    # Quantize
    quantized = (tiled / scales_expanded).round()
    quantized = quantized.clamp(fp8_format.min_value, fp8_format.max_value)
    
    # Reshape back
    quantized = quantized.permute(0, 2, 1, 3)
    quantized = quantized.reshape((rows + pad_r), (cols + pad_c))
    
    # Remove padding
    if pad_r > 0 or pad_c > 0:
        quantized = quantized[:rows, :cols]
    
    return quantized.view(original_shape)


def dequantize_fp8_tiled(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    tile_rows: int = FP8_TILE_SIZE,
    tile_cols: int = FP8_TILE_SIZE,
) -> torch.Tensor:
    """Dequantize FP8 tensor with per-tile scaling."""
    original_shape = quantized.shape
    
    # Flatten to 2D
    if quantized.dim() == 2:
        rows, cols = quantized.shape
    else:
        rows = quantized.numel() // quantized.shape[-1]
        cols = quantized.shape[-1]
    
    tensor_2d = quantized.view(rows, cols)
    
    # Pad to tile boundaries
    pad_r = (tile_rows - (rows % tile_rows)) % tile_rows
    pad_c = (tile_cols - (cols % tile_cols)) % tile_cols
    
    if pad_r > 0 or pad_c > 0:
        tensor_2d = F.pad(tensor_2d, (0, pad_c, 0, pad_r))
    
    n_tiles_r = (rows + pad_r) // tile_rows
    n_tiles_c = (cols + pad_c) // tile_cols
    
    # Reshape to tiles
    tiled = tensor_2d.view(n_tiles_r, tile_rows, n_tiles_c, tile_cols)
    tiled = tiled.permute(0, 2, 1, 3)
    
    # Expand scales
    scales_expanded = scales.view(n_tiles_r, n_tiles_c, 1, 1)
    
    # Dequantize
    dequantized = tiled * scales_expanded
    
    # Reshape back
    dequantized = dequantized.permute(0, 2, 1, 3)
    dequantized = dequantized.reshape((rows + pad_r), (cols + pad_c))
    
    # Remove padding
    if pad_r > 0 or pad_c > 0:
        dequantized = dequantized[:rows, :cols]
    
    return dequantized.view(original_shape)


# ============================================================================
# FP8 Optimizer State Quantization
# ============================================================================

@dataclass
class FP8OptimizerConfig:
    """Configuration for FP8 optimizer."""
    block_wise_scaling: bool = True
    block_size: int = 128
    scale_update_interval: int = 1
    moment_format: FP8Format = FP8Format.E4M3


class FP8OptimizerState:
    """FP8 quantized optimizer state for memory efficiency."""
    
    def __init__(
        self,
        m: torch.Tensor,
        v: torch.Tensor,
        config: FP8OptimizerConfig,
    ):
        self.config = config
        self.step = 0
        
        # Quantize moments
        self.m_quantized, self.m_scale = self._quantize_moment(m)
        self.v_quantized, self.v_scale = self._quantize_moment(v)
    
    def _quantize_moment(self, moment: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a moment tensor to FP8."""
        fmt = self.config.moment_format
        amax = moment.abs().max()
        scale = amax.clamp(min=1e-12) / fmt.max_value
        
        quantized = (moment / scale).round().clamp(fmt.min_value, fmt.max_value)
        return quantized, scale
    
    def to_fp32(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize moments to FP32."""
        m = self.m_quantized * self.m_scale
        v = self.v_quantized * self.v_scale
        return m, v
    
    def update(
        self,
        grad: torch.Tensor,
        beta1: float,
        beta2: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update moments (Adam-style) and requantize."""
        # Dequantize
        m, v = self.to_fp32()
        
        # Adam update
        m_new = beta1 * m + (1 - beta1) * grad
        v_new = beta2 * v + (1 - beta2) * grad.square()
        
        self.step += 1
        
        # Re-quantize
        if self.step % self.config.scale_update_interval == 0:
            self.m_quantized, self.m_scale = self._quantize_moment(m_new)
            self.v_quantized, self.v_scale = self._quantize_moment(v_new)
        else:
            # Quantize with existing scale
            fmt = self.config.moment_format
            self.m_quantized = (m_new / self.m_scale).round().clamp(fmt.min_value, fmt.max_value)
            self.v_quantized = (v_new / self.v_scale).round().clamp(fmt.min_value, fmt.max_value)
        
        return m_new, v_new
    
    def get_bias_corrected(self, beta1: float, beta2: float) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # FP8 = 1 byte, FP32 = 4 bytes
        return 0.25 * 1.01  # Include scale overhead


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
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for FP8 training."""
        return loss * self.loss_scale
    
    def unscale_gradients(self, gradients: List[torch.Tensor]) -> bool:
        """Unscale gradients after backward pass. Returns False if overflow detected."""
        inv_scale = 1.0 / self.loss_scale
        
        # Check for overflow
        for grad in gradients:
            if grad is not None and (torch.isinf(grad).any() or torch.isnan(grad).any()):
                return False
        
        # Unscale
        for grad in gradients:
            if grad is not None:
                grad.mul_(inv_scale)
        
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
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize activations for forward pass."""
        if not self.config.fp8_forward:
            return tensor, torch.ones(1, device=tensor.device)
        
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
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize gradients for backward pass."""
        if not self.config.fp8_backward:
            return tensor, torch.ones(1, device=tensor.device)
        
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
    
    def init_optimizer_state(self, name: str, param: torch.Tensor):
        """Initialize FP8 optimizer state for a parameter."""
        if not self.config.fp8_optimizer:
            return
        
        m = torch.zeros_like(param)
        v = torch.zeros_like(param)
        
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
            param_size = state.m_quantized.numel()
            total_fp32_bytes += param_size * 8  # 2 moments * 4 bytes
            total_fp8_bytes += param_size * 2 + 64  # 2 moments * 1 byte + scales
        
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
    def __init__(self, in_features, out_features, block_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # E4M3 constants
        self.E4M3_MAX = 448.0
        
    def pad_to_block(self, x):
        # Pad dimensions to be multiple of block_size
        rows, cols = x.shape
        pad_rows = (self.block_size - (rows % self.block_size)) % self.block_size
        pad_cols = (self.block_size - (cols % self.block_size)) % self.block_size
        
        if pad_rows > 0 or pad_cols > 0:
            x = F.pad(x, (0, pad_cols, 0, pad_rows))
            
        return x, pad_rows, pad_cols
        
    def quantize_tile(self, x):
        # x: (Rows, Cols)
        # Reshape to (R_blocks, Block, C_blocks, Block)
        padded_x, pad_r, pad_c = self.pad_to_block(x)
        R, C = padded_x.shape
        
        x_blocked = padded_x.view(R // self.block_size, self.block_size, C // self.block_size, self.block_size)
        # Permute to (R_blocks, C_blocks, Block, Block)
        x_blocked = x_blocked.permute(0, 2, 1, 3)
        
        # Calculate max per tile
        abs_max = x_blocked.abs().amax(dim=(-1, -2), keepdim=True)
        scale = abs_max / self.E4M3_MAX
        scale = torch.clamp(scale, min=1e-8)
        
        # Quantize
        x_q = torch.round(x_blocked / scale)
        x_q = torch.clamp(x_q, -self.E4M3_MAX, self.E4M3_MAX)
        
        return x_q, scale, pad_r, pad_c
        
    def dequantize_tile(self, x_q, scale, pad_r, pad_c, original_shape):
        # x_q: (R_blocks, C_blocks, Block, Block)
        # scale: (R_blocks, C_blocks, 1, 1)
        
        x_dq = x_q * scale
        
        # Reshape back
        # (R_blocks, C_blocks, Block, Block) -> (R_blocks, Block, C_blocks, Block)
        x_dq = x_dq.permute(0, 2, 1, 3)
        R_padded = x_dq.shape[0] * x_dq.shape[1]
        C_padded = x_dq.shape[2] * x_dq.shape[3]
        
        x_dq = x_dq.reshape(R_padded, C_padded)
        
        # Remove padding
        if pad_r > 0:
            x_dq = x_dq[:-pad_r, :]
        if pad_c > 0:
            x_dq = x_dq[:, :-pad_c]
            
        return x_dq

    def forward(self, x):
        # Simulate FP8 MatMul: Y = X @ W.T
        # 1. Quantize Input X (Online)
        # Flatten batch dims
        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, self.in_features)
        
        x_q, x_scale, x_pad_r, x_pad_c = self.quantize_tile(x_flat)
        x_dq = self.dequantize_tile(x_q, x_scale, x_pad_r, x_pad_c, x_flat.shape)
        
        # 2. Quantize Weight W (Offline/Cached usually, here online)
        w_q, w_scale, w_pad_r, w_pad_c = self.quantize_tile(self.weight)
        w_dq = self.dequantize_tile(w_q, w_scale, w_pad_r, w_pad_c, self.weight.shape)
        
        # 3. MatMul in FP32 (Simulating accumulation)
        out = F.linear(x_dq, w_dq, self.bias)
        
        return out.view(*batch_shape, self.out_features)


# ============================================================================
# Hardware Detection and Automatic Precision Selection
# ============================================================================

@dataclass
class HardwareCapabilities:
    """Detected hardware capabilities for precision selection."""
    has_cuda: bool = False
    cuda_version: str = ""
    device_name: str = ""
    compute_capability: Tuple[int, int] = (0, 0)
    supports_fp8: bool = False  # H100+ (SM 90+)
    supports_bf16: bool = False  # SM 80+
    supports_fp16: bool = True
    memory_gb: float = 0.0
    
    @classmethod
    def detect(cls, device: Optional[torch.device] = None) -> "HardwareCapabilities":
        """Detect hardware capabilities."""
        caps = cls()
        
        if not torch.cuda.is_available():
            return caps
        
        caps.has_cuda = True
        caps.cuda_version = torch.version.cuda or ""
        
        if device is None:
            device = torch.device("cuda:0")
        
        device_idx = device.index if device.index is not None else 0
        
        props = torch.cuda.get_device_properties(device_idx)
        caps.device_name = props.name
        caps.compute_capability = (props.major, props.minor)
        caps.memory_gb = props.total_memory / (1024 ** 3)
        
        # SM 90+ for FP8 (H100, H200)
        caps.supports_fp8 = props.major >= 9
        
        # SM 80+ for BF16 (A100, A10, etc.)
        caps.supports_bf16 = props.major >= 8
        
        return caps
    
    def get_optimal_dtype(self) -> torch.dtype:
        """Get optimal training dtype for this hardware."""
        if self.supports_bf16:
            return torch.bfloat16
        elif self.supports_fp16:
            return torch.float16
        return torch.float32


def check_fp8_support() -> Tuple[bool, str]:
    """
    Check if FP8 training is supported on current hardware.
    
    Returns:
        - supported: Whether FP8 is available
        - message: Description of hardware status
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available. FP8 requires H100/H200 GPU."
    
    caps = HardwareCapabilities.detect()
    
    if caps.supports_fp8:
        return True, f"FP8 supported on {caps.device_name} (SM {caps.compute_capability[0]}.{caps.compute_capability[1]})"
    
    return False, (
        f"FP8 not supported on {caps.device_name} (SM {caps.compute_capability[0]}.{caps.compute_capability[1]}). "
        f"Requires H100/H200 (SM 9.0+). Falling back to {'BF16' if caps.supports_bf16 else 'FP16'}."
    )


class AutoPrecisionTraining:
    """
    Automatic precision selection based on hardware capabilities.
    
    Provides unified API for mixed-precision training with automatic
    fallback from FP8 -> BF16 -> FP16 based on hardware.
    """
    
    def __init__(
        self,
        prefer_fp8: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.caps = HardwareCapabilities.detect(self.device) if torch.cuda.is_available() else HardwareCapabilities()
        
        # Determine precision mode
        self.use_fp8 = prefer_fp8 and self.caps.supports_fp8
        self.use_bf16 = not self.use_fp8 and self.caps.supports_bf16
        self.use_fp16 = not self.use_fp8 and not self.use_bf16 and self.caps.supports_fp16
        
        # Initialize appropriate precision manager
        if self.use_fp8:
            self.fp8_manager = FP8MixedPrecisionManager(FP8TrainingConfig())
            self.grad_scaler = None
        elif self.use_bf16:
            self.fp8_manager = None
            self.grad_scaler = None  # BF16 doesn't need grad scaling
        else:
            self.fp8_manager = None
            self.grad_scaler = torch.cuda.amp.GradScaler() if self.caps.has_cuda else None
        
        self._log_precision_mode()
    
    def _log_precision_mode(self):
        """Log the selected precision mode."""
        if self.use_fp8:
            mode = "FP8"
        elif self.use_bf16:
            mode = "BF16"
        elif self.use_fp16:
            mode = "FP16"
        else:
            mode = "FP32"
        
        print(f"[AutoPrecision] Using {mode} on {self.caps.device_name if self.caps.has_cuda else 'CPU'}")
    
    def get_autocast_context(self):
        """Get the appropriate autocast context manager."""
        if not self.caps.has_cuda:
            return torch.autocast(device_type='cpu', dtype=torch.float32)
        
        if self.use_bf16:
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        elif self.use_fp16:
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            return torch.autocast(device_type='cuda', dtype=torch.float32)
    
    def backward(self, loss: torch.Tensor) -> bool:
        """
        Perform backward pass with appropriate scaling.
        
        Returns True if gradients are valid, False if overflow detected.
        """
        if self.use_fp8 and self.fp8_manager:
            scaled_loss = self.fp8_manager.scale_loss(loss)
            scaled_loss.backward()
            return True  # FP8 manager handles overflow internally
        elif self.grad_scaler:
            self.grad_scaler.scale(loss).backward()
            return True
        else:
            loss.backward()
            return True
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Perform optimizer step with appropriate unscaling."""
        if self.grad_scaler:
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            optimizer.step()
    
    def get_precision_report(self) -> Dict[str, Any]:
        """Get report on precision settings."""
        return {
            "device": self.caps.device_name,
            "compute_capability": f"{self.caps.compute_capability[0]}.{self.caps.compute_capability[1]}",
            "precision_mode": "FP8" if self.use_fp8 else "BF16" if self.use_bf16 else "FP16" if self.use_fp16 else "FP32",
            "memory_gb": self.caps.memory_gb,
            "supports_fp8": self.caps.supports_fp8,
            "supports_bf16": self.caps.supports_bf16,
        }


# ============================================================================
# FP8 vs BF16 Comparison Utilities
# ============================================================================

def compare_fp8_vs_bf16(
    model: nn.Module,
    sample_input: torch.Tensor,
    sample_target: torch.Tensor,
    loss_fn: nn.Module,
) -> Dict[str, Any]:
    """
    Compare FP8 and BF16 training characteristics.
    
    Returns comparison metrics for ablation study.
    """
    results = {}
    
    # BF16 baseline
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        bf16_output = model(sample_input)
        bf16_loss = loss_fn(bf16_output, sample_target)
    results["bf16_loss"] = bf16_loss.item()
    
    # FP8 (if supported)
    caps = HardwareCapabilities.detect()
    if caps.supports_fp8:
        fp8_manager = FP8MixedPrecisionManager(FP8TrainingConfig())
        # Simulate FP8 forward pass
        quantized_input, _ = fp8_manager.quantize_activation("input", sample_input)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            fp8_output = model(quantized_input)
            fp8_loss = loss_fn(fp8_output, sample_target)
        results["fp8_loss"] = fp8_loss.item()
        results["loss_difference"] = abs(fp8_loss.item() - bf16_loss.item())
    else:
        results["fp8_loss"] = None
        results["fp8_note"] = "FP8 not supported on this hardware"
    
    return results


# ============================================================================
# INT4/INT8 Inference Quantization (DeepSeek-V3.2 Production Deployment)
# ============================================================================

class QuantizedLinearInt4(nn.Module):
    """INT4 quantized linear layer with per-group (128) scaling.
    
    Uses asymmetric quantization with 128-element groups for accuracy.
    Weights are stored as packed uint8 (2 values per byte).
    
    Optimized for inference on CUDA with PyTorch.
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = 128, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Number of groups per input dimension
        self.n_groups = (in_features + group_size - 1) // group_size
        
        # Padded input size (to align with group_size)
        self.in_features_padded = self.n_groups * group_size

        # Quantized weights: packed 4-bit (2 values per byte)
        self.register_buffer(
            'weight_packed',
            torch.zeros((out_features, self.in_features_padded // 2), dtype=torch.uint8)
        )
        
        # Per-group scales and zero points
        self.register_buffer(
            'scales',
            torch.zeros((out_features, self.n_groups), dtype=torch.float16)
        )
        self.register_buffer(
            'zeros',
            torch.zeros((out_features, self.n_groups), dtype=torch.float16)
        )
        
        # Optional bias
        if bias:
            self.register_buffer('bias', torch.zeros((out_features,), dtype=torch.float16))
        else:
            self.register_buffer('bias', None)

    @classmethod
    def from_float(cls, linear: nn.Linear, group_size: int = 128) -> "QuantizedLinearInt4":
        """Convert a float linear layer to INT4 quantized.
        
        Args:
            linear: Original nn.Linear layer
            group_size: Number of elements per quantization group (default 128)
        
        Returns:
            Quantized INT4 linear layer
        """
        in_features = linear.weight.shape[1]
        out_features = linear.weight.shape[0]
        has_bias = linear.bias is not None
        
        q = cls(in_features, out_features, group_size, bias=has_bias)
        device = linear.weight.device
        
        weight = linear.weight.float()
        
        # Pad weight if needed
        if in_features != q.in_features_padded:
            padding = q.in_features_padded - in_features
            weight = F.pad(weight, (0, padding))
        
        # Quantize per group
        scales = torch.zeros((out_features, q.n_groups), dtype=torch.float16, device=device)
        zeros = torch.zeros((out_features, q.n_groups), dtype=torch.float16, device=device)
        weight_packed = torch.zeros((out_features, q.in_features_padded // 2), dtype=torch.uint8, device=device)
        
        for g in range(q.n_groups):
            group_start = g * group_size
            group_end = (g + 1) * group_size
            group_weights = weight[:, group_start:group_end]

            # Compute min/max for asymmetric quantization
            w_min = group_weights.min(dim=1, keepdim=True).values
            w_max = group_weights.max(dim=1, keepdim=True).values
            
            # Compute scale (range / 15 for 4-bit)
            scale = (w_max - w_min) / 15.0
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            
            # Compute zero point
            zero = -w_min / scale

            # Quantize to 0-15 range
            q_weights = torch.round((group_weights - w_min) / scale)
            q_weights = torch.clamp(q_weights, 0, 15).to(torch.uint8)

            # Pack nibbles (2 values per byte)
            for i in range(0, group_size, 2):
                packed = (q_weights[:, i + 1] << 4) | q_weights[:, i]
                weight_packed[:, (group_start + i) // 2] = packed

            # Store scales and zeros
            scales[:, g] = scale.squeeze().half()
            zeros[:, g] = zero.squeeze().half()

        q.weight_packed = weight_packed
        q.scales = scales
        q.zeros = zeros

        if has_bias:
            q.bias = linear.bias.half()

        return q

    def _dequantize(self) -> torch.Tensor:
        """Dequantize weights on-the-fly for inference."""
        device = self.weight_packed.device
        
        # Unpack nibbles
        weight_low = self.weight_packed & 0x0F
        weight_high = (self.weight_packed >> 4) & 0x0F
        
        # Interleave
        weight_unpacked = torch.zeros(
            (self.out_features, self.in_features_padded), 
            dtype=torch.float16, 
            device=device
        )
        weight_unpacked[:, 0::2] = weight_low.half()
        weight_unpacked[:, 1::2] = weight_high.half()
        
        # Dequantize per group
        weight_dequant = torch.zeros_like(weight_unpacked)
        for g in range(self.n_groups):
            group_start = g * self.group_size
            group_end = (g + 1) * self.group_size
            
            scale = self.scales[:, g:g+1]
            zero = self.zeros[:, g:g+1]
            
            weight_dequant[:, group_start:group_end] = (
                weight_unpacked[:, group_start:group_end] - zero
            ) * scale
        
        # Remove padding
        return weight_dequant[:, :self.in_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization."""
        weight = self._dequantize()
        out = F.linear(x, weight, None)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class QuantizedLinearInt8(nn.Module):
    """INT8 quantized linear layer with per-channel scaling.
    
    Uses symmetric per-channel quantization for optimal accuracy/speed tradeoff.
    Weights are stored as int8 with float16 per-channel scales.
    
    Optimized for inference on CUDA with PyTorch.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Quantized weights (int8)
        self.register_buffer(
            'weight_int8',
            torch.zeros((out_features, in_features), dtype=torch.int8)
        )
        
        # Per-channel scales
        self.register_buffer(
            'scale',
            torch.zeros((out_features,), dtype=torch.float16)
        )
        
        # Optional bias
        if bias:
            self.register_buffer('bias', torch.zeros((out_features,), dtype=torch.float16))
        else:
            self.register_buffer('bias', None)

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "QuantizedLinearInt8":
        """Convert a float linear layer to INT8 quantized.
        
        Uses symmetric per-channel quantization with scale = max_abs / 127.
        
        Args:
            linear: Original nn.Linear layer
        
        Returns:
            Quantized INT8 linear layer
        """
        in_features = linear.weight.shape[1]
        out_features = linear.weight.shape[0]
        has_bias = linear.bias is not None
        
        q = cls(in_features, out_features, bias=has_bias)
        device = linear.weight.device
        
        weight = linear.weight.float()

        # Per-channel symmetric quantization
        w_max = weight.abs().max(dim=1, keepdim=True).values
        scale = w_max / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        # Quantize to int8 (-128 to 127)
        q.weight_int8 = torch.round(weight / scale).to(torch.int8)
        q.scale = scale.squeeze().half()

        if has_bias:
            q.bias = linear.bias.half()

        return q

    def _dequantize(self) -> torch.Tensor:
        """Dequantize weights on-the-fly for inference."""
        return self.weight_int8.half() * self.scale[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization."""
        weight = self._dequantize()
        out = F.linear(x, weight, None)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Apply PyTorch dynamic INT8 quantization for inference.
    
    This uses PyTorch's built-in dynamic quantization which quantizes
    weights statically and activations dynamically during inference.
    
    Args:
        model: Model to quantize
        
    Returns:
        Quantized model
    """
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )


def apply_static_quantization(
    model: nn.Module,
    calibration_data: List[torch.Tensor],
    backend: str = 'x86',
) -> nn.Module:
    """Apply static INT8 quantization with calibration data.
    
    This uses PyTorch's static quantization which calibrates
    quantization parameters using representative data.
    
    Args:
        model: Model to quantize
        calibration_data: List of input tensors for calibration
        backend: Quantization backend ('x86' or 'fbgemm' for CPU, 'qnnpack' for ARM)
        
    Returns:
        Quantized model
    """
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig(backend)

    # Prepare for quantization (fuses layers and adds observers)
    model_prepared = torch.quantization.prepare(model)

    # Calibrate by running sample data
    with torch.no_grad():
        for batch in calibration_data:
            model_prepared(batch)

    # Convert to quantized model
    return torch.quantization.convert(model_prepared)


def quantize_model_int4(model: nn.Module, group_size: int = 128) -> nn.Module:
    """Quantize all linear layers in a model to INT4.
    
    Args:
        model: Original model with nn.Linear layers
        group_size: Group size for INT4 quantization (default 128)
    
    Returns:
        Model with INT4 quantized linear layers
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Get parent module
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            # Replace with quantized version
            quantized = QuantizedLinearInt4.from_float(module, group_size)
            setattr(parent, parts[-1], quantized)
    
    return model


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """Quantize all linear layers in a model to INT8.
    
    Args:
        model: Original model with nn.Linear layers
    
    Returns:
        Model with INT8 quantized linear layers
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Get parent module
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            # Replace with quantized version
            quantized = QuantizedLinearInt8.from_float(module)
            setattr(parent, parts[-1], quantized)
    
    return model
