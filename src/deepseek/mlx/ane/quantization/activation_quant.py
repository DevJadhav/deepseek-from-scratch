"""
ANE Activation Quantization Module

This module provides activation quantization optimized for Apple Neural Engine:
- FP16 activations (default for ANE precision)
- INT8 activations for non-critical paths
- Per-tensor and per-channel quantization
- Observer patterns for calibration

ANE prefers FP16 activations for numerical stability, with INT8 available
for throughput optimization on non-critical computation paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class ActivationQuantType(Enum):
    """Activation quantization types for ANE."""
    
    FP16 = "fp16"  # Default for ANE
    INT8_PER_TENSOR = "int8_per_tensor"  # Per-tensor INT8
    INT8_PER_CHANNEL = "int8_per_channel"  # Per-channel INT8
    DYNAMIC = "dynamic"  # Dynamic quantization (runtime calibration)


@dataclass
class ActivationQuantConfig:
    """Configuration for activation quantization."""
    
    # Quantization type
    quant_type: ActivationQuantType = ActivationQuantType.FP16
    
    # Symmetric quantization (for INT8)
    symmetric: bool = True
    
    # Per-channel axis (for per-channel INT8)
    channel_axis: int = -1
    
    # Observer for statistics
    use_observer: bool = True
    observer_epochs: int = 100
    
    # Clipping percentile for outliers
    clip_percentile: float = 99.99
    
    # Enable/disable quantization (for toggling)
    enabled: bool = True


@dataclass
class ActivationStats:
    """Statistics collected for activation quantization."""
    
    min_val: float = float('inf')
    max_val: float = float('-inf')
    abs_max: float = 0.0
    running_mean: float = 0.0
    running_var: float = 1.0
    num_batches: int = 0
    
    def update(self, x: torch.Tensor):
        """Update statistics with new activation tensor."""
        with torch.no_grad():
            x_float = x.float()
            
            self.min_val = min(self.min_val, x_float.min().item())
            self.max_val = max(self.max_val, x_float.max().item())
            self.abs_max = max(self.abs_max, x_float.abs().max().item())
            
            # Running mean and variance
            batch_mean = x_float.mean().item()
            batch_var = x_float.var().item()
            
            if self.num_batches == 0:
                self.running_mean = batch_mean
                self.running_var = batch_var
            else:
                # Exponential moving average
                alpha = 0.1
                self.running_mean = (1 - alpha) * self.running_mean + alpha * batch_mean
                self.running_var = (1 - alpha) * self.running_var + alpha * batch_var
            
            self.num_batches += 1


@dataclass
class QuantizedActivation:
    """Container for quantized activation tensor."""
    
    # Quantized data
    data: torch.Tensor
    
    # Scale factor(s)
    scale: torch.Tensor
    
    # Zero point (None for symmetric or FP16)
    zero_point: torch.Tensor | None
    
    # Original shape
    original_shape: tuple[int, ...]
    
    # Quantization type used
    quant_type: ActivationQuantType
    
    @property
    def is_fp16(self) -> bool:
        """Check if this is FP16 (no quantization)."""
        return self.quant_type == ActivationQuantType.FP16


def quantize_activation_fp16(
    x: torch.Tensor,
) -> QuantizedActivation:
    """
    Convert activation to FP16.
    
    This is the default for ANE which prefers FP16 activations
    for numerical precision while still being efficient.
    
    Args:
        x: Input activation tensor
        
    Returns:
        QuantizedActivation with FP16 data
    """
    return QuantizedActivation(
        data=x.to(torch.float16),
        scale=torch.ones(1, device=x.device),
        zero_point=None,
        original_shape=x.shape,
        quant_type=ActivationQuantType.FP16,
    )


def quantize_activation_int8(
    x: torch.Tensor,
    symmetric: bool = True,
    per_channel: bool = False,
    channel_axis: int = -1,
    scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
) -> QuantizedActivation:
    """
    Quantize activation to INT8.
    
    For symmetric quantization:
        scale = max(|x|) / 127
        x_quant = round(x / scale)
    
    For asymmetric quantization:
        scale = (max(x) - min(x)) / 255
        zero_point = round(-min(x) / scale)
        x_quant = round(x / scale + zero_point)
    
    Args:
        x: Input activation tensor
        symmetric: Use symmetric quantization (default True)
        per_channel: Use per-channel quantization (default False)
        channel_axis: Channel axis for per-channel (default -1)
        scale: Pre-computed scale (optional, for calibrated quantization)
        zero_point: Pre-computed zero point (optional)
        
    Returns:
        QuantizedActivation with INT8 data
    """
    original_shape = x.shape
    device = x.device
    
    if scale is None:
        # Compute scale dynamically
        if per_channel:
            # Per-channel quantization
            if channel_axis < 0:
                channel_axis = x.ndim + channel_axis
            
            # Move channel axis to last for easier processing
            dims = list(range(x.ndim))
            dims.remove(channel_axis)
            x_permuted = x.permute(dims + [channel_axis])
            x_flat = x_permuted.reshape(-1, x.shape[channel_axis])
            
            if symmetric:
                qmax = 127
                abs_max = x_flat.abs().max(dim=0).values
                computed_scale = abs_max / qmax
                computed_scale = torch.clamp(computed_scale, min=1e-10)
                computed_zp = None
            else:
                qmax = 255
                x_min = x_flat.min(dim=0).values
                x_max = x_flat.max(dim=0).values
                computed_scale = (x_max - x_min) / qmax
                computed_scale = torch.clamp(computed_scale, min=1e-10)
                computed_zp = torch.round(-x_min / computed_scale).to(torch.int8)
            
            scale = computed_scale
            zero_point = computed_zp
        else:
            # Per-tensor quantization
            if symmetric:
                qmax = 127
                abs_max = x.abs().max()
                computed_scale = abs_max / qmax
                computed_scale = torch.clamp(computed_scale, min=torch.tensor(1e-10, device=device))
                computed_zp = None
            else:
                qmax = 255
                x_min = x.min()
                x_max = x.max()
                computed_scale = (x_max - x_min) / qmax
                computed_scale = torch.clamp(computed_scale, min=torch.tensor(1e-10, device=device))
                computed_zp = torch.round(-x_min / computed_scale).to(torch.int8)
            
            scale = computed_scale.reshape(1)
            zero_point = computed_zp.reshape(1) if computed_zp is not None else None
    
    # Quantize
    if per_channel:
        # Reshape scale for broadcasting
        scale_shape = [1] * x.ndim
        scale_shape[channel_axis] = scale.shape[0]
        scale_broadcast = scale.reshape(scale_shape)
        
        if symmetric:
            x_quant = torch.round(x / scale_broadcast).clamp(-128, 127).to(torch.int8)
        else:
            zp_broadcast = zero_point.float().reshape(scale_shape)
            x_quant = torch.round(x / scale_broadcast + zp_broadcast).clamp(0, 255).to(torch.uint8)
    else:
        if symmetric:
            x_quant = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
        else:
            x_quant = torch.round(x / scale + zero_point.float()).clamp(0, 255).to(torch.uint8)
    
    quant_type = ActivationQuantType.INT8_PER_CHANNEL if per_channel else ActivationQuantType.INT8_PER_TENSOR
    
    return QuantizedActivation(
        data=x_quant,
        scale=scale,
        zero_point=zero_point,
        original_shape=original_shape,
        quant_type=quant_type,
    )


def dequantize_activation(
    quantized: QuantizedActivation,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Dequantize activation tensor.
    
    Args:
        quantized: QuantizedActivation to dequantize
        output_dtype: Output data type (default FP16)
        
    Returns:
        Dequantized activation tensor
    """
    if quantized.is_fp16:
        return quantized.data.to(output_dtype)
    
    x_quant = quantized.data.float()
    scale = quantized.scale
    
    if quantized.quant_type == ActivationQuantType.INT8_PER_CHANNEL:
        # Find the channel axis (axis with non-1 scale size)
        for axis, size in enumerate(quantized.original_shape):
            if scale.numel() == size:
                channel_axis = axis
                break
        else:
            channel_axis = -1
        
        # Reshape scale for broadcasting
        scale_shape = [1] * len(quantized.original_shape)
        scale_shape[channel_axis] = scale.shape[0]
        scale_broadcast = scale.reshape(scale_shape)
        
        if quantized.zero_point is not None:
            zp_broadcast = quantized.zero_point.float().reshape(scale_shape)
            x_dequant = (x_quant - zp_broadcast) * scale_broadcast
        else:
            x_dequant = x_quant * scale_broadcast
    else:
        # Per-tensor
        if quantized.zero_point is not None:
            x_dequant = (x_quant - quantized.zero_point.float()) * scale
        else:
            x_dequant = x_quant * scale
    
    return x_dequant.to(output_dtype)


class ActivationObserver(nn.Module):
    """
    Observer module for collecting activation statistics.
    
    Used during calibration to determine optimal quantization parameters.
    """
    
    def __init__(
        self,
        config: ActivationQuantConfig,
        averaging_constant: float = 0.01,
    ):
        super().__init__()
        self.config = config
        self.averaging_constant = averaging_constant
        
        # Statistics
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        self.register_buffer('running_min', torch.tensor(float('inf')))
        self.register_buffer('running_max', torch.tensor(float('-inf')))
        self.register_buffer('num_batches', torch.tensor(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Observe activation and update statistics."""
        if self.training:
            with torch.no_grad():
                x_float = x.float()
                min_val = x_float.min()
                max_val = x_float.max()
                
                # Update global min/max
                self.min_val = torch.min(self.min_val, min_val)
                self.max_val = torch.max(self.max_val, max_val)
                
                # Update running min/max with EMA
                if self.num_batches == 0:
                    self.running_min = min_val
                    self.running_max = max_val
                else:
                    self.running_min = self.running_min + self.averaging_constant * (min_val - self.running_min)
                    self.running_max = self.running_max + self.averaging_constant * (max_val - self.running_max)
                
                self.num_batches = self.num_batches + 1
        
        return x
    
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Calculate quantization parameters from observed statistics."""
        if self.config.symmetric:
            abs_max = max(abs(self.running_min.item()), abs(self.running_max.item()))
            scale = torch.tensor(abs_max / 127, dtype=torch.float32)
            zero_point = None
        else:
            scale = (self.running_max - self.running_min) / 255
            zero_point = torch.round(-self.running_min / scale).to(torch.int8)
        
        return scale, zero_point
    
    def reset(self):
        """Reset observer statistics."""
        self.min_val.fill_(float('inf'))
        self.max_val.fill_(float('-inf'))
        self.running_min.fill_(float('inf'))
        self.running_max.fill_(float('-inf'))
        self.num_batches.zero_()


class ANEActivationQuantizer(nn.Module):
    """
    ANE-optimized activation quantizer.
    
    Default behavior:
    - FP16 activations for critical paths (attention, final output)
    - INT8 activations for non-critical paths (FFN intermediate)
    
    Example:
        quantizer = ANEActivationQuantizer(
            config=ActivationQuantConfig(quant_type=ActivationQuantType.FP16),
        )
        
        # Quantize activations
        quant_act = quantizer.quantize(hidden_states)
        
        # Dequantize for next layer
        hidden_states = quantizer.dequantize(quant_act)
    """
    
    def __init__(
        self,
        config: ActivationQuantConfig | None = None,
        critical_config: ActivationQuantConfig | None = None,
        non_critical_config: ActivationQuantConfig | None = None,
    ):
        """
        Initialize activation quantizer.
        
        Args:
            config: Default configuration
            critical_config: Config for critical paths (default: FP16)
            non_critical_config: Config for non-critical paths (default: INT8)
        """
        super().__init__()
        
        self.config = config or ActivationQuantConfig(
            quant_type=ActivationQuantType.FP16,
        )
        
        self.critical_config = critical_config or ActivationQuantConfig(
            quant_type=ActivationQuantType.FP16,
        )
        
        self.non_critical_config = non_critical_config or ActivationQuantConfig(
            quant_type=ActivationQuantType.INT8_PER_TENSOR,
            symmetric=True,
        )
        
        # Observers for calibration
        self._observers: dict[str, ActivationObserver] = {}
        self._calibration_mode = False
        
        # Cached quantization parameters
        self._qparams: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
    
    def quantize(
        self,
        x: torch.Tensor,
        name: str | None = None,
        config: ActivationQuantConfig | None = None,
    ) -> QuantizedActivation:
        """
        Quantize an activation tensor.
        
        Args:
            x: Activation tensor to quantize
            name: Optional name for caching/calibration
            config: Optional config (uses default if None)
            
        Returns:
            QuantizedActivation
        """
        if config is None:
            config = self.config
        
        if not config.enabled:
            return quantize_activation_fp16(x)
        
        # Calibration mode: observe and pass through
        if self._calibration_mode and name is not None:
            if name not in self._observers:
                self._observers[name] = ActivationObserver(config)
            self._observers[name](x)
            return quantize_activation_fp16(x)
        
        # Get cached qparams if available
        scale, zero_point = None, None
        if name is not None and name in self._qparams:
            scale, zero_point = self._qparams[name]
        
        if config.quant_type == ActivationQuantType.FP16:
            return quantize_activation_fp16(x)
        elif config.quant_type == ActivationQuantType.INT8_PER_TENSOR:
            return quantize_activation_int8(
                x, symmetric=config.symmetric, per_channel=False,
                scale=scale, zero_point=zero_point,
            )
        elif config.quant_type == ActivationQuantType.INT8_PER_CHANNEL:
            return quantize_activation_int8(
                x, symmetric=config.symmetric, per_channel=True,
                channel_axis=config.channel_axis,
                scale=scale, zero_point=zero_point,
            )
        elif config.quant_type == ActivationQuantType.DYNAMIC:
            # Dynamic: always recompute scale
            return quantize_activation_int8(x, symmetric=config.symmetric)
        else:
            raise ValueError(f"Unsupported quantization type: {config.quant_type}")
    
    def dequantize(
        self,
        quantized: QuantizedActivation,
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Dequantize an activation tensor.
        
        Args:
            quantized: QuantizedActivation to dequantize
            output_dtype: Output data type (default FP16)
            
        Returns:
            Dequantized activation tensor
        """
        return dequantize_activation(quantized, output_dtype)
    
    def quantize_critical(self, x: torch.Tensor, name: str | None = None) -> QuantizedActivation:
        """Quantize critical path activation (FP16)."""
        return self.quantize(x, name=name, config=self.critical_config)
    
    def quantize_non_critical(self, x: torch.Tensor, name: str | None = None) -> QuantizedActivation:
        """Quantize non-critical path activation (INT8)."""
        return self.quantize(x, name=name, config=self.non_critical_config)
    
    def enable_calibration(self):
        """Enable calibration mode to collect statistics."""
        self._calibration_mode = True
        self._observers.clear()
    
    def disable_calibration(self):
        """Disable calibration mode and compute qparams."""
        self._calibration_mode = False
        
        # Compute qparams from observers
        for name, observer in self._observers.items():
            self._qparams[name] = observer.calculate_qparams()
    
    def get_calibrated_params(self) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
        """Get calibrated quantization parameters."""
        return self._qparams.copy()
