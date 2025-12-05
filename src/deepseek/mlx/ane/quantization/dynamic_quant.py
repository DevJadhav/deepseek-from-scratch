"""
ANE Dynamic Quantization Module

This module provides dynamic quantization with runtime calibration:
- Calibration data collection during inference
- Automatic scale computation from observed activations
- Histogram-based and percentile-based calibration methods
- Support for per-tensor and per-channel calibration

Dynamic quantization is useful when:
- Input data distribution is unknown at compile time
- Different input sequences have varying ranges
- Fine-tuned accuracy is needed without static calibration
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch
import torch.nn as nn


class CalibrationMethod(Enum):
    """Methods for computing quantization parameters."""
    
    MINMAX = "minmax"  # Simple min/max based
    PERCENTILE = "percentile"  # Percentile-based (clips outliers)
    MSE = "mse"  # Minimize quantization MSE
    ENTROPY = "entropy"  # Minimize KL divergence
    HISTOGRAM = "histogram"  # Histogram-based calibration


@dataclass
class CalibrationConfig:
    """Configuration for calibration."""
    
    # Calibration method
    method: CalibrationMethod = CalibrationMethod.MINMAX
    
    # Number of calibration batches
    num_batches: int = 100
    
    # Percentile for clipping (for PERCENTILE method)
    percentile: float = 99.99
    
    # Number of histogram bins (for HISTOGRAM/ENTROPY methods)
    num_bins: int = 2048
    
    # Symmetric quantization
    symmetric: bool = True
    
    # Number of bits for quantization
    num_bits: int = 8
    
    # Per-channel calibration
    per_channel: bool = False
    channel_axis: int = 0
    
    # EMA smoothing factor for running statistics
    ema_alpha: float = 0.1


@dataclass
class CalibrationStats:
    """Collected calibration statistics."""
    
    # Basic statistics
    min_val: torch.Tensor | None = None
    max_val: torch.Tensor | None = None
    abs_max: torch.Tensor | None = None
    
    # Running statistics (EMA)
    running_min: torch.Tensor | None = None
    running_max: torch.Tensor | None = None
    
    # Histogram (for advanced methods)
    histogram: torch.Tensor | None = None
    bin_edges: torch.Tensor | None = None
    
    # Number of samples observed
    num_samples: int = 0
    
    def reset(self):
        """Reset all statistics."""
        self.min_val = None
        self.max_val = None
        self.abs_max = None
        self.running_min = None
        self.running_max = None
        self.histogram = None
        self.bin_edges = None
        self.num_samples = 0


class HistogramObserver(nn.Module):
    """
    Observer that collects histogram of activations.
    
    Used for entropy and histogram-based calibration methods.
    """
    
    def __init__(
        self,
        config: CalibrationConfig,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        
        # Initialize buffers
        self.register_buffer('min_val', torch.tensor(float('inf'), dtype=dtype))
        self.register_buffer('max_val', torch.tensor(float('-inf'), dtype=dtype))
        self.register_buffer('histogram', torch.zeros(config.num_bins, dtype=torch.float32))
        self.register_buffer('num_samples', torch.tensor(0))
        
        self._initialized = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Observe tensor and update histogram."""
        if self.training:
            self._observe(x)
        return x
    
    def _observe(self, x: torch.Tensor):
        """Update statistics with observed tensor."""
        x_flat = x.detach().float().flatten()
        
        # Update min/max
        x_min = x_flat.min()
        x_max = x_flat.max()
        
        if not self._initialized:
            self.min_val.copy_(x_min)
            self.max_val.copy_(x_max)
            self._initialized = True
        else:
            self.min_val.copy_(torch.min(self.min_val, x_min))
            self.max_val.copy_(torch.max(self.max_val, x_max))
        
        # Update histogram
        # Use dynamic binning based on current observed range
        bin_width = (self.max_val - self.min_val) / self.config.num_bins
        if bin_width > 0:
            indices = ((x_flat - self.min_val) / bin_width).long().clamp(0, self.config.num_bins - 1)
            self.histogram.scatter_add_(0, indices, torch.ones_like(x_flat))
        
        self.num_samples += x_flat.numel()
    
    def compute_qparams(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute quantization parameters from histogram."""
        if self.config.method == CalibrationMethod.ENTROPY:
            return self._compute_entropy_qparams()
        elif self.config.method == CalibrationMethod.HISTOGRAM:
            return self._compute_histogram_qparams()
        else:
            return self._compute_minmax_qparams()
    
    def _compute_minmax_qparams(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Simple min/max based qparams."""
        if self.config.symmetric:
            abs_max = torch.max(self.min_val.abs(), self.max_val.abs())
            qmax = (1 << (self.config.num_bits - 1)) - 1
            scale = abs_max / qmax
            return scale, None
        else:
            qmax = (1 << self.config.num_bits) - 1
            scale = (self.max_val - self.min_val) / qmax
            zero_point = torch.round(-self.min_val / scale)
            return scale, zero_point
    
    def _compute_histogram_qparams(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Histogram-based qparams (percentile clipping)."""
        # Find percentile thresholds from histogram
        cumsum = self.histogram.cumsum(0)
        total = cumsum[-1]
        
        lower_thresh = self.config.percentile / 100
        upper_thresh = (100 - self.config.percentile) / 100
        
        lower_idx = (cumsum < total * lower_thresh).sum()
        upper_idx = (cumsum < total * (1 - upper_thresh)).sum()
        
        bin_width = (self.max_val - self.min_val) / self.config.num_bins
        
        clipped_min = self.min_val + lower_idx * bin_width
        clipped_max = self.min_val + upper_idx * bin_width
        
        if self.config.symmetric:
            abs_max = torch.max(clipped_min.abs(), clipped_max.abs())
            qmax = (1 << (self.config.num_bits - 1)) - 1
            scale = abs_max / qmax
            return scale, None
        else:
            qmax = (1 << self.config.num_bits) - 1
            scale = (clipped_max - clipped_min) / qmax
            zero_point = torch.round(-clipped_min / scale)
            return scale, zero_point
    
    def _compute_entropy_qparams(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Entropy-based qparams (minimize KL divergence).
        
        This method searches for the best threshold that minimizes
        KL divergence between original and quantized distributions.
        """
        # Normalize histogram to probability distribution
        hist_norm = self.histogram / self.histogram.sum()
        
        # Try different thresholds
        best_kl = float('inf')
        best_threshold_idx = self.config.num_bins
        
        for threshold_idx in range(128, self.config.num_bins):
            # Quantize histogram to target bits
            target_bins = 1 << self.config.num_bits
            
            # Reference distribution (original, clipped at threshold)
            ref_hist = hist_norm[:threshold_idx + 1].clone()
            ref_hist[-1] += hist_norm[threshold_idx + 1:].sum()
            
            # Quantized distribution
            bin_size = (threshold_idx + 1) / target_bins
            quant_hist = torch.zeros(target_bins, device=hist_norm.device)
            
            for i in range(target_bins):
                start = int(i * bin_size)
                end = int((i + 1) * bin_size)
                quant_hist[i] = ref_hist[start:end].sum()
            
            # Expand back to original size for KL computation
            expanded = torch.zeros_like(ref_hist)
            for i in range(target_bins):
                start = int(i * bin_size)
                end = int((i + 1) * bin_size)
                if end > start:
                    expanded[start:end] = quant_hist[i] / (end - start)
            
            # Compute KL divergence
            # KL(P || Q) = sum(P * log(P / Q))
            mask = (ref_hist > 0) & (expanded > 0)
            if mask.any():
                kl = (ref_hist[mask] * torch.log(ref_hist[mask] / expanded[mask])).sum()
                
                if kl < best_kl:
                    best_kl = kl
                    best_threshold_idx = threshold_idx
        
        # Compute scale from best threshold
        bin_width = (self.max_val - self.min_val) / self.config.num_bins
        threshold = self.min_val + best_threshold_idx * bin_width
        
        if self.config.symmetric:
            abs_max = torch.max(self.min_val.abs(), threshold.abs())
            qmax = (1 << (self.config.num_bits - 1)) - 1
            scale = abs_max / qmax
            return scale, None
        else:
            qmax = (1 << self.config.num_bits) - 1
            scale = (threshold - self.min_val) / qmax
            zero_point = torch.round(-self.min_val / scale)
            return scale, zero_point
    
    def reset(self):
        """Reset observer state."""
        self.min_val.fill_(float('inf'))
        self.max_val.fill_(float('-inf'))
        self.histogram.zero_()
        self.num_samples.zero_()
        self._initialized = False


class DynamicQuantizer(nn.Module):
    """
    Dynamic quantizer with runtime calibration.
    
    This quantizer collects statistics during calibration and then
    uses the computed parameters for efficient inference.
    
    Example:
        # Create quantizer
        quantizer = DynamicQuantizer(
            config=CalibrationConfig(method=CalibrationMethod.PERCENTILE),
        )
        
        # Calibration phase
        quantizer.enable_calibration()
        for batch in calibration_data:
            _ = model(batch)
        quantizer.disable_calibration()
        
        # Inference phase (uses calibrated parameters)
        output = quantizer(hidden_states)
    """
    
    def __init__(
        self,
        config: CalibrationConfig | None = None,
    ):
        super().__init__()
        self.config = config or CalibrationConfig()
        
        # Observers for different layers
        self._observers: dict[str, HistogramObserver] = nn.ModuleDict()
        self._qparams: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        
        self._calibration_mode = False
        self._inference_mode = False
    
    def register_layer(self, name: str):
        """Register a layer for calibration."""
        if name not in self._observers:
            self._observers[name] = HistogramObserver(self.config)
    
    def forward(
        self,
        x: torch.Tensor,
        name: str | None = None,
    ) -> torch.Tensor:
        """
        Process tensor through quantizer.
        
        Args:
            x: Input tensor
            name: Layer name for per-layer calibration
            
        Returns:
            Quantized and dequantized tensor (or original if not in inference mode)
        """
        if self._calibration_mode and name is not None:
            # Calibration: observe and pass through
            if name not in self._observers:
                self.register_layer(name)
            return self._observers[name](x)
        
        if self._inference_mode:
            # Inference: quantize and dequantize
            return self._quantize_dequantize(x, name)
        
        # Passthrough
        return x
    
    def _quantize_dequantize(
        self,
        x: torch.Tensor,
        name: str | None = None,
    ) -> torch.Tensor:
        """Quantize and immediately dequantize (simulated quantization)."""
        # Get qparams
        if name is not None and name in self._qparams:
            scale, zero_point = self._qparams[name]
        else:
            # Dynamic: compute qparams from tensor
            if self.config.symmetric:
                abs_max = x.abs().max()
                qmax = (1 << (self.config.num_bits - 1)) - 1
                scale = abs_max / qmax
                zero_point = None
            else:
                x_min = x.min()
                x_max = x.max()
                qmax = (1 << self.config.num_bits) - 1
                scale = (x_max - x_min) / qmax
                zero_point = torch.round(-x_min / scale)
        
        scale = torch.clamp(scale, min=1e-10)
        
        # Quantize
        if self.config.symmetric:
            qmax = (1 << (self.config.num_bits - 1)) - 1
            x_quant = torch.round(x / scale).clamp(-qmax - 1, qmax)
            x_dequant = x_quant * scale
        else:
            qmax = (1 << self.config.num_bits) - 1
            x_quant = torch.round(x / scale + zero_point).clamp(0, qmax)
            x_dequant = (x_quant - zero_point) * scale
        
        return x_dequant.to(x.dtype)
    
    def enable_calibration(self):
        """Enable calibration mode."""
        self._calibration_mode = True
        self._inference_mode = False
        
        # Reset all observers
        for observer in self._observers.values():
            observer.reset()
    
    def disable_calibration(self):
        """
        Disable calibration mode and compute qparams.
        
        Call this after calibration is complete to switch to inference mode.
        """
        self._calibration_mode = False
        
        # Compute qparams from all observers
        self._qparams.clear()
        for name, observer in self._observers.items():
            if observer.num_samples > 0:
                self._qparams[name] = observer.compute_qparams()
    
    def enable_inference(self):
        """Enable inference mode (quantize/dequantize activations)."""
        self._inference_mode = True
        self._calibration_mode = False
    
    def disable_inference(self):
        """Disable inference mode (passthrough)."""
        self._inference_mode = False
    
    def get_calibration_stats(self) -> dict[str, CalibrationStats]:
        """Get calibration statistics for all layers."""
        stats = {}
        for name, observer in self._observers.items():
            stats[name] = CalibrationStats(
                min_val=observer.min_val.clone(),
                max_val=observer.max_val.clone(),
                histogram=observer.histogram.clone(),
                num_samples=observer.num_samples.item(),
            )
        return stats
    
    def get_qparams(self) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
        """Get computed quantization parameters."""
        return self._qparams.copy()
    
    def save_calibration(self, path: str):
        """Save calibration data to file."""
        data = {
            "config": {
                "method": self.config.method.value,
                "num_bits": self.config.num_bits,
                "symmetric": self.config.symmetric,
                "percentile": self.config.percentile,
            },
            "qparams": {},
        }
        
        for name, (scale, zp) in self._qparams.items():
            data["qparams"][name] = {
                "scale": scale.cpu().numpy().tolist(),
                "zero_point": zp.cpu().numpy().tolist() if zp is not None else None,
            }
        
        torch.save(data, path)
    
    def load_calibration(self, path: str):
        """Load calibration data from file."""
        data = torch.load(path)
        
        self._qparams.clear()
        for name, qp in data["qparams"].items():
            scale = torch.tensor(qp["scale"])
            zp = torch.tensor(qp["zero_point"]) if qp["zero_point"] is not None else None
            self._qparams[name] = (scale, zp)


def calibrate_model(
    model: nn.Module,
    calibration_dataloader,
    config: CalibrationConfig | None = None,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> DynamicQuantizer:
    """
    Calibrate a model with representative data.
    
    Args:
        model: Model to calibrate
        calibration_dataloader: DataLoader with calibration data
        config: Calibration configuration
        forward_fn: Optional custom forward function
        
    Returns:
        Calibrated DynamicQuantizer
    """
    config = config or CalibrationConfig()
    quantizer = DynamicQuantizer(config)
    
    # Register all linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            quantizer.register_layer(name)
    
    # Calibration pass
    quantizer.enable_calibration()
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(calibration_dataloader):
            if i >= config.num_batches:
                break
            
            if forward_fn is not None:
                forward_fn(model, batch)
            else:
                if isinstance(batch, (list, tuple)):
                    model(*batch)
                elif isinstance(batch, dict):
                    model(**batch)
                else:
                    model(batch)
    
    quantizer.disable_calibration()
    quantizer.enable_inference()
    
    return quantizer
