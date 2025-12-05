"""
ANE Mixed Precision Module

This module provides mixed precision management for ANE inference:
- Layer-wise precision assignment (FP16 for critical, INT8/INT4 for others)
- Automatic layer classification (attention vs FFN vs embeddings)
- Configuration presets for different accuracy/efficiency trade-offs

Mixed Precision Strategy (as per production_hardening.md):
- Attention projections: INT8 per-channel weights, FP16 activations
- FFN/Expert weights: INT4 block-wise, FP16 activations
- Embedding: FP16 (CPU-side)
- LayerNorm weights: FP16
- Critical computations: FP16 (softmax, attention scores)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from .weight_quant import WeightQuantConfig, WeightQuantType
from .activation_quant import ActivationQuantConfig, ActivationQuantType


class LayerType(Enum):
    """Types of layers for mixed precision assignment."""
    
    EMBEDDING = "embedding"
    ATTENTION_QKV = "attention_qkv"
    ATTENTION_OUT = "attention_out"
    FFN_GATE = "ffn_gate"
    FFN_UP = "ffn_up"
    FFN_DOWN = "ffn_down"
    EXPERT = "expert"
    LAYERNORM = "layernorm"
    LM_HEAD = "lm_head"
    OTHER = "other"


@dataclass
class LayerPrecision:
    """Precision configuration for a layer."""
    
    # Weight precision
    weight_quant: WeightQuantConfig
    
    # Activation precision
    activation_quant: ActivationQuantConfig
    
    # Whether this is a critical layer (affects accuracy significantly)
    is_critical: bool = False
    
    # Layer type
    layer_type: LayerType = LayerType.OTHER


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision inference."""
    
    # Default precisions by layer type
    embedding_precision: LayerPrecision | None = None
    attention_precision: LayerPrecision | None = None
    ffn_precision: LayerPrecision | None = None
    expert_precision: LayerPrecision | None = None
    layernorm_precision: LayerPrecision | None = None
    lm_head_precision: LayerPrecision | None = None
    
    # Global settings
    default_weight_quant: WeightQuantType = WeightQuantType.INT8_PER_CHANNEL
    default_activation_quant: ActivationQuantType = ActivationQuantType.FP16
    
    # Critical layer handling
    keep_critical_fp16: bool = True
    
    # Expert-specific settings
    expert_weight_quant: WeightQuantType = WeightQuantType.INT4_PER_BLOCK
    expert_block_size: int = 128
    
    def __post_init__(self):
        """Initialize default precisions if not provided."""
        if self.embedding_precision is None:
            self.embedding_precision = LayerPrecision(
                weight_quant=WeightQuantConfig(quant_type=WeightQuantType.NONE),
                activation_quant=ActivationQuantConfig(quant_type=ActivationQuantType.FP16),
                is_critical=True,
                layer_type=LayerType.EMBEDDING,
            )
        
        if self.attention_precision is None:
            self.attention_precision = LayerPrecision(
                weight_quant=WeightQuantConfig(quant_type=WeightQuantType.INT8_PER_CHANNEL),
                activation_quant=ActivationQuantConfig(quant_type=ActivationQuantType.FP16),
                is_critical=True,
                layer_type=LayerType.ATTENTION_QKV,
            )
        
        if self.ffn_precision is None:
            self.ffn_precision = LayerPrecision(
                weight_quant=WeightQuantConfig(
                    quant_type=WeightQuantType.INT4_PER_BLOCK,
                    block_size=128,
                ),
                activation_quant=ActivationQuantConfig(quant_type=ActivationQuantType.FP16),
                is_critical=False,
                layer_type=LayerType.FFN_GATE,
            )
        
        if self.expert_precision is None:
            self.expert_precision = LayerPrecision(
                weight_quant=WeightQuantConfig(
                    quant_type=self.expert_weight_quant,
                    block_size=self.expert_block_size,
                ),
                activation_quant=ActivationQuantConfig(quant_type=ActivationQuantType.FP16),
                is_critical=False,
                layer_type=LayerType.EXPERT,
            )
        
        if self.layernorm_precision is None:
            self.layernorm_precision = LayerPrecision(
                weight_quant=WeightQuantConfig(quant_type=WeightQuantType.NONE),
                activation_quant=ActivationQuantConfig(quant_type=ActivationQuantType.FP16),
                is_critical=True,
                layer_type=LayerType.LAYERNORM,
            )
        
        if self.lm_head_precision is None:
            self.lm_head_precision = LayerPrecision(
                weight_quant=WeightQuantConfig(quant_type=WeightQuantType.INT8_PER_CHANNEL),
                activation_quant=ActivationQuantConfig(quant_type=ActivationQuantType.FP16),
                is_critical=True,
                layer_type=LayerType.LM_HEAD,
            )
    
    @classmethod
    def default(cls) -> MixedPrecisionConfig:
        """Create default mixed precision config."""
        return cls()
    
    @classmethod
    def accuracy_focused(cls) -> MixedPrecisionConfig:
        """Create accuracy-focused config (more FP16, less quantization)."""
        return cls(
            default_weight_quant=WeightQuantType.INT8_PER_CHANNEL,
            default_activation_quant=ActivationQuantType.FP16,
            expert_weight_quant=WeightQuantType.INT8_PER_CHANNEL,
            keep_critical_fp16=True,
        )
    
    @classmethod
    def efficiency_focused(cls) -> MixedPrecisionConfig:
        """Create efficiency-focused config (more INT4/INT8)."""
        return cls(
            default_weight_quant=WeightQuantType.INT4_PER_BLOCK,
            default_activation_quant=ActivationQuantType.INT8_PER_TENSOR,
            expert_weight_quant=WeightQuantType.INT4_PER_BLOCK,
            keep_critical_fp16=False,
        )


class MixedPrecisionManager:
    """
    Manager for mixed precision inference on ANE.
    
    This class handles:
    - Automatic layer type classification
    - Precision assignment based on configuration
    - Weight/activation quantization coordination
    
    Example:
        config = MixedPrecisionConfig.default()
        manager = MixedPrecisionManager(config)
        
        # Get precision for a layer
        precision = manager.get_layer_precision("model.layers.0.self_attn.q_proj")
        
        # Apply quantization to model
        manager.quantize_model(model)
    """
    
    def __init__(self, config: MixedPrecisionConfig | None = None):
        """
        Initialize mixed precision manager.
        
        Args:
            config: Mixed precision configuration
        """
        self.config = config or MixedPrecisionConfig.default()
        
        # Layer name patterns for classification
        self._patterns = {
            LayerType.EMBEDDING: ["embed", "embedding", "wte", "wpe"],
            LayerType.ATTENTION_QKV: ["q_proj", "k_proj", "v_proj", "qkv", "query", "key", "value"],
            LayerType.ATTENTION_OUT: ["o_proj", "out_proj", "output"],
            LayerType.FFN_GATE: ["gate", "w1", "gate_proj"],
            LayerType.FFN_UP: ["up", "w3", "up_proj", "fc1"],
            LayerType.FFN_DOWN: ["down", "w2", "down_proj", "fc2"],
            LayerType.EXPERT: ["expert", "moe"],
            LayerType.LAYERNORM: ["norm", "layernorm", "ln", "rms"],
            LayerType.LM_HEAD: ["lm_head", "head", "output_proj"],
        }
    
    def classify_layer(self, name: str) -> LayerType:
        """
        Classify a layer by its name.
        
        Args:
            name: Layer name (e.g., "model.layers.0.self_attn.q_proj")
            
        Returns:
            LayerType classification
        """
        name_lower = name.lower()
        
        # Check patterns in priority order
        for layer_type, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern in name_lower:
                    return layer_type
        
        return LayerType.OTHER
    
    def get_layer_precision(self, name: str) -> LayerPrecision:
        """
        Get precision configuration for a layer.
        
        Args:
            name: Layer name
            
        Returns:
            LayerPrecision configuration
        """
        layer_type = self.classify_layer(name)
        
        if layer_type == LayerType.EMBEDDING:
            return self.config.embedding_precision
        elif layer_type in (LayerType.ATTENTION_QKV, LayerType.ATTENTION_OUT):
            return self.config.attention_precision
        elif layer_type in (LayerType.FFN_GATE, LayerType.FFN_UP, LayerType.FFN_DOWN):
            return self.config.ffn_precision
        elif layer_type == LayerType.EXPERT:
            return self.config.expert_precision
        elif layer_type == LayerType.LAYERNORM:
            return self.config.layernorm_precision
        elif layer_type == LayerType.LM_HEAD:
            return self.config.lm_head_precision
        else:
            # Default precision
            return LayerPrecision(
                weight_quant=WeightQuantConfig(quant_type=self.config.default_weight_quant),
                activation_quant=ActivationQuantConfig(quant_type=self.config.default_activation_quant),
                is_critical=False,
                layer_type=LayerType.OTHER,
            )
    
    def should_quantize_weights(self, name: str) -> bool:
        """Check if layer weights should be quantized."""
        precision = self.get_layer_precision(name)
        
        if self.config.keep_critical_fp16 and precision.is_critical:
            if precision.layer_type in (LayerType.EMBEDDING, LayerType.LAYERNORM):
                return False
        
        return precision.weight_quant.quant_type != WeightQuantType.NONE
    
    def should_quantize_activations(self, name: str) -> bool:
        """Check if layer activations should be quantized."""
        precision = self.get_layer_precision(name)
        return precision.activation_quant.quant_type != ActivationQuantType.FP16
    
    def get_model_precision_summary(self, model: nn.Module) -> dict:
        """
        Get precision summary for all model layers.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with precision assignments
        """
        summary = {
            "by_type": {},
            "layers": [],
        }
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                precision = self.get_layer_precision(name)
                layer_type = precision.layer_type.value
                
                if layer_type not in summary["by_type"]:
                    summary["by_type"][layer_type] = {
                        "count": 0,
                        "weight_quant": precision.weight_quant.quant_type.value,
                        "activation_quant": precision.activation_quant.quant_type.value,
                    }
                summary["by_type"][layer_type]["count"] += 1
                
                summary["layers"].append({
                    "name": name,
                    "type": layer_type,
                    "weight_quant": precision.weight_quant.quant_type.value,
                    "activation_quant": precision.activation_quant.quant_type.value,
                    "is_critical": precision.is_critical,
                })
        
        return summary
    
    def estimate_memory_savings(
        self,
        model: nn.Module,
        baseline_dtype: torch.dtype = torch.float16,
    ) -> dict:
        """
        Estimate memory savings from mixed precision.
        
        Args:
            model: PyTorch model
            baseline_dtype: Baseline data type for comparison
            
        Returns:
            Dictionary with memory estimates
        """
        baseline_bytes = baseline_dtype.itemsize if hasattr(baseline_dtype, 'itemsize') else 2
        
        estimates = {
            "baseline_bytes": 0,
            "quantized_bytes": 0,
            "by_type": {},
        }
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight
                numel = weight.numel()
                original_bytes = numel * baseline_bytes
                
                precision = self.get_layer_precision(name)
                wq = precision.weight_quant.quant_type
                
                if wq == WeightQuantType.NONE:
                    quant_bytes = original_bytes
                elif wq in (WeightQuantType.INT8_PER_CHANNEL, WeightQuantType.INT8_PER_TENSOR):
                    # INT8 + scale overhead
                    quant_bytes = numel + weight.shape[0] * 4  # INT8 data + FP32 scales
                elif wq in (WeightQuantType.INT4_PER_BLOCK, WeightQuantType.INT4_PER_CHANNEL):
                    # INT4 packed + scale overhead
                    block_size = precision.weight_quant.block_size
                    num_blocks = (numel + block_size - 1) // block_size
                    quant_bytes = numel // 2 + num_blocks * 4  # Packed INT4 + FP32 scales
                else:
                    quant_bytes = original_bytes
                
                estimates["baseline_bytes"] += original_bytes
                estimates["quantized_bytes"] += quant_bytes
                
                layer_type = precision.layer_type.value
                if layer_type not in estimates["by_type"]:
                    estimates["by_type"][layer_type] = {"baseline": 0, "quantized": 0}
                estimates["by_type"][layer_type]["baseline"] += original_bytes
                estimates["by_type"][layer_type]["quantized"] += quant_bytes
        
        if estimates["quantized_bytes"] > 0:
            estimates["compression_ratio"] = estimates["baseline_bytes"] / estimates["quantized_bytes"]
            estimates["savings_percent"] = (1 - estimates["quantized_bytes"] / estimates["baseline_bytes"]) * 100
        else:
            estimates["compression_ratio"] = 1.0
            estimates["savings_percent"] = 0.0
        
        return estimates
