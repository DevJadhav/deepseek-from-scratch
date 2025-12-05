"""
CoreML Export Pipeline for ANE-Optimized Models

This module provides a complete pipeline for converting ANE-optimized
PyTorch models to Core ML format for deployment on Apple Neural Engine.

Key Features:
- PyTorch → TorchScript → Core ML conversion
- Automatic optimization passes (op fusion, constant folding)
- Compute unit selection (ANE, CPU, GPU, or ALL)
- Model validation with numeric tolerance checks
- .mlpackage export with metadata

Usage:
    from ane_impl.export import CoreMLExporter, ANEExportConfig
    from ane_impl.model import ANEDeepSeekModel, ANEDeepSeekConfig
    
    # Create model
    model_config = ANEDeepSeekConfig.tiny()
    model = ANEDeepSeekModel(model_config)
    
    # Export to CoreML
    export_config = ANEExportConfig(
        sequence_length=128,
        compute_units=ComputeUnit.ALL,
    )
    exporter = CoreMLExporter(model, export_config)
    exporter.export("deepseek_ane.mlpackage")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class ComputeUnit(Enum):
    """Target compute units for CoreML model execution."""
    
    ALL = "ALL"  # ANE + CPU + GPU (let CoreML decide)
    CPU_AND_GPU = "CPU_AND_GPU"  # Exclude ANE
    CPU_AND_NE = "CPU_AND_NE"  # CPU + ANE (Neural Engine)
    CPU_ONLY = "CPU_ONLY"  # CPU only
    
    def to_coreml(self) -> str:
        """Convert to coremltools compute unit string."""
        mapping = {
            ComputeUnit.ALL: "ALL",
            ComputeUnit.CPU_AND_GPU: "CPU_AND_GPU",
            ComputeUnit.CPU_AND_NE: "CPU_AND_NE",
            ComputeUnit.CPU_ONLY: "CPU_ONLY",
        }
        return mapping[self]


@dataclass
class CoreMLOptimizationConfig:
    """Configuration for CoreML optimization passes."""
    
    # Op fusion
    fuse_matmul_add: bool = True
    fuse_conv_bn: bool = True
    fuse_linear_bn: bool = True
    fuse_gelu: bool = True
    fuse_swish: bool = True
    
    # Constant folding
    fold_constants: bool = True
    
    # Quantization (applied during conversion)
    quantize_weights: bool = False  # INT8 weight quantization
    quantize_activations: bool = False  # INT8 activation quantization
    
    # Memory optimization
    optimize_memory: bool = True
    
    # Numerical precision
    float16_inference: bool = True  # Use FP16 for ANE


@dataclass
class ANEExportConfig:
    """Configuration for ANE model export to CoreML."""
    
    # Input shapes
    batch_size: int = 1
    sequence_length: int = 128  # Fixed for ANE (multiples of 16 preferred)
    
    # Compute target
    compute_units: ComputeUnit = ComputeUnit.ALL
    
    # Model metadata
    model_name: str = "DeepSeekANE"
    model_description: str = "DeepSeek model optimized for Apple Neural Engine"
    model_author: str = "ANE Implementation"
    model_version: str = "1.0"
    
    # Optimization
    optimization: CoreMLOptimizationConfig | None = None
    
    # Validation
    validate_numerics: bool = True
    numeric_tolerance: float = 1e-3  # Tolerance for validation
    
    # Output
    include_metadata: bool = True
    include_tokenizer: bool = False  # Include tokenizer config
    
    def __post_init__(self):
        if self.optimization is None:
            self.optimization = CoreMLOptimizationConfig()


class TracingWrapper(nn.Module):
    """
    Wrapper for tracing ANE models with fixed shapes.
    
    This wrapper handles:
    - Fixed sequence length (required for ANE)
    - Removed KV cache (full recompute for tracing)
    - Simplified interface for TorchScript
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ANEExportConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward for tracing.
        
        Args:
            input_ids: Token indices [batch, seq_len]
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        # Forward without cache (full recompute)
        logits, _ = self.model(
            input_ids,
            past_key_values=None,
            use_cache=False,
        )
        return logits


class CoreMLExporter:
    """
    Export ANE-optimized PyTorch models to Core ML format.
    
    The export pipeline:
    1. Wrap model for tracing (fixed shapes, no cache)
    2. Trace with torch.jit.trace
    3. Convert to CoreML with coremltools
    4. Apply optimization passes
    5. Validate numerics (optional)
    6. Package as .mlpackage
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ANEExportConfig,
    ):
        self.model = model
        self.config = config
        self._traced_model: torch.jit.ScriptModule | None = None
        self._coreml_model: Any | None = None  # ct.models.MLModel
        
    def trace(self) -> torch.jit.ScriptModule:
        """
        Trace the model with sample inputs.
        
        Returns:
            TorchScript traced model
        """
        # Create tracing wrapper
        wrapper = TracingWrapper(self.model, self.config)
        wrapper.eval()
        
        # Create sample input
        sample_input = torch.randint(
            0, 32000,  # Assume vocab size < 32000 for sample
            (self.config.batch_size, self.config.sequence_length),
            dtype=torch.long,
        )
        
        # Trace the model
        with torch.no_grad():
            self._traced_model = torch.jit.trace(
                wrapper,
                (sample_input,),
                strict=False,
            )
        
        return self._traced_model
    
    def convert_to_coreml(self) -> Any:
        """
        Convert traced model to CoreML format.
        
        Returns:
            CoreML model (ct.models.MLModel)
        """
        # Check if coremltools is available
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError(
                "coremltools is required for CoreML export. "
                "Install with: pip install coremltools"
            )
        
        # Trace if not already done
        if self._traced_model is None:
            self.trace()
        
        # Define input type
        input_shape = ct.Shape(
            shape=(
                self.config.batch_size,
                self.config.sequence_length,
            )
        )
        
        # Convert to CoreML
        self._coreml_model = ct.convert(
            self._traced_model,
            inputs=[
                ct.TensorType(
                    name="input_ids",
                    shape=input_shape,
                    dtype=int,
                ),
            ],
            outputs=[
                ct.TensorType(name="logits"),
            ],
            convert_to="mlprogram",
            compute_units=self._get_compute_units(),
            minimum_deployment_target=ct.target.iOS16,
        )
        
        # Apply optimizations
        self._apply_optimizations()
        
        # Add metadata
        if self.config.include_metadata:
            self._add_metadata()
        
        return self._coreml_model
    
    def _get_compute_units(self) -> Any:
        """Get CoreML compute units enum."""
        try:
            import coremltools as ct
            
            mapping = {
                ComputeUnit.ALL: ct.ComputeUnit.ALL,
                ComputeUnit.CPU_AND_GPU: ct.ComputeUnit.CPU_AND_GPU,
                ComputeUnit.CPU_AND_NE: ct.ComputeUnit.CPU_AND_NE,
                ComputeUnit.CPU_ONLY: ct.ComputeUnit.CPU_ONLY,
            }
            return mapping[self.config.compute_units]
        except ImportError:
            return None
    
    def _apply_optimizations(self):
        """Apply CoreML optimization passes."""
        if self._coreml_model is None:
            return
            
        try:
            from coremltools.optimize.coreml import (
                OpLinearQuantizerConfig,
                OptimizationConfig,
                linear_quantize_weights,
            )
            
            opt_config = self.config.optimization
            
            # Weight quantization
            if opt_config.quantize_weights:
                op_config = OpLinearQuantizerConfig(mode="linear_symmetric")
                config = OptimizationConfig(global_config=op_config)
                self._coreml_model = linear_quantize_weights(
                    self._coreml_model,
                    config=config,
                )
        except ImportError:
            # Optimization tools not available
            warnings.warn(
                "CoreML optimization tools not available. "
                "Skipping weight quantization."
            )
    
    def _add_metadata(self):
        """Add metadata to CoreML model."""
        if self._coreml_model is None:
            return
            
        self._coreml_model.short_description = self.config.model_description
        self._coreml_model.author = self.config.model_author
        self._coreml_model.version = self.config.model_version
        
        # Add custom metadata
        metadata = {
            "model_type": "causal_lm",
            "architecture": "deepseek_ane",
            "batch_size": str(self.config.batch_size),
            "sequence_length": str(self.config.sequence_length),
            "compute_units": self.config.compute_units.value,
        }
        
        # Try to get model config info
        if hasattr(self.model, 'config'):
            model_config = self.model.config
            metadata.update({
                "vocab_size": str(getattr(model_config, 'vocab_size', 'unknown')),
                "num_layers": str(getattr(model_config, 'num_layers', 'unknown')),
                "d_model": str(getattr(model_config, 'd_model', 'unknown')),
                "num_heads": str(getattr(model_config, 'num_heads', 'unknown')),
            })
        
        self._coreml_model.user_defined_metadata.update(metadata)
    
    def validate(self) -> dict:
        """
        Validate CoreML model against PyTorch model.
        
        Returns:
            Dictionary with validation results
        """
        if self._coreml_model is None:
            raise RuntimeError("Must convert to CoreML first")
        
        try:
            import coremltools as ct
            import numpy as np
        except ImportError:
            warnings.warn("Cannot validate without coremltools and numpy")
            return {"status": "skipped", "reason": "missing dependencies"}
        
        # Create test input
        test_input = torch.randint(
            0, 32000,
            (self.config.batch_size, self.config.sequence_length),
            dtype=torch.long,
        )
        
        # PyTorch forward
        self.model.eval()
        with torch.no_grad():
            wrapper = TracingWrapper(self.model, self.config)
            pt_output = wrapper(test_input)
            pt_output = pt_output.numpy()
        
        # CoreML forward
        coreml_input = {"input_ids": test_input.numpy().astype(np.int32)}
        coreml_output = self._coreml_model.predict(coreml_input)
        coreml_logits = coreml_output["logits"]
        
        # Compare outputs
        max_diff = np.max(np.abs(pt_output - coreml_logits))
        mean_diff = np.mean(np.abs(pt_output - coreml_logits))
        
        passed = max_diff < self.config.numeric_tolerance
        
        return {
            "status": "passed" if passed else "failed",
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
            "tolerance": self.config.numeric_tolerance,
            "output_shape": list(pt_output.shape),
        }
    
    def export(
        self,
        output_path: str | Path,
        validate: bool | None = None,
    ) -> dict:
        """
        Complete export pipeline: trace, convert, validate, save.
        
        Args:
            output_path: Path for .mlpackage output
            validate: Whether to validate (default: use config setting)
            
        Returns:
            Export results dictionary
        """
        output_path = Path(output_path)
        
        results = {
            "output_path": str(output_path),
            "config": {
                "batch_size": self.config.batch_size,
                "sequence_length": self.config.sequence_length,
                "compute_units": self.config.compute_units.value,
            },
        }
        
        # Step 1: Trace
        self.trace()
        results["tracing"] = "success"
        
        # Step 2: Convert
        self.convert_to_coreml()
        results["conversion"] = "success"
        
        # Step 3: Validate (optional)
        should_validate = validate if validate is not None else self.config.validate_numerics
        if should_validate:
            try:
                validation = self.validate()
                results["validation"] = validation
            except Exception as e:
                results["validation"] = {"status": "error", "error": str(e)}
        
        # Step 4: Save
        try:
            self._coreml_model.save(str(output_path))
            results["save"] = "success"
        except Exception as e:
            results["save"] = {"status": "error", "error": str(e)}
        
        return results
    
    def get_model_info(self) -> dict:
        """Get information about the CoreML model."""
        if self._coreml_model is None:
            return {"error": "No CoreML model available"}
        
        try:
            spec = self._coreml_model.get_spec()
            
            return {
                "description": spec.description.ShortDescription,
                "inputs": [
                    {
                        "name": inp.name,
                        "type": str(inp.type),
                    }
                    for inp in spec.description.input
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "type": str(out.type),
                    }
                    for out in spec.description.output
                ],
                "metadata": dict(self._coreml_model.user_defined_metadata),
            }
        except Exception as e:
            return {"error": str(e)}


def validate_coreml_export(
    pytorch_model: nn.Module,
    coreml_path: str | Path,
    test_inputs: torch.Tensor | None = None,
    tolerance: float = 1e-3,
) -> dict:
    """
    Validate a CoreML export against its PyTorch source.
    
    Args:
        pytorch_model: Original PyTorch model
        coreml_path: Path to .mlpackage
        test_inputs: Optional test inputs (will generate random if None)
        tolerance: Numeric tolerance for comparison
        
    Returns:
        Validation results dictionary
    """
    try:
        import coremltools as ct
        import numpy as np
    except ImportError:
        return {"status": "error", "reason": "missing coremltools or numpy"}
    
    # Load CoreML model
    coreml_model = ct.models.MLModel(str(coreml_path))
    
    # Get input shape from model
    spec = coreml_model.get_spec()
    input_spec = spec.description.input[0]
    
    # Generate test input if not provided
    if test_inputs is None:
        # Try to get shape from model spec
        if hasattr(input_spec.type, "multiArrayType"):
            shape = list(input_spec.type.multiArrayType.shape)
        else:
            shape = [1, 128]  # Default
        
        test_inputs = torch.randint(0, 32000, shape, dtype=torch.long)
    
    # PyTorch forward
    pytorch_model.eval()
    with torch.no_grad():
        pt_output, _ = pytorch_model(test_inputs, use_cache=False)
        pt_output = pt_output.numpy()
    
    # CoreML forward
    coreml_input = {input_spec.name: test_inputs.numpy().astype(np.int32)}
    coreml_output = coreml_model.predict(coreml_input)
    coreml_logits = list(coreml_output.values())[0]
    
    # Compare
    max_diff = np.max(np.abs(pt_output - coreml_logits))
    mean_diff = np.mean(np.abs(pt_output - coreml_logits))
    
    return {
        "status": "passed" if max_diff < tolerance else "failed",
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "tolerance": tolerance,
        "pytorch_shape": list(pt_output.shape),
        "coreml_shape": list(coreml_logits.shape),
    }


class CoreMLInference:
    """
    Helper class for running inference with exported CoreML models.
    
    This provides a simple interface for loading and running CoreML models
    that mirrors the PyTorch model interface.
    """
    
    def __init__(self, model_path: str | Path):
        """
        Load a CoreML model from .mlpackage.
        
        Args:
            model_path: Path to .mlpackage file
        """
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError("coremltools required for CoreML inference")
        
        self.model_path = Path(model_path)
        self.model = ct.models.MLModel(str(model_path))
        self._metadata = dict(self.model.user_defined_metadata)
        
    @property
    def metadata(self) -> dict:
        """Get model metadata."""
        return self._metadata
    
    @property
    def sequence_length(self) -> int:
        """Get expected sequence length from metadata."""
        return int(self._metadata.get("sequence_length", 128))
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from metadata."""
        return int(self._metadata.get("vocab_size", 32000))
    
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run inference.
        
        Args:
            input_ids: Token indices [batch, seq_len]
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        import numpy as np
        
        # Validate input shape
        if input_ids.shape[1] != self.sequence_length:
            # Pad or truncate to expected length
            current_len = input_ids.shape[1]
            if current_len < self.sequence_length:
                # Pad with zeros
                padding = torch.zeros(
                    input_ids.shape[0],
                    self.sequence_length - current_len,
                    dtype=input_ids.dtype,
                )
                input_ids = torch.cat([input_ids, padding], dim=1)
            else:
                # Truncate
                input_ids = input_ids[:, :self.sequence_length]
        
        # Convert to numpy
        input_array = input_ids.numpy().astype(np.int32)
        
        # Run inference
        output = self.model.predict({"input_ids": input_array})
        
        # Convert back to torch
        logits = torch.from_numpy(output["logits"])
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: int = 1,
    ) -> torch.Tensor:
        """
        Simple autoregressive generation.
        
        Note: This is a simple implementation. For production, consider
        using speculative decoding or other optimizations.
        
        Args:
            input_ids: Initial tokens [batch, seq_len]
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            eos_token_id: End of sequence token
            
        Returns:
            generated_ids: Generated token sequence [batch, new_len]
        """
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get logits for current sequence
            # Note: Full recompute each step (no KV cache in CoreML export)
            if generated.shape[1] > self.sequence_length:
                # Use last sequence_length tokens
                current_input = generated[:, -self.sequence_length:]
            else:
                current_input = generated
            
            logits = self(current_input)
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k = min(top_k, next_logits.shape[-1])
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check EOS
            if (next_token == eos_token_id).all():
                break
        
        return generated
