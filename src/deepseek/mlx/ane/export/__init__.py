"""
ANE Export Module

This module provides utilities for exporting ANE-optimized models to various
formats including CoreML for Apple Neural Engine deployment.

Components:
- ANEExportConfig: Configuration for model export
- CoreMLExporter: Export models to Core ML format
- Optimization passes: Op fusion, constant folding, quantization
- Validation utilities: Numeric validation between PyTorch and CoreML

Usage:
    from ane_impl.export import (
        ANEExportConfig,
        CoreMLExporter,
        validate_coreml_export,
    )
    
    # Export model to CoreML
    config = ANEExportConfig()
    exporter = CoreMLExporter(model, config)
    exporter.export("model.mlpackage")
"""

from .coreml_export import (
    ANEExportConfig,
    ComputeUnit,
    CoreMLExporter,
    CoreMLInference,
    CoreMLOptimizationConfig,
    validate_coreml_export,
)

__all__ = [
    # Config
    "ANEExportConfig",
    "ComputeUnit",
    "CoreMLOptimizationConfig",
    # Exporter
    "CoreMLExporter",
    "CoreMLInference",
    # Validation
    "validate_coreml_export",
]
