"""
ANE-Optimized Mixture of Experts (MoE) Module

This module provides ANE-optimized MoE implementations with:
- Expert fusion for small expert counts (< 16)
- Batched processing for medium counts (16-64)
- Distillation support for large counts (256)
- FP16 computation and INT8 weights
- Pre-computed routing for ANE constraints
"""

from .expert import (
    ANEExpert,
    ANEExpertConfig,
    ANESharedExpert,
    ANEExpertGroup,
    ANEFusedExpert,
    ActivationType,
)
from .router import (
    ANERouter,
    ANERouterConfig,
    ANEHierarchicalRouter,
    ANEHierarchicalRouterConfig,
    RoutingStrategy,
)
from .moe import (
    ANEMoE,
    ANEMoEConfig,
    ANEMoEFused,
    ANEMoEBatched,
    ANEMoEHierarchical,
    MoEStrategy,
    ExpertDistillation,
)

__all__ = [
    # Expert types
    "ANEExpert",
    "ANEExpertConfig",
    "ANESharedExpert",
    "ANEExpertGroup",
    "ANEFusedExpert",
    "ActivationType",
    # Router types
    "ANERouter",
    "ANERouterConfig",
    "ANEHierarchicalRouter",
    "ANEHierarchicalRouterConfig",
    "RoutingStrategy",
    # MoE layers
    "ANEMoE",
    "ANEMoEConfig",
    "ANEMoEFused",
    "ANEMoEBatched",
    "ANEMoEHierarchical",
    "MoEStrategy",
    "ExpertDistillation",
]
