//! Ablation Study Module for DeepSeek-From-Scratch
//!
//! This module provides ablation study infrastructure for comparing
//! different architectural choices and configurations:
//!
//! - RoPE scaling strategies (Standard, Linear, NTK-aware, YaRN, Dynamic)
//! - Attention mechanisms (MLA vs GQA vs MHA)
//! - Expert configurations for MoE
//! - Load balancing strategies
//! - Multi-token prediction configurations
//! - Precision/quantization comparisons
//!
//! Paper Experiments (Section 4.3):
//! - A1: Rust vs PyTorch-MPS Backend Comparison
//! - A2: Zero-copy vs Serialized Tensor Interop
//! - A3: Metal SIMD vs Naive Kernel Implementation
//! - A4: Heterogeneous vs Homogeneous Cluster Cost
//! - A5: MLA Latent Dimension Pareto Frontier
//! - A6: Bias-update vs Auxiliary-loss Load Balancing

pub mod rope_ablation;
pub mod attention_ablation;
pub mod moe_ablation;
pub mod paper_experiments;

pub use rope_ablation::*;
pub use attention_ablation::*;
pub use moe_ablation::*;
pub use paper_experiments::*;
