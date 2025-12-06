//! DeepSeek Rust - High-performance implementation of DeepSeek components
//!
//! This library provides:
//! - Multi-head Latent Attention (MLA)
//! - DeepSeek MoE with expert selection
//! - FP8 quantization
//! - CUDA Hopper optimizations
//! - Zero-copy Python interop via PyO3

// Allow dead code and unused variables during development
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
// Allow clippy warnings that are common in this codebase
#![allow(clippy::redundant_closure)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::map_clone)]
#![allow(clippy::new_without_default)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::vec_box)]
#![allow(clippy::missing_const_for_thread_local)]
#![allow(clippy::manual_c_str_literals)]

pub mod model;
pub mod training;
pub mod utils;
pub mod benchmarks;
pub mod distributed;
pub mod ablation;

#[cfg(feature = "pyo3-bindings")]
pub mod pyo3_bindings;

#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;

/// DeepSeek Rust Python Module
///
/// This module provides high-performance Rust implementations of DeepSeek
/// components with zero-copy Python interop via PyO3.
#[cfg(feature = "pyo3-bindings")]
#[pymodule]
fn deepseek_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register all PyO3 bindings
    pyo3_bindings::register_bindings(m)?;

    // Add module version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "DeepSeek Rust implementation with zero-copy Python interop")?;

    Ok(())
}
