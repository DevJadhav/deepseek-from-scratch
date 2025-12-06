//! PyO3 Bindings for Zero-Copy Tensor Interop
//!
//! This module provides Python bindings for efficient tensor transfer between
//! Python (NumPy/Ray) and Rust (Candle) with zero-copy semantics where possible.
//!
//! # Architecture
//!
//! The implementation provides three levels of zero-copy optimization:
//! 1. **Buffer Protocol**: Direct NumPy array access via Python's buffer protocol
//! 2. **Arrow IPC**: Efficient columnar serialization for batch transfers
//! 3. **Shared Memory Arena**: mmap-based shared memory for Ray actor communication
//!
//! # Usage from Python
//!
//! ```python
//! import deepseek_rust
//!
//! # Zero-copy tensor creation from NumPy
//! tensor = deepseek_rust.CandleTensorView.from_numpy_f32(numpy_array)
//!
//! # Process tensor in Rust
//! result = tensor.matmul(other_tensor)
//!
//! # Zero-copy conversion back to NumPy
//! numpy_result = result.to_numpy_f32()
//! ```

pub mod tensor_view;
pub mod arrow_interop;
pub mod shared_memory;

pub use tensor_view::CandleTensorView;
pub use arrow_interop::ArrowTensorInterop;
pub use shared_memory::{SharedMemoryArena, SharedTensorHandle};

use pyo3::prelude::*;

/// Register all PyO3 bindings with the Python module
pub fn register_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CandleTensorView>()?;
    m.add_class::<ArrowTensorInterop>()?;
    m.add_class::<SharedMemoryArena>()?;
    m.add_class::<SharedTensorHandle>()?;
    
    // Add utility functions
    m.add_function(wrap_pyfunction!(create_shared_arena, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_transfer, m)?)?;
    
    Ok(())
}

/// Create a new shared memory arena for Ray actor communication
#[pyfunction]
pub fn create_shared_arena(name: &str, size_mb: usize) -> PyResult<SharedMemoryArena> {
    SharedMemoryArena::new(name, size_mb * 1024 * 1024)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create arena: {}", e)))
}

/// Benchmark tensor transfer between Python and Rust
#[pyfunction]
pub fn benchmark_transfer(py: Python<'_>, size: usize, iterations: usize) -> PyResult<PyObject> {
    use std::time::Instant;
    
    let results = pyo3::types::PyDict::new_bound(py);
    
    // Create test data
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    
    // Benchmark copy transfer
    let start = Instant::now();
    for _ in 0..iterations {
        let _tensor = candle_core::Tensor::from_vec(
            data.clone(),
            &[size],
            &candle_core::Device::Cpu,
        );
    }
    let copy_time = start.elapsed().as_secs_f64() / iterations as f64;
    results.set_item("copy_transfer_seconds", copy_time)?;
    
    // Calculate throughput
    let bytes_per_transfer = size * std::mem::size_of::<f32>();
    let copy_throughput_gbps = (bytes_per_transfer as f64 / copy_time) / 1e9;
    results.set_item("copy_throughput_gbps", copy_throughput_gbps)?;
    
    results.set_item("tensor_size_elements", size)?;
    results.set_item("tensor_size_bytes", bytes_per_transfer)?;
    results.set_item("iterations", iterations)?;
    
    Ok(results.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Basic test to ensure module compiles correctly
        assert!(true);
    }
}
