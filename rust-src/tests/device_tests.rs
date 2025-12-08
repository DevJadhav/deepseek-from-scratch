//! Device Selection Tests - TDD Approach
//!
//! Tests for unified device selection with CUDA→Metal→CPU priority chain
//! and binary search batch size optimization.
//!
//! These tests pass on all systems regardless of hardware availability
//! by testing the actual hardware detection logic.

use candle_core::{DType, Tensor};
use deepseek_rust::utils::device::{DeviceSelector, DeviceConfig, DevicePriority};
use deepseek_rust::utils::memory::MemoryBudget;

// =============================================================================
// Device Selection Tests
// =============================================================================

#[test]
fn test_device_selector_returns_best_available() {
    // DeviceSelector should always return a valid device
    let device = DeviceSelector::get_device().expect("Should get a device");
    
    // Verify we can create a tensor on the device
    let tensor = Tensor::zeros((2, 2), DType::F32, &device)
        .expect("Should create tensor on device");
    assert_eq!(tensor.dims(), &[2, 2]);
}

#[test]
fn test_device_selector_priority_cuda_first() {
    // Test that CUDA is prioritized when available
    let config = DeviceConfig::default();
    assert_eq!(config.priority, DevicePriority::CudaFirst);
    
    // Verify priority order is CUDA → Metal → CPU
    let priorities = config.get_priority_order();
    assert_eq!(priorities, vec!["cuda", "metal", "cpu"]);
}

#[test]
fn test_device_selector_with_custom_config() {
    // Test custom configuration
    let config = DeviceConfig {
        priority: DevicePriority::CudaFirst,
        cuda_device_id: 0,
        metal_device_id: 0,
        ..Default::default()
    };
    
    let device = DeviceSelector::get_device_with_config(&config)
        .expect("Should get device with config");
    
    // Should return a valid device
    let tensor = Tensor::ones((3, 3), DType::F32, &device)
        .expect("Should create tensor");
    assert_eq!(tensor.dims(), &[3, 3]);
}

#[test]
fn test_device_selector_info() {
    // Test device information retrieval
    let info = DeviceSelector::get_device_info();
    
    // Should have information about availability
    assert!(info.contains_key("cuda_available"));
    assert!(info.contains_key("metal_available"));
    assert!(info.contains_key("selected_device"));
}

#[test]
fn test_device_availability_detection() {
    // Test that availability detection works
    let cuda_available = DeviceSelector::is_cuda_available();
    let metal_available = DeviceSelector::is_metal_available();
    
    // At least CPU should always be available (implicit)
    // These should not panic
    println!("CUDA available: {}", cuda_available);
    println!("Metal available: {}", metal_available);
}

// =============================================================================
// Binary Search Batch Size Tests  
// =============================================================================

#[test]
fn test_binary_search_batch_size_basic() {
    let budget = MemoryBudget {
        max_memory_bytes: 1024 * 1024 * 1024, // 1 GB
        memory_fraction: 0.9,
        auto_adjust_batch: true,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    
    // Test binary search finds a valid batch size
    let memory_per_sample = 1024 * 1024 * 10; // 10 MB per sample
    let optimal = budget.find_optimal_batch_size_binary(memory_per_sample);
    
    // Should be within bounds
    assert!(optimal >= budget.min_batch_size);
    assert!(optimal <= budget.max_batch_size);
    
    // Should fit within budget
    let total_memory = (optimal as u64) * memory_per_sample;
    assert!(budget.within_budget(total_memory));
}

#[test]
fn test_binary_search_batch_size_small_memory() {
    let budget = MemoryBudget {
        max_memory_bytes: 1024 * 1024 * 100, // 100 MB
        memory_fraction: 0.9,
        auto_adjust_batch: true,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    
    // Large memory per sample should result in small batch
    let memory_per_sample = 1024 * 1024 * 50; // 50 MB per sample
    let optimal = budget.find_optimal_batch_size_binary(memory_per_sample);
    
    // Should be 1 (only 1 sample fits in 90MB budget)
    assert_eq!(optimal, 1);
}

#[test]
fn test_binary_search_batch_size_large_memory() {
    let budget = MemoryBudget {
        max_memory_bytes: 1024 * 1024 * 1024 * 10, // 10 GB
        memory_fraction: 0.9,
        auto_adjust_batch: true,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    
    // Small memory per sample should result in max batch
    let memory_per_sample = 1024 * 1024; // 1 MB per sample
    let optimal = budget.find_optimal_batch_size_binary(memory_per_sample);
    
    // Should be max (9GB budget / 1MB = 9000, but capped at 256)
    assert_eq!(optimal, 256);
}

#[test]
fn test_binary_search_respects_bounds() {
    let budget = MemoryBudget {
        max_memory_bytes: 1024 * 1024 * 1024,
        memory_fraction: 0.9,
        auto_adjust_batch: true,
        min_batch_size: 8,  // Custom minimum
        max_batch_size: 64, // Custom maximum
    };
    
    let memory_per_sample = 1024 * 1024 * 5; // 5 MB per sample
    let optimal = budget.find_optimal_batch_size_binary(memory_per_sample);
    
    assert!(optimal >= 8);
    assert!(optimal <= 64);
}

#[test]
fn test_binary_search_disabled() {
    let budget = MemoryBudget {
        max_memory_bytes: 1024 * 1024 * 1024,
        memory_fraction: 0.9,
        auto_adjust_batch: false, // Disabled
        min_batch_size: 1,
        max_batch_size: 256,
    };
    
    // When disabled, should return provided default
    let optimal = budget.find_optimal_batch_size_binary_with_default(
        1024 * 1024 * 10,
        32, // default batch size
    );
    
    assert_eq!(optimal, 32);
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_device_with_tensor_operations() {
    let device = DeviceSelector::get_device().expect("Should get device");
    
    // Test basic tensor operations on selected device
    let a = Tensor::randn(0f32, 1f32, (4, 4), &device).expect("Create tensor a");
    let b = Tensor::randn(0f32, 1f32, (4, 4), &device).expect("Create tensor b");
    
    // Matrix multiplication should work
    let c = a.matmul(&b).expect("Matmul should work");
    assert_eq!(c.dims(), &[4, 4]);
}

#[test]
fn test_device_type_string() {
    let device = DeviceSelector::get_device().expect("Should get device");
    let device_str = DeviceSelector::device_type_string(&device);
    
    // Should be one of the expected types
    assert!(
        device_str == "cuda" || device_str == "metal" || device_str == "cpu",
        "Unexpected device type: {}",
        device_str
    );
}

// =============================================================================
// Environment Variable Configuration Tests
// =============================================================================

#[test]
fn test_config_from_env() {
    // Test that config can read from environment variables
    // (uses defaults if not set)
    let config = DeviceConfig::from_env();
    
    // Should have valid defaults
    assert!(config.min_batch_size >= 1);
    assert!(config.max_batch_size >= config.min_batch_size);
}
