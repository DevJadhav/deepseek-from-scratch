//! Integration tests for Metal SIMD-Group Kernels
//!
//! These tests verify the Metal kernel implementations without requiring
//! actual Metal hardware - they test the API and fallback paths.

use candle_core::{Device, Tensor, DType};
use deepseek_rust::model::metal_kernels::{
    AppleSiliconGeneration, MetalFeatures, SIMDGroupConfig, TileConfig,
    MetalPipelineCache, MetalKernelDispatcher,
    softmax_simd, softmax_simd_stable, rms_norm_simd, attention_tiled,
    matmul_fp16_tiled, matmul_bf16_tiled, swiglu_fused,
};

// =============================================================================
// Metal Feature Detection Tests
// =============================================================================

#[test]
fn test_apple_silicon_generation() {
    assert!(!AppleSiliconGeneration::M1.supports_bf16());
    assert!(!AppleSiliconGeneration::M2.supports_bf16());
    assert!(AppleSiliconGeneration::M3.supports_bf16());
    assert!(AppleSiliconGeneration::M4.supports_bf16());
    
    assert_eq!(AppleSiliconGeneration::M1.simd_group_size(), 32);
    assert_eq!(AppleSiliconGeneration::M4.simd_group_size(), 32);
    
    assert_eq!(AppleSiliconGeneration::M1.max_threadgroup_memory(), 32 * 1024);
    assert_eq!(AppleSiliconGeneration::M3.max_threadgroup_memory(), 64 * 1024);
}

#[test]
fn test_metal_features_presets() {
    let m1 = MetalFeatures::m1();
    assert_eq!(m1.generation, AppleSiliconGeneration::M1);
    assert!(!m1.has_bf16);
    assert!(m1.has_fp16);
    assert!(m1.has_simd_group_reduce);
    
    let m3 = MetalFeatures::m3();
    assert_eq!(m3.generation, AppleSiliconGeneration::M3);
    assert!(m3.has_bf16);
    assert_eq!(m3.max_threadgroup_memory, 64 * 1024);
    
    let m4 = MetalFeatures::m4();
    assert!(m4.has_bf16);
}

#[test]
fn test_metal_features_detection_cpu() {
    // On CPU, should return default features
    let device = Device::Cpu;
    let features = MetalFeatures::detect(&device);
    
    assert_eq!(features.simd_width, 32);
    assert!(features.has_simd_group_reduce);
}

#[test]
fn test_optimal_tile_size() {
    let m1_features = MetalFeatures::m1();
    let (m, n, k) = m1_features.optimal_tile_size();
    assert_eq!((m, n, k), (32, 32, 32));
    
    let m3_features = MetalFeatures::m3();
    let (m, n, k) = m3_features.optimal_tile_size();
    assert_eq!((m, n, k), (64, 64, 32));
}

// =============================================================================
// SIMD Config Tests
// =============================================================================

#[test]
fn test_simd_config_softmax() {
    let config = SIMDGroupConfig::for_softmax(4096);
    assert_eq!(config.simd_size, 32);
    assert!(config.simd_groups_per_threadgroup <= 8);
    assert!(config.threadgroup_size <= 256);
}

#[test]
fn test_simd_config_rms_norm() {
    let config = SIMDGroupConfig::for_rms_norm(2048);
    assert_eq!(config.simd_size, 32);
    assert!(config.simd_groups_per_threadgroup <= 4);
}

// =============================================================================
// Tile Config Tests
// =============================================================================

#[test]
fn test_tile_config_matmul() {
    let features = MetalFeatures::default();
    
    // Small matrix
    let config = TileConfig::for_matmul(32, 32, 32, &features);
    assert_eq!(config.tile_m, 16);
    assert_eq!(config.tile_n, 16);
    
    // Large matrix
    let config = TileConfig::for_matmul(4096, 4096, 4096, &features);
    assert!(config.tile_m >= 32);
    assert!(config.tile_n >= 32);
}

#[test]
fn test_tile_config_attention() {
    // Short sequence
    let config = TileConfig::for_attention(256, 64);
    assert_eq!(config.tile_m, 32);
    assert_eq!(config.tile_k, 64);
    
    // Long sequence
    let config = TileConfig::for_attention(4096, 128);
    assert_eq!(config.tile_m, 64);
    assert_eq!(config.tile_n, 64);
}

// =============================================================================
// Pipeline Cache Tests
// =============================================================================

#[test]
fn test_pipeline_cache_basic() {
    let cache = MetalPipelineCache::new();
    
    // First access creates entry
    assert!(cache.get_or_create("test_kernel").is_ok());
    
    // Stats should show one entry
    let stats = cache.stats().unwrap();
    assert_eq!(stats.len(), 1);
    assert!(stats.iter().any(|(name, _)| name == "test_kernel"));
}

#[test]
fn test_pipeline_cache_invocation_count() {
    let cache = MetalPipelineCache::new();
    
    // Multiple accesses should increment count
    for _ in 0..5 {
        assert!(cache.get_or_create("repeated_kernel").is_ok());
    }
    
    let stats = cache.stats().unwrap();
    let (_, count) = stats.iter().find(|(name, _)| name == "repeated_kernel").unwrap();
    assert_eq!(*count, 5);
}

#[test]
fn test_pipeline_cache_clear() {
    let cache = MetalPipelineCache::new();
    
    cache.get_or_create("kernel1").unwrap();
    cache.get_or_create("kernel2").unwrap();
    
    assert!(cache.clear().is_ok());
    
    let stats = cache.stats().unwrap();
    assert!(stats.is_empty());
}

// =============================================================================
// Kernel Function Tests (CPU Fallback)
// =============================================================================

#[test]
fn test_softmax_simd_cpu() {
    let device = Device::Cpu;
    let input = Tensor::randn(0.0f32, 1.0, (2, 8, 512), &device).unwrap();
    
    // Use positive index (2) instead of -1 for last dimension
    let output = softmax_simd(&input, 2).unwrap();
    
    assert_eq!(output.dims(), input.dims());
    
    // Sum along last dim should be ~1.0
    let sum = output.sum_keepdim(2).unwrap();
    let sum_vals: Vec<f32> = sum.flatten_all().unwrap().to_vec1().unwrap();
    for val in sum_vals {
        assert!((val - 1.0).abs() < 1e-5, "Softmax sum should be 1.0, got {}", val);
    }
}

#[test]
fn test_softmax_simd_stable_cpu() {
    let device = Device::Cpu;
    
    // Test with large values that could cause overflow without stability
    let input = Tensor::new(&[1000.0f32, 1001.0, 1002.0], &device).unwrap();
    
    let output = softmax_simd_stable(&input, 0).unwrap();
    
    let sum: f32 = output.sum_all().unwrap().to_scalar().unwrap();
    assert!((sum - 1.0).abs() < 1e-5, "Stable softmax sum should be 1.0");
}

#[test]
fn test_rms_norm_simd_cpu() {
    let device = Device::Cpu;
    let input = Tensor::randn(0.0f32, 1.0, (2, 4, 256), &device).unwrap();
    let weight = Tensor::ones((256,), DType::F32, &device).unwrap();
    
    let output = rms_norm_simd(&input, &weight, 1e-6).unwrap();
    
    assert_eq!(output.dims(), input.dims());
}

#[test]
fn test_attention_tiled_cpu() {
    let device = Device::Cpu;
    let batch = 2;
    let heads = 4;
    let seq_len = 64;
    let head_dim = 32;
    
    let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &device).unwrap();
    
    let output = attention_tiled(&q, &k, &v, None, None).unwrap();
    
    assert_eq!(output.dims(), &[batch, heads, seq_len, head_dim]);
}

#[test]
#[ignore = "f16 matmul not supported on CPU"]
fn test_matmul_fp16_tiled_cpu() {
    let device = Device::Cpu;
    let a = Tensor::randn(0.0f32, 1.0, (32, 64), &device).unwrap();
    let b = Tensor::randn(0.0f32, 1.0, (64, 48), &device).unwrap();
    
    let output = matmul_fp16_tiled(&a, &b, None).unwrap();
    
    assert_eq!(output.dims(), &[32, 48]);
    assert_eq!(output.dtype(), DType::F16);
}

#[test]
#[ignore = "bf16/f16 matmul not supported on CPU"]
fn test_matmul_bf16_fallback_cpu() {
    let device = Device::Cpu;
    let features = MetalFeatures::m1();  // No BF16 support
    
    let a = Tensor::randn(0.0f32, 1.0, (16, 32), &device).unwrap();
    let b = Tensor::randn(0.0f32, 1.0, (32, 24), &device).unwrap();
    
    // Should fallback to FP16 since M1 doesn't have BF16
    let output = matmul_bf16_tiled(&a, &b, &features).unwrap();
    
    assert_eq!(output.dims(), &[16, 24]);
}

#[test]
fn test_swiglu_fused_cpu() {
    let device = Device::Cpu;
    let gate = Tensor::randn(0.0f32, 1.0, (2, 4, 128), &device).unwrap();
    let up = Tensor::randn(0.0f32, 1.0, (2, 4, 128), &device).unwrap();
    
    let output = swiglu_fused(&gate, &up).unwrap();
    
    assert_eq!(output.dims(), &[2, 4, 128]);
}

// =============================================================================
// Kernel Dispatcher Tests
// =============================================================================

#[test]
fn test_metal_kernel_dispatcher_cpu() {
    let device = Device::Cpu;
    let dispatcher = MetalKernelDispatcher::new(&device);
    
    // Test feature detection
    let features = dispatcher.features();
    assert!(features.has_simd_group_reduce);
}

#[test]
fn test_dispatcher_softmax() {
    let device = Device::Cpu;
    let dispatcher = MetalKernelDispatcher::new(&device);
    
    let input = Tensor::randn(0.0f32, 1.0, (4, 256), &device).unwrap();
    // Use positive index (1) instead of -1 for last dimension
    let output = dispatcher.softmax(&input, 1).unwrap();
    
    assert_eq!(output.dims(), input.dims());
}

#[test]
fn test_dispatcher_rms_norm() {
    let device = Device::Cpu;
    let dispatcher = MetalKernelDispatcher::new(&device);
    
    let input = Tensor::randn(0.0f32, 1.0, (2, 128), &device).unwrap();
    let weight = Tensor::ones((128,), DType::F32, &device).unwrap();
    
    let output = dispatcher.rms_norm(&input, &weight, 1e-6).unwrap();
    assert_eq!(output.dims(), input.dims());
}

#[test]
fn test_dispatcher_attention() {
    let device = Device::Cpu;
    let dispatcher = MetalKernelDispatcher::new(&device);
    
    let q = Tensor::randn(0.0f32, 1.0, (1, 2, 32, 16), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (1, 2, 32, 16), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (1, 2, 32, 16), &device).unwrap();
    
    let output = dispatcher.attention(&q, &k, &v, None).unwrap();
    assert_eq!(output.dims(), q.dims());
}

#[test]
#[ignore = "f16 matmul not supported on CPU"]
fn test_dispatcher_matmul() {
    let device = Device::Cpu;
    let dispatcher = MetalKernelDispatcher::new(&device);
    
    let a = Tensor::randn(0.0f32, 1.0, (8, 16), &device).unwrap();
    let b = Tensor::randn(0.0f32, 1.0, (16, 12), &device).unwrap();
    
    let output = dispatcher.matmul(&a, &b).unwrap();
    assert_eq!(output.dims(), &[8, 12]);
}

#[test]
fn test_dispatcher_swiglu() {
    let device = Device::Cpu;
    let dispatcher = MetalKernelDispatcher::new(&device);
    
    let gate = Tensor::randn(0.0f32, 1.0, (4, 64), &device).unwrap();
    let up = Tensor::randn(0.0f32, 1.0, (4, 64), &device).unwrap();
    
    let output = dispatcher.swiglu(&gate, &up).unwrap();
    assert_eq!(output.dims(), &[4, 64]);
}

// =============================================================================
// Numerical Correctness Tests
// =============================================================================

#[test]
fn test_softmax_correctness() {
    let device = Device::Cpu;
    
    // Known values
    let input = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
    let output = softmax_simd_stable(&input, 0).unwrap();
    
    let values: Vec<f32> = output.to_vec1().unwrap();
    
    // Expected: softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
    assert!((values[0] - 0.0900).abs() < 0.01);
    assert!((values[1] - 0.2447).abs() < 0.01);
    assert!((values[2] - 0.6652).abs() < 0.01);
}

#[test]
fn test_rms_norm_correctness() {
    let device = Device::Cpu;
    
    // Simple case: input with known RMS
    let input = Tensor::new(&[3.0f32, 4.0], &device).unwrap().unsqueeze(0).unwrap();
    let weight = Tensor::ones((2,), DType::F32, &device).unwrap();
    
    let output = rms_norm_simd(&input, &weight, 1e-6).unwrap();
    let values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
    
    // RMS of [3, 4] = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
    // Normalized: [3/3.536, 4/3.536] ≈ [0.848, 1.131]
    assert!((values[0] - 0.848).abs() < 0.01);
    assert!((values[1] - 1.131).abs() < 0.01);
}
