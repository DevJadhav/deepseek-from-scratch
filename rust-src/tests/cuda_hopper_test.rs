//! Integration tests for CUDA Hopper Features
//!
//! These tests verify the CUDA Hopper feature implementations
//! using CPU fallback paths when CUDA is not available.

use candle_core::{Device, Tensor, DType};
use deepseek_rust::model::cuda_hopper::{
    ComputeCapability, CUDAFeatures, KernelBackend,
    TMADescriptor, TMASwizzleMode, WarpgroupConfig,
    FP8Format, FP8Config,
    CUDAKernelDispatcher, CUDAStreamPool,
    create_tma_descriptor, warpgroup_matmul, fp8_matmul,
    dispatch_attention, dispatch_softmax, dispatch_rms_norm,
};

// =============================================================================
// Compute Capability Tests
// =============================================================================

#[test]
fn test_compute_capability_creation() {
    let cc = ComputeCapability::new(9, 0);
    assert_eq!(cc.major, 9);
    assert_eq!(cc.minor, 0);
}

#[test]
fn test_compute_capability_at_least() {
    let hopper = ComputeCapability::new(9, 0);
    assert!(hopper.at_least(9, 0));
    assert!(hopper.at_least(8, 0));
    assert!(hopper.at_least(7, 5));
    assert!(!hopper.at_least(9, 1));
    assert!(!hopper.at_least(10, 0));
}

#[test]
fn test_compute_capability_generation_detection() {
    let hopper = ComputeCapability::new(9, 0);
    assert!(hopper.is_hopper());
    assert!(hopper.is_ampere());
    assert!(hopper.is_turing());
    assert!(hopper.is_volta());

    let ampere = ComputeCapability::new(8, 0);
    assert!(!ampere.is_hopper());
    assert!(ampere.is_ampere());
    assert!(ampere.is_turing());
    assert!(ampere.is_volta());

    let turing = ComputeCapability::new(7, 5);
    assert!(!turing.is_hopper());
    assert!(!turing.is_ampere());
    assert!(turing.is_turing());
    assert!(turing.is_volta());

    let volta = ComputeCapability::new(7, 0);
    assert!(!volta.is_hopper());
    assert!(!volta.is_ampere());
    assert!(!volta.is_turing());
    assert!(volta.is_volta());
}

#[test]
fn test_compute_capability_display() {
    let cc = ComputeCapability::new(9, 0);
    assert_eq!(format!("{}", cc), "SM 9.0");
}

// =============================================================================
// CUDA Features Tests
// =============================================================================

#[test]
fn test_cuda_features_h100() {
    let features = CUDAFeatures::h100();
    
    assert!(features.has_tma);
    assert!(features.has_warpgroup);
    assert!(features.has_fp8_tensor_core);
    assert!(features.has_cluster);
    assert!(features.has_bf16_tensor_core);
    assert!(features.has_tf32);
    assert!(features.has_async_copy);
    assert!(features.compute_capability.is_hopper());
    assert_eq!(features.sm_count, 132);
    assert_eq!(features.total_memory_gb, 80.0);
}

#[test]
fn test_cuda_features_a100() {
    let features = CUDAFeatures::a100();
    
    assert!(!features.has_tma);
    assert!(!features.has_warpgroup);
    assert!(!features.has_fp8_tensor_core);
    assert!(features.has_bf16_tensor_core);
    assert!(features.has_tf32);
    assert!(features.has_async_copy);
    assert!(features.compute_capability.is_ampere());
    assert_eq!(features.sm_count, 108);
}

#[test]
fn test_cuda_features_v100() {
    let features = CUDAFeatures::v100();
    
    assert!(!features.has_tma);
    assert!(!features.has_warpgroup);
    assert!(!features.has_bf16_tensor_core);
    assert!(features.has_fp16_tensor_core);
    assert!(features.compute_capability.is_volta());
}

#[test]
fn test_cuda_features_default() {
    let features = CUDAFeatures::default();
    assert!(!features.has_tma);
    assert_eq!(features.warp_size, 32);
    assert_eq!(features.max_threads_per_block, 1024);
}

#[test]
fn test_optimal_tile_size() {
    let hopper = CUDAFeatures::h100();
    assert_eq!(hopper.optimal_tile_size(), (256, 128, 64));
    
    let ampere = CUDAFeatures::a100();
    assert_eq!(ampere.optimal_tile_size(), (128, 128, 32));
    
    let volta = CUDAFeatures::v100();
    assert_eq!(volta.optimal_tile_size(), (64, 64, 16));
}

#[test]
fn test_optimal_block_size() {
    let hopper = CUDAFeatures::h100();
    assert_eq!(hopper.optimal_block_size(256), 256);
    assert_eq!(hopper.optimal_block_size(100), 128);
    
    let ampere = CUDAFeatures::a100();
    assert_eq!(ampere.optimal_block_size(100), 128);
}

// =============================================================================
// Kernel Backend Tests
// =============================================================================

#[test]
fn test_kernel_backend_selection() {
    let hopper = CUDAFeatures::h100();
    let dispatcher = CUDAKernelDispatcher::with_features(hopper);
    assert_eq!(dispatcher.backend(), KernelBackend::CudaHopper);

    let ampere = CUDAFeatures::a100();
    let dispatcher = CUDAKernelDispatcher::with_features(ampere);
    assert_eq!(dispatcher.backend(), KernelBackend::CudaAmpere);

    let volta = CUDAFeatures::v100();
    let dispatcher = CUDAKernelDispatcher::with_features(volta);
    assert_eq!(dispatcher.backend(), KernelBackend::CudaStandard);
}

#[test]
fn test_kernel_backend_cpu() {
    let device = Device::Cpu;
    let backend = KernelBackend::select(&device, None);
    assert_eq!(backend, KernelBackend::Cpu);
}

// =============================================================================
// TMA Tests
// =============================================================================

#[test]
fn test_tma_descriptor_creation() {
    let desc = TMADescriptor::new(
        vec![128, 4096],
        vec![32, 32],
        TMASwizzleMode::Swizzle128B,
        4,  // F32
    );
    
    assert_eq!(desc.shape, vec![128, 4096]);
    assert_eq!(desc.tile_shape, vec![32, 32]);
    assert_eq!(desc.element_size, 4);
}

#[test]
fn test_tma_descriptor_validation() {
    // Valid: tile size is 32*32*4 = 4096 bytes (divisible by 16)
    let valid_desc = TMADescriptor::new(
        vec![256, 256],
        vec![32, 32],
        TMASwizzleMode::Swizzle128B,
        4,
    );
    assert!(valid_desc.is_valid());
    
    // Should still be valid for F16
    let fp16_desc = TMADescriptor::new(
        vec![256, 256],
        vec![32, 32],
        TMASwizzleMode::Swizzle64B,
        2,
    );
    assert!(fp16_desc.is_valid());
}

#[test]
fn test_tma_descriptor_num_tiles() {
    let desc = TMADescriptor::new(
        vec![128, 4096],
        vec![32, 32],
        TMASwizzleMode::None,
        4,
    );
    
    let tiles = desc.num_tiles();
    assert_eq!(tiles, vec![4, 128]);  // 128/32=4, 4096/32=128
}

#[test]
fn test_create_tma_descriptor_from_tensor() {
    let device = Device::Cpu;
    let tensor = Tensor::zeros((128, 256), DType::F32, &device).unwrap();
    
    let result = create_tma_descriptor(&tensor, vec![32, 32], TMASwizzleMode::Swizzle128B);
    assert!(result.is_ok());
    
    let desc = result.unwrap();
    assert_eq!(desc.shape, vec![128, 256]);
}

// =============================================================================
// Warpgroup Tests
// =============================================================================

#[test]
fn test_warpgroup_config_default() {
    let config = WarpgroupConfig::default();
    assert_eq!(config.warpgroup_size, 128);
    assert_eq!(config.warp_size, 32);
    assert_eq!(config.warps_per_warpgroup, 4);
}

#[test]
fn test_warpgroup_num_warpgroups() {
    let config = WarpgroupConfig::default();
    
    assert_eq!(config.num_warpgroups(128), 1);
    assert_eq!(config.num_warpgroups(256), 2);
    assert_eq!(config.num_warpgroups(384), 3);
    assert_eq!(config.num_warpgroups(64), 1);  // Rounds up
}

#[test]
fn test_warpgroup_for_matmul() {
    let (config, num_warpgroups) = WarpgroupConfig::for_matmul(1024, 1024, 512);
    
    assert_eq!(config.warpgroup_size, 128);
    assert!(num_warpgroups >= 1);
    assert!(num_warpgroups <= 16);
}

#[test]
fn test_warpgroup_matmul_cpu() {
    let device = Device::Cpu;
    let features = CUDAFeatures::default();
    
    let a = Tensor::randn(0.0f32, 1.0, (32, 64), &device).unwrap();
    let b = Tensor::randn(0.0f32, 1.0, (64, 48), &device).unwrap();
    
    let result = warpgroup_matmul(&a, &b, &features).unwrap();
    
    assert_eq!(result.dims(), &[32, 48]);
}

// =============================================================================
// FP8 Tests
// =============================================================================

#[test]
fn test_fp8_format() {
    assert_eq!(FP8Format::E4M3.max_value(), 240.0);
    assert_eq!(FP8Format::E5M2.max_value(), 57344.0);
}

#[test]
fn test_fp8_config_default() {
    let config = FP8Config::default();
    
    assert_eq!(config.format, FP8Format::E4M3);
    assert_eq!(config.scale, 1.0);
    assert_eq!(config.amax_history_len, 16);
}

#[test]
fn test_fp8_config_quantize() {
    let device = Device::Cpu;
    let config = FP8Config::default();
    
    let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
    let (quantized, scale) = config.quantize(&tensor).unwrap();
    
    assert_eq!(quantized.dims(), tensor.dims());
    assert_eq!(scale, 1.0);
}

#[test]
fn test_fp8_config_update_scale() {
    let device = Device::Cpu;
    let mut config = FP8Config::default();
    
    let tensor = Tensor::new(&[100.0f32, 200.0, -150.0], &device).unwrap();
    config.update_scale(&tensor).unwrap();
    
    // Scale should be set to use full FP8 range
    assert!(config.scale > 0.0);
}

#[test]
fn test_fp8_matmul_cpu() {
    let device = Device::Cpu;
    let features = CUDAFeatures::default();
    
    let a = Tensor::randn(0.0f32, 1.0, (16, 32), &device).unwrap();
    let b = Tensor::randn(0.0f32, 1.0, (32, 24), &device).unwrap();
    
    let result = fp8_matmul(&a, &b, None, &features).unwrap();
    
    assert_eq!(result.dims(), &[16, 24]);
}

// =============================================================================
// Kernel Dispatcher Tests
// =============================================================================

#[test]
fn test_cuda_kernel_dispatcher_cpu() {
    let device = Device::Cpu;
    let dispatcher = CUDAKernelDispatcher::new(&device);
    
    assert_eq!(dispatcher.backend(), KernelBackend::Cpu);
}

#[test]
fn test_dispatcher_with_hopper_features() {
    let features = CUDAFeatures::h100();
    let dispatcher = CUDAKernelDispatcher::with_features(features);
    
    assert_eq!(dispatcher.backend(), KernelBackend::CudaHopper);
    assert!(dispatcher.features().has_tma);
}

#[test]
fn test_dispatch_attention_cpu() {
    let device = Device::Cpu;
    let features = CUDAFeatures::default();
    
    let q = Tensor::randn(0.0f32, 1.0, (1, 2, 32, 16), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (1, 2, 32, 16), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (1, 2, 32, 16), &device).unwrap();
    
    let output = dispatch_attention(&q, &k, &v, None, &features).unwrap();
    
    assert_eq!(output.dims(), q.dims());
}

#[test]
fn test_dispatch_softmax_cpu() {
    let device = Device::Cpu;
    let features = CUDAFeatures::default();
    
    let input = Tensor::randn(0.0f32, 1.0, (4, 256), &device).unwrap();
    let output = dispatch_softmax(&input, 1, &features).unwrap();
    
    assert_eq!(output.dims(), input.dims());
    
    // Verify softmax sums to 1
    let sum = output.sum_keepdim(1).unwrap();
    let vals: Vec<f32> = sum.flatten_all().unwrap().to_vec1().unwrap();
    for v in vals {
        assert!((v - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_dispatch_rms_norm_cpu() {
    let device = Device::Cpu;
    let features = CUDAFeatures::default();
    
    let input = Tensor::randn(0.0f32, 1.0, (2, 128), &device).unwrap();
    let weight = Tensor::ones((128,), DType::F32, &device).unwrap();
    
    let output = dispatch_rms_norm(&input, &weight, 1e-6, &features).unwrap();
    
    assert_eq!(output.dims(), input.dims());
}

#[test]
fn test_dispatcher_matmul() {
    let device = Device::Cpu;
    let dispatcher = CUDAKernelDispatcher::new(&device);
    
    let a = Tensor::randn(0.0f32, 1.0, (16, 32), &device).unwrap();
    let b = Tensor::randn(0.0f32, 1.0, (32, 24), &device).unwrap();
    
    let output = dispatcher.matmul(&a, &b).unwrap();
    
    assert_eq!(output.dims(), &[16, 24]);
}

#[test]
fn test_dispatcher_matmul_fp8() {
    let device = Device::Cpu;
    let dispatcher = CUDAKernelDispatcher::new(&device);
    let config = FP8Config::default();
    
    let a = Tensor::randn(0.0f32, 1.0, (8, 16), &device).unwrap();
    let b = Tensor::randn(0.0f32, 1.0, (16, 12), &device).unwrap();
    
    let output = dispatcher.matmul_fp8(&a, &b, &config).unwrap();
    
    assert_eq!(output.dims(), &[8, 12]);
}

// =============================================================================
// CUDA Stream Pool Tests
// =============================================================================

#[test]
fn test_cuda_stream_pool() {
    let pool = CUDAStreamPool::new(4);
    assert_eq!(pool.num_streams(), 4);
}

// =============================================================================
// Feature-Based Dispatch Tests
// =============================================================================

#[test]
fn test_attention_hopper_path() {
    let features = CUDAFeatures::h100();
    let device = Device::Cpu;  // Using CPU device but with Hopper features
    
    let q = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 8), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 8), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 8), &device).unwrap();
    
    // This should exercise the Hopper code path (even if running on CPU)
    let output = dispatch_attention(&q, &k, &v, None, &features).unwrap();
    
    assert_eq!(output.dims(), q.dims());
}

#[test]
fn test_attention_ampere_path() {
    // Use A100 features but disable bf16 tensor cores for CPU testing
    // (CPU backend doesn't support BF16 matmul)
    let mut features = CUDAFeatures::a100();
    features.has_bf16_tensor_core = false;
    let device = Device::Cpu;
    
    let q = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 8), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 8), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 8), &device).unwrap();
    
    let output = dispatch_attention(&q, &k, &v, None, &features).unwrap();
    
    assert_eq!(output.dims(), q.dims());
}
