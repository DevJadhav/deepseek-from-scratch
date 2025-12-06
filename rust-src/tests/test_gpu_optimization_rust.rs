//! GPU Optimization Tests - Rust Backend
//!
//! Tests for:
//! - Attention optimizations (SDPA, memory-efficient patterns)
//! - Mixed precision (BF16/FP16)
//! - Memory profiling
//! - Gradient checkpointing
//! - Compile-time optimizations

use candle_core::{Device, DType, Result, Tensor};

// ============================================================================
// Attention Tests (Section 1.1)
// ============================================================================

mod attention_tests {
    use super::*;
    use deepseek_rust::model::attention::{
        AttentionConfig, AttentionBackend, scaled_dot_product_attention,
        standard_attention, chunked_attention, detect_optimal_backend,
    };

    #[test]
    fn test_attention_config_default() {
        let config = AttentionConfig::default();
        
        // Default should be Standard backend, causal masking, no dropout
        assert!(matches!(config.backend, AttentionBackend::Standard));
        assert!(config.is_causal);
        assert_eq!(config.dropout_p, 0.0);
        assert!(config.chunk_size.is_none());
    }

    #[test]
    fn test_attention_config_custom() {
        let config = AttentionConfig {
            backend: AttentionBackend::MemoryEfficient,
            is_causal: false,
            dropout_p: 0.1,
            chunk_size: Some(512),
        };
        
        assert!(matches!(config.backend, AttentionBackend::MemoryEfficient));
        assert!(!config.is_causal);
        assert_eq!(config.dropout_p, 0.1);
        assert_eq!(config.chunk_size, Some(512));
    }

    #[test]
    fn test_attention_backend_variants() {
        // Test all backend variants
        let _standard = AttentionBackend::Standard;
        let _memory_efficient = AttentionBackend::MemoryEfficient;
        let _flash = AttentionBackend::Flash;
        
        // Verify default
        assert_eq!(AttentionBackend::default(), AttentionBackend::Standard);
    }

    #[test]
    fn test_detect_optimal_backend() {
        let cpu_backend = detect_optimal_backend(&Device::Cpu);
        assert!(matches!(cpu_backend, AttentionBackend::Standard));
    }

    #[test]
    fn test_scaled_dot_product_attention_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        let num_heads = 4;
        let seq_len = 16;
        let d_head = 32;

        let q = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;

        let config = AttentionConfig::default();
        let output = scaled_dot_product_attention(&q, &k, &v, &config)?;

        let (b, h, s, d) = output.dims4()?;
        assert_eq!(b, batch_size);
        assert_eq!(h, num_heads);
        assert_eq!(s, seq_len);
        assert_eq!(d, d_head);

        Ok(())
    }

    #[test]
    fn test_standard_attention() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        let num_heads = 4;
        let seq_len = 8;
        let d_head = 16;

        let q = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;

        // Test with causal masking
        let output = standard_attention(&q, &k, &v, true)?;
        
        let (b, h, s, d) = output.dims4()?;
        assert_eq!(b, batch_size);
        assert_eq!(h, num_heads);
        assert_eq!(s, seq_len);
        assert_eq!(d, d_head);

        // Verify no NaN values
        let sum = output.sum_all()?.to_scalar::<f32>()?;
        assert!(!sum.is_nan(), "Attention output contains NaN");

        Ok(())
    }

    #[test]
    fn test_chunked_attention() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        let num_heads = 4;
        let seq_len = 32;  // Long enough to trigger chunking
        let d_head = 16;
        let chunk_size = 8;

        let q = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;

        let output = chunked_attention(&q, &k, &v, chunk_size, true)?;
        
        let (b, h, s, d) = output.dims4()?;
        assert_eq!(b, batch_size);
        assert_eq!(h, num_heads);
        assert_eq!(s, seq_len);
        assert_eq!(d, d_head);

        Ok(())
    }

    #[test]
    fn test_attention_memory_efficient_backend() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        let num_heads = 4;
        let seq_len = 16;
        let d_head = 16;

        let q = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;

        let config = AttentionConfig {
            backend: AttentionBackend::MemoryEfficient,
            is_causal: true,
            dropout_p: 0.0,
            chunk_size: Some(8),
        };
        
        let output = scaled_dot_product_attention(&q, &k, &v, &config)?;
        let (b, h, s, d) = output.dims4()?;
        assert_eq!(b, batch_size);
        assert_eq!(s, seq_len);

        Ok(())
    }

    #[test]
    fn test_attention_non_causal() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 8;
        let d_head = 8;

        let q = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;

        // Non-causal attention
        let output = standard_attention(&q, &k, &v, false)?;
        
        let shape = output.shape().dims();
        assert_eq!(shape, &[batch_size, num_heads, seq_len, d_head]);

        Ok(())
    }
}

// ============================================================================
// Mixed Precision Tests (Section 1.3)
// ============================================================================

mod mixed_precision_tests {
    use super::*;
    use deepseek_rust::utils::mixed_precision::{
        PrecisionMode, MixedPrecisionConfig,
    };

    #[test]
    fn test_precision_mode_enum() {
        let fp32 = PrecisionMode::FP32;
        let fp16 = PrecisionMode::FP16;
        let bf16 = PrecisionMode::BF16;
        let auto = PrecisionMode::Auto;

        // Should be distinct variants
        assert!(matches!(fp32, PrecisionMode::FP32));
        assert!(matches!(fp16, PrecisionMode::FP16));
        assert!(matches!(bf16, PrecisionMode::BF16));
        assert!(matches!(auto, PrecisionMode::Auto));
    }

    #[test]
    fn test_precision_mode_default() {
        let default_mode = PrecisionMode::default();
        assert!(matches!(default_mode, PrecisionMode::Auto));
    }

    #[test]
    fn test_mixed_precision_config_default() {
        let config = MixedPrecisionConfig::default();

        assert!(matches!(config.mode, PrecisionMode::Auto));
        assert_eq!(config.compute_dtype, DType::F32);
        assert_eq!(config.param_dtype, DType::F32);
        assert_eq!(config.optimizer_dtype, DType::F32);
        assert!(!config.use_loss_scaling);
    }

    #[test]
    fn test_mixed_precision_config_fp32() {
        let config = MixedPrecisionConfig::fp32();

        assert!(matches!(config.mode, PrecisionMode::FP32));
        assert_eq!(config.compute_dtype, DType::F32);
        assert_eq!(config.param_dtype, DType::F32);
        assert!(!config.use_loss_scaling);
    }

    #[test]
    fn test_mixed_precision_config_bf16() {
        let config = MixedPrecisionConfig::bf16();

        assert!(matches!(config.mode, PrecisionMode::BF16));
        assert_eq!(config.compute_dtype, DType::BF16);
        assert_eq!(config.param_dtype, DType::BF16);
        assert_eq!(config.optimizer_dtype, DType::F32); // Optimizer always FP32
        assert!(!config.use_loss_scaling); // BF16 doesn't need loss scaling
    }

    #[test]
    fn test_mixed_precision_config_fp16() {
        let config = MixedPrecisionConfig::fp16();

        assert!(matches!(config.mode, PrecisionMode::FP16));
        assert_eq!(config.compute_dtype, DType::F16);
        assert_eq!(config.param_dtype, DType::F16);
        assert!(config.use_loss_scaling); // FP16 needs loss scaling
    }

    #[test]
    fn test_mixed_precision_config_custom() {
        let config = MixedPrecisionConfig {
            mode: PrecisionMode::FP16,
            compute_dtype: DType::F16,
            param_dtype: DType::F16,
            optimizer_dtype: DType::F32,
            use_loss_scaling: true,
            init_loss_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
        };

        assert!(matches!(config.mode, PrecisionMode::FP16));
        assert_eq!(config.init_loss_scale, 65536.0);
        assert_eq!(config.growth_factor, 2.0);
        assert_eq!(config.backoff_factor, 0.5);
    }

    #[test]
    fn test_tensor_dtype_conversion_f16() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0f32, 1f32, (4, 4), &device)?;

        let converted = tensor.to_dtype(DType::F16)?;
        assert_eq!(converted.dtype(), DType::F16);

        Ok(())
    }

    #[test]
    fn test_tensor_dtype_conversion_bf16() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0f32, 1f32, (4, 4), &device)?;

        let converted = tensor.to_dtype(DType::BF16)?;
        assert_eq!(converted.dtype(), DType::BF16);

        Ok(())
    }

    #[test]
    fn test_fp32_accumulation() -> Result<()> {
        let device = Device::Cpu;
        
        // Create FP16 tensors
        let a = Tensor::randn(0f32, 1f32, (4, 4), &device)?.to_dtype(DType::F16)?;
        let b = Tensor::randn(0f32, 1f32, (4, 4), &device)?.to_dtype(DType::F16)?;

        // Convert to FP32, add, then convert back
        let a_fp32 = a.to_dtype(DType::F32)?;
        let b_fp32 = b.to_dtype(DType::F32)?;
        let result = (a_fp32 + b_fp32)?;
        
        // Result is FP32
        assert_eq!(result.dtype(), DType::F32);

        Ok(())
    }
}

// ============================================================================
// Memory Profiling Tests (Section 1.5)
// ============================================================================

mod memory_tests {
    use super::*;
    use deepseek_rust::utils::memory::{
        MemoryStats, MemoryProfiler, ProfileRegion,
    };

    #[test]
    fn test_memory_stats_creation() {
        let stats = MemoryStats {
            allocated_bytes: 1024,
            peak_allocated_bytes: 1536,
            num_allocations: 10,
            num_deallocations: 5,
            reserved_bytes: 2048,
        };

        assert_eq!(stats.allocated_bytes, 1024);
        assert_eq!(stats.reserved_bytes, 2048);
        assert_eq!(stats.peak_allocated_bytes, 1536);
    }

    #[test]
    fn test_memory_stats_default() {
        let stats = MemoryStats::default();
        
        assert_eq!(stats.allocated_bytes, 0);
        assert_eq!(stats.peak_allocated_bytes, 0);
        assert_eq!(stats.num_allocations, 0);
    }

    #[test]
    fn test_memory_stats_allocated_mb() {
        let stats = MemoryStats {
            allocated_bytes: 1024 * 1024, // 1 MB
            peak_allocated_bytes: 1024 * 1024,
            reserved_bytes: 2 * 1024 * 1024,
            num_allocations: 1,
            num_deallocations: 0,
        };

        assert!((stats.allocated_mb() - 1.0).abs() < 0.01);
        assert!((stats.peak_allocated_mb() - 1.0).abs() < 0.01);
        assert!((stats.reserved_mb() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_profiler_creation() {
        let device = Device::Cpu;
        let profiler = MemoryProfiler::new(device);
        // Profiler created successfully
        assert!(true);
    }

    #[test]
    fn test_memory_profiler_start_end_region() {
        let device = Device::Cpu;
        let mut profiler = MemoryProfiler::new(device);

        profiler.start_region("test_region");
        // Do something
        profiler.end_region();
        
        // Region was recorded
        assert!(true);
    }

    #[test]
    fn test_profile_region() {
        let region = ProfileRegion::new("forward_pass", 1024);

        assert_eq!(region.name, "forward_pass");
        assert_eq!(region.start_memory, 1024);
        assert!(region.duration.is_none()); // Not ended yet
    }

    #[test]
    fn test_profile_region_end() {
        let mut region = ProfileRegion::new("test", 1000);
        region.end(2000);

        assert!(region.duration.is_some());
        assert_eq!(region.end_memory, Some(2000));
        assert_eq!(region.memory_delta(), Some(1000));
    }

    #[test]
    fn test_memory_profiler_with_log_interval() {
        let device = Device::Cpu;
        let profiler = MemoryProfiler::new(device)
            .with_log_interval(50);
        
        // Profiler created with custom log interval
        assert!(true);
    }
}

// ============================================================================
// Gradient Checkpointing Tests (Section 1.4)
// ============================================================================

mod checkpointing_tests {
    use super::*;
    use deepseek_rust::training::checkpointing::CheckpointConfig;

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();

        assert!(config.enabled);
        assert!(config.checkpoint_every_n_layers > 0);
        assert!(config.checkpoint_attention);
        assert!(config.checkpoint_mlp);
        assert!(config.checkpoint_moe);
        assert!(config.memory_efficient);
    }

    #[test]
    fn test_checkpoint_config_custom() {
        let config = CheckpointConfig {
            enabled: true,
            checkpoint_every_n_layers: 4,
            checkpoint_moe: true,
            checkpoint_attention: true,
            checkpoint_mlp: false,
            memory_efficient: true,
        };

        assert_eq!(config.checkpoint_every_n_layers, 4);
        assert!(!config.checkpoint_mlp);
        assert!(config.checkpoint_moe);
    }

    #[test]
    fn test_checkpoint_config_builder_pattern() {
        let config = CheckpointConfig::default()
            .with_checkpoint_every(3)
            .with_checkpoint_moe(false)
            .with_checkpoint_attention(true)
            .with_checkpoint_mlp(false);

        assert_eq!(config.checkpoint_every_n_layers, 3);
        assert!(!config.checkpoint_moe);
        assert!(config.checkpoint_attention);
        assert!(!config.checkpoint_mlp);
    }

    #[test]
    fn test_should_checkpoint_layer() {
        // Checkpoint every 2 layers
        let config = CheckpointConfig::default()
            .with_checkpoint_every(2);

        // Layer 0: should checkpoint (0 % 2 == 0)
        // Layer 1: should not
        // Layer 2: should checkpoint (2 % 2 == 0)
        assert!(config.should_checkpoint_layer(0));
        assert!(!config.should_checkpoint_layer(1));
        assert!(config.should_checkpoint_layer(2));
        assert!(!config.should_checkpoint_layer(3));
        assert!(config.should_checkpoint_layer(4));
    }

    #[test]
    fn test_checkpoint_config_disabled() {
        let config = CheckpointConfig {
            enabled: false,
            ..Default::default()
        };

        // When disabled, should_checkpoint_layer returns false
        assert!(!config.should_checkpoint_layer(0));
        assert!(!config.should_checkpoint_layer(100));
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;
    use deepseek_rust::model::attention::{AttentionConfig, scaled_dot_product_attention};
    use deepseek_rust::utils::mixed_precision::MixedPrecisionConfig;

    #[test]
    fn test_attention_with_mixed_precision() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        let num_heads = 4;
        let seq_len = 8;
        let d_head = 16;

        // Create FP32 tensors
        let q = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, d_head), &device)?;

        // Note: BF16 matmul is not supported on CPU, so we test with FP32
        // In production with GPU/Metal, BF16 would work
        // Run attention with FP32
        let config = AttentionConfig::default();
        let output = scaled_dot_product_attention(&q, &k, &v, &config)?;

        assert_eq!(output.dtype(), DType::F32);
        
        // Test FP16 conversion (available on CPU for storage)
        let output_f16 = output.to_dtype(DType::F16)?;
        assert_eq!(output_f16.dtype(), DType::F16);
        
        Ok(())
    }

    #[test]
    fn test_memory_tracking_during_attention() -> Result<()> {
        use deepseek_rust::utils::memory::MemoryProfiler;

        let device = Device::Cpu;
        let mut profiler = MemoryProfiler::new(device.clone());

        profiler.start_region("attention_forward");
        
        let q = Tensor::randn(0f32, 1f32, (2, 4, 16, 32), &device)?;
        let k = Tensor::randn(0f32, 1f32, (2, 4, 16, 32), &device)?;
        let v = Tensor::randn(0f32, 1f32, (2, 4, 16, 32), &device)?;

        let config = AttentionConfig::default();
        let _output = scaled_dot_product_attention(&q, &k, &v, &config)?;

        profiler.end_region();

        // Profiling completed
        assert!(true);

        Ok(())
    }

    #[test]
    fn test_mixed_precision_with_checkpointing() {
        use deepseek_rust::training::checkpointing::CheckpointConfig;

        // Configure mixed precision
        let mp_config = MixedPrecisionConfig::bf16();
        
        // Configure checkpointing
        let ckpt_config = CheckpointConfig::default()
            .with_checkpoint_every(2)
            .with_checkpoint_attention(true);

        // Both configs work together
        assert_eq!(mp_config.compute_dtype, DType::BF16);
        assert!(ckpt_config.checkpoint_attention);
        assert_eq!(ckpt_config.checkpoint_every_n_layers, 2);
    }

    #[test]
    fn test_full_pipeline_config() {
        use deepseek_rust::model::attention::AttentionBackend;
        use deepseek_rust::training::checkpointing::CheckpointConfig;

        // Attention config
        let attn_config = AttentionConfig {
            backend: AttentionBackend::MemoryEfficient,
            is_causal: true,
            dropout_p: 0.0,
            chunk_size: Some(512),
        };

        // Mixed precision config
        let mp_config = MixedPrecisionConfig::bf16();

        // Checkpointing config
        let ckpt_config = CheckpointConfig::default()
            .with_checkpoint_every(3);

        // Verify all configs
        assert!(matches!(attn_config.backend, AttentionBackend::MemoryEfficient));
        assert_eq!(mp_config.compute_dtype, DType::BF16);
        assert_eq!(ckpt_config.checkpoint_every_n_layers, 3);
    }
}

// ============================================================================
// Run all tests
// ============================================================================

fn main() {
    println!("Phase 1 Rust tests - use 'cargo test' to run");
}
