//! Distributed Training Infrastructure Tests - Rust Backend
//!
//! Comprehensive tests for distributed training components.

use candle_core::{Device, Tensor};
use std::time::Duration;

use deepseek_rust::distributed::{
    expert::{ExpertParallelConfig, ExpertParallelDispatch, LoadBalancer},
    pipeline::DualPipeConfig,
    ring_attention::{
        DistributedLayerNorm, DistributedRMSNorm, RingAttention, RingAttentionConfig,
        SequenceParallelConfig,
    },
    fault_tolerance::{
        ElasticConfig, HeartbeatMonitor, PreemptionHandler, PreemptionSignal,
        WorkerInfo, WorkerState,
    },
    distributed_checkpoint::{
        CheckpointConfig, CheckpointMetadata,
    },
};

// ============================================================================
// Expert Parallelism Tests
// ============================================================================

mod expert_parallelism_tests {
    use super::*;

    #[test]
    fn test_expert_parallel_config_creation() {
        let config = ExpertParallelConfig::new(256);
        assert_eq!(config.num_experts, 256);
        assert!(config.capacity_factor > 0.0);
    }

    #[test]
    fn test_expert_parallel_config_with_capacity() {
        let config = ExpertParallelConfig::new(256)
            .with_capacity_factor(1.5);
        assert_eq!(config.capacity_factor, 1.5);
    }

    #[test]
    fn test_expert_parallel_config_with_load_balance() {
        let config = ExpertParallelConfig::new(64)
            .with_load_balance(true, 0.02);
        assert!(config.load_balance);
        assert_eq!(config.load_balance_weight, 0.02);
    }

    #[test]
    fn test_expert_parallel_config_with_auxiliary_loss() {
        let config = ExpertParallelConfig::new(64)
            .with_auxiliary_loss(true, 0.001);
        assert!(config.auxiliary_loss);
        assert_eq!(config.auxiliary_loss_weight, 0.001);
    }

    #[test]
    fn test_expert_parallel_config_local_expert_ids() {
        let config = ExpertParallelConfig::new(16);
        let local_ids = config.local_expert_ids();
        assert!(!local_ids.is_empty());
    }

    #[test]
    fn test_expert_parallel_config_capacity() {
        let config = ExpertParallelConfig::new(8)
            .with_capacity_factor(1.25);
        
        let capacity = config.expert_capacity(100);
        // With 8 experts, avg = 100/8 = 12.5, capacity = 12.5 * 1.25 ≈ 16
        assert!(capacity >= 15);
    }

    #[test]
    fn test_load_balancer_creation() {
        let config = ExpertParallelConfig::new(16);
        let balancer = LoadBalancer::new(config);
        // Should initialize without panic
        let score = balancer.compute_imbalance_score();
        assert!(score >= 0.0 || score.is_nan());
    }

    #[test]
    fn test_load_balancer_update() {
        let config = ExpertParallelConfig::new(4);
        let mut balancer = LoadBalancer::new(config);
        
        balancer.update(&[0, 1, 2, 0, 0, 1, 3]);
        
        let score = balancer.compute_imbalance_score();
        assert!(score >= 0.0);
    }

    #[test]
    fn test_load_balancer_reset() {
        let config = ExpertParallelConfig::new(4);
        let mut balancer = LoadBalancer::new(config);
        
        balancer.update(&[0, 1, 2, 3]);
        balancer.reset();
        
        let score = balancer.compute_imbalance_score();
        assert!(score.is_nan() || score >= 0.0);
    }

    #[test]
    fn test_expert_parallel_dispatch_creation() {
        let dispatch = ExpertParallelDispatch::new(8);
        let config = dispatch.config();
        assert_eq!(config.num_experts, 8);
    }
}

// ============================================================================
// DualPipe Pipeline Parallelism Tests
// ============================================================================

mod dualpipe_tests {
    use super::*;

    #[test]
    fn test_dualpipe_config_creation() {
        let config = DualPipeConfig::new(8);
        assert_eq!(config.num_micro_batches, 8);
        assert!(config.overlap_communication);
    }

    #[test]
    fn test_dualpipe_config_with_stages() {
        let config = DualPipeConfig::new(4)
            .with_stages(4, 1);
        assert_eq!(config.num_stages, 4);
        assert_eq!(config.stage_rank, 1);
        assert!(!config.is_first_stage());
        assert!(!config.is_last_stage());
    }

    #[test]
    fn test_dualpipe_config_first_last_stage() {
        let config_first = DualPipeConfig::new(4).with_stages(4, 0);
        assert!(config_first.is_first_stage());
        assert!(!config_first.is_last_stage());
        
        let config_last = DualPipeConfig::new(4).with_stages(4, 3);
        assert!(!config_last.is_first_stage());
        assert!(config_last.is_last_stage());
    }

    #[test]
    fn test_dualpipe_config_with_checkpointing() {
        let config = DualPipeConfig::new(8)
            .with_checkpointing(true, 4);
        assert!(config.activation_checkpointing);
        assert_eq!(config.checkpoint_chunk_size, 4);
    }

    #[test]
    fn test_dualpipe_micro_batches_per_stream() {
        let config = DualPipeConfig::new(8);
        assert_eq!(config.micro_batches_per_stream(), 4);
    }
}

// ============================================================================
// Sequence Parallelism Tests
// ============================================================================

mod sequence_parallelism_tests {
    use super::*;

    #[test]
    fn test_sequence_parallel_config_creation() {
        let config = SequenceParallelConfig::new(2, 0);
        assert_eq!(config.sp_size, 2);
        assert_eq!(config.sp_rank, 0);
    }

    #[test]
    fn test_ring_attention_config() {
        let config = RingAttentionConfig::new(true, 0.0, None);
        assert!(config.causal);
        assert_eq!(config.dropout, 0.0);
    }

    #[test]
    fn test_ring_attention_creation() {
        let config = RingAttentionConfig::new(true, 0.0, None);
        let _attention = RingAttention::new(config);
        // RingAttention created successfully - config holds causal flag
    }

    #[test]
    fn test_ring_attention_forward() {
        let config = RingAttentionConfig::new(false, 0.0, None);
        let attention = RingAttention::new(config);
        
        let device = Device::Cpu;
        let batch = 2;
        let heads = 4;
        let seq_len = 8;
        let head_dim = 16;
        
        // Ring attention expects 4D tensors: (batch, heads, seq_len, head_dim)
        let query = Tensor::randn(0.0f32, 0.1, (batch, heads, seq_len, head_dim), &device).unwrap();
        let key = Tensor::randn(0.0f32, 0.1, (batch, heads, seq_len, head_dim), &device).unwrap();
        let value = Tensor::randn(0.0f32, 0.1, (batch, heads, seq_len, head_dim), &device).unwrap();
        
        let output = attention.forward(&query, &key, &value).unwrap();
        assert_eq!(output.dims(), &[batch, heads, seq_len, head_dim]);
    }

    #[test]
    fn test_distributed_layer_norm_creation() {
        let device = Device::Cpu;
        let norm = DistributedLayerNorm::new(vec![64], 1e-5, &device).unwrap();
        let input = Tensor::zeros((2, 4, 64), candle_core::DType::F32, &device).unwrap();
        assert!(norm.forward(&input).is_ok());
    }

    #[test]
    fn test_distributed_layer_norm_forward() {
        let device = Device::Cpu;
        let norm = DistributedLayerNorm::new(vec![8], 1e-6, &device).unwrap();
        
        let input = Tensor::randn(0.0f32, 1.0, (2, 4, 8), &device).unwrap();
        let output = norm.forward(&input).unwrap();
        
        assert_eq!(output.dims(), input.dims());
    }

    #[test]
    fn test_distributed_rms_norm_creation() {
        let device = Device::Cpu;
        let norm = DistributedRMSNorm::new(64, 1e-5, &device).unwrap();
        let input = Tensor::ones((2, 4, 64), candle_core::DType::F32, &device).unwrap();
        assert!(norm.forward(&input).is_ok());
    }

    #[test]
    fn test_distributed_rms_norm_forward() {
        let device = Device::Cpu;
        let norm = DistributedRMSNorm::new(8, 1e-6, &device).unwrap();
        
        let input = Tensor::randn(0.0f32, 1.0, (2, 4, 8), &device).unwrap();
        let output = norm.forward(&input).unwrap();
        
        assert_eq!(output.dims(), input.dims());
    }
}

// ============================================================================
// Fault Tolerance Tests
// ============================================================================

mod fault_tolerance_tests {
    use super::*;

    #[test]
    fn test_elastic_config_creation() {
        let config = ElasticConfig::new(2, 8);
        assert_eq!(config.min_workers, 2);
        assert_eq!(config.max_workers, 8);
    }

    #[test]
    fn test_elastic_config_can_continue() {
        let mut config = ElasticConfig::new(2, 8);
        config.current_workers = 3;
        assert!(config.can_continue());
        
        config.current_workers = 1;
        assert!(!config.can_continue());
    }

    #[test]
    fn test_elastic_config_batch_size() {
        let config = ElasticConfig::new(1, 8)
            .with_batch_scaling(true, 32);
        
        let mut config = config;
        config.current_workers = 4;
        
        assert_eq!(config.effective_batch_size(), 128); // 32 * 4
    }

    #[test]
    fn test_heartbeat_monitor_creation() {
        let monitor = HeartbeatMonitor::new(
            Duration::from_secs(5),
            Duration::from_secs(30),
        );
        assert_eq!(monitor.worker_count(), 0);
    }

    #[test]
    fn test_heartbeat_monitor_register_worker() {
        let monitor = HeartbeatMonitor::new(
            Duration::from_secs(5),
            Duration::from_secs(30),
        );
        
        let info = WorkerInfo::new(0, "cuda:0", "worker-0");
        monitor.register_worker(info);
        
        assert_eq!(monitor.worker_count(), 1);
    }

    #[test]
    fn test_heartbeat_monitor_heartbeat() {
        let monitor = HeartbeatMonitor::new(
            Duration::from_secs(5),
            Duration::from_secs(30),
        );
        
        let info = WorkerInfo::new(0, "cuda:0", "worker-0");
        monitor.register_worker(info);
        // Set worker state to Running (new workers start in Initializing)
        monitor.set_state(0, WorkerState::Running);
        monitor.heartbeat(0, 100);
        
        let healthy = monitor.get_healthy_workers();
        assert_eq!(healthy.len(), 1);
    }

    #[test]
    fn test_heartbeat_monitor_unregister() {
        let monitor = HeartbeatMonitor::new(
            Duration::from_secs(5),
            Duration::from_secs(30),
        );
        
        let info = WorkerInfo::new(0, "cuda:0", "worker-0");
        monitor.register_worker(info);
        monitor.unregister_worker(0);
        
        assert_eq!(monitor.worker_count(), 0);
    }

    #[test]
    fn test_preemption_handler_creation() {
        let handler = PreemptionHandler::new(Duration::from_secs(60));
        assert!(!handler.is_preempted());
    }

    #[test]
    fn test_preemption_handler_preempt() {
        let handler = PreemptionHandler::new(Duration::from_secs(60));
        handler.signal_preemption(PreemptionSignal::SigTerm);
        assert!(handler.is_preempted());
    }

    #[test]
    fn test_preemption_handler_time_remaining() {
        let handler = PreemptionHandler::new(Duration::from_secs(60));
        let remaining = handler.time_remaining();
        assert_eq!(remaining, Duration::from_secs(60));
    }

    #[test]
    fn test_worker_info_creation() {
        let info = WorkerInfo::new(0, "cuda:0", "hostname");
        assert_eq!(info.worker_id, 0);
        assert_eq!(info.device, "cuda:0");
    }

    #[test]
    fn test_worker_state_transitions() {
        let mut info = WorkerInfo::new(0, "cuda:0", "host");
        
        assert!(matches!(info.state, WorkerState::Initializing));
        
        info.state = WorkerState::Running;
        assert!(matches!(info.state, WorkerState::Running));
    }
}

// ============================================================================
// Distributed Checkpointing Tests
// ============================================================================

mod distributed_checkpoint_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_checkpoint_config_creation() {
        let config = CheckpointConfig::new("/tmp/checkpoints");
        assert_eq!(config.checkpoint_dir, PathBuf::from("/tmp/checkpoints"));
    }

    #[test]
    fn test_checkpoint_config_with_options() {
        let config = CheckpointConfig::new("/tmp/ckpt")
            .with_async_save(true)
            .with_max_checkpoints(5)
            .with_save_interval(500, 3600);
        
        assert!(config.async_save);
        assert_eq!(config.max_checkpoints, 5);
        assert_eq!(config.save_interval_steps, 500);
    }

    #[test]
    fn test_checkpoint_config_with_sharding() {
        let config = CheckpointConfig::new("/tmp/ckpt")
            .with_sharding(true)
            .with_compression(6);
        
        assert!(config.sharded);
        assert_eq!(config.compression_level, 6);
    }

    #[test]
    fn test_checkpoint_metadata_creation() {
        let metadata = CheckpointMetadata::new(1000, 5, 0.5, 4, 0);
        
        assert_eq!(metadata.step, 1000);
        assert_eq!(metadata.epoch, 5);
        assert_eq!(metadata.world_size, 4);
        assert_eq!(metadata.rank, 0);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_expert_with_load_balancing() {
        let config = ExpertParallelConfig::new(4)
            .with_capacity_factor(1.25)
            .with_load_balance(true, 0.01);
        
        let mut balancer = LoadBalancer::new(config.clone());
        
        // Simulate token routing
        balancer.update(&[0, 0, 0, 1, 1, 2, 3, 3]);
        
        let imbalance = balancer.compute_imbalance_score();
        assert!(imbalance >= 0.0);
    }

    #[test]
    fn test_sequence_parallel_with_normalization() {
        let device = Device::Cpu;
        let _sp_config = SequenceParallelConfig::new(1, 0);
        
        let norm = DistributedRMSNorm::new(32, 1e-6, &device).unwrap();
        let attention_config = RingAttentionConfig::new(true, 0.0, None);
        let attention = RingAttention::new(attention_config);
        
        let batch = 2;
        let heads = 4;
        let seq_len = 8;
        let head_dim = 8;  // 32 / 4 heads = 8
        
        // Create 4D input for attention
        let input = Tensor::randn(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &device).unwrap();
        
        // Self-attention (ring attention expects 4D)
        let output = attention.forward(&input, &input, &input).unwrap();
        assert_eq!(output.dims(), &[batch, heads, seq_len, head_dim]);
        
        // Test normalization separately with 3D tensors
        let norm_input = Tensor::randn(0.0f32, 1.0, (batch, seq_len, 32), &device).unwrap();
        let normed = norm.forward(&norm_input).unwrap();
        assert_eq!(normed.dims(), &[batch, seq_len, 32]);
    }

    #[test]
    fn test_fault_tolerance_workflow() {
        let elastic_config = ElasticConfig::new(1, 4);
        
        let monitor = HeartbeatMonitor::new(
            Duration::from_secs(5),
            Duration::from_secs(30),
        );
        
        // Register workers
        for i in 0..elastic_config.current_workers {
            let info = WorkerInfo::new(i, &format!("cuda:{}", i), &format!("worker-{}", i));
            monitor.register_worker(info);
            // Set workers to Running state (new workers start in Initializing)
            monitor.set_state(i, WorkerState::Running);
        }
        
        assert_eq!(monitor.worker_count(), elastic_config.current_workers);
        
        // Simulate heartbeats
        for i in 0..elastic_config.current_workers {
            monitor.heartbeat(i, 100);
        }
        
        assert_eq!(monitor.healthy_worker_count(), elastic_config.current_workers);
    }

    #[test]
    fn test_pipeline_config_integration() {
        let config = DualPipeConfig::new(4)
            .with_stages(2, 0)
            .with_checkpointing(true, 2);
        
        assert!(config.activation_checkpointing);
        assert_eq!(config.num_stages, 2);
        assert!(config.is_first_stage());
    }
}

// ============================================================================
// Performance Tests (ignored by default)
// ============================================================================

mod performance_tests {
    use super::*;

    #[test]
    #[ignore]
    fn bench_ring_attention() {
        let config = RingAttentionConfig::new(true, 0.0, None);
        let attention = RingAttention::new(config);
        
        let device = Device::Cpu;
        let batch = 4;
        let heads = 8;
        let seq_len = 256;
        let head_dim = 64;
        
        // Ring attention expects 4D tensors
        let query = Tensor::randn(0.0f32, 0.1, (batch, heads, seq_len, head_dim), &device).unwrap();
        let key = Tensor::randn(0.0f32, 0.1, (batch, heads, seq_len, head_dim), &device).unwrap();
        let value = Tensor::randn(0.0f32, 0.1, (batch, heads, seq_len, head_dim), &device).unwrap();
        
        let start = std::time::Instant::now();
        let iterations = 10;
        
        for _ in 0..iterations {
            let _ = attention.forward(&query, &key, &value).unwrap();
        }
        
        let elapsed = start.elapsed();
        println!(
            "Ring attention: {:?} per iteration",
            elapsed / iterations
        );
    }

    #[test]
    #[ignore]
    fn bench_distributed_norm() {
        let device = Device::Cpu;
        let norm = DistributedRMSNorm::new(4096, 1e-6, &device).unwrap();
        
        let input = Tensor::randn(0.0f32, 1.0, (32, 256, 4096), &device).unwrap();
        
        let start = std::time::Instant::now();
        let iterations = 100;
        
        for _ in 0..iterations {
            let _ = norm.forward(&input).unwrap();
        }
        
        let elapsed = start.elapsed();
        println!(
            "Distributed RMSNorm: {:?} per iteration",
            elapsed / iterations
        );
    }
}
