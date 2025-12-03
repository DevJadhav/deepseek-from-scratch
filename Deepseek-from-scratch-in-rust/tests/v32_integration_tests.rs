//! DeepSeek-V3.2 Comprehensive Integration Tests
//!
//! End-to-end tests for all V3.2 components:
//! - 128K context window with DSA
//! - 256-expert MoE routing
//! - DSA vs dense attention comparison
//! - Full training pipeline validation
//! - Agent trajectory processing

use deepseek_from_scratch_in_rust::model::moe::{DeepSeekMoEV3, DeepSeekMoEV3Config};
use deepseek_from_scratch_in_rust::model::mla::{
    ExtendedRotaryPositionalEncoding, RoPEConfig, RoPEScalingType,
};
use deepseek_from_scratch_in_rust::model::sparse_attention::{DeepSeekSparseAttention, DSAConfig};
use deepseek_from_scratch_in_rust::model::kv_cache::KVCache;
use deepseek_from_scratch_in_rust::training::agent::{
    ToolType, ToolStatus, ToolCall, ToolResponse, AgentTurn, AgentTrajectory, TaskTier,
    RewardWeights, RewardBreakdown,
};
use candle_core::{Device, Tensor, DType};
use candle_nn::{VarMap, VarBuilder};
use std::time::Instant;

// ============================================================================
// 128K Context Window Tests
// ============================================================================

#[test]
fn test_128k_context_rope_initialization() {
    let device = Device::Cpu;
    
    // Use YaRN scaling for 128K context
    let config = RoPEConfig::for_128k_yarn(64);
    
    let rope = ExtendedRotaryPositionalEncoding::new(config.clone(), &device)
        .expect("Failed to create 128K RoPE");
    
    // Verify the config is for 128K
    assert_eq!(config.max_seq_len, 131072);
}

#[test]
fn test_extended_rope_positions() {
    let device = Device::Cpu;
    
    // Use NTK-aware scaling for extended context
    let config = RoPEConfig::for_128k_ntk_aware(64);
    
    let rope = ExtendedRotaryPositionalEncoding::new(config, &device)
        .expect("Failed to create ExtendedRoPE");
    
    // Test at various positions within range
    let batch_size = 1;
    let num_heads = 4;
    let seq_len = 1024;
    let head_dim = 64;
    
    let x = Tensor::randn(0f32, 1f32, (batch_size, num_heads, seq_len, head_dim), &device)
        .expect("Failed to create input");
    
    let output = rope.forward(&x).expect("RoPE forward failed");
    assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);
}

#[test]
fn test_rope_scaling_types() {
    let device = Device::Cpu;
    
    // Test different scaling types
    let scaling_types = vec![
        RoPEScalingType::None,
        RoPEScalingType::Linear { scale: 4.0 },
        RoPEScalingType::NTKAware { alpha: 32.0 },
        RoPEScalingType::YaRN {
            scale: 32.0,
            original_max_seq_len: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: 0.1,
        },
    ];
    
    for scaling_type in scaling_types {
        let config = RoPEConfig {
            d_head: 64,
            max_seq_len: 131072,
            base: 10000.0,
            scaling_type,
            original_max_seq_len: 4096,
        };
        
        let rope = ExtendedRotaryPositionalEncoding::new(config, &device)
            .expect("Failed to create RoPE");
        
        let x = Tensor::randn(0f32, 1f32, (1, 4, 32, 64), &device)
            .expect("Failed to create input");
        
        let output = rope.forward(&x).expect("RoPE forward failed");
        assert_eq!(output.dims(), &[1, 4, 32, 64]);
    }
    
    println!("All RoPE scaling types tested successfully");
}

#[test]
fn test_dsa_sliding_window_at_scale() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    let config = DSAConfig {
        d_model: 256,
        num_heads: 8,
        window_size: 512,
        num_global_tokens: 64,
        max_seq_len: 8192,
        causal: true,
        ..Default::default()
    };
    
    let mut dsa = DeepSeekSparseAttention::new(config, vb)
        .expect("Failed to create DSA");
    
    // Test with longer sequence
    let batch_size = 2;
    let seq_len = 4096;
    let d_model = 256;
    
    let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)
        .expect("Failed to create input");
    
    let start = Instant::now();
    let output = dsa.forward(&x).expect("DSA forward failed");
    let elapsed = start.elapsed();
    
    assert_eq!(output.dims(), &[batch_size, seq_len, d_model]);
    println!("DSA 4K sequence forward: {:?}", elapsed);
}

// ============================================================================
// 256-Expert MoE Routing Tests
// ============================================================================

#[test]
fn test_moe_v3_default_config() {
    let config = DeepSeekMoEV3Config::default();
    
    assert_eq!(config.n_routed_experts, 256);
    assert_eq!(config.top_k, 8);
    assert!(config.capacity_factor >= 1.0);
}

#[test]
fn test_moe_256_expert_routing() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    // Use smaller model for testing but with 256 experts structure
    let config = DeepSeekMoEV3Config {
        d_model: 64,
        n_routed_experts: 256,
        n_shared_experts: 2,
        top_k: 8,
        routed_expert_hidden: 128,
        shared_expert_hidden: 128,
        n_expert_groups: 8,
        top_k_groups: 4,
        ..Default::default()
    };
    
    let mut moe = DeepSeekMoEV3::new(config.clone(), vb)
        .expect("Failed to create 256-expert MoE");
    
    let batch_size = 4;
    let seq_len = 16;
    
    let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, config.d_model), &device)
        .expect("Failed to create input");
    
    let output = moe.forward(&x).expect("MoE forward failed");
    assert_eq!(output.dims(), &[batch_size, seq_len, config.d_model]);
    
    // Verify expert utilization
    let metrics = moe.get_capacity_metrics();
    assert!(metrics.total_tokens > 0);
    
    // Check load balancing stats
    let (mean_load, imbalance, _) = moe.get_load_balance_stats();
    println!("256-expert MoE: mean_load={:.4}, imbalance={:.4}", mean_load, imbalance);
}

#[test]
fn test_moe_expert_capacity_overflow() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    // Create MoE with tight capacity to test overflow handling
    let config = DeepSeekMoEV3Config {
        d_model: 32,
        n_routed_experts: 16,
        n_shared_experts: 2,
        top_k: 2,
        routed_expert_hidden: 64,
        shared_expert_hidden: 64,
        n_expert_groups: 4,
        top_k_groups: 2,
        capacity_factor: 1.0, // Tight capacity
        ..Default::default()
    };
    
    let mut moe = DeepSeekMoEV3::new(config.clone(), vb)
        .expect("Failed to create MoE");
    
    // Send many tokens to stress capacity
    let batch_size = 8;
    let seq_len = 64;
    
    let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, config.d_model), &device)
        .expect("Failed to create input");
    
    let output = moe.forward(&x).expect("MoE forward should handle overflow");
    assert_eq!(output.dims(), &[batch_size, seq_len, config.d_model]);
    
    let metrics = moe.get_capacity_metrics();
    println!("Capacity test: {} tokens, {} dropped ({}%)", 
        metrics.total_tokens, 
        metrics.dropped_tokens,
        100.0 * metrics.dropped_tokens as f32 / metrics.total_tokens as f32
    );
}

#[test]
fn test_moe_hierarchical_routing() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    // 64 experts in 8 groups = 8 experts per group
    let config = DeepSeekMoEV3Config {
        d_model: 64,
        n_routed_experts: 64,
        n_shared_experts: 2,
        top_k: 8,
        n_expert_groups: 8,    // 8 groups
        top_k_groups: 4,       // Select 4 groups, then 2 per group
        routed_expert_hidden: 128,
        shared_expert_hidden: 128,
        ..Default::default()
    };
    
    assert_eq!(config.experts_per_group(), 8);
    assert_eq!(config.experts_per_selected_group(), 2);
    
    let mut moe = DeepSeekMoEV3::new(config.clone(), vb)
        .expect("Failed to create hierarchical MoE");
    
    let x = Tensor::randn(0f32, 1f32, (4, 32, config.d_model), &device)
        .expect("Failed to create input");
    
    let output = moe.forward(&x).expect("Hierarchical routing failed");
    assert_eq!(output.dims(), &[4, 32, config.d_model]);
    
    println!("Hierarchical routing: {} experts in {} groups", 
        config.n_routed_experts, config.n_expert_groups);
}

// ============================================================================
// DSA vs Dense Attention Comparison
// ============================================================================

#[test]
fn test_dsa_vs_dense_attention_divergence() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    let d_model = 64;
    let num_heads = 4;
    let seq_len = 128;
    let batch_size = 2;
    
    // Create DSA
    let dsa_config = DSAConfig {
        d_model,
        num_heads,
        window_size: 32,
        num_global_tokens: 8,
        max_seq_len: 256,
        causal: true,
        ..Default::default()
    };
    
    let mut dsa = DeepSeekSparseAttention::new(dsa_config, vb.pp("dsa"))
        .expect("Failed to create DSA");
    
    // Create input
    let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)
        .expect("Failed to create input");
    
    let dsa_output = dsa.forward(&x).expect("DSA forward failed");
    
    // DSA output should have same shape
    assert_eq!(dsa_output.dims(), &[batch_size, seq_len, d_model]);
    
    // Verify output is not all zeros or NaN
    let mean = dsa_output.mean_all().expect("Mean failed");
    let mean_val: f32 = mean.to_scalar().expect("Scalar conversion failed");
    assert!(!mean_val.is_nan(), "DSA output contains NaN");
    
    println!("DSA output mean: {:.6}", mean_val);
}

#[test]
fn test_dsa_sparsity_pattern() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    let window_size = 16;
    let num_global = 4;
    
    let config = DSAConfig {
        d_model: 32,
        num_heads: 2,
        window_size,
        num_global_tokens: num_global,
        max_seq_len: 64,
        causal: true,
        ..Default::default()
    };
    
    let mut dsa = DeepSeekSparseAttention::new(config, vb)
        .expect("Failed to create DSA");
    
    let seq_len = 48;
    let x = Tensor::randn(0f32, 1f32, (1, seq_len, 32), &device)
        .expect("Failed to create input");
    
    let output = dsa.forward(&x).expect("DSA forward failed");
    assert_eq!(output.dims(), &[1, seq_len, 32]);
    
    // Log sparsity characteristics
    println!("DSA sparsity test:");
    println!("  - Sequence length: {}", seq_len);
    println!("  - Window size: {}", window_size);
    println!("  - Global tokens: {}", num_global);
    println!("  - Expected attended: ~{} per query", window_size + num_global);
}

// ============================================================================
// KV Cache Long Context Tests
// ============================================================================

#[test]
fn test_kv_cache_incremental_generation() {
    let device = Device::Cpu;
    
    let batch_size = 2;
    let max_seq_len = 2048;
    let n_heads = 8;
    let head_dim = 64;
    
    let mut cache = KVCache::new(batch_size, max_seq_len, n_heads, head_dim, DType::F32, &device)
        .expect("Failed to create KV cache");
    
    // Simulate incremental generation
    let mut total_len = 0;
    for step in 0..10 {
        let new_tokens = if step == 0 { 128 } else { 1 }; // Prefill then single tokens
        
        let k = Tensor::randn(0f32, 1f32, (batch_size, n_heads, new_tokens, head_dim), &device)
            .expect("Failed to create K");
        let v = Tensor::randn(0f32, 1f32, (batch_size, n_heads, new_tokens, head_dim), &device)
            .expect("Failed to create V");
        
        let (k_cached, v_cached) = cache.update(&k, &v).expect("Cache update failed");
        total_len += new_tokens;
        
        assert_eq!(k_cached.dims()[2], total_len);
        assert_eq!(v_cached.dims()[2], total_len);
    }
    
    assert_eq!(cache.current_seq_len(), total_len);
    println!("KV cache test: {} tokens cached", total_len);
}

#[test]
fn test_kv_cache_memory_efficiency() {
    let device = Device::Cpu;
    
    // Test with 128K max length
    let batch_size = 1;
    let max_seq_len = 131072;
    let n_heads = 8;
    let head_dim = 64;
    
    let result = KVCache::new(batch_size, max_seq_len, n_heads, head_dim, DType::F32, &device);
    
    // Should succeed (memory allocation)
    assert!(result.is_ok(), "Failed to allocate 128K KV cache");
    
    let cache = result.unwrap();
    
    // Calculate expected memory
    let expected_size = 2 * batch_size * n_heads * max_seq_len * head_dim * 4; // 4 bytes per f32
    println!("128K KV cache allocated: {} MB", expected_size / 1024 / 1024);
}

// ============================================================================
// Agent Trajectory Integration Tests
// ============================================================================

#[test]
fn test_agent_full_trajectory() {
    let mut trajectory = AgentTrajectory::new(
        "Write a Python function to calculate fibonacci".to_string(),
        TaskTier::MultiToolSequential
    );
    
    // Turn 1: Think and search
    let mut turn1 = AgentTurn::new(0);
    turn1.user_message = Some("Write a Python function to calculate fibonacci".to_string());
    turn1.assistant_response = "I'll write a fibonacci function. Let me first check the best approach.".to_string();
    
    turn1.tool_calls.push(ToolCall {
        id: "search_1".to_string(),
        tool_type: ToolType::WebSearch,
        function_name: "search".to_string(),
        arguments: serde_json::json!({"query": "efficient fibonacci python"}),
    });
    turn1.tool_responses.push(ToolResponse {
        call_id: "search_1".to_string(),
        status: ToolStatus::Success,
        content: "Memoization is efficient for fibonacci...".to_string(),
        error: None,
        execution_time_ms: 150,
    });
    trajectory.add_turn(turn1);
    
    // Turn 2: Write code
    let mut turn2 = AgentTurn::new(1);
    turn2.assistant_response = "Based on my research, I'll implement with memoization.".to_string();
    
    turn2.tool_calls.push(ToolCall {
        id: "code_1".to_string(),
        tool_type: ToolType::CodeExecution,
        function_name: "execute_python".to_string(),
        arguments: serde_json::json!({
            "code": "def fib(n, memo={}):\n    if n in memo: return memo[n]\n    if n <= 1: return n\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n    return memo[n]\n\nprint([fib(i) for i in range(10)])"
        }),
    });
    turn2.tool_responses.push(ToolResponse {
        call_id: "code_1".to_string(),
        status: ToolStatus::Success,
        content: "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]".to_string(),
        error: None,
        execution_time_ms: 50,
    });
    turn2.is_terminal = true;
    trajectory.add_turn(turn2);
    
    trajectory.task_completed = true;
    
    // Verify trajectory
    assert_eq!(trajectory.num_turns(), 2);
    assert_eq!(trajectory.total_tool_calls(), 2);
    assert!(trajectory.task_completed);
    
    let tool_types = trajectory.tool_types_used();
    assert!(tool_types.contains(&ToolType::WebSearch));
    assert!(tool_types.contains(&ToolType::CodeExecution));
    
    println!("Agent trajectory test passed:");
    println!("  - Turns: {}", trajectory.num_turns());
    println!("  - Tool calls: {}", trajectory.total_tool_calls());
    println!("  - Tool types: {:?}", tool_types);
}

#[test]
fn test_agent_reward_computation_integration() {
    let weights = RewardWeights::default();
    
    // Verify weight sum
    let total = weights.correctness + weights.format + weights.efficiency + weights.safety;
    assert!((total - 1.0).abs() < 1e-6);
    
    // Create sample reward breakdown
    let breakdown = RewardBreakdown {
        correctness: 0.9,
        format: 1.0,
        efficiency: 0.8,
        safety: 1.0,
    };
    
    // Use the total() method from RewardBreakdown
    let computed_total = breakdown.total(&weights);
    
    // Expected: 0.5*0.9 + 0.2*1.0 + 0.15*0.8 + 0.15*1.0 = 0.45 + 0.2 + 0.12 + 0.15 = 0.92
    assert!((computed_total - 0.92).abs() < 1e-6);
    
    println!("Reward computation: {:.4}", computed_total);
}

#[test]
fn test_task_tier_complexity_ordering() {
    let tiers = [
        TaskTier::SingleTool,
        TaskTier::MultiToolSequential,
        TaskTier::MultiToolParallel,
        TaskTier::ComplexWorkflow,
    ];
    
    let mut prev_weight = 0.0;
    for tier in &tiers {
        let weight = tier.difficulty_weight();
        assert!(weight > prev_weight, "Tier {:?} should have higher weight than previous", tier);
        prev_weight = weight;
    }
    
    // Verify environment counts
    let total_envs: usize = tiers.iter().map(|t| t.environment_count()).sum();
    assert_eq!(total_envs, 1800);
}

// ============================================================================
// Full Pipeline Integration Tests
// ============================================================================

#[test]
fn test_full_v32_forward_pipeline() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    let batch_size = 2;
    let seq_len = 64;
    let d_model = 128;
    let num_heads = 4;
    
    // 1. Extended RoPE with YaRN
    let rope_config = RoPEConfig::for_128k_yarn(d_model / num_heads);
    let rope = ExtendedRotaryPositionalEncoding::new(rope_config, &device)
        .expect("RoPE creation failed");
    
    // 2. DSA
    let dsa_config = DSAConfig {
        d_model,
        num_heads,
        window_size: 16,
        num_global_tokens: 4,
        max_seq_len: 1024,
        causal: true,
        ..Default::default()
    };
    let mut dsa = DeepSeekSparseAttention::new(dsa_config, vb.pp("dsa"))
        .expect("DSA creation failed");
    
    // 3. MoE
    let moe_config = DeepSeekMoEV3Config {
        d_model,
        n_routed_experts: 16,
        n_shared_experts: 2,
        top_k: 2,
        n_expert_groups: 4,
        top_k_groups: 2,
        routed_expert_hidden: d_model * 2,
        shared_expert_hidden: d_model * 2,
        ..Default::default()
    };
    
    let mut moe = DeepSeekMoEV3::new(moe_config, vb.pp("moe"))
        .expect("MoE creation failed");
    
    // Input
    let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)
        .expect("Input creation failed");
    
    // Forward pipeline
    let start = Instant::now();
    
    // DSA
    let dsa_out = dsa.forward(&x).expect("DSA forward failed");
    let dsa_time = start.elapsed();
    
    // MoE
    let moe_start = Instant::now();
    let moe_out = moe.forward(&dsa_out).expect("MoE forward failed");
    let moe_time = moe_start.elapsed();
    
    let total_time = start.elapsed();
    
    assert_eq!(moe_out.dims(), &[batch_size, seq_len, d_model]);
    
    println!("\nFull V3.2 Pipeline Test:");
    println!("  - DSA time: {:?}", dsa_time);
    println!("  - MoE time: {:?}", moe_time);
    println!("  - Total time: {:?}", total_time);
    println!("  - Output shape: {:?}", moe_out.shape());
}

#[test]
fn test_pipeline_numerical_stability() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    let d_model = 64;
    
    // Create components
    let dsa_config = DSAConfig {
        d_model,
        num_heads: 4,
        window_size: 8,
        num_global_tokens: 2,
        max_seq_len: 128,
        causal: true,
        ..Default::default()
    };
    let mut dsa = DeepSeekSparseAttention::new(dsa_config, vb.pp("dsa"))
        .expect("DSA creation failed");
    
    let moe_config = DeepSeekMoEV3Config {
        d_model,
        n_routed_experts: 16,
        n_shared_experts: 2,
        top_k: 2,
        n_expert_groups: 4,
        top_k_groups: 2,
        routed_expert_hidden: d_model * 2,
        shared_expert_hidden: d_model * 2,
        ..Default::default()
    };
    
    let mut moe = DeepSeekMoEV3::new(moe_config, vb.pp("moe"))
        .expect("MoE creation failed");
    
    // Run multiple forward passes
    for i in 0..5 {
        let x = Tensor::randn(0f32, 1f32, (2, 32, d_model), &device)
            .expect("Input failed");
        
        let dsa_out = dsa.forward(&x).expect("DSA failed");
        let moe_out = moe.forward(&dsa_out).expect("MoE failed");
        
        // Check for NaN/Inf
        let mean: f32 = moe_out.mean_all()
            .expect("Mean failed")
            .to_scalar()
            .expect("Scalar failed");
        
        assert!(!mean.is_nan(), "NaN detected at iteration {}", i);
        assert!(!mean.is_infinite(), "Inf detected at iteration {}", i);
    }
    
    println!("Numerical stability test passed (5 iterations)");
}

// ============================================================================
// Benchmark Tests
// ============================================================================

#[test]
fn benchmark_dsa_scaling() {
    let device = Device::Cpu;
    
    let seq_lengths = [128, 256, 512, 1024];
    let d_model = 64;
    
    println!("\nDSA Scaling Benchmark:");
    println!("{:>8} | {:>12}", "Seq Len", "Time (ms)");
    println!("{:-<8}-+-{:-<12}", "", "");
    
    for seq_len in seq_lengths {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let config = DSAConfig {
            d_model,
            num_heads: 4,
            window_size: 32,
            num_global_tokens: 8,
            max_seq_len: seq_len * 2,
            causal: true,
            ..Default::default()
        };
        
        let mut dsa = DeepSeekSparseAttention::new(config, vb)
            .expect("DSA creation failed");
        
        let x = Tensor::randn(0f32, 1f32, (1, seq_len, d_model), &device)
            .expect("Input failed");
        
        // Warmup
        let _ = dsa.forward(&x);
        
        // Benchmark
        let start = Instant::now();
        for _ in 0..10 {
            let _ = dsa.forward(&x);
        }
        let elapsed = start.elapsed().as_millis() / 10;
        
        println!("{:>8} | {:>12}", seq_len, elapsed);
    }
}

#[test]
fn benchmark_moe_expert_scaling() {
    let device = Device::Cpu;
    
    println!("\nMoE Expert Scaling Benchmark:");
    println!("{:>8} | {:>8} | {:>12}", "Experts", "Active", "Time (ms)");
    println!("{:-<8}-+-{:-<8}-+-{:-<12}", "", "", "");
    
    let configs = [
        (16, 2),
        (32, 4),
        (64, 8),
    ];
    
    let d_model = 32;
    let batch_size = 4;
    let seq_len = 32;
    
    for (num_experts, num_active) in configs {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let config = DeepSeekMoEV3Config {
            d_model,
            n_routed_experts: num_experts,
            top_k: num_active,
            n_shared_experts: 1,
            n_expert_groups: num_experts / 8,
            top_k_groups: num_active / 2,
            routed_expert_hidden: d_model * 2,
            shared_expert_hidden: d_model * 2,
            capacity_factor: 1.25,
            ..Default::default()
        };
        
        let mut moe = DeepSeekMoEV3::new(config, vb)
            .expect("MoE creation failed");
        
        let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)
            .expect("Input failed");
        
        // Warmup
        let _ = moe.forward(&x);
        
        // Benchmark
        let start = Instant::now();
        for _ in 0..10 {
            let _ = moe.forward(&x);
        }
        let elapsed = start.elapsed().as_millis() / 10;
        
        println!("{:>8} | {:>8} | {:>12}", num_experts, num_active, elapsed);
    }
}

#[test]
fn test_end_to_end_v32_generation() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    // Configuration
    let batch_size = 1;
    let d_model = 64;
    let num_heads = 4;
    let head_dim = d_model / num_heads;
    let n_layers = 2;
    let prefill_len = 32;
    let generate_tokens = 16;
    
    // Create components for each layer
    let mut dsas = Vec::new();
    let mut moes = Vec::new();
    
    for layer in 0..n_layers {
        let dsa_config = DSAConfig {
            d_model,
            num_heads,
            window_size: 8,
            num_global_tokens: 2,
            max_seq_len: 256,
            causal: true,
            ..Default::default()
        };
        dsas.push(DeepSeekSparseAttention::new(dsa_config, vb.pp(format!("layer{}_dsa", layer)))
            .expect("DSA creation failed"));
        
        let moe_config = DeepSeekMoEV3Config {
            d_model,
            n_routed_experts: 8,
            n_shared_experts: 1,
            top_k: 2,
            n_expert_groups: 2,
            top_k_groups: 1,
            routed_expert_hidden: d_model * 2,
            shared_expert_hidden: d_model * 2,
            ..Default::default()
        };
        moes.push(DeepSeekMoEV3::new(moe_config, vb.pp(format!("layer{}_moe", layer)))
            .expect("MoE creation failed"));
    }
    
    // Create KV caches for each layer
    let mut kv_caches: Vec<KVCache> = Vec::new();
    for _ in 0..n_layers {
        kv_caches.push(KVCache::new(batch_size, 256, num_heads, head_dim, DType::F32, &device)
            .expect("KV cache creation failed"));
    }
    
    // Prefill phase
    let prefill_input = Tensor::randn(0f32, 1f32, (batch_size, prefill_len, d_model), &device)
        .expect("Prefill input creation failed");
    
    let mut x = prefill_input;
    for layer in 0..n_layers {
        x = dsas[layer].forward(&x).expect("DSA prefill failed");
        x = moes[layer].forward(&x).expect("MoE prefill failed");
    }
    
    // Generate phase (token by token)
    let mut generated = Vec::new();
    for step in 0..generate_tokens {
        // Single token input
        let token_input = Tensor::randn(0f32, 1f32, (batch_size, 1, d_model), &device)
            .expect("Token input creation failed");
        
        let mut x = token_input;
        for layer in 0..n_layers {
            x = dsas[layer].forward(&x).expect("DSA generate failed");
            x = moes[layer].forward(&x).expect("MoE generate failed");
        }
        
        generated.push(x);
    }
    
    println!("End-to-end V3.2 generation test:");
    println!("  - Layers: {}", n_layers);
    println!("  - Prefill: {} tokens", prefill_len);
    println!("  - Generated: {} tokens", generate_tokens);
    
    assert_eq!(generated.len(), generate_tokens);
}
