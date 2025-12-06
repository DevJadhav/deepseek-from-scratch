use deepseek_rust::utils::retry::{RetryPolicy, retry_with_backoff};
use deepseek_rust::utils::error::DeepSeekError;
use deepseek_rust::model::moe::{DeepSeekMoEV3, DeepSeekMoEV3Config};
use deepseek_rust::model::mla::ExtendedRotaryPositionalEncoding;
use deepseek_rust::model::mla::RoPEConfig;
use deepseek_rust::model::sparse_attention::{DeepSeekSparseAttention, DSAConfig};
use candle_core::{Device, Tensor, DType};
use candle_nn::{VarMap, VarBuilder};
use std::time::Duration;

#[tokio::test]
async fn test_retry_integration() {
    let policy = RetryPolicy {
        max_retries: 2,
        initial_delay: Duration::from_millis(10),
        ..Default::default()
    };

    let result = retry_with_backoff(|| async {
        Ok::<_, DeepSeekError>("Success")
    }, &policy).await;

    assert_eq!(result.unwrap(), "Success");
}

/// Integration test for DeepSeek-V3.2 components working together
#[test]
fn test_deepseek_v32_integration() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    // Config for testing (small scale)
    let batch_size = 2;
    let seq_len = 32;
    let d_model = 64;
    let num_heads = 4;
    
    // 1. Create Extended RoPE for 128K context
    let rope_config = RoPEConfig {
        d_head: d_model / num_heads,
        max_seq_len: 131072,
        ..Default::default()
    };
    let rope = ExtendedRotaryPositionalEncoding::new(rope_config, &device)
        .expect("Failed to create ExtendedRoPE");
    
    // 2. Create DSA (Sparse Attention)
    let dsa_config = DSAConfig {
        d_model,
        num_heads,
        window_size: 8,
        num_global_tokens: 4,
        max_seq_len: 1024,
        causal: true,
        ..Default::default()
    };
    let mut dsa = DeepSeekSparseAttention::new(dsa_config, vb.pp("dsa"))
        .expect("Failed to create DSA");
    
    // 3. Create MoE V3
    let mut moe_config = DeepSeekMoEV3Config::small_16_2();
    moe_config.d_model = d_model;
    moe_config.routed_expert_hidden = d_model * 2;
    moe_config.shared_expert_hidden = d_model * 2;
    
    let mut moe = DeepSeekMoEV3::new(moe_config, vb.pp("moe"))
        .expect("Failed to create MoE V3");
    
    // Create input tensor
    let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)
        .expect("Failed to create input");
    
    // Test DSA forward pass
    let dsa_out = dsa.forward(&x).expect("DSA forward failed");
    assert_eq!(dsa_out.dims(), &[batch_size, seq_len, d_model]);
    
    // Test MoE forward pass
    let moe_out = moe.forward(&dsa_out).expect("MoE forward failed");
    assert_eq!(moe_out.dims(), &[batch_size, seq_len, d_model]);
    
    // Test RoPE on attention heads (reshape for head dimension)
    let x_heads = x.reshape((batch_size, seq_len, num_heads, d_model / num_heads))
        .expect("Reshape failed")
        .transpose(1, 2)
        .expect("Transpose failed");
    let rope_out = rope.forward(&x_heads).expect("RoPE forward failed");
    assert_eq!(rope_out.dims(), &[batch_size, num_heads, seq_len, d_model / num_heads]);
    
    // Verify capacity metrics exist
    let metrics = moe.get_capacity_metrics();
    // Just verify we can access the metrics (usize is always >= 0)
    let _total = metrics.total_tokens;
    let _dropped = metrics.dropped_tokens;
    
    // Verify load balancing
    let (mean, imbalance, steps) = moe.get_load_balance_stats();
    assert!(mean >= 0.0);
    assert!(imbalance >= 1.0);  // Minimum imbalance is 1.0 (perfect balance)
    assert!(steps >= 0.0);
    
    println!("DeepSeek-V3.2 Integration Test Passed:");
    println!("  - DSA output: {:?}", dsa_out.shape());
    println!("  - MoE output: {:?}", moe_out.shape());
    println!("  - RoPE output: {:?}", rope_out.shape());
    println!("  - Load balance: mean={:.4}, imbalance={:.4}", mean, imbalance);
    println!("  - Capacity: {} tokens, {} dropped", metrics.total_tokens, metrics.dropped_tokens);
}

// ============================================================================
// RouterBiasController Tests (Section 1.2 Auxiliary-Loss-Free Load Balancing)
// ============================================================================

use deepseek_rust::model::moe::{
    RouterBiasController, LoadBalancingState, BIAS_UPDATE_ALPHA_RECOMMENDED
};

/// Test RouterBiasController creation and basic properties
#[test]
fn test_router_bias_controller_creation() {
    let device = Device::Cpu;
    let config = DeepSeekMoEV3Config::small_16_2();
    
    let controller = RouterBiasController::new(&config, &device)
        .expect("Failed to create RouterBiasController");
    
    // Verify initial bias is zeros
    let bias = controller.get_bias();
    let bias_vec = bias.to_vec1::<f32>().unwrap();
    assert_eq!(bias_vec.len(), config.n_routed_experts);
    for &b in &bias_vec {
        assert!((b - 0.0).abs() < 1e-6, "Initial bias should be zero");
    }
    
    // Verify auxiliary loss is disabled
    assert!(!controller.use_auxiliary_loss(), 
        "RouterBiasController should disable auxiliary loss");
    
    // Verify initial step is 0
    assert_eq!(controller.step(), 0);
    
    println!("RouterBiasController creation test passed");
}

/// Test bias updates with uniform expert counts (should result in minimal bias changes)
#[test]
fn test_router_bias_controller_uniform_update() {
    let device = Device::Cpu;
    let config = DeepSeekMoEV3Config::small_16_2();
    
    let mut controller = RouterBiasController::new(&config, &device)
        .expect("Failed to create RouterBiasController");
    
    // Uniform distribution: all experts receive same count
    let uniform_counts: Vec<f32> = vec![10.0; config.n_routed_experts];
    
    // Update with uniform counts
    controller.update_after_batch(&uniform_counts, &device)
        .expect("Update failed");
    
    // Bias should remain close to zero for uniform distribution
    let bias = controller.get_bias();
    let bias_vec = bias.to_vec1::<f32>().unwrap();
    for &b in &bias_vec {
        assert!(b.abs() < 0.1, "Bias should be near zero for uniform distribution, got {}", b);
    }
    
    assert_eq!(controller.step(), 1);
    
    println!("RouterBiasController uniform update test passed");
}

/// Test bias updates with imbalanced expert counts (should adjust biases)
#[test]
fn test_router_bias_controller_imbalanced_update() {
    let device = Device::Cpu;
    let mut config = DeepSeekMoEV3Config::small_16_2();
    config.bias_lr = 0.1; // Higher LR for visible effect
    
    let mut controller = RouterBiasController::new(&config, &device)
        .expect("Failed to create RouterBiasController");
    
    // Imbalanced: first expert gets all tokens
    let n_experts = config.n_routed_experts;
    let mut imbalanced_counts = vec![0.0; n_experts];
    imbalanced_counts[0] = 100.0; // First expert is overloaded
    
    // Update multiple times to build up bias
    for _ in 0..10 {
        controller.update_after_batch(&imbalanced_counts, &device)
            .expect("Update failed");
    }
    
    // Overloaded expert (index 0) should have negative bias (discourage selection)
    // Underloaded experts should have positive bias (encourage selection)
    let bias = controller.get_bias();
    let bias_vec = bias.to_vec1::<f32>().unwrap();
    
    assert!(bias_vec[0] < 0.0, 
        "Overloaded expert should have negative bias, got {}", bias_vec[0]);
    
    // At least some underloaded experts should have positive bias
    let positive_biases: Vec<&f32> = bias_vec[1..].iter().filter(|&&b| b > 0.0).collect();
    assert!(!positive_biases.is_empty(), 
        "Some underloaded experts should have positive bias");
    
    println!("RouterBiasController imbalanced update test passed");
    println!("  Overloaded expert bias: {:.4}", bias_vec[0]);
    println!("  Underloaded expert bias (example): {:.4}", bias_vec[1]);
}

/// Test bias clamping
#[test]
fn test_router_bias_controller_clamping() {
    let device = Device::Cpu;
    let mut config = DeepSeekMoEV3Config::small_16_2();
    config.bias_lr = 1.0; // Very high LR to trigger clamping
    config.bias_clamp = 2.0;
    
    let mut controller = RouterBiasController::new(&config, &device)
        .expect("Failed to create RouterBiasController");
    
    // Extreme imbalance
    let n_experts = config.n_routed_experts;
    let mut extreme_counts = vec![0.0; n_experts];
    extreme_counts[0] = 10000.0;
    
    // Many updates with extreme imbalance
    for _ in 0..100 {
        controller.update_after_batch(&extreme_counts, &device)
            .expect("Update failed");
    }
    
    // Verify all biases are within clamp range
    let bias = controller.get_bias();
    let bias_vec = bias.to_vec1::<f32>().unwrap();
    for &b in &bias_vec {
        assert!(b >= -config.bias_clamp && b <= config.bias_clamp,
            "Bias {} should be clamped to [-{}, {}]", b, config.bias_clamp, config.bias_clamp);
    }
    
    println!("RouterBiasController clamping test passed");
}

/// Test that BIAS_UPDATE_ALPHA_RECOMMENDED constant is correct
#[test]
fn test_bias_update_alpha_constant() {
    assert!((BIAS_UPDATE_ALPHA_RECOMMENDED - 0.001).abs() < 1e-9,
        "BIAS_UPDATE_ALPHA_RECOMMENDED should be 0.001");
    
    println!("BIAS_UPDATE_ALPHA_RECOMMENDED = {}", BIAS_UPDATE_ALPHA_RECOMMENDED);
}

/// Test load balancing statistics
#[test]
fn test_router_bias_controller_stats() {
    let device = Device::Cpu;
    let config = DeepSeekMoEV3Config::small_16_2();
    
    let mut controller = RouterBiasController::new(&config, &device)
        .expect("Failed to create RouterBiasController");
    
    // Add some data
    let counts: Vec<f32> = (0..config.n_routed_experts)
        .map(|i| (i + 1) as f32)
        .collect();
    
    controller.update_after_batch(&counts, &device).unwrap();
    
    let (mean, imbalance, steps) = controller.get_stats();
    
    assert!(mean > 0.0, "Mean should be positive");
    assert!(imbalance >= 1.0, "Imbalance should be >= 1.0");
    assert_eq!(steps, 1.0, "Steps should be 1");
    
    let detailed = controller.get_detailed_stats();
    assert!(detailed.mean_count > 0.0);
    assert_eq!(detailed.step, 1);
    
    println!("RouterBiasController stats test passed");
    println!("  Mean: {:.4}, Imbalance: {:.4}, Steps: {}", mean, imbalance, steps);
}

/// Test LoadBalancingState directly (underlying implementation)
#[test]
fn test_load_balancing_state_direct() {
    let device = Device::Cpu;
    let config = DeepSeekMoEV3Config::small_16_2();
    
    let mut state = LoadBalancingState::new(&config, &device)
        .expect("Failed to create LoadBalancingState");
    
    // Test EMA decay
    let counts: Vec<f32> = vec![1.0; config.n_routed_experts];
    state.update(&counts, &device).unwrap();
    
    let bias = state.get_bias();
    assert_eq!(bias.dims(), &[config.n_routed_experts]);
    
    println!("LoadBalancingState direct test passed");
}
