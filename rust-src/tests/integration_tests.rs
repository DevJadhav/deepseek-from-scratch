use deepseek_from_scratch_in_rust::utils::retry::{RetryPolicy, retry_with_backoff};
use deepseek_from_scratch_in_rust::utils::error::DeepSeekError;
use deepseek_from_scratch_in_rust::model::moe::{DeepSeekMoEV3, DeepSeekMoEV3Config};
use deepseek_from_scratch_in_rust::model::mla::ExtendedRotaryPositionalEncoding;
use deepseek_from_scratch_in_rust::model::mla::RoPEConfig;
use deepseek_from_scratch_in_rust::model::sparse_attention::{DeepSeekSparseAttention, DSAConfig};
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
