//! Attention mechanism benchmarks for DeepSeek Rust implementation
//!
//! Run with: cargo bench --bench attention_bench --no-default-features
//! Profile with: cargo bench --bench attention_bench --profile release-with-debug --no-default-features

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};

// Import the MLA module
use deepseek_from_scratch_in_rust::model::mla::{
    MultiHeadLatentAttention,
    DeepSeekAttention,
    RotaryPositionalEncoding,
    ExtendedRotaryPositionalEncoding,
    RoPEConfig,
    RoPEScalingType,
};
use deepseek_from_scratch_in_rust::model::kv_cache::{KVCache, LatentKVCache};

fn benchmark_vanilla_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_vanilla");
    
    // Use CPU for benchmarks to avoid CUDA dependency issues
    let device = Device::Cpu;
    
    for seq_len in [128, 256, 512, 1024].iter() {
        group.throughput(Throughput::Elements(*seq_len as u64));
        
        let d_model = 256;
        let num_heads = 8;
        let batch_size = 4;
        
        // Create random Q, K, V tensors
        let q = Tensor::randn(0f32, 1f32, (batch_size, num_heads, *seq_len, d_model / num_heads), &device).unwrap();
        let k = Tensor::randn(0f32, 1f32, (batch_size, num_heads, *seq_len, d_model / num_heads), &device).unwrap();
        let v = Tensor::randn(0f32, 1f32, (batch_size, num_heads, *seq_len, d_model / num_heads), &device).unwrap();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            seq_len,
            |b, &_seq_len| {
                b.iter(|| {
                    // Vanilla attention: softmax(Q @ K^T / sqrt(d)) @ V
                    let scale = 1.0 / ((d_model / num_heads) as f64).sqrt();
                    let scores = q.matmul(&k.transpose(2, 3).unwrap()).unwrap();
                    let scores = (scores * scale).unwrap();
                    let weights = candle_nn::ops::softmax(&scores, 3).unwrap();
                    let output = weights.matmul(&v).unwrap();
                    black_box(output)
                })
            },
        );
    }
    group.finish();
}

fn benchmark_mla_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_mla");
    group.sample_size(20);  // Reduce sample size for complex operations
    
    let device = Device::Cpu;
    
    let configs = [
        (256, 128, 4, 8),  // (d_model, d_latent, num_heads, seq_len)
        (512, 256, 8, 16),
    ];
    
    for (d_model, d_latent, num_heads, batch_size) in configs.iter() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let mla = MultiHeadLatentAttention::new(*d_model, *num_heads, *d_latent, vb).unwrap();
        
        for seq_len in [64, 128, 256].iter() {
            let x = Tensor::randn(0f32, 1f32, (*batch_size, *seq_len, *d_model), &device).unwrap();
            
            group.bench_with_input(
                BenchmarkId::new("mla", format!("d{}l{}s{}", d_model, d_latent, seq_len)),
                &(*d_model, *d_latent, *seq_len),
                |b, _| {
                    b.iter(|| {
                        let output = mla.forward(black_box(&x), None).unwrap();
                        black_box(output)
                    })
                },
            );
        }
    }
    group.finish();
}

fn benchmark_mla_with_latent_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_mla_latent_cache");
    group.sample_size(20);
    
    let device = Device::Cpu;
    
    let d_model = 256;
    let d_latent = 64;
    let num_heads = 4;
    let batch_size = 4;
    let max_seq_len = 512;
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mla = MultiHeadLatentAttention::new(d_model, num_heads, d_latent, vb).unwrap();
    
    // Test incremental generation (1 token at a time)
    for context_len in [64, 128, 256].iter() {
        let mut cache = LatentKVCache::new(batch_size, max_seq_len, d_latent, DType::F32, &device).unwrap();
        
        // Prime the cache with context
        let context = Tensor::randn(0f32, 1f32, (batch_size, *context_len, d_model), &device).unwrap();
        let _ = mla.forward_with_latent_cache(&context, Some(&mut cache)).unwrap();
        
        // Benchmark single token generation
        let single_token = Tensor::randn(0f32, 1f32, (batch_size, 1, d_model), &device).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("incremental", format!("ctx{}", context_len)),
            context_len,
            |b, _| {
                b.iter(|| {
                    let mut cache_clone = LatentKVCache::new(batch_size, max_seq_len, d_latent, DType::F32, &device).unwrap();
                    // Re-prime cache
                    let _ = mla.forward_with_latent_cache(&context, Some(&mut cache_clone)).unwrap();
                    let output = mla.forward_with_latent_cache(black_box(&single_token), Some(&mut cache_clone)).unwrap();
                    black_box(output)
                })
            },
        );
    }
    group.finish();
}

fn benchmark_rope(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope_encoding");
    
    let device = Device::Cpu;
    
    for d_head in [32, 64, 128].iter() {
        let rope = RotaryPositionalEncoding::new(*d_head, 2048, &device).unwrap();
        
        let batch = 4;
        let num_heads = 8;
        let seq_len = 256;
        let x = Tensor::randn(0f32, 1f32, (batch, num_heads, seq_len, *d_head), &device).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("standard", format!("d{}", d_head)),
            d_head,
            |b, _| {
                b.iter(|| {
                    let output = rope.forward(black_box(&x)).unwrap();
                    black_box(output)
                })
            },
        );
    }
    
    // Benchmark extended RoPE with different scaling types
    for scaling_name in ["ntk", "yarn", "linear"].iter() {
        let d_head = 64;
        let scaling_type = match *scaling_name {
            "ntk" => RoPEScalingType::NTKAware { alpha: 8.0 },
            "yarn" => RoPEScalingType::YaRN {
                scale: 4.0,
                original_max_seq_len: 4096,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: 0.1,
            },
            "linear" => RoPEScalingType::Linear { scale: 4.0 },
            _ => RoPEScalingType::None,
        };
        
        let config = RoPEConfig {
            d_head,
            max_seq_len: 32768,
            base: 10000.0,
            scaling_type,
            original_max_seq_len: 4096,
        };
        
        let rope = ExtendedRotaryPositionalEncoding::new(config, &device).unwrap();
        
        let batch = 4;
        let num_heads = 8;
        let seq_len = 256;
        let x = Tensor::randn(0f32, 1f32, (batch, num_heads, seq_len, d_head), &device).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("extended", *scaling_name),
            scaling_name,
            |b, _| {
                b.iter(|| {
                    let output = rope.forward(black_box(&x)).unwrap();
                    black_box(output)
                })
            },
        );
    }
    group.finish();
}

fn benchmark_deepseek_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("deepseek_attention");
    group.sample_size(20);
    
    let device = Device::Cpu;
    
    let d_model = 256;
    let num_heads = 4;
    let d_latent = 64;
    let d_rope = 32;
    let batch_size = 4;
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let attn = DeepSeekAttention::new(d_model, num_heads, d_latent, d_rope, vb).unwrap();
    
    for seq_len in [64, 128, 256].iter() {
        let x = Tensor::randn(0f32, 1f32, (batch_size, *seq_len, d_model), &device).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("decoupled_rope", format!("s{}", seq_len)),
            seq_len,
            |b, _| {
                b.iter(|| {
                    let output = attn.forward(black_box(&x), None).unwrap();
                    black_box(output)
                })
            },
        );
    }
    group.finish();
}

fn benchmark_kv_cache_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_memory");
    
    let device = Device::Cpu;
    let batch_size = 4;
    let n_heads = 8;
    let head_dim = 64;
    let d_model = n_heads * head_dim;
    
    for d_latent in [64, 128, 256].iter() {
        let max_seq_len = 1024;
        
        // Compare latent cache vs full KV cache memory
        let latent_cache = LatentKVCache::new(batch_size, max_seq_len, *d_latent, DType::F32, &device).unwrap();
        let (latent_bytes, full_bytes, ratio) = latent_cache.memory_stats(d_model, n_heads, 4);
        
        group.bench_with_input(
            BenchmarkId::new("compression_ratio", format!("d{}", d_latent)),
            d_latent,
            |b, _| {
                b.iter(|| {
                    black_box((latent_bytes, full_bytes, ratio))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_vanilla_attention,
    benchmark_mla_attention,
    benchmark_mla_with_latent_cache,
    benchmark_rope,
    benchmark_deepseek_attention,
    benchmark_kv_cache_memory,
);
criterion_main!(benches);
