//! MoE (Mixture of Experts) benchmarks for DeepSeek Rust implementation
//!
//! Run with: cargo bench --bench moe_bench --no-default-features
//! Profile with: cargo bench --bench moe_bench --profile release-with-debug --no-default-features

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};

// Import the MoE module
use deepseek_rust::model::moe::{
    DeepSeekMoE,
    DeepSeekMoEV3,
    DeepSeekMoEV3Config,
    LoadBalancingState,
    StandardMoE,
    Expert,
};

fn benchmark_expert_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_expert_forward");
    group.sample_size(30);
    
    let device = Device::Cpu;
    
    let d_model = 256;
    let hidden_dim = 512;
    
    for batch_size in [1, 4, 16, 64].iter() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let expert = Expert::new(d_model, hidden_dim, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (*batch_size, d_model), &device).unwrap();
        
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let output = expert.forward(black_box(&x)).unwrap();
                    black_box(output)
                })
            },
        );
    }
    group.finish();
}

fn benchmark_expert_routing(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_routing");
    group.sample_size(20);
    
    let device = Device::Cpu;
    
    let configs = [
        (8, 2, 64),    // (num_experts, top_k, d_model)
        (16, 4, 128),
        (64, 6, 256),
    ];
    
    let batch_size = 32;
    let seq_len = 16;
    
    for (num_experts, top_k, d_model) in configs.iter() {
        // Create router weights (centroids)
        let centroids = Tensor::randn(0f32, 1f32, (*num_experts, *d_model), &device).unwrap();
        let x = Tensor::randn(0f32, 1f32, (batch_size * seq_len, *d_model), &device).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("routing", format!("e{}k{}", num_experts, top_k)),
            &(*num_experts, *top_k),
            |b, &(_, top_k)| {
                b.iter(|| {
                    // Compute routing logits
                    let logits = x.matmul(&centroids.transpose(0, 1).unwrap()).unwrap();
                    
                    // Top-K selection
                    let topk_idx = logits.arg_sort_last_dim(true).unwrap()
                        .narrow(1, 0, top_k).unwrap()
                        .contiguous().unwrap();
                    let topk_vals = logits.gather(&topk_idx, 1).unwrap();
                    
                    // Softmax over top-k
                    let gate = candle_nn::ops::softmax(&topk_vals, 1).unwrap();
                    
                    black_box((topk_idx, gate))
                })
            },
        );
    }
    group.finish();
}

fn benchmark_deepseek_moe(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_deepseek");
    group.sample_size(10);  // Reduce for complex operations
    
    let device = Device::Cpu;
    
    let d_model = 128;
    let n_routed = 8;
    let n_shared = 1;
    let top_k = 2;
    let routed_hidden = 256;
    let shared_hidden = 256;
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let moe = DeepSeekMoE::new(d_model, n_routed, n_shared, top_k, routed_hidden, shared_hidden, vb).unwrap();
    
    for batch_size in [2, 4, 8].iter() {
        let seq_len = 16;
        let x = Tensor::randn(0f32, 1f32, (*batch_size, seq_len, d_model), &device).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("forward", format!("b{}", batch_size)),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let output = moe.forward(black_box(&x)).unwrap();
                    black_box(output)
                })
            },
        );
    }
    group.finish();
}

fn benchmark_deepseek_moe_capacity(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_capacity");
    group.sample_size(10);
    
    let device = Device::Cpu;
    
    let d_model = 128;
    let n_routed = 8;
    let n_shared = 1;
    let top_k = 2;
    let routed_hidden = 256;
    let shared_hidden = 256;
    
    for capacity_factor in [1.0, 1.25, 1.5, 2.0].iter() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let moe = DeepSeekMoE::with_config(
            d_model, n_routed, n_shared, top_k, routed_hidden, shared_hidden,
            *capacity_factor, true, vb
        ).unwrap();
        
        let batch_size = 4;
        let seq_len = 16;
        let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("capacity", format!("cf{}", capacity_factor)),
            capacity_factor,
            |b, _| {
                b.iter(|| {
                    let output = moe.forward_with_capacity(black_box(&x), true).unwrap();
                    black_box(output)
                })
            },
        );
    }
    group.finish();
}

fn benchmark_auxiliary_loss_free(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_aux_loss_free");
    group.sample_size(50);
    
    let device = Device::Cpu;
    
    let num_experts_list = [8, 16, 64];
    
    for num_experts in num_experts_list.iter() {
        let config = DeepSeekMoEV3Config::default();
        
        // Simulate expert counts
        let counts: Vec<f32> = (0..*num_experts).map(|i| (i as f32 + 1.0)).collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_experts),
            num_experts,
            |b, num_experts| {
                b.iter(|| {
                    // Create fresh state each iteration for clean benchmark
                    let mut local_config = config.clone();
                    local_config.n_routed_experts = *num_experts;
                    let mut state = LoadBalancingState::new(&local_config, &device).unwrap();
                    state.update(black_box(&counts), &device).unwrap();
                    // Return shape dims to verify result without borrowing issues
                    black_box(state.get_bias().dims().len())
                })
            },
        );
    }
    group.finish();
}

fn benchmark_standard_moe(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_standard");
    group.sample_size(15);
    
    let device = Device::Cpu;
    
    let d_model = 128;
    let hidden_dim = 256;
    let top_k = 2;
    
    for n_experts in [4, 8, 16].iter() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let moe = StandardMoE::new(d_model, *n_experts, top_k, hidden_dim, vb).unwrap();
        
        let batch_size = 4;
        let seq_len = 16;
        let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("forward", format!("e{}", n_experts)),
            n_experts,
            |b, _| {
                b.iter(|| {
                    let (output, aux_loss) = moe.forward(black_box(&x)).unwrap();
                    black_box((output, aux_loss))
                })
            },
        );
    }
    group.finish();
}

fn benchmark_deepseek_moe_v3(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_v3");
    group.sample_size(10);
    
    let device = Device::Cpu;
    
    // Use small config for CPU benchmarks
    let mut config = DeepSeekMoEV3Config::small_16_2();
    config.d_model = 64;
    config.routed_expert_hidden = 128;
    config.shared_expert_hidden = 128;
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut moe = DeepSeekMoEV3::new(config, vb).unwrap();
    
    for batch_size in [2, 4].iter() {
        let seq_len = 8;
        let x = Tensor::randn(0f32, 1f32, (*batch_size, seq_len, 64), &device).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("hierarchical", format!("b{}", batch_size)),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let output = moe.forward(black_box(&x)).unwrap();
                    black_box(output)
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_expert_forward,
    benchmark_expert_routing,
    benchmark_deepseek_moe,
    benchmark_deepseek_moe_capacity,
    benchmark_auxiliary_loss_free,
    benchmark_standard_moe,
    benchmark_deepseek_moe_v3,
);
criterion_main!(benches);
