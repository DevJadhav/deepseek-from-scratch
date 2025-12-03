use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use crate::model::moe::{DeepSeekMoE, StandardMoE};

// Configuration struct
#[derive(Clone, Copy)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub vocab_size: usize,
    pub block_size: usize,
    pub dropout: f32,
    // MoE params
    pub moe_n_routed_experts: usize,
    pub moe_top_k: usize,
    pub moe_expert_hidden_dim: usize,
    pub ds_moe_n_shared_experts: usize,
    pub ds_moe_shared_expert_hidden_dim: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_layer: 2, // Reduced for benchmark speed
            n_head: 8,
            n_embd: 512,
            vocab_size: 50257,
            block_size: 256,
            dropout: 0.1,
            moe_n_routed_experts: 16,
            moe_top_k: 2,
            moe_expert_hidden_dim: 512,
            ds_moe_n_shared_experts: 2,
            ds_moe_shared_expert_hidden_dim: 1024,
        }
    }
}

// Wrapper for Standard MoE to fit common interface (ignoring aux loss for timing)
struct StandardMoEWrapper {
    moe: StandardMoE,
}

impl StandardMoEWrapper {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // Adjust n_routed to match parameter count roughly if needed, 
        // but for now we stick to config or the notebook's 22.
        // Notebook uses 22 for Standard to match DeepSeek's 16+2(large).
        let n_routed = 22; 
        let moe = StandardMoE::new(
            cfg.n_embd,
            n_routed,
            cfg.moe_top_k,
            cfg.moe_expert_hidden_dim,
            vb,
        )?;
        Ok(Self { moe })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (out, _) = self.moe.forward(x)?;
        Ok(out)
    }
}

pub fn run_benchmark() -> Result<()> {
    println!("\n=== Chapter 3 Bonus: MoE Benchmarking ===");
    let device = if candle_core::utils::metal_is_available() {
        println!("Using Metal GPU");
        Device::new_metal(0)?
    } else {
        println!("Using CPU");
        Device::Cpu
    };
    let cfg = Config::default();
    
    let batch_size = 4;
    let seq_len = 64;
    let warmup_iters = 5;
    let measure_iters = 20;
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, cfg.n_embd), &device)?;
    
    println!("Config: batch={}, seq_len={}, d_model={}, warmup={}, measure={} iterations", 
        batch_size, seq_len, cfg.n_embd, warmup_iters, measure_iters);

    // Benchmark Standard MoE
    println!("\nBenchmarking Standard MoE...");
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let std_moe = StandardMoEWrapper::new(&cfg, vb.pp("std_moe"))?;
    
    // Warmup
    for _ in 0..warmup_iters {
        let _ = std_moe.forward(&input)?;
    }
    
    let start = std::time::Instant::now();
    for _ in 0..measure_iters {
        let _ = std_moe.forward(&input)?;
    }
    let duration = start.elapsed();
    println!("Standard MoE Average Time: {:.2?} per forward pass", duration / measure_iters as u32);

    // Benchmark DeepSeek MoE
    println!("\nBenchmarking DeepSeek MoE...");
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let ds_moe = DeepSeekMoE::new(
        cfg.n_embd,
        cfg.moe_n_routed_experts,
        cfg.ds_moe_n_shared_experts,
        cfg.moe_top_k,
        cfg.moe_expert_hidden_dim,
        cfg.ds_moe_shared_expert_hidden_dim,
        vb.pp("ds_moe"),
    )?;

    // Warmup
    for _ in 0..warmup_iters {
        let _ = ds_moe.forward(&input)?;
    }

    let start = std::time::Instant::now();
    for _ in 0..measure_iters {
        let _ = ds_moe.forward(&input)?;
    }
    let duration = start.elapsed();
    println!("DeepSeek MoE Average Time: {:.2?} per forward pass", duration / measure_iters as u32);

    Ok(())
}
