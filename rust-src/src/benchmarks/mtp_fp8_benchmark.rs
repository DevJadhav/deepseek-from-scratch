use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use crate::model::mtp::MTPModel;
use crate::model::quantization::FP8Linear;

pub fn run_benchmark() -> Result<()> {
    println!("\n=== Chapter 4 Bonus: MTP & FP8 Benchmarking ===");
    let device = if candle_core::utils::metal_is_available() {
        println!("Using Metal GPU");
        Device::new_metal(0)?
    } else {
        println!("Using CPU");
        Device::Cpu
    };
    
    let batch_size = 4;
    let seq_len = 64;
    let warmup_iters = 5;
    let measure_iters = 20;
    
    println!("Config: batch={}, seq_len={}, warmup={}, measure={} iterations", 
        batch_size, seq_len, warmup_iters, measure_iters);

    // 1. MTP Benchmark
    println!("\n--- Multi-Token Prediction (MTP) ---");
    let vocab_size = 1000;
    let d_model = 512;
    let n_layers = 2;
    let k_predictions = 1;
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mtp_model = MTPModel::new(vocab_size, d_model, n_layers, k_predictions, vb.pp("mtp"))?;
    let input = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
    
    // Warmup
    for _ in 0..warmup_iters {
        let _ = mtp_model.forward(&input)?;
    }
    
    // Measure
    let start = std::time::Instant::now();
    for _ in 0..measure_iters {
        let _ = mtp_model.forward(&input)?;
    }
    let duration = start.elapsed();
    println!("MTP (k=1) Average Time: {:.2?} per forward pass", duration / measure_iters as u32);

    // 2. FP8 Quantization Benchmark
    println!("\n--- FP8 Quantization Simulation ---");
    let in_dim = 512;
    let out_dim = 512;
    let block_size = 128;
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let fp8_linear = FP8Linear::new(in_dim, out_dim, block_size, vb.pp("fp8"))?;
    let input_fp8 = Tensor::randn(0f32, 1f32, (batch_size, seq_len, in_dim), &device)?;
    
    // Warmup
    for _ in 0..warmup_iters {
        let _ = fp8_linear.forward(&input_fp8)?;
    }
    
    // Measure
    let start = std::time::Instant::now();
    for _ in 0..measure_iters {
        let _ = fp8_linear.forward(&input_fp8)?;
    }
    let duration = start.elapsed();
    println!("FP8 Linear Average Time: {:.2?} per forward pass", duration / measure_iters as u32);

    Ok(())
}
