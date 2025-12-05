use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{VarBuilder, VarMap, Module};
use crate::training::grpo::GRPOTrainer;
use crate::model::r1::ReasoningModel;
use crate::training::distillation::{kd_loss_kl, kd_loss_mse, kd_loss_jsd};

/// Run comprehensive benchmarks for Chapter 5-9 components
pub fn run_benchmark() -> Result<()> {
    println!("\n=== Chapter 5-9: Training & Post-Training Benchmarks ===");
    let device = if candle_core::utils::metal_is_available() {
        println!("Using Metal GPU");
        Device::new_metal(0)?
    } else {
        println!("Using CPU");
        Device::Cpu
    };
    
    let batch_size = 4;
    let seq_len = 64;
    let d_model = 512;
    let vocab_size = 1000;
    let warmup_iters = 5;
    let measure_iters = 20;
    
    println!("Config: batch={}, seq_len={}, d_model={}, warmup={}, measure={} iterations", 
        batch_size, seq_len, d_model, warmup_iters, measure_iters);

    // 1. GRPO Benchmark (Chapter 9)
    println!("\n--- GRPO (Group Relative Policy Optimization) ---");
    {
        let grpo = GRPOTrainer::new(0.01);
        
        let logits = Tensor::randn(0f32, 1f32, (batch_size, seq_len, vocab_size), &device)?;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
        let rewards = Tensor::randn(0f32, 1f32, (batch_size,), &device)?;
        let ref_logits = Tensor::randn(0f32, 1f32, (batch_size, seq_len, vocab_size), &device)?;
        
        // Warmup
        for _ in 0..warmup_iters {
            let _ = grpo.compute_loss(&logits, &input_ids, &rewards, &ref_logits)?;
        }
        
        // Measure
        let start = std::time::Instant::now();
        for _ in 0..measure_iters {
            let _ = grpo.compute_loss(&logits, &input_ids, &rewards, &ref_logits)?;
        }
        let duration = start.elapsed();
        println!("GRPO Loss Average Time: {:.2?} per forward pass", duration / measure_iters as u32);
    }

    // 2. R1 Reasoning Model Benchmark (Chapter 9)
    println!("\n--- R1 Reasoning Model ---");
    {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let r1 = ReasoningModel::new(vocab_size, d_model, vb.pp("r1"))?;
        
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
        
        // Warmup
        for _ in 0..warmup_iters {
            let _ = r1.forward(&input_ids)?;
        }
        
        // Measure
        let start = std::time::Instant::now();
        for _ in 0..measure_iters {
            let _ = r1.forward(&input_ids)?;
        }
        let duration = start.elapsed();
        println!("R1 Forward Average Time: {:.2?} per forward pass", duration / measure_iters as u32);
    }

    // 3. Knowledge Distillation Benchmarks (Chapter 13)
    println!("\n--- Knowledge Distillation Losses ---");
    {
        let student_logits = Tensor::randn(0f32, 1f32, (batch_size, seq_len, vocab_size), &device)?;
        let teacher_logits = Tensor::randn(0f32, 1f32, (batch_size, seq_len, vocab_size), &device)?;
        let temperature = 4.0;
        
        // KL Divergence Loss
        for _ in 0..warmup_iters {
            let _ = kd_loss_kl(&student_logits, &teacher_logits, temperature)?;
        }
        let start = std::time::Instant::now();
        for _ in 0..measure_iters {
            let _ = kd_loss_kl(&student_logits, &teacher_logits, temperature)?;
        }
        let duration = start.elapsed();
        println!("KD KL Loss Average Time: {:.2?} per forward pass", duration / measure_iters as u32);
        
        // MSE Loss
        for _ in 0..warmup_iters {
            let _ = kd_loss_mse(&student_logits, &teacher_logits, temperature)?;
        }
        let start = std::time::Instant::now();
        for _ in 0..measure_iters {
            let _ = kd_loss_mse(&student_logits, &teacher_logits, temperature)?;
        }
        let duration = start.elapsed();
        println!("KD MSE Loss Average Time: {:.2?} per forward pass", duration / measure_iters as u32);
        
        // JSD Loss
        for _ in 0..warmup_iters {
            let _ = kd_loss_jsd(&student_logits, &teacher_logits, temperature)?;
        }
        let start = std::time::Instant::now();
        for _ in 0..measure_iters {
            let _ = kd_loss_jsd(&student_logits, &teacher_logits, temperature)?;
        }
        let duration = start.elapsed();
        println!("KD JSD Loss Average Time: {:.2?} per forward pass", duration / measure_iters as u32);
    }

    // 4. Reward Model (Reward Head) Benchmark (Chapter 12)
    println!("\n--- Reward Model (Head Only) ---");
    {
        // Test just the reward head for fair comparison
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        // Simple reward head: Linear(d_model -> d_model/4) -> GELU -> Linear(d_model/4 -> 1)
        let linear1 = candle_nn::linear(d_model, d_model / 4, vb.pp("l1"))?;
        let linear2 = candle_nn::linear(d_model / 4, 1, vb.pp("l2"))?;
        
        let hidden_states = Tensor::randn(0f32, 1f32, (batch_size, d_model), &device)?;
        
        let reward_head_forward = || -> Result<Tensor> {
            let x = linear1.forward(&hidden_states)?;
            let x = x.gelu_erf()?;
            linear2.forward(&x)
        };
        
        // Warmup
        for _ in 0..warmup_iters {
            let _ = reward_head_forward()?;
        }
        
        // Measure
        let start = std::time::Instant::now();
        for _ in 0..measure_iters {
            let _ = reward_head_forward()?;
        }
        let duration = start.elapsed();
        println!("Reward Forward Average Time: {:.2?} per forward pass", duration / measure_iters as u32);
    }

    // 5. DPO Loss Benchmark (Chapter 12)
    println!("\n--- DPO (Direct Preference Optimization) ---");
    {
        let beta = 0.1f64;
        
        let policy_chosen_logps = Tensor::randn(0f32, 1f32, (batch_size,), &device)?;
        let policy_rejected_logps = Tensor::randn(0f32, 1f32, (batch_size,), &device)?;
        let ref_chosen_logps = Tensor::randn(0f32, 1f32, (batch_size,), &device)?;
        let ref_rejected_logps = Tensor::randn(0f32, 1f32, (batch_size,), &device)?;
        
        // DPO loss computation
        let compute_dpo_loss = || -> Result<Tensor> {
            let chosen_logratios = (&policy_chosen_logps - &ref_chosen_logps)?;
            let rejected_logratios = (&policy_rejected_logps - &ref_rejected_logps)?;
            let logits = ((chosen_logratios - rejected_logratios)? * beta)?;
            
            // Sigmoid DPO loss: -log(sigmoid(logits))
            let sigmoid = (logits.neg()?.exp()? + 1.0)?.recip()?;
            let losses = sigmoid.log()?.neg()?;
            losses.mean_all()
        };
        
        // Warmup
        for _ in 0..warmup_iters {
            let _ = compute_dpo_loss()?;
        }
        
        // Measure
        let start = std::time::Instant::now();
        for _ in 0..measure_iters {
            let _ = compute_dpo_loss()?;
        }
        let duration = start.elapsed();
        println!("DPO Loss Average Time: {:.2?} per forward pass", duration / measure_iters as u32);
    }

    Ok(())
}
