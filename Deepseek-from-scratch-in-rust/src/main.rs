use deepseek_from_scratch_in_rust::model::{r1, reward_model};
use deepseek_from_scratch_in_rust::training::{grpo, pipeline, sft, distillation};
use deepseek_from_scratch_in_rust::benchmarks::{attention_benchmark, moe_benchmark, mtp_fp8_benchmark, training_benchmark};

use candle_core::{Device, Tensor, Result, DType};
use candle_nn::{VarBuilder, VarMap};
use deepseek_from_scratch_in_rust::model::attention::{MultiQueryAttention, GroupedQueryAttention};
use deepseek_from_scratch_in_rust::model::mla::{MultiHeadLatentAttention, DeepSeekAttention};
use deepseek_from_scratch_in_rust::model::moe::DeepSeekMoE;
use deepseek_from_scratch_in_rust::model::mtp::MTPModel;
use deepseek_from_scratch_in_rust::utils::logging;
use tracing::info;

fn main() -> Result<()> {
    logging::init_logging();
    info!("Starting DeepSeek from Scratch (Rust)");

    // CUDA-first device selection for production 3-GPU pipeline
    let device = if candle_core::utils::cuda_is_available() {
        info!("Using CUDA GPU");
        Device::new_cuda(0)?  // Default to GPU 0, PP stages use CUDA_VISIBLE_DEVICES
    // TODO: Remove after production - Metal fallback for local dev
    } else if candle_core::utils::metal_is_available() {
        info!("Using Metal GPU (fallback)");
        Device::new_metal(0)?
    } else {
        // TODO: Remove after production - CPU fallback
        println!("Using CPU (fallback)");
        Device::Cpu
    };
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    println!("--- Chapter 1: Multi-Query Attention (MQA) ---");
    let d_model = 512;
    let num_heads = 8;
    let batch_size = 4;
    let seq_len = 64;

    let mqa = MultiQueryAttention::new(d_model, num_heads, vb.pp("mqa"))?;
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;
    let output = mqa.forward(&input)?;

    println!("MQA Input shape: {:?}", input.shape());
    println!("MQA Output shape: {:?}", output.shape());
    println!("MQA Layer successful!");

    println!("\n--- Chapter 1: Grouped-Query Attention (GQA) ---");
    let num_heads = 32;
    let num_groups = 4;
    
    let gqa = GroupedQueryAttention::new(d_model, num_heads, num_groups, vb.pp("gqa"))?;
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;
    let output = gqa.forward(&input)?;

    println!("GQA Input shape: {:?}", input.shape());
    println!("GQA Output shape: {:?}", output.shape());
    println!("GQA Layer successful!");

    println!("\n--- Chapter 2: Multi-Head Latent Attention (MLA) ---");
    let d_model = 512;
    let num_heads = 8;
    let d_latent = 128;
    
    let mla = MultiHeadLatentAttention::new(d_model, num_heads, d_latent, vb.pp("mla"))?;
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;
    let output = mla.forward(&input, None)?;

    println!("MLA Input shape: {:?}", input.shape());
    println!("MLA Output shape: {:?}", output.shape());
    println!("MLA Layer successful!");

    println!("\n--- Chapter 2: DeepSeek Attention (Fused MLA + RoPE) ---");
    let d_rope = 64;
    
    let deepseek_attn = DeepSeekAttention::new(d_model, num_heads, d_latent, d_rope, vb.pp("deepseek"))?;
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;
    let output = deepseek_attn.forward(&input, None)?;

    println!("DeepSeek Attention Input shape: {:?}", input.shape());
    println!("DeepSeek Attention Output shape: {:?}", output.shape());
    println!("DeepSeek Attention Layer successful!");

    attention_benchmark::run_benchmark()?;

    println!("\n--- Chapter 3: DeepSeek MoE ---");
    let n_routed = 16;
    let n_shared = 2;
    let top_k = 2;
    let routed_hidden = 512;
    let shared_hidden = 1024;

    let ds_moe = DeepSeekMoE::new(
        d_model,
        n_routed,
        n_shared,
        top_k,
        routed_hidden,
        shared_hidden,
        vb.pp("ds_moe_demo"),
    )?;
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;
    let output = ds_moe.forward(&input)?;
    
    println!("DeepSeek MoE Input shape: {:?}", input.shape());
    println!("DeepSeek MoE Output shape: {:?}", output.shape());
    println!("DeepSeek MoE Layer successful!");

    moe_benchmark::run_benchmark()?;

    println!("\n--- Chapter 4: Multi-Token Prediction (MTP) ---");
    let vocab_size = 1000;
    let k_predictions = 1;
    let mtp_model = MTPModel::new(vocab_size, d_model, 2, k_predictions, vb.pp("mtp_demo"))?;
    let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
    let (main_logits, mtp_logits) = mtp_model.forward(&input_ids)?;
    
    println!("MTP Main Logits shape: {:?}", main_logits.shape());
    println!("MTP Future Logits count: {}", mtp_logits.len());
    if !mtp_logits.is_empty() {
        println!("MTP Future Logits[0] shape: {:?}", mtp_logits[0].shape());
    }
    println!("MTP Layer successful!");

    mtp_fp8_benchmark::run_benchmark()?;

    println!("\n--- Chapter 5: DeepSeek-R1 (Reasoning) ---");
    let vocab_size = 1000;
    let r1_model = r1::ReasoningModel::new(vocab_size, d_model, vb.pp("r1_demo"))?;
    let prompt = "DeepSeek architecture";
    let output = r1_model.generate_with_reasoning(prompt, &device)?;
    
    println!("Input Prompt: \"{}\"", prompt);
    println!("Generated Output (Simulated):\n{}", output);
    println!("DeepSeek-R1 Reasoning Layer successful!");

    println!("\n--- Chapter 6: GRPO (Group Relative Policy Optimization) ---");
    let group_size = 4;
    let seq_len = 10;
    let vocab_size = 100;
    let beta = 0.01;
    
    let grpo = grpo::GRPOTrainer::new(beta);
    
    // Simulate data
    let logits = Tensor::randn(0f32, 1f32, (group_size, seq_len, vocab_size), &device)?;
    let ref_logits = Tensor::randn(0f32, 1f32, (group_size, seq_len, vocab_size), &device)?;
    let input_ids = Tensor::zeros((group_size, seq_len), DType::U32, &device)?;
    let rewards = Tensor::new(&[1.0f32, 0.5, -0.5, 0.0], &device)?; // 4 rewards for group size 4
    
    let loss = grpo.compute_loss(&logits, &input_ids, &rewards, &ref_logits)?;
    
    println!("GRPO Group Size: {}", group_size);
    println!("Rewards: {:?}", rewards.to_vec1::<f32>()?);
    println!("Computed GRPO Loss: {:.4}", loss.to_scalar::<f32>()?);
    println!("GRPO Step successful!");

    println!("\n--- Chapter 7: Training Pipeline ---");
    println!("{}", "=".repeat(60));
    println!("DeepSeek Training Pipeline Demo");
    println!("{}\n", "=".repeat(60));
    
    // Scaling Laws
    println!("1. Scaling Laws");
    println!("{}", "-".repeat(40));
    let scaling = pipeline::ScalingLaws::deepseek();
    
    println!("Predicted loss (7B, 2T tokens): {:.4}", 
        scaling.predict_loss(7e9, 2e12));
    println!("Recommended LR for 7B: {:.2e}",
        scaling.recommended_lr(7e9));
    
    let (opt_n, opt_d) = scaling.optimal_config(1e23);
    println!("Optimal config for 1e23 FLOPs:");
    println!("  Parameters: {:.1}B", opt_n as f64 / 1e9);
    println!("  Tokens: {:.1}T", opt_d as f64 / 1e12);
    
    // Data Mixing
    println!("\n2. Data Mixing");
    println!("{}", "-".repeat(40));
    let mixing = pipeline::DataMixingConfig::deepseek_default();
    
    println!("Sampling probabilities:");
    let mut probs = mixing.get_sampling_probs();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (domain, prob) in probs {
        println!("  {}: {:.1}%", domain, prob * 100.0);
    }
    
    // WSD Scheduler
    println!("\n3. WSD Learning Rate Schedule");
    println!("{}", "-".repeat(40));
    let config = pipeline::PipelineConfig::default();
    let scheduler = config.create_wsd_scheduler();
    
    println!("LR at different steps:");
    for step in [0, 1000, 2000, 50000, 82000, 100000] {
        println!("  Step {:6}: {:.2e}", step, scheduler.get_lr(step));
    }
    
    // Curriculum Learning
    println!("\n4. Curriculum Learning");
    println!("{}", "-".repeat(40));
    let curriculum = pipeline::CurriculumScheduler::new(512, 4096, 10000, 5000);
    
    println!("Sequence length progression:");
    for step in [0, 2500, 5000, 7500, 10000, 15000] {
        println!("  Step {:5}: {}", step, curriculum.get_seq_length(step));
    }
    
    println!("Difficulty weight progression:");
    for step in [0, 1000, 2500, 5000, 10000] {
        println!("  Step {:5}: {:.2}", step, curriculum.get_difficulty_weight(step));
    }
    
    // Pipeline Config
    println!("\n5. Pipeline Configuration");
    println!("{}", "-".repeat(40));
    println!("Model size: {:.1}B", config.model_size as f64 / 1e9);
    println!("Max steps: {}", config.max_steps);
    println!("Learning rate: {:.0e}", config.learning_rate);
    println!("Effective batch size: {}", config.effective_batch_size());
    println!("Tokens per step: {}", config.tokens_per_step());
    
    // Distributed Config
    println!("\n6. Distributed Configuration");
    println!("{}", "-".repeat(40));
    let dist = pipeline::DistributedConfig::multi_gpu(8, 2);
    println!("World size: {}", dist.world_size);
    println!("Data parallelism: {}", dist.dp_size);
    println!("ZeRO stage: {}", dist.zero_stage);
    
    println!("\n{}", "=".repeat(60));
    println!("Pipeline Demo Complete!");
    println!("{}\n", "=".repeat(60));

    println!("\n--- Chapter 8: SFT and DPO ---");
    println!("{}", "=".repeat(60));
    println!("SFT and DPO Demo (Rust)");
    println!("{}\n", "=".repeat(60));
    
    let device = Device::Cpu;
    
    // Chat Template
    println!("1. Chat Template");
    println!("{}", "-".repeat(40));
    
    let template = sft::ChatTemplate::default();
    let messages = vec![
        ("system".to_string(), "You are a helpful assistant.".to_string()),
        ("user".to_string(), "What is 2+2?".to_string()),
        ("assistant".to_string(), "2+2 equals 4.".to_string()),
    ];
    
    let formatted = template.format_conversation(&messages);
    println!("Formatted conversation:");
    println!("{}", formatted);
    
    // SFT Loss
    println!("2. SFT Loss Computation");
    println!("{}", "-".repeat(40));
    
    let batch_size = 2;
    let seq_len = 10;
    let vocab_size = 1000;
    
    let logits = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, vocab_size), &device)?;
    let labels = Tensor::from_vec(
        vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10,
             -100, -100, 3, 4, 5, 6, 7, 8, 9, 10],
        (batch_size, seq_len),
        &device,
    )?;
    
    let sft_loss = sft::compute_sft_loss(&logits, &labels)?;
    println!("SFT Loss: {:.4}", sft_loss.to_scalar::<f32>()?);
    
    // DPO Loss
    println!("\n3. DPO Loss Computation");
    println!("{}", "-".repeat(40));
    
    let config = sft::DPOConfig::default();
    println!("Beta: {}", config.beta);
    println!("Loss type: {:?}", config.loss_type);
    
    // Simulated log probs
    let policy_chosen = Tensor::from_vec(vec![-10.0f32, -8.0, -12.0], 3, &device)?;
    let policy_rejected = Tensor::from_vec(vec![-15.0f32, -14.0, -11.0], 3, &device)?;
    let ref_chosen = Tensor::from_vec(vec![-11.0f32, -9.0, -13.0], 3, &device)?;
    let ref_rejected = Tensor::from_vec(vec![-16.0f32, -15.0, -12.0], 3, &device)?;
    
    let (dpo_loss, metrics) = sft::compute_dpo_loss(
        &policy_chosen,
        &policy_rejected,
        &ref_chosen,
        &ref_rejected,
        &config,
    )?;
    
    println!("DPO Loss: {:.4}", dpo_loss.to_scalar::<f32>()?);
    println!("Chosen reward: {:.4}", metrics.chosen_reward);
    println!("Rejected reward: {:.4}", metrics.rejected_reward);
    println!("Accuracy: {:.2}%", metrics.accuracy * 100.0);
    println!("Margin: {:.4}", metrics.margin);
    
    // LoRA Parameter Count
    println!("\n4. LoRA Parameter Count");
    println!("{}", "-".repeat(40));
    
    let hidden_size = 4096;
    let num_layers = 32;
    let lora_r = 64;
    let num_modules = 7;  // q, k, v, o, gate, up, down
    
    let lora_params = sft::lora_param_count(hidden_size, num_layers, lora_r, num_modules);
    let full_params = hidden_size * hidden_size * num_modules * num_layers;
    
    println!("Full parameters: {}", full_params);
    println!("LoRA parameters: {}", lora_params);
    println!("Reduction: {:.2}%", 100.0 * (1.0 - lora_params as f64 / full_params as f64));
    
    println!("\n{}", "=".repeat(60));
    println!("SFT/DPO Demo Complete!");
    println!("{}\n", "=".repeat(60));

    println!("\n--- Chapter 8: Reward Model ---");
    println!("{}", "=".repeat(60));
    println!("Reward Model Demo (Rust)");
    println!("{}\n", "=".repeat(60));
    
    let device = Device::Cpu;
    
    // Configuration
    println!("1. Configuration");
    println!("{}", "-".repeat(40));
    
    let config = reward_model::RewardModelConfig::default();
    println!("Hidden size: {}", config.hidden_size);
    println!("Intermediate size: {}", config.intermediate_size);
    
    let train_config = reward_model::RewardTrainingConfig::default();
    println!("Learning rate: {}", train_config.learning_rate);
    println!("Batch size: {}", train_config.batch_size);
    
    // Preference Loss
    println!("\n2. Preference Loss");
    println!("{}", "-".repeat(40));
    
    let chosen = Tensor::from_vec(vec![1.5f32, 2.0, 0.8, 1.2], 4, &device)?;
    let rejected = Tensor::from_vec(vec![0.5f32, 0.8, 1.0, 0.3], 4, &device)?;
    
    let (loss, metrics) = reward_model::compute_preference_loss(&chosen, &rejected, 0.0)?;
    
    println!("Chosen rewards: {:?}", chosen.to_vec1::<f32>()?);
    println!("Rejected rewards: {:?}", rejected.to_vec1::<f32>()?);
    println!("Preference loss: {:.4}", loss.to_scalar::<f32>()?);
    println!("Accuracy: {:.2}%", metrics.accuracy * 100.0);
    println!("Margin: {:.4}", metrics.margin);
    
    // Margin Loss
    println!("\n3. Margin Loss");
    println!("{}", "-".repeat(40));
    
    let margin_loss = reward_model::compute_margin_loss(&chosen, &rejected, 0.5)?;
    println!("Margin loss (margin=0.5): {:.4}", margin_loss.to_scalar::<f32>()?);
    
    // Reward Normalizer
    println!("\n4. Reward Normalization");
    println!("{}", "-".repeat(40));
    
    let mut normalizer = reward_model::RewardNormalizer::new();
    let rewards = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    normalizer.update(&rewards);
    
    println!("Running mean: {:.4}", normalizer.mean);
    println!("Running std: {:.4}", normalizer.std());
    
    let test_rewards = Tensor::from_vec(vec![3.0f32, 4.0], 2, &device)?;
    let normalized = normalizer.normalize(&test_rewards)?;
    println!("Normalized [3.0, 4.0]: {:?}", normalized.to_vec1::<f32>()?);
    
    // Reward Aggregation
    println!("\n5. Reward Aggregation");
    println!("{}", "-".repeat(40));
    
    let batch_size = 2;
    let seq_len = 5;
    let hidden_size = 3;
    
    let token_rewards = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
    let mask = Tensor::from_vec(
        vec![1i64, 1, 1, 0, 0, 1, 1, 1, 1, 1],
        (batch_size, seq_len),
        &device,
    )?;
    
    for agg in [
        reward_model::RewardAggregation::MeanPool,
        reward_model::RewardAggregation::MaxPool,
    ] {
        let result = agg.aggregate(&token_rewards, &mask)?;
        println!("{:?} shape: {:?}", agg, result.dims());
    }
    
    println!("\n{}", "=".repeat(60));
    println!("Reward Model Demo Complete!");
    println!("{}\n", "=".repeat(60));

    println!("\n--- Chapter 9: Knowledge Distillation ---");
    println!("{}", "=".repeat(60));
    println!("Knowledge Distillation Demo (Rust)");
    println!("{}\n", "=".repeat(60));
    
    let device = Device::Cpu;
    
    // Configuration
    println!("1. Distillation Configuration");
    println!("{}", "-".repeat(40));
    
    let config = distillation::DistillationConfig::default();
    println!("Temperature: {}", config.temperature);
    println!("Alpha: {}", config.alpha);
    println!("Loss type: {:?}", config.kd_loss_type);
    
    // KD Loss Types
    println!("\n2. Knowledge Distillation Losses");
    println!("{}", "-".repeat(40));
    
    let batch_size = 2;
    let seq_len = 10;
    let vocab_size = 1000;
    
    let student_logits = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, vocab_size), &device)?;
    let teacher_logits = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, vocab_size), &device)?;
    
    for loss_type in [
        distillation::KDLossType::KLDivergence,
        distillation::KDLossType::MSE,
        distillation::KDLossType::JSD,
        distillation::KDLossType::Cosine
    ] {
        let cfg = distillation::DistillationConfig {
            kd_loss_type: loss_type.clone(),
            ..Default::default()
        };
        let loss = distillation::compute_distillation_loss(&student_logits, &teacher_logits, &cfg)?;
        println!("{:?} loss: {:.4}", loss_type, loss.to_scalar::<f32>()?);
    }
    
    // Combined Loss
    println!("\n3. Combined Distillation Loss");
    println!("{}", "-".repeat(40));
    
    let labels = Tensor::from_vec(
        (0..batch_size * seq_len).map(|x| (x % vocab_size) as i64).collect::<Vec<_>>(),
        (batch_size, seq_len),
        &device,
    )?;
    
    let metrics = distillation::combined_distillation_loss(
        &student_logits,
        &teacher_logits,
        &labels,
        &config,
    )?;
    
    println!("Total loss: {:.4}", metrics.total_loss);
    println!("KD loss: {:.4}", metrics.kd_loss);
    println!("CE loss: {:.4}", metrics.ce_loss);
    
    // Hidden State Distillation
    println!("\n4. Hidden State Distillation");
    println!("{}", "-".repeat(40));
    
    let hidden_size = 512;
    let teacher_hidden_size = 768;
    
    let student_hidden = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
    let teacher_hidden = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, teacher_hidden_size), &device)?;
    
    // Projection matrix
    let projection = Tensor::randn(0.0f32, 0.02, (hidden_size, teacher_hidden_size), &device)?;
    
    let hidden_loss = distillation::hidden_state_distillation(
        &student_hidden,
        &teacher_hidden,
        Some(&projection),
    )?;
    println!("Hidden state loss: {:.4}", hidden_loss.to_scalar::<f32>()?);
    
    // Progressive Distillation
    println!("\n5. Progressive Distillation");
    println!("{}", "-".repeat(40));
    
    let prog_config = distillation::ProgressiveConfig::default();
    let total_steps = 1000;
    let mut distiller = distillation::ProgressiveDistiller::new(prog_config, total_steps);
    
    println!("Stages: {}", distiller.config.num_stages); // Accessing config field directly might need pub
    
    for step in [0, 333, 666, 999] {
        while distiller.current_step < step {
            distiller.step();
        }
        println!("Step {}: stage={}, temp={:.2}, size={}", 
            step, 
            distiller.current_stage(),
            distiller.current_temperature(),
            distiller.current_intermediate_size()
        );
    }
    
    // Layer Mapping
    println!("\n6. Layer Mapping");
    println!("{}", "-".repeat(40));
    
    let student_layers = 12;
    let teacher_layers = 60;
    
    let uniform_map = distillation::compute_layer_mapping(student_layers, teacher_layers);
    let top_heavy_map = distillation::compute_layer_mapping_top_heavy(student_layers, teacher_layers);
    
    println!("Uniform mapping (student → teacher):");
    for (s, t) in &uniform_map[..4] {
        println!("  Layer {} → Layer {}", s, t);
    }
    println!("  ...");
    
    println!("\nTop-heavy mapping (student → teacher):");
    for (s, t) in &top_heavy_map[..4] {
        println!("  Layer {} → Layer {}", s, t);
    }
    println!("  ...");
    
    // Temperature Schedule
    println!("\n7. Temperature Schedule");
    println!("{}", "-".repeat(40));
    
    let schedules = vec![
        ("Constant", distillation::TemperatureSchedule::Constant(4.0)),
        ("Linear", distillation::TemperatureSchedule::Linear { start: 6.0, end: 2.0 }),
        ("Cosine", distillation::TemperatureSchedule::Cosine { start: 6.0, end: 2.0 }),
    ];
    
    for (name, schedule) in schedules {
        println!("{}:", name);
        for step in [0, 250, 500, 750, 1000] {
            let temp = schedule.get_temperature(step, 1000);
            println!("  Step {}: T={:.2}", step, temp);
        }
    }
    
    println!("\n{}", "=".repeat(60));
    println!("Distillation Demo Complete!");
    println!("{}\n", "=".repeat(60));

    // Run training benchmarks for Chapter 5-9
    training_benchmark::run_benchmark()?;

    Ok(())
}
