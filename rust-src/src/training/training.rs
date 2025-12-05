//! Training Infrastructure for DeepSeek From Scratch (Rust/Candle)
//!
//! This module provides:
//! - Metal GPU / CPU device support
//! - Cosine annealing LR scheduler with warmup
//! - Checkpoint saving and loading utilities
//! - Cross-entropy loss for language modeling
//! - DSA alignment loss for sparse attention training
//!
//! Note: Candle provides SGD/AdamW through candle-optimizers or manual implementation.
//! This module focuses on the training loop infrastructure.

use candle_core::{DType, Device, Result, Tensor, D, Module};
use candle_nn::{VarBuilder, VarMap, ops};
use std::fs;

/// Get the best available device (Metal GPU on macOS, otherwise CPU)
pub fn get_device() -> Result<Device> {
    if candle_core::utils::metal_is_available() {
        println!("Using Metal GPU");
        Device::new_metal(0)
    } else {
        println!("Using CPU");
        Ok(Device::Cpu)
    }
}

/// Training configuration
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    // Optimizer
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub max_grad_norm: f64,
    
    // Scheduler
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub min_lr_ratio: f64,
    
    // Batch
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    
    // Checkpointing
    pub save_every: usize,
    pub checkpoint_dir: String,
    
    // Logging
    pub log_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            max_grad_norm: 1.0,
            warmup_steps: 100,
            max_steps: 10000,
            min_lr_ratio: 0.1,
            batch_size: 8,
            gradient_accumulation_steps: 4,
            save_every: 1000,
            checkpoint_dir: "./checkpoints".to_string(),
            log_every: 10,
        }
    }
}

/// Learning rate scheduler with warmup and cosine decay
pub struct LRScheduler {
    base_lr: f64,
    warmup_steps: usize,
    max_steps: usize,
    min_lr_ratio: f64,
}

impl LRScheduler {
    pub fn new(config: &TrainingConfig) -> Self {
        Self {
            base_lr: config.learning_rate,
            warmup_steps: config.warmup_steps,
            max_steps: config.max_steps,
            min_lr_ratio: config.min_lr_ratio,
        }
    }
    
    /// Compute learning rate with cosine schedule and warmup
    pub fn get_lr(&self, step: usize) -> f64 {
        let min_lr = self.base_lr * self.min_lr_ratio;
        
        if step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (step as f64) / (self.warmup_steps.max(1) as f64)
        } else if step >= self.max_steps {
            // After max steps, use min LR
            min_lr
        } else {
            // Cosine decay
            let progress = (step - self.warmup_steps) as f64 / ((self.max_steps - self.warmup_steps).max(1) as f64);
            let cosine_decay = 0.5 * (1.0 + (progress * std::f64::consts::PI).cos());
            min_lr + (self.base_lr - min_lr) * cosine_decay
        }
    }
}

/// Cross-entropy loss for language modeling
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len, vocab_size) = logits.dims3()?;
    
    // Shift for next-token prediction (causal LM)
    let shift_logits = logits.narrow(1, 0, seq_len - 1)?;
    let shift_targets = targets.narrow(1, 1, seq_len - 1)?;
    
    // Flatten
    let flat_logits = shift_logits.reshape((batch_size * (seq_len - 1), vocab_size))?;
    let flat_targets = shift_targets.reshape((batch_size * (seq_len - 1),))?;
    
    // Log softmax
    let log_probs = ops::log_softmax(&flat_logits, D::Minus1)?;
    
    // Gather target log probs (negative log likelihood)
    let flat_targets_u32 = flat_targets.to_dtype(DType::U32)?;
    let target_log_probs = log_probs.gather(&flat_targets_u32.unsqueeze(1)?, 1)?.squeeze(1)?;
    
    // Negative mean
    let loss = target_log_probs.neg()?.mean_all()?;
    
    Ok(loss)
}

/// Trainer struct - holds training state
pub struct Trainer {
    pub config: TrainingConfig,
    pub device: Device,
    pub varmap: VarMap,
    pub scheduler: LRScheduler,
    pub global_step: usize,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Result<Self> {
        let device = get_device()?;
        let varmap = VarMap::new();
        let scheduler = LRScheduler::new(&config);
        
        // Create checkpoint directory
        fs::create_dir_all(&config.checkpoint_dir).ok();
        
        println!("Trainer initialized on device: {:?}", device);
        println!("Gradient accumulation steps: {}", config.gradient_accumulation_steps);
        println!("Effective batch size: {}", config.batch_size * config.gradient_accumulation_steps);
        
        Ok(Self {
            config,
            device,
            varmap,
            scheduler,
            global_step: 0,
        })
    }
    
    pub fn vb(&self) -> VarBuilder {
        VarBuilder::from_varmap(&self.varmap, DType::F32, &self.device)
    }
    
    /// Get current learning rate
    pub fn current_lr(&self) -> f64 {
        self.scheduler.get_lr(self.global_step)
    }
    
    /// Save checkpoint
    pub fn save_checkpoint(&self, suffix: &str) -> Result<()> {
        let path = format!("{}/checkpoint_{}.safetensors", self.config.checkpoint_dir, suffix);
        self.varmap.save(&path)?;
        println!("Checkpoint saved: {}", path);
        Ok(())
    }
    
    /// Load checkpoint
    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        self.varmap.load(path)?;
        println!("Checkpoint loaded: {}", path);
        Ok(())
    }
    
    /// Count total parameters
    pub fn num_parameters(&self) -> usize {
        self.varmap.data().lock().unwrap().iter()
            .map(|(_, v)| v.as_tensor().elem_count())
            .sum()
    }
}

// ============================================================================
// DSA (DeepSeek Sparse Attention) Alignment Loss
// ============================================================================

/// Configuration for DSA alignment loss during continued pre-training
#[derive(Clone, Debug)]
pub struct DSAAlignmentConfig {
    /// Weight for the alignment loss in total loss
    pub alignment_weight: f64,
    /// Fraction of positions to sample for alignment (0.0-1.0)
    pub sample_fraction: f64,
    /// Type of alignment loss to use
    pub loss_type: DSAAlignmentLossType,
    /// Temperature for KL divergence (if applicable)
    pub temperature: f64,
    /// Whether to use gradient stopping on dense attention
    pub stop_gradient_dense: bool,
}

impl Default for DSAAlignmentConfig {
    fn default() -> Self {
        Self {
            alignment_weight: 0.1,
            sample_fraction: 0.1, // Sample 10% of positions for alignment
            loss_type: DSAAlignmentLossType::MSE,
            temperature: 1.0,
            stop_gradient_dense: true, // Dense attention is the target
        }
    }
}

/// Type of alignment loss for DSA training
#[derive(Clone, Debug, PartialEq)]
pub enum DSAAlignmentLossType {
    /// Mean Squared Error between sparse and dense attention outputs
    MSE,
    /// KL Divergence between attention weight distributions
    KLDivergence,
    /// Cosine similarity loss
    CosineSimilarity,
    /// Combined MSE on outputs + KL on attention weights
    Combined { output_weight: f64, attn_weight: f64 },
}

/// DSA Alignment Loss for training sparse attention to match dense attention
/// 
/// This loss aligns the outputs of sparse attention to dense attention,
/// allowing the sparse model to learn the same attention patterns while
/// being more efficient. Used during continued pre-training phase.
pub struct DSAAlignmentLoss {
    config: DSAAlignmentConfig,
}

impl DSAAlignmentLoss {
    pub fn new(config: DSAAlignmentConfig) -> Self {
        Self { config }
    }

    /// Compute alignment loss between sparse and dense attention outputs
    /// 
    /// # Arguments
    /// * `sparse_output` - Output from sparse attention (batch, seq_len, d_model)
    /// * `dense_output` - Output from dense attention (batch, seq_len, d_model)
    /// * `sample_positions` - Optional positions to sample (if None, samples randomly)
    /// 
    /// # Returns
    /// Alignment loss scalar
    pub fn compute_output_loss(
        &self,
        sparse_output: &Tensor,
        dense_output: &Tensor,
        sample_positions: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_batch_size, seq_len, _d_model) = sparse_output.dims3()?;
        
        // Optionally stop gradient on dense (it's the target)
        // Note: In Candle, we use clone() and rely on the computation graph
        // For true gradient stopping, we'd need to handle this in backward pass
        let dense_target = dense_output.clone();

        // Sample positions if needed
        let (sparse_sampled, dense_sampled) = if let Some(positions) = sample_positions {
            // Use provided positions
            let sparse_sampled = self.gather_positions(sparse_output, positions)?;
            let dense_sampled = self.gather_positions(&dense_target, positions)?;
            (sparse_sampled, dense_sampled)
        } else if self.config.sample_fraction < 1.0 {
            // Random sampling
            let num_samples = ((seq_len as f64) * self.config.sample_fraction) as usize;
            let num_samples = num_samples.max(1);
            
            // Generate random positions (simplified: use evenly spaced)
            let step = seq_len / num_samples;
            let positions: Vec<i64> = (0..num_samples).map(|i| (i * step) as i64).collect();
            let positions = Tensor::from_vec(positions, (num_samples,), sparse_output.device())?;
            
            let sparse_sampled = self.gather_positions(sparse_output, &positions)?;
            let dense_sampled = self.gather_positions(&dense_target, &positions)?;
            (sparse_sampled, dense_sampled)
        } else {
            // Use all positions
            (sparse_output.clone(), dense_target)
        };

        // Compute loss based on type
        match &self.config.loss_type {
            DSAAlignmentLossType::MSE => {
                self.mse_loss(&sparse_sampled, &dense_sampled)
            }
            DSAAlignmentLossType::CosineSimilarity => {
                self.cosine_loss(&sparse_sampled, &dense_sampled)
            }
            DSAAlignmentLossType::KLDivergence => {
                // For output alignment, treat as soft targets
                self.kl_output_loss(&sparse_sampled, &dense_sampled)
            }
            DSAAlignmentLossType::Combined { output_weight, attn_weight: _ } => {
                // Combined loss uses MSE for outputs
                let mse = self.mse_loss(&sparse_sampled, &dense_sampled)?;
                mse * *output_weight
            }
        }
    }

    /// Compute alignment loss between sparse and dense attention weight distributions
    /// 
    /// # Arguments
    /// * `sparse_attn_weights` - Attention weights from sparse attention (batch, heads, q_len, k_len)
    /// * `dense_attn_weights` - Attention weights from dense attention (batch, heads, q_len, k_len)
    /// * `sparse_mask` - Mask indicating which positions sparse attention attends to
    /// 
    /// # Returns
    /// Attention distribution alignment loss
    pub fn compute_attention_loss(
        &self,
        sparse_attn_weights: &Tensor,
        dense_attn_weights: &Tensor,
        sparse_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Note: In Candle, we use clone() - gradient stopping handled in backward pass
        let dense_target = dense_attn_weights.clone();

        // If sparse mask is provided, only compare on sparse positions
        let (sparse_w, dense_w) = if let Some(mask) = sparse_mask {
            // Mask out non-attended positions in dense
            let mask_broadcast = mask.broadcast_as(dense_target.shape())?;
            let dense_masked = mask_broadcast.where_cond(&dense_target, &Tensor::zeros_like(&dense_target)?)?;
            
            // Renormalize dense attention on sparse positions
            let dense_sum = dense_masked.sum_keepdim(D::Minus1)?;
            let dense_normalized = (dense_masked / (dense_sum + 1e-10)?)?;
            
            (sparse_attn_weights.clone(), dense_normalized)
        } else {
            (sparse_attn_weights.clone(), dense_target)
        };

        match &self.config.loss_type {
            DSAAlignmentLossType::KLDivergence | 
            DSAAlignmentLossType::Combined { .. } => {
                self.kl_divergence_loss(&sparse_w, &dense_w)
            }
            DSAAlignmentLossType::MSE => {
                self.mse_loss(&sparse_w, &dense_w)
            }
            DSAAlignmentLossType::CosineSimilarity => {
                self.cosine_loss(&sparse_w, &dense_w)
            }
        }
    }

    /// Combined alignment loss for outputs and attention weights
    pub fn compute_combined_loss(
        &self,
        sparse_output: &Tensor,
        dense_output: &Tensor,
        sparse_attn_weights: Option<&Tensor>,
        dense_attn_weights: Option<&Tensor>,
        sparse_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let output_loss = self.compute_output_loss(sparse_output, dense_output, None)?;

        if let (
            Some(sparse_attn), 
            Some(dense_attn), 
            DSAAlignmentLossType::Combined { output_weight, attn_weight }
        ) = (sparse_attn_weights, dense_attn_weights, &self.config.loss_type) {
            let attn_loss = self.compute_attention_loss(sparse_attn, dense_attn, sparse_mask)?;
            (output_loss * *output_weight)? + (attn_loss * *attn_weight)?
        } else {
            Ok(output_loss)
        }
    }

    /// Get the alignment weight for combining with main loss
    pub fn get_weight(&self) -> f64 {
        self.config.alignment_weight
    }

    // ---- Private helper functions ----

    fn mse_loss(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        let diff = (pred - target)?;
        let sq = diff.sqr()?;
        sq.mean_all()
    }

    fn cosine_loss(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        // Cosine similarity loss: 1 - cos_sim
        let pred_norm = self.l2_normalize(pred)?;
        let target_norm = self.l2_normalize(target)?;
        
        let cos_sim = (pred_norm * target_norm)?.sum(D::Minus1)?.mean_all()?;
        
        // Loss is 1 - similarity (so lower is better)
        let one = Tensor::new(1.0f32, pred.device())?;
        one - cos_sim
    }

    fn kl_divergence_loss(&self, p: &Tensor, q: &Tensor) -> Result<Tensor> {
        // KL(P || Q) = sum(P * log(P/Q))
        // Add small epsilon to avoid log(0)
        let eps = 1e-10;
        let p_safe = (p + eps)?;
        let q_safe = (q + eps)?;
        
        let log_ratio = (p_safe.log()? - q_safe.log()?)?;
        let kl = (p * log_ratio)?.sum(D::Minus1)?.mean_all()?;
        
        Ok(kl)
    }

    fn kl_output_loss(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        // For output alignment, use squared difference as KL-like loss
        // (since outputs aren't probability distributions)
        let temp = self.config.temperature;
        
        // Scale by temperature
        let pred_scaled = (pred / temp)?;
        let target_scaled = (target / temp)?;
        
        // Softmax to get "soft" distributions
        let pred_soft = ops::softmax(&pred_scaled, D::Minus1)?;
        let target_soft = ops::softmax(&target_scaled, D::Minus1)?;
        
        self.kl_divergence_loss(&pred_soft, &target_soft)
    }

    fn l2_normalize(&self, x: &Tensor) -> Result<Tensor> {
        let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let norm_broadcast = norm.broadcast_as(x.shape())?;
        x / (norm_broadcast + 1e-10)?
    }

    fn gather_positions(&self, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
        // Gather along sequence dimension
        let (batch_size, _seq_len, d_model) = x.dims3()?;
        let num_positions = positions.dim(0)?;
        
        // Make tensor contiguous before gathering
        let x_contig = x.contiguous()?;
        
        // Expand positions for gathering
        let positions_i64 = positions.to_dtype(DType::I64)?;
        let positions_expanded = positions_i64
            .unsqueeze(0)?
            .unsqueeze(2)?
            .broadcast_as((batch_size, num_positions, d_model))?
            .contiguous()?;
        
        x_contig.gather(&positions_expanded, 1)
    }
}

/// Training configuration for DSA continued pre-training
#[derive(Clone, Debug)]
pub struct DSATrainingConfig {
    /// Base training config
    pub base: TrainingConfig,
    /// DSA alignment config
    pub alignment: DSAAlignmentConfig,
    /// How often to compute full dense attention for alignment (expensive)
    pub dense_alignment_every: usize,
    /// Warmup steps before introducing alignment loss
    pub alignment_warmup_steps: usize,
    /// Linear ramp for alignment weight during warmup
    pub alignment_ramp: bool,
}

impl Default for DSATrainingConfig {
    fn default() -> Self {
        Self {
            base: TrainingConfig::default(),
            alignment: DSAAlignmentConfig::default(),
            dense_alignment_every: 100, // Compute dense attention every 100 steps
            alignment_warmup_steps: 1000,
            alignment_ramp: true,
        }
    }
}

impl DSATrainingConfig {
    /// Get the effective alignment weight at a given step
    pub fn get_alignment_weight(&self, step: usize) -> f64 {
        if step < self.alignment_warmup_steps {
            if self.alignment_ramp {
                // Linear ramp
                self.alignment.alignment_weight * (step as f64 / self.alignment_warmup_steps as f64)
            } else {
                0.0
            }
        } else {
            self.alignment.alignment_weight
        }
    }
}


/// Demo function showing training infrastructure
pub fn demo_training() -> Result<()> {
    println!("\n=== Training Infrastructure Demo ===\n");
    
    let config = TrainingConfig {
        learning_rate: 1e-4,
        warmup_steps: 5,
        max_steps: 20,
        batch_size: 4,
        gradient_accumulation_steps: 2,
        log_every: 5,
        save_every: 10,
        checkpoint_dir: "./demo_checkpoints".to_string(),
        ..Default::default()
    };
    
    let trainer = Trainer::new(config)?;
    
    // Create a simple model for demo
    let vb = trainer.vb();
    let embed = candle_nn::embedding(1000, 256, vb.pp("embed"))?;
    let linear = candle_nn::linear(256, 1000, vb.pp("head"))?;
    
    println!("Model created with {} parameters", trainer.num_parameters());
    
    // Simulate a forward pass
    let input_ids = Tensor::zeros((4, 64), DType::U32, &trainer.device)?;
    let hidden = embed.forward(&input_ids)?;
    let logits = linear.forward(&hidden)?;
    
    println!("Forward pass complete. Logits shape: {:?}", logits.shape());
    
    // Compute loss
    let targets = Tensor::zeros((4, 64), DType::U32, &trainer.device)?;
    let loss = cross_entropy_loss(&logits, &targets)?;
    println!("Loss: {:.4}", loss.to_scalar::<f32>()?);
    
    // Demo LR schedule
    println!("\nLR Schedule:");
    for step in [0, 5, 10, 15, 20] {
        println!("  Step {}: LR = {:.2e}", step, trainer.scheduler.get_lr(step));
    }
    
    // Save checkpoint
    trainer.save_checkpoint("demo")?;
    
    println!("\n=== Demo Complete ===");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lr_schedule() {
        let config = TrainingConfig {
            learning_rate: 1e-4,
            warmup_steps: 100,
            max_steps: 1000,
            min_lr_ratio: 0.1,
            ..Default::default()
        };
        
        let scheduler = LRScheduler::new(&config);
        
        // Warmup
        assert!((scheduler.get_lr(0) - 0.0).abs() < 1e-10);
        assert!((scheduler.get_lr(50) - 0.5e-4).abs() < 1e-10);
        assert!((scheduler.get_lr(100) - 1e-4).abs() < 1e-10);
        
        // Decay
        let lr_mid = scheduler.get_lr(550);
        assert!(lr_mid > 1e-5 && lr_mid < 1e-4);
        
        // End
        let lr_end = scheduler.get_lr(1000);
        assert!((lr_end - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_dsa_alignment_loss_mse() -> Result<()> {
        let device = Device::Cpu;
        
        let config = DSAAlignmentConfig {
            alignment_weight: 0.1,
            sample_fraction: 1.0, // Use all positions
            loss_type: DSAAlignmentLossType::MSE,
            ..Default::default()
        };
        
        let loss_fn = DSAAlignmentLoss::new(config);
        
        // Create test tensors
        let batch_size = 2;
        let seq_len = 16;
        let d_model = 64;
        
        let sparse_output = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;
        let dense_output = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;
        
        let loss = loss_fn.compute_output_loss(&sparse_output, &dense_output, None)?;
        
        // Loss should be a scalar
        assert_eq!(loss.dims(), &[] as &[usize]);
        
        // Loss should be positive
        let loss_val = loss.to_scalar::<f32>()?;
        assert!(loss_val > 0.0);
        
        // When outputs are identical, loss should be ~0
        let identical_loss = loss_fn.compute_output_loss(&sparse_output, &sparse_output, None)?;
        let identical_val = identical_loss.to_scalar::<f32>()?;
        assert!(identical_val < 1e-6);
        
        Ok(())
    }

    #[test]
    fn test_dsa_alignment_loss_cosine() -> Result<()> {
        let device = Device::Cpu;
        
        let config = DSAAlignmentConfig {
            loss_type: DSAAlignmentLossType::CosineSimilarity,
            sample_fraction: 1.0,
            ..Default::default()
        };
        
        let loss_fn = DSAAlignmentLoss::new(config);
        
        let batch_size = 2;
        let seq_len = 8;
        let d_model = 32;
        
        let sparse_output = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;
        
        // When outputs are identical, cosine loss should be ~0
        let loss = loss_fn.compute_output_loss(&sparse_output, &sparse_output, None)?;
        let loss_val = loss.to_scalar::<f32>()?;
        assert!(loss_val.abs() < 1e-5, "Cosine loss for identical tensors should be ~0, got {}", loss_val);
        
        Ok(())
    }

    #[test]
    fn test_dsa_alignment_loss_with_sampling() -> Result<()> {
        let device = Device::Cpu;
        
        let config = DSAAlignmentConfig {
            sample_fraction: 0.5, // Sample 50% of positions
            loss_type: DSAAlignmentLossType::MSE,
            ..Default::default()
        };
        
        let loss_fn = DSAAlignmentLoss::new(config);
        
        let batch_size = 2;
        let seq_len = 16;
        let d_model = 64;
        
        let sparse_output = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;
        let dense_output = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;
        
        let loss = loss_fn.compute_output_loss(&sparse_output, &dense_output, None)?;
        
        // Should still produce valid loss
        assert_eq!(loss.dims(), &[] as &[usize]);
        let loss_val = loss.to_scalar::<f32>()?;
        assert!(loss_val > 0.0);
        
        Ok(())
    }

    #[test]
    fn test_dsa_training_config_warmup() {
        let config = DSATrainingConfig {
            alignment: DSAAlignmentConfig {
                alignment_weight: 0.1,
                ..Default::default()
            },
            alignment_warmup_steps: 1000,
            alignment_ramp: true,
            ..Default::default()
        };
        
        // Before warmup
        assert!((config.get_alignment_weight(0) - 0.0).abs() < 1e-10);
        
        // During warmup (linear ramp)
        assert!((config.get_alignment_weight(500) - 0.05).abs() < 1e-10);
        
        // After warmup
        assert!((config.get_alignment_weight(1000) - 0.1).abs() < 1e-10);
        assert!((config.get_alignment_weight(2000) - 0.1).abs() < 1e-10);
    }
}
