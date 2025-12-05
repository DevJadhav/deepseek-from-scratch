//! Knowledge Distillation for DeepSeek (Rust/Candle)
//!
//! This module implements:
//! - Standard knowledge distillation (KD)
//! - Sequence-level knowledge distillation (SeqKD)
//! - Progressive distillation
//! - Feature distillation
//!
//! Based on: DeepSeek distillation techniques and Hinton et al.

use candle_core::{DType, Result, Tensor, D};
use candle_nn::ops;

// =============================================================================
// Configuration
// =============================================================================

/// Knowledge distillation configuration
#[derive(Clone, Debug)]
pub struct DistillationConfig {
    pub temperature: f64,           // Softmax temperature
    pub alpha: f64,                 // Weight for distillation loss vs task loss
    pub kd_loss_type: KDLossType,   // Type of KD loss
    pub use_hard_labels: bool,      // Include hard label loss
    pub hidden_distill: bool,       // Distill hidden states
    pub attention_distill: bool,    // Distill attention patterns
}

#[derive(Clone, Debug, PartialEq)]
pub enum KDLossType {
    KLDivergence,
    MSE,
    Cosine,
    JSD,  // Jensen-Shannon Divergence
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.5,
            kd_loss_type: KDLossType::KLDivergence,
            use_hard_labels: true,
            hidden_distill: false,
            attention_distill: false,
        }
    }
}

/// Progressive distillation configuration
#[derive(Clone, Debug)]
pub struct ProgressiveConfig {
    pub num_stages: usize,
    pub layer_mapping: LayerMapping,
    pub intermediate_sizes: Vec<usize>,
    pub temperature_schedule: TemperatureSchedule,
}

#[derive(Clone, Debug)]
pub enum LayerMapping {
    Uniform,              // Evenly spaced layers
    TopLayers,            // Focus on top layers
    Custom(Vec<usize>),   // Custom mapping
}

#[derive(Clone, Debug)]
pub enum TemperatureSchedule {
    Constant(f64),
    Linear { start: f64, end: f64 },
    Cosine { start: f64, end: f64 },
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            num_stages: 3,
            layer_mapping: LayerMapping::Uniform,
            intermediate_sizes: vec![7168, 4096, 2048],
            temperature_schedule: TemperatureSchedule::Linear { start: 6.0, end: 2.0 },
        }
    }
}

impl TemperatureSchedule {
    pub fn get_temperature(&self, step: usize, total_steps: usize) -> f64 {
        let progress = step as f64 / total_steps.max(1) as f64;
        
        match self {
            TemperatureSchedule::Constant(t) => *t,
            TemperatureSchedule::Linear { start, end } => {
                start + (end - start) * progress
            }
            TemperatureSchedule::Cosine { start, end } => {
                let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
                end + (start - end) * cosine
            }
        }
    }
}

// =============================================================================
// Core Distillation Losses
// =============================================================================

/// Standard knowledge distillation loss (KL divergence)
/// 
/// L_KD = T² * KL(softmax(s_t/T) || softmax(s_s/T))
/// 
/// where s_t = teacher logits, s_s = student logits, T = temperature
pub fn kd_loss_kl(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    temperature: f64,
) -> Result<Tensor> {
    // Scale by temperature
    let student_scaled = (student_logits / temperature)?;
    let teacher_scaled = (teacher_logits / temperature)?;
    
    // Compute log softmax for student, softmax for teacher
    let student_log_probs = ops::log_softmax(&student_scaled, D::Minus1)?;
    let teacher_probs = ops::softmax(&teacher_scaled, D::Minus1)?;
    
    // KL divergence: sum(p * (log(p) - log(q)))
    let teacher_log_probs = ops::log_softmax(&teacher_scaled, D::Minus1)?;
    let kl = (teacher_probs * (teacher_log_probs - student_log_probs)?)?;
    
    // Sum over vocab, mean over batch and sequence
    let loss = kl.sum(D::Minus1)?.mean_all()?;
    
    // Scale by T² (Hinton et al.)
    let scaled_loss = (loss * (temperature * temperature))?;
    
    Ok(scaled_loss)
}

/// Jensen-Shannon divergence for distillation
/// JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q)
pub fn kd_loss_jsd(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    temperature: f64,
) -> Result<Tensor> {
    let student_scaled = (student_logits / temperature)?;
    let teacher_scaled = (teacher_logits / temperature)?;
    
    let student_probs = ops::softmax(&student_scaled, D::Minus1)?;
    let teacher_probs = ops::softmax(&teacher_scaled, D::Minus1)?;
    
    // M = (P + Q) / 2
    let m_probs = ((student_probs.clone() + teacher_probs.clone())? / 2.0)?;
    let m_log_probs = m_probs.log()?;
    
    // KL(P || M)
    let student_log_probs = student_probs.log()?;
    let kl_pm = (student_probs * (student_log_probs - &m_log_probs)?)?.sum(D::Minus1)?;
    
    // KL(Q || M)  
    let teacher_log_probs = teacher_probs.log()?;
    let kl_qm = (teacher_probs * (teacher_log_probs - m_log_probs)?)?.sum(D::Minus1)?;
    
    // JSD = 0.5 * (KL_PM + KL_QM)
    ((kl_pm + kl_qm)? / 2.0)?.mean_all()
}

/// MSE loss for distillation (on logits or probabilities)
pub fn kd_loss_mse(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    temperature: f64,
) -> Result<Tensor> {
    let student_scaled = (student_logits / temperature)?;
    let teacher_scaled = (teacher_logits / temperature)?;
    
    // Apply softmax to get probabilities
    let student_probs = ops::softmax(&student_scaled, D::Minus1)?;
    let teacher_probs = ops::softmax(&teacher_scaled, D::Minus1)?;
    
    // MSE
    let diff = (student_probs - teacher_probs)?;
    diff.sqr()?.mean_all()
}

/// Cosine similarity loss for distillation
pub fn kd_loss_cosine(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
) -> Result<Tensor> {
    // Normalize along vocab dimension
    let student_norm = (student_logits.sqr()?.sum(D::Minus1)?.sqrt()? + 1e-8)?;
    let teacher_norm = (teacher_logits.sqr()?.sum(D::Minus1)?.sqrt()? + 1e-8)?;
    
    let student_normalized = (student_logits / student_norm.unsqueeze(D::Minus1)?.broadcast_as(student_logits.shape())?)?;
    let teacher_normalized = (teacher_logits / teacher_norm.unsqueeze(D::Minus1)?.broadcast_as(teacher_logits.shape())?)?;
    
    // Cosine similarity
    let cos_sim = (student_normalized * teacher_normalized)?.sum(D::Minus1)?;
    
    // Loss = 1 - cos_sim (minimize distance)
    let one = Tensor::ones(cos_sim.shape(), cos_sim.dtype(), cos_sim.device())?;
    (one - cos_sim)?.mean_all()
}

/// Compute distillation loss based on config
pub fn compute_distillation_loss(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    config: &DistillationConfig,
) -> Result<Tensor> {
    match config.kd_loss_type {
        KDLossType::KLDivergence => kd_loss_kl(student_logits, teacher_logits, config.temperature),
        KDLossType::MSE => kd_loss_mse(student_logits, teacher_logits, config.temperature),
        KDLossType::Cosine => kd_loss_cosine(student_logits, teacher_logits),
        KDLossType::JSD => kd_loss_jsd(student_logits, teacher_logits, config.temperature),
    }
}

// =============================================================================
// Combined Distillation Loss
// =============================================================================

/// Combined distillation loss with hard labels
/// 
/// L = α * L_KD + (1 - α) * L_CE
pub fn combined_distillation_loss(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    labels: &Tensor,
    config: &DistillationConfig,
) -> Result<DistillationMetrics> {
    // Distillation loss
    let kd_loss = compute_distillation_loss(student_logits, teacher_logits, config)?;
    
    // Hard label loss (cross-entropy)
    let ce_loss = if config.use_hard_labels {
        let log_probs = ops::log_softmax(student_logits, D::Minus1)?;
        
        // Gather target log probs
        let labels_u32 = labels.to_dtype(DType::U32)?;
        let target_log_probs = log_probs
            .gather(&labels_u32.unsqueeze(D::Minus1)?, D::Minus1)?
            .squeeze(D::Minus1)?;
        
        target_log_probs.neg()?.mean_all()?
    } else {
        Tensor::zeros((), DType::F32, student_logits.device())?
    };
    
    // Combined loss
    let alpha = config.alpha;
    let total_loss = if config.use_hard_labels {
        ((kd_loss.clone() * alpha)? + (ce_loss.clone() * (1.0 - alpha))?)?
    } else {
        kd_loss.clone()
    };
    
    let metrics = DistillationMetrics {
        total_loss: total_loss.to_scalar::<f32>()?,
        kd_loss: kd_loss.to_scalar::<f32>()?,
        ce_loss: ce_loss.to_scalar::<f32>()?,
    };
    
    Ok(metrics)
}

#[derive(Debug, Clone)]
pub struct DistillationMetrics {
    pub total_loss: f32,
    pub kd_loss: f32,
    pub ce_loss: f32,
}

// =============================================================================
// Hidden State Distillation
// =============================================================================

/// Distill hidden states between teacher and student
/// 
/// L_hidden = MSE(proj(student_hidden), teacher_hidden)
pub fn hidden_state_distillation(
    student_hidden: &Tensor,
    teacher_hidden: &Tensor,
    projection: Option<&Tensor>,
) -> Result<Tensor> {
    let student_proj = if let Some(proj) = projection {
        // Project student to teacher dimension
        let (b, s, h) = student_hidden.dims3()?;
        let flat = student_hidden.reshape((b * s, h))?;
        let projected = flat.matmul(proj)?;
        projected.reshape((b, s, proj.dim(1)?))?
    } else {
        student_hidden.clone()
    };
    
    // MSE loss
    let diff = (student_proj - teacher_hidden)?;
    diff.sqr()?.mean_all()
}

/// Distill attention patterns
/// 
/// L_attn = KL(student_attn || teacher_attn)
pub fn attention_distillation(
    student_attn: &Tensor,  // (batch, heads, seq, seq)
    teacher_attn: &Tensor,
) -> Result<Tensor> {
    // Average over heads if different
    let s_heads = student_attn.dim(1)?;
    let t_heads = teacher_attn.dim(1)?;
    
    let student_avg = if s_heads != t_heads {
        student_attn.mean(1)?
    } else {
        student_attn.clone()
    };
    
    let teacher_avg = if s_heads != t_heads {
        teacher_attn.mean(1)?
    } else {
        teacher_attn.clone()
    };
    
    // KL divergence over attention distribution
    let student_log = (student_avg.clone() + 1e-10)?.log()?;
    let teacher_log = (teacher_avg.clone() + 1e-10)?.log()?;
    
    let kl = (teacher_avg * (teacher_log - student_log)?)?;
    kl.sum(D::Minus1)?.mean_all()
}

// =============================================================================
// Sequence-Level Distillation
// =============================================================================

/// Sequence-level knowledge distillation
/// Uses teacher-generated sequences for training
#[derive(Clone, Debug)]
pub struct SeqKDConfig {
    pub num_samples: usize,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub mix_ratio: f64,  // Ratio of teacher vs ground truth sequences
}

impl Default for SeqKDConfig {
    fn default() -> Self {
        Self {
            num_samples: 4,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.95,
            mix_ratio: 0.5,
        }
    }
}

/// Generate teacher sequences for SeqKD
/// Note: Simplified - actual implementation would use autoregressive generation
pub fn generate_teacher_sequences(
    _prompt_ids: &Tensor,
    _config: &SeqKDConfig,
) -> Result<Tensor> {
    // Placeholder - actual implementation would:
    // 1. Generate sequences autoregressively from teacher
    // 2. Apply top-k/top-p sampling
    // 3. Return generated token IDs
    unimplemented!("Teacher generation requires full model inference")
}

// =============================================================================
// Progressive Distillation
// =============================================================================

/// Progressive distillation manager
pub struct ProgressiveDistiller {
    pub config: ProgressiveConfig,
    pub current_stage: usize,
    pub steps_per_stage: usize,
    pub current_step: usize,
}

impl ProgressiveDistiller {
    pub fn new(config: ProgressiveConfig, total_steps: usize) -> Self {
        let steps_per_stage = total_steps / config.num_stages.max(1);
        Self {
            config,
            current_stage: 0,
            steps_per_stage,
            current_step: 0,
        }
    }
    
    /// Get current stage index
    pub fn current_stage(&self) -> usize {
        self.current_stage
    }
    
    /// Get current temperature based on schedule
    pub fn current_temperature(&self) -> f64 {
        let total_steps = self.steps_per_stage * self.config.num_stages;
        self.config.temperature_schedule.get_temperature(self.current_step, total_steps)
    }
    
    /// Get layer indices to distill from teacher
    pub fn get_teacher_layers(&self) -> Vec<usize> {
        let total_teacher_layers = 60;  // Example: DeepSeek-V2 has 60 layers
        
        match &self.config.layer_mapping {
            LayerMapping::Uniform => {
                let step = total_teacher_layers / self.config.num_stages;
                (0..total_teacher_layers).step_by(step).collect()
            }
            LayerMapping::TopLayers => {
                let start = total_teacher_layers.saturating_sub(self.config.num_stages * 4);
                (start..total_teacher_layers).step_by(4).collect()
            }
            LayerMapping::Custom(layers) => layers.clone(),
        }
    }
    
    /// Step the distiller
    pub fn step(&mut self) {
        self.current_step += 1;
        if self.current_step >= self.steps_per_stage * (self.current_stage + 1) {
            self.current_stage = (self.current_stage + 1).min(self.config.num_stages - 1);
        }
    }
    
    /// Check if should advance to next stage
    pub fn should_advance_stage(&self) -> bool {
        self.current_step > 0 && 
        self.current_step % self.steps_per_stage == 0 &&
        self.current_stage < self.config.num_stages - 1
    }
    
    /// Get current intermediate size for student
    pub fn current_intermediate_size(&self) -> usize {
        self.config.intermediate_sizes.get(self.current_stage)
            .copied()
            .unwrap_or(self.config.intermediate_sizes[self.config.intermediate_sizes.len() - 1])
    }
}

// =============================================================================
// Layer Matching for Different Model Sizes
// =============================================================================

/// Match student layers to teacher layers
pub fn compute_layer_mapping(
    student_layers: usize,
    teacher_layers: usize,
) -> Vec<(usize, usize)> {
    // Simple uniform mapping
    let ratio = teacher_layers as f64 / student_layers as f64;
    
    (0..student_layers)
        .map(|s| {
            let t = ((s as f64 + 0.5) * ratio).floor() as usize;
            (s, t.min(teacher_layers - 1))
        })
        .collect()
}

/// Match layers with emphasis on top layers
pub fn compute_layer_mapping_top_heavy(
    student_layers: usize,
    teacher_layers: usize,
) -> Vec<(usize, usize)> {
    // More student layers map to top teacher layers
    let mut mapping = Vec::with_capacity(student_layers);
    
    let bottom_student = student_layers / 2;
    let bottom_teacher = teacher_layers / 3;
    
    // Bottom half of student -> bottom third of teacher
    for s in 0..bottom_student {
        let t = (s as f64 / bottom_student as f64 * bottom_teacher as f64) as usize;
        mapping.push((s, t));
    }
    
    // Top half of student -> top two-thirds of teacher
    let remaining_student = student_layers - bottom_student;
    let remaining_teacher = teacher_layers - bottom_teacher;
    
    for s in bottom_student..student_layers {
        let local_s = s - bottom_student;
        let t = bottom_teacher + 
            (local_s as f64 / remaining_student as f64 * remaining_teacher as f64) as usize;
        mapping.push((s, t.min(teacher_layers - 1)));
    }
    
    mapping
}

// =============================================================================
// Distillation Trainer State
// =============================================================================

/// Distillation training state
#[derive(Debug, Clone)]
pub struct DistillationState {
    pub step: usize,
    pub epoch: usize,
    pub total_loss: f32,
    pub kd_loss: f32,
    pub ce_loss: f32,
    pub hidden_loss: f32,
    pub attention_loss: f32,
    pub temperature: f64,
}

impl Default for DistillationState {
    fn default() -> Self {
        Self {
            step: 0,
            epoch: 0,
            total_loss: 0.0,
            kd_loss: 0.0,
            ce_loss: 0.0,
            hidden_loss: 0.0,
            attention_loss: 0.0,
            temperature: 4.0,
        }
    }
}

// =============================================================================
// Demo
// =============================================================================



#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_temperature_schedule() {
        let linear = TemperatureSchedule::Linear { start: 6.0, end: 2.0 };
        assert!((linear.get_temperature(0, 1000) - 6.0).abs() < 0.01);
        assert!((linear.get_temperature(500, 1000) - 4.0).abs() < 0.01);
        assert!((linear.get_temperature(1000, 1000) - 2.0).abs() < 0.01);
    }
    
    #[test]
    fn test_layer_mapping() {
        let mapping = compute_layer_mapping(12, 60);
        assert_eq!(mapping.len(), 12);
        assert_eq!(mapping[0].0, 0);
        assert!(mapping.last().unwrap().1 < 60);
    }
    
    #[test]
    fn test_progressive_distiller() {
        let config = ProgressiveConfig::default();
        let mut distiller = ProgressiveDistiller::new(config, 900);
        
        assert_eq!(distiller.current_stage(), 0);
        
        for _ in 0..301 {
            distiller.step();
        }
        assert_eq!(distiller.current_stage(), 1);
        
        for _ in 0..300 {
            distiller.step();
        }
        assert_eq!(distiller.current_stage(), 2);
    }
}
