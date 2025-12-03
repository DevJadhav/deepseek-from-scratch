//! Supervised Fine-Tuning (SFT) and DPO for DeepSeek (Rust/Candle)
//!
//! This module implements:
//! - SFT loss computation
//! - Chat template formatting
//! - DPO loss computation
//! - Training utilities
//!
//! Based on: DeepSeek LLM paper and DPO paper

use candle_core::{DType, Result, Tensor, D};
use candle_nn::ops;
use std::collections::HashMap;

// =============================================================================
// Configuration
// =============================================================================

/// SFT Configuration
#[derive(Clone, Debug)]
pub struct SFTConfig {
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub max_seq_length: usize,
    pub gradient_accumulation_steps: usize,
    pub max_grad_norm: f64,
    pub weight_decay: f64,
    pub warmup_ratio: f64,
    pub use_lora: bool,
    pub lora_r: usize,
    pub lora_alpha: usize,
}

impl Default for SFTConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-5,
            num_epochs: 3,
            batch_size: 4,
            max_seq_length: 4096,
            gradient_accumulation_steps: 8,
            max_grad_norm: 1.0,
            weight_decay: 0.01,
            warmup_ratio: 0.03,
            use_lora: true,
            lora_r: 64,
            lora_alpha: 128,
        }
    }
}

/// DPO Configuration
#[derive(Clone, Debug)]
pub struct DPOConfig {
    pub beta: f64,              // KL penalty coefficient
    pub label_smoothing: f64,   // Label smoothing
    pub loss_type: DPOLossType,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub max_length: usize,
    pub max_grad_norm: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DPOLossType {
    Sigmoid,
    IPO,
    KTO,
}

impl Default for DPOConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            label_smoothing: 0.0,
            loss_type: DPOLossType::Sigmoid,
            learning_rate: 1e-6,
            batch_size: 4,
            max_length: 2048,
            max_grad_norm: 1.0,
        }
    }
}

// =============================================================================
// Chat Template
// =============================================================================

/// DeepSeek chat template
pub struct ChatTemplate {
    pub system_token: String,
    pub user_token: String,
    pub assistant_token: String,
    pub end_token: String,
}

impl Default for ChatTemplate {
    fn default() -> Self {
        Self {
            system_token: "<|system|>".to_string(),
            user_token: "<|user|>".to_string(),
            assistant_token: "<|assistant|>".to_string(),
            end_token: "<|end|>".to_string(),
        }
    }
}

impl ChatTemplate {
    /// Format a single message
    pub fn format_message(&self, role: &str, content: &str) -> String {
        let token = match role {
            "system" => &self.system_token,
            "user" => &self.user_token,
            "assistant" => &self.assistant_token,
            _ => &self.user_token,
        };
        format!("{}\n{}\n{}\n", token, content, self.end_token)
    }
    
    /// Format a conversation
    pub fn format_conversation(&self, messages: &[(String, String)]) -> String {
        messages.iter()
            .map(|(role, content)| self.format_message(role, content))
            .collect()
    }
    
    /// Format with generation prompt
    pub fn format_for_generation(&self, messages: &[(String, String)]) -> String {
        let mut formatted = self.format_conversation(messages);
        formatted.push_str(&format!("{}\n", self.assistant_token));
        formatted
    }
}

// =============================================================================
// SFT Loss
// =============================================================================

/// Compute SFT loss (cross-entropy on response tokens)
/// 
/// Args:
///   logits: (batch, seq, vocab) - Model output logits
///   labels: (batch, seq) - Target token IDs, -100 for masked positions
pub fn compute_sft_loss(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len, vocab_size) = logits.dims3()?;
    
    // Shift for next-token prediction
    let shift_logits = logits.narrow(1, 0, seq_len - 1)?;
    let shift_labels = labels.narrow(1, 1, seq_len - 1)?;
    
    // Flatten
    let flat_logits = shift_logits.reshape((batch_size * (seq_len - 1), vocab_size))?;
    let flat_labels = shift_labels.reshape((batch_size * (seq_len - 1),))?;
    
    // Log softmax
    let log_probs = ops::log_softmax(&flat_logits, D::Minus1)?;
    
    // Create mask for valid labels (not -100)
    // In Candle, we need to handle this differently
    let valid_mask = flat_labels.ge(&Tensor::zeros(flat_labels.shape(), DType::I64, flat_labels.device())?)?;
    let valid_mask_f32 = valid_mask.to_dtype(DType::F32)?;
    
    // Clamp labels for gathering (replace -100 with 0)
    let safe_labels = flat_labels.clamp(0i64, (vocab_size - 1) as i64)?;
    let safe_labels_u32 = safe_labels.to_dtype(DType::U32)?;
    
    // Gather target log probs
    let target_log_probs = log_probs
        .gather(&safe_labels_u32.unsqueeze(1)?, 1)?
        .squeeze(1)?;
    
    // Apply mask
    let masked_log_probs = (target_log_probs * &valid_mask_f32)?;
    
    // Compute mean (only over valid positions)
    let num_valid = valid_mask_f32.sum_all()?;
    let loss = (masked_log_probs.sum_all()?.neg()? / num_valid)?;
    
    Ok(loss)
}

/// Compute per-token log probabilities for a sequence
pub fn compute_token_log_probs(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
    let (_batch_size, seq_len, _vocab_size) = logits.dims3()?;
    
    // Log softmax
    let log_probs = ops::log_softmax(logits, D::Minus1)?;
    
    // Gather target log probs
    let labels_u32 = labels.to_dtype(DType::U32)?;
    let token_log_probs = log_probs
        .gather(&labels_u32.unsqueeze(D::Minus1)?, D::Minus1)?
        .squeeze(D::Minus1)?;
    
    Ok(token_log_probs)
}

/// Compute sequence log probability (sum of token log probs)
pub fn compute_seq_log_prob(
    logits: &Tensor,
    labels: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let token_log_probs = compute_token_log_probs(logits, labels)?;
    
    // Apply mask if provided
    let masked_log_probs = if let Some(m) = mask {
        (token_log_probs * m.to_dtype(DType::F32)?)?
    } else {
        token_log_probs
    };
    
    // Sum over sequence dimension
    masked_log_probs.sum(D::Minus1)
}

// =============================================================================
// DPO Loss
// =============================================================================

/// DPO training metrics
#[derive(Debug, Clone)]
pub struct DPOMetrics {
    pub loss: f32,
    pub chosen_reward: f32,
    pub rejected_reward: f32,
    pub accuracy: f32,
    pub margin: f32,
}

/// Compute DPO loss
/// 
/// L = -log σ(β * (log π/π_ref(chosen) - log π/π_ref(rejected)))
pub fn compute_dpo_loss(
    policy_chosen_logps: &Tensor,
    policy_rejected_logps: &Tensor,
    ref_chosen_logps: &Tensor,
    ref_rejected_logps: &Tensor,
    config: &DPOConfig,
) -> Result<(Tensor, DPOMetrics)> {
    // Log ratios
    let chosen_logratios = (policy_chosen_logps - ref_chosen_logps)?;
    let rejected_logratios = (policy_rejected_logps - ref_rejected_logps)?;
    
    // DPO logits: β * (chosen_logratio - rejected_logratio)
    let logits = ((chosen_logratios.clone() - rejected_logratios.clone())? * config.beta)?;
    
    // Compute loss based on type
    let loss = match config.loss_type {
        DPOLossType::Sigmoid => {
            // Standard DPO: -log σ(logits)
            // log σ(x) = x - log(1 + exp(x)) = -softplus(-x)
            let neg_logits = logits.clone().neg()?;
            // softplus(-x) = log(1 + exp(-x))
            let exp_neg = neg_logits.exp()?;
            let one_plus_exp = (exp_neg + 1.0)?;
            one_plus_exp.log()?.mean_all()?
        }
        DPOLossType::IPO => {
            // IPO: (logits - 1/(2β))²
            let target = 1.0 / (2.0 * config.beta);
            let diff = (logits.clone() - target)?;
            diff.sqr()?.mean_all()?
        }
        DPOLossType::KTO => {
            // KTO: 1 - σ(β * logratio) for both
            let sigmoid_chosen = ops::sigmoid(&(chosen_logratios.clone() * config.beta)?)?;
            let sigmoid_rejected = ops::sigmoid(&(rejected_logratios.neg()? * config.beta)?)?;
            let chosen_loss = 1.0 - sigmoid_chosen.mean_all()?.to_scalar::<f32>()?;
            let rejected_loss = 1.0 - sigmoid_rejected.mean_all()?.to_scalar::<f32>()?;
            Tensor::new(chosen_loss + rejected_loss, logits.device())?
        }
    };
    
    // Apply label smoothing for sigmoid loss
    let final_loss = if config.label_smoothing > 0.0 && config.loss_type == DPOLossType::Sigmoid {
        let smooth_loss = {
            let exp_logits = logits.exp()?;
            let one_plus_exp = (exp_logits + 1.0)?;
            one_plus_exp.log()?.mean_all()?
        };
        ((loss * (1.0 - config.label_smoothing))? + (smooth_loss * config.label_smoothing)?)?
    } else {
        loss
    };
    
    // Compute metrics
    let chosen_rewards = (chosen_logratios * config.beta)?;
    let rejected_rewards = (rejected_logratios * config.beta)?;
    let margin = (chosen_rewards.clone() - rejected_rewards.clone())?;
    
    let accuracy = chosen_rewards.gt(&rejected_rewards)?
        .to_dtype(DType::F32)?
        .mean_all()?
        .to_scalar::<f32>()?;
    
    let metrics = DPOMetrics {
        loss: final_loss.to_scalar::<f32>()?,
        chosen_reward: chosen_rewards.mean_all()?.to_scalar::<f32>()?,
        rejected_reward: rejected_rewards.mean_all()?.to_scalar::<f32>()?,
        accuracy,
        margin: margin.mean_all()?.to_scalar::<f32>()?,
    };
    
    Ok((final_loss, metrics))
}

// =============================================================================
// LoRA Components (Simplified)
// =============================================================================

/// LoRA configuration
#[derive(Clone, Debug)]
pub struct LoRAConfig {
    pub r: usize,          // Rank
    pub alpha: usize,      // Scaling factor
    pub dropout: f64,      // Dropout probability
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            r: 64,
            alpha: 128,
            dropout: 0.05,
        }
    }
}

impl LoRAConfig {
    pub fn scaling(&self) -> f64 {
        self.alpha as f64 / self.r as f64
    }
}

/// Compute LoRA parameter count
pub fn lora_param_count(
    hidden_size: usize,
    num_layers: usize,
    lora_r: usize,
    num_target_modules: usize,
) -> usize {
    // Each LoRA layer has A (in x r) and B (r x out) matrices
    // For attention: q, k, v, o projections = 4 modules
    // For MLP: gate, up, down = 3 modules
    let params_per_layer = 2 * hidden_size * lora_r * num_target_modules;
    params_per_layer * num_layers
}

// =============================================================================
// Training Utilities
// =============================================================================

/// Training step result
#[derive(Debug, Clone)]
pub struct TrainStepResult {
    pub loss: f32,
    pub grad_norm: f32,
    pub metrics: HashMap<String, f32>,
}

/// Compute gradient norm
pub fn compute_grad_norm(tensors: &[&Tensor]) -> Result<f32> {
    let mut total_norm_sq = 0.0f32;
    
    for t in tensors {
        let norm_sq = t.sqr()?.sum_all()?.to_scalar::<f32>()?;
        total_norm_sq += norm_sq;
    }
    
    Ok(total_norm_sq.sqrt())
}

// =============================================================================
// Demo
// =============================================================================



#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chat_template() {
        let template = ChatTemplate::default();
        let msg = template.format_message("user", "Hello");
        assert!(msg.contains("<|user|>"));
        assert!(msg.contains("Hello"));
        assert!(msg.contains("<|end|>"));
    }
    
    #[test]
    fn test_lora_config() {
        let config = LoRAConfig::default();
        assert_eq!(config.scaling(), 2.0);  // 128/64
    }
    
    #[test]
    fn test_dpo_config() {
        let config = DPOConfig::default();
        assert_eq!(config.beta, 0.1);
        assert_eq!(config.loss_type, DPOLossType::Sigmoid);
    }
}
