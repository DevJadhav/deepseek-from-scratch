//! Reward Model for RLHF (Rust/Candle)
//!
//! This module implements:
//! - Bradley-Terry reward model architecture
//! - Preference loss computation  
//! - Reward head for language models
//!
//! Based on: InstructGPT and DeepSeek RLHF papers

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear, ops, Linear, Module, VarBuilder};

// =============================================================================
// Configuration
// =============================================================================

/// Reward Model Configuration
#[derive(Clone, Debug)]
pub struct RewardModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub use_layernorm: bool,
    pub dropout: f64,
}

impl Default for RewardModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 2,
            use_layernorm: true,
            dropout: 0.1,
        }
    }
}

/// Reward Training Configuration
#[derive(Clone, Debug)]
pub struct RewardTrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub max_length: usize,
    pub num_epochs: usize,
    pub warmup_ratio: f64,
    pub weight_decay: f64,
    pub max_grad_norm: f64,
    pub label_smoothing: f64,
}

impl Default for RewardTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-5,
            batch_size: 4,
            max_length: 2048,
            num_epochs: 1,
            warmup_ratio: 0.03,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            label_smoothing: 0.0,
        }
    }
}

// =============================================================================
// Reward Head
// =============================================================================

/// Simple reward head (single linear layer)
pub struct SimpleRewardHead {
    linear: Linear,
}

impl SimpleRewardHead {
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear = linear(hidden_size, 1, vb.pp("linear"))?;
        Ok(Self { linear })
    }
}

impl Module for SimpleRewardHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)?.squeeze(D::Minus1)
    }
}

/// MLP reward head (more expressive)
pub struct MLPRewardHead {
    fc1: Linear,
    fc2: Linear,
}

impl MLPRewardHead {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(hidden_size, intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(intermediate_size, 1, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for MLPRewardHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?;
        // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let h = {
            let x3 = h.powf(3.0)?;
            let inner = ((&h + (x3 * 0.044715)?)? * 0.7978845608)?;
            let tanh_inner = inner.tanh()?;
            let gate = ((tanh_inner + 1.0)? * 0.5)?;
            (&h * gate)?
        };
        self.fc2.forward(&h)?.squeeze(D::Minus1)
    }
}

// =============================================================================
// Reward Model
// =============================================================================

/// Full reward model with backbone + head
/// In practice, the backbone would be a pretrained LLM
pub struct RewardModel {
    head: MLPRewardHead,
    config: RewardModelConfig,
}

impl RewardModel {
    pub fn new(config: &RewardModelConfig, vb: VarBuilder) -> Result<Self> {
        let head = MLPRewardHead::new(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("reward_head"),
        )?;
        
        Ok(Self {
            head,
            config: config.clone(),
        })
    }
    
    /// Compute reward from last hidden states
    /// 
    /// Args:
    ///   hidden_states: (batch, seq, hidden) - Last layer hidden states
    ///   attention_mask: (batch, seq) - 1 for real tokens, 0 for padding
    /// 
    /// Returns:
    ///   rewards: (batch,) - Scalar reward for each sequence
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden_size) = hidden_states.dims3()?;
        
        // Find last non-padding position for each sequence
        // Sum along seq dimension to get counts
        let seq_lengths = attention_mask.sum(D::Minus1)?;
        
        // Gather last hidden states
        // Create indices for gathering: (batch, 1, hidden)
        let indices = (seq_lengths - 1.0)?
            .to_dtype(DType::U32)?
            .unsqueeze(1)?;
        
        // Expand indices to match hidden size
        let hidden_size = hidden_states.dim(2)?;
        let indices_expanded = indices.broadcast_as((batch_size, 1, hidden_size))?;
        
        // Gather last token hidden states
        let last_hidden = hidden_states.gather(&indices_expanded, 1)?.squeeze(1)?;
        
        // Pass through reward head
        self.head.forward(&last_hidden)
    }
    
    /// Alternative: Pool over all positions (mean pooling)
    pub fn forward_mean_pool(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let mask = attention_mask.unsqueeze(D::Minus1)?.to_dtype(DType::F32)?;
        
        // Masked sum
        let masked_hidden = (hidden_states * &mask)?;
        let summed = masked_hidden.sum(1)?;
        
        // Normalize by sequence length
        let lengths = mask.sum(1)?;
        let pooled = (summed / lengths)?;
        
        self.head.forward(&pooled)
    }
}

// =============================================================================
// Preference Loss
// =============================================================================

/// Bradley-Terry preference loss
/// 
/// P(chosen > rejected) = σ(r_chosen - r_rejected)
/// Loss = -log P(chosen > rejected) = -log σ(r_chosen - r_rejected)
pub fn compute_preference_loss(
    chosen_rewards: &Tensor,
    rejected_rewards: &Tensor,
    label_smoothing: f64,
) -> Result<(Tensor, PreferenceLossMetrics)> {
    // Reward difference
    let diff = (chosen_rewards - rejected_rewards)?;
    
    // Preference probability: σ(diff)
    let preference_prob = ops::sigmoid(&diff)?;
    
    // Log preference probability
    let log_prob = preference_prob.log()?;
    
    // Base loss: -log σ(diff)
    let loss = log_prob.neg()?.mean_all()?;
    
    // Apply label smoothing if specified
    let smoothed_loss = if label_smoothing > 0.0 {
        // Also consider the opposite preference
        let reverse_log_prob = (rejected_rewards - chosen_rewards)?.log()?;
        let reverse_loss = reverse_log_prob.neg()?.mean_all()?;
        ((loss * (1.0 - label_smoothing))? + (reverse_loss * label_smoothing)?)?
    } else {
        loss
    };
    
    // Compute metrics
    let accuracy = diff.gt(&Tensor::zeros(diff.shape(), DType::F32, diff.device())?)?
        .to_dtype(DType::F32)?
        .mean_all()?
        .to_scalar::<f32>()?;
    
    let metrics = PreferenceLossMetrics {
        loss: smoothed_loss.to_scalar::<f32>()?,
        chosen_mean: chosen_rewards.mean_all()?.to_scalar::<f32>()?,
        rejected_mean: rejected_rewards.mean_all()?.to_scalar::<f32>()?,
        margin: diff.mean_all()?.to_scalar::<f32>()?,
        accuracy,
    };
    
    Ok((smoothed_loss, metrics))
}

/// Preference loss with margin
/// Loss = max(0, margin - (r_chosen - r_rejected)) + reg
pub fn compute_margin_loss(
    chosen_rewards: &Tensor,
    rejected_rewards: &Tensor,
    margin: f64,
) -> Result<Tensor> {
    let diff = (chosen_rewards - rejected_rewards)?;
    let margin_tensor = Tensor::new(margin as f32, diff.device())?.broadcast_as(diff.shape())?;
    let loss = (margin_tensor - diff)?.relu()?.mean_all()?;
    Ok(loss)
}

/// Preference loss metrics
#[derive(Debug, Clone)]
pub struct PreferenceLossMetrics {
    pub loss: f32,
    pub chosen_mean: f32,
    pub rejected_mean: f32,
    pub margin: f32,
    pub accuracy: f32,
}

// =============================================================================
// Reward Normalization
// =============================================================================

/// Running statistics for reward normalization
pub struct RewardNormalizer {
    pub mean: f32,
    pub var: f32,
    pub count: usize,
    pub eps: f32,
}

impl Default for RewardNormalizer {
    fn default() -> Self {
        Self {
            mean: 0.0,
            var: 1.0,
            count: 0,
            eps: 1e-8,
        }
    }
}

impl RewardNormalizer {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Update running statistics with a batch of rewards
    pub fn update(&mut self, rewards: &[f32]) {
        for r in rewards {
            self.count += 1;
            let delta = r - self.mean;
            self.mean += delta / self.count as f32;
            let delta2 = r - self.mean;
            self.var += (delta * delta2 - self.var) / self.count as f32;
        }
    }
    
    /// Normalize rewards using running statistics
    pub fn normalize(&self, rewards: &Tensor) -> Result<Tensor> {
        let std = (self.var + self.eps).sqrt();
        ((rewards - self.mean as f64)? / std as f64).map(|t| t)
    }
    
    /// Get standard deviation
    pub fn std(&self) -> f32 {
        (self.var + self.eps).sqrt()
    }
}

// =============================================================================
// Preference Data
// =============================================================================

/// A single preference example
#[derive(Clone, Debug)]
pub struct PreferencePair {
    pub prompt: String,
    pub chosen: String,
    pub rejected: String,
    pub chosen_score: Option<f32>,
    pub rejected_score: Option<f32>,
}

/// Batch of preference pairs (tokenized)
#[derive(Clone, Debug)]
pub struct PreferenceBatch {
    pub chosen_ids: Vec<Vec<i64>>,
    pub rejected_ids: Vec<Vec<i64>>,
    pub chosen_mask: Vec<Vec<i64>>,
    pub rejected_mask: Vec<Vec<i64>>,
}

impl PreferenceBatch {
    /// Convert to tensors
    pub fn to_tensors(&self, device: &Device) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let batch_size = self.chosen_ids.len();
        let max_len_chosen = self.chosen_ids.iter().map(|x| x.len()).max().unwrap_or(0);
        let max_len_rejected = self.rejected_ids.iter().map(|x| x.len()).max().unwrap_or(0);
        
        // Flatten and pad
        let mut chosen_flat: Vec<i64> = vec![0; batch_size * max_len_chosen];
        let mut rejected_flat: Vec<i64> = vec![0; batch_size * max_len_rejected];
        let mut chosen_mask_flat: Vec<i64> = vec![0; batch_size * max_len_chosen];
        let mut rejected_mask_flat: Vec<i64> = vec![0; batch_size * max_len_rejected];
        
        for (i, ids) in self.chosen_ids.iter().enumerate() {
            for (j, &id) in ids.iter().enumerate() {
                chosen_flat[i * max_len_chosen + j] = id;
            }
        }
        
        for (i, ids) in self.rejected_ids.iter().enumerate() {
            for (j, &id) in ids.iter().enumerate() {
                rejected_flat[i * max_len_rejected + j] = id;
            }
        }
        
        for (i, mask) in self.chosen_mask.iter().enumerate() {
            for (j, &m) in mask.iter().enumerate() {
                chosen_mask_flat[i * max_len_chosen + j] = m;
            }
        }
        
        for (i, mask) in self.rejected_mask.iter().enumerate() {
            for (j, &m) in mask.iter().enumerate() {
                rejected_mask_flat[i * max_len_rejected + j] = m;
            }
        }
        
        let chosen_ids = Tensor::from_vec(chosen_flat, (batch_size, max_len_chosen), device)?;
        let rejected_ids = Tensor::from_vec(rejected_flat, (batch_size, max_len_rejected), device)?;
        let chosen_mask = Tensor::from_vec(chosen_mask_flat, (batch_size, max_len_chosen), device)?;
        let rejected_mask = Tensor::from_vec(rejected_mask_flat, (batch_size, max_len_rejected), device)?;
        
        Ok((chosen_ids, rejected_ids, chosen_mask, rejected_mask))
    }
}

// =============================================================================
// Reward Aggregation
// =============================================================================

/// Different methods for aggregating per-token rewards
#[derive(Clone, Debug)]
pub enum RewardAggregation {
    LastToken,     // Use reward at EOS token
    MeanPool,      // Average over all tokens
    MaxPool,       // Maximum reward over tokens
    WeightedMean,  // Weighted by position
}

impl RewardAggregation {
    /// Aggregate token-level rewards to sequence-level
    pub fn aggregate(
        &self,
        token_rewards: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let mask_f32 = mask.to_dtype(DType::F32)?;
        
        match self {
            RewardAggregation::LastToken => {
                // Find last token position
                let seq_lens = mask_f32.sum(D::Minus1)?;
                let batch_size = token_rewards.dim(0)?;
                let hidden_size = token_rewards.dim(2)?;
                
                let indices = (seq_lens - 1.0)?
                    .to_dtype(DType::U32)?
                    .unsqueeze(1)?
                    .broadcast_as((batch_size, 1, hidden_size))?;
                
                token_rewards.gather(&indices, 1)?.squeeze(1)
            }
            RewardAggregation::MeanPool => {
                let mask_3d = mask_f32.unsqueeze(D::Minus1)?.broadcast_as(token_rewards.shape())?;
                let masked = (token_rewards * &mask_3d)?;
                let summed = masked.sum(1)?;
                let lengths = mask_3d.sum(1)?;
                summed / lengths
            }
            RewardAggregation::MaxPool => {
                // Mask out padding with large negative value
                let mask_3d = mask_f32.unsqueeze(D::Minus1)?.broadcast_as(token_rewards.shape())?;
                let neg_inf_val = Tensor::new(-1e9f32, token_rewards.device())?.broadcast_as(token_rewards.shape())?;
                let neg_inf = ((mask_3d.neg()? + 1.0)? * neg_inf_val)?;
                let masked = (token_rewards + neg_inf)?;
                masked.max(1)
            }
            RewardAggregation::WeightedMean => {
                // Weight by position (later tokens weighted more)
                let seq_len = token_rewards.dim(1)?;
                let positions: Vec<f32> = (1..=seq_len).map(|x| x as f32).collect();
                let weights = Tensor::from_vec(positions, seq_len, token_rewards.device())?
                    .unsqueeze(0)?;
                
                let mask_3d = mask_f32.unsqueeze(D::Minus1)?.broadcast_as(token_rewards.shape())?;
                let weights_3d = (weights * &mask_f32)?.unsqueeze(D::Minus1)?.broadcast_as(token_rewards.shape())?;
                
                let weighted = (token_rewards * &weights_3d)?;
                let summed = weighted.sum(1)?;
                let total_weight = weights_3d.sum(1)?;
                summed / total_weight
            }
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
    fn test_preference_loss() -> Result<()> {
        let device = Device::Cpu;
        let chosen = Tensor::from_vec(vec![1.0f32, 2.0], 2, &device)?;
        let rejected = Tensor::from_vec(vec![0.5f32, 0.8], 2, &device)?;
        
        let (loss, metrics) = compute_preference_loss(&chosen, &rejected, 0.0)?;
        
        assert!(loss.to_scalar::<f32>()? > 0.0);
        assert_eq!(metrics.accuracy, 1.0);
        Ok(())
    }
    
    #[test]
    fn test_reward_normalizer() {
        let mut normalizer = RewardNormalizer::new();
        normalizer.update(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        
        assert!((normalizer.mean - 3.0).abs() < 0.01);
        assert!(normalizer.std() > 0.0);
    }
    
    #[test]
    fn test_margin_loss() -> Result<()> {
        let device = Device::Cpu;
        
        // When chosen > rejected by more than margin, loss should be 0
        let chosen = Tensor::from_vec(vec![2.0f32], 1, &device)?;
        let rejected = Tensor::from_vec(vec![0.5f32], 1, &device)?;
        
        let loss = compute_margin_loss(&chosen, &rejected, 0.5)?;
        assert!(loss.to_scalar::<f32>()? < 0.01);
        
        Ok(())
    }
}
