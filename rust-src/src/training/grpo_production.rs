//! Production GRPO (Group Relative Policy Optimization) - Rust/Candle Implementation
//!
//! This module implements production-ready GRPO with:
//! 1. PPO-style clipping for policy ratio
//! 2. Dynamic beta adjustment based on KL divergence
//! 3. Reference model update strategies
//! 4. Entropy bonus for exploration
//!
//! Reference: production_hardening.md Section 3.3 Phase 3: Post-Training (RLHF/GRPO)

use candle_core::{Device, DType, Result, Tensor};
use candle_nn as nn;
use std::collections::HashMap;

// =============================================================================
// Configuration Types
// =============================================================================

/// Strategy for updating the reference model
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceUpdateStrategy {
    /// Hard copy every N steps
    HardCopy,
    /// Polyak averaging (soft update)
    SoftUpdate,
    /// Never update reference
    NoUpdate,
}

/// KL penalty beta scheduling strategy
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BetaSchedule {
    /// Constant beta throughout training
    Constant,
    /// Adaptive beta based on KL divergence
    Adaptive,
    /// Linear decay schedule
    LinearDecay,
}

/// Configuration for production GRPO training
#[derive(Clone, Debug)]
pub struct GRPOConfig {
    /// KL penalty coefficient
    pub beta: f64,
    /// PPO clipping ratio
    pub clip_ratio: f64,
    /// Whether to clip value loss
    pub clip_value_loss: bool,
    /// Value clip range
    pub value_clip_range: f64,
    
    // Dynamic beta adjustment
    /// Beta scheduling strategy
    pub beta_schedule: BetaSchedule,
    /// Target KL for adaptive scheduling
    pub target_kl: f64,
    /// Minimum beta value
    pub beta_min: f64,
    /// Maximum beta value
    pub beta_max: f64,
    /// Beta adjustment factor
    pub beta_adjustment_factor: f64,
    
    // Reference model update
    /// Reference update strategy
    pub ref_update_strategy: ReferenceUpdateStrategy,
    /// Steps between hard copy updates
    pub ref_update_interval: usize,
    /// Polyak averaging coefficient
    pub soft_update_tau: f64,
    
    // Training parameters
    /// Group size for GRPO
    pub group_size: usize,
    /// Whether to normalize advantages
    pub advantage_normalization: bool,
    /// Entropy coefficient
    pub entropy_coef: f64,
    /// Max gradient norm for clipping
    pub max_grad_norm: f64,
    
    // Logging
    /// Log interval
    pub log_interval: usize,
}

impl Default for GRPOConfig {
    fn default() -> Self {
        Self {
            beta: 0.01,
            clip_ratio: 0.2,
            clip_value_loss: true,
            value_clip_range: 0.2,
            
            beta_schedule: BetaSchedule::Adaptive,
            target_kl: 0.01,
            beta_min: 0.001,
            beta_max: 0.1,
            beta_adjustment_factor: 1.5,
            
            ref_update_strategy: ReferenceUpdateStrategy::SoftUpdate,
            ref_update_interval: 100,
            soft_update_tau: 0.001,
            
            group_size: 4,
            advantage_normalization: true,
            entropy_coef: 0.01,
            max_grad_norm: 1.0,
            
            log_interval: 10,
        }
    }
}

/// Tracks GRPO training state
#[derive(Clone, Debug)]
pub struct GRPOState {
    /// Current training step
    pub step: usize,
    /// Current beta value
    pub current_beta: f64,
    /// Mean KL divergence
    pub mean_kl: f64,
    /// Mean entropy
    pub mean_entropy: f64,
    /// Mean probability ratio
    pub mean_ratio: f64,
    /// Fraction of clipped samples
    pub clip_fraction: f64,
    /// Mean advantage
    pub mean_advantage: f64,
    /// Total updates performed
    pub total_updates: usize,
    /// Step of last reference update
    pub last_ref_update_step: usize,
    
    // History for logging
    /// KL divergence history
    pub kl_history: Vec<f64>,
    /// Loss history
    pub loss_history: Vec<f64>,
    /// Reward history
    pub reward_history: Vec<f64>,
}

impl GRPOState {
    pub fn new(initial_beta: f64) -> Self {
        Self {
            step: 0,
            current_beta: initial_beta,
            mean_kl: 0.0,
            mean_entropy: 0.0,
            mean_ratio: 1.0,
            clip_fraction: 0.0,
            mean_advantage: 0.0,
            total_updates: 0,
            last_ref_update_step: 0,
            kl_history: Vec::new(),
            loss_history: Vec::new(),
            reward_history: Vec::new(),
        }
    }
}

/// Batch of rollout data for GRPO training
#[derive(Clone)]
pub struct RolloutBatch {
    /// Input token IDs (G, Seq)
    pub input_ids: Tensor,
    /// Attention mask (G, Seq)
    pub attention_mask: Option<Tensor>,
    /// Log probabilities from behavior policy
    pub behavior_log_probs: Option<Tensor>,
    /// Rewards for each sequence (G,)
    pub rewards: Tensor,
    /// Pre-computed advantages (G,)
    pub advantages: Option<Tensor>,
    /// Generation time in seconds
    pub generation_time: f64,
}

// =============================================================================
// Production GRPO Trainer
// =============================================================================

/// Production-ready GRPO trainer with PPO-style clipping
pub struct ProductionGRPOTrainer {
    config: GRPOConfig,
    state: GRPOState,
    device: Device,
}

impl ProductionGRPOTrainer {
    /// Create a new production GRPO trainer
    pub fn new(config: GRPOConfig, device: Device) -> Self {
        let initial_beta = config.beta;
        Self {
            config,
            state: GRPOState::new(initial_beta),
            device,
        }
    }

    /// Compute normalized advantages from rewards
    pub fn compute_advantages(&self, rewards: &Tensor) -> Result<Tensor> {
        let g = rewards.dims()[0] as f64;
        
        // Convert to f64 for computation, then convert back
        let rewards_f64 = rewards.to_dtype(DType::F64)?;
        
        // Compute mean
        let mean_r = rewards_f64.sum_all()?.to_scalar::<f64>()? / g;
        let mean_tensor = Tensor::new(&[mean_r], &self.device)?
            .broadcast_as(rewards.shape())?;
        
        if self.config.advantage_normalization {
            // Compute std
            let diff = rewards_f64.sub(&mean_tensor)?;
            let var = diff.sqr()?.sum_all()?.to_scalar::<f64>()? / g;
            let std = (var.sqrt() + 1e-8).max(1e-8);
            let std_tensor = Tensor::new(&[std], &self.device)?
                .broadcast_as(rewards.shape())?;
            
            diff.div(&std_tensor)?.to_dtype(rewards.dtype())
        } else {
            rewards_f64.sub(&mean_tensor)?.to_dtype(rewards.dtype())
        }
    }

    /// Compute log probabilities of tokens under the policy
    pub fn compute_log_probs(
        &self,
        logits: &Tensor,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Log softmax over vocabulary
        let log_probs = nn::ops::log_softmax(logits, 2)?;
        
        // Gather log probs for actual tokens
        let token_log_probs = log_probs
            .gather(&input_ids.unsqueeze(2)?, 2)?
            .squeeze(2)?;
        
        // Apply attention mask if provided
        if let Some(mask) = attention_mask {
            token_log_probs.mul(mask)
        } else {
            Ok(token_log_probs)
        }
    }

    /// Compute KL divergence between policy and reference
    pub fn compute_kl_divergence(
        &self,
        policy_logits: &Tensor,
        ref_logits: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let seq = policy_logits.dims()[1] as f64;
        
        let policy_log_probs = nn::ops::log_softmax(policy_logits, 2)?;
        let ref_log_probs = nn::ops::log_softmax(ref_logits, 2)?;
        
        // KL per token: sum(p * (log p - log q))
        let policy_probs = policy_log_probs.exp()?;
        let kl = policy_probs
            .mul(&policy_log_probs.sub(&ref_log_probs)?)?
            .sum(2)?;
        
        // Apply mask and average
        if let Some(mask) = attention_mask {
            let masked_kl = kl.mul(mask)?;
            let seq_len = mask.sum(1)?;
            let eps = Tensor::new(&[1e-8], &self.device)?.broadcast_as(seq_len.shape())?;
            masked_kl.sum(1)?.div(&seq_len.add(&eps)?)
        } else {
            Ok((kl.sum(1)? / seq)?)
        }
    }

    /// Compute entropy of the policy distribution
    pub fn compute_entropy(
        &self,
        logits: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let seq = logits.dims()[1] as f64;
        
        let log_probs = nn::ops::log_softmax(logits, 2)?;
        let probs = log_probs.exp()?;
        
        // Entropy: -sum(p * log p)
        let entropy = probs.mul(&log_probs)?.neg()?.sum(2)?;
        
        // Apply mask and average
        if let Some(mask) = attention_mask {
            let masked_entropy = entropy.mul(mask)?;
            let seq_len = mask.sum(1)?;
            let eps = Tensor::new(&[1e-8], &self.device)?.broadcast_as(seq_len.shape())?;
            masked_entropy.sum(1)?.div(&seq_len.add(&eps)?)
        } else {
            Ok((entropy.sum(1)? / seq)?)
        }
    }

    /// Compute PPO-style clipped policy loss
    pub fn compute_ppo_loss(
        &self,
        policy_log_probs: &Tensor,
        behavior_log_probs: &Tensor,
        advantages: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, f64, f64)> {
        let g = advantages.dims()[0] as f64;
        
        // Compute log ratio
        let log_ratio = policy_log_probs.sub(behavior_log_probs)?;
        
        // Average over sequence
        let seq_log_ratio = if let Some(mask) = attention_mask {
            let masked = log_ratio.mul(mask)?;
            let seq_len = mask.sum(1)?;
            let eps = Tensor::new(&[1e-8], &self.device)?.broadcast_as(seq_len.shape())?;
            masked.sum(1)?.div(&seq_len.add(&eps)?)?
        } else {
            log_ratio.mean(1)?
        };
        
        // Compute ratio
        let ratio = seq_log_ratio.exp()?;
        
        // Clip ratio
        let eps = self.config.clip_ratio;
        let clipped_ratio = ratio.clamp(1.0 - eps, 1.0 + eps)?;
        
        // Surrogate objectives
        let surrogate1 = ratio.mul(advantages)?;
        let surrogate2 = clipped_ratio.mul(advantages)?;
        
        // Take minimum (pessimistic bound)
        let policy_loss = surrogate1.minimum(&surrogate2)?.neg()?.mean_all()?;
        
        // Compute clip fraction: count where |ratio - 1| > eps
        let ones = Tensor::ones_like(&ratio)?;
        let ratio_diff = ratio.sub(&ones)?.abs()?;
        let eps_tensor = Tensor::new(&[eps], &self.device)?
            .to_dtype(ratio_diff.dtype())?
            .broadcast_as(ratio_diff.shape())?;
        let clipped = ratio_diff.gt(&eps_tensor)?;
        let clip_fraction = clipped.to_dtype(DType::F64)?.sum_all()?.to_scalar::<f64>()? / g;
        
        // Mean ratio
        let mean_ratio = ratio.mean_all()?.to_scalar::<f64>()?;
        
        Ok((policy_loss, mean_ratio, clip_fraction))
    }

    /// Update beta based on KL divergence
    pub fn update_beta(&mut self, mean_kl: f64) {
        if self.config.beta_schedule != BetaSchedule::Adaptive {
            return;
        }
        
        let target = self.config.target_kl;
        let factor = self.config.beta_adjustment_factor;
        
        let new_beta = if mean_kl > target * 1.5 {
            self.state.current_beta * factor
        } else if mean_kl < target * 0.5 {
            self.state.current_beta / factor
        } else {
            return;
        };
        
        self.state.current_beta = new_beta
            .max(self.config.beta_min)
            .min(self.config.beta_max);
    }

    /// Compute full GRPO loss with PPO clipping
    pub fn compute_grpo_loss(
        &self,
        rollouts: &RolloutBatch,
        policy_logits: &Tensor,
        ref_logits: &Tensor,
    ) -> Result<HashMap<String, f64>> {
        let g = rollouts.rewards.dims()[0] as f64;
        
        // Compute advantages
        let advantages = if let Some(ref adv) = rollouts.advantages {
            adv.clone()
        } else {
            self.compute_advantages(&rollouts.rewards)?
        };
        
        // Compute policy log probs
        let policy_log_probs = self.compute_log_probs(
            policy_logits,
            &rollouts.input_ids,
            rollouts.attention_mask.as_ref(),
        )?;
        
        // Compute KL divergence
        let kl = self.compute_kl_divergence(
            policy_logits,
            ref_logits,
            rollouts.attention_mask.as_ref(),
        )?;
        let mean_kl = kl.mean_all()?.to_scalar::<f64>()?;
        
        // Compute entropy bonus
        let entropy = self.compute_entropy(
            policy_logits,
            rollouts.attention_mask.as_ref(),
        )?;
        let mean_entropy = entropy.mean_all()?.to_scalar::<f64>()?;
        
        // Compute policy loss
        let (policy_loss, mean_ratio, clip_fraction) = if let Some(ref behavior_lp) = rollouts.behavior_log_probs {
            self.compute_ppo_loss(
                &policy_log_probs,
                behavior_lp,
                &advantages,
                rollouts.attention_mask.as_ref(),
            )?
        } else {
            // Fallback to REINFORCE-style loss
            let seq_log_probs = if let Some(ref mask) = rollouts.attention_mask {
                policy_log_probs.mul(mask)?.sum(1)?
            } else {
                policy_log_probs.sum(1)?
            };
            let loss = advantages.mul(&seq_log_probs)?.neg()?.mean_all()?;
            (loss, 1.0, 0.0)
        };
        
        // Get scalar policy loss
        let policy_loss_val = policy_loss.to_scalar::<f64>()?;
        
        // KL penalty
        let kl_loss = self.state.current_beta * mean_kl;
        
        // Entropy bonus
        let entropy_loss = -self.config.entropy_coef * mean_entropy;
        
        // Total loss
        let total_loss = policy_loss_val + kl_loss + entropy_loss;
        
        // Mean advantage
        let mean_advantage = advantages.mean_all()?.to_scalar::<f64>()?;
        
        // Mean reward
        let mean_reward = rollouts.rewards.sum_all()?.to_scalar::<f64>()? / g;
        
        let mut metrics = HashMap::new();
        metrics.insert("total_loss".to_string(), total_loss);
        metrics.insert("policy_loss".to_string(), policy_loss_val);
        metrics.insert("kl_loss".to_string(), kl_loss);
        metrics.insert("entropy_loss".to_string(), entropy_loss);
        metrics.insert("mean_kl".to_string(), mean_kl);
        metrics.insert("mean_entropy".to_string(), mean_entropy);
        metrics.insert("mean_ratio".to_string(), mean_ratio);
        metrics.insert("clip_fraction".to_string(), clip_fraction);
        metrics.insert("mean_advantage".to_string(), mean_advantage);
        metrics.insert("mean_reward".to_string(), mean_reward);
        metrics.insert("beta".to_string(), self.state.current_beta);
        metrics.insert("step".to_string(), self.state.step as f64);
        
        Ok(metrics)
    }

    /// Update trainer state after a training step
    pub fn update_state(&mut self, metrics: &HashMap<String, f64>) {
        self.state.step += 1;
        self.state.total_updates += 1;
        
        if let Some(&kl) = metrics.get("mean_kl") {
            self.state.mean_kl = kl;
            self.state.kl_history.push(kl);
        }
        if let Some(&entropy) = metrics.get("mean_entropy") {
            self.state.mean_entropy = entropy;
        }
        if let Some(&ratio) = metrics.get("mean_ratio") {
            self.state.mean_ratio = ratio;
        }
        if let Some(&clip_frac) = metrics.get("clip_fraction") {
            self.state.clip_fraction = clip_frac;
        }
        if let Some(&adv) = metrics.get("mean_advantage") {
            self.state.mean_advantage = adv;
        }
        if let Some(&loss) = metrics.get("total_loss") {
            self.state.loss_history.push(loss);
        }
        if let Some(&reward) = metrics.get("mean_reward") {
            self.state.reward_history.push(reward);
        }
        
        // Update beta based on KL
        self.update_beta(self.state.mean_kl);
    }

    /// Check if reference model should be updated (for hard copy strategy)
    pub fn should_update_reference(&self) -> bool {
        match self.config.ref_update_strategy {
            ReferenceUpdateStrategy::HardCopy => {
                // Also update on first step (step 0)
                if self.state.step == 0 && self.state.last_ref_update_step == 0 {
                    return true;
                }
                let steps_since = self.state.step - self.state.last_ref_update_step;
                steps_since >= self.config.ref_update_interval
            }
            ReferenceUpdateStrategy::SoftUpdate => true,
            ReferenceUpdateStrategy::NoUpdate => false,
        }
    }

    /// Mark reference model as updated
    pub fn mark_reference_updated(&mut self) {
        self.state.last_ref_update_step = self.state.step;
    }

    /// Get current configuration
    pub fn config(&self) -> &GRPOConfig {
        &self.config
    }

    /// Get current state
    pub fn state(&self) -> &GRPOState {
        &self.state
    }

    /// Get soft update tau for Polyak averaging
    pub fn soft_update_tau(&self) -> f64 {
        self.config.soft_update_tau
    }
}

// =============================================================================
// Soft Update Utility
// =============================================================================

/// Perform Polyak averaging (soft update) between two sets of parameters
/// ref_params = (1 - tau) * ref_params + tau * policy_params
pub fn soft_update_params(
    ref_params: &mut Tensor,
    policy_params: &Tensor,
    tau: f64,
) -> Result<()> {
    let one_minus_tau = 1.0 - tau;
    let updated = (ref_params.clone() * one_minus_tau)?
        .add(&(policy_params.clone() * tau)?)?;
    *ref_params = updated;
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_device() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_config_default() {
        let config = GRPOConfig::default();
        assert!((config.beta - 0.01).abs() < 1e-6);
        assert!((config.clip_ratio - 0.2).abs() < 1e-6);
        assert_eq!(config.beta_schedule, BetaSchedule::Adaptive);
        assert_eq!(config.ref_update_strategy, ReferenceUpdateStrategy::SoftUpdate);
    }

    #[test]
    fn test_state_initialization() {
        let state = GRPOState::new(0.01);
        assert_eq!(state.step, 0);
        assert!((state.current_beta - 0.01).abs() < 1e-6);
        assert_eq!(state.total_updates, 0);
    }

    #[test]
    fn test_trainer_creation() {
        let device = create_test_device();
        let config = GRPOConfig::default();
        let trainer = ProductionGRPOTrainer::new(config.clone(), device);
        
        assert!((trainer.state().current_beta - config.beta).abs() < 1e-6);
        assert_eq!(trainer.state().step, 0);
    }

    #[test]
    fn test_compute_advantages() -> Result<()> {
        let device = create_test_device();
        let config = GRPOConfig::default();
        let trainer = ProductionGRPOTrainer::new(config, device.clone());
        
        let rewards = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
        let advantages = trainer.compute_advantages(&rewards)?;
        
        // Check shape
        assert_eq!(advantages.dims(), rewards.dims());
        
        // Check normalization (should have ~0 mean and ~1 std)
        let mean = advantages.mean_all()?.to_scalar::<f32>()?;
        assert!(mean.abs() < 0.1);
        
        Ok(())
    }

    #[test]
    fn test_update_beta_increase() {
        let device = create_test_device();
        let config = GRPOConfig {
            beta_schedule: BetaSchedule::Adaptive,
            target_kl: 0.01,
            beta_adjustment_factor: 2.0,
            ..GRPOConfig::default()
        };
        
        let mut trainer = ProductionGRPOTrainer::new(config, device);
        let initial_beta = trainer.state().current_beta;
        
        // KL much higher than target should increase beta
        trainer.update_beta(0.02);
        assert!(trainer.state().current_beta > initial_beta);
    }

    #[test]
    fn test_update_beta_decrease() {
        let device = create_test_device();
        let config = GRPOConfig {
            beta_schedule: BetaSchedule::Adaptive,
            target_kl: 0.01,
            beta_adjustment_factor: 2.0,
            ..GRPOConfig::default()
        };
        
        let mut trainer = ProductionGRPOTrainer::new(config, device);
        let initial_beta = trainer.state().current_beta;
        
        // KL much lower than target should decrease beta
        trainer.update_beta(0.001);
        assert!(trainer.state().current_beta < initial_beta);
    }

    #[test]
    fn test_update_beta_constant_schedule() {
        let device = create_test_device();
        let config = GRPOConfig {
            beta_schedule: BetaSchedule::Constant,
            ..GRPOConfig::default()
        };
        
        let mut trainer = ProductionGRPOTrainer::new(config, device);
        let initial_beta = trainer.state().current_beta;
        
        // Should not change with constant schedule
        trainer.update_beta(0.02);
        assert!((trainer.state().current_beta - initial_beta).abs() < 1e-8);
    }

    #[test]
    fn test_should_update_reference() {
        let device = create_test_device();
        
        // Hard copy strategy
        let config = GRPOConfig {
            ref_update_strategy: ReferenceUpdateStrategy::HardCopy,
            ref_update_interval: 10,
            ..GRPOConfig::default()
        };
        let trainer = ProductionGRPOTrainer::new(config, device.clone());
        assert!(trainer.should_update_reference()); // Step 0, should update
        
        // No update strategy
        let config2 = GRPOConfig {
            ref_update_strategy: ReferenceUpdateStrategy::NoUpdate,
            ..GRPOConfig::default()
        };
        let trainer2 = ProductionGRPOTrainer::new(config2, device.clone());
        assert!(!trainer2.should_update_reference());
        
        // Soft update always returns true
        let config3 = GRPOConfig {
            ref_update_strategy: ReferenceUpdateStrategy::SoftUpdate,
            ..GRPOConfig::default()
        };
        let trainer3 = ProductionGRPOTrainer::new(config3, device);
        assert!(trainer3.should_update_reference());
    }

    #[test]
    fn test_soft_update_params() -> Result<()> {
        let device = create_test_device();
        let mut ref_params = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
        let policy_params = Tensor::new(&[4.0f32, 5.0, 6.0], &device)?;
        
        soft_update_params(&mut ref_params, &policy_params, 0.1)?;
        
        // ref = 0.9 * [1,2,3] + 0.1 * [4,5,6] = [0.9+0.4, 1.8+0.5, 2.7+0.6] = [1.3, 2.3, 3.3]
        let expected = vec![1.3f32, 2.3, 3.3];
        let result: Vec<f32> = ref_params.to_vec1()?;
        
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5, "Expected {} but got {}", e, r);
        }
        
        Ok(())
    }
}
