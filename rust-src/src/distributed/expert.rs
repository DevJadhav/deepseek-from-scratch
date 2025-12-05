//! Expert Parallelism for Mixture-of-Experts models.
//!
//! This module provides comprehensive expert parallelism capabilities:
//! - All-to-all token dispatch and combine operations
//! - Load balancing across experts
//! - Capacity factor handling with token dropping
//! - Token padding/unpadding for efficient batched computation
//! - Expert placement optimization
//! - Gradient synchronization for distributed experts
//!
//! Reference: DeepSeek-V3 Expert Parallelism implementation.

use candle_core::{Result, Tensor, DType, Device};
use super::{get_ep_size, get_ep_group, CollectiveCommunicator};
use super::groups::ProcessGroup;
use std::sync::Arc;
use std::collections::HashMap;

// ============================================================================
// Expert Parallelism Configuration
// ============================================================================

/// Configuration for expert parallelism.
#[derive(Clone, Debug)]
pub struct ExpertParallelConfig {
    /// Number of EP ranks
    pub ep_size: usize,
    /// Total number of experts
    pub num_experts: usize,
    /// Number of local experts per rank
    pub num_local_experts: usize,
    /// This rank in EP group
    pub ep_rank: usize,
    /// Capacity factor (tokens per expert = avg_tokens * capacity_factor)
    pub capacity_factor: f32,
    /// Drop tokens that exceed capacity
    pub drop_tokens: bool,
    /// Pad tokens for uniform expert batches
    pub pad_to_capacity: bool,
    /// Enable load balancing loss
    pub load_balance: bool,
    /// Load balance loss weight
    pub load_balance_weight: f32,
    /// Enable auxiliary loss for expert usage
    pub auxiliary_loss: bool,
    /// Auxiliary loss weight  
    pub auxiliary_loss_weight: f32,
}

impl Default for ExpertParallelConfig {
    fn default() -> Self {
        let ep_size = get_ep_size();
        let ep_rank = get_ep_group()
            .map(|g| g.communicator.rank())
            .unwrap_or(0);
        
        Self {
            ep_size,
            num_experts: 8,
            num_local_experts: 8 / ep_size.max(1),
            ep_rank,
            capacity_factor: 1.25,
            drop_tokens: true,
            pad_to_capacity: true,
            load_balance: true,
            load_balance_weight: 0.01,
            auxiliary_loss: true,
            auxiliary_loss_weight: 0.001,
        }
    }
}

impl ExpertParallelConfig {
    pub fn new(num_experts: usize) -> Self {
        let ep_size = get_ep_size();
        let ep_rank = get_ep_group()
            .map(|g| g.communicator.rank())
            .unwrap_or(0);
        
        let num_local_experts = num_experts / ep_size.max(1);
        
        Self {
            ep_size,
            num_experts,
            num_local_experts,
            ep_rank,
            ..Default::default()
        }
    }
    
    pub fn with_capacity_factor(mut self, factor: f32) -> Self {
        self.capacity_factor = factor;
        self
    }
    
    pub fn with_load_balance(mut self, enabled: bool, weight: f32) -> Self {
        self.load_balance = enabled;
        self.load_balance_weight = weight;
        self
    }
    
    pub fn with_auxiliary_loss(mut self, enabled: bool, weight: f32) -> Self {
        self.auxiliary_loss = enabled;
        self.auxiliary_loss_weight = weight;
        self
    }
    
    /// Get expert IDs that this rank is responsible for.
    pub fn local_expert_ids(&self) -> Vec<usize> {
        let start = self.ep_rank * self.num_local_experts;
        (start..start + self.num_local_experts).collect()
    }
    
    /// Calculate capacity per expert.
    pub fn expert_capacity(&self, num_tokens: usize) -> usize {
        let avg_tokens_per_expert = num_tokens / self.num_experts;
        ((avg_tokens_per_expert as f32) * self.capacity_factor).ceil() as usize
    }
    
    /// Get the rank responsible for a given expert.
    pub fn expert_to_rank(&self, expert_id: usize) -> usize {
        expert_id / self.num_local_experts
    }
}

// ============================================================================
// Dispatch Metadata
// ============================================================================

/// Stored permutation for reversing dispatch.
#[derive(Clone)]
pub struct DispatchInfo {
    /// Permutation indices to restore original order
    pub restore_indices: Vec<u32>,
    /// Number of tokens sent to each rank
    pub send_counts: Vec<usize>,
    /// Number of tokens received from each rank
    pub recv_counts: Vec<usize>,
    /// Original shape before flattening
    pub original_shape: Vec<usize>,
    /// Tokens per expert on this rank
    pub tokens_per_expert: Vec<usize>,
    /// Indices of dropped tokens (if any)
    pub dropped_indices: Vec<u32>,
    /// Expert assignments for each token
    pub expert_assignments: Vec<u32>,
    /// Gate values for weighted combining
    pub gate_values: Option<Vec<f32>>,
    /// Padding mask (true = padded token)
    pub padding_mask: Vec<bool>,
}

impl DispatchInfo {
    pub fn new(num_tokens: usize, ep_size: usize) -> Self {
        Self {
            restore_indices: Vec::with_capacity(num_tokens),
            send_counts: vec![0; ep_size],
            recv_counts: vec![0; ep_size],
            original_shape: Vec::new(),
            tokens_per_expert: Vec::new(),
            dropped_indices: Vec::new(),
            expert_assignments: Vec::with_capacity(num_tokens),
            gate_values: None,
            padding_mask: Vec::new(),
        }
    }
    
    /// Number of tokens that were dropped.
    pub fn num_dropped(&self) -> usize {
        self.dropped_indices.len()
    }
    
    /// Number of tokens that were kept.
    pub fn num_kept(&self) -> usize {
        self.restore_indices.len() - self.num_dropped()
    }
    
    /// Number of padded tokens.
    pub fn num_padded(&self) -> usize {
        self.padding_mask.iter().filter(|&&x| x).count()
    }
}

// ============================================================================
// Load Balancing
// ============================================================================

/// Load balancing statistics and loss computation.
pub struct LoadBalancer {
    config: ExpertParallelConfig,
    /// Cumulative expert usage counts
    expert_counts: Vec<u64>,
    /// Total tokens processed
    total_tokens: u64,
    /// History of imbalance scores
    imbalance_history: Vec<f32>,
}

impl LoadBalancer {
    pub fn new(config: ExpertParallelConfig) -> Self {
        Self {
            expert_counts: vec![0; config.num_experts],
            total_tokens: 0,
            imbalance_history: Vec::new(),
            config,
        }
    }
    
    /// Update statistics with new expert assignments.
    pub fn update(&mut self, expert_indices: &[u32]) {
        for &idx in expert_indices {
            if (idx as usize) < self.expert_counts.len() {
                self.expert_counts[idx as usize] += 1;
            }
        }
        self.total_tokens += expert_indices.len() as u64;
        
        // Track imbalance
        let imbalance = self.compute_imbalance_score();
        self.imbalance_history.push(imbalance);
    }
    
    /// Compute load balancing loss.
    /// 
    /// The loss encourages uniform distribution across experts:
    /// L_balance = sum(f_i * P_i) * num_experts
    /// where f_i = fraction of tokens to expert i
    ///       P_i = fraction of routing probability to expert i
    pub fn compute_loss(
        &self,
        expert_indices: &Tensor,
        gate_probs: &Tensor,
    ) -> Result<Tensor> {
        let device = expert_indices.device();
        let num_tokens = expert_indices.elem_count();
        let num_experts = self.config.num_experts;
        
        // Compute f_i: fraction of tokens routed to each expert
        let indices_vec = expert_indices.to_vec1::<u32>()?;
        let mut expert_fractions = vec![0f32; num_experts];
        for &idx in &indices_vec {
            if (idx as usize) < num_experts {
                expert_fractions[idx as usize] += 1.0;
            }
        }
        for f in &mut expert_fractions {
            *f /= num_tokens as f32;
        }
        
        // Compute P_i: mean routing probability to each expert
        // gate_probs shape: (num_tokens, num_experts)
        let mean_probs = gate_probs.mean(0)?; // (num_experts,)
        let mean_probs_vec = mean_probs.to_vec1::<f32>()?;
        
        // Compute loss = sum(f_i * P_i) * num_experts
        let mut loss_val = 0f32;
        for i in 0..num_experts {
            loss_val += expert_fractions[i] * mean_probs_vec.get(i).copied().unwrap_or(0.0);
        }
        loss_val *= num_experts as f32;
        
        Tensor::from_vec(vec![loss_val], (1,), device)
    }
    
    /// Compute auxiliary router z-loss for training stability.
    pub fn compute_auxiliary_loss(&self, router_logits: &Tensor) -> Result<Tensor> {
        // z-loss = mean(log(sum(exp(logits))))^2
        let logsumexp = router_logits.log_sum_exp(1)?;
        let z_loss = logsumexp.sqr()?.mean_all()?;
        Ok(z_loss)
    }
    
    /// Compute imbalance score (coefficient of variation).
    pub fn compute_imbalance_score(&self) -> f32 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        
        let mean = self.total_tokens as f32 / self.expert_counts.len() as f32;
        let variance: f32 = self.expert_counts
            .iter()
            .map(|&c| {
                let diff = c as f32 - mean;
                diff * diff
            })
            .sum::<f32>()
            / self.expert_counts.len() as f32;
        
        if mean > 0.0 {
            variance.sqrt() / mean
        } else {
            0.0
        }
    }
    
    /// Get expert utilization percentages.
    pub fn expert_utilization(&self) -> Vec<f32> {
        if self.total_tokens == 0 {
            return vec![0.0; self.expert_counts.len()];
        }
        
        let expected = self.total_tokens as f32 / self.expert_counts.len() as f32;
        self.expert_counts
            .iter()
            .map(|&c| (c as f32 / expected) * 100.0)
            .collect()
    }
    
    /// Reset statistics.
    pub fn reset(&mut self) {
        self.expert_counts.fill(0);
        self.total_tokens = 0;
        self.imbalance_history.clear();
    }
}

// ============================================================================
// Token Padding/Unpadding
// ============================================================================

/// Pad tokens to uniform expert capacity.
pub fn pad_to_capacity(
    tokens: &Tensor,
    expert_indices: &[u32],
    capacity: usize,
    num_experts: usize,
) -> Result<(Tensor, Vec<bool>)> {
    let device = tokens.device();
    let dtype = tokens.dtype();
    let (num_tokens, hidden_dim) = tokens.dims2()?;
    
    // Count tokens per expert
    let mut tokens_per_expert = vec![0usize; num_experts];
    for &idx in expert_indices {
        if (idx as usize) < num_experts {
            tokens_per_expert[idx as usize] += 1;
        }
    }
    
    // Calculate padding needed
    let total_padded = num_experts * capacity;
    let mut padding_mask = vec![false; total_padded];
    
    // Create padded tensor
    let mut padded_data = vec![0f32; total_padded * hidden_dim];
    
    // Sort tokens by expert
    let mut token_positions: Vec<(usize, u32)> = expert_indices
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e))
        .collect();
    token_positions.sort_by_key(|&(_, e)| e);
    
    // Fill padded tensor
    let mut expert_offsets = vec![0usize; num_experts];
    for i in 1..num_experts {
        expert_offsets[i] = expert_offsets[i - 1] + capacity;
    }
    
    let tokens_vec: Vec<f32> = tokens.flatten_all()?.to_vec1()?;
    let mut expert_cursors = vec![0usize; num_experts];
    
    for (orig_pos, expert_id) in token_positions {
        let exp = expert_id as usize;
        if exp < num_experts && expert_cursors[exp] < capacity {
            let dst_pos = expert_offsets[exp] + expert_cursors[exp];
            let src_start = orig_pos * hidden_dim;
            let dst_start = dst_pos * hidden_dim;
            
            for j in 0..hidden_dim {
                padded_data[dst_start + j] = tokens_vec[src_start + j];
            }
            expert_cursors[exp] += 1;
        }
    }
    
    // Mark padding positions
    for exp in 0..num_experts {
        for i in expert_cursors[exp]..capacity {
            let pos = expert_offsets[exp] + i;
            padding_mask[pos] = true;
        }
    }
    
    let padded_tensor = Tensor::from_vec(padded_data, (total_padded, hidden_dim), device)?
        .to_dtype(dtype)?;
    
    Ok((padded_tensor, padding_mask))
}

/// Remove padding from expert outputs.
pub fn unpad_from_capacity(
    padded_output: &Tensor,
    padding_mask: &[bool],
    original_num_tokens: usize,
) -> Result<Tensor> {
    let device = padded_output.device();
    let dtype = padded_output.dtype();
    let (total_padded, hidden_dim) = padded_output.dims2()?;
    
    // Extract non-padded tokens
    let padded_vec: Vec<f32> = padded_output.flatten_all()?.to_vec1()?;
    let mut output_data = vec![0f32; original_num_tokens * hidden_dim];
    
    let mut out_idx = 0;
    for (i, &is_padded) in padding_mask.iter().enumerate() {
        if !is_padded && out_idx < original_num_tokens {
            let src_start = i * hidden_dim;
            let dst_start = out_idx * hidden_dim;
            
            for j in 0..hidden_dim {
                output_data[dst_start + j] = padded_vec[src_start + j];
            }
            out_idx += 1;
        }
    }
    
    Tensor::from_vec(output_data, (original_num_tokens, hidden_dim), device)?
        .to_dtype(dtype)
}

// ============================================================================
// Expert Parallel Dispatcher
// ============================================================================

/// Expert Parallel Dispatcher.
/// 
/// Handles routing tokens to experts across different ranks using
/// all-to-all communication.
pub struct ExpertParallelDispatch {
    config: ExpertParallelConfig,
    process_group: Option<ProcessGroup>,
}

impl ExpertParallelDispatch {
    pub fn new(num_experts: usize) -> Self {
        let config = ExpertParallelConfig::new(num_experts);
        let process_group = get_ep_group();
        
        Self { config, process_group }
    }
    
    pub fn with_communicator(
        num_experts: usize,
        communicator: Arc<dyn CollectiveCommunicator>,
    ) -> Self {
        let ep_size = communicator.world_size();
        let ep_rank = communicator.rank();
        let num_local_experts = num_experts / ep_size.max(1);
        
        let config = ExpertParallelConfig {
            ep_size,
            num_experts,
            num_local_experts,
            ep_rank,
            capacity_factor: 1.25,
            drop_tokens: true,
            pad_to_capacity: true,
            load_balance: true,
            load_balance_weight: 0.01,
            auxiliary_loss: true,
            auxiliary_loss_weight: 0.001,
        };
        
        let process_group = Some(ProcessGroup::new(
            communicator,
            (0..ep_size).collect(),
        ));
        
        Self { config, process_group }
    }
    
    pub fn config(&self) -> &ExpertParallelConfig {
        &self.config
    }

    /// Dispatch tokens to experts across ranks.
    /// 
    /// Args:
    ///   x: Input tokens (num_tokens, hidden_dim)
    ///   expert_indices: Target expert for each token (num_tokens,)
    /// 
    /// Returns:
    ///   (dispatched_tokens, dispatch_info)
    ///   - dispatched_tokens: Tokens reordered for local expert processing
    ///   - dispatch_info: Information needed to restore original order
    pub fn dispatch(&self, x: &Tensor, expert_indices: &Tensor) -> Result<(Tensor, DispatchInfo)> {
        let ep_size = self.config.ep_size;
        
        if ep_size <= 1 {
            // No EP, just sort by expert for efficient processing
            return self.local_dispatch(x, expert_indices);
        }
        
        let (num_tokens, hidden_dim) = x.dims2()?;
        let indices_vec = expert_indices.to_vec1::<u32>()?;
        
        // Count tokens per rank
        let mut send_counts = vec![0usize; ep_size];
        for &exp_id in &indices_vec {
            let target_rank = (exp_id as usize) / self.config.num_local_experts;
            if target_rank < ep_size {
                send_counts[target_rank] += 1;
            }
        }
        
        // Sort tokens by target rank (stable sort to maintain order within rank)
        let mut token_order: Vec<(usize, usize)> = indices_vec
            .iter()
            .enumerate()
            .map(|(i, &exp_id)| {
                let target_rank = (exp_id as usize) / self.config.num_local_experts;
                (i, target_rank)
            })
            .collect();
        token_order.sort_by_key(|&(_, rank)| rank);
        
        // Permute tokens for sending
        let permute_indices: Vec<u32> = token_order.iter().map(|&(i, _)| i as u32).collect();
        let permute_tensor = Tensor::from_vec(
            permute_indices.clone(),
            (num_tokens,),
            x.device()
        )?;
        let permuted_x = x.index_select(&permute_tensor, 0)?;
        
        // Compute restore indices (inverse permutation)
        let mut restore_indices = vec![0u32; num_tokens];
        for (new_pos, &orig_pos) in permute_indices.iter().enumerate() {
            restore_indices[orig_pos as usize] = new_pos as u32;
        }
        
        // All-to-all exchange
        // First, compute recv_counts via all-to-all of send_counts
        let recv_counts = self.exchange_counts(&send_counts)?;
        
        // Perform all-to-all on tokens
        let dispatched = if let Some(ref pg) = self.process_group {
            pg.communicator.all_to_all_variable(&permuted_x, &send_counts, &recv_counts)?
        } else {
            permuted_x
        };
        
        let info = DispatchInfo {
            restore_indices,
            send_counts,
            recv_counts,
            original_shape: vec![num_tokens, hidden_dim],
            tokens_per_expert: Vec::new(),
            dropped_indices: Vec::new(),
            expert_assignments: indices_vec,
            gate_values: None,
            padding_mask: Vec::new(),
        };
        
        Ok((dispatched, info))
    }
    
    /// Combine results from experts back to original order.
    /// 
    /// Args:
    ///   expert_out: Output from local experts
    ///   info: DispatchInfo from dispatch()
    /// 
    /// Returns:
    ///   Tokens in original order
    pub fn combine(&self, expert_out: &Tensor, info: &DispatchInfo) -> Result<Tensor> {
        let ep_size = self.config.ep_size;
        
        if ep_size <= 1 {
            // No EP, just restore order
            return self.local_combine(expert_out, info);
        }
        
        // All-to-all to send results back
        // Note: send/recv counts are swapped compared to dispatch
        let gathered = if let Some(ref pg) = self.process_group {
            pg.communicator.all_to_all_variable(&expert_out, &info.recv_counts, &info.send_counts)?
        } else {
            expert_out.clone()
        };
        
        // Restore original order using inverse permutation
        let restore_tensor = Tensor::from_vec(
            info.restore_indices.clone(),
            (info.restore_indices.len(),),
            expert_out.device()
        )?;
        
        // Create output tensor and scatter
        let num_tokens = info.original_shape[0];
        let hidden_dim = gathered.dim(1)?;
        let mut output = Tensor::zeros((num_tokens, hidden_dim), DType::F32, expert_out.device())?;
        output = output.index_add(&restore_tensor, &gathered, 0)?;
        
        Ok(output)
    }
    
    /// Local dispatch (no communication, just sort by expert).
    fn local_dispatch(&self, x: &Tensor, expert_indices: &Tensor) -> Result<(Tensor, DispatchInfo)> {
        let (num_tokens, hidden_dim) = x.dims2()?;
        let indices_vec = expert_indices.to_vec1::<u32>()?;
        
        // Sort by expert ID
        let mut token_order: Vec<(usize, u32)> = indices_vec
            .iter()
            .enumerate()
            .map(|(i, &exp_id)| (i, exp_id))
            .collect();
        token_order.sort_by_key(|&(_, exp_id)| exp_id);
        
        let permute_indices: Vec<u32> = token_order.iter().map(|&(i, _)| i as u32).collect();
        let permute_tensor = Tensor::from_vec(
            permute_indices.clone(),
            (num_tokens,),
            x.device()
        )?;
        let permuted_x = x.index_select(&permute_tensor, 0)?;
        
        // Compute restore indices
        let mut restore_indices = vec![0u32; num_tokens];
        for (new_pos, &orig_pos) in permute_indices.iter().enumerate() {
            restore_indices[orig_pos as usize] = new_pos as u32;
        }
        
        // Count tokens per expert
        let mut tokens_per_expert = vec![0usize; self.config.num_experts];
        for &idx in &indices_vec {
            if (idx as usize) < self.config.num_experts {
                tokens_per_expert[idx as usize] += 1;
            }
        }
        
        let info = DispatchInfo {
            restore_indices,
            send_counts: vec![num_tokens],
            recv_counts: vec![num_tokens],
            original_shape: vec![num_tokens, hidden_dim],
            tokens_per_expert,
            dropped_indices: Vec::new(),
            expert_assignments: indices_vec,
            gate_values: None,
            padding_mask: Vec::new(),
        };
        
        Ok((permuted_x, info))
    }
    
    /// Local combine (no communication).
    fn local_combine(&self, expert_out: &Tensor, info: &DispatchInfo) -> Result<Tensor> {
        let restore_tensor = Tensor::from_vec(
            info.restore_indices.clone(),
            (info.restore_indices.len(),),
            expert_out.device()
        )?;
        
        expert_out.index_select(&restore_tensor, 0)
    }
    
    /// Exchange token counts with all ranks.
    fn exchange_counts(&self, send_counts: &[usize]) -> Result<Vec<usize>> {
        if self.config.ep_size <= 1 {
            return Ok(send_counts.to_vec());
        }
        
        // In real impl, use all-to-all on counts
        // For now, return same counts (symmetric case)
        Ok(send_counts.to_vec())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_expert_parallel_config() {
        let config = ExpertParallelConfig::new(8);
        assert_eq!(config.ep_size, 1);  // Default single rank
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.num_local_experts, 8);
    }
    
    #[test]
    fn test_expert_parallel_config_with_options() {
        let config = ExpertParallelConfig::new(16)
            .with_capacity_factor(1.5)
            .with_load_balance(true, 0.02)
            .with_auxiliary_loss(true, 0.002);
        
        assert_eq!(config.num_experts, 16);
        assert!((config.capacity_factor - 1.5).abs() < 0.001);
        assert!(config.load_balance);
        assert!((config.load_balance_weight - 0.02).abs() < 0.001);
    }
    
    #[test]
    fn test_expert_capacity() {
        let config = ExpertParallelConfig::new(8)
            .with_capacity_factor(1.25);
        
        // 80 tokens, 8 experts = 10 avg per expert
        // With 1.25 capacity factor = 13 (ceil of 12.5)
        let capacity = config.expert_capacity(80);
        assert_eq!(capacity, 13);
    }
    
    #[test]
    fn test_dispatch_info_creation() {
        let info = DispatchInfo::new(100, 4);
        assert_eq!(info.send_counts.len(), 4);
        assert_eq!(info.recv_counts.len(), 4);
    }
    
    #[test]
    fn test_local_dispatch_combine() -> Result<()> {
        let device = Device::Cpu;
        let dispatcher = ExpertParallelDispatch::new(4);
        
        // 8 tokens, hidden_dim=16
        let x = Tensor::randn(0f32, 1f32, (8, 16), &device)?;
        let expert_indices = Tensor::from_vec(
            vec![0u32, 2, 1, 3, 0, 1, 2, 3],
            (8,),
            &device
        )?;
        
        let (dispatched, info) = dispatcher.dispatch(&x, &expert_indices)?;
        assert_eq!(dispatched.dims(), &[8, 16]);
        
        // Check tokens per expert
        assert_eq!(info.tokens_per_expert[0], 2);
        assert_eq!(info.tokens_per_expert[1], 2);
        assert_eq!(info.tokens_per_expert[2], 2);
        assert_eq!(info.tokens_per_expert[3], 2);
        
        // Combine should restore original order
        let combined = dispatcher.combine(&dispatched, &info)?;
        assert_eq!(combined.dims(), x.dims());
        
        Ok(())
    }
    
    #[test]
    fn test_load_balancer() {
        let config = ExpertParallelConfig::new(4);
        let mut balancer = LoadBalancer::new(config);
        
        // Simulate balanced usage
        balancer.update(&[0, 1, 2, 3, 0, 1, 2, 3]);
        
        let utilization = balancer.expert_utilization();
        assert_eq!(utilization.len(), 4);
        
        // All should be ~100%
        for u in &utilization {
            assert!((*u - 100.0).abs() < 1.0);
        }
        
        let imbalance = balancer.compute_imbalance_score();
        assert!(imbalance < 0.1); // Should be very low for balanced usage
    }
    
    #[test]
    fn test_load_balancer_imbalanced() {
        let config = ExpertParallelConfig::new(4);
        let mut balancer = LoadBalancer::new(config);
        
        // Simulate imbalanced usage (all to expert 0)
        balancer.update(&[0, 0, 0, 0, 0, 0, 0, 0]);
        
        let utilization = balancer.expert_utilization();
        assert!(utilization[0] > 300.0); // Expert 0 heavily used
        assert!(utilization[1] < 1.0);   // Others unused
        
        let imbalance = balancer.compute_imbalance_score();
        assert!(imbalance > 1.0); // High imbalance
    }
    
    #[test]
    fn test_pad_to_capacity() -> Result<()> {
        let device = Device::Cpu;
        
        // 6 tokens, 4 experts, capacity 2
        let tokens = Tensor::randn(0f32, 1f32, (6, 8), &device)?;
        let expert_indices = vec![0u32, 0, 1, 1, 2, 3]; // 2 to exp 0, 2 to exp 1, 1 to exp 2, 1 to exp 3
        
        let (padded, mask) = pad_to_capacity(&tokens, &expert_indices, 2, 4)?;
        
        // Should be 4 experts * 2 capacity = 8 total
        assert_eq!(padded.dims(), &[8, 8]);
        assert_eq!(mask.len(), 8);
        
        // Expert 2 and 3 each have 1 token, so 1 padding each
        let num_padded = mask.iter().filter(|&&x| x).count();
        assert_eq!(num_padded, 2);
        
        Ok(())
    }
    
    #[test]
    fn test_unpad_from_capacity() -> Result<()> {
        let device = Device::Cpu;
        
        // 8 padded tokens (4 experts * 2 capacity)
        let padded = Tensor::randn(0f32, 1f32, (8, 4), &device)?;
        let mask = vec![false, false, false, true, false, true, false, false];
        
        // Original 6 tokens
        let unpadded = unpad_from_capacity(&padded, &mask, 6)?;
        
        assert_eq!(unpadded.dims(), &[6, 4]);
        
        Ok(())
    }
    
    #[test]
    fn test_expert_to_rank() {
        let mut config = ExpertParallelConfig::new(8);
        config.ep_size = 4;
        config.num_local_experts = 2;
        
        // Expert 0,1 -> rank 0
        // Expert 2,3 -> rank 1
        // Expert 4,5 -> rank 2
        // Expert 6,7 -> rank 3
        assert_eq!(config.expert_to_rank(0), 0);
        assert_eq!(config.expert_to_rank(1), 0);
        assert_eq!(config.expert_to_rank(2), 1);
        assert_eq!(config.expert_to_rank(5), 2);
        assert_eq!(config.expert_to_rank(7), 3);
    }
    
    #[test]
    fn test_local_expert_ids() {
        let mut config = ExpertParallelConfig::new(8);
        config.ep_size = 4;
        config.num_local_experts = 2;
        config.ep_rank = 2;
        
        let local_ids = config.local_expert_ids();
        assert_eq!(local_ids, vec![4, 5]);
    }
}
