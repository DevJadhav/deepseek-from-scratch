//! Expert Parallelism for Mixture-of-Experts models.
//!
//! Implements all-to-all token dispatch and combine operations
//! for distributing tokens to experts across ranks.

use candle_core::{Result, Tensor, DType};
use super::{get_ep_size, get_ep_group, CollectiveCommunicator};
use super::groups::ProcessGroup;
use std::sync::Arc;

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
}

impl ExpertParallelConfig {
    pub fn new(num_experts: usize) -> Self {
        let ep_size = get_ep_size();
        let ep_rank = get_ep_group()
            .map(|g| g.communicator.rank())
            .unwrap_or(0);
        
        let num_local_experts = num_experts / ep_size;
        
        Self {
            ep_size,
            num_experts,
            num_local_experts,
            ep_rank,
        }
    }
    
    /// Get expert IDs that this rank is responsible for.
    pub fn local_expert_ids(&self) -> Vec<usize> {
        let start = self.ep_rank * self.num_local_experts;
        (start..start + self.num_local_experts).collect()
    }
}

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
}

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
        let num_local_experts = num_experts / ep_size;
        
        let config = ExpertParallelConfig {
            ep_size,
            num_experts,
            num_local_experts,
            ep_rank,
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
        
        let info = DispatchInfo {
            restore_indices,
            send_counts: vec![num_tokens],
            recv_counts: vec![num_tokens],
            original_shape: vec![num_tokens, hidden_dim],
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
        
        // Combine should restore original order
        let combined = dispatcher.combine(&dispatched, &info)?;
        assert_eq!(combined.dims(), x.dims());
        
        Ok(())
    }
}
