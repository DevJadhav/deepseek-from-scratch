//! Distributed training infrastructure for DeepSeek.
//!
//! Provides 5D parallelism support:
//! - Data Parallelism (DP)
//! - Tensor Parallelism (TP)
//! - Pipeline Parallelism (PP)
//! - Expert Parallelism (EP)
//! - Sequence Parallelism (SP)

use candle_core::{Result, Tensor};

/// Trait for collective communications.
/// 
/// Implementations include:
/// - `LocalCommunicator`: In-process simulation for testing
/// - `NcclCommunicator`: NCCL-based GPU communication (requires cuda feature)
pub trait CollectiveCommunicator: Send + Sync {
    /// Get the rank of this process
    fn rank(&self) -> usize;
    
    /// Get the total number of processes
    fn world_size(&self) -> usize;
    
    /// All-reduce: Sum tensor across all ranks and broadcast result.
    fn all_reduce(&self, tensor: &Tensor) -> Result<Tensor>;
    
    /// All-gather: Gather tensors from all ranks, concatenating along dim 0.
    fn all_gather(&self, tensor: &Tensor) -> Result<Tensor>;
    
    /// Broadcast: Send tensor from root rank to all others.
    fn broadcast(&self, tensor: &Tensor, root_rank: usize) -> Result<Tensor>;
    
    /// Reduce-scatter: Reduce and scatter result to all ranks.
    fn reduce_scatter(&self, tensor: &Tensor) -> Result<Tensor> {
        // Default implementation: all-reduce then slice
        let reduced = self.all_reduce(tensor)?;
        let chunk_size = reduced.dim(0)? / self.world_size();
        let start = self.rank() * chunk_size;
        reduced.narrow(0, start, chunk_size)
    }
    
    /// All-to-all: Each rank sends different data to each other rank.
    /// Input shape: (world_size, tokens_per_rank, hidden_dim)
    /// Output shape: (world_size, tokens_per_rank, hidden_dim) but transposed across ranks
    fn all_to_all(&self, tensor: &Tensor) -> Result<Tensor> {
        // Default implementation for single rank
        if self.world_size() == 1 {
            return Ok(tensor.clone());
        }
        Err(candle_core::Error::Msg("all_to_all not implemented for this backend".to_string()))
    }
    
    /// All-to-all with variable split sizes (for MoE token dispatch).
    /// 
    /// Args:
    ///   tensor: Input tensor to send
    ///   input_splits: Number of elements to send to each rank
    ///   output_splits: Number of elements to receive from each rank
    fn all_to_all_variable(
        &self,
        tensor: &Tensor,
        input_splits: &[usize],
        output_splits: &[usize],
    ) -> Result<Tensor> {
        // Default implementation for single rank
        if self.world_size() == 1 {
            return Ok(tensor.clone());
        }
        Err(candle_core::Error::Msg("all_to_all_variable not implemented for this backend".to_string()))
    }
    
    /// Point-to-point send
    fn send(&self, tensor: &Tensor, dst_rank: usize) -> Result<()> {
        let _ = (tensor, dst_rank);
        Err(candle_core::Error::Msg("send not implemented for this backend".to_string()))
    }
    
    /// Point-to-point receive
    fn recv(&self, shape: &[usize], device: &candle_core::Device, src_rank: usize) -> Result<Tensor> {
        let _ = (shape, device, src_rank);
        Err(candle_core::Error::Msg("recv not implemented for this backend".to_string()))
    }
}

pub mod parallel;
pub mod expert;
pub mod backend;
pub mod groups;
pub mod tp_ops;
pub mod pipeline;
pub mod ring_attention;
pub mod nccl_sys;
pub mod nccl_backend;

// Re-export commonly used items
pub use groups::{
    initialize_distributed, cleanup_distributed, is_initialized,
    get_rank, get_world_size, get_tp_size, get_tp_rank, get_dp_size, get_ep_size,
    get_pp_size, get_pp_rank, get_sp_size, get_sp_rank,
    get_dp_group, get_tp_group, get_pp_group, get_ep_group, get_sp_group,
    ParallelismConfig, ProcessGroup,
};
pub use nccl_backend::NcclCommunicator;
pub use backend::LocalCommunicator;
pub use tp_ops::{
    CopyToModelParallelRegion, ReduceFromModelParallelRegion,
    ScatterToModelParallelRegion, GatherFromModelParallelRegion,
};
pub use pipeline::{PipelineStage, OneFOneBScheduler, GPipeScheduler, ScheduleAction};
pub use ring_attention::{RingAttention, RingAttentionConfig, DistributedLayerNorm};

/// Synchronize gradients across ranks using all-reduce.
/// 
/// This iterates over all gradients and performs all-reduce to average them.
/// All ranks must call this with gradients in the same order.
pub fn synchronize_gradients(
    grads: &mut std::collections::HashMap<candle_core::TensorId, Tensor>,
    communicator: &dyn CollectiveCommunicator,
) -> Result<()> {
    if communicator.world_size() == 1 {
        return Ok(());
    }
    
    // Sort by TensorId for deterministic ordering across ranks
    let mut ids: Vec<candle_core::TensorId> = grads.keys().cloned().collect();
    ids.sort_by_key(|id| format!("{:?}", id));

    for id in ids {
        if let Some(grad) = grads.get(&id) {
            let summed = communicator.all_reduce(grad)?;
            let avg = (summed / communicator.world_size() as f64)?;
            grads.insert(id, avg);
        }
    }
    Ok(())
}

/// Reduce a scalar value across all ranks.
pub fn all_reduce_scalar(
    value: f64,
    communicator: &dyn CollectiveCommunicator,
) -> Result<f64> {
    if communicator.world_size() == 1 {
        return Ok(value);
    }
    
    let tensor = Tensor::new(&[value as f32], &candle_core::Device::Cpu)?;
    let reduced = communicator.all_reduce(&tensor)?;
    let result = reduced.to_vec1::<f32>()?[0] as f64;
    Ok(result / communicator.world_size() as f64)
}
