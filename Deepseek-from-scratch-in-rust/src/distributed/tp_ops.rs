//! Tensor Parallel autograd operations.
//!
//! These operations handle the forward/backward communication patterns
//! for tensor parallelism without explicit backward pass (since Candle
//! uses eager execution).

use candle_core::{Result, Tensor};
use super::{get_tp_group, get_tp_size};

/// Copy input to model parallel region.
/// 
/// Forward: Identity (no-op)
/// Backward: All-reduce gradients
/// 
/// Use this at the START of a column-parallel layer sequence.
pub struct CopyToModelParallelRegion;

impl CopyToModelParallelRegion {
    /// Forward pass - just returns the input unchanged.
    /// In PyTorch autograd, this would register an all-reduce for backward.
    pub fn apply(x: &Tensor) -> Result<Tensor> {
        // Forward is identity
        Ok(x.clone())
    }
    
    /// Manual backward: All-reduce gradients across TP group.
    /// Call this when computing gradients for the input tensor.
    pub fn backward(grad: &Tensor) -> Result<Tensor> {
        let tp_size = get_tp_size();
        if tp_size <= 1 {
            return Ok(grad.clone());
        }
        
        // All-reduce the gradient
        if let Some(group) = get_tp_group() {
            group.communicator.all_reduce(grad)
        } else {
            Ok(grad.clone())
        }
    }
}

/// Reduce output from model parallel region.
/// 
/// Forward: All-reduce across TP ranks
/// Backward: Identity (no-op)
/// 
/// Use this at the END of a row-parallel layer sequence.
pub struct ReduceFromModelParallelRegion;

impl ReduceFromModelParallelRegion {
    /// Forward pass - all-reduce the tensor across TP group.
    pub fn apply(x: &Tensor) -> Result<Tensor> {
        let tp_size = get_tp_size();
        if tp_size <= 1 {
            return Ok(x.clone());
        }
        
        if let Some(group) = get_tp_group() {
            group.communicator.all_reduce(x)
        } else {
            Ok(x.clone())
        }
    }
    
    /// Manual backward: Identity (gradients flow unchanged).
    pub fn backward(grad: &Tensor) -> Result<Tensor> {
        Ok(grad.clone())
    }
}

/// Scatter input to model parallel region.
/// 
/// Forward: Split input along last dimension, each rank gets its slice
/// Backward: All-gather gradients
/// 
/// Use when feeding non-parallel data into a row-parallel layer.
pub struct ScatterToModelParallelRegion;

impl ScatterToModelParallelRegion {
    /// Forward pass - scatter (split) the tensor along last dimension.
    pub fn apply(x: &Tensor) -> Result<Tensor> {
        let tp_size = get_tp_size();
        let tp_rank = super::get_tp_rank();
        
        if tp_size <= 1 {
            return Ok(x.clone());
        }
        
        let dims = x.dims();
        let last_dim = dims.len() - 1;
        let last_size = dims[last_dim];
        
        if last_size % tp_size != 0 {
            return Err(candle_core::Error::Msg(format!(
                "Last dimension ({}) must be divisible by tp_size ({})",
                last_size, tp_size
            )));
        }
        
        let chunk_size = last_size / tp_size;
        let start = tp_rank * chunk_size;
        
        x.narrow(last_dim, start, chunk_size)
    }
    
    /// Manual backward: All-gather gradients to reconstruct full gradient.
    pub fn backward(grad: &Tensor) -> Result<Tensor> {
        let tp_size = get_tp_size();
        if tp_size <= 1 {
            return Ok(grad.clone());
        }
        
        if let Some(group) = get_tp_group() {
            group.communicator.all_gather(grad)
        } else {
            Ok(grad.clone())
        }
    }
}

/// Gather output from model parallel region.
/// 
/// Forward: All-gather across TP ranks along last dimension
/// Backward: Scatter gradients (each rank gets its slice)
/// 
/// Use when collecting parallel outputs for non-parallel consumption.
pub struct GatherFromModelParallelRegion;

impl GatherFromModelParallelRegion {
    /// Forward pass - all-gather the tensor along last dimension.
    pub fn apply(x: &Tensor) -> Result<Tensor> {
        let tp_size = get_tp_size();
        if tp_size <= 1 {
            return Ok(x.clone());
        }
        
        if let Some(group) = get_tp_group() {
            group.communicator.all_gather(x)
        } else {
            Ok(x.clone())
        }
    }
    
    /// Manual backward: Scatter (split) gradient, each rank gets its slice.
    pub fn backward(grad: &Tensor) -> Result<Tensor> {
        let tp_size = get_tp_size();
        let tp_rank = super::get_tp_rank();
        
        if tp_size <= 1 {
            return Ok(grad.clone());
        }
        
        let dims = grad.dims();
        let last_dim = dims.len() - 1;
        let last_size = dims[last_dim];
        
        let chunk_size = last_size / tp_size;
        let start = tp_rank * chunk_size;
        
        grad.narrow(last_dim, start, chunk_size)
    }
}

/// Reduce-scatter operation for tensor parallelism.
/// 
/// Forward: Reduce (sum) and scatter result across TP ranks
/// Backward: All-gather gradients
/// 
/// Use for efficient communication in certain TP patterns.
pub struct ReduceScatterToModelParallelRegion;

impl ReduceScatterToModelParallelRegion {
    /// Forward pass - reduce-scatter the tensor.
    pub fn apply(x: &Tensor) -> Result<Tensor> {
        let tp_size = get_tp_size();
        if tp_size <= 1 {
            return Ok(x.clone());
        }
        
        if let Some(group) = get_tp_group() {
            group.communicator.reduce_scatter(x)
        } else {
            Ok(x.clone())
        }
    }
    
    /// Manual backward: All-gather gradients.
    pub fn backward(grad: &Tensor) -> Result<Tensor> {
        let tp_size = get_tp_size();
        if tp_size <= 1 {
            return Ok(grad.clone());
        }
        
        if let Some(group) = get_tp_group() {
            group.communicator.all_gather(grad)
        } else {
            Ok(grad.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    
    #[test]
    fn test_copy_to_parallel_region() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (2, 4, 8), &device)?;
        
        // With tp_size=1 (default), this should be identity
        let y = CopyToModelParallelRegion::apply(&x)?;
        assert_eq!(x.dims(), y.dims());
        
        Ok(())
    }
    
    #[test]
    fn test_reduce_from_parallel_region() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (2, 4, 8), &device)?;
        
        // With tp_size=1 (default), this should be identity
        let y = ReduceFromModelParallelRegion::apply(&x)?;
        assert_eq!(x.dims(), y.dims());
        
        Ok(())
    }
    
    #[test]
    fn test_gather_from_parallel_region() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (2, 4, 8), &device)?;
        
        // With tp_size=1 (default), this should be identity
        let y = GatherFromModelParallelRegion::apply(&x)?;
        assert_eq!(x.dims(), y.dims());
        
        Ok(())
    }
}
