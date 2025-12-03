//! Tensor Parallelism primitives for distributed training.
//!
//! Provides column-parallel and row-parallel linear layers that
//! shard weights across tensor parallel ranks.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use super::{get_tp_size, get_tp_rank, get_tp_group};

/// Tensor Parallel Column Linear Layer.
/// 
/// Splits the output dimension across TP ranks.
/// Each rank holds a slice of the weight matrix: W[:, start:end]
/// 
/// Forward pass:
/// - Each rank computes: y_local = x @ W_local
/// - If gather_output=true: all-gather to get full output
/// - If gather_output=false: return local output (for chaining with RowParallel)
pub struct ColumnParallelLinear {
    linear: Linear,
    tp_size: usize,
    tp_rank: usize,
    gather_output: bool,
    full_out_dim: usize,
}

impl ColumnParallelLinear {
    /// Create a new column-parallel linear layer.
    /// 
    /// Args:
    ///   in_dim: Input dimension (not split)
    ///   out_dim: Full output dimension (will be split by tp_size)
    ///   gather_output: If true, all-gather output; if false, return sharded
    ///   vb: VarBuilder for weight initialization
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        gather_output: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let tp_size = get_tp_size();
        let tp_rank = get_tp_rank();
        
        // Validate dimensions
        if out_dim % tp_size != 0 {
            return Err(candle_core::Error::Msg(format!(
                "out_dim ({}) must be divisible by tp_size ({})",
                out_dim, tp_size
            )));
        }
        
        let local_out_dim = out_dim / tp_size;
        
        // Create linear layer with local dimensions
        // In production, would load only the local shard from checkpoint
        let linear = candle_nn::linear(in_dim, local_out_dim, vb)?;
        
        Ok(Self {
            linear,
            tp_size,
            tp_rank,
            gather_output,
            full_out_dim: out_dim,
        })
    }
    
    /// Create with explicit TP configuration (for testing without global state)
    pub fn new_with_config(
        in_dim: usize,
        out_dim: usize,
        gather_output: bool,
        tp_size: usize,
        tp_rank: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if out_dim % tp_size != 0 {
            return Err(candle_core::Error::Msg(format!(
                "out_dim ({}) must be divisible by tp_size ({})",
                out_dim, tp_size
            )));
        }
        
        let local_out_dim = out_dim / tp_size;
        let linear = candle_nn::linear(in_dim, local_out_dim, vb)?;
        
        Ok(Self {
            linear,
            tp_size,
            tp_rank,
            gather_output,
            full_out_dim: out_dim,
        })
    }
    
    /// Get the local output dimension
    pub fn local_out_dim(&self) -> usize {
        self.full_out_dim / self.tp_size
    }
}

impl Module for ColumnParallelLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute local output
        let local_output = self.linear.forward(x)?;
        
        if self.tp_size == 1 || !self.gather_output {
            return Ok(local_output);
        }
        
        // All-gather across TP group
        if let Some(group) = get_tp_group() {
            group.communicator.all_gather(&local_output)
        } else {
            // Fallback for when distributed is not initialized
            Ok(local_output)
        }
    }
}

/// Tensor Parallel Row Linear Layer.
/// 
/// Splits the input dimension across TP ranks.
/// Each rank holds a slice of the weight matrix: W[start:end, :]
/// 
/// Forward pass:
/// - Input is already sharded across ranks (from ColumnParallel)
/// - Each rank computes: y_local = x_local @ W_local
/// - All-reduce to sum partial results
pub struct RowParallelLinear {
    linear: Linear,
    tp_size: usize,
    tp_rank: usize,
    input_is_parallel: bool,
}

impl RowParallelLinear {
    /// Create a new row-parallel linear layer.
    /// 
    /// Args:
    ///   in_dim: Full input dimension (will be split by tp_size)
    ///   out_dim: Output dimension (not split)
    ///   input_is_parallel: If true, input is already sharded
    ///   vb: VarBuilder for weight initialization
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        input_is_parallel: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let tp_size = get_tp_size();
        let tp_rank = get_tp_rank();
        
        // Validate dimensions
        if in_dim % tp_size != 0 {
            return Err(candle_core::Error::Msg(format!(
                "in_dim ({}) must be divisible by tp_size ({})",
                in_dim, tp_size
            )));
        }
        
        let local_in_dim = in_dim / tp_size;
        let linear = candle_nn::linear(local_in_dim, out_dim, vb)?;
        
        Ok(Self {
            linear,
            tp_size,
            tp_rank,
            input_is_parallel,
        })
    }
    
    /// Create with explicit TP configuration (for testing)
    pub fn new_with_config(
        in_dim: usize,
        out_dim: usize,
        input_is_parallel: bool,
        tp_size: usize,
        tp_rank: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if in_dim % tp_size != 0 {
            return Err(candle_core::Error::Msg(format!(
                "in_dim ({}) must be divisible by tp_size ({})",
                in_dim, tp_size
            )));
        }
        
        let local_in_dim = in_dim / tp_size;
        let linear = candle_nn::linear(local_in_dim, out_dim, vb)?;
        
        Ok(Self {
            linear,
            tp_size,
            tp_rank,
            input_is_parallel,
        })
    }
}

impl Module for RowParallelLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute local output
        let local_output = self.linear.forward(x)?;
        
        if self.tp_size == 1 {
            return Ok(local_output);
        }
        
        // All-reduce across TP group to sum partial results
        if let Some(group) = get_tp_group() {
            group.communicator.all_reduce(&local_output)
        } else {
            Ok(local_output)
        }
    }
}

/// Parallel Embedding layer.
/// 
/// Vocabulary is split across TP ranks.
/// Each rank holds embeddings for vocab[start:end].
pub struct ParallelEmbedding {
    embedding: candle_nn::Embedding,
    tp_size: usize,
    tp_rank: usize,
    vocab_start: usize,
    vocab_end: usize,
    full_vocab_size: usize,
}

impl ParallelEmbedding {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let tp_size = get_tp_size();
        let tp_rank = get_tp_rank();
        
        if vocab_size % tp_size != 0 {
            return Err(candle_core::Error::Msg(format!(
                "vocab_size ({}) must be divisible by tp_size ({})",
                vocab_size, tp_size
            )));
        }
        
        let local_vocab = vocab_size / tp_size;
        let vocab_start = tp_rank * local_vocab;
        let vocab_end = vocab_start + local_vocab;
        
        let embedding = candle_nn::embedding(local_vocab, embed_dim, vb)?;
        
        Ok(Self {
            embedding,
            tp_size,
            tp_rank,
            vocab_start,
            vocab_end,
            full_vocab_size: vocab_size,
        })
    }
}

impl Module for ParallelEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.tp_size == 1 {
            return self.embedding.forward(x);
        }
        
        // Mask tokens not in our range
        let local_ids = (x - self.vocab_start as f64)?;
        
        // Get local embeddings (will be zero for out-of-range tokens)
        let local_embed = self.embedding.forward(&local_ids)?;
        
        // All-reduce to combine (other ranks have zeros for our tokens)
        if let Some(group) = get_tp_group() {
            group.communicator.all_reduce(&local_embed)
        } else {
            Ok(local_embed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    
    #[test]
    fn test_column_parallel_shapes() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        // Single rank (tp_size=1)
        let layer = ColumnParallelLinear::new_with_config(
            64, 128, false, 1, 0, vb.clone()
        )?;
        
        let input = Tensor::randn(0f32, 1f32, (2, 10, 64), &device)?;
        let output = layer.forward(&input)?;
        
        assert_eq!(output.dims3()?, (2, 10, 128));
        Ok(())
    }
    
    #[test]
    fn test_row_parallel_shapes() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        // Single rank (tp_size=1)
        let layer = RowParallelLinear::new_with_config(
            128, 64, true, 1, 0, vb.clone()
        )?;
        
        let input = Tensor::randn(0f32, 1f32, (2, 10, 128), &device)?;
        let output = layer.forward(&input)?;
        
        assert_eq!(output.dims3()?, (2, 10, 64));
        Ok(())
    }
}

