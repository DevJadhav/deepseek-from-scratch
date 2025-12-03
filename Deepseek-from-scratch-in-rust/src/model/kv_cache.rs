use candle_core::{Result, Tensor, Device, DType, IndexOp};

/// Key-Value Cache for efficient generation.
pub struct KVCache {
    k_cache: Tensor,
    v_cache: Tensor,
    current_seq_len: usize,
    max_seq_len: usize,
}

impl KVCache {
    pub fn new(
        batch_size: usize,
        max_seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let k_cache = Tensor::zeros((batch_size, n_heads, max_seq_len, head_dim), dtype, device)?;
        let v_cache = Tensor::zeros((batch_size, n_heads, max_seq_len, head_dim), dtype, device)?;
        
        Ok(Self {
            k_cache,
            v_cache,
            current_seq_len: 0,
            max_seq_len,
        })
    }

    /// Update cache with new k, v and return the full cached sequence (up to current position).
    /// k, v: (B, H, S_new, D)
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, h, seq_len, d) = k.dims4()?;
        let start_pos = self.current_seq_len;
        let end_pos = start_pos + seq_len;
        
        if end_pos > self.max_seq_len {
            candle_core::bail!("KV Cache overflow");
        }
        
        // Insert new data into cache
        // Note: Candle tensors are immutable. We use slice_assign which returns a new tensor 
        // if we were operating on a variable, but here we have to be clever.
        // Actually, for this implementation without `Var`, we might have to reconstruct.
        // BUT, to be efficient, we should use `slice_assign` on the underlying storage if possible.
        // Candle's `slice_assign` works on Tensors.
        
        self.k_cache = self.k_cache.slice_assign(&[0..b, 0..h, start_pos..end_pos, 0..d], k)?;
        self.v_cache = self.v_cache.slice_assign(&[0..b, 0..h, start_pos..end_pos, 0..d], v)?;
        
        self.current_seq_len = end_pos;
        
        // Return the valid part of the cache
        let k_out = self.k_cache.i((.., .., 0..end_pos, ..))?;
        let v_out = self.v_cache.i((.., .., 0..end_pos, ..))?;
        
        Ok((k_out, v_out))
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }
    
    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    
    /// Reset the cache to empty state
    pub fn reset(&mut self) {
        self.current_seq_len = 0;
    }
    
    /// Trim cache to a specific length (used in speculative decoding rejection)
    pub fn trim_to(&mut self, length: usize) -> Result<()> {
        if length > self.current_seq_len {
            candle_core::bail!("Cannot trim to length {} > current {}", length, self.current_seq_len);
        }
        self.current_seq_len = length;
        Ok(())
    }
}
