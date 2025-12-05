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

// ============================================================================
// Latent KV Cache for MLA (Task 1.1: Fix MLA KV Cache Compression)
// ============================================================================

/// Latent KV Cache for Multi-Head Latent Attention.
/// 
/// Instead of storing full K/V tensors (d_model × seq_len), we store the compressed
/// latent representation C_KV (d_latent × seq_len). This achieves ~14× memory reduction
/// when d_latent = d_model/14 (e.g., d_latent=512 for d_model=7168).
/// 
/// The K and V tensors are up-projected on-demand during attention computation.
pub struct LatentKVCache {
    /// Compressed latent cache: (batch, seq_len, d_latent)
    latent_cache: Tensor,
    /// Current sequence length in cache
    current_seq_len: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Latent dimension (d_latent)
    d_latent: usize,
    /// Batch size
    batch_size: usize,
}

impl LatentKVCache {
    /// Create a new latent KV cache.
    /// 
    /// # Arguments
    /// * `batch_size` - Batch size
    /// * `max_seq_len` - Maximum sequence length to cache
    /// * `d_latent` - Latent dimension (compressed KV size)
    /// * `dtype` - Data type for the cache
    /// * `device` - Device to allocate cache on
    pub fn new(
        batch_size: usize,
        max_seq_len: usize,
        d_latent: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let latent_cache = Tensor::zeros((batch_size, max_seq_len, d_latent), dtype, device)?;
        
        Ok(Self {
            latent_cache,
            current_seq_len: 0,
            max_seq_len,
            d_latent,
            batch_size,
        })
    }

    /// Update cache with new compressed latent representation.
    /// 
    /// # Arguments
    /// * `c_kv` - Compressed KV latent: (batch, seq_len_new, d_latent)
    /// 
    /// # Returns
    /// The full cached latent sequence up to current position: (batch, seq_len_total, d_latent)
    pub fn update(&mut self, c_kv: &Tensor) -> Result<Tensor> {
        let (b, seq_len, d) = c_kv.dims3()?;
        
        if b > self.batch_size {
            candle_core::bail!(
                "Batch size {} exceeds cache batch size {}",
                b, self.batch_size
            );
        }
        
        if d != self.d_latent {
            candle_core::bail!(
                "Latent dimension {} doesn't match cache dimension {}",
                d, self.d_latent
            );
        }
        
        let start_pos = self.current_seq_len;
        let end_pos = start_pos + seq_len;
        
        if end_pos > self.max_seq_len {
            candle_core::bail!(
                "Latent KV Cache overflow: {} + {} > {}",
                start_pos, seq_len, self.max_seq_len
            );
        }
        
        // Insert new latent data into cache
        self.latent_cache = self.latent_cache.slice_assign(
            &[0..b, start_pos..end_pos, 0..d],
            c_kv
        )?;
        
        self.current_seq_len = end_pos;
        
        // Return the valid part of the cache
        let latent_out = self.latent_cache.i((.., 0..end_pos, ..))?;
        
        Ok(latent_out)
    }

    /// Get current sequence length in cache
    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }
    
    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    
    /// Get latent dimension
    pub fn d_latent(&self) -> usize {
        self.d_latent
    }
    
    /// Reset the cache to empty state
    pub fn reset(&mut self) {
        self.current_seq_len = 0;
    }
    
    /// Trim cache to a specific length (used in speculative decoding rejection)
    pub fn trim_to(&mut self, length: usize) -> Result<()> {
        if length > self.current_seq_len {
            candle_core::bail!(
                "Cannot trim to length {} > current {}",
                length, self.current_seq_len
            );
        }
        self.current_seq_len = length;
        Ok(())
    }
    
    /// Calculate memory savings compared to full KV cache.
    /// 
    /// Returns (latent_cache_bytes, full_kv_bytes, compression_ratio)
    pub fn memory_stats(&self, d_model: usize, n_heads: usize, dtype_bytes: usize) -> (usize, usize, f64) {
        let head_dim = d_model / n_heads;
        
        // Latent cache: batch × seq × d_latent
        let latent_bytes = self.batch_size * self.max_seq_len * self.d_latent * dtype_bytes;
        
        // Full KV cache: batch × n_heads × seq × head_dim × 2 (K and V)
        let full_kv_bytes = self.batch_size * n_heads * self.max_seq_len * head_dim * 2 * dtype_bytes;
        
        let compression_ratio = full_kv_bytes as f64 / latent_bytes as f64;
        
        (latent_bytes, full_kv_bytes, compression_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kv_cache_basic() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        let max_seq_len = 128;
        let n_heads = 4;
        let head_dim = 32;
        
        let mut cache = KVCache::new(batch_size, max_seq_len, n_heads, head_dim, DType::F32, &device)?;
        
        // Add first sequence
        let k1 = Tensor::randn(0f32, 1f32, (batch_size, n_heads, 16, head_dim), &device)?;
        let v1 = Tensor::randn(0f32, 1f32, (batch_size, n_heads, 16, head_dim), &device)?;
        
        let (k_out, v_out) = cache.update(&k1, &v1)?;
        assert_eq!(k_out.dims(), &[batch_size, n_heads, 16, head_dim]);
        assert_eq!(v_out.dims(), &[batch_size, n_heads, 16, head_dim]);
        assert_eq!(cache.current_seq_len(), 16);
        
        // Add more tokens
        let k2 = Tensor::randn(0f32, 1f32, (batch_size, n_heads, 8, head_dim), &device)?;
        let v2 = Tensor::randn(0f32, 1f32, (batch_size, n_heads, 8, head_dim), &device)?;
        
        let (k_out, v_out) = cache.update(&k2, &v2)?;
        assert_eq!(k_out.dims(), &[batch_size, n_heads, 24, head_dim]);
        assert_eq!(v_out.dims(), &[batch_size, n_heads, 24, head_dim]);
        assert_eq!(cache.current_seq_len(), 24);
        
        Ok(())
    }
    
    #[test]
    fn test_latent_kv_cache_basic() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        let max_seq_len = 128;
        let d_latent = 64;  // Compressed dimension
        
        let mut cache = LatentKVCache::new(batch_size, max_seq_len, d_latent, DType::F32, &device)?;
        
        // Add first sequence of compressed latents
        let c_kv1 = Tensor::randn(0f32, 1f32, (batch_size, 16, d_latent), &device)?;
        
        let latent_out = cache.update(&c_kv1)?;
        assert_eq!(latent_out.dims(), &[batch_size, 16, d_latent]);
        assert_eq!(cache.current_seq_len(), 16);
        
        // Add more tokens
        let c_kv2 = Tensor::randn(0f32, 1f32, (batch_size, 8, d_latent), &device)?;
        
        let latent_out = cache.update(&c_kv2)?;
        assert_eq!(latent_out.dims(), &[batch_size, 24, d_latent]);
        assert_eq!(cache.current_seq_len(), 24);
        
        Ok(())
    }
    
    #[test]
    fn test_latent_kv_cache_memory_savings() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 4;
        let max_seq_len = 4096;
        let d_model = 4096;
        let d_latent = 512;  // ~8× compression
        let n_heads = 32;
        let dtype_bytes = 4;  // f32
        
        let cache = LatentKVCache::new(batch_size, max_seq_len, d_latent, DType::F32, &device)?;
        
        let (latent_bytes, full_kv_bytes, ratio) = cache.memory_stats(d_model, n_heads, dtype_bytes);
        
        // Verify significant memory savings
        assert!(ratio > 7.0, "Expected >7× compression, got {:.2}×", ratio);
        assert!(latent_bytes < full_kv_bytes, "Latent cache should be smaller");
        
        println!("Memory stats:");
        println!("  Latent cache: {} bytes ({:.2} MB)", latent_bytes, latent_bytes as f64 / 1e6);
        println!("  Full KV cache: {} bytes ({:.2} MB)", full_kv_bytes, full_kv_bytes as f64 / 1e6);
        println!("  Compression ratio: {:.2}×", ratio);
        
        Ok(())
    }
    
    #[test]
    fn test_latent_kv_cache_reset() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = LatentKVCache::new(2, 128, 64, DType::F32, &device)?;
        
        let c_kv = Tensor::randn(0f32, 1f32, (2, 32, 64), &device)?;
        cache.update(&c_kv)?;
        assert_eq!(cache.current_seq_len(), 32);
        
        cache.reset();
        assert_eq!(cache.current_seq_len(), 0);
        
        Ok(())
    }
    
    #[test]
    fn test_latent_kv_cache_trim() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = LatentKVCache::new(2, 128, 64, DType::F32, &device)?;
        
        let c_kv = Tensor::randn(0f32, 1f32, (2, 50, 64), &device)?;
        cache.update(&c_kv)?;
        assert_eq!(cache.current_seq_len(), 50);
        
        // Trim to smaller length (e.g., after speculative decoding rejection)
        cache.trim_to(30)?;
        assert_eq!(cache.current_seq_len(), 30);
        
        // Cannot trim to larger length
        assert!(cache.trim_to(40).is_err());
        
        Ok(())
    }
    
    #[test]
    fn test_latent_kv_cache_overflow() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = LatentKVCache::new(2, 32, 64, DType::F32, &device)?;
        
        // First update within bounds
        let c_kv1 = Tensor::randn(0f32, 1f32, (2, 20, 64), &device)?;
        cache.update(&c_kv1)?;
        
        // Second update causes overflow
        let c_kv2 = Tensor::randn(0f32, 1f32, (2, 20, 64), &device)?;
        assert!(cache.update(&c_kv2).is_err(), "Should fail due to overflow");
        
        Ok(())
    }
}
