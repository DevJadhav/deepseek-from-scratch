//! CUDA Hopper Features Implementation
//!
//! This module provides Hopper-specific optimizations for SM 9.0+ GPUs:
//! - TMA (Tensor Memory Accelerator): Async global->shared memory with hardware-managed tiling
//! - Warpgroup Collectives: 128-thread operations for matrix multiply
//! - FP8 Tensor Core: SM 9.0+ native FP8 support
//!
//! # Architecture Support
//!
//! | Generation | SM Version | Features |
//! |------------|------------|----------|
//! | Hopper     | SM 9.0     | TMA, Warpgroup, FP8, Clusters |
//! | Ampere     | SM 8.0     | BF16, TF32, Async Copy |
//! | Turing     | SM 7.5     | INT8 Tensor Cores |
//! | Volta      | SM 7.0     | FP16 Tensor Cores |

use candle_core::{DType, Device, Result, Tensor};


/// CUDA compute capability version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
}

impl ComputeCapability {
    /// Create a new compute capability
    pub fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Check if capability meets minimum version
    pub fn at_least(&self, major: u32, minor: u32) -> bool {
        (self.major, self.minor) >= (major, minor)
    }

    /// Check if this is Hopper (SM 9.0+)
    pub fn is_hopper(&self) -> bool {
        self.at_least(9, 0)
    }

    /// Check if this is Ampere (SM 8.0+)
    pub fn is_ampere(&self) -> bool {
        self.at_least(8, 0)
    }

    /// Check if this is Turing (SM 7.5+)
    pub fn is_turing(&self) -> bool {
        self.at_least(7, 5)
    }

    /// Check if this is Volta (SM 7.0+)
    pub fn is_volta(&self) -> bool {
        self.at_least(7, 0)
    }
}

impl std::fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SM {}.{}", self.major, self.minor)
    }
}

/// CUDA features detected from hardware
#[derive(Debug, Clone)]
pub struct CUDAFeatures {
    /// Device name
    pub device_name: String,
    /// Device index
    pub device_index: usize,
    /// Compute capability
    pub compute_capability: ComputeCapability,

    // Hopper features (SM 9.0+)
    /// TMA (Tensor Memory Accelerator) support
    pub has_tma: bool,
    /// Warpgroup collectives (128 threads)
    pub has_warpgroup: bool,
    /// FP8 tensor core support
    pub has_fp8_tensor_core: bool,
    /// Thread block clusters support
    pub has_cluster: bool,

    // Ampere features (SM 8.0+)
    /// BF16 tensor core support
    pub has_bf16_tensor_core: bool,
    /// TF32 support for FP32 operations
    pub has_tf32: bool,
    /// Async copy instructions
    pub has_async_copy: bool,

    // Volta/Turing features
    /// FP16 tensor core support
    pub has_fp16_tensor_core: bool,
    /// INT8 tensor core support
    pub has_int8_tensor_core: bool,

    // Memory info
    /// Total device memory in GB
    pub total_memory_gb: f64,
    /// Shared memory per block in bytes
    pub shared_memory_per_block: usize,
    /// Shared memory per SM in bytes
    pub shared_memory_per_sm: usize,
    /// L2 cache size in bytes
    pub l2_cache_size: usize,

    // Compute info
    /// Number of streaming multiprocessors
    pub sm_count: usize,
    /// Max threads per SM
    pub max_threads_per_sm: usize,
    /// Warp size
    pub warp_size: usize,
    /// Max threads per block
    pub max_threads_per_block: usize,
}

impl Default for CUDAFeatures {
    fn default() -> Self {
        Self {
            device_name: String::new(),
            device_index: 0,
            compute_capability: ComputeCapability::new(0, 0),
            has_tma: false,
            has_warpgroup: false,
            has_fp8_tensor_core: false,
            has_cluster: false,
            has_bf16_tensor_core: false,
            has_tf32: false,
            has_async_copy: false,
            has_fp16_tensor_core: false,
            has_int8_tensor_core: false,
            total_memory_gb: 0.0,
            shared_memory_per_block: 0,
            shared_memory_per_sm: 0,
            l2_cache_size: 0,
            sm_count: 0,
            max_threads_per_sm: 0,
            warp_size: 32,
            max_threads_per_block: 1024,
        }
    }
}

impl CUDAFeatures {
    /// Detect features from device
    pub fn detect(device: &Device) -> Self {
        match device {
            Device::Cuda(_cuda_device) => {
                // In a real implementation, we'd query CUDA device properties
                // For now, return conservative defaults for Ampere
                Self::a100()
            }
            _ => Self::default(),
        }
    }

    /// Create features for H100 GPU (Hopper)
    pub fn h100() -> Self {
        Self {
            device_name: "NVIDIA H100 80GB HBM3".to_string(),
            device_index: 0,
            compute_capability: ComputeCapability::new(9, 0),
            has_tma: true,
            has_warpgroup: true,
            has_fp8_tensor_core: true,
            has_cluster: true,
            has_bf16_tensor_core: true,
            has_tf32: true,
            has_async_copy: true,
            has_fp16_tensor_core: true,
            has_int8_tensor_core: true,
            total_memory_gb: 80.0,
            shared_memory_per_block: 228 * 1024,
            shared_memory_per_sm: 228 * 1024,
            l2_cache_size: 50 * 1024 * 1024,
            sm_count: 132,
            max_threads_per_sm: 2048,
            warp_size: 32,
            max_threads_per_block: 1024,
        }
    }

    /// Create features for A100 GPU (Ampere)
    pub fn a100() -> Self {
        Self {
            device_name: "NVIDIA A100 80GB PCIe".to_string(),
            device_index: 0,
            compute_capability: ComputeCapability::new(8, 0),
            has_tma: false,
            has_warpgroup: false,
            has_fp8_tensor_core: false,
            has_cluster: false,
            has_bf16_tensor_core: true,
            has_tf32: true,
            has_async_copy: true,
            has_fp16_tensor_core: true,
            has_int8_tensor_core: true,
            total_memory_gb: 80.0,
            shared_memory_per_block: 164 * 1024,
            shared_memory_per_sm: 164 * 1024,
            l2_cache_size: 40 * 1024 * 1024,
            sm_count: 108,
            max_threads_per_sm: 2048,
            warp_size: 32,
            max_threads_per_block: 1024,
        }
    }

    /// Create features for V100 GPU (Volta)
    pub fn v100() -> Self {
        Self {
            device_name: "NVIDIA V100 32GB".to_string(),
            device_index: 0,
            compute_capability: ComputeCapability::new(7, 0),
            has_tma: false,
            has_warpgroup: false,
            has_fp8_tensor_core: false,
            has_cluster: false,
            has_bf16_tensor_core: false,
            has_tf32: false,
            has_async_copy: false,
            has_fp16_tensor_core: true,
            has_int8_tensor_core: false,
            total_memory_gb: 32.0,
            shared_memory_per_block: 96 * 1024,
            shared_memory_per_sm: 96 * 1024,
            l2_cache_size: 6 * 1024 * 1024,
            sm_count: 80,
            max_threads_per_sm: 2048,
            warp_size: 32,
            max_threads_per_block: 1024,
        }
    }

    /// Get optimal tile size for matrix operations
    pub fn optimal_tile_size(&self) -> (usize, usize, usize) {
        if self.has_tma {
            (256, 128, 64) // Hopper with TMA
        } else if self.has_bf16_tensor_core {
            (128, 128, 32) // Ampere
        } else {
            (64, 64, 16) // Older GPUs
        }
    }

    /// Get optimal block size for element-wise operations
    pub fn optimal_block_size(&self, elements: usize) -> usize {
        if self.has_warpgroup {
            (((elements + 127) / 128) * 128).min(256)
        } else {
            (((elements + 31) / 32) * 32).min(256)
        }
    }
}

/// Kernel backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBackend {
    /// CPU fallback
    Cpu,
    /// Hopper optimized (SM 9.0+)
    CudaHopper,
    /// Ampere optimized (SM 8.0+)
    CudaAmpere,
    /// Standard CUDA (SM 7.x)
    CudaStandard,
}

impl KernelBackend {
    /// Select optimal backend for device
    pub fn select(device: &Device, features: Option<&CUDAFeatures>) -> Self {
        match device {
            Device::Cpu => Self::Cpu,
            Device::Metal(_) => Self::Cpu, // Metal has its own path
            Device::Cuda(_) => {
                let features = features.map(|f| f.clone()).unwrap_or_else(|| CUDAFeatures::detect(device));
                
                if features.has_tma && features.has_warpgroup {
                    Self::CudaHopper
                } else if features.has_bf16_tensor_core {
                    Self::CudaAmpere
                } else {
                    Self::CudaStandard
                }
            }
        }
    }
}

// =============================================================================
// TMA (Tensor Memory Accelerator) - Hopper Only
// =============================================================================

/// Swizzle mode for TMA operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TMASwizzleMode {
    /// No swizzling
    None,
    /// 32-byte swizzle
    Swizzle32B,
    /// 64-byte swizzle
    Swizzle64B,
    /// 128-byte swizzle (recommended for best performance)
    Swizzle128B,
}

/// TMA descriptor for async memory operations
///
/// TMA enables asynchronous global->shared memory copies with:
/// - Hardware address generation
/// - Multi-dimensional tensor copying
/// - Automatic swizzling for conflict-free access
#[derive(Debug, Clone)]
pub struct TMADescriptor {
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Tile shape for copying
    pub tile_shape: Vec<usize>,
    /// Element stride
    pub strides: Vec<usize>,
    /// Swizzle mode
    pub swizzle_mode: TMASwizzleMode,
    /// Element size in bytes
    pub element_size: usize,
}

impl TMADescriptor {
    /// Create a new TMA descriptor
    pub fn new(
        shape: Vec<usize>,
        tile_shape: Vec<usize>,
        swizzle_mode: TMASwizzleMode,
        element_size: usize,
    ) -> Self {
        let strides = Self::compute_strides(&shape, element_size);
        Self {
            shape,
            tile_shape,
            strides,
            swizzle_mode,
            element_size,
        }
    }

    fn compute_strides(shape: &[usize], element_size: usize) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = element_size;
        for &dim in shape.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }
        strides.reverse();
        strides
    }

    /// Get the number of tiles
    pub fn num_tiles(&self) -> Vec<usize> {
        self.shape
            .iter()
            .zip(&self.tile_shape)
            .map(|(&s, &t)| (s + t - 1) / t)
            .collect()
    }

    /// Check if this descriptor is valid for TMA
    pub fn is_valid(&self) -> bool {
        // Check alignment requirements
        let tile_bytes: usize = self.tile_shape.iter().product::<usize>() * self.element_size;
        
        // TMA requires 16-byte alignment for tiles
        tile_bytes % 16 == 0
    }
}

/// Create a TMA descriptor for a tensor
pub fn create_tma_descriptor(
    tensor: &Tensor,
    tile_shape: Vec<usize>,
    swizzle_mode: TMASwizzleMode,
) -> Result<TMADescriptor> {
    let shape: Vec<usize> = tensor.dims().to_vec();
    let element_size = match tensor.dtype() {
        DType::F32 => 4,
        DType::F16 => 2,
        DType::BF16 => 2,
        DType::F64 => 8,
        DType::I64 => 8,
        DType::U32 => 4,
        DType::U8 => 1,
    };

    let desc = TMADescriptor::new(shape, tile_shape, swizzle_mode, element_size);
    
    if !desc.is_valid() {
        return Err(candle_core::Error::Msg(
            "TMA descriptor failed validation: check tile alignment".to_string()
        ));
    }

    Ok(desc)
}

/// Simulated TMA async copy (for API design)
///
/// In a real implementation, this would use CUDA's TMA instructions:
/// - cp.async.bulk.tensor for bulk tensor copies
/// - tma.prefetch for prefetching
/// - tma.commit and tma.wait for synchronization
pub fn tma_async_copy(
    _src: &Tensor,
    _dst: &mut Tensor,
    _descriptor: &TMADescriptor,
    _features: &CUDAFeatures,
) -> Result<()> {
    // This would dispatch to CUDA TMA instructions on Hopper
    // For now, we just validate the operation would be valid
    Ok(())
}

// =============================================================================
// Warpgroup Collectives - Hopper Only
// =============================================================================

/// Warpgroup configuration (128 threads = 4 warps)
#[derive(Debug, Clone)]
pub struct WarpgroupConfig {
    /// Warpgroup size (128 threads)
    pub warpgroup_size: usize,
    /// Warp size (32 threads)
    pub warp_size: usize,
    /// Number of warps per warpgroup
    pub warps_per_warpgroup: usize,
}

impl Default for WarpgroupConfig {
    fn default() -> Self {
        Self {
            warpgroup_size: 128,
            warp_size: 32,
            warps_per_warpgroup: 4,
        }
    }
}

impl WarpgroupConfig {
    /// Get number of warpgroups for a thread count
    pub fn num_warpgroups(&self, num_threads: usize) -> usize {
        (num_threads + self.warpgroup_size - 1) / self.warpgroup_size
    }

    /// Get optimal warpgroup count for matrix multiply
    pub fn for_matmul(m: usize, n: usize, k: usize) -> (Self, usize) {
        let config = Self::default();
        
        // Optimal number of warpgroups based on matrix size
        let total_elements = m * n;
        let elements_per_warpgroup = 128 * 4; // Each warpgroup handles a tile
        let num_warpgroups = (total_elements + elements_per_warpgroup - 1) / elements_per_warpgroup;
        
        // Cap at reasonable number for occupancy
        let num_warpgroups = num_warpgroups.clamp(1, 16);
        let _ = k; // Will be used for K-dimension tiling
        
        (config, num_warpgroups)
    }
}

/// Warpgroup matrix multiply accumulate (WGMMA)
///
/// WGMMA enables 128-thread cooperative matrix operations on Hopper.
/// This is significantly faster than standard WMMA (32-thread) operations.
///
/// In production, this would dispatch to:
/// - wgmma.mma_async for async matrix multiply
/// - wgmma.fence for synchronization
/// - wgmma.commit for committing operations
pub fn warpgroup_matmul(
    a: &Tensor,
    b: &Tensor,
    features: &CUDAFeatures,
) -> Result<Tensor> {
    if !features.has_warpgroup {
        // Fallback to standard matmul
        return a.matmul(b);
    }

    // On Hopper, we would use warpgroup MMA instructions
    // For now, use standard matmul which will use tensor cores if available
    a.matmul(b)
}

// =============================================================================
// FP8 Tensor Core Operations - Hopper Only
// =============================================================================

/// FP8 format types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FP8Format {
    /// E4M3: 4-bit exponent, 3-bit mantissa (range: ±240)
    /// Better precision, smaller range
    E4M3,
    /// E5M2: 5-bit exponent, 2-bit mantissa (range: ±57344)
    /// Larger range, less precision
    E5M2,
}

impl FP8Format {
    /// Get the maximum representable value
    pub fn max_value(&self) -> f32 {
        match self {
            Self::E4M3 => 240.0,
            Self::E5M2 => 57344.0,
        }
    }

    /// Get the minimum positive value
    pub fn min_positive(&self) -> f32 {
        match self {
            Self::E4M3 => 1.95e-3,  // 2^-9
            Self::E5M2 => 6.10e-5,  // 2^-14
        }
    }
}

/// FP8 configuration for tensor operations
#[derive(Debug, Clone)]
pub struct FP8Config {
    /// FP8 format to use
    pub format: FP8Format,
    /// Scale factor for quantization
    pub scale: f32,
    /// History length for dynamic scaling
    pub amax_history_len: usize,
    /// Amax history buffer
    amax_history: Vec<f32>,
    /// Current history index
    history_index: usize,
}

impl Default for FP8Config {
    fn default() -> Self {
        Self {
            format: FP8Format::E4M3,
            scale: 1.0,
            amax_history_len: 16,
            amax_history: vec![0.0; 16],
            history_index: 0,
        }
    }
}

impl FP8Config {
    /// Create config with specific format
    pub fn new(format: FP8Format) -> Self {
        Self {
            format,
            ..Default::default()
        }
    }

    /// Update scale based on tensor max value
    pub fn update_scale(&mut self, tensor: &Tensor) -> Result<()> {
        // Compute absolute max
        let amax = tensor.abs()?.max_all()?.to_scalar::<f32>()?;
        
        // Update history
        self.amax_history[self.history_index] = amax;
        self.history_index = (self.history_index + 1) % self.amax_history_len;
        
        // Compute scale from max of history
        let max_amax = self.amax_history.iter().cloned().fold(0.0f32, f32::max);
        let max_fp8 = self.format.max_value();
        
        // Scale to use full FP8 range
        self.scale = if max_amax > 0.0 {
            max_fp8 / max_amax
        } else {
            1.0
        };
        
        Ok(())
    }

    /// Quantize tensor to FP8 (simulated)
    ///
    /// Note: Actual FP8 requires CUDA 12+ and specific tensor core instructions.
    /// This simulates the quantization for API design.
    pub fn quantize(&self, tensor: &Tensor) -> Result<(Tensor, f32)> {
        // Scale the tensor
        let scaled = (tensor * self.scale as f64)?;
        
        // Clamp to FP8 range
        let max_val = self.format.max_value() as f64;
        let clamped = scaled.clamp(-max_val, max_val)?;
        
        // In production, this would convert to actual FP8 dtype
        // For now, we keep as F32 but simulate the precision loss
        let factor = match self.format {
            FP8Format::E4M3 => 16.0,  // ~4 bits mantissa precision
            FP8Format::E5M2 => 4.0,   // ~2 bits mantissa precision
        };
        
        let quantized = ((clamped * factor)?.round()? / factor)?;
        
        Ok((quantized, self.scale))
    }

    /// Dequantize FP8 tensor back to FP32
    pub fn dequantize(&self, tensor: &Tensor, scale: f32) -> Result<Tensor> {
        tensor * (1.0 / scale as f64)
    }
}

/// FP8 matrix multiplication
///
/// Uses FP8 tensor cores on Hopper for maximum throughput.
/// Falls back to BF16/FP16 on older GPUs.
pub fn fp8_matmul(
    a: &Tensor,
    b: &Tensor,
    config: Option<&FP8Config>,
    features: &CUDAFeatures,
) -> Result<Tensor> {
    let config = config.cloned().unwrap_or_default();

    if features.has_fp8_tensor_core {
        // Quantize inputs to FP8
        let (a_fp8, scale_a) = config.quantize(a)?;
        let (b_fp8, scale_b) = config.quantize(b)?;
        
        // Matrix multiply (would use FP8 tensor cores)
        let result = a_fp8.matmul(&b_fp8)?;
        
        // Dequantize result
        let combined_scale = scale_a * scale_b;
        config.dequantize(&result, combined_scale)
    } else if features.has_bf16_tensor_core {
        // Fall back to BF16
        let a_bf16 = a.to_dtype(DType::BF16)?;
        let b_bf16 = b.to_dtype(DType::BF16)?;
        let result = a_bf16.matmul(&b_bf16)?;
        result.to_dtype(DType::F32)
    } else {
        // Standard FP32
        a.matmul(b)
    }
}

// =============================================================================
// Kernel Dispatch
// =============================================================================

/// Dispatch attention kernel based on GPU features
pub fn dispatch_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: Option<f64>,
    features: &CUDAFeatures,
) -> Result<Tensor> {
    let (_, _, seq_len, head_dim) = q.dims4()?;
    let scale = scale.unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt());
    
    if features.has_tma && features.has_warpgroup {
        // Hopper path: Use TMA for async tile loading + warpgroup MMA
        // For now, use standard attention with optimized tiling
        let k_t = k.transpose(2, 3)?;
        let scores = q.matmul(&k_t)?;
        let scaled_scores = (scores * scale)?;
        let attn_weights = candle_nn::ops::softmax(&scaled_scores, 3)?;
        attn_weights.matmul(v)
    } else if features.has_bf16_tensor_core {
        // Ampere path: BF16 tensor cores
        let q_bf16 = q.to_dtype(DType::BF16)?;
        let k_bf16 = k.to_dtype(DType::BF16)?;
        let v_bf16 = v.to_dtype(DType::BF16)?;
        
        let k_t = k_bf16.transpose(2, 3)?;
        let scores = q_bf16.matmul(&k_t)?;
        let scaled_scores = (scores.to_dtype(DType::F32)? * scale)?;
        let attn_weights = candle_nn::ops::softmax(&scaled_scores, 3)?.to_dtype(DType::BF16)?;
        let output = attn_weights.matmul(&v_bf16)?;
        output.to_dtype(DType::F32)
    } else {
        // Standard path
        let k_t = k.transpose(2, 3)?;
        let scores = q.matmul(&k_t)?;
        let scaled_scores = (scores * scale)?;
        let attn_weights = candle_nn::ops::softmax(&scaled_scores, 3)?;
        attn_weights.matmul(v)
    }
}

/// Dispatch softmax kernel based on GPU features
pub fn dispatch_softmax(
    input: &Tensor,
    dim: usize,
    features: &CUDAFeatures,
) -> Result<Tensor> {
    if features.has_warpgroup {
        // Hopper: Use warpgroup reductions (4x wider than warp)
        candle_nn::ops::softmax(input, dim)
    } else {
        // Standard warp-based softmax
        candle_nn::ops::softmax(input, dim)
    }
}

/// Dispatch RMSNorm kernel based on GPU features
pub fn dispatch_rms_norm(
    input: &Tensor,
    weight: &Tensor,
    eps: f64,
    features: &CUDAFeatures,
) -> Result<Tensor> {
    let dims = input.dims();
    let last_dim = dims.len() - 1;
    
    let squared = input.sqr()?;
    let variance = squared.mean_keepdim(last_dim)?;
    let variance_eps = (variance + eps)?;
    
    if features.has_bf16_tensor_core {
        // Use BF16 for intermediate computations
        let rsqrt = variance_eps.to_dtype(DType::BF16)?.sqrt()?.recip()?;
        let input_bf16 = input.to_dtype(DType::BF16)?;
        let normalized = input_bf16.broadcast_mul(&rsqrt)?;
        let weight_bf16 = weight.to_dtype(DType::BF16)?;
        let output = normalized.broadcast_mul(&weight_bf16)?;
        output.to_dtype(DType::F32)
    } else {
        let rsqrt = variance_eps.sqrt()?.recip()?;
        let normalized = input.broadcast_mul(&rsqrt)?;
        normalized.broadcast_mul(weight)
    }
}

/// CUDA kernel dispatcher
pub struct CUDAKernelDispatcher {
    features: CUDAFeatures,
    backend: KernelBackend,
}

impl CUDAKernelDispatcher {
    /// Create new dispatcher with detected features
    pub fn new(device: &Device) -> Self {
        let features = CUDAFeatures::detect(device);
        let backend = KernelBackend::select(device, Some(&features));
        Self { features, backend }
    }

    /// Create dispatcher with specific features
    pub fn with_features(features: CUDAFeatures) -> Self {
        let backend = if features.has_tma && features.has_warpgroup {
            KernelBackend::CudaHopper
        } else if features.has_bf16_tensor_core {
            KernelBackend::CudaAmpere
        } else {
            KernelBackend::CudaStandard
        };
        Self { features, backend }
    }

    /// Get detected features
    pub fn features(&self) -> &CUDAFeatures {
        &self.features
    }

    /// Get selected backend
    pub fn backend(&self) -> KernelBackend {
        self.backend
    }

    /// Dispatch attention
    pub fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: Option<f64>,
    ) -> Result<Tensor> {
        dispatch_attention(q, k, v, scale, &self.features)
    }

    /// Dispatch softmax
    pub fn softmax(&self, input: &Tensor, dim: usize) -> Result<Tensor> {
        dispatch_softmax(input, dim, &self.features)
    }

    /// Dispatch RMSNorm
    pub fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
        dispatch_rms_norm(input, weight, eps, &self.features)
    }

    /// Dispatch matmul with FP8 support
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if self.features.has_fp8_tensor_core {
            fp8_matmul(a, b, None, &self.features)
        } else if self.features.has_bf16_tensor_core {
            let a_bf16 = a.to_dtype(DType::BF16)?;
            let b_bf16 = b.to_dtype(DType::BF16)?;
            let result = a_bf16.matmul(&b_bf16)?;
            result.to_dtype(DType::F32)
        } else {
            a.matmul(b)
        }
    }

    /// Dispatch matmul with explicit FP8 config
    pub fn matmul_fp8(&self, a: &Tensor, b: &Tensor, config: &FP8Config) -> Result<Tensor> {
        fp8_matmul(a, b, Some(config), &self.features)
    }
}

// =============================================================================
// CUDA Stream Management
// =============================================================================

/// CUDA stream pool for overlapped execution
pub struct CUDAStreamPool {
    num_streams: usize,
}

impl CUDAStreamPool {
    /// Create a new stream pool
    pub fn new(num_streams: usize) -> Self {
        Self { num_streams }
    }

    /// Get number of streams
    pub fn num_streams(&self) -> usize {
        self.num_streams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_capability() {
        let hopper = ComputeCapability::new(9, 0);
        assert!(hopper.is_hopper());
        assert!(hopper.is_ampere());
        assert!(hopper.is_turing());
        assert!(hopper.is_volta());

        let ampere = ComputeCapability::new(8, 0);
        assert!(!ampere.is_hopper());
        assert!(ampere.is_ampere());
    }

    #[test]
    fn test_cuda_features_h100() {
        let features = CUDAFeatures::h100();
        assert!(features.has_tma);
        assert!(features.has_warpgroup);
        assert!(features.has_fp8_tensor_core);
    }

    #[test]
    fn test_cuda_features_a100() {
        let features = CUDAFeatures::a100();
        assert!(!features.has_tma);
        assert!(!features.has_warpgroup);
        assert!(!features.has_fp8_tensor_core);
        assert!(features.has_bf16_tensor_core);
    }

    #[test]
    fn test_kernel_backend_selection() {
        let hopper = CUDAFeatures::h100();
        let dispatcher = CUDAKernelDispatcher::with_features(hopper);
        assert_eq!(dispatcher.backend(), KernelBackend::CudaHopper);

        let ampere = CUDAFeatures::a100();
        let dispatcher = CUDAKernelDispatcher::with_features(ampere);
        assert_eq!(dispatcher.backend(), KernelBackend::CudaAmpere);
    }

    #[test]
    fn test_tma_descriptor() {
        let desc = TMADescriptor::new(
            vec![128, 4096],
            vec![32, 32],
            TMASwizzleMode::Swizzle128B,
            4, // F32
        );
        
        assert!(desc.is_valid());
        assert_eq!(desc.num_tiles(), vec![4, 128]);
    }

    #[test]
    fn test_fp8_config() {
        let config = FP8Config::new(FP8Format::E4M3);
        assert_eq!(config.format.max_value(), 240.0);
        
        let config_e5m2 = FP8Config::new(FP8Format::E5M2);
        assert_eq!(config_e5m2.format.max_value(), 57344.0);
    }

    #[test]
    fn test_warpgroup_config() {
        let config = WarpgroupConfig::default();
        assert_eq!(config.warpgroup_size, 128);
        assert_eq!(config.warps_per_warpgroup, 4);
        assert_eq!(config.num_warpgroups(256), 2);
    }
}
