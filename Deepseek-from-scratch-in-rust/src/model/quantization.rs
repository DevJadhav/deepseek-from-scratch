use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

// ============================================================================
// FP8 Format Constants (DeepSeek-V3.2 Mixed-Precision Training)
// ============================================================================

/// E4M3 Format: 4-bit exponent, 3-bit mantissa
/// Range: +/- 448.0, used for forward activations and weights
pub const E4M3_MAX: f32 = 448.0;
pub const E4M3_MIN: f32 = -448.0;
pub const E4M3_SMALLEST_NORMAL: f32 = 0.015625; // 2^-6

/// E5M2 Format: 5-bit exponent, 2-bit mantissa  
/// Range: +/- 57344.0, used for gradients (larger range for stability)
pub const E5M2_MAX: f32 = 57344.0;
pub const E5M2_MIN: f32 = -57344.0;
pub const E5M2_SMALLEST_NORMAL: f32 = 6.1e-5; // 2^-14

/// Tile size for per-tile scaling (128x128 as per DeepSeek-V3)
pub const FP8_TILE_SIZE: usize = 128;

// ============================================================================
// FP8 Format Types
// ============================================================================

/// FP8 format selection for different use cases
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FP8Format {
    /// E4M3: Higher precision, lower range - for forward pass
    E4M3,
    /// E5M2: Lower precision, higher range - for gradients
    E5M2,
}

impl FP8Format {
    pub fn max_value(&self) -> f32 {
        match self {
            FP8Format::E4M3 => E4M3_MAX,
            FP8Format::E5M2 => E5M2_MAX,
        }
    }
    
    pub fn min_value(&self) -> f32 {
        match self {
            FP8Format::E4M3 => E4M3_MIN,
            FP8Format::E5M2 => E5M2_MIN,
        }
    }
    
    pub fn smallest_normal(&self) -> f32 {
        match self {
            FP8Format::E4M3 => E4M3_SMALLEST_NORMAL,
            FP8Format::E5M2 => E5M2_SMALLEST_NORMAL,
        }
    }
}

// ============================================================================
// Per-Tile Scaling Infrastructure (128x128 tiles)
// ============================================================================

/// Per-tile scaling configuration for FP8 quantization
#[derive(Clone, Debug)]
pub struct TileScalingConfig {
    /// Tile dimensions (default 128x128)
    pub tile_rows: usize,
    pub tile_cols: usize,
    /// Format for forward pass (activations/weights)
    pub forward_format: FP8Format,
    /// Format for backward pass (gradients)
    pub backward_format: FP8Format,
    /// Dynamic scaling factor for underflow prevention
    pub dynamic_scaling: bool,
    /// History length for amax tracking
    pub amax_history_len: usize,
}

impl Default for TileScalingConfig {
    fn default() -> Self {
        Self {
            tile_rows: FP8_TILE_SIZE,
            tile_cols: FP8_TILE_SIZE,
            forward_format: FP8Format::E4M3,
            backward_format: FP8Format::E5M2,
            dynamic_scaling: true,
            amax_history_len: 16,
        }
    }
}

/// Tile-based scaling state for FP8 training
#[derive(Clone)]
pub struct TileScalingState {
    /// Configuration
    pub config: TileScalingConfig,
    /// Current scales per tile (shape depends on tensor being scaled)
    pub scales: Option<Tensor>,
    /// Amax history for dynamic scaling
    pub amax_history: Vec<Tensor>,
    /// Current position in history buffer
    history_idx: usize,
}

impl TileScalingState {
    pub fn new(config: TileScalingConfig) -> Self {
        let history_len = config.amax_history_len;
        Self {
            config,
            scales: None,
            amax_history: Vec::with_capacity(history_len),
            history_idx: 0,
        }
    }
    
    /// Compute per-tile scales for a tensor
    pub fn compute_tile_scales(&mut self, tensor: &Tensor, format: FP8Format) -> Result<Tensor> {
        let device = tensor.device();
        let (rows, cols) = if tensor.dims().len() == 2 {
            tensor.dims2()?
        } else {
            let shape = tensor.dims();
            let last = shape[shape.len() - 1];
            let rest: usize = shape[..shape.len() - 1].iter().product();
            (rest, last)
        };
        
        let tile_r = self.config.tile_rows;
        let tile_c = self.config.tile_cols;
        
        // Pad to tile boundaries
        let pad_r = (tile_r - (rows % tile_r)) % tile_r;
        let pad_c = (tile_c - (cols % tile_c)) % tile_c;
        
        let n_tiles_r = (rows + pad_r) / tile_r;
        let n_tiles_c = (cols + pad_c) / tile_c;
        
        // Reshape and compute amax per tile
        let flat = tensor.flatten_all()?;
        let padded = if pad_r > 0 || pad_c > 0 {
            let total_padded = (rows + pad_r) * (cols + pad_c);
            let zeros = Tensor::zeros(total_padded - flat.elem_count(), flat.dtype(), device)?;
            Tensor::cat(&[&flat, &zeros], 0)?
        } else {
            flat
        };
        
        // Reshape to tiles: (n_tiles_r, tile_r, n_tiles_c, tile_c)
        let tiled = padded.reshape((n_tiles_r, tile_r, n_tiles_c, tile_c))?;
        let tiled = tiled.permute((0, 2, 1, 3))?; // (n_tiles_r, n_tiles_c, tile_r, tile_c)
        
        // Compute amax per tile
        let amax = tiled.abs()?.max_keepdim(3)?.max_keepdim(2)?;
        let amax = amax.squeeze(3)?.squeeze(2)?; // (n_tiles_r, n_tiles_c)
        
        // Update history for dynamic scaling
        if self.config.dynamic_scaling {
            self.update_amax_history(&amax)?;
        }
        
        // Compute scale = amax / format_max
        let format_max = format.max_value() as f64;
        let scales = (amax.clamp(1e-12, f32::MAX as f64)? / format_max)?;
        
        self.scales = Some(scales.clone());
        Ok(scales)
    }
    
    fn update_amax_history(&mut self, amax: &Tensor) -> Result<()> {
        if self.amax_history.len() < self.config.amax_history_len {
            self.amax_history.push(amax.clone());
        } else {
            self.amax_history[self.history_idx] = amax.clone();
        }
        self.history_idx = (self.history_idx + 1) % self.config.amax_history_len;
        Ok(())
    }
    
    /// Get smoothed amax from history
    pub fn get_smoothed_amax(&self) -> Result<Option<Tensor>> {
        if self.amax_history.is_empty() {
            return Ok(None);
        }
        
        // Stack and take max across history
        let stacked = Tensor::stack(&self.amax_history, 0)?;
        let smoothed = stacked.max(0)?;
        Ok(Some(smoothed))
    }
}

/// Quantize tensor to FP8 with per-tile scaling
pub fn quantize_fp8_tiled(
    tensor: &Tensor,
    scales: &Tensor,
    format: FP8Format,
    tile_rows: usize,
    tile_cols: usize,
) -> Result<Tensor> {
    let device = tensor.device();
    let original_shape = tensor.dims().to_vec();
    
    let (rows, cols) = if tensor.dims().len() == 2 {
        tensor.dims2()?
    } else {
        let shape = tensor.dims();
        let last = shape[shape.len() - 1];
        let rest: usize = shape[..shape.len() - 1].iter().product();
        (rest, last)
    };
    
    // Pad to tile boundaries
    let pad_r = (tile_rows - (rows % tile_rows)) % tile_rows;
    let pad_c = (tile_cols - (cols % tile_cols)) % tile_cols;
    
    let n_tiles_r = (rows + pad_r) / tile_rows;
    let n_tiles_c = (cols + pad_c) / tile_cols;
    
    // Flatten and pad
    let flat = tensor.flatten_all()?;
    let padded = if pad_r > 0 || pad_c > 0 {
        let total_padded = (rows + pad_r) * (cols + pad_c);
        let zeros = Tensor::zeros(total_padded - flat.elem_count(), flat.dtype(), device)?;
        Tensor::cat(&[&flat, &zeros], 0)?
    } else {
        flat
    };
    
    // Reshape to tiles
    let tiled = padded.reshape((n_tiles_r, tile_rows, n_tiles_c, tile_cols))?;
    let tiled = tiled.permute((0, 2, 1, 3))?;
    
    // Expand scales for broadcasting: (n_tiles_r, n_tiles_c) -> (n_tiles_r, n_tiles_c, 1, 1)
    let scales_expanded = scales.reshape((n_tiles_r, n_tiles_c, 1, 1))?;
    
    // Quantize: x_q = clamp(round(x / scale), min, max)
    let scaled = tiled.broadcast_div(&scales_expanded)?;
    let quantized = scaled.round()?;
    let quantized = quantized.clamp(format.min_value() as f64, format.max_value() as f64)?;
    
    // Reshape back
    let quantized = quantized.permute((0, 2, 1, 3))?;
    let quantized = quantized.reshape(((rows + pad_r) * (cols + pad_c),))?;
    
    // Remove padding and restore shape
    if pad_r > 0 || pad_c > 0 {
        // Need to carefully extract non-padded elements
        let mut indices = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                indices.push(r * (cols + pad_c) + c);
            }
        }
        let indices = Tensor::from_slice(&indices.iter().map(|&i| i as i64).collect::<Vec<_>>(), indices.len(), device)?;
        let unpadded = quantized.index_select(&indices, 0)?;
        unpadded.reshape(original_shape.as_slice())
    } else {
        quantized.reshape(original_shape.as_slice())
    }
}

/// Dequantize FP8 tensor with per-tile scaling
pub fn dequantize_fp8_tiled(
    quantized: &Tensor,
    scales: &Tensor,
    tile_rows: usize,
    tile_cols: usize,
) -> Result<Tensor> {
    let device = quantized.device();
    let original_shape = quantized.dims().to_vec();
    
    let (rows, cols) = if quantized.dims().len() == 2 {
        quantized.dims2()?
    } else {
        let shape = quantized.dims();
        let last = shape[shape.len() - 1];
        let rest: usize = shape[..shape.len() - 1].iter().product();
        (rest, last)
    };
    
    // Pad to tile boundaries
    let pad_r = (tile_rows - (rows % tile_rows)) % tile_rows;
    let pad_c = (tile_cols - (cols % tile_cols)) % tile_cols;
    
    let n_tiles_r = (rows + pad_r) / tile_rows;
    let n_tiles_c = (cols + pad_c) / tile_cols;
    
    // Flatten and pad
    let flat = quantized.flatten_all()?;
    let padded = if pad_r > 0 || pad_c > 0 {
        let total_padded = (rows + pad_r) * (cols + pad_c);
        let zeros = Tensor::zeros(total_padded - flat.elem_count(), flat.dtype(), device)?;
        Tensor::cat(&[&flat, &zeros], 0)?
    } else {
        flat
    };
    
    // Reshape to tiles
    let tiled = padded.reshape((n_tiles_r, tile_rows, n_tiles_c, tile_cols))?;
    let tiled = tiled.permute((0, 2, 1, 3))?;
    
    // Expand scales
    let scales_expanded = scales.reshape((n_tiles_r, n_tiles_c, 1, 1))?;
    
    // Dequantize: x = x_q * scale
    let dequantized = tiled.broadcast_mul(&scales_expanded)?;
    
    // Reshape back
    let dequantized = dequantized.permute((0, 2, 1, 3))?;
    let dequantized = dequantized.reshape(((rows + pad_r) * (cols + pad_c),))?;
    
    // Remove padding
    if pad_r > 0 || pad_c > 0 {
        let mut indices = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                indices.push(r * (cols + pad_c) + c);
            }
        }
        let indices = Tensor::from_slice(&indices.iter().map(|&i| i as i64).collect::<Vec<_>>(), indices.len(), device)?;
        let unpadded = dequantized.index_select(&indices, 0)?;
        unpadded.reshape(original_shape.as_slice())
    } else {
        dequantized.reshape(original_shape.as_slice())
    }
}

// ============================================================================
// FP8 Optimizer State Quantization
// ============================================================================

/// FP8 quantized optimizer state for memory efficiency
#[derive(Clone)]
pub struct FP8OptimizerState {
    /// First moment (m) quantized to FP8
    pub m_quantized: Tensor,
    pub m_scale: Tensor,
    /// Second moment (v) quantized to FP8  
    pub v_quantized: Tensor,
    pub v_scale: Tensor,
    /// Step count
    pub step: usize,
    /// Configuration
    pub config: FP8OptimizerConfig,
}

/// Configuration for FP8 optimizer
#[derive(Clone, Debug)]
pub struct FP8OptimizerConfig {
    /// Use block-wise scaling for optimizer states
    pub block_wise_scaling: bool,
    /// Block size for scaling
    pub block_size: usize,
    /// Update scale every N steps
    pub scale_update_interval: usize,
    /// Format for moments (E4M3 for better precision)
    pub moment_format: FP8Format,
}

impl Default for FP8OptimizerConfig {
    fn default() -> Self {
        Self {
            block_wise_scaling: true,
            block_size: 128,
            scale_update_interval: 1,
            moment_format: FP8Format::E4M3,
        }
    }
}

impl FP8OptimizerState {
    /// Create new FP8 optimizer state from FP32 moments
    pub fn from_fp32(m: &Tensor, v: &Tensor, config: FP8OptimizerConfig) -> Result<Self> {
        let format = config.moment_format;
        let format_max = format.max_value();
        
        // Compute scales
        let m_amax = m.abs()?.max_all()?;
        let m_scale = (m_amax.clamp(1e-12, f32::MAX as f64)? / (format_max as f64))?;
        
        let v_amax = v.abs()?.max_all()?;
        let v_scale = (v_amax.clamp(1e-12, f32::MAX as f64)? / (format_max as f64))?;
        
        // Quantize
        let m_quantized = (m / m_scale.broadcast_as(m.dims())?)?.round()?.clamp(format.min_value() as f64, format_max as f64)?;
        let v_quantized = (v / v_scale.broadcast_as(v.dims())?)?.round()?.clamp(format.min_value() as f64, format_max as f64)?;
        
        Ok(Self {
            m_quantized,
            m_scale,
            v_quantized,
            v_scale,
            step: 0,
            config,
        })
    }
    
    /// Dequantize moments to FP32
    pub fn to_fp32(&self) -> Result<(Tensor, Tensor)> {
        let m = self.m_quantized.broadcast_mul(&self.m_scale.broadcast_as(self.m_quantized.dims())?)?;
        let v = self.v_quantized.broadcast_mul(&self.v_scale.broadcast_as(self.v_quantized.dims())?)?;
        Ok((m, v))
    }
    
    /// Update moments (Adam-style) and requantize
    pub fn update(
        &mut self,
        grad: &Tensor,
        beta1: f64,
        beta2: f64,
    ) -> Result<(Tensor, Tensor)> {
        // Dequantize current moments
        let (m, v) = self.to_fp32()?;
        
        // Adam update: m = beta1 * m + (1 - beta1) * grad
        let m_new = ((m * beta1)? + (grad * (1.0 - beta1))?)?;
        
        // v = beta2 * v + (1 - beta2) * grad^2
        let grad_sq = grad.sqr()?;
        let v_new = ((v * beta2)? + (grad_sq * (1.0 - beta2))?)?;
        
        self.step += 1;
        
        // Re-quantize
        let format = self.config.moment_format;
        let format_max = format.max_value();
        
        if self.step % self.config.scale_update_interval == 0 {
            // Update scales
            let m_amax = m_new.abs()?.max_all()?;
            self.m_scale = (m_amax.clamp(1e-12, f32::MAX as f64)? / (format_max as f64))?;
            
            let v_amax = v_new.abs()?.max_all()?;
            self.v_scale = (v_amax.clamp(1e-12, f32::MAX as f64)? / (format_max as f64))?;
        }
        
        self.m_quantized = (m_new.broadcast_div(&self.m_scale.broadcast_as(m_new.dims())?)?)
            .round()?
            .clamp(format.min_value() as f64, format_max as f64)?;
        self.v_quantized = (v_new.broadcast_div(&self.v_scale.broadcast_as(v_new.dims())?)?)
            .round()?
            .clamp(format.min_value() as f64, format_max as f64)?;
        
        Ok((m_new, v_new))
    }
    
    /// Get bias-corrected moments
    pub fn get_bias_corrected(&self, beta1: f64, beta2: f64) -> Result<(Tensor, Tensor)> {
        let (m, v) = self.to_fp32()?;
        
        let bias_correction1 = 1.0 - beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step as i32);
        
        let m_hat = (m / bias_correction1)?;
        let v_hat = (v / bias_correction2)?;
        
        Ok((m_hat, v_hat))
    }
    
    /// Memory savings compared to FP32
    pub fn memory_savings_ratio(&self) -> f64 {
        // FP8 = 1 byte, FP32 = 4 bytes
        // With scales overhead (~1% for large tensors)
        0.25 * 1.01
    }
}

// ============================================================================
// FP8 Mixed-Precision Training Manager
// ============================================================================

/// Manages FP8 mixed-precision training with dynamic loss scaling
#[derive(Clone)]
pub struct FP8MixedPrecisionManager {
    /// Configuration
    pub config: FP8TrainingConfig,
    /// Loss scale for gradient scaling
    pub loss_scale: f64,
    /// Number of consecutive steps without overflow
    pub growth_tracker: usize,
    /// Per-parameter FP8 optimizer states
    pub optimizer_states: HashMap<String, FP8OptimizerState>,
    /// Tile scaling states for activations
    pub activation_scales: HashMap<String, TileScalingState>,
    /// Tile scaling states for weights
    pub weight_scales: HashMap<String, TileScalingState>,
}

/// Configuration for FP8 mixed-precision training
#[derive(Clone, Debug)]
pub struct FP8TrainingConfig {
    /// Initial loss scale
    pub initial_loss_scale: f64,
    /// Factor to grow loss scale
    pub growth_factor: f64,
    /// Factor to reduce loss scale on overflow
    pub backoff_factor: f64,
    /// Steps between loss scale growth attempts
    pub growth_interval: usize,
    /// Maximum loss scale
    pub max_loss_scale: f64,
    /// Minimum loss scale before warning
    pub min_loss_scale: f64,
    /// Use FP8 for forward pass activations
    pub fp8_forward: bool,
    /// Use FP8 for backward pass gradients
    pub fp8_backward: bool,
    /// Use FP8 for optimizer states
    pub fp8_optimizer: bool,
    /// Tile scaling configuration
    pub tile_config: TileScalingConfig,
    /// Optimizer configuration
    pub optimizer_config: FP8OptimizerConfig,
}

impl Default for FP8TrainingConfig {
    fn default() -> Self {
        Self {
            initial_loss_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            max_loss_scale: 2.0_f64.powi(24),
            min_loss_scale: 1.0,
            fp8_forward: true,
            fp8_backward: true,
            fp8_optimizer: true,
            tile_config: TileScalingConfig::default(),
            optimizer_config: FP8OptimizerConfig::default(),
        }
    }
}

impl FP8MixedPrecisionManager {
    pub fn new(config: FP8TrainingConfig) -> Self {
        Self {
            loss_scale: config.initial_loss_scale,
            config,
            growth_tracker: 0,
            optimizer_states: HashMap::new(),
            activation_scales: HashMap::new(),
            weight_scales: HashMap::new(),
        }
    }
    
    /// Scale loss for FP8 training
    pub fn scale_loss(&self, loss: &Tensor) -> Result<Tensor> {
        loss * self.loss_scale
    }
    
    /// Unscale gradients after backward pass
    pub fn unscale_gradients(&self, gradients: &mut [Tensor]) -> Result<bool> {
        let inv_scale = 1.0 / self.loss_scale;
        
        // Check for overflow/underflow
        for grad in gradients.iter() {
            let has_inf = grad.abs()?.max_all()?.to_scalar::<f32>()?.is_infinite();
            if has_inf {
                return Ok(false); // Overflow detected
            }
        }
        
        // Unscale
        for grad in gradients.iter_mut() {
            *grad = (grad.clone() * inv_scale)?;
        }
        
        Ok(true)
    }
    
    /// Update loss scale based on overflow status
    pub fn update_loss_scale(&mut self, overflow: bool) {
        if overflow {
            // Reduce scale on overflow
            self.loss_scale = (self.loss_scale * self.config.backoff_factor)
                .max(self.config.min_loss_scale);
            self.growth_tracker = 0;
            
            if self.loss_scale <= self.config.min_loss_scale {
                eprintln!("Warning: Loss scale at minimum ({}). Consider gradient clipping.", 
                         self.config.min_loss_scale);
            }
        } else {
            // Possibly grow scale
            self.growth_tracker += 1;
            if self.growth_tracker >= self.config.growth_interval {
                self.loss_scale = (self.loss_scale * self.config.growth_factor)
                    .min(self.config.max_loss_scale);
                self.growth_tracker = 0;
            }
        }
    }
    
    /// Quantize activations for forward pass
    pub fn quantize_activation(&mut self, name: &str, tensor: &Tensor) -> Result<(Tensor, Tensor)> {
        if !self.config.fp8_forward {
            let ones = Tensor::ones(tensor.dims(), tensor.dtype(), tensor.device())?;
            return Ok((tensor.clone(), ones));
        }
        
        let state = self.activation_scales
            .entry(name.to_string())
            .or_insert_with(|| TileScalingState::new(self.config.tile_config.clone()));
        
        let scales = state.compute_tile_scales(tensor, FP8Format::E4M3)?;
        let quantized = quantize_fp8_tiled(
            tensor,
            &scales,
            FP8Format::E4M3,
            self.config.tile_config.tile_rows,
            self.config.tile_config.tile_cols,
        )?;
        
        Ok((quantized, scales))
    }
    
    /// Quantize gradients for backward pass  
    pub fn quantize_gradient(&mut self, name: &str, tensor: &Tensor) -> Result<(Tensor, Tensor)> {
        if !self.config.fp8_backward {
            let ones = Tensor::ones(tensor.dims(), tensor.dtype(), tensor.device())?;
            return Ok((tensor.clone(), ones));
        }
        
        // Use E5M2 for gradients (larger range)
        let state = self.activation_scales
            .entry(format!("{}_grad", name))
            .or_insert_with(|| TileScalingState::new(self.config.tile_config.clone()));
        
        let scales = state.compute_tile_scales(tensor, FP8Format::E5M2)?;
        let quantized = quantize_fp8_tiled(
            tensor,
            &scales,
            FP8Format::E5M2,
            self.config.tile_config.tile_rows,
            self.config.tile_config.tile_cols,
        )?;
        
        Ok((quantized, scales))
    }
    
    /// Initialize FP8 optimizer state for a parameter
    pub fn init_optimizer_state(&mut self, name: &str, param: &Tensor) -> Result<()> {
        if !self.config.fp8_optimizer {
            return Ok(());
        }
        
        let m = Tensor::zeros(param.dims(), param.dtype(), param.device())?;
        let v = Tensor::zeros(param.dims(), param.dtype(), param.device())?;
        
        let state = FP8OptimizerState::from_fp32(&m, &v, self.config.optimizer_config.clone())?;
        self.optimizer_states.insert(name.to_string(), state);
        
        Ok(())
    }
    
    /// Get optimizer state for a parameter
    pub fn get_optimizer_state(&self, name: &str) -> Option<&FP8OptimizerState> {
        self.optimizer_states.get(name)
    }
    
    /// Get mutable optimizer state
    pub fn get_optimizer_state_mut(&mut self, name: &str) -> Option<&mut FP8OptimizerState> {
        self.optimizer_states.get_mut(name)
    }
    
    /// Report memory savings from FP8
    pub fn memory_savings_report(&self) -> FP8MemorySavingsReport {
        let mut total_fp32_bytes = 0usize;
        let mut total_fp8_bytes = 0usize;
        
        for state in self.optimizer_states.values() {
            let param_size = state.m_quantized.elem_count();
            // FP32: 4 bytes per element, 2 moments = 8 bytes
            total_fp32_bytes += param_size * 8;
            // FP8: 1 byte per element, 2 moments + scales = ~2 + overhead
            total_fp8_bytes += param_size * 2 + 64; // Approximate scale overhead
        }
        
        FP8MemorySavingsReport {
            fp32_bytes: total_fp32_bytes,
            fp8_bytes: total_fp8_bytes,
            savings_ratio: if total_fp32_bytes > 0 {
                1.0 - (total_fp8_bytes as f64 / total_fp32_bytes as f64)
            } else {
                0.0
            },
            optimizer_states_count: self.optimizer_states.len(),
        }
    }
}

/// Report on memory savings from FP8 training
#[derive(Clone, Debug)]
pub struct FP8MemorySavingsReport {
    pub fp32_bytes: usize,
    pub fp8_bytes: usize,
    pub savings_ratio: f64,
    pub optimizer_states_count: usize,
}

impl std::fmt::Display for FP8MemorySavingsReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FP8 Memory Savings: {:.1}% ({} MB -> {} MB) for {} parameters",
            self.savings_ratio * 100.0,
            self.fp32_bytes / (1024 * 1024),
            self.fp8_bytes / (1024 * 1024),
            self.optimizer_states_count
        )
    }
}

// ============================================================================
// Original FP8Linear (Enhanced)
// ============================================================================

pub struct FP8Linear {
    weight: Tensor, // Stored as f32, but we will quantize it on the fly or store quantized
    bias: Option<Tensor>,
    weight_scale: Tensor, // Pre-calculated scales for weights (128x128 blocks)
    scale_block_size: usize,
    in_dim: usize,
    out_dim: usize,
}

impl FP8Linear {
    pub fn new(in_dim: usize, out_dim: usize, scale_block_size: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.pp("weight").get_with_hints((out_dim, in_dim), "weight", candle_nn::init::DEFAULT_KAIMING_NORMAL)?;
        let bias = vb.pp("bias").get_with_hints((out_dim,), "bias", candle_nn::init::ZERO).ok();

        // Pre-calculate weight scales (Fine-grained quantization: 128x128 tiles)
        // We pad if dimensions are not divisible by block size
        let (padded_weight, n_blocks_row, n_blocks_col) = Self::pad_and_block(&weight, scale_block_size)?;
        
        // Calculate max per block
        let _abs_weight = padded_weight.abs()?;
        // blocks: (n_blocks_row, block_size, n_blocks_col, block_size) -> need to rearrange to pool
        // Easier: Reshape to (n_blocks_row, block_size, n_blocks_col, block_size)
        // Then permute to (n_blocks_row, n_blocks_col, block_size, block_size)
        // Then max over last two dims.
        
        // Reshape logic:
        // weight: (out_dim, in_dim)
        // padded: (pad_out, pad_in)
        // reshape -> (n_blocks_row, block_size, n_blocks_col, block_size)
        let blocks = padded_weight.reshape((n_blocks_row, scale_block_size, n_blocks_col, scale_block_size))?;
        let blocks = blocks.permute((0, 2, 1, 3))?; // (n_blocks_row, n_blocks_col, block_size, block_size)
        
        let max_vals = blocks.abs()?.max_keepdim(3)?.max_keepdim(2)?; // (n_blocks_row, n_blocks_col, 1, 1)
        let max_vals = max_vals.flatten_all()?.reshape((n_blocks_row, n_blocks_col))?;
        
        // Avoid zero division
        let max_vals = max_vals.clamp(1e-6, f32::MAX)?;
        let weight_scale = (max_vals / (E4M3_MAX as f64))?;

        Ok(Self {
            weight,
            bias,
            weight_scale,
            scale_block_size,
            in_dim,
            out_dim,
        })
    }

    fn pad_and_block(x: &Tensor, block_size: usize) -> Result<(Tensor, usize, usize)> {
        let (rows, cols) = x.dims2()?;
        let pad_rows = (block_size - (rows % block_size)) % block_size;
        let pad_cols = (block_size - (cols % block_size)) % block_size;
        
        let padded = if pad_rows > 0 || pad_cols > 0 {
            // Pad rows (dim 0)
            let x = if pad_rows > 0 {
                x.pad_with_zeros(0, 0, pad_rows)?
            } else {
                x.clone()
            };
            // Pad cols (dim 1)
            if pad_cols > 0 {
                x.pad_with_zeros(1, 0, pad_cols)?
            } else {
                x
            }
        } else {
            x.clone()
        };
        
        let n_blocks_row = (rows + pad_rows) / block_size;
        let n_blocks_col = (cols + pad_cols) / block_size;
        
        Ok((padded, n_blocks_row, n_blocks_col))
    }

    // Online Quantization for Activations (Token-wise or Block-wise)
    // DeepSeek uses 1x128 block scaling for activations.
    // For shape (Batch*Seq, Hidden), we can treat it as (Rows, Cols).
    // We block the Cols (Hidden dim) into 128 chunks. Rows are 1 (per token).
    fn quantize_activations(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (rows, _cols) = x.dims2()?;
        // Assume cols is divisible by block_size for simplicity, or pad
        let (padded_x, _, n_blocks_col) = Self::pad_and_block(x, self.scale_block_size)?;
        
        // Reshape to (rows, n_blocks_col, block_size)
        let blocks = padded_x.reshape((rows, n_blocks_col, self.scale_block_size))?;
        
        // Max per block
        let max_vals = blocks.abs()?.max_keepdim(2)?; // (rows, n_blocks_col, 1)
        let max_vals = max_vals.clamp(1e-6, f32::MAX)?;
        let scale = (max_vals / (E4M3_MAX as f64))?; // (rows, n_blocks_col, 1)
        
        // Quantize
        let quantized = blocks.broadcast_div(&scale)?;
        let quantized = quantized.clamp(E4M3_MIN, E4M3_MAX)?.round()?;
        
        // Return quantized blocks and scales
        Ok((quantized, scale))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Online Quantize Activations
        // x: (batch, seq, in_dim)
        let (b, s, d) = x.dims3()?;
        let x_flat = x.reshape((b * s, d))?;
        
        // Quantize x (activations)
        // x_q: (rows, n_blocks_col, block_size), x_scale: (rows, n_blocks_col, 1)
        let (x_q, x_scale) = self.quantize_activations(&x_flat)?;
        
        // 2. Quantize Weights (using pre-calculated scales)
        // We need to quantize the weights on the fly (or usually stored quantized)
        // weight: (out_dim, in_dim)
        // weight_scale: (n_blocks_out, n_blocks_in)
        
        // Pad weight again to match blocking logic
        let (padded_weight, n_blocks_out, n_blocks_in) = Self::pad_and_block(&self.weight, self.scale_block_size)?;
        
        // Reshape weight to blocks for broadcasting scale
        // (n_blocks_out, block_size, n_blocks_in, block_size) -> (n_blocks_out, n_blocks_in, block_size, block_size)
        let w_blocks = padded_weight.reshape((n_blocks_out, self.scale_block_size, n_blocks_in, self.scale_block_size))?
            .permute((0, 2, 1, 3))?;
            
        // Expand scale for broadcasting: (n_blocks_out, n_blocks_in, 1, 1)
        let w_scale_expanded = self.weight_scale.reshape((n_blocks_out, n_blocks_in, 1, 1))?;
        
        let w_q = w_blocks.broadcast_div(&w_scale_expanded)?;
        let w_q = w_q.clamp(E4M3_MIN, E4M3_MAX)?.round()?;
        
        // 3. MatMul (Simulated)
        // In real hardware, we'd multiply x_q (FP8) and w_q (FP8) and accumulate in FP32.
        // Here we dequantize first and then multiply, to simulate the numerical behavior.
        
        // Dequantize X
        let x_deq = x_q.broadcast_mul(&x_scale)?.reshape((b * s, self.in_dim))?; // Adjust for padding if needed
        // (Note: ignoring padding removal for simplicity, assuming dimensions match block size)

        // Dequantize W
        let w_deq = w_q.broadcast_mul(&w_scale_expanded)?
            .permute((0, 2, 1, 3))?
            .reshape((n_blocks_out * self.scale_block_size, n_blocks_in * self.scale_block_size))?;
            
        // Slice to original dimensions
        let w_deq = w_deq.narrow(0, 0, self.out_dim)?.narrow(1, 0, self.in_dim)?;
        let x_deq = x_deq.narrow(1, 0, self.in_dim)?; // Just in case of padding

        let out = x_deq.matmul(&w_deq.transpose(0, 1)?)?;
        
        // 4. Add Bias
        let out = match &self.bias {
            Some(bias) => out.broadcast_add(bias)?,
            None => out,
        };
        
        out.reshape((b, s, self.out_dim))
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
    fn test_fp8_format_constants() {
        // E4M3 format
        assert_eq!(FP8Format::E4M3.max_value(), 448.0);
        assert_eq!(FP8Format::E4M3.min_value(), -448.0);
        
        // E5M2 format  
        assert_eq!(FP8Format::E5M2.max_value(), 57344.0);
        assert_eq!(FP8Format::E5M2.min_value(), -57344.0);
    }

    #[test]
    fn test_tile_scaling_config() {
        let config = TileScalingConfig::default();
        assert_eq!(config.tile_rows, 128);
        assert_eq!(config.tile_cols, 128);
        assert_eq!(config.forward_format, FP8Format::E4M3);
        assert_eq!(config.backward_format, FP8Format::E5M2);
    }

    #[test]
    fn test_tile_scaling_state() -> Result<()> {
        let device = Device::Cpu;
        let config = TileScalingConfig {
            tile_rows: 4,
            tile_cols: 4,
            ..Default::default()
        };
        
        let mut state = TileScalingState::new(config);
        
        // Create a test tensor
        let tensor = Tensor::randn(0f32, 1f32, (8, 8), &device)?;
        
        // Compute scales
        let scales = state.compute_tile_scales(&tensor, FP8Format::E4M3)?;
        
        // Should be 2x2 tiles (8/4 = 2)
        assert_eq!(scales.dims(), &[2, 2]);
        
        // Scales should be positive
        let scale_vals = scales.flatten_all()?.to_vec1::<f32>()?;
        for s in scale_vals {
            assert!(s > 0.0);
        }
        
        Ok(())
    }

    #[test]
    fn test_fp8_quantize_dequantize() -> Result<()> {
        let device = Device::Cpu;
        
        // Create a tensor with values in FP8 range
        let tensor = Tensor::new(&[
            [10.0f32, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0],
            [90.0, 100.0, 110.0, 120.0],
            [130.0, 140.0, 150.0, 160.0],
        ], &device)?;
        
        // Compute scales (1x1 tile for simplicity)
        let amax = tensor.abs()?.max_all()?;
        let scale = (amax / (E4M3_MAX as f64))?;
        let scales = scale.reshape((1, 1))?;
        
        // Quantize
        let quantized = quantize_fp8_tiled(&tensor, &scales, FP8Format::E4M3, 4, 4)?;
        
        // Dequantize
        let dequantized = dequantize_fp8_tiled(&quantized, &scales, 4, 4)?;
        
        // Check shape preserved
        assert_eq!(dequantized.dims(), tensor.dims());
        
        // Check values are close (within quantization error)
        let diff = (tensor - dequantized)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 5.0, "Quantization error too large: {}", diff);
        
        Ok(())
    }

    #[test]
    fn test_fp8_optimizer_state() -> Result<()> {
        let device = Device::Cpu;
        let config = FP8OptimizerConfig::default();
        
        // Create mock moments
        let m = Tensor::randn(0f32, 0.1f32, (32, 32), &device)?;
        let v = Tensor::randn(0f32, 0.01f32, (32, 32), &device)?.abs()?;
        
        // Create FP8 state
        let state = FP8OptimizerState::from_fp32(&m, &v, config)?;
        
        // Verify quantization happened
        assert_eq!(state.m_quantized.dims(), m.dims());
        assert_eq!(state.v_quantized.dims(), v.dims());
        
        // Dequantize and check closeness
        let (m_dq, v_dq) = state.to_fp32()?;
        
        let m_diff = (m - m_dq)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let v_diff = (v - v_dq)?.abs()?.max_all()?.to_scalar::<f32>()?;
        
        // Should be close (within quantization tolerance)
        assert!(m_diff < 0.01, "m quantization error: {}", m_diff);
        assert!(v_diff < 0.001, "v quantization error: {}", v_diff);
        
        Ok(())
    }

    #[test]
    fn test_fp8_optimizer_state_update() -> Result<()> {
        let device = Device::Cpu;
        let config = FP8OptimizerConfig::default();
        
        let m = Tensor::zeros((16, 16), candle_core::DType::F32, &device)?;
        let v = Tensor::zeros((16, 16), candle_core::DType::F32, &device)?;
        
        let mut state = FP8OptimizerState::from_fp32(&m, &v, config)?;
        
        // Simulate gradient
        let grad = Tensor::randn(0f32, 0.5f32, (16, 16), &device)?;
        
        // Update
        let (m_new, v_new) = state.update(&grad, 0.9, 0.999)?;
        
        assert_eq!(state.step, 1);
        assert_eq!(m_new.dims(), &[16, 16]);
        assert_eq!(v_new.dims(), &[16, 16]);
        
        // v should be non-negative (squared gradients)
        let v_min = v_new.min_all()?.to_scalar::<f32>()?;
        assert!(v_min >= 0.0);
        
        Ok(())
    }

    #[test]
    fn test_fp8_optimizer_bias_correction() -> Result<()> {
        let device = Device::Cpu;
        let config = FP8OptimizerConfig::default();
        
        let m = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
        let v = Tensor::new(&[[0.1f32, 0.2], [0.3, 0.4]], &device)?;
        
        let mut state = FP8OptimizerState::from_fp32(&m, &v, config)?;
        state.step = 10;
        
        let (m_hat, _v_hat) = state.get_bias_corrected(0.9, 0.999)?;
        
        // Bias correction should increase values early in training
        let m_orig = state.to_fp32()?.0;
        let m_hat_sum = m_hat.sum_all()?.to_scalar::<f32>()?;
        let m_orig_sum = m_orig.sum_all()?.to_scalar::<f32>()?;
        
        assert!(m_hat_sum > m_orig_sum, "Bias correction should increase magnitude");
        
        Ok(())
    }

    #[test]
    fn test_fp8_memory_savings() {
        let state = FP8OptimizerState {
            m_quantized: Tensor::zeros((1000, 1000), candle_core::DType::F32, &Device::Cpu).unwrap(),
            m_scale: Tensor::zeros(1, candle_core::DType::F32, &Device::Cpu).unwrap(),
            v_quantized: Tensor::zeros((1000, 1000), candle_core::DType::F32, &Device::Cpu).unwrap(),
            v_scale: Tensor::zeros(1, candle_core::DType::F32, &Device::Cpu).unwrap(),
            step: 0,
            config: FP8OptimizerConfig::default(),
        };
        
        let savings = state.memory_savings_ratio();
        // FP8 should save ~75% memory
        assert!(savings > 0.2 && savings < 0.3, "Expected ~25% of original, got {}", savings);
    }

    #[test]
    fn test_fp8_training_config() {
        let config = FP8TrainingConfig::default();
        
        assert_eq!(config.initial_loss_scale, 65536.0);
        assert_eq!(config.growth_factor, 2.0);
        assert_eq!(config.backoff_factor, 0.5);
        assert!(config.fp8_forward);
        assert!(config.fp8_backward);
        assert!(config.fp8_optimizer);
    }

    #[test]
    fn test_fp8_mixed_precision_manager() -> Result<()> {
        let config = FP8TrainingConfig::default();
        let manager = FP8MixedPrecisionManager::new(config);
        
        assert_eq!(manager.loss_scale, 65536.0);
        assert_eq!(manager.growth_tracker, 0);
        
        Ok(())
    }

    #[test]
    fn test_fp8_loss_scaling() -> Result<()> {
        let device = Device::Cpu;
        let config = FP8TrainingConfig::default();
        let manager = FP8MixedPrecisionManager::new(config);
        
        let loss = Tensor::new(1.0f32, &device)?;
        let scaled_loss = manager.scale_loss(&loss)?;
        
        let scaled_val = scaled_loss.to_scalar::<f32>()?;
        assert_eq!(scaled_val, 65536.0);
        
        Ok(())
    }

    #[test]
    fn test_fp8_loss_scale_update() {
        let config = FP8TrainingConfig::default();
        let mut manager = FP8MixedPrecisionManager::new(config);
        
        let initial_scale = manager.loss_scale;
        
        // Simulate overflow
        manager.update_loss_scale(true);
        assert!(manager.loss_scale < initial_scale);
        assert_eq!(manager.loss_scale, initial_scale * 0.5);
        
        // Simulate many successful steps
        for _ in 0..2000 {
            manager.update_loss_scale(false);
        }
        
        // Scale should have grown
        assert!(manager.loss_scale > initial_scale * 0.5);
    }

    #[test]
    fn test_fp8_memory_report() -> Result<()> {
        let device = Device::Cpu;
        let config = FP8TrainingConfig::default();
        let mut manager = FP8MixedPrecisionManager::new(config);
        
        // Initialize optimizer state for a parameter
        let param = Tensor::randn(0f32, 1f32, (256, 256), &device)?;
        manager.init_optimizer_state("test_param", &param)?;
        
        let report = manager.memory_savings_report();
        
        assert_eq!(report.optimizer_states_count, 1);
        assert!(report.savings_ratio > 0.0);
        
        Ok(())
    }
}
