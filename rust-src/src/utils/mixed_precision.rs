//! Mixed Precision Training Utilities for DeepSeek Rust
//!
//! This module provides BF16/FP16 mixed precision training support:
//! - Automatic precision selection based on hardware
//! - Type conversion utilities
//! - FP32 accumulation for critical operations
//!
//! # Example
//! ```rust,ignore
//! use deepseek::utils::mixed_precision::{MixedPrecisionConfig, get_optimal_dtype};
//!
//! let config = MixedPrecisionConfig::auto_detect(&device);
//! let compute_dtype = config.compute_dtype;
//! ```

use candle_core::{DType, Device, Result, Tensor};

/// Precision modes for training
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    /// Full FP32 precision
    FP32,
    /// FP16 with loss scaling
    FP16,
    /// BF16 (no loss scaling needed)
    BF16,
    /// Automatic selection based on hardware
    Auto,
}

impl Default for PrecisionMode {
    fn default() -> Self {
        Self::Auto
    }
}

/// Configuration for mixed precision training
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Precision mode
    pub mode: PrecisionMode,
    /// Data type for compute (forward/backward)
    pub compute_dtype: DType,
    /// Data type for parameters storage
    pub param_dtype: DType,
    /// Data type for optimizer states (always FP32)
    pub optimizer_dtype: DType,
    /// Whether to use loss scaling (for FP16)
    pub use_loss_scaling: bool,
    /// Initial loss scale value
    pub init_loss_scale: f64,
    /// Loss scale growth factor
    pub growth_factor: f64,
    /// Loss scale backoff factor
    pub backoff_factor: f64,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            mode: PrecisionMode::Auto,
            compute_dtype: DType::F32,
            param_dtype: DType::F32,
            optimizer_dtype: DType::F32,
            use_loss_scaling: false,
            init_loss_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
        }
    }
}

impl MixedPrecisionConfig {
    /// Create FP32 configuration (no mixed precision)
    pub fn fp32() -> Self {
        Self {
            mode: PrecisionMode::FP32,
            compute_dtype: DType::F32,
            param_dtype: DType::F32,
            optimizer_dtype: DType::F32,
            use_loss_scaling: false,
            ..Default::default()
        }
    }

    /// Create BF16 configuration
    pub fn bf16() -> Self {
        Self {
            mode: PrecisionMode::BF16,
            compute_dtype: DType::BF16,
            param_dtype: DType::BF16,
            optimizer_dtype: DType::F32,
            use_loss_scaling: false,
            ..Default::default()
        }
    }

    /// Create FP16 configuration with loss scaling
    pub fn fp16() -> Self {
        Self {
            mode: PrecisionMode::FP16,
            compute_dtype: DType::F16,
            param_dtype: DType::F16,
            optimizer_dtype: DType::F32,
            use_loss_scaling: true,
            ..Default::default()
        }
    }

    /// Auto-detect optimal configuration for device
    pub fn auto_detect(device: &Device) -> Self {
        match device {
            Device::Cuda(cuda_dev) => {
                // BF16 supported on Ampere+ (SM 80+)
                // For simplicity, default to BF16 on CUDA
                // In production, would query device capabilities
                Self::bf16()
            }
            Device::Metal(_) => {
                // Metal supports both FP16 and BF16
                // FP16 is more widely supported
                Self::fp16()
            }
            Device::Cpu => Self::fp32(),
        }
    }
}

/// Get optimal dtype for current device
pub fn get_optimal_dtype(device: &Device) -> DType {
    MixedPrecisionConfig::auto_detect(device).compute_dtype
}

/// Convert tensor to compute dtype
pub fn to_compute_dtype(tensor: &Tensor, config: &MixedPrecisionConfig) -> Result<Tensor> {
    tensor.to_dtype(config.compute_dtype)
}

/// Convert tensor to FP32 for accumulation
pub fn to_fp32_for_accumulation(tensor: &Tensor) -> Result<Tensor> {
    tensor.to_dtype(DType::F32)
}

/// Convert tensor back to storage dtype
pub fn to_storage_dtype(tensor: &Tensor, config: &MixedPrecisionConfig) -> Result<Tensor> {
    tensor.to_dtype(config.param_dtype)
}

/// Loss scaler for FP16 training
#[derive(Debug)]
pub struct LossScaler {
    /// Current scale value
    scale: f64,
    /// Growth factor
    growth_factor: f64,
    /// Backoff factor
    backoff_factor: f64,
    /// Number of steps since last scale change
    growth_interval: usize,
    /// Steps counter
    steps: usize,
    /// Whether overflow was detected
    overflow_detected: bool,
}

impl LossScaler {
    /// Create a new loss scaler
    pub fn new(init_scale: f64, growth_factor: f64, backoff_factor: f64) -> Self {
        Self {
            scale: init_scale,
            growth_factor,
            backoff_factor,
            growth_interval: 2000,
            steps: 0,
            overflow_detected: false,
        }
    }

    /// Create from config
    pub fn from_config(config: &MixedPrecisionConfig) -> Self {
        Self::new(
            config.init_loss_scale,
            config.growth_factor,
            config.backoff_factor,
        )
    }

    /// Get current scale
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Scale loss before backward
    pub fn scale_loss(&self, loss: &Tensor) -> Result<Tensor> {
        loss.affine(self.scale, 0.0)
    }

    /// Unscale gradients after backward
    pub fn unscale_gradients(&self, grads: &[Tensor]) -> Result<Vec<Tensor>> {
        let inv_scale = 1.0 / self.scale;
        grads.iter().map(|g| g.affine(inv_scale, 0.0)).collect()
    }

    /// Check for overflow in gradients
    pub fn check_overflow(&mut self, grads: &[Tensor]) -> Result<bool> {
        for grad in grads {
            let grad_fp32 = grad.to_dtype(DType::F32)?;
            let sum = grad_fp32.sum_all()?.to_scalar::<f32>()?;
            if sum.is_nan() || sum.is_infinite() {
                self.overflow_detected = true;
                return Ok(true);
            }
        }
        self.overflow_detected = false;
        Ok(false)
    }

    /// Update scale after step
    pub fn update(&mut self) {
        if self.overflow_detected {
            // Reduce scale on overflow
            self.scale *= self.backoff_factor;
            self.steps = 0;
        } else {
            self.steps += 1;
            if self.steps >= self.growth_interval {
                // Increase scale periodically
                self.scale *= self.growth_factor;
                self.steps = 0;
            }
        }
    }
}

/// Mixed precision training context
pub struct MixedPrecisionContext {
    /// Configuration
    pub config: MixedPrecisionConfig,
    /// Loss scaler (for FP16)
    pub scaler: Option<LossScaler>,
}

impl MixedPrecisionContext {
    /// Create a new context
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let scaler = if config.use_loss_scaling {
            Some(LossScaler::from_config(&config))
        } else {
            None
        };

        Self { config, scaler }
    }

    /// Auto-detect and create context for device
    pub fn auto(device: &Device) -> Self {
        Self::new(MixedPrecisionConfig::auto_detect(device))
    }

    /// Convert input to compute dtype
    pub fn prepare_input(&self, tensor: &Tensor) -> Result<Tensor> {
        to_compute_dtype(tensor, &self.config)
    }

    /// Prepare loss for backward (apply scaling if needed)
    pub fn prepare_loss(&self, loss: &Tensor) -> Result<Tensor> {
        if let Some(scaler) = &self.scaler {
            scaler.scale_loss(loss)
        } else {
            Ok(loss.clone())
        }
    }

    /// Get current loss scale
    pub fn loss_scale(&self) -> Option<f64> {
        self.scaler.as_ref().map(|s| s.scale())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_precision_config_fp32() {
        let config = MixedPrecisionConfig::fp32();
        assert_eq!(config.compute_dtype, DType::F32);
        assert!(!config.use_loss_scaling);
    }

    #[test]
    fn test_precision_config_bf16() {
        let config = MixedPrecisionConfig::bf16();
        assert_eq!(config.compute_dtype, DType::BF16);
        assert!(!config.use_loss_scaling);
    }

    #[test]
    fn test_precision_config_fp16() {
        let config = MixedPrecisionConfig::fp16();
        assert_eq!(config.compute_dtype, DType::F16);
        assert!(config.use_loss_scaling);
    }

    #[test]
    fn test_loss_scaler() {
        let mut scaler = LossScaler::new(65536.0, 2.0, 0.5);
        assert_eq!(scaler.scale(), 65536.0);

        // Simulate overflow
        scaler.overflow_detected = true;
        scaler.update();
        assert_eq!(scaler.scale(), 32768.0);
    }
}
