//! Unified Device Selection for DeepSeek Rust
//!
//! This module provides a centralized device selection mechanism with
//! CUDA → Metal → CPU priority chain.
//!
//! # Features
//! - Automatic device detection with consistent priority (CUDA first)
//! - Configurable device selection via environment variables
//! - Device information and availability checking
//!
//! # Example
//! ```rust,ignore
//! use deepseek_rust::utils::device::{DeviceSelector, DeviceConfig};
//!
//! // Get best available device (CUDA → Metal → CPU)
//! let device = DeviceSelector::get_device()?;
//!
//! // Or with custom configuration
//! let config = DeviceConfig::from_env();
//! let device = DeviceSelector::get_device_with_config(&config)?;
//! ```

use candle_core::{Device, Result};
use std::collections::HashMap;
use std::env;
use tracing::info;

/// Device selection priority
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePriority {
    /// CUDA → Metal → CPU (recommended for training)
    CudaFirst,
    /// Metal → CUDA → CPU (for Apple Silicon optimization)
    MetalFirst,
    /// Always use CPU
    CpuOnly,
}

impl Default for DevicePriority {
    fn default() -> Self {
        Self::CudaFirst
    }
}

/// Configuration for device selection
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// Device selection priority
    pub priority: DevicePriority,
    /// CUDA device ID (for multi-GPU systems)
    pub cuda_device_id: usize,
    /// Metal device ID
    pub metal_device_id: usize,
    /// Minimum batch size for auto-adjustment
    pub min_batch_size: usize,
    /// Maximum batch size for auto-adjustment
    pub max_batch_size: usize,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            priority: DevicePriority::CudaFirst,
            cuda_device_id: 0,
            metal_device_id: 0,
            min_batch_size: 1,
            max_batch_size: 256,
        }
    }
}

impl DeviceConfig {
    /// Create configuration from environment variables
    ///
    /// Environment variables:
    /// - `DEEPSEEK_DEVICE_PRIORITY`: "cuda", "metal", or "cpu"
    /// - `DEEPSEEK_CUDA_DEVICE`: CUDA device ID (default: 0)
    /// - `DEEPSEEK_METAL_DEVICE`: Metal device ID (default: 0)
    /// - `DEEPSEEK_MIN_BATCH_SIZE`: Minimum batch size (default: 1)
    /// - `DEEPSEEK_MAX_BATCH_SIZE`: Maximum batch size (default: 256)
    pub fn from_env() -> Self {
        let priority = match env::var("DEEPSEEK_DEVICE_PRIORITY")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "metal" => DevicePriority::MetalFirst,
            "cpu" => DevicePriority::CpuOnly,
            _ => DevicePriority::CudaFirst, // Default
        };

        let cuda_device_id = env::var("DEEPSEEK_CUDA_DEVICE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let metal_device_id = env::var("DEEPSEEK_METAL_DEVICE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let min_batch_size = env::var("DEEPSEEK_MIN_BATCH_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let max_batch_size = env::var("DEEPSEEK_MAX_BATCH_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);

        Self {
            priority,
            cuda_device_id,
            metal_device_id,
            min_batch_size,
            max_batch_size,
        }
    }

    /// Get the priority order as a list of device type names
    pub fn get_priority_order(&self) -> Vec<&'static str> {
        match self.priority {
            DevicePriority::CudaFirst => vec!["cuda", "metal", "cpu"],
            DevicePriority::MetalFirst => vec!["metal", "cuda", "cpu"],
            DevicePriority::CpuOnly => vec!["cpu"],
        }
    }
}

/// Centralized device selection with CUDA → Metal → CPU priority
pub struct DeviceSelector;

impl DeviceSelector {
    /// Get the best available device using default configuration (CUDA first)
    ///
    /// Priority: CUDA → Metal → CPU
    ///
    /// # Returns
    /// The best available device
    ///
    /// # Errors
    /// Returns error if device initialization fails (rare, usually GPU driver issues)
    pub fn get_device() -> Result<Device> {
        Self::get_device_with_config(&DeviceConfig::default())
    }

    /// Get device with custom configuration
    pub fn get_device_with_config(config: &DeviceConfig) -> Result<Device> {
        match config.priority {
            DevicePriority::CudaFirst => Self::get_cuda_first(config),
            DevicePriority::MetalFirst => Self::get_metal_first(config),
            DevicePriority::CpuOnly => {
                info!("Using CPU (configured)");
                Ok(Device::Cpu)
            }
        }
    }

    /// CUDA → Metal → CPU priority chain
    fn get_cuda_first(config: &DeviceConfig) -> Result<Device> {
        // Try CUDA first
        if candle_core::utils::cuda_is_available() {
            info!("Using CUDA GPU (device {})", config.cuda_device_id);
            return Device::new_cuda(config.cuda_device_id);
        }

        // Fallback to Metal
        if candle_core::utils::metal_is_available() {
            info!("Using Metal GPU (device {}) - CUDA not available", config.metal_device_id);
            return Device::new_metal(config.metal_device_id);
        }

        // Final fallback to CPU
        info!("Using CPU - no GPU available");
        Ok(Device::Cpu)
    }

    /// Metal → CUDA → CPU priority chain
    fn get_metal_first(config: &DeviceConfig) -> Result<Device> {
        // Try Metal first
        if candle_core::utils::metal_is_available() {
            info!("Using Metal GPU (device {})", config.metal_device_id);
            return Device::new_metal(config.metal_device_id);
        }

        // Fallback to CUDA
        if candle_core::utils::cuda_is_available() {
            info!("Using CUDA GPU (device {}) - Metal not available", config.cuda_device_id);
            return Device::new_cuda(config.cuda_device_id);
        }

        // Final fallback to CPU
        info!("Using CPU - no GPU available");
        Ok(Device::Cpu)
    }

    /// Check if CUDA is available
    pub fn is_cuda_available() -> bool {
        candle_core::utils::cuda_is_available()
    }

    /// Check if Metal is available
    pub fn is_metal_available() -> bool {
        candle_core::utils::metal_is_available()
    }

    /// Get device information as a HashMap
    pub fn get_device_info() -> HashMap<String, String> {
        let mut info = HashMap::new();
        
        info.insert(
            "cuda_available".to_string(),
            Self::is_cuda_available().to_string(),
        );
        info.insert(
            "metal_available".to_string(),
            Self::is_metal_available().to_string(),
        );

        // Determine selected device
        let selected = if Self::is_cuda_available() {
            "cuda"
        } else if Self::is_metal_available() {
            "metal"
        } else {
            "cpu"
        };
        info.insert("selected_device".to_string(), selected.to_string());

        info
    }

    /// Get string representation of device type
    pub fn device_type_string(device: &Device) -> &'static str {
        match device {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Metal(_) => "metal",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DeviceConfig::default();
        assert_eq!(config.priority, DevicePriority::CudaFirst);
        assert_eq!(config.cuda_device_id, 0);
        assert_eq!(config.metal_device_id, 0);
        assert_eq!(config.min_batch_size, 1);
        assert_eq!(config.max_batch_size, 256);
    }

    #[test]
    fn test_priority_order_cuda_first() {
        let config = DeviceConfig {
            priority: DevicePriority::CudaFirst,
            ..Default::default()
        };
        assert_eq!(config.get_priority_order(), vec!["cuda", "metal", "cpu"]);
    }

    #[test]
    fn test_priority_order_metal_first() {
        let config = DeviceConfig {
            priority: DevicePriority::MetalFirst,
            ..Default::default()
        };
        assert_eq!(config.get_priority_order(), vec!["metal", "cuda", "cpu"]);
    }

    #[test]
    fn test_priority_order_cpu_only() {
        let config = DeviceConfig {
            priority: DevicePriority::CpuOnly,
            ..Default::default()
        };
        assert_eq!(config.get_priority_order(), vec!["cpu"]);
    }

    #[test]
    fn test_get_device_returns_valid() {
        let device = DeviceSelector::get_device().expect("Should get device");
        // Verify device is usable
        let _tensor = candle_core::Tensor::zeros((2, 2), candle_core::DType::F32, &device)
            .expect("Should create tensor");
    }

    #[test]
    fn test_device_info_contains_required_keys() {
        let info = DeviceSelector::get_device_info();
        assert!(info.contains_key("cuda_available"));
        assert!(info.contains_key("metal_available"));
        assert!(info.contains_key("selected_device"));
    }

    #[test]
    fn test_device_type_string() {
        assert_eq!(DeviceSelector::device_type_string(&Device::Cpu), "cpu");
    }
}
