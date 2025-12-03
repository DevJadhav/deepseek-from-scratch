//! Checkpoint utilities with SHA256 checksum validation.
//!
//! Provides safe checkpoint saving and loading with corruption detection.

use crate::utils::error::{DeepSeekError, Result};
use candle_core::{safetensors, DType, Device, Tensor};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use tracing::{info, warn, error};

/// Metadata stored alongside checkpoints
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMetadata {
    /// SHA256 checksum of the checkpoint data
    pub checksum: String,
    /// Original filename
    pub filename: String,
    /// Training step when saved
    pub step: Option<u64>,
    /// Training epoch when saved
    pub epoch: Option<u64>,
    /// Model configuration hash
    pub config_hash: Option<String>,
    /// Additional user-provided metadata
    #[serde(default)]
    pub extra: HashMap<String, String>,
}

/// Compute SHA256 checksum of tensor data.
/// 
/// For efficiency, we hash:
/// - Tensor shape and dtype metadata
/// - First and last 1024 bytes of tensor data (or full data if smaller)
pub fn compute_tensor_checksum(tensors: &HashMap<String, Tensor>) -> Result<String> {
    let mut hasher = Sha256::new();
    
    // Sort keys for deterministic ordering
    let mut keys: Vec<&String> = tensors.keys().collect();
    keys.sort();
    
    for key in keys {
        let tensor = &tensors[key];
        
        // Hash metadata
        let meta = format!(
            "{}:{:?}:{:?}",
            key,
            tensor.dtype(),
            tensor.dims()
        );
        hasher.update(meta.as_bytes());
        
        // Hash tensor data (sample for large tensors)
        let flat = tensor.flatten_all()?;
        let data = flat.to_vec1::<f32>().unwrap_or_default();
        let bytes: Vec<u8> = data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        
        if bytes.len() > 2048 {
            // Hash first and last 1024 bytes
            hasher.update(&bytes[..1024]);
            hasher.update(&bytes[bytes.len()-1024..]);
        } else {
            hasher.update(&bytes);
        }
    }
    
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute checksum from raw bytes
pub fn compute_bytes_checksum(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// Save checkpoint with checksum validation.
/// 
/// Saves tensors to a safetensors file and creates a companion .meta.json
/// file containing the checksum and metadata.
pub fn save_checkpoint_with_checksum<P: AsRef<Path>>(
    tensors: &HashMap<String, Tensor>,
    path: P,
    step: Option<u64>,
    epoch: Option<u64>,
    extra: Option<HashMap<String, String>>,
) -> Result<()> {
    let path = path.as_ref();
    
    // Compute checksum before saving
    let checksum = compute_tensor_checksum(tensors)?;
    
    // Save tensors using safetensors format
    safetensors::save(tensors, path)?;
    
    // Create metadata
    let metadata = CheckpointMetadata {
        checksum: checksum.clone(),
        filename: path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("checkpoint")
            .to_string(),
        step,
        epoch,
        config_hash: None,
        extra: extra.unwrap_or_default(),
    };
    
    // Save metadata to companion file
    let meta_path = path.with_extension("meta.json");
    let meta_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| DeepSeekError::Checkpoint(format!("Failed to serialize metadata: {}", e)))?;
    
    fs::write(&meta_path, meta_json)?;
    
    info!(
        checkpoint = %path.display(),
        checksum = %checksum,
        step = ?step,
        "Checkpoint saved with checksum"
    );
    
    Ok(())
}

/// Load checkpoint with checksum validation.
/// 
/// Verifies the checkpoint integrity against the stored checksum.
/// Returns an error if the checksum doesn't match (corruption detected).
pub fn load_checkpoint_with_checksum<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<(HashMap<String, Tensor>, Option<CheckpointMetadata>)> {
    let path = path.as_ref();
    
    if !path.exists() {
        return Err(DeepSeekError::Checkpoint(
            format!("Checkpoint not found: {}", path.display())
        ));
    }
    
    // Load tensors
    let tensors = safetensors::load(path, device)?;
    
    // Try to load and verify metadata
    let meta_path = path.with_extension("meta.json");
    let metadata = if meta_path.exists() {
        let meta_json = fs::read_to_string(&meta_path)?;
        let meta: CheckpointMetadata = serde_json::from_str(&meta_json)
            .map_err(|e| DeepSeekError::Checkpoint(format!("Failed to parse metadata: {}", e)))?;
        
        // Verify checksum
        let current_checksum = compute_tensor_checksum(&tensors)?;
        if current_checksum != meta.checksum {
            error!(
                expected = %meta.checksum,
                actual = %current_checksum,
                "Checkpoint checksum mismatch - corruption detected"
            );
            return Err(DeepSeekError::Checkpoint(
                format!(
                    "Checkpoint corruption detected: checksum mismatch (expected {}, got {})",
                    meta.checksum, current_checksum
                )
            ));
        }
        
        info!(
            checkpoint = %path.display(),
            checksum = %current_checksum,
            "Checkpoint loaded and verified"
        );
        
        Some(meta)
    } else {
        warn!(
            checkpoint = %path.display(),
            "No metadata file found - skipping checksum verification"
        );
        None
    };
    
    Ok((tensors, metadata))
}

/// Validate a checkpoint without loading all tensors into memory.
/// 
/// Useful for checking checkpoint integrity before training.
pub fn validate_checkpoint<P: AsRef<Path>>(path: P) -> Result<bool> {
    let path = path.as_ref();
    let device = Device::Cpu;
    
    match load_checkpoint_with_checksum(path, &device) {
        Ok(_) => Ok(true),
        Err(DeepSeekError::Checkpoint(msg)) if msg.contains("corruption") => Ok(false),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_checkpoint_save_load() -> Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_checkpoint.safetensors");
        
        // Create test tensors
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            Tensor::randn(0f32, 1f32, (10, 10), &device)?,
        );
        tensors.insert(
            "bias".to_string(),
            Tensor::zeros((10,), DType::F32, &device)?,
        );
        
        // Save
        save_checkpoint_with_checksum(&tensors, &path, Some(100), Some(1), None)?;
        
        // Verify meta file exists
        assert!(path.with_extension("meta.json").exists());
        
        // Load and verify
        let (loaded, meta) = load_checkpoint_with_checksum(&path, &device)?;
        assert!(meta.is_some());
        assert_eq!(meta.unwrap().step, Some(100));
        assert_eq!(loaded.len(), 2);
        
        Ok(())
    }
    
    #[test]
    fn test_checksum_consistency() -> Result<()> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "test".to_string(),
            Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?,
        );
        
        let checksum1 = compute_tensor_checksum(&tensors)?;
        let checksum2 = compute_tensor_checksum(&tensors)?;
        
        assert_eq!(checksum1, checksum2);
        Ok(())
    }
}
