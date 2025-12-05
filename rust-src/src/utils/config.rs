//! Configuration loading with JSON file support and environment variable overrides.
//!
//! Pattern: `DEEPSEEK_*` environment variables override config file values.
//! Example: `DEEPSEEK_BATCH_SIZE=32` overrides `batch_size` in config.

use crate::utils::error::{DeepSeekError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use tracing::{debug, info};

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Batch size per device
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    
    /// Maximum sequence length
    #[serde(default = "default_seq_len")]
    pub max_seq_len: usize,
    
    /// Learning rate
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    
    /// Weight decay
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    
    /// Number of training steps
    #[serde(default = "default_max_steps")]
    pub max_steps: u64,
    
    /// Warmup steps
    #[serde(default = "default_warmup_steps")]
    pub warmup_steps: u64,
    
    /// Gradient accumulation steps
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: usize,
    
    /// Checkpoint save interval (steps)
    #[serde(default = "default_save_interval")]
    pub save_interval: u64,
    
    /// Logging interval (steps)
    #[serde(default = "default_log_interval")]
    pub log_interval: u64,
    
    /// Checkpoint directory
    #[serde(default = "default_checkpoint_dir")]
    pub checkpoint_dir: String,
    
    /// Enable gradient checkpointing
    #[serde(default)]
    pub gradient_checkpointing: bool,
    
    /// Mixed precision training
    #[serde(default = "default_mixed_precision")]
    pub mixed_precision: String,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model dimension
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    
    /// Number of attention heads
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    
    /// Number of transformer layers
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    
    /// Vocabulary size
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    
    /// MLA latent dimension
    #[serde(default = "default_d_latent")]
    pub d_latent: usize,
    
    /// RoPE dimension
    #[serde(default = "default_d_rope")]
    pub d_rope: usize,
    
    /// Number of routed experts
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    
    /// Number of shared experts
    #[serde(default = "default_num_shared_experts")]
    pub num_shared_experts: usize,
    
    /// Top-K experts to activate
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

/// Distributed training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// World size (total number of processes)
    #[serde(default = "default_world_size")]
    pub world_size: usize,
    
    /// Data parallelism degree
    #[serde(default = "default_one")]
    pub dp_size: usize,
    
    /// Tensor parallelism degree
    #[serde(default = "default_one")]
    pub tp_size: usize,
    
    /// Pipeline parallelism degree
    #[serde(default = "default_one")]
    pub pp_size: usize,
    
    /// Expert parallelism degree
    #[serde(default = "default_one")]
    pub ep_size: usize,
    
    /// Sequence parallelism degree
    #[serde(default = "default_one")]
    pub sp_size: usize,
    
    /// Communication backend
    #[serde(default = "default_backend")]
    pub backend: String,
    
    /// Master address
    #[serde(default = "default_master_addr")]
    pub master_addr: String,
    
    /// Master port
    #[serde(default = "default_master_port")]
    pub master_port: u16,
}

/// Complete configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeepSeekConfig {
    #[serde(default)]
    pub training: TrainingConfig,
    
    #[serde(default)]
    pub model: ModelConfig,
    
    #[serde(default)]
    pub distributed: DistributedConfig,
    
    /// Additional key-value configuration
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

// Default value functions
fn default_batch_size() -> usize { 4 }
fn default_seq_len() -> usize { 2048 }
fn default_lr() -> f64 { 1e-4 }
fn default_weight_decay() -> f64 { 0.1 }
fn default_max_steps() -> u64 { 100000 }
fn default_warmup_steps() -> u64 { 2000 }
fn default_grad_accum() -> usize { 1 }
fn default_save_interval() -> u64 { 1000 }
fn default_log_interval() -> u64 { 10 }
fn default_checkpoint_dir() -> String { "./checkpoints".to_string() }
fn default_mixed_precision() -> String { "bf16".to_string() }
fn default_d_model() -> usize { 4096 }
fn default_num_heads() -> usize { 32 }
fn default_num_layers() -> usize { 32 }
fn default_vocab_size() -> usize { 102400 }
fn default_d_latent() -> usize { 512 }
fn default_d_rope() -> usize { 64 }
fn default_num_experts() -> usize { 256 }
fn default_num_shared_experts() -> usize { 1 }
fn default_top_k() -> usize { 8 }
fn default_world_size() -> usize { 1 }
fn default_one() -> usize { 1 }
fn default_backend() -> String { "nccl".to_string() }
fn default_master_addr() -> String { "localhost".to_string() }
fn default_master_port() -> u16 { 29500 }

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: default_batch_size(),
            max_seq_len: default_seq_len(),
            learning_rate: default_lr(),
            weight_decay: default_weight_decay(),
            max_steps: default_max_steps(),
            warmup_steps: default_warmup_steps(),
            gradient_accumulation_steps: default_grad_accum(),
            save_interval: default_save_interval(),
            log_interval: default_log_interval(),
            checkpoint_dir: default_checkpoint_dir(),
            gradient_checkpointing: false,
            mixed_precision: default_mixed_precision(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            d_model: default_d_model(),
            num_heads: default_num_heads(),
            num_layers: default_num_layers(),
            vocab_size: default_vocab_size(),
            d_latent: default_d_latent(),
            d_rope: default_d_rope(),
            num_experts: default_num_experts(),
            num_shared_experts: default_num_shared_experts(),
            top_k: default_top_k(),
        }
    }
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            world_size: default_world_size(),
            dp_size: default_one(),
            tp_size: default_one(),
            pp_size: default_one(),
            ep_size: default_one(),
            sp_size: default_one(),
            backend: default_backend(),
            master_addr: default_master_addr(),
            master_port: default_master_port(),
        }
    }
}

impl DeepSeekConfig {
    /// Load configuration from a JSON file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)
            .map_err(|e| DeepSeekError::Config(format!("Failed to parse config: {}", e)))?;
        
        info!(config_file = %path.display(), "Loaded configuration from file");
        Ok(config)
    }
    
    /// Load configuration with environment variable overrides.
    /// 
    /// Environment variables are prefixed with `DEEPSEEK_` and use uppercase.
    /// Nested keys use double underscore: `DEEPSEEK_TRAINING__BATCH_SIZE`.
    pub fn from_file_with_env<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut config = Self::from_file(path)?;
        config.apply_env_overrides();
        Ok(config)
    }
    
    /// Load configuration from environment only (no file).
    pub fn from_env() -> Self {
        let mut config = Self::default();
        config.apply_env_overrides();
        config
    }
    
    /// Apply environment variable overrides.
    fn apply_env_overrides(&mut self) {
        for (key, value) in env::vars() {
            if !key.starts_with("DEEPSEEK_") {
                continue;
            }
            
            let config_key = key[9..].to_lowercase(); // Remove "DEEPSEEK_" prefix
            
            // Handle nested keys with double underscore
            let parts: Vec<&str> = config_key.split("__").collect();
            
            match parts.as_slice() {
                ["training", field] => self.apply_training_override(field, &value),
                ["model", field] => self.apply_model_override(field, &value),
                ["distributed", field] => self.apply_distributed_override(field, &value),
                [field] => {
                    // Try each section for simple keys
                    self.apply_training_override(field, &value);
                    self.apply_model_override(field, &value);
                    self.apply_distributed_override(field, &value);
                }
                _ => {
                    debug!(key = %key, "Unknown config key pattern");
                }
            }
        }
    }
    
    fn apply_training_override(&mut self, field: &str, value: &str) {
        match field {
            "batch_size" => if let Ok(v) = value.parse() { self.training.batch_size = v; }
            "max_seq_len" => if let Ok(v) = value.parse() { self.training.max_seq_len = v; }
            "learning_rate" | "lr" => if let Ok(v) = value.parse() { self.training.learning_rate = v; }
            "weight_decay" => if let Ok(v) = value.parse() { self.training.weight_decay = v; }
            "max_steps" => if let Ok(v) = value.parse() { self.training.max_steps = v; }
            "warmup_steps" => if let Ok(v) = value.parse() { self.training.warmup_steps = v; }
            "gradient_accumulation_steps" | "grad_accum" => {
                if let Ok(v) = value.parse() { self.training.gradient_accumulation_steps = v; }
            }
            "save_interval" => if let Ok(v) = value.parse() { self.training.save_interval = v; }
            "log_interval" => if let Ok(v) = value.parse() { self.training.log_interval = v; }
            "checkpoint_dir" => self.training.checkpoint_dir = value.to_string(),
            "gradient_checkpointing" => {
                self.training.gradient_checkpointing = value.to_lowercase() == "true" || value == "1";
            }
            "mixed_precision" => self.training.mixed_precision = value.to_string(),
            _ => {}
        }
    }
    
    fn apply_model_override(&mut self, field: &str, value: &str) {
        match field {
            "d_model" => if let Ok(v) = value.parse() { self.model.d_model = v; }
            "num_heads" => if let Ok(v) = value.parse() { self.model.num_heads = v; }
            "num_layers" => if let Ok(v) = value.parse() { self.model.num_layers = v; }
            "vocab_size" => if let Ok(v) = value.parse() { self.model.vocab_size = v; }
            "d_latent" => if let Ok(v) = value.parse() { self.model.d_latent = v; }
            "d_rope" => if let Ok(v) = value.parse() { self.model.d_rope = v; }
            "num_experts" => if let Ok(v) = value.parse() { self.model.num_experts = v; }
            "num_shared_experts" => if let Ok(v) = value.parse() { self.model.num_shared_experts = v; }
            "top_k" => if let Ok(v) = value.parse() { self.model.top_k = v; }
            _ => {}
        }
    }
    
    fn apply_distributed_override(&mut self, field: &str, value: &str) {
        match field {
            "world_size" => if let Ok(v) = value.parse() { self.distributed.world_size = v; }
            "dp_size" => if let Ok(v) = value.parse() { self.distributed.dp_size = v; }
            "tp_size" => if let Ok(v) = value.parse() { self.distributed.tp_size = v; }
            "pp_size" => if let Ok(v) = value.parse() { self.distributed.pp_size = v; }
            "ep_size" => if let Ok(v) = value.parse() { self.distributed.ep_size = v; }
            "sp_size" | "sequence_parallel_size" => if let Ok(v) = value.parse() { self.distributed.sp_size = v; }
            "backend" => self.distributed.backend = value.to_string(),
            "master_addr" => self.distributed.master_addr = value.to_string(),
            "master_port" => if let Ok(v) = value.parse() { self.distributed.master_port = v; }
            _ => {}
        }
    }
    
    /// Validate configuration consistency.
    pub fn validate(&self) -> Result<()> {
        // Validate parallelism dimensions
        let computed = self.distributed.dp_size 
            * self.distributed.tp_size 
            * self.distributed.pp_size;
        
        if computed != self.distributed.world_size && self.distributed.world_size > 1 {
            return Err(DeepSeekError::Config(format!(
                "DP({}) x TP({}) x PP({}) = {} != world_size({})",
                self.distributed.dp_size,
                self.distributed.tp_size,
                self.distributed.pp_size,
                computed,
                self.distributed.world_size
            )));
        }
        
        // Validate model dimensions
        if self.model.d_model % self.model.num_heads != 0 {
            return Err(DeepSeekError::Config(format!(
                "d_model ({}) must be divisible by num_heads ({})",
                self.model.d_model, self.model.num_heads
            )));
        }
        
        // Validate TP compatibility
        if self.distributed.tp_size > 1 {
            if self.model.num_heads % self.distributed.tp_size != 0 {
                return Err(DeepSeekError::Config(format!(
                    "num_heads ({}) must be divisible by tp_size ({})",
                    self.model.num_heads, self.distributed.tp_size
                )));
            }
        }
        
        Ok(())
    }
    
    /// Save configuration to a JSON file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| DeepSeekError::Config(format!("Failed to serialize config: {}", e)))?;
        fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_default_config() {
        let config = DeepSeekConfig::default();
        assert_eq!(config.training.batch_size, 4);
        assert_eq!(config.model.d_model, 4096);
        assert_eq!(config.distributed.world_size, 1);
    }
    
    #[test]
    fn test_config_save_load() -> Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.json");
        
        let config = DeepSeekConfig::default();
        config.save(&path)?;
        
        let loaded = DeepSeekConfig::from_file(&path)?;
        assert_eq!(loaded.training.batch_size, config.training.batch_size);
        
        Ok(())
    }
    
    #[test]
    fn test_env_override() {
        env::set_var("DEEPSEEK_BATCH_SIZE", "32");
        env::set_var("DEEPSEEK_TRAINING__LEARNING_RATE", "0.001");
        
        let config = DeepSeekConfig::from_env();
        assert_eq!(config.training.batch_size, 32);
        assert_eq!(config.training.learning_rate, 0.001);
        
        env::remove_var("DEEPSEEK_BATCH_SIZE");
        env::remove_var("DEEPSEEK_TRAINING__LEARNING_RATE");
    }
    
    #[test]
    fn test_validation() {
        let mut config = DeepSeekConfig::default();
        config.distributed.world_size = 8;
        config.distributed.dp_size = 2;
        config.distributed.tp_size = 2;
        config.distributed.pp_size = 2;
        
        assert!(config.validate().is_ok());
        
        config.distributed.pp_size = 4; // Now 2*2*4=16 != 8
        assert!(config.validate().is_err());
    }
}
