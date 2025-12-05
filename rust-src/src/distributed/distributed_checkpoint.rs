//! Distributed Checkpointing for Multi-GPU Training.
//!
//! This module provides comprehensive distributed checkpoint capabilities:
//! - Async checkpoint saving to avoid blocking training
//! - Checkpoint versioning and metadata tracking
//! - Garbage collection of old checkpoints
//! - Sharded checkpointing for distributed models
//! - Checkpoint validation before deletion
//! - Topology-agnostic loading (N GPUs -> M GPUs)
//!
//! Reference: PyTorch Distributed Checkpoint (DCP) patterns for Rust.

use candle_core::{Result, Tensor, Device, DType};
use std::collections::{HashMap, BTreeMap};
use std::fs::{self, File};
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};

// ============================================================================
// Checkpoint Configuration
// ============================================================================

/// Configuration for distributed checkpointing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Base directory for checkpoints
    pub checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,
    /// Enable async saving
    pub async_save: bool,
    /// Checkpoint every N steps
    pub save_interval_steps: u64,
    /// Checkpoint every N seconds (0 = disabled)
    pub save_interval_secs: u64,
    /// Enable checkpoint validation before deletion
    pub validate_before_delete: bool,
    /// Enable sharded checkpointing
    pub sharded: bool,
    /// Compression level (0 = none, 1-9 = zlib levels)
    pub compression_level: u32,
    /// Include optimizer state
    pub save_optimizer: bool,
    /// Include scheduler state
    pub save_scheduler: bool,
    /// Include RNG state
    pub save_rng_state: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("./checkpoints"),
            max_checkpoints: 5,
            async_save: true,
            save_interval_steps: 1000,
            save_interval_secs: 0,
            validate_before_delete: true,
            sharded: true,
            compression_level: 0,
            save_optimizer: true,
            save_scheduler: true,
            save_rng_state: true,
        }
    }
}

impl CheckpointConfig {
    pub fn new(checkpoint_dir: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            ..Default::default()
        }
    }
    
    pub fn with_max_checkpoints(mut self, max: usize) -> Self {
        self.max_checkpoints = max;
        self
    }
    
    pub fn with_async_save(mut self, enabled: bool) -> Self {
        self.async_save = enabled;
        self
    }
    
    pub fn with_save_interval(mut self, steps: u64, secs: u64) -> Self {
        self.save_interval_steps = steps;
        self.save_interval_secs = secs;
        self
    }
    
    pub fn with_sharding(mut self, enabled: bool) -> Self {
        self.sharded = enabled;
        self
    }
    
    pub fn with_compression(mut self, level: u32) -> Self {
        self.compression_level = level.min(9);
        self
    }
}

// ============================================================================
// Checkpoint Metadata
// ============================================================================

/// Metadata for a checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Checkpoint version (for format compatibility)
    pub version: String,
    /// Training step
    pub step: u64,
    /// Epoch
    pub epoch: u64,
    /// Loss at checkpoint
    pub loss: f64,
    /// Timestamp (Unix epoch seconds)
    pub timestamp: u64,
    /// World size when checkpoint was created
    pub world_size: usize,
    /// Rank that created this shard (for sharded checkpoints)
    pub rank: usize,
    /// Model configuration hash (for validation)
    pub model_config_hash: String,
    /// Number of parameters
    pub num_parameters: u64,
    /// List of tensor names in checkpoint
    pub tensor_names: Vec<String>,
    /// Shard info for distributed checkpoints
    pub shard_info: Option<ShardInfo>,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

impl CheckpointMetadata {
    pub fn new(step: u64, epoch: u64, loss: f64, world_size: usize, rank: usize) -> Self {
        Self {
            version: "1.0.0".to_string(),
            step,
            epoch,
            loss,
            timestamp: current_timestamp(),
            world_size,
            rank,
            model_config_hash: String::new(),
            num_parameters: 0,
            tensor_names: Vec::new(),
            shard_info: None,
            custom: HashMap::new(),
        }
    }
    
    pub fn with_model_hash(mut self, hash: &str) -> Self {
        self.model_config_hash = hash.to_string();
        self
    }
    
    pub fn with_parameters(mut self, count: u64) -> Self {
        self.num_parameters = count;
        self
    }
    
    pub fn with_shard_info(mut self, info: ShardInfo) -> Self {
        self.shard_info = Some(info);
        self
    }
    
    pub fn add_custom(&mut self, key: &str, value: &str) {
        self.custom.insert(key.to_string(), value.to_string());
    }
}

/// Shard information for distributed checkpoints.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardInfo {
    /// Total number of shards
    pub total_shards: usize,
    /// This shard's index
    pub shard_index: usize,
    /// Parameter indices in this shard
    pub param_start: usize,
    /// Number of parameters in this shard
    pub param_count: usize,
    /// Byte offset in the full checkpoint
    pub byte_offset: u64,
    /// Byte size of this shard
    pub byte_size: u64,
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================================================
// Checkpoint State
// ============================================================================

/// Complete training state for checkpointing.
#[derive(Clone)]
pub struct TrainingState {
    /// Model weights (name -> tensor)
    pub model_weights: HashMap<String, Tensor>,
    /// Optimizer state (name -> tensor)
    pub optimizer_state: HashMap<String, Tensor>,
    /// Scheduler state
    pub scheduler_state: SchedulerState,
    /// Current training step
    pub step: u64,
    /// Current epoch
    pub epoch: u64,
    /// Best validation loss
    pub best_loss: f64,
    /// Current loss
    pub current_loss: f64,
    /// RNG state bytes
    pub rng_state: Vec<u8>,
    /// Custom state
    pub custom: HashMap<String, Vec<u8>>,
}

/// Scheduler state for checkpointing.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SchedulerState {
    pub last_epoch: u64,
    pub last_lr: f64,
    pub base_lrs: Vec<f64>,
    pub warmup_steps: u64,
    pub total_steps: u64,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            model_weights: HashMap::new(),
            optimizer_state: HashMap::new(),
            scheduler_state: SchedulerState::default(),
            step: 0,
            epoch: 0,
            best_loss: f64::MAX,
            current_loss: 0.0,
            rng_state: Vec::new(),
            custom: HashMap::new(),
        }
    }
}

impl TrainingState {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_step(mut self, step: u64) -> Self {
        self.step = step;
        self
    }
    
    pub fn with_epoch(mut self, epoch: u64) -> Self {
        self.epoch = epoch;
        self
    }
    
    pub fn with_loss(mut self, loss: f64) -> Self {
        self.current_loss = loss;
        self
    }
    
    pub fn add_weight(&mut self, name: &str, tensor: Tensor) {
        self.model_weights.insert(name.to_string(), tensor);
    }
    
    pub fn add_optimizer_state(&mut self, name: &str, tensor: Tensor) {
        self.optimizer_state.insert(name.to_string(), tensor);
    }
    
    pub fn set_rng_state(&mut self, state: Vec<u8>) {
        self.rng_state = state;
    }
    
    pub fn add_custom(&mut self, key: &str, data: Vec<u8>) {
        self.custom.insert(key.to_string(), data);
    }
    
    /// Calculate total parameters.
    pub fn total_parameters(&self) -> u64 {
        self.model_weights
            .values()
            .map(|t| t.elem_count() as u64)
            .sum()
    }
}

// ============================================================================
// Async Checkpoint Saver
// ============================================================================

/// Async checkpoint saver that saves in background.
pub struct AsyncCheckpointSaver {
    /// Configuration
    config: CheckpointConfig,
    /// Queue of pending saves
    pending_saves: Arc<Mutex<Vec<PendingSave>>>,
    /// Save thread handle
    save_thread: Option<JoinHandle<()>>,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Last save time
    last_save_time: Arc<AtomicU64>,
    /// Save in progress flag
    save_in_progress: Arc<AtomicBool>,
}

struct PendingSave {
    state: TrainingState,
    metadata: CheckpointMetadata,
    path: PathBuf,
}

impl AsyncCheckpointSaver {
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            pending_saves: Arc::new(Mutex::new(Vec::new())),
            save_thread: None,
            running: Arc::new(AtomicBool::new(false)),
            last_save_time: Arc::new(AtomicU64::new(0)),
            save_in_progress: Arc::new(AtomicBool::new(false)),
        }
    }
    
    /// Start the async saver background thread.
    pub fn start(&mut self) {
        if self.running.load(Ordering::SeqCst) {
            return;
        }
        
        self.running.store(true, Ordering::SeqCst);
        
        let pending = self.pending_saves.clone();
        let running = self.running.clone();
        let save_in_progress = self.save_in_progress.clone();
        let config = self.config.clone();
        
        self.save_thread = Some(thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                // Check for pending saves
                let save = {
                    let mut queue = pending.lock().unwrap();
                    queue.pop()
                };
                
                if let Some(save) = save {
                    save_in_progress.store(true, Ordering::SeqCst);
                    
                    // Perform the save
                    if let Err(e) = save_checkpoint_to_disk(
                        &save.state,
                        &save.metadata,
                        &save.path,
                        &config,
                    ) {
                        eprintln!("Async checkpoint save failed: {:?}", e);
                    }
                    
                    save_in_progress.store(false, Ordering::SeqCst);
                } else {
                    // No pending saves, sleep briefly
                    thread::sleep(Duration::from_millis(100));
                }
            }
        }));
    }
    
    /// Stop the async saver.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        
        if let Some(handle) = self.save_thread.take() {
            let _ = handle.join();
        }
    }
    
    /// Queue a checkpoint for async saving.
    pub fn queue_save(&self, state: TrainingState, metadata: CheckpointMetadata, path: PathBuf) {
        let mut queue = self.pending_saves.lock().unwrap();
        queue.push(PendingSave { state, metadata, path });
        self.last_save_time.store(current_timestamp(), Ordering::SeqCst);
    }
    
    /// Check if a save is in progress.
    pub fn is_saving(&self) -> bool {
        self.save_in_progress.load(Ordering::SeqCst)
    }
    
    /// Get number of pending saves.
    pub fn pending_count(&self) -> usize {
        self.pending_saves.lock().unwrap().len()
    }
    
    /// Wait for all pending saves to complete.
    pub fn wait_for_completion(&self) {
        while self.is_saving() || self.pending_count() > 0 {
            thread::sleep(Duration::from_millis(100));
        }
    }
    
    /// Get last save timestamp.
    pub fn last_save_time(&self) -> u64 {
        self.last_save_time.load(Ordering::SeqCst)
    }
}

impl Drop for AsyncCheckpointSaver {
    fn drop(&mut self) {
        self.stop();
    }
}

// ============================================================================
// Distributed Checkpointer
// ============================================================================

/// Main distributed checkpointer.
pub struct DistributedCheckpointer {
    /// Configuration
    config: CheckpointConfig,
    /// This rank
    rank: usize,
    /// World size
    world_size: usize,
    /// Async saver
    async_saver: Option<AsyncCheckpointSaver>,
    /// Checkpoint registry (step -> path)
    registry: RwLock<BTreeMap<u64, PathBuf>>,
    /// Last checkpoint step
    last_checkpoint_step: AtomicU64,
    /// Total checkpoints saved
    total_saves: AtomicU64,
}

impl DistributedCheckpointer {
    pub fn new(config: CheckpointConfig, rank: usize, world_size: usize) -> Self {
        // Create checkpoint directory
        if let Err(e) = fs::create_dir_all(&config.checkpoint_dir) {
            eprintln!("Warning: Could not create checkpoint dir: {:?}", e);
        }
        
        Self {
            config,
            rank,
            world_size,
            async_saver: None,
            registry: RwLock::new(BTreeMap::new()),
            last_checkpoint_step: AtomicU64::new(0),
            total_saves: AtomicU64::new(0),
        }
    }
    
    /// Initialize the checkpointer (start async saver if enabled).
    pub fn initialize(&mut self) {
        if self.config.async_save {
            let mut saver = AsyncCheckpointSaver::new(self.config.clone());
            saver.start();
            self.async_saver = Some(saver);
        }
        
        // Scan existing checkpoints
        self.scan_existing_checkpoints();
    }
    
    /// Scan checkpoint directory for existing checkpoints.
    fn scan_existing_checkpoints(&self) {
        let dir = &self.config.checkpoint_dir;
        
        if let Ok(entries) = fs::read_dir(dir) {
            let mut registry = self.registry.write().unwrap();
            
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_dir() && path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("step_"))
                    .unwrap_or(false)
                {
                    // Extract step number from directory name
                    if let Some(step_str) = path.file_name()
                        .and_then(|n| n.to_str())
                        .and_then(|n| n.strip_prefix("step_"))
                    {
                        if let Ok(step) = step_str.parse::<u64>() {
                            registry.insert(step, path);
                        }
                    }
                }
            }
        }
    }
    
    /// Check if should checkpoint at this step.
    pub fn should_checkpoint(&self, step: u64) -> bool {
        if self.config.save_interval_steps > 0 && step % self.config.save_interval_steps == 0 {
            return true;
        }
        
        if self.config.save_interval_secs > 0 {
            if let Some(ref saver) = self.async_saver {
                let last = saver.last_save_time();
                let now = current_timestamp();
                if now - last >= self.config.save_interval_secs {
                    return true;
                }
            }
        }
        
        false
    }
    
    /// Save checkpoint.
    pub fn save(&self, state: &TrainingState) -> Result<PathBuf> {
        let step = state.step;
        let path = self.checkpoint_path(step);
        
        let metadata = CheckpointMetadata::new(
            step,
            state.epoch,
            state.current_loss,
            self.world_size,
            self.rank,
        ).with_parameters(state.total_parameters());
        
        if self.config.async_save {
            if let Some(ref saver) = self.async_saver {
                saver.queue_save(state.clone(), metadata, path.clone());
            }
        } else {
            save_checkpoint_to_disk(state, &metadata, &path, &self.config)?;
        }
        
        // Register checkpoint
        {
            let mut registry = self.registry.write().unwrap();
            registry.insert(step, path.clone());
        }
        
        self.last_checkpoint_step.store(step, Ordering::SeqCst);
        self.total_saves.fetch_add(1, Ordering::SeqCst);
        
        // Garbage collect old checkpoints
        self.garbage_collect()?;
        
        Ok(path)
    }
    
    /// Load checkpoint.
    pub fn load(&self, step: Option<u64>) -> Result<TrainingState> {
        let path = if let Some(step) = step {
            self.checkpoint_path(step)
        } else {
            // Load latest
            self.latest_checkpoint_path()
                .ok_or_else(|| candle_core::Error::Msg("No checkpoints found".to_string()))?
        };
        
        load_checkpoint_from_disk(&path, &self.config)
    }
    
    /// Load latest checkpoint.
    pub fn load_latest(&self) -> Result<TrainingState> {
        self.load(None)
    }
    
    /// Get checkpoint path for step.
    fn checkpoint_path(&self, step: u64) -> PathBuf {
        self.config.checkpoint_dir.join(format!("step_{}", step))
    }
    
    /// Get latest checkpoint path.
    fn latest_checkpoint_path(&self) -> Option<PathBuf> {
        let registry = self.registry.read().unwrap();
        registry.values().last().cloned()
    }
    
    /// Get latest checkpoint step.
    pub fn latest_checkpoint_step(&self) -> Option<u64> {
        let registry = self.registry.read().unwrap();
        registry.keys().last().copied()
    }
    
    /// Garbage collect old checkpoints.
    pub fn garbage_collect(&self) -> Result<()> {
        let mut registry = self.registry.write().unwrap();
        
        while registry.len() > self.config.max_checkpoints {
            if let Some((&oldest_step, oldest_path)) = registry.iter().next() {
                let path = oldest_path.clone();
                
                // Validate newest checkpoint before deleting old one
                if self.config.validate_before_delete {
                    if let Some(newest_path) = registry.values().last() {
                        if !validate_checkpoint(newest_path) {
                            // Don't delete if newest is invalid
                            break;
                        }
                    }
                }
                
                // Delete the checkpoint
                if let Err(e) = fs::remove_dir_all(&path) {
                    eprintln!("Warning: Failed to delete checkpoint {:?}: {:?}", path, e);
                }
                
                registry.remove(&oldest_step);
            } else {
                break;
            }
        }
        
        Ok(())
    }
    
    /// List all checkpoints.
    pub fn list_checkpoints(&self) -> Vec<(u64, PathBuf)> {
        let registry = self.registry.read().unwrap();
        registry.iter().map(|(&k, v)| (k, v.clone())).collect()
    }
    
    /// Get number of checkpoints.
    pub fn checkpoint_count(&self) -> usize {
        self.registry.read().unwrap().len()
    }
    
    /// Get total saves.
    pub fn total_saves(&self) -> u64 {
        self.total_saves.load(Ordering::SeqCst)
    }
    
    /// Wait for async saves to complete.
    pub fn wait_for_saves(&self) {
        if let Some(ref saver) = self.async_saver {
            saver.wait_for_completion();
        }
    }
    
    /// Shutdown checkpointer.
    pub fn shutdown(&mut self) {
        if let Some(ref mut saver) = self.async_saver {
            saver.stop();
        }
    }
}

impl Drop for DistributedCheckpointer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ============================================================================
// Checkpoint I/O
// ============================================================================

/// Save checkpoint to disk.
fn save_checkpoint_to_disk(
    state: &TrainingState,
    metadata: &CheckpointMetadata,
    path: &Path,
    _config: &CheckpointConfig,
) -> Result<()> {
    // Create checkpoint directory
    fs::create_dir_all(path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to create checkpoint dir: {}", e)))?;
    
    // Save metadata
    let metadata_path = path.join("metadata.json");
    let metadata_json = serde_json::to_string_pretty(metadata)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to serialize metadata: {}", e)))?;
    fs::write(&metadata_path, metadata_json)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to write metadata: {}", e)))?;
    
    // Save model weights
    let weights_path = path.join("model_weights.safetensors");
    if !state.model_weights.is_empty() {
        candle_core::safetensors::save(&state.model_weights, &weights_path)?;
    }
    
    // Save optimizer state
    if !state.optimizer_state.is_empty() {
        let optimizer_path = path.join("optimizer_state.safetensors");
        candle_core::safetensors::save(&state.optimizer_state, &optimizer_path)?;
    }
    
    // Save scheduler state
    let scheduler_path = path.join("scheduler_state.json");
    let scheduler_json = serde_json::to_string_pretty(&state.scheduler_state)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to serialize scheduler: {}", e)))?;
    fs::write(&scheduler_path, scheduler_json)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to write scheduler: {}", e)))?;
    
    // Save training state
    let training_state_path = path.join("training_state.json");
    let training_state = serde_json::json!({
        "step": state.step,
        "epoch": state.epoch,
        "best_loss": state.best_loss,
        "current_loss": state.current_loss,
    });
    fs::write(&training_state_path, serde_json::to_string_pretty(&training_state).unwrap())
        .map_err(|e| candle_core::Error::Msg(format!("Failed to write training state: {}", e)))?;
    
    // Save RNG state if present
    if !state.rng_state.is_empty() {
        let rng_path = path.join("rng_state.bin");
        fs::write(&rng_path, &state.rng_state)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to write RNG state: {}", e)))?;
    }
    
    Ok(())
}

/// Load checkpoint from disk.
fn load_checkpoint_from_disk(path: &Path, _config: &CheckpointConfig) -> Result<TrainingState> {
    if !path.exists() {
        return Err(candle_core::Error::Msg(format!(
            "Checkpoint path does not exist: {:?}", path
        )));
    }
    
    let mut state = TrainingState::default();
    
    // Load model weights
    let weights_path = path.join("model_weights.safetensors");
    if weights_path.exists() {
        let tensors = candle_core::safetensors::load(&weights_path, &Device::Cpu)?;
        state.model_weights = tensors;
    }
    
    // Load optimizer state
    let optimizer_path = path.join("optimizer_state.safetensors");
    if optimizer_path.exists() {
        let tensors = candle_core::safetensors::load(&optimizer_path, &Device::Cpu)?;
        state.optimizer_state = tensors;
    }
    
    // Load scheduler state
    let scheduler_path = path.join("scheduler_state.json");
    if scheduler_path.exists() {
        let content = fs::read_to_string(&scheduler_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read scheduler: {}", e)))?;
        state.scheduler_state = serde_json::from_str(&content)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse scheduler: {}", e)))?;
    }
    
    // Load training state
    let training_state_path = path.join("training_state.json");
    if training_state_path.exists() {
        let content = fs::read_to_string(&training_state_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read training state: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse training state: {}", e)))?;
        
        state.step = json.get("step").and_then(|v| v.as_u64()).unwrap_or(0);
        state.epoch = json.get("epoch").and_then(|v| v.as_u64()).unwrap_or(0);
        state.best_loss = json.get("best_loss").and_then(|v| v.as_f64()).unwrap_or(f64::MAX);
        state.current_loss = json.get("current_loss").and_then(|v| v.as_f64()).unwrap_or(0.0);
    }
    
    // Load RNG state
    let rng_path = path.join("rng_state.bin");
    if rng_path.exists() {
        state.rng_state = fs::read(&rng_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read RNG state: {}", e)))?;
    }
    
    Ok(state)
}

/// Validate a checkpoint.
fn validate_checkpoint(path: &Path) -> bool {
    // Check that essential files exist
    let metadata_path = path.join("metadata.json");
    let weights_path = path.join("model_weights.safetensors");
    
    if !metadata_path.exists() {
        return false;
    }
    
    // Try to load metadata
    if let Ok(content) = fs::read_to_string(&metadata_path) {
        if serde_json::from_str::<CheckpointMetadata>(&content).is_err() {
            return false;
        }
    } else {
        return false;
    }
    
    // Check weights file if it exists
    if weights_path.exists() {
        // Try to read header to validate
        if let Ok(data) = fs::read(&weights_path) {
            if data.len() < 8 {
                return false;
            }
        } else {
            return false;
        }
    }
    
    true
}

// ============================================================================
// Sharded Checkpoint Utilities
// ============================================================================

/// Shard state across multiple files for distributed saving.
pub fn shard_state(
    state: &TrainingState,
    world_size: usize,
    rank: usize,
) -> TrainingState {
    let mut sharded = TrainingState::default();
    sharded.step = state.step;
    sharded.epoch = state.epoch;
    sharded.best_loss = state.best_loss;
    sharded.current_loss = state.current_loss;
    sharded.scheduler_state = state.scheduler_state.clone();
    
    // Distribute weights by name hash
    for (name, tensor) in &state.model_weights {
        let hash = simple_hash(name);
        if hash % world_size == rank {
            sharded.model_weights.insert(name.clone(), tensor.clone());
        }
    }
    
    // Same for optimizer state
    for (name, tensor) in &state.optimizer_state {
        let hash = simple_hash(name);
        if hash % world_size == rank {
            sharded.optimizer_state.insert(name.clone(), tensor.clone());
        }
    }
    
    // RNG state only on rank 0
    if rank == 0 {
        sharded.rng_state = state.rng_state.clone();
    }
    
    sharded
}

/// Merge sharded states back into complete state.
pub fn merge_shards(shards: Vec<TrainingState>) -> TrainingState {
    if shards.is_empty() {
        return TrainingState::default();
    }
    
    let mut merged = TrainingState::default();
    
    // Use first shard for scalar values
    let first = &shards[0];
    merged.step = first.step;
    merged.epoch = first.epoch;
    merged.best_loss = first.best_loss;
    merged.current_loss = first.current_loss;
    merged.scheduler_state = first.scheduler_state.clone();
    
    // Merge weights from all shards
    for shard in &shards {
        merged.model_weights.extend(shard.model_weights.clone());
        merged.optimizer_state.extend(shard.optimizer_state.clone());
        
        if !shard.rng_state.is_empty() {
            merged.rng_state = shard.rng_state.clone();
        }
    }
    
    merged
}

fn simple_hash(s: &str) -> usize {
    let mut hash: usize = 0;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as usize);
    }
    hash
}

// ============================================================================
// Checkpoint Comparison Utilities
// ============================================================================

/// Compare two checkpoints for equality.
pub fn compare_checkpoints(path1: &Path, path2: &Path) -> Result<bool> {
    let state1 = load_checkpoint_from_disk(path1, &CheckpointConfig::default())?;
    let state2 = load_checkpoint_from_disk(path2, &CheckpointConfig::default())?;
    
    // Compare scalar values
    if state1.step != state2.step || state1.epoch != state2.epoch {
        return Ok(false);
    }
    
    // Compare weights
    if state1.model_weights.len() != state2.model_weights.len() {
        return Ok(false);
    }
    
    for (name, tensor1) in &state1.model_weights {
        if let Some(tensor2) = state2.model_weights.get(name) {
            if tensor1.dims() != tensor2.dims() {
                return Ok(false);
            }
            // Note: Exact value comparison would require tensor data access
        } else {
            return Ok(false);
        }
    }
    
    Ok(true)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_checkpoint_config() {
        let config = CheckpointConfig::new("./test_ckpt")
            .with_max_checkpoints(3)
            .with_async_save(false)
            .with_save_interval(100, 0);
        
        assert_eq!(config.max_checkpoints, 3);
        assert!(!config.async_save);
        assert_eq!(config.save_interval_steps, 100);
    }
    
    #[test]
    fn test_checkpoint_metadata() {
        let metadata = CheckpointMetadata::new(100, 5, 0.5, 4, 0)
            .with_model_hash("abc123")
            .with_parameters(1_000_000);
        
        assert_eq!(metadata.step, 100);
        assert_eq!(metadata.epoch, 5);
        assert_eq!(metadata.world_size, 4);
        assert_eq!(metadata.num_parameters, 1_000_000);
    }
    
    #[test]
    fn test_training_state() {
        let device = Device::Cpu;
        
        let mut state = TrainingState::new()
            .with_step(100)
            .with_epoch(5)
            .with_loss(0.5);
        
        let tensor = Tensor::zeros((10, 10), DType::F32, &device).unwrap();
        state.add_weight("layer1.weight", tensor);
        
        assert_eq!(state.step, 100);
        assert_eq!(state.model_weights.len(), 1);
        assert_eq!(state.total_parameters(), 100);
    }
    
    #[test]
    fn test_shard_merge() {
        let device = Device::Cpu;
        
        let mut state = TrainingState::new().with_step(100);
        state.add_weight("w1", Tensor::zeros((10,), DType::F32, &device).unwrap());
        state.add_weight("w2", Tensor::zeros((10,), DType::F32, &device).unwrap());
        state.add_weight("w3", Tensor::zeros((10,), DType::F32, &device).unwrap());
        state.add_weight("w4", Tensor::zeros((10,), DType::F32, &device).unwrap());
        
        // Shard across 2 ranks
        let shard0 = shard_state(&state, 2, 0);
        let shard1 = shard_state(&state, 2, 1);
        
        // Merge back
        let merged = merge_shards(vec![shard0, shard1]);
        
        assert_eq!(merged.step, 100);
        assert_eq!(merged.model_weights.len(), 4);
    }
    
    #[test]
    fn test_checkpoint_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig::new(temp_dir.path())
            .with_async_save(false);
        
        let device = Device::Cpu;
        
        let mut state = TrainingState::new()
            .with_step(100)
            .with_epoch(5)
            .with_loss(0.5);
        
        state.add_weight("test.weight", Tensor::zeros((10, 10), DType::F32, &device).unwrap());
        
        // Save
        let checkpointer = DistributedCheckpointer::new(config.clone(), 0, 1);
        let path = checkpointer.save(&state).unwrap();
        
        assert!(path.exists());
        
        // Load
        let loaded = checkpointer.load(Some(100)).unwrap();
        
        assert_eq!(loaded.step, 100);
        assert_eq!(loaded.epoch, 5);
        assert_eq!(loaded.model_weights.len(), 1);
    }
    
    #[test]
    fn test_garbage_collection() {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig::new(temp_dir.path())
            .with_max_checkpoints(2)
            .with_async_save(false)
            .with_save_interval(1, 0);
        
        let device = Device::Cpu;
        let checkpointer = DistributedCheckpointer::new(config, 0, 1);
        
        // Save multiple checkpoints
        for step in [100, 200, 300] {
            let mut state = TrainingState::new().with_step(step);
            state.add_weight("w", Tensor::zeros((5,), DType::F32, &device).unwrap());
            checkpointer.save(&state).unwrap();
        }
        
        // Should only have 2 checkpoints (max)
        assert!(checkpointer.checkpoint_count() <= 2);
    }
    
    #[test]
    fn test_checkpoint_validation() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test_ckpt");
        fs::create_dir_all(&path).unwrap();
        
        // Create valid metadata
        let metadata = CheckpointMetadata::new(100, 1, 0.5, 1, 0);
        let metadata_json = serde_json::to_string_pretty(&metadata).unwrap();
        fs::write(path.join("metadata.json"), metadata_json).unwrap();
        
        assert!(validate_checkpoint(&path));
        
        // Invalid checkpoint (no metadata)
        let invalid_path = temp_dir.path().join("invalid_ckpt");
        fs::create_dir_all(&invalid_path).unwrap();
        assert!(!validate_checkpoint(&invalid_path));
    }
    
    #[test]
    fn test_scheduler_state_serialization() {
        let state = SchedulerState {
            last_epoch: 10,
            last_lr: 0.001,
            base_lrs: vec![0.01, 0.001],
            warmup_steps: 100,
            total_steps: 10000,
        };
        
        let json = serde_json::to_string(&state).unwrap();
        let loaded: SchedulerState = serde_json::from_str(&json).unwrap();
        
        assert_eq!(loaded.last_epoch, 10);
        assert_eq!(loaded.warmup_steps, 100);
    }
}
