//! Fault Tolerance and Elastic Training for Distributed Training.
//!
//! This module provides comprehensive fault tolerance capabilities:
//! - Elastic training with dynamic worker scaling
//! - Heartbeat monitoring for health checks
//! - Preemption handling for spot instances
//! - Graceful degradation on failures
//! - Automatic recovery mechanisms
//!
//! Reference: TorchElastic-style fault tolerance for Rust training.

use candle_core::{Result, Tensor, Device, DType};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock, atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering}};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::thread::{self, JoinHandle};

// ============================================================================
// Elastic Training Configuration
// ============================================================================

/// Configuration for elastic training.
#[derive(Clone, Debug)]
pub struct ElasticConfig {
    /// Minimum number of workers to continue training
    pub min_workers: usize,
    /// Maximum number of workers
    pub max_workers: usize,
    /// Current number of workers
    pub current_workers: usize,
    /// Timeout for worker join (seconds)
    pub join_timeout_secs: u64,
    /// Maximum restarts before giving up
    pub max_restarts: usize,
    /// Rendezvous backend (e.g., "c10d", "etcd")
    pub rendezvous_backend: String,
    /// Rendezvous endpoint
    pub rendezvous_endpoint: String,
    /// Enable automatic batch size adjustment on worker change
    pub auto_scale_batch_size: bool,
    /// Base batch size per worker
    pub base_batch_size_per_worker: usize,
}

impl Default for ElasticConfig {
    fn default() -> Self {
        Self {
            min_workers: 1,
            max_workers: 8,
            current_workers: 1,
            join_timeout_secs: 300, // 5 minutes
            max_restarts: 3,
            rendezvous_backend: "c10d".to_string(),
            rendezvous_endpoint: "localhost:29400".to_string(),
            auto_scale_batch_size: true,
            base_batch_size_per_worker: 32,
        }
    }
}

impl ElasticConfig {
    pub fn new(min_workers: usize, max_workers: usize) -> Self {
        Self {
            min_workers,
            max_workers,
            current_workers: min_workers,
            ..Default::default()
        }
    }
    
    pub fn with_rendezvous(mut self, backend: &str, endpoint: &str) -> Self {
        self.rendezvous_backend = backend.to_string();
        self.rendezvous_endpoint = endpoint.to_string();
        self
    }
    
    pub fn with_max_restarts(mut self, max_restarts: usize) -> Self {
        self.max_restarts = max_restarts;
        self
    }
    
    pub fn with_batch_scaling(mut self, enabled: bool, base_size: usize) -> Self {
        self.auto_scale_batch_size = enabled;
        self.base_batch_size_per_worker = base_size;
        self
    }
    
    /// Calculate effective batch size based on current workers.
    pub fn effective_batch_size(&self) -> usize {
        if self.auto_scale_batch_size {
            self.base_batch_size_per_worker * self.current_workers
        } else {
            self.base_batch_size_per_worker
        }
    }
    
    /// Check if we have enough workers to continue.
    pub fn can_continue(&self) -> bool {
        self.current_workers >= self.min_workers
    }
}

// ============================================================================
// Worker State
// ============================================================================

/// State of a distributed worker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerState {
    /// Worker is initializing
    Initializing,
    /// Worker is ready and healthy
    Ready,
    /// Worker is running training
    Running,
    /// Worker is paused (e.g., during checkpoint)
    Paused,
    /// Worker has failed
    Failed { error: String },
    /// Worker is shutting down gracefully
    ShuttingDown,
    /// Worker has terminated
    Terminated,
}

/// Information about a worker.
#[derive(Clone, Debug)]
pub struct WorkerInfo {
    /// Worker ID (rank)
    pub worker_id: usize,
    /// Worker state
    pub state: WorkerState,
    /// Last heartbeat timestamp (Unix epoch millis)
    pub last_heartbeat: u64,
    /// Number of restarts
    pub restart_count: usize,
    /// Device this worker is using
    pub device: String,
    /// Hostname
    pub hostname: String,
    /// Current training step
    pub training_step: u64,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl WorkerInfo {
    pub fn new(worker_id: usize, device: &str, hostname: &str) -> Self {
        Self {
            worker_id,
            state: WorkerState::Initializing,
            last_heartbeat: current_time_millis(),
            restart_count: 0,
            device: device.to_string(),
            hostname: hostname.to_string(),
            training_step: 0,
            metadata: HashMap::new(),
        }
    }
    
    pub fn update_heartbeat(&mut self) {
        self.last_heartbeat = current_time_millis();
    }
    
    pub fn time_since_heartbeat(&self) -> Duration {
        let now = current_time_millis();
        Duration::from_millis(now.saturating_sub(self.last_heartbeat))
    }
    
    pub fn is_healthy(&self, timeout: Duration) -> bool {
        matches!(self.state, WorkerState::Ready | WorkerState::Running | WorkerState::Paused)
            && self.time_since_heartbeat() < timeout
    }
}

/// Get current time in milliseconds since Unix epoch.
fn current_time_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ============================================================================
// Heartbeat Monitor
// ============================================================================

/// Heartbeat monitor for detecting worker failures.
pub struct HeartbeatMonitor {
    /// Worker information
    workers: RwLock<HashMap<usize, WorkerInfo>>,
    /// Heartbeat interval
    interval: Duration,
    /// Timeout before declaring worker dead
    timeout: Duration,
    /// Whether monitor is running
    running: AtomicBool,
    /// Failure callbacks
    failure_handlers: Mutex<Vec<Box<dyn Fn(usize) + Send + Sync>>>,
    /// Recovery callbacks
    recovery_handlers: Mutex<Vec<Box<dyn Fn(usize) + Send + Sync>>>,
}

impl HeartbeatMonitor {
    pub fn new(interval: Duration, timeout: Duration) -> Arc<Self> {
        Arc::new(Self {
            workers: RwLock::new(HashMap::new()),
            interval,
            timeout,
            running: AtomicBool::new(false),
            failure_handlers: Mutex::new(Vec::new()),
            recovery_handlers: Mutex::new(Vec::new()),
        })
    }
    
    /// Register a worker.
    pub fn register_worker(&self, info: WorkerInfo) {
        let mut workers = self.workers.write().unwrap();
        workers.insert(info.worker_id, info);
    }
    
    /// Unregister a worker.
    pub fn unregister_worker(&self, worker_id: usize) {
        let mut workers = self.workers.write().unwrap();
        workers.remove(&worker_id);
    }
    
    /// Record heartbeat from worker.
    pub fn heartbeat(&self, worker_id: usize, training_step: u64) {
        let mut workers = self.workers.write().unwrap();
        if let Some(info) = workers.get_mut(&worker_id) {
            info.update_heartbeat();
            info.training_step = training_step;
            if matches!(info.state, WorkerState::Failed { .. }) {
                info.state = WorkerState::Running;
                // Trigger recovery handlers
                drop(workers);
                self.trigger_recovery(worker_id);
            }
        }
    }
    
    /// Set worker state.
    pub fn set_state(&self, worker_id: usize, state: WorkerState) {
        let mut workers = self.workers.write().unwrap();
        if let Some(info) = workers.get_mut(&worker_id) {
            info.state = state;
        }
    }
    
    /// Get all healthy workers.
    pub fn get_healthy_workers(&self) -> Vec<usize> {
        let workers = self.workers.read().unwrap();
        workers
            .iter()
            .filter(|(_, info)| info.is_healthy(self.timeout))
            .map(|(id, _)| *id)
            .collect()
    }
    
    /// Get all failed workers.
    pub fn get_failed_workers(&self) -> Vec<usize> {
        let workers = self.workers.read().unwrap();
        workers
            .iter()
            .filter(|(_, info)| !info.is_healthy(self.timeout))
            .map(|(id, _)| *id)
            .collect()
    }
    
    /// Get worker count.
    pub fn worker_count(&self) -> usize {
        self.workers.read().unwrap().len()
    }
    
    /// Get healthy worker count.
    pub fn healthy_worker_count(&self) -> usize {
        self.get_healthy_workers().len()
    }
    
    /// Add failure handler.
    pub fn on_failure(&self, handler: Box<dyn Fn(usize) + Send + Sync>) {
        self.failure_handlers.lock().unwrap().push(handler);
    }
    
    /// Add recovery handler.
    pub fn on_recovery(&self, handler: Box<dyn Fn(usize) + Send + Sync>) {
        self.recovery_handlers.lock().unwrap().push(handler);
    }
    
    fn trigger_failure(&self, worker_id: usize) {
        let handlers = self.failure_handlers.lock().unwrap();
        for handler in handlers.iter() {
            handler(worker_id);
        }
    }
    
    fn trigger_recovery(&self, worker_id: usize) {
        let handlers = self.recovery_handlers.lock().unwrap();
        for handler in handlers.iter() {
            handler(worker_id);
        }
    }
    
    /// Check all workers and trigger failure handlers for dead workers.
    pub fn check_workers(&self) -> Vec<usize> {
        let failed = self.get_failed_workers();
        for worker_id in &failed {
            let mut workers = self.workers.write().unwrap();
            if let Some(info) = workers.get_mut(worker_id) {
                if !matches!(info.state, WorkerState::Failed { .. }) {
                    info.state = WorkerState::Failed {
                        error: "Heartbeat timeout".to_string(),
                    };
                    drop(workers);
                    self.trigger_failure(*worker_id);
                }
            }
        }
        failed
    }
    
    /// Start monitoring in background thread.
    pub fn start_monitoring(self: Arc<Self>) -> JoinHandle<()> {
        self.running.store(true, Ordering::SeqCst);
        let monitor = self.clone();
        
        thread::spawn(move || {
            while monitor.running.load(Ordering::SeqCst) {
                monitor.check_workers();
                thread::sleep(monitor.interval);
            }
        })
    }
    
    /// Stop monitoring.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
    
    /// Check if monitoring is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

// ============================================================================
// Preemption Handler
// ============================================================================

/// Signal types that can trigger preemption handling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreemptionSignal {
    /// SIGTERM - graceful termination
    SigTerm,
    /// SIGINT - interrupt
    SigInt,
    /// SIGUSR1 - user-defined, often used for checkpoint
    SigUsr1,
    /// SIGUSR2 - user-defined
    SigUsr2,
    /// Spot instance preemption warning
    SpotPreemption,
    /// Cloud provider maintenance
    Maintenance,
    /// Custom signal
    Custom(i32),
}

/// Handler for preemption events.
pub struct PreemptionHandler {
    /// Flag indicating preemption was received
    preemption_received: AtomicBool,
    /// The signal that triggered preemption
    signal: Mutex<Option<PreemptionSignal>>,
    /// Timestamp when preemption was received
    preemption_time: AtomicU64,
    /// Grace period for shutdown
    grace_period: Duration,
    /// Checkpoint callback
    checkpoint_callback: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    /// Cleanup callbacks
    cleanup_callbacks: Mutex<Vec<Box<dyn Fn() + Send + Sync>>>,
}

impl PreemptionHandler {
    pub fn new(grace_period: Duration) -> Arc<Self> {
        Arc::new(Self {
            preemption_received: AtomicBool::new(false),
            signal: Mutex::new(None),
            preemption_time: AtomicU64::new(0),
            grace_period,
            checkpoint_callback: Mutex::new(None),
            cleanup_callbacks: Mutex::new(Vec::new()),
        })
    }
    
    /// Check if preemption was signaled.
    pub fn is_preempted(&self) -> bool {
        self.preemption_received.load(Ordering::SeqCst)
    }
    
    /// Signal preemption.
    pub fn signal_preemption(&self, signal: PreemptionSignal) {
        if !self.preemption_received.swap(true, Ordering::SeqCst) {
            *self.signal.lock().unwrap() = Some(signal);
            self.preemption_time.store(current_time_millis(), Ordering::SeqCst);
        }
    }
    
    /// Get the preemption signal.
    pub fn get_signal(&self) -> Option<PreemptionSignal> {
        *self.signal.lock().unwrap()
    }
    
    /// Time remaining until forced termination.
    pub fn time_remaining(&self) -> Duration {
        let preempt_time = self.preemption_time.load(Ordering::SeqCst);
        if preempt_time == 0 {
            return self.grace_period;
        }
        
        let elapsed = Duration::from_millis(current_time_millis() - preempt_time);
        self.grace_period.saturating_sub(elapsed)
    }
    
    /// Set checkpoint callback.
    pub fn set_checkpoint_callback(&self, callback: Box<dyn Fn() + Send + Sync>) {
        *self.checkpoint_callback.lock().unwrap() = Some(callback);
    }
    
    /// Add cleanup callback.
    pub fn add_cleanup_callback(&self, callback: Box<dyn Fn() + Send + Sync>) {
        self.cleanup_callbacks.lock().unwrap().push(callback);
    }
    
    /// Handle preemption: checkpoint and cleanup.
    pub fn handle_preemption(&self) {
        // First checkpoint
        if let Some(ref callback) = *self.checkpoint_callback.lock().unwrap() {
            callback();
        }
        
        // Then cleanup
        for callback in self.cleanup_callbacks.lock().unwrap().iter() {
            callback();
        }
    }
    
    /// Reset preemption state.
    pub fn reset(&self) {
        self.preemption_received.store(false, Ordering::SeqCst);
        *self.signal.lock().unwrap() = None;
        self.preemption_time.store(0, Ordering::SeqCst);
    }
}

// ============================================================================
// Graceful Degradation
// ============================================================================

/// Strategy for handling worker failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegradationStrategy {
    /// Abort training immediately
    Abort,
    /// Continue with remaining workers if above minimum
    ContinueWithLess,
    /// Wait for workers to recover
    WaitForRecovery { timeout_secs: u64 },
    /// Try to restart failed workers
    RestartFailed { max_attempts: usize },
}

impl Default for DegradationStrategy {
    fn default() -> Self {
        Self::ContinueWithLess
    }
}

/// Graceful degradation manager.
pub struct GracefulDegradation {
    /// Elastic config
    config: RwLock<ElasticConfig>,
    /// Degradation strategy
    strategy: RwLock<DegradationStrategy>,
    /// Heartbeat monitor
    monitor: Arc<HeartbeatMonitor>,
    /// Current active workers
    active_workers: RwLock<HashSet<usize>>,
    /// Training state
    training_active: AtomicBool,
    /// Restart counts per worker
    restart_counts: RwLock<HashMap<usize, usize>>,
}

impl GracefulDegradation {
    pub fn new(
        config: ElasticConfig,
        strategy: DegradationStrategy,
        monitor: Arc<HeartbeatMonitor>,
    ) -> Self {
        Self {
            config: RwLock::new(config),
            strategy: RwLock::new(strategy),
            monitor,
            active_workers: RwLock::new(HashSet::new()),
            training_active: AtomicBool::new(false),
            restart_counts: RwLock::new(HashMap::new()),
        }
    }
    
    /// Start training with initial workers.
    pub fn start_training(&self, worker_ids: &[usize]) {
        let mut active = self.active_workers.write().unwrap();
        active.clear();
        active.extend(worker_ids);
        self.training_active.store(true, Ordering::SeqCst);
        
        // Update config
        let mut config = self.config.write().unwrap();
        config.current_workers = worker_ids.len();
    }
    
    /// Handle worker failure.
    pub fn handle_failure(&self, worker_id: usize) -> FailureAction {
        let strategy = *self.strategy.read().unwrap();
        let config = self.config.read().unwrap();
        
        // Remove from active workers
        {
            let mut active = self.active_workers.write().unwrap();
            active.remove(&worker_id);
        }
        
        let remaining = self.active_workers.read().unwrap().len();
        
        match strategy {
            DegradationStrategy::Abort => FailureAction::AbortTraining,
            
            DegradationStrategy::ContinueWithLess => {
                if remaining >= config.min_workers {
                    // Update batch size if auto-scaling
                    drop(config);
                    self.update_worker_count(remaining);
                    FailureAction::ContinueWithLess { remaining_workers: remaining }
                } else {
                    FailureAction::AbortTraining
                }
            }
            
            DegradationStrategy::WaitForRecovery { timeout_secs } => {
                FailureAction::WaitForRecovery {
                    timeout: Duration::from_secs(timeout_secs),
                }
            }
            
            DegradationStrategy::RestartFailed { max_attempts } => {
                let mut counts = self.restart_counts.write().unwrap();
                let count = counts.entry(worker_id).or_insert(0);
                *count += 1;
                
                if *count <= max_attempts {
                    FailureAction::RestartWorker {
                        worker_id,
                        attempt: *count,
                    }
                } else if remaining >= config.min_workers {
                    drop(config);
                    self.update_worker_count(remaining);
                    FailureAction::ContinueWithLess { remaining_workers: remaining }
                } else {
                    FailureAction::AbortTraining
                }
            }
        }
    }
    
    /// Handle worker recovery.
    pub fn handle_recovery(&self, worker_id: usize) -> RecoveryAction {
        let mut active = self.active_workers.write().unwrap();
        
        if active.contains(&worker_id) {
            return RecoveryAction::AlreadyActive;
        }
        
        let config = self.config.read().unwrap();
        if active.len() < config.max_workers {
            active.insert(worker_id);
            let new_count = active.len();
            drop(active);
            drop(config);
            
            self.update_worker_count(new_count);
            RecoveryAction::AddedToPool { new_worker_count: new_count }
        } else {
            RecoveryAction::PoolFull
        }
    }
    
    /// Update worker count and reconfigure.
    fn update_worker_count(&self, count: usize) {
        let mut config = self.config.write().unwrap();
        config.current_workers = count;
    }
    
    /// Get current effective batch size.
    pub fn effective_batch_size(&self) -> usize {
        self.config.read().unwrap().effective_batch_size()
    }
    
    /// Get current worker count.
    pub fn worker_count(&self) -> usize {
        self.active_workers.read().unwrap().len()
    }
    
    /// Check if training can continue.
    pub fn can_continue(&self) -> bool {
        self.config.read().unwrap().can_continue()
    }
    
    /// Get active worker IDs.
    pub fn active_worker_ids(&self) -> Vec<usize> {
        self.active_workers.read().unwrap().iter().copied().collect()
    }
}

/// Action to take on worker failure.
#[derive(Clone, Debug)]
pub enum FailureAction {
    /// Abort training immediately
    AbortTraining,
    /// Continue with remaining workers
    ContinueWithLess { remaining_workers: usize },
    /// Wait for worker to recover
    WaitForRecovery { timeout: Duration },
    /// Attempt to restart the failed worker
    RestartWorker { worker_id: usize, attempt: usize },
}

/// Action to take on worker recovery.
#[derive(Clone, Debug)]
pub enum RecoveryAction {
    /// Worker added back to pool
    AddedToPool { new_worker_count: usize },
    /// Worker already active
    AlreadyActive,
    /// Pool is at maximum capacity
    PoolFull,
}

// ============================================================================
// Elastic Trainer
// ============================================================================

/// Elastic trainer with fault tolerance.
pub struct ElasticTrainer {
    /// Elastic configuration
    config: ElasticConfig,
    /// This worker's rank
    rank: usize,
    /// World size (total workers)
    world_size: AtomicUsize,
    /// Heartbeat monitor
    monitor: Arc<HeartbeatMonitor>,
    /// Preemption handler
    preemption: Arc<PreemptionHandler>,
    /// Graceful degradation
    degradation: GracefulDegradation,
    /// Current training step
    training_step: AtomicU64,
    /// Number of restarts
    restart_count: AtomicUsize,
    /// Is training active
    is_training: AtomicBool,
}

impl ElasticTrainer {
    pub fn new(config: ElasticConfig, rank: usize) -> Self {
        let monitor = HeartbeatMonitor::new(
            Duration::from_secs(5),
            Duration::from_secs(30),
        );
        let preemption = PreemptionHandler::new(Duration::from_secs(60));
        let degradation = GracefulDegradation::new(
            config.clone(),
            DegradationStrategy::ContinueWithLess,
            monitor.clone(),
        );
        
        Self {
            config: config.clone(),
            rank,
            world_size: AtomicUsize::new(config.current_workers),
            monitor,
            preemption,
            degradation,
            training_step: AtomicU64::new(0),
            restart_count: AtomicUsize::new(0),
            is_training: AtomicBool::new(false),
        }
    }
    
    /// Initialize elastic training.
    pub fn initialize(&self) -> Result<()> {
        // Get hostname using std::process::Command
        let hostname = std::process::Command::new("hostname")
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|_| format!("worker-{}", self.rank));
        
        let info = WorkerInfo::new(self.rank, "cuda", &hostname);
        self.monitor.register_worker(info);
        self.monitor.set_state(self.rank, WorkerState::Ready);
        
        // Start heartbeat monitoring
        self.monitor.clone().start_monitoring();
        
        Ok(())
    }
    
    /// Start training loop.
    pub fn start_training(&self, worker_ids: &[usize]) {
        self.degradation.start_training(worker_ids);
        self.is_training.store(true, Ordering::SeqCst);
        self.monitor.set_state(self.rank, WorkerState::Running);
    }
    
    /// Send heartbeat.
    pub fn heartbeat(&self) {
        let step = self.training_step.load(Ordering::SeqCst);
        self.monitor.heartbeat(self.rank, step);
    }
    
    /// Advance training step.
    pub fn step(&self) {
        self.training_step.fetch_add(1, Ordering::SeqCst);
        self.heartbeat();
    }
    
    /// Check if should checkpoint (preemption or periodic).
    pub fn should_checkpoint(&self) -> bool {
        self.preemption.is_preempted()
    }
    
    /// Check if training should continue.
    pub fn should_continue(&self) -> bool {
        !self.preemption.is_preempted() && self.degradation.can_continue()
    }
    
    /// Handle preemption signal.
    pub fn handle_signal(&self, signal: PreemptionSignal) {
        self.preemption.signal_preemption(signal);
    }
    
    /// Get current world size.
    pub fn world_size(&self) -> usize {
        self.world_size.load(Ordering::SeqCst)
    }
    
    /// Get current training step.
    pub fn current_step(&self) -> u64 {
        self.training_step.load(Ordering::SeqCst)
    }
    
    /// Get effective batch size.
    pub fn effective_batch_size(&self) -> usize {
        self.degradation.effective_batch_size()
    }
    
    /// Shutdown gracefully.
    pub fn shutdown(&self) {
        self.monitor.set_state(self.rank, WorkerState::ShuttingDown);
        self.is_training.store(false, Ordering::SeqCst);
        
        // Handle any preemption cleanup
        if self.preemption.is_preempted() {
            self.preemption.handle_preemption();
        }
        
        self.monitor.stop();
        self.monitor.set_state(self.rank, WorkerState::Terminated);
    }
    
    /// Get restart count.
    pub fn restart_count(&self) -> usize {
        self.restart_count.load(Ordering::SeqCst)
    }
    
    /// Get monitor reference.
    pub fn monitor(&self) -> &Arc<HeartbeatMonitor> {
        &self.monitor
    }
    
    /// Get preemption handler reference.
    pub fn preemption_handler(&self) -> &Arc<PreemptionHandler> {
        &self.preemption
    }
}

// ============================================================================
// Failure Injection for Testing
// ============================================================================

/// Failure injection for testing fault tolerance.
pub struct FailureInjector {
    /// Probability of injecting failure (0.0 to 1.0)
    failure_probability: f64,
    /// Types of failures to inject
    failure_types: Vec<InjectedFailure>,
    /// Random seed
    seed: u64,
    /// Counter for deterministic failures
    counter: AtomicU64,
    /// Enabled flag
    enabled: AtomicBool,
}

/// Types of failures that can be injected.
#[derive(Clone, Debug)]
pub enum InjectedFailure {
    /// Simulate worker crash
    WorkerCrash,
    /// Simulate network partition
    NetworkPartition { duration_secs: u64 },
    /// Simulate slow worker (straggler)
    Straggler { slowdown_factor: f64 },
    /// Simulate OOM
    OutOfMemory,
    /// Simulate CUDA error
    CudaError,
}

impl FailureInjector {
    pub fn new(failure_probability: f64, seed: u64) -> Self {
        Self {
            failure_probability,
            failure_types: vec![InjectedFailure::WorkerCrash],
            seed,
            counter: AtomicU64::new(0),
            enabled: AtomicBool::new(false),
        }
    }
    
    pub fn with_failure_types(mut self, types: Vec<InjectedFailure>) -> Self {
        self.failure_types = types;
        self
    }
    
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::SeqCst);
    }
    
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
    }
    
    /// Check if should inject failure this step.
    pub fn should_inject(&self) -> bool {
        if !self.enabled.load(Ordering::SeqCst) {
            return false;
        }
        
        let count = self.counter.fetch_add(1, Ordering::SeqCst);
        // Simple pseudo-random based on counter and seed
        let hash = (count ^ self.seed).wrapping_mul(0x517cc1b727220a95);
        let rand = (hash as f64) / (u64::MAX as f64);
        
        rand < self.failure_probability
    }
    
    /// Get a random failure type.
    pub fn get_failure(&self) -> Option<&InjectedFailure> {
        if self.failure_types.is_empty() {
            return None;
        }
        
        let count = self.counter.load(Ordering::SeqCst);
        let idx = (count as usize) % self.failure_types.len();
        self.failure_types.get(idx)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_elastic_config() {
        let config = ElasticConfig::new(2, 8)
            .with_max_restarts(5)
            .with_batch_scaling(true, 32);
        
        assert_eq!(config.min_workers, 2);
        assert_eq!(config.max_workers, 8);
        assert_eq!(config.max_restarts, 5);
        assert_eq!(config.effective_batch_size(), 64); // 32 * 2 workers
    }
    
    #[test]
    fn test_worker_info() {
        let mut info = WorkerInfo::new(0, "cuda:0", "localhost");
        assert_eq!(info.worker_id, 0);
        assert!(matches!(info.state, WorkerState::Initializing));
        
        info.update_heartbeat();
        assert!(info.time_since_heartbeat() < Duration::from_secs(1));
    }
    
    #[test]
    fn test_heartbeat_monitor() {
        let monitor = HeartbeatMonitor::new(
            Duration::from_millis(100),
            Duration::from_secs(1),
        );
        
        let info = WorkerInfo::new(0, "cuda:0", "localhost");
        monitor.register_worker(info);
        
        assert_eq!(monitor.worker_count(), 1);
        
        // Set worker to Running state (new workers start in Initializing)
        monitor.set_state(0, WorkerState::Running);
        monitor.heartbeat(0, 100);
        assert_eq!(monitor.healthy_worker_count(), 1);
    }
    
    #[test]
    fn test_preemption_handler() {
        let handler = PreemptionHandler::new(Duration::from_secs(60));
        
        assert!(!handler.is_preempted());
        
        handler.signal_preemption(PreemptionSignal::SigTerm);
        
        assert!(handler.is_preempted());
        assert_eq!(handler.get_signal(), Some(PreemptionSignal::SigTerm));
        assert!(handler.time_remaining() <= Duration::from_secs(60));
    }
    
    #[test]
    fn test_graceful_degradation() {
        let config = ElasticConfig::new(2, 4);
        let monitor = HeartbeatMonitor::new(
            Duration::from_millis(100),
            Duration::from_secs(1),
        );
        
        let degradation = GracefulDegradation::new(
            config,
            DegradationStrategy::ContinueWithLess,
            monitor,
        );
        
        degradation.start_training(&[0, 1, 2, 3]);
        assert_eq!(degradation.worker_count(), 4);
        
        // Simulate worker failure
        let action = degradation.handle_failure(3);
        assert!(matches!(action, FailureAction::ContinueWithLess { remaining_workers: 3 }));
        assert_eq!(degradation.worker_count(), 3);
        
        // Fail another, still above minimum
        let action = degradation.handle_failure(2);
        assert!(matches!(action, FailureAction::ContinueWithLess { remaining_workers: 2 }));
        
        // Fail to below minimum
        let action = degradation.handle_failure(1);
        assert!(matches!(action, FailureAction::AbortTraining));
    }
    
    #[test]
    fn test_elastic_trainer_creation() {
        let config = ElasticConfig::new(1, 4);
        let trainer = ElasticTrainer::new(config, 0);
        
        assert_eq!(trainer.rank, 0);
        assert_eq!(trainer.world_size(), 1);
        assert_eq!(trainer.current_step(), 0);
    }
    
    #[test]
    fn test_failure_injector() {
        let injector = FailureInjector::new(1.0, 42)
            .with_failure_types(vec![
                InjectedFailure::WorkerCrash,
                InjectedFailure::OutOfMemory,
            ]);
        
        assert!(!injector.should_inject()); // Not enabled
        
        injector.enable();
        assert!(injector.should_inject()); // 100% probability
        
        injector.disable();
        assert!(!injector.should_inject());
    }
    
    #[test]
    fn test_degradation_strategy_restart() {
        let config = ElasticConfig::new(1, 4);
        let monitor = HeartbeatMonitor::new(
            Duration::from_millis(100),
            Duration::from_secs(1),
        );
        
        let degradation = GracefulDegradation::new(
            config,
            DegradationStrategy::RestartFailed { max_attempts: 2 },
            monitor,
        );
        
        degradation.start_training(&[0, 1]);
        
        // First failure should trigger restart
        let action = degradation.handle_failure(1);
        assert!(matches!(action, FailureAction::RestartWorker { worker_id: 1, attempt: 1 }));
        
        // Add worker back
        degradation.handle_recovery(1);
        
        // Second failure
        let action = degradation.handle_failure(1);
        assert!(matches!(action, FailureAction::RestartWorker { worker_id: 1, attempt: 2 }));
        
        // Third failure exceeds max, should continue with less
        degradation.handle_recovery(1);
        let action = degradation.handle_failure(1);
        assert!(matches!(action, FailureAction::ContinueWithLess { .. }));
    }
}
