//! Fault Tolerance and Elastic Training for Distributed Training.
//!
//! This module provides comprehensive fault tolerance capabilities:
//! - Elastic training with dynamic worker scaling
//! - Heartbeat monitoring for health checks
//! - Preemption handling for spot instances
//! - Graceful degradation on failures
//! - Automatic recovery mechanisms
//! - RetryManager with exponential backoff (F1)
//! - NaN loss detection and rollback (F6)
//! - Checkpoint validation (F7)
//! - Health check HTTP endpoint (F8)
//! - Cross-backend coordination (F9)
//! - Retry budget tracking (F10)
//!
//! Reference: TorchElastic-style fault tolerance for Rust training.

use candle_core::{Result, Tensor, Device, DType};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock, atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering}};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::thread::{self, JoinHandle};
use std::path::Path;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

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
// F1: RetryManager with Exponential Backoff
// ============================================================================

/// Types of failures that can occur during training.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FailureType {
    /// Out of memory
    Oom,
    /// NaN or Inf loss
    NanLoss,
    /// Loss divergence (too high)
    Divergence,
    /// Timeout
    Timeout,
    /// Network error
    Network,
    /// Preemption
    Preemption,
    /// Corrupt checkpoint
    CheckpointCorrupt,
    /// Unknown error
    Unknown,
}

/// Record of a training failure.
#[derive(Clone, Debug)]
pub struct FailureRecord {
    pub failure_type: FailureType,
    pub timestamp: u64,
    pub step: u64,
    pub loss: Option<f64>,
    pub error_message: String,
    pub backend: String,
    pub rank: usize,
}

impl FailureRecord {
    pub fn new(
        failure_type: FailureType,
        step: u64,
        loss: Option<f64>,
        error_message: &str,
        backend: &str,
        rank: usize,
    ) -> Self {
        Self {
            failure_type,
            timestamp: current_time_millis(),
            step,
            loss,
            error_message: error_message.to_string(),
            backend: backend.to_string(),
            rank,
        }
    }
}

/// Retry manager with exponential backoff (F1).
pub struct RetryManager {
    /// Maximum retry attempts
    max_attempts: usize,
    /// Base delay in milliseconds
    base_delay_ms: u64,
    /// Maximum delay in milliseconds
    max_delay_ms: u64,
    /// Backoff factor
    backoff_factor: f64,
    /// Current attempt count
    attempt_count: AtomicUsize,
    /// Failure history
    failure_history: Mutex<Vec<FailureRecord>>,
    /// Total retry time in milliseconds
    total_retry_time_ms: AtomicU64,
}

impl RetryManager {
    pub fn new(max_attempts: usize) -> Self {
        Self {
            max_attempts,
            base_delay_ms: 1000,
            max_delay_ms: 60000,
            backoff_factor: 2.0,
            attempt_count: AtomicUsize::new(0),
            failure_history: Mutex::new(Vec::new()),
            total_retry_time_ms: AtomicU64::new(0),
        }
    }
    
    pub fn with_delays(mut self, base_ms: u64, max_ms: u64, factor: f64) -> Self {
        self.base_delay_ms = base_ms;
        self.max_delay_ms = max_ms;
        self.backoff_factor = factor;
        self
    }
    
    /// Check if should retry after failure.
    pub fn should_retry(&self, error: &str, step: u64, loss: Option<f64>) -> bool {
        let attempt = self.attempt_count.fetch_add(1, Ordering::SeqCst) + 1;
        
        // Classify and record failure
        let failure_type = self.classify_failure(error, loss);
        let record = FailureRecord::new(
            failure_type.clone(),
            step,
            loss,
            error,
            "rust",
            0,
        );
        self.failure_history.lock().unwrap().push(record);
        
        // Don't retry fatal errors
        if matches!(failure_type, FailureType::CheckpointCorrupt) {
            return false;
        }
        
        attempt < self.max_attempts
    }
    
    /// Get delay before next retry using exponential backoff.
    pub fn get_retry_delay(&self) -> Duration {
        let attempt = self.attempt_count.load(Ordering::SeqCst);
        let delay = (self.base_delay_ms as f64) * self.backoff_factor.powi(attempt as i32 - 1);
        let delay_ms = (delay as u64).min(self.max_delay_ms);
        
        self.total_retry_time_ms.fetch_add(delay_ms, Ordering::SeqCst);
        Duration::from_millis(delay_ms)
    }
    
    /// Classify failure type from error message.
    fn classify_failure(&self, error: &str, loss: Option<f64>) -> FailureType {
        let error_lower = error.to_lowercase();
        
        // Check loss first
        if let Some(l) = loss {
            if l.is_nan() || l.is_infinite() {
                return FailureType::NanLoss;
            }
            if l > 100.0 {
                return FailureType::Divergence;
            }
        }
        
        if error_lower.contains("out of memory") || error_lower.contains("oom") {
            FailureType::Oom
        } else if error_lower.contains("nan") || error_lower.contains("inf") {
            FailureType::NanLoss
        } else if error_lower.contains("timeout") || error_lower.contains("timed out") {
            FailureType::Timeout
        } else if error_lower.contains("connection") || error_lower.contains("network") {
            FailureType::Network
        } else if error_lower.contains("checkpoint") && error_lower.contains("corrupt") {
            FailureType::CheckpointCorrupt
        } else {
            FailureType::Unknown
        }
    }
    
    /// Execute function with retry logic.
    pub fn execute_with_retry<F, T, E>(&self, mut f: F, step: u64) -> std::result::Result<T, E>
    where
        F: FnMut() -> std::result::Result<T, E>,
        E: std::fmt::Display,
    {
        loop {
            match f() {
                Ok(result) => {
                    self.attempt_count.store(0, Ordering::SeqCst);
                    return Ok(result);
                }
                Err(e) => {
                    let error_str = e.to_string();
                    if !self.should_retry(&error_str, step, None) {
                        return Err(e);
                    }
                    
                    let delay = self.get_retry_delay();
                    thread::sleep(delay);
                }
            }
        }
    }
    
    /// Get statistics.
    pub fn stats(&self) -> (usize, usize, u64) {
        (
            self.attempt_count.load(Ordering::SeqCst),
            self.failure_history.lock().unwrap().len(),
            self.total_retry_time_ms.load(Ordering::SeqCst),
        )
    }
    
    /// Reset state.
    pub fn reset(&self) {
        self.attempt_count.store(0, Ordering::SeqCst);
        self.failure_history.lock().unwrap().clear();
        self.total_retry_time_ms.store(0, Ordering::SeqCst);
    }
}

// ============================================================================
// F6: NaN Loss Detection and Rollback
// ============================================================================

/// NaN loss detector with rollback capability (F6).
pub struct NaNLossDetector {
    /// Loss threshold for divergence warning
    loss_threshold: f64,
    /// Number of consecutive NaN before rollback
    nan_streak_limit: usize,
    /// Current NaN streak
    nan_streak: AtomicUsize,
    /// Last valid loss
    last_valid_loss: Mutex<Option<f64>>,
    /// Last valid checkpoint path
    last_valid_checkpoint: Mutex<Option<String>>,
    /// Loss history
    loss_history: Mutex<Vec<(u64, f64)>>,
}

impl NaNLossDetector {
    pub fn new(loss_threshold: f64, nan_streak_limit: usize) -> Self {
        Self {
            loss_threshold,
            nan_streak_limit,
            nan_streak: AtomicUsize::new(0),
            last_valid_loss: Mutex::new(None),
            last_valid_checkpoint: Mutex::new(None),
            loss_history: Mutex::new(Vec::new()),
        }
    }
    
    /// Check loss value and return action needed.
    /// Returns (is_valid, action) where action is "none", "warn", or "rollback".
    pub fn check_loss(&self, loss: f64, step: u64) -> (bool, &'static str) {
        // Check for NaN/Inf
        if loss.is_nan() || loss.is_infinite() {
            let streak = self.nan_streak.fetch_add(1, Ordering::SeqCst) + 1;
            
            if streak >= self.nan_streak_limit {
                return (false, "rollback");
            }
            return (false, "warn");
        }
        
        // Check for divergence
        if loss > self.loss_threshold {
            return (false, "warn");
        }
        
        // Valid loss - reset streak and update tracking
        self.nan_streak.store(0, Ordering::SeqCst);
        *self.last_valid_loss.lock().unwrap() = Some(loss);
        
        let mut history = self.loss_history.lock().unwrap();
        history.push((step, loss));
        if history.len() > 1000 {
            *history = history.split_off(500);
        }
        
        (true, "none")
    }
    
    /// Set valid checkpoint for potential rollback.
    pub fn set_valid_checkpoint(&self, path: &str) {
        *self.last_valid_checkpoint.lock().unwrap() = Some(path.to_string());
    }
    
    /// Get checkpoint path for rollback.
    pub fn get_rollback_checkpoint(&self) -> Option<String> {
        self.last_valid_checkpoint.lock().unwrap().clone()
    }
    
    /// Reset detector state.
    pub fn reset(&self) {
        self.nan_streak.store(0, Ordering::SeqCst);
        self.loss_history.lock().unwrap().clear();
    }
}

// ============================================================================
// F7: Checkpoint Validation
// ============================================================================

/// Checkpoint validator (F7).
pub struct CheckpointValidator;

impl CheckpointValidator {
    /// Find latest checkpoint in directory.
    pub fn find_latest_checkpoint(dir: &str) -> Option<String> {
        let path = Path::new(dir);
        if !path.exists() {
            return None;
        }
        
        let mut checkpoints: Vec<_> = fs::read_dir(path)
            .ok()?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                let name = entry.file_name().to_string_lossy().to_string();
                (name.starts_with("step_") || name.starts_with("checkpoint_"))
                    && (name.ends_with(".safetensors") || name.ends_with(".pt") || name.ends_with(".bin"))
            })
            .collect();
        
        if checkpoints.is_empty() {
            return None;
        }
        
        // Sort by step number
        checkpoints.sort_by_key(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            Self::extract_step(&name).unwrap_or(0)
        });
        
        checkpoints.last().map(|e| e.path().to_string_lossy().to_string())
    }
    
    /// Extract step number from checkpoint filename.
    fn extract_step(name: &str) -> Option<u64> {
        let name = name.replace("checkpoint_", "").replace("step_", "");
        name.split('_')
            .next()
            .and_then(|s| s.split('.').next())
            .and_then(|s| s.parse().ok())
    }
    
    /// Validate checkpoint file exists and has content.
    pub fn validate_checkpoint(path: &str) -> (bool, String) {
        let path = Path::new(path);
        
        if !path.exists() {
            return (false, "Checkpoint file does not exist".to_string());
        }
        
        match fs::metadata(path) {
            Ok(meta) => {
                if meta.len() == 0 {
                    return (false, "Checkpoint file is empty".to_string());
                }
                
                // Basic format validation
                let name = path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                    
                if name.ends_with(".safetensors") {
                    // Try to read safetensors header
                    match fs::File::open(path) {
                        Ok(mut file) => {
                            let mut header_size = [0u8; 8];
                            if file.read_exact(&mut header_size).is_ok() {
                                (true, String::new())
                            } else {
                                (false, "Cannot read safetensors header".to_string())
                            }
                        }
                        Err(e) => (false, format!("Cannot open file: {}", e)),
                    }
                } else {
                    // For .pt/.bin files, just check it's not empty
                    (true, String::new())
                }
            }
            Err(e) => (false, format!("Cannot read file metadata: {}", e)),
        }
    }
}

// ============================================================================
// F8: Health Check HTTP Endpoint
// ============================================================================

/// Training status for health endpoint.
#[derive(Clone, Debug)]
pub struct TrainingStatus {
    pub status: String,
    pub step: u64,
    pub loss: Option<f64>,
    pub uptime_secs: u64,
    pub retry_count: usize,
    pub backend: String,
}

impl Default for TrainingStatus {
    fn default() -> Self {
        Self {
            status: "unknown".to_string(),
            step: 0,
            loss: None,
            uptime_secs: 0,
            retry_count: 0,
            backend: "rust".to_string(),
        }
    }
}

/// Simple HTTP health check server (F8).
pub struct HealthCheckServer {
    port: u16,
    status: Arc<RwLock<TrainingStatus>>,
    running: Arc<AtomicBool>,
    start_time: Instant,
}

impl HealthCheckServer {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            status: Arc::new(RwLock::new(TrainingStatus::default())),
            running: Arc::new(AtomicBool::new(false)),
            start_time: Instant::now(),
        }
    }
    
    /// Start the health check server in background.
    pub fn start(&self) -> Option<JoinHandle<()>> {
        let listener = match TcpListener::bind(format!("0.0.0.0:{}", self.port)) {
            Ok(l) => {
                // Set non-blocking for responsive shutdown
                let _ = l.set_nonblocking(true);
                l
            },
            Err(_) => return None,
        };
        
        self.running.store(true, Ordering::SeqCst);
        let running = self.running.clone();
        let status = self.status.clone();
        let start_time = self.start_time;
        
        Some(thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let status = status.read().unwrap();
                        let uptime = start_time.elapsed().as_secs();
                        
                        let response_body = format!(
                            r#"{{"status":"{}","step":{},"loss":{},"uptime_secs":{},"retry_count":{},"backend":"{}","timestamp":{}}}"#,
                            status.status,
                            status.step,
                            status.loss.map(|l| l.to_string()).unwrap_or("null".to_string()),
                            uptime,
                            status.retry_count,
                            status.backend,
                            current_time_millis()
                        );
                        
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            response_body.len(),
                            response_body
                        );
                        
                        let _ = stream.write_all(response.as_bytes());
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        // No connection, sleep briefly
                        thread::sleep(Duration::from_millis(100));
                    }
                    Err(_) => break,
                }
            }
        }))
    }
    
    /// Stop the server.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
    
    /// Update status.
    pub fn update_status(&self, status: &str, step: u64, loss: Option<f64>, retry_count: usize) {
        let mut s = self.status.write().unwrap();
        s.status = status.to_string();
        s.step = step;
        s.loss = loss;
        s.retry_count = retry_count;
        s.uptime_secs = self.start_time.elapsed().as_secs();
    }
}

// ============================================================================
// F9: Cross-Backend Coordination
// ============================================================================

/// Cross-backend coordinator for handling failures across PyTorch/Rust (F9).
pub struct CrossBackendCoordinator {
    backends: Vec<String>,
    status: RwLock<HashMap<String, String>>,
    steps: RwLock<HashMap<String, u64>>,
    failed: RwLock<Vec<String>>,
    checkpoint_dir: String,
}

impl CrossBackendCoordinator {
    pub fn new(backends: Vec<String>, checkpoint_dir: &str) -> Self {
        let status: HashMap<String, String> = backends.iter()
            .map(|b| (b.clone(), "unknown".to_string()))
            .collect();
        let steps: HashMap<String, u64> = backends.iter()
            .map(|b| (b.clone(), 0))
            .collect();
        
        Self {
            backends,
            status: RwLock::new(status),
            steps: RwLock::new(steps),
            failed: RwLock::new(Vec::new()),
            checkpoint_dir: checkpoint_dir.to_string(),
        }
    }
    
    /// Report status from a backend.
    pub fn report_status(&self, backend: &str, status: &str, step: u64) {
        self.status.write().unwrap().insert(backend.to_string(), status.to_string());
        self.steps.write().unwrap().insert(backend.to_string(), step);
        
        if status == "failed" {
            let mut failed = self.failed.write().unwrap();
            if !failed.contains(&backend.to_string()) {
                failed.push(backend.to_string());
            }
        }
    }
    
    /// Check if training can continue with remaining backends.
    pub fn can_continue(&self) -> bool {
        let status = self.status.read().unwrap();
        status.values().any(|s| s != "failed" && s != "terminated")
    }
    
    /// Get healthy backends.
    pub fn get_healthy_backends(&self) -> Vec<String> {
        let status = self.status.read().unwrap();
        status.iter()
            .filter(|(_, s)| *s != "failed" && *s != "terminated")
            .map(|(b, _)| b.clone())
            .collect()
    }
    
    /// Find checkpoint from a healthy backend.
    pub fn sync_checkpoint_from_healthy(&self, failed_backend: &str) -> Option<String> {
        let healthy = self.get_healthy_backends();
        if healthy.is_empty() {
            return None;
        }
        
        let mut best_checkpoint: Option<String> = None;
        let mut best_step = 0u64;
        
        for backend in healthy {
            let dir = format!("{}/{}", self.checkpoint_dir, backend);
            if let Some(checkpoint) = CheckpointValidator::find_latest_checkpoint(&dir) {
                if let Some(step) = CheckpointValidator::extract_step(&checkpoint) {
                    if step > best_step {
                        best_step = step;
                        best_checkpoint = Some(checkpoint);
                    }
                }
            }
        }
        
        best_checkpoint
    }
}

// ============================================================================
// F10: Retry Budget Tracking
// ============================================================================

/// Retry budget tracker (F10).
pub struct RetryBudgetTracker {
    /// Maximum cost allowed for retries
    max_retry_cost: f64,
    /// Total retry cost so far
    total_retry_cost: AtomicU64, // Stored as cents to avoid float atomics
    /// Cost per GPU hour
    cost_per_gpu_hour: f64,
    /// GPU count
    gpu_count: usize,
    /// Retry start time
    retry_start: Mutex<Option<Instant>>,
    /// Retry history
    retry_history: Mutex<Vec<(Duration, f64, bool)>>, // duration, cost, success
}

impl RetryBudgetTracker {
    pub fn new(max_retry_cost: f64, cost_per_gpu_hour: f64, gpu_count: usize) -> Self {
        Self {
            max_retry_cost,
            total_retry_cost: AtomicU64::new(0),
            cost_per_gpu_hour,
            gpu_count,
            retry_start: Mutex::new(None),
            retry_history: Mutex::new(Vec::new()),
        }
    }
    
    /// Mark start of a retry attempt.
    pub fn start_retry(&self) {
        *self.retry_start.lock().unwrap() = Some(Instant::now());
    }
    
    /// Mark end of retry and calculate cost.
    pub fn end_retry(&self, success: bool) -> f64 {
        let start = self.retry_start.lock().unwrap().take();
        let duration = start.map(|s| s.elapsed()).unwrap_or_default();
        
        let hours = duration.as_secs_f64() / 3600.0;
        let cost = hours * self.cost_per_gpu_hour * self.gpu_count as f64;
        
        // Store as cents
        let cost_cents = (cost * 100.0) as u64;
        self.total_retry_cost.fetch_add(cost_cents, Ordering::SeqCst);
        
        self.retry_history.lock().unwrap().push((duration, cost, success));
        
        cost
    }
    
    /// Check if we can afford another retry.
    pub fn can_afford_retry(&self) -> bool {
        let total = self.total_retry_cost.load(Ordering::SeqCst) as f64 / 100.0;
        total < self.max_retry_cost
    }
    
    /// Get total retry cost.
    pub fn total_cost(&self) -> f64 {
        self.total_retry_cost.load(Ordering::SeqCst) as f64 / 100.0
    }
    
    /// Get remaining budget.
    pub fn remaining_budget(&self) -> f64 {
        self.max_retry_cost - self.total_cost()
    }
    
    /// Get retry count.
    pub fn retry_count(&self) -> usize {
        self.retry_history.lock().unwrap().len()
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
    
    // ========================================================================
    // F1: RetryManager Tests
    // ========================================================================
    
    #[test]
    fn test_retry_manager_creation() {
        let manager = RetryManager::new(3);
        assert_eq!(manager.max_attempts, 3);
        let (attempts, failures, _) = manager.stats();
        assert_eq!(attempts, 0);
        assert_eq!(failures, 0);
    }
    
    #[test]
    fn test_retry_manager_should_retry() {
        let manager = RetryManager::new(3);
        
        // First retry should be allowed
        assert!(manager.should_retry("some error", 100, None));
        assert!(manager.should_retry("another error", 101, None));
        
        // Third retry should not be allowed (at max)
        assert!(!manager.should_retry("third error", 102, None));
    }
    
    #[test]
    fn test_retry_manager_exponential_backoff() {
        let manager = RetryManager::new(5)
            .with_delays(1000, 60000, 2.0);
        
        manager.attempt_count.store(1, Ordering::SeqCst);
        let delay1 = manager.get_retry_delay();
        assert_eq!(delay1.as_millis(), 1000);
        
        manager.attempt_count.store(2, Ordering::SeqCst);
        let delay2 = manager.get_retry_delay();
        assert_eq!(delay2.as_millis(), 2000);
        
        manager.attempt_count.store(3, Ordering::SeqCst);
        let delay3 = manager.get_retry_delay();
        assert_eq!(delay3.as_millis(), 4000);
    }
    
    #[test]
    fn test_retry_manager_failure_classification() {
        let manager = RetryManager::new(3);
        
        assert_eq!(
            manager.classify_failure("CUDA out of memory", None),
            FailureType::Oom
        );
        assert_eq!(
            manager.classify_failure("connection timeout", None),
            FailureType::Timeout
        );
        assert_eq!(
            manager.classify_failure("network error", None),
            FailureType::Network
        );
        assert_eq!(
            manager.classify_failure("", Some(f64::NAN)),
            FailureType::NanLoss
        );
        assert_eq!(
            manager.classify_failure("", Some(200.0)),
            FailureType::Divergence
        );
    }
    
    // ========================================================================
    // F6: NaNLossDetector Tests
    // ========================================================================
    
    #[test]
    fn test_nan_detector_valid_loss() {
        let detector = NaNLossDetector::new(100.0, 3);
        
        let (valid, action) = detector.check_loss(1.5, 100);
        assert!(valid);
        assert_eq!(action, "none");
    }
    
    #[test]
    fn test_nan_detector_nan_loss() {
        let detector = NaNLossDetector::new(100.0, 3);
        
        let (valid, action) = detector.check_loss(f64::NAN, 100);
        assert!(!valid);
        assert_eq!(action, "warn");
    }
    
    #[test]
    fn test_nan_detector_streak_rollback() {
        let detector = NaNLossDetector::new(100.0, 3);
        
        detector.check_loss(f64::NAN, 100);
        detector.check_loss(f64::NAN, 101);
        let (valid, action) = detector.check_loss(f64::NAN, 102);
        
        assert!(!valid);
        assert_eq!(action, "rollback");
    }
    
    #[test]
    fn test_nan_detector_streak_reset() {
        let detector = NaNLossDetector::new(100.0, 3);
        
        detector.check_loss(f64::NAN, 100);
        detector.check_loss(f64::NAN, 101);
        detector.check_loss(1.0, 102); // Valid loss resets streak
        
        let (_, action) = detector.check_loss(f64::NAN, 103);
        assert_eq!(action, "warn"); // Back to warn, not rollback
    }
    
    #[test]
    fn test_nan_detector_checkpoint() {
        let detector = NaNLossDetector::new(100.0, 3);
        
        detector.set_valid_checkpoint("/checkpoints/step_100.pt");
        assert_eq!(
            detector.get_rollback_checkpoint(),
            Some("/checkpoints/step_100.pt".to_string())
        );
    }
    
    // ========================================================================
    // F9: CrossBackendCoordinator Tests
    // ========================================================================
    
    #[test]
    fn test_cross_backend_coordinator() {
        let coordinator = CrossBackendCoordinator::new(
            vec!["pytorch".to_string(), "rust".to_string()],
            "/checkpoints",
        );
        
        coordinator.report_status("pytorch", "running", 100);
        coordinator.report_status("rust", "running", 150);
        
        assert!(coordinator.can_continue());
        assert_eq!(coordinator.get_healthy_backends().len(), 2);
    }
    
    #[test]
    fn test_cross_backend_one_failed() {
        let coordinator = CrossBackendCoordinator::new(
            vec!["pytorch".to_string(), "rust".to_string()],
            "/checkpoints",
        );
        
        coordinator.report_status("pytorch", "running", 100);
        coordinator.report_status("rust", "failed", 50);
        
        assert!(coordinator.can_continue());
        let healthy = coordinator.get_healthy_backends();
        assert_eq!(healthy.len(), 1);
        assert!(healthy.contains(&"pytorch".to_string()));
    }
    
    #[test]
    fn test_cross_backend_all_failed() {
        let coordinator = CrossBackendCoordinator::new(
            vec!["pytorch".to_string(), "rust".to_string()],
            "/checkpoints",
        );
        
        coordinator.report_status("pytorch", "failed", 100);
        coordinator.report_status("rust", "failed", 50);
        
        assert!(!coordinator.can_continue());
    }
    
    // ========================================================================
    // F10: RetryBudgetTracker Tests
    // ========================================================================
    
    #[test]
    fn test_retry_budget_tracker() {
        let tracker = RetryBudgetTracker::new(50.0, 2.78, 1);
        
        assert!(tracker.can_afford_retry());
        assert_eq!(tracker.remaining_budget(), 50.0);
    }
    
    #[test]
    fn test_retry_budget_tracking() {
        let tracker = RetryBudgetTracker::new(100.0, 2.78, 1);
        
        tracker.start_retry();
        std::thread::sleep(Duration::from_millis(10));
        let cost = tracker.end_retry(true);
        
        assert!(cost > 0.0);
        assert!(cost < 1.0); // Very small cost for 10ms
        assert_eq!(tracker.retry_count(), 1);
    }
    
    #[test]
    fn test_retry_budget_exhausted() {
        let tracker = RetryBudgetTracker::new(0.001, 2.78, 1);
        
        // Simulate exhausting budget
        tracker.total_retry_cost.store(100, Ordering::SeqCst); // 1 dollar
        
        assert!(!tracker.can_afford_retry());
    }
}
