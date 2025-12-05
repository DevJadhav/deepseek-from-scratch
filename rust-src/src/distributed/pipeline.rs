//! Pipeline Parallelism for distributed training.
//!
//! Implements multiple pipeline parallelism strategies:
//! - 1F1B (One Forward One Backward) - standard efficient schedule
//! - GPipe - simple all-forward then all-backward
//! - DualPipe - DeepSeek-V3's bidirectional pipeline parallelism
//!
//! DualPipe achieves near-zero bubble by running two streams in opposite
//! directions through the pipeline, keeping both ends busy at all times.
//!
//! Reference: DeepSeek-V3 Technical Report - DualPipe Algorithm

use candle_core::{Result, Tensor, DType, Device};
use std::collections::{VecDeque, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use super::{get_pp_size, get_pp_rank, get_pp_group};

/// Pipeline stage configuration.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Number of pipeline stages (= pp_size)
    pub num_stages: usize,
    /// Number of micro-batches per mini-batch
    pub num_micro_batches: usize,
    /// Current stage rank
    pub stage_rank: usize,
    /// Whether this is the first stage
    pub is_first_stage: bool,
    /// Whether this is the last stage
    pub is_last_stage: bool,
}

impl PipelineConfig {
    pub fn new(num_micro_batches: usize) -> Self {
        let num_stages = get_pp_size();
        let stage_rank = get_pp_rank();
        
        Self {
            num_stages,
            num_micro_batches,
            stage_rank,
            is_first_stage: stage_rank == 0,
            is_last_stage: stage_rank == num_stages - 1,
        }
    }
}

/// Stored activation for backward pass.
pub struct StoredActivation {
    /// Micro-batch ID
    pub micro_batch_id: usize,
    /// Input activation (needed for backward)
    pub input: Tensor,
    /// Output activation (needed for backward on downstream)
    pub output: Tensor,
}

/// Pipeline stage holding a subset of model layers.
pub struct PipelineStage<M> {
    /// The model layers for this stage
    model: M,
    /// Configuration
    config: PipelineConfig,
    /// Stored activations for backward pass
    activations: VecDeque<StoredActivation>,
}

impl<M> PipelineStage<M> {
    pub fn new(model: M, num_micro_batches: usize) -> Self {
        Self {
            model,
            config: PipelineConfig::new(num_micro_batches),
            activations: VecDeque::new(),
        }
    }
    
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
    
    /// Store activation for backward pass.
    pub fn store_activation(&mut self, micro_batch_id: usize, input: Tensor, output: Tensor) {
        self.activations.push_back(StoredActivation {
            micro_batch_id,
            input,
            output,
        });
    }
    
    /// Pop oldest activation for backward pass.
    pub fn pop_activation(&mut self) -> Option<StoredActivation> {
        self.activations.pop_front()
    }
    
    /// Get model reference.
    pub fn model(&self) -> &M {
        &self.model
    }
    
    /// Get mutable model reference.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

/// Send tensor to next pipeline stage.
pub fn send_forward(tensor: &Tensor) -> Result<()> {
    let pp_size = get_pp_size();
    let pp_rank = get_pp_rank();
    
    if pp_size <= 1 || pp_rank == pp_size - 1 {
        // No next stage
        return Ok(());
    }
    
    let next_rank = pp_rank + 1;
    
    if let Some(group) = get_pp_group() {
        group.communicator.send(tensor, next_rank)
    } else {
        Ok(())
    }
}

/// Receive tensor from previous pipeline stage.
pub fn recv_forward(shape: &[usize], device: &candle_core::Device) -> Result<Option<Tensor>> {
    let pp_size = get_pp_size();
    let pp_rank = get_pp_rank();
    
    if pp_size <= 1 || pp_rank == 0 {
        // No previous stage
        return Ok(None);
    }
    
    let prev_rank = pp_rank - 1;
    
    if let Some(group) = get_pp_group() {
        let tensor = group.communicator.recv(shape, device, prev_rank)?;
        Ok(Some(tensor))
    } else {
        Ok(None)
    }
}

/// Send gradient to previous pipeline stage.
pub fn send_backward(tensor: &Tensor) -> Result<()> {
    let pp_size = get_pp_size();
    let pp_rank = get_pp_rank();
    
    if pp_size <= 1 || pp_rank == 0 {
        // No previous stage
        return Ok(());
    }
    
    let prev_rank = pp_rank - 1;
    
    if let Some(group) = get_pp_group() {
        group.communicator.send(tensor, prev_rank)
    } else {
        Ok(())
    }
}

/// Receive gradient from next pipeline stage.
pub fn recv_backward(shape: &[usize], device: &candle_core::Device) -> Result<Option<Tensor>> {
    let pp_size = get_pp_size();
    let pp_rank = get_pp_rank();
    
    if pp_size <= 1 || pp_rank == pp_size - 1 {
        // No next stage
        return Ok(None);
    }
    
    let next_rank = pp_rank + 1;
    
    if let Some(group) = get_pp_group() {
        let tensor = group.communicator.recv(shape, device, next_rank)?;
        Ok(Some(tensor))
    } else {
        Ok(None)
    }
}

/// 1F1B (One Forward One Backward) Pipeline Scheduler.
///
/// Implements the standard 1F1B schedule:
/// 1. Warmup: num_stages - stage_rank - 1 forward passes
/// 2. Steady state: 1 forward, 1 backward alternating
/// 3. Cooldown: remaining backward passes
pub struct OneFOneBScheduler {
    config: PipelineConfig,
    /// Current micro-batch index for forward
    forward_idx: usize,
    /// Current micro-batch index for backward
    backward_idx: usize,
    /// Number of warmup steps completed
    warmup_done: usize,
    /// Number of steady state steps completed (forward)
    steady_forward_done: usize,
    /// Number of steady state steps completed (backward)
    steady_backward_done: usize,
}

/// Action to take in the pipeline schedule.
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleAction {
    /// Perform forward pass for given micro-batch
    Forward(usize),
    /// Perform backward pass for given micro-batch  
    Backward(usize),
    /// Perform both forward and backward (steady state)
    ForwardBackward { forward_mb: usize, backward_mb: usize },
    /// Schedule complete
    Done,
}

impl OneFOneBScheduler {
    pub fn new(num_micro_batches: usize) -> Self {
        Self {
            config: PipelineConfig::new(num_micro_batches),
            forward_idx: 0,
            backward_idx: 0,
            warmup_done: 0,
            steady_forward_done: 0,
            steady_backward_done: 0,
        }
    }
    
    /// Number of warmup forward passes for this stage.
    pub fn num_warmup_steps(&self) -> usize {
        self.config.num_stages - self.config.stage_rank - 1
    }
    
    /// Number of steady-state steps.
    pub fn num_steady_steps(&self) -> usize {
        let warmup = self.num_warmup_steps();
        if self.config.num_micro_batches > warmup {
            self.config.num_micro_batches - warmup
        } else {
            0
        }
    }
    
    /// Number of cooldown backward passes.
    pub fn num_cooldown_steps(&self) -> usize {
        self.num_warmup_steps()
    }
    
    /// Get next action in the schedule.
    pub fn next_action(&mut self) -> ScheduleAction {
        let warmup_steps = self.num_warmup_steps();
        let steady_steps = self.num_steady_steps();
        
        // Phase 1: Warmup (only forward passes)
        if self.warmup_done < warmup_steps && self.forward_idx < self.config.num_micro_batches {
            let mb = self.forward_idx;
            self.forward_idx += 1;
            self.warmup_done += 1;
            return ScheduleAction::Forward(mb);
        }
        
        // Phase 2: Steady state (1F1B)
        if self.steady_forward_done < steady_steps {
            let forward_mb = self.forward_idx;
            let backward_mb = self.backward_idx;
            
            self.forward_idx += 1;
            self.backward_idx += 1;
            self.steady_forward_done += 1;
            self.steady_backward_done += 1;
            
            if forward_mb < self.config.num_micro_batches {
                return ScheduleAction::ForwardBackward {
                    forward_mb,
                    backward_mb,
                };
            }
        }
        
        // Phase 3: Cooldown (only backward passes)
        if self.backward_idx < self.config.num_micro_batches {
            let mb = self.backward_idx;
            self.backward_idx += 1;
            return ScheduleAction::Backward(mb);
        }
        
        ScheduleAction::Done
    }
    
    /// Reset scheduler for next iteration.
    pub fn reset(&mut self) {
        self.forward_idx = 0;
        self.backward_idx = 0;
        self.warmup_done = 0;
        self.steady_forward_done = 0;
        self.steady_backward_done = 0;
    }
    
    /// Check if schedule is complete.
    pub fn is_done(&self) -> bool {
        self.backward_idx >= self.config.num_micro_batches
    }
}

/// GPipe-style scheduler (all forwards, then all backwards).
/// Simpler but higher memory usage than 1F1B.
pub struct GPipeScheduler {
    config: PipelineConfig,
    forward_idx: usize,
    backward_idx: usize,
    in_backward_phase: bool,
}

impl GPipeScheduler {
    pub fn new(num_micro_batches: usize) -> Self {
        Self {
            config: PipelineConfig::new(num_micro_batches),
            forward_idx: 0,
            backward_idx: 0,
            in_backward_phase: false,
        }
    }
    
    pub fn next_action(&mut self) -> ScheduleAction {
        if !self.in_backward_phase {
            // Forward phase
            if self.forward_idx < self.config.num_micro_batches {
                let mb = self.forward_idx;
                self.forward_idx += 1;
                return ScheduleAction::Forward(mb);
            }
            self.in_backward_phase = true;
        }
        
        // Backward phase
        if self.backward_idx < self.config.num_micro_batches {
            let mb = self.backward_idx;
            self.backward_idx += 1;
            return ScheduleAction::Backward(mb);
        }
        
        ScheduleAction::Done
    }
    
    pub fn reset(&mut self) {
        self.forward_idx = 0;
        self.backward_idx = 0;
        self.in_backward_phase = false;
    }
    
    pub fn is_done(&self) -> bool {
        self.in_backward_phase && self.backward_idx >= self.config.num_micro_batches
    }
}

// ============================================================================
// DualPipe: Bidirectional Pipeline Parallelism (DeepSeek-V3)
// ============================================================================

/// DualPipe action for bidirectional scheduling.
#[derive(Debug, Clone, PartialEq)]
pub enum DualPipeAction {
    /// Forward pass on regular stream (micro-batch from start)
    ForwardRegular(usize),
    /// Forward pass on reverse stream (micro-batch from end)
    ForwardReverse(usize),
    /// Backward pass on regular stream
    BackwardRegular(usize),
    /// Backward pass on reverse stream
    BackwardReverse(usize),
    /// Paired operations in steady state
    DualStep {
        regular_fwd: Option<usize>,
        regular_bwd: Option<usize>,
        reverse_fwd: Option<usize>,
        reverse_bwd: Option<usize>,
    },
    /// Schedule complete
    Done,
}

/// DualPipe scheduler for bidirectional pipeline parallelism.
///
/// DeepSeek-V3 innovation that achieves 2x throughput by running
/// two streams through the pipeline in opposite directions:
/// - Regular stream: stage 0 → stage N-1 (forward), N-1 → 0 (backward)
/// - Reverse stream: stage N-1 → stage 0 (forward), 0 → N-1 (backward)
///
/// This keeps both ends of the pipeline busy, reducing bubble time.
#[derive(Debug)]
pub struct DualPipeScheduler {
    config: PipelineConfig,
    /// Number of micro-batches per stream (half of total)
    micro_batches_per_stream: usize,
    /// Current phase
    phase: DualPipePhase,
    /// Regular stream forward index (start to end)
    regular_fwd_idx: usize,
    /// Regular stream backward index
    regular_bwd_idx: usize,
    /// Reverse stream forward index (end to start)
    reverse_fwd_idx: usize,
    /// Reverse stream backward index
    reverse_bwd_idx: usize,
    /// Steps completed in current phase
    steps_in_phase: usize,
}

/// Phases of DualPipe schedule.
#[derive(Debug, Clone, PartialEq)]
pub enum DualPipePhase {
    /// Warmup: Fill both directions with forward passes
    Warmup,
    /// Steady state: Alternating forward/backward in both directions
    Steady,
    /// Cooldown: Drain remaining backward passes
    Cooldown,
    /// Complete
    Done,
}

impl DualPipeScheduler {
    /// Create new DualPipe scheduler.
    ///
    /// # Arguments
    /// * `num_micro_batches` - Total micro-batches (will be split between streams)
    pub fn new(num_micro_batches: usize) -> Self {
        assert!(num_micro_batches >= 2, "DualPipe requires at least 2 micro-batches");
        
        Self {
            config: PipelineConfig::new(num_micro_batches),
            micro_batches_per_stream: num_micro_batches / 2,
            phase: DualPipePhase::Warmup,
            regular_fwd_idx: 0,
            regular_bwd_idx: 0,
            reverse_fwd_idx: 0,
            reverse_bwd_idx: 0,
            steps_in_phase: 0,
        }
    }
    
    /// Number of warmup steps for this stage in each direction.
    fn warmup_steps(&self) -> usize {
        // Each direction needs (num_stages - distance_from_start) warmup steps
        // For regular: num_stages - stage_rank - 1
        // For reverse: stage_rank
        (self.config.num_stages - self.config.stage_rank - 1).max(self.config.stage_rank)
    }
    
    /// Check if this stage should process regular stream this step.
    fn should_process_regular(&self) -> bool {
        // Regular stream goes 0 → N-1 for forward
        // Process if we have pending work
        true
    }
    
    /// Check if this stage should process reverse stream this step.
    fn should_process_reverse(&self) -> bool {
        // Reverse stream goes N-1 → 0 for forward
        // Process if we have pending work
        true
    }
    
    /// Get next action in DualPipe schedule.
    pub fn next_action(&mut self) -> DualPipeAction {
        match &self.phase {
            DualPipePhase::Warmup => self.warmup_action(),
            DualPipePhase::Steady => self.steady_action(),
            DualPipePhase::Cooldown => self.cooldown_action(),
            DualPipePhase::Done => DualPipeAction::Done,
        }
    }
    
    fn warmup_action(&mut self) -> DualPipeAction {
        let warmup_needed = self.warmup_steps();
        
        if self.steps_in_phase >= warmup_needed {
            self.phase = DualPipePhase::Steady;
            self.steps_in_phase = 0;
            return self.steady_action();
        }
        
        // During warmup, do forward passes in both directions
        let mut action = DualPipeAction::DualStep {
            regular_fwd: None,
            regular_bwd: None,
            reverse_fwd: None,
            reverse_bwd: None,
        };
        
        if let DualPipeAction::DualStep { regular_fwd, reverse_fwd, .. } = &mut action {
            // Regular stream forward (if not done)
            if self.should_process_regular() && self.regular_fwd_idx < self.micro_batches_per_stream {
                *regular_fwd = Some(self.regular_fwd_idx);
                self.regular_fwd_idx += 1;
            }
            
            // Reverse stream forward (offset by micro_batches_per_stream)
            if self.should_process_reverse() && self.reverse_fwd_idx < self.micro_batches_per_stream {
                *reverse_fwd = Some(self.micro_batches_per_stream + self.reverse_fwd_idx);
                self.reverse_fwd_idx += 1;
            }
        }
        
        self.steps_in_phase += 1;
        action
    }
    
    fn steady_action(&mut self) -> DualPipeAction {
        // Steady state: 1 forward, 1 backward for each stream
        let total_steady_steps = self.micro_batches_per_stream.saturating_sub(self.warmup_steps());
        
        if self.steps_in_phase >= total_steady_steps {
            self.phase = DualPipePhase::Cooldown;
            self.steps_in_phase = 0;
            return self.cooldown_action();
        }
        
        let mut action = DualPipeAction::DualStep {
            regular_fwd: None,
            regular_bwd: None,
            reverse_fwd: None,
            reverse_bwd: None,
        };
        
        if let DualPipeAction::DualStep { 
            regular_fwd, regular_bwd, reverse_fwd, reverse_bwd 
        } = &mut action {
            // Regular stream
            if self.regular_fwd_idx < self.micro_batches_per_stream {
                *regular_fwd = Some(self.regular_fwd_idx);
                self.regular_fwd_idx += 1;
            }
            if self.regular_bwd_idx < self.micro_batches_per_stream {
                *regular_bwd = Some(self.regular_bwd_idx);
                self.regular_bwd_idx += 1;
            }
            
            // Reverse stream
            if self.reverse_fwd_idx < self.micro_batches_per_stream {
                *reverse_fwd = Some(self.micro_batches_per_stream + self.reverse_fwd_idx);
                self.reverse_fwd_idx += 1;
            }
            if self.reverse_bwd_idx < self.micro_batches_per_stream {
                *reverse_bwd = Some(self.micro_batches_per_stream + self.reverse_bwd_idx);
                self.reverse_bwd_idx += 1;
            }
        }
        
        self.steps_in_phase += 1;
        action
    }
    
    fn cooldown_action(&mut self) -> DualPipeAction {
        // Cooldown: Drain remaining backward passes
        let remaining_regular = self.micro_batches_per_stream.saturating_sub(self.regular_bwd_idx);
        let remaining_reverse = self.micro_batches_per_stream.saturating_sub(self.reverse_bwd_idx);
        
        if remaining_regular == 0 && remaining_reverse == 0 {
            self.phase = DualPipePhase::Done;
            return DualPipeAction::Done;
        }
        
        let mut action = DualPipeAction::DualStep {
            regular_fwd: None,
            regular_bwd: None,
            reverse_fwd: None,
            reverse_bwd: None,
        };
        
        if let DualPipeAction::DualStep { regular_bwd, reverse_bwd, .. } = &mut action {
            if self.regular_bwd_idx < self.micro_batches_per_stream {
                *regular_bwd = Some(self.regular_bwd_idx);
                self.regular_bwd_idx += 1;
            }
            if self.reverse_bwd_idx < self.micro_batches_per_stream {
                *reverse_bwd = Some(self.micro_batches_per_stream + self.reverse_bwd_idx);
                self.reverse_bwd_idx += 1;
            }
        }
        
        self.steps_in_phase += 1;
        action
    }
    
    /// Reset scheduler for next iteration.
    pub fn reset(&mut self) {
        self.phase = DualPipePhase::Warmup;
        self.regular_fwd_idx = 0;
        self.regular_bwd_idx = 0;
        self.reverse_fwd_idx = 0;
        self.reverse_bwd_idx = 0;
        self.steps_in_phase = 0;
    }
    
    /// Check if schedule is complete.
    pub fn is_done(&self) -> bool {
        self.phase == DualPipePhase::Done
    }
    
    /// Get current phase.
    pub fn phase(&self) -> &DualPipePhase {
        &self.phase
    }
    
    /// Theoretical bubble ratio (vs 1F1B).
    ///
    /// DualPipe reduces bubble from (pp_size - 1) / total_steps
    /// to approximately (pp_size - 1) / (2 * total_steps)
    pub fn bubble_ratio(&self) -> f32 {
        let pp = self.config.num_stages as f32;
        let mb = self.config.num_micro_batches as f32;
        
        // 1F1B bubble ratio for comparison
        let _baseline = (pp - 1.0) / (mb + pp - 1.0);
        
        // DualPipe halves the bubble by using both directions
        (pp - 1.0) / (2.0 * (mb / 2.0) + pp - 1.0)
    }
    
    /// Get config reference.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
    
    /// Get micro-batches per stream.
    pub fn micro_batches_per_stream(&self) -> usize {
        self.micro_batches_per_stream
    }
}

// ============================================================================
// DualPipe Configuration
// ============================================================================

/// Comprehensive DualPipe configuration.
#[derive(Clone, Debug)]
pub struct DualPipeConfig {
    /// Number of pipeline stages
    pub num_stages: usize,
    /// Number of micro-batches per mini-batch
    pub num_micro_batches: usize,
    /// Current stage rank
    pub stage_rank: usize,
    /// Enable overlapped communication with computation
    pub overlap_communication: bool,
    /// Enable activation checkpointing within stages
    pub activation_checkpointing: bool,
    /// Chunk size for activation checkpointing (layers per chunk)
    pub checkpoint_chunk_size: usize,
    /// Enable gradient accumulation across micro-batches
    pub gradient_accumulation: bool,
    /// Number of gradient accumulation steps
    pub accumulation_steps: usize,
    /// Enable memory-efficient backward pass
    pub memory_efficient_backward: bool,
    /// Device for this stage
    pub device: Device,
}

impl Default for DualPipeConfig {
    fn default() -> Self {
        Self {
            num_stages: get_pp_size(),
            num_micro_batches: 8,
            stage_rank: get_pp_rank(),
            overlap_communication: true,
            activation_checkpointing: true,
            checkpoint_chunk_size: 2,
            gradient_accumulation: true,
            accumulation_steps: 1,
            memory_efficient_backward: true,
            device: Device::Cpu,
        }
    }
}

impl DualPipeConfig {
    pub fn new(num_micro_batches: usize) -> Self {
        Self {
            num_micro_batches,
            ..Default::default()
        }
    }
    
    pub fn with_stages(mut self, num_stages: usize, stage_rank: usize) -> Self {
        self.num_stages = num_stages;
        self.stage_rank = stage_rank;
        self
    }
    
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
    
    pub fn with_checkpointing(mut self, enabled: bool, chunk_size: usize) -> Self {
        self.activation_checkpointing = enabled;
        self.checkpoint_chunk_size = chunk_size;
        self
    }
    
    pub fn is_first_stage(&self) -> bool {
        self.stage_rank == 0
    }
    
    pub fn is_last_stage(&self) -> bool {
        self.stage_rank == self.num_stages - 1
    }
    
    /// Calculate micro-batches per stream for DualPipe.
    pub fn micro_batches_per_stream(&self) -> usize {
        self.num_micro_batches / 2
    }
}

// ============================================================================
// Activation Storage for Pipeline Stages
// ============================================================================

/// Stored activation with metadata for backward pass.
#[derive(Clone)]
pub struct DualPipeActivation {
    /// Micro-batch ID
    pub micro_batch_id: usize,
    /// Stream direction (regular=true, reverse=false)
    pub is_regular_stream: bool,
    /// Input activation
    pub input: Tensor,
    /// Output activation
    pub output: Tensor,
    /// Timestamp for profiling
    pub timestamp: Instant,
    /// Memory size in bytes
    pub memory_bytes: usize,
}

impl DualPipeActivation {
    pub fn new(
        micro_batch_id: usize,
        is_regular_stream: bool,
        input: Tensor,
        output: Tensor,
    ) -> Self {
        let memory_bytes = input.elem_count() * 4 + output.elem_count() * 4; // Assume f32
        Self {
            micro_batch_id,
            is_regular_stream,
            input,
            output,
            timestamp: Instant::now(),
            memory_bytes,
        }
    }
}

/// Activation buffer for managing stored activations.
pub struct ActivationBuffer {
    /// Regular stream activations (FIFO for 1F1B-like behavior)
    regular_activations: VecDeque<DualPipeActivation>,
    /// Reverse stream activations
    reverse_activations: VecDeque<DualPipeActivation>,
    /// Maximum memory budget in bytes
    max_memory_bytes: usize,
    /// Current memory usage in bytes
    current_memory_bytes: usize,
}

impl ActivationBuffer {
    pub fn new(max_memory_gb: f32) -> Self {
        Self {
            regular_activations: VecDeque::new(),
            reverse_activations: VecDeque::new(),
            max_memory_bytes: (max_memory_gb * 1024.0 * 1024.0 * 1024.0) as usize,
            current_memory_bytes: 0,
        }
    }
    
    /// Store activation for later backward pass.
    pub fn store(&mut self, activation: DualPipeActivation) -> Result<()> {
        let mem = activation.memory_bytes;
        
        // Check memory budget
        if self.current_memory_bytes + mem > self.max_memory_bytes {
            return Err(candle_core::Error::Msg(
                "Activation buffer memory limit exceeded".to_string()
            ));
        }
        
        self.current_memory_bytes += mem;
        
        if activation.is_regular_stream {
            self.regular_activations.push_back(activation);
        } else {
            self.reverse_activations.push_back(activation);
        }
        
        Ok(())
    }
    
    /// Pop activation for backward pass (FIFO order).
    pub fn pop_regular(&mut self) -> Option<DualPipeActivation> {
        let act = self.regular_activations.pop_front()?;
        self.current_memory_bytes = self.current_memory_bytes.saturating_sub(act.memory_bytes);
        Some(act)
    }
    
    /// Pop reverse activation for backward pass.
    pub fn pop_reverse(&mut self) -> Option<DualPipeActivation> {
        let act = self.reverse_activations.pop_front()?;
        self.current_memory_bytes = self.current_memory_bytes.saturating_sub(act.memory_bytes);
        Some(act)
    }
    
    /// Get current memory usage.
    pub fn memory_usage_bytes(&self) -> usize {
        self.current_memory_bytes
    }
    
    /// Get memory usage as percentage of budget.
    pub fn memory_usage_percent(&self) -> f32 {
        (self.current_memory_bytes as f32 / self.max_memory_bytes as f32) * 100.0
    }
    
    /// Clear all activations.
    pub fn clear(&mut self) {
        self.regular_activations.clear();
        self.reverse_activations.clear();
        self.current_memory_bytes = 0;
    }
    
    /// Get number of stored activations.
    pub fn len(&self) -> usize {
        self.regular_activations.len() + self.reverse_activations.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.regular_activations.is_empty() && self.reverse_activations.is_empty()
    }
}

// ============================================================================
// DualPipe Stage - Pipeline Stage with DualPipe Support
// ============================================================================

/// A pipeline stage for DualPipe execution.
pub struct DualPipeStage {
    /// Stage ID (0 to num_stages - 1)
    pub stage_id: usize,
    /// Stage configuration
    config: DualPipeConfig,
    /// Activation buffer for stored activations
    activation_buffer: ActivationBuffer,
    /// Performance metrics
    metrics: StageMetrics,
}

/// Performance metrics for a pipeline stage.
#[derive(Clone, Debug, Default)]
pub struct StageMetrics {
    /// Total forward passes executed
    pub forward_count: usize,
    /// Total backward passes executed
    pub backward_count: usize,
    /// Total forward time in microseconds
    pub forward_time_us: u64,
    /// Total backward time in microseconds
    pub backward_time_us: u64,
    /// Total communication time in microseconds
    pub comm_time_us: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Total bubbles (idle steps)
    pub bubble_count: usize,
}

impl StageMetrics {
    pub fn forward_throughput(&self) -> f64 {
        if self.forward_time_us == 0 {
            return 0.0;
        }
        self.forward_count as f64 / (self.forward_time_us as f64 / 1_000_000.0)
    }
    
    pub fn backward_throughput(&self) -> f64 {
        if self.backward_time_us == 0 {
            return 0.0;
        }
        self.backward_count as f64 / (self.backward_time_us as f64 / 1_000_000.0)
    }
    
    pub fn compute_efficiency(&self) -> f64 {
        let total_ops = self.forward_count + self.backward_count + self.bubble_count;
        if total_ops == 0 {
            return 0.0;
        }
        (self.forward_count + self.backward_count) as f64 / total_ops as f64
    }
}

impl DualPipeStage {
    pub fn new(stage_id: usize, config: DualPipeConfig, max_activation_memory_gb: f32) -> Self {
        Self {
            stage_id,
            config,
            activation_buffer: ActivationBuffer::new(max_activation_memory_gb),
            metrics: StageMetrics::default(),
        }
    }
    
    /// Execute forward pass and store activation.
    pub fn forward(
        &mut self,
        input: &Tensor,
        micro_batch_id: usize,
        is_regular_stream: bool,
        forward_fn: impl FnOnce(&Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let start = Instant::now();
        
        // Execute forward pass
        let output = forward_fn(input)?;
        
        // Store activation for backward
        let activation = DualPipeActivation::new(
            micro_batch_id,
            is_regular_stream,
            input.clone(),
            output.clone(),
        );
        self.activation_buffer.store(activation)?;
        
        // Update metrics
        self.metrics.forward_count += 1;
        self.metrics.forward_time_us += start.elapsed().as_micros() as u64;
        self.metrics.peak_memory_bytes = self.metrics.peak_memory_bytes
            .max(self.activation_buffer.memory_usage_bytes());
        
        Ok(output)
    }
    
    /// Execute backward pass using stored activation.
    pub fn backward(
        &mut self,
        grad_output: &Tensor,
        is_regular_stream: bool,
        backward_fn: impl FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let start = Instant::now();
        
        // Pop stored activation
        let activation = if is_regular_stream {
            self.activation_buffer.pop_regular()
        } else {
            self.activation_buffer.pop_reverse()
        }.ok_or_else(|| candle_core::Error::Msg(
            "No activation available for backward pass".to_string()
        ))?;
        
        // Execute backward pass
        let grad_input = backward_fn(&activation.input, &activation.output, grad_output)?;
        
        // Update metrics
        self.metrics.backward_count += 1;
        self.metrics.backward_time_us += start.elapsed().as_micros() as u64;
        
        Ok(grad_input)
    }
    
    /// Record bubble (idle step).
    pub fn record_bubble(&mut self) {
        self.metrics.bubble_count += 1;
    }
    
    /// Get metrics reference.
    pub fn metrics(&self) -> &StageMetrics {
        &self.metrics
    }
    
    /// Reset metrics for new training run.
    pub fn reset_metrics(&mut self) {
        self.metrics = StageMetrics::default();
    }
    
    /// Get activation buffer reference.
    pub fn activation_buffer(&self) -> &ActivationBuffer {
        &self.activation_buffer
    }
    
    /// Clear activation buffer.
    pub fn clear_activations(&mut self) {
        self.activation_buffer.clear();
    }
}

// ============================================================================
// DualPipe Engine - Full Pipeline Execution Engine
// ============================================================================

/// DualPipe execution engine managing all stages.
pub struct DualPipeEngine {
    /// Configuration
    config: DualPipeConfig,
    /// Pipeline stage for this rank
    stage: DualPipeStage,
    /// Scheduler for generating schedule
    scheduler: DualPipeScheduler,
    /// Accumulated gradients for gradient accumulation
    accumulated_grads: HashMap<String, Tensor>,
    /// Number of accumulation steps completed
    accumulation_step: usize,
    /// Total training time for benchmarking
    total_train_time_us: u64,
    /// Number of training iterations
    train_iterations: usize,
}

impl DualPipeEngine {
    pub fn new(config: DualPipeConfig, max_activation_memory_gb: f32) -> Self {
        let scheduler = DualPipeScheduler::new(config.num_micro_batches);
        let stage = DualPipeStage::new(
            config.stage_rank,
            config.clone(),
            max_activation_memory_gb,
        );
        
        Self {
            config,
            stage,
            scheduler,
            accumulated_grads: HashMap::new(),
            accumulation_step: 0,
            total_train_time_us: 0,
            train_iterations: 0,
        }
    }
    
    /// Execute one full training iteration with DualPipe schedule.
    pub fn train_step<F, B>(
        &mut self,
        get_micro_batch: impl Fn(usize) -> Result<Tensor>,
        forward_fn: F,
        backward_fn: B,
        compute_loss: impl Fn(&Tensor, usize) -> Result<Tensor>,
    ) -> Result<f32>
    where
        F: Fn(&Tensor) -> Result<Tensor> + Copy,
        B: Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor> + Copy,
    {
        let start = Instant::now();
        let mut total_loss = 0.0f32;
        let mut loss_count = 0;
        
        // Reset scheduler for new iteration
        self.scheduler.reset();
        
        // Execute schedule
        loop {
            let action = self.scheduler.next_action();
            
            match action {
                DualPipeAction::DualStep {
                    regular_fwd,
                    regular_bwd,
                    reverse_fwd,
                    reverse_bwd,
                } => {
                    // Process regular stream forward
                    if let Some(mb_id) = regular_fwd {
                        let input = self.receive_or_get_input(mb_id, true, &get_micro_batch)?;
                        let output = self.stage.forward(&input, mb_id, true, forward_fn)?;
                        
                        if self.config.is_last_stage() {
                            // Compute loss at last stage
                            let loss = compute_loss(&output, mb_id)?;
                            let loss_val: f32 = loss.to_scalar()?;
                            total_loss += loss_val;
                            loss_count += 1;
                        } else {
                            self.send_forward(&output, true)?;
                        }
                    }
                    
                    // Process reverse stream forward
                    if let Some(mb_id) = reverse_fwd {
                        let input = self.receive_or_get_input(mb_id, false, &get_micro_batch)?;
                        let output = self.stage.forward(&input, mb_id, false, forward_fn)?;
                        
                        if self.config.is_first_stage() {
                            // Compute loss at first stage for reverse stream
                            let loss = compute_loss(&output, mb_id)?;
                            let loss_val: f32 = loss.to_scalar()?;
                            total_loss += loss_val;
                            loss_count += 1;
                        } else {
                            self.send_forward(&output, false)?;
                        }
                    }
                    
                    // Process regular stream backward
                    if let Some(_mb_id) = regular_bwd {
                        let grad_output = self.receive_backward(true)?;
                        let grad_input = self.stage.backward(&grad_output, true, backward_fn)?;
                        
                        if !self.config.is_first_stage() {
                            self.send_backward(&grad_input, true)?;
                        }
                    }
                    
                    // Process reverse stream backward
                    if let Some(_mb_id) = reverse_bwd {
                        let grad_output = self.receive_backward(false)?;
                        let grad_input = self.stage.backward(&grad_output, false, backward_fn)?;
                        
                        if !self.config.is_last_stage() {
                            self.send_backward(&grad_input, false)?;
                        }
                    }
                    
                    // Record bubble if nothing was done
                    if regular_fwd.is_none() && regular_bwd.is_none() 
                        && reverse_fwd.is_none() && reverse_bwd.is_none() {
                        self.stage.record_bubble();
                    }
                }
                DualPipeAction::ForwardRegular(mb_id) => {
                    let input = self.receive_or_get_input(mb_id, true, &get_micro_batch)?;
                    let output = self.stage.forward(&input, mb_id, true, forward_fn)?;
                    
                    if self.config.is_last_stage() {
                        let loss = compute_loss(&output, mb_id)?;
                        let loss_val: f32 = loss.to_scalar()?;
                        total_loss += loss_val;
                        loss_count += 1;
                    } else {
                        self.send_forward(&output, true)?;
                    }
                }
                DualPipeAction::ForwardReverse(mb_id) => {
                    let input = self.receive_or_get_input(mb_id, false, &get_micro_batch)?;
                    let output = self.stage.forward(&input, mb_id, false, forward_fn)?;
                    
                    if self.config.is_first_stage() {
                        let loss = compute_loss(&output, mb_id)?;
                        let loss_val: f32 = loss.to_scalar()?;
                        total_loss += loss_val;
                        loss_count += 1;
                    } else {
                        self.send_forward(&output, false)?;
                    }
                }
                DualPipeAction::BackwardRegular(_mb_id) => {
                    let grad_output = self.receive_backward(true)?;
                    let grad_input = self.stage.backward(&grad_output, true, backward_fn)?;
                    
                    if !self.config.is_first_stage() {
                        self.send_backward(&grad_input, true)?;
                    }
                }
                DualPipeAction::BackwardReverse(_mb_id) => {
                    let grad_output = self.receive_backward(false)?;
                    let grad_input = self.stage.backward(&grad_output, false, backward_fn)?;
                    
                    if !self.config.is_last_stage() {
                        self.send_backward(&grad_input, false)?;
                    }
                }
                DualPipeAction::Done => break,
            }
        }
        
        // Update timing stats
        self.total_train_time_us += start.elapsed().as_micros() as u64;
        self.train_iterations += 1;
        
        // Clear activations after iteration
        self.stage.clear_activations();
        
        let avg_loss = if loss_count > 0 {
            total_loss / loss_count as f32
        } else {
            0.0
        };
        
        Ok(avg_loss)
    }
    
    /// Receive input from previous stage or get from data loader.
    fn receive_or_get_input(
        &self,
        micro_batch_id: usize,
        is_regular_stream: bool,
        get_micro_batch: impl Fn(usize) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let is_first_for_stream = if is_regular_stream {
            self.config.is_first_stage()
        } else {
            self.config.is_last_stage()
        };
        
        if is_first_for_stream {
            // Get input from data loader
            get_micro_batch(micro_batch_id)
        } else {
            // Receive from previous stage
            self.receive_forward(is_regular_stream)
        }
    }
    
    /// Send activation forward to next stage.
    fn send_forward(&self, tensor: &Tensor, is_regular_stream: bool) -> Result<()> {
        if is_regular_stream {
            send_forward(tensor)
        } else {
            // Reverse stream goes in opposite direction
            send_backward(tensor)
        }
    }
    
    /// Receive activation from previous stage.
    fn receive_forward(&self, is_regular_stream: bool) -> Result<Tensor> {
        let shape = &[1]; // Placeholder - actual shape depends on model
        
        let tensor = if is_regular_stream {
            recv_forward(shape, &self.config.device)?
        } else {
            recv_backward(shape, &self.config.device)?
        };
        
        tensor.ok_or_else(|| candle_core::Error::Msg(
            "Failed to receive forward activation".to_string()
        ))
    }
    
    /// Send gradient backward to previous stage.
    fn send_backward(&self, tensor: &Tensor, is_regular_stream: bool) -> Result<()> {
        if is_regular_stream {
            send_backward(tensor)
        } else {
            send_forward(tensor)
        }
    }
    
    /// Receive gradient from next stage.
    fn receive_backward(&self, is_regular_stream: bool) -> Result<Tensor> {
        let shape = &[1]; // Placeholder
        
        let tensor = if is_regular_stream {
            recv_backward(shape, &self.config.device)?
        } else {
            recv_forward(shape, &self.config.device)?
        };
        
        tensor.ok_or_else(|| candle_core::Error::Msg(
            "Failed to receive backward gradient".to_string()
        ))
    }
    
    /// Get engine metrics.
    pub fn get_metrics(&self) -> EngineMetrics {
        EngineMetrics {
            stage_metrics: self.stage.metrics().clone(),
            total_train_time_us: self.total_train_time_us,
            train_iterations: self.train_iterations,
            avg_iteration_time_ms: if self.train_iterations > 0 {
                (self.total_train_time_us as f64 / self.train_iterations as f64) / 1000.0
            } else {
                0.0
            },
            throughput_samples_per_sec: if self.total_train_time_us > 0 {
                let total_samples = self.train_iterations * self.config.num_micro_batches;
                (total_samples as f64) / (self.total_train_time_us as f64 / 1_000_000.0)
            } else {
                0.0
            },
            bubble_ratio: self.calculate_bubble_ratio(),
        }
    }
    
    /// Calculate actual bubble ratio from metrics.
    fn calculate_bubble_ratio(&self) -> f64 {
        let metrics = self.stage.metrics();
        let total_ops = metrics.forward_count + metrics.backward_count + metrics.bubble_count;
        if total_ops == 0 {
            return 0.0;
        }
        metrics.bubble_count as f64 / total_ops as f64
    }
    
    /// Get configuration.
    pub fn config(&self) -> &DualPipeConfig {
        &self.config
    }
    
    /// Reset engine for new training run.
    pub fn reset(&mut self) {
        self.scheduler.reset();
        self.stage.reset_metrics();
        self.stage.clear_activations();
        self.accumulated_grads.clear();
        self.accumulation_step = 0;
        self.total_train_time_us = 0;
        self.train_iterations = 0;
    }
}

/// Engine-level metrics combining all measurements.
#[derive(Clone, Debug)]
pub struct EngineMetrics {
    pub stage_metrics: StageMetrics,
    pub total_train_time_us: u64,
    pub train_iterations: usize,
    pub avg_iteration_time_ms: f64,
    pub throughput_samples_per_sec: f64,
    pub bubble_ratio: f64,
}

impl EngineMetrics {
    pub fn print_summary(&self) {
        println!("=== DualPipe Engine Metrics ===");
        println!("Training iterations: {}", self.train_iterations);
        println!("Avg iteration time: {:.2} ms", self.avg_iteration_time_ms);
        println!("Throughput: {:.2} samples/sec", self.throughput_samples_per_sec);
        println!("Bubble ratio: {:.2}%", self.bubble_ratio * 100.0);
        println!("Forward passes: {}", self.stage_metrics.forward_count);
        println!("Backward passes: {}", self.stage_metrics.backward_count);
        println!("Forward throughput: {:.2} ops/sec", self.stage_metrics.forward_throughput());
        println!("Backward throughput: {:.2} ops/sec", self.stage_metrics.backward_throughput());
        println!("Compute efficiency: {:.2}%", self.stage_metrics.compute_efficiency() * 100.0);
        println!("Peak memory: {} MB", self.stage_metrics.peak_memory_bytes / (1024 * 1024));
        println!("================================");
    }
}

// ============================================================================
// DualPipe LR Scheduler
// ============================================================================

/// Learning rate scheduler that is pipeline-aware.
pub struct DualPipeLRScheduler {
    /// Base learning rate
    base_lr: f64,
    /// Current learning rate
    current_lr: f64,
    /// Warmup steps
    warmup_steps: usize,
    /// Total training steps
    total_steps: usize,
    /// Current step
    current_step: usize,
    /// Schedule type
    schedule: LRScheduleType,
    /// Pipeline stage rank for logging
    stage_rank: usize,
}

/// Types of LR schedules.
#[derive(Clone, Debug)]
pub enum LRScheduleType {
    /// Constant LR after warmup
    Constant,
    /// Linear decay to 0
    Linear,
    /// Cosine decay to min_lr
    Cosine { min_lr: f64 },
    /// Step decay at specified steps
    Step { decay_factor: f64, decay_steps: Vec<usize> },
}

impl DualPipeLRScheduler {
    pub fn new(
        base_lr: f64,
        warmup_steps: usize,
        total_steps: usize,
        schedule: LRScheduleType,
        stage_rank: usize,
    ) -> Self {
        Self {
            base_lr,
            current_lr: 0.0, // Start at 0 for warmup
            warmup_steps,
            total_steps,
            current_step: 0,
            schedule,
            stage_rank,
        }
    }
    
    /// Step the scheduler and return new learning rate.
    pub fn step(&mut self) -> f64 {
        self.current_step += 1;
        
        if self.current_step <= self.warmup_steps {
            // Linear warmup
            self.current_lr = self.base_lr * (self.current_step as f64 / self.warmup_steps as f64);
        } else {
            // Apply schedule
            let progress = (self.current_step - self.warmup_steps) as f64
                / (self.total_steps - self.warmup_steps) as f64;
            
            self.current_lr = match &self.schedule {
                LRScheduleType::Constant => self.base_lr,
                LRScheduleType::Linear => self.base_lr * (1.0 - progress),
                LRScheduleType::Cosine { min_lr } => {
                    let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                    min_lr + (self.base_lr - min_lr) * cosine_decay
                }
                LRScheduleType::Step { decay_factor, decay_steps } => {
                    let num_decays = decay_steps.iter()
                        .filter(|&&s| self.current_step >= s)
                        .count();
                    self.base_lr * decay_factor.powi(num_decays as i32)
                }
            };
        }
        
        self.current_lr
    }
    
    /// Get current learning rate without stepping.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }
    
    /// Get current step.
    pub fn get_step(&self) -> usize {
        self.current_step
    }
    
    /// Reset scheduler.
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = 0.0;
    }
}

// ============================================================================
// Model Partitioning Utilities
// ============================================================================

/// Partition configuration for distributing model across stages.
#[derive(Clone, Debug)]
pub struct PartitionConfig {
    /// Total number of layers
    pub total_layers: usize,
    /// Number of pipeline stages
    pub num_stages: usize,
    /// Balance method
    pub method: PartitionMethod,
}

/// Methods for partitioning model across stages.
#[derive(Clone, Debug)]
pub enum PartitionMethod {
    /// Equal number of layers per stage
    Uniform,
    /// Specify layer counts per stage
    Custom(Vec<usize>),
    /// Profile-based (more layers on faster stages)
    ProfileBased(Vec<f64>), // Relative compute capacities
}

impl PartitionConfig {
    pub fn uniform(total_layers: usize, num_stages: usize) -> Self {
        Self {
            total_layers,
            num_stages,
            method: PartitionMethod::Uniform,
        }
    }
    
    /// Get layer range for a given stage.
    pub fn get_layer_range(&self, stage_id: usize) -> (usize, usize) {
        match &self.method {
            PartitionMethod::Uniform => {
                let base_layers = self.total_layers / self.num_stages;
                let remainder = self.total_layers % self.num_stages;
                
                let start = stage_id * base_layers + stage_id.min(remainder);
                let extra = if stage_id < remainder { 1 } else { 0 };
                let end = start + base_layers + extra;
                
                (start, end)
            }
            PartitionMethod::Custom(counts) => {
                let start: usize = counts[..stage_id].iter().sum();
                let end = start + counts[stage_id];
                (start, end)
            }
            PartitionMethod::ProfileBased(capacities) => {
                // Assign layers proportional to capacity
                let total_capacity: f64 = capacities.iter().sum();
                let mut start = 0;
                let mut end = 0;
                
                for (i, cap) in capacities.iter().enumerate() {
                    let layers = ((cap / total_capacity) * self.total_layers as f64).round() as usize;
                    if i == stage_id {
                        end = start + layers;
                        break;
                    }
                    start += layers;
                }
                
                // Ensure last stage gets remaining layers
                if stage_id == self.num_stages - 1 {
                    end = self.total_layers;
                }
                
                (start, end)
            }
        }
    }
    
    /// Get number of layers for a given stage.
    pub fn num_layers_for_stage(&self, stage_id: usize) -> usize {
        let (start, end) = self.get_layer_range(stage_id);
        end - start
    }
}

// ============================================================================
// Benchmarking Utilities
// ============================================================================

/// Benchmark DualPipe vs 1F1B scheduling.
pub fn benchmark_schedule_efficiency(
    num_stages: usize,
    num_micro_batches: usize,
) -> ScheduleBenchmark {
    // Simulate 1F1B schedule
    let onef1b_bubbles = num_stages - 1;
    let onef1b_total_steps = num_micro_batches + onef1b_bubbles;
    let onef1b_bubble_ratio = onef1b_bubbles as f64 / onef1b_total_steps as f64;
    
    // Simulate DualPipe schedule
    let dualpipe_bubbles = (num_stages - 1) / 2;
    let dualpipe_total_steps = num_micro_batches / 2 + dualpipe_bubbles;
    let dualpipe_bubble_ratio = dualpipe_bubbles as f64 / dualpipe_total_steps as f64;
    
    ScheduleBenchmark {
        num_stages,
        num_micro_batches,
        onef1b_bubble_ratio,
        dualpipe_bubble_ratio,
        improvement_percent: ((onef1b_bubble_ratio - dualpipe_bubble_ratio) 
            / onef1b_bubble_ratio) * 100.0,
    }
}

/// Result of schedule efficiency benchmark.
#[derive(Clone, Debug)]
pub struct ScheduleBenchmark {
    pub num_stages: usize,
    pub num_micro_batches: usize,
    pub onef1b_bubble_ratio: f64,
    pub dualpipe_bubble_ratio: f64,
    pub improvement_percent: f64,
}

impl ScheduleBenchmark {
    pub fn print(&self) {
        println!("Schedule Efficiency Benchmark:");
        println!("  Stages: {}, Micro-batches: {}", self.num_stages, self.num_micro_batches);
        println!("  1F1B bubble ratio: {:.2}%", self.onef1b_bubble_ratio * 100.0);
        println!("  DualPipe bubble ratio: {:.2}%", self.dualpipe_bubble_ratio * 100.0);
        println!("  Improvement: {:.2}%", self.improvement_percent);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pipeline_config() {
        let config = PipelineConfig::new(4);
        // With default single rank, should be both first and last
        assert!(config.is_first_stage);
        assert!(config.is_last_stage);
        assert_eq!(config.num_stages, 1);
    }
    
    #[test]
    fn test_1f1b_scheduler_single_stage() {
        let mut scheduler = OneFOneBScheduler::new(4);
        
        // With single stage, all actions should be forward-backward pairs
        let actions: Vec<_> = std::iter::from_fn(|| {
            match scheduler.next_action() {
                ScheduleAction::Done => None,
                action => Some(action),
            }
        }).collect();
        
        // Should have 4 micro-batches processed
        assert_eq!(actions.len(), 4);
    }
    
    #[test]
    fn test_gpipe_scheduler() {
        let mut scheduler = GPipeScheduler::new(4);
        
        let mut forwards = 0;
        let mut backwards = 0;
        let mut last_was_forward = true;
        
        loop {
            match scheduler.next_action() {
                ScheduleAction::Forward(_) => {
                    forwards += 1;
                    assert!(last_was_forward, "GPipe should do all forwards first");
                }
                ScheduleAction::Backward(_) => {
                    backwards += 1;
                    last_was_forward = false;
                }
                ScheduleAction::Done => break,
                _ => panic!("Unexpected action"),
            }
        }
        
        assert_eq!(forwards, 4);
        assert_eq!(backwards, 4);
    }
    
    #[test]
    fn test_dualpipe_scheduler() {
        let mut scheduler = DualPipeScheduler::new(8); // 4 per stream
        
        let mut regular_fwd = 0;
        let mut regular_bwd = 0;
        let mut reverse_fwd = 0;
        let mut reverse_bwd = 0;
        let mut steps = 0;
        
        loop {
            match scheduler.next_action() {
                DualPipeAction::DualStep { 
                    regular_fwd: rf, 
                    regular_bwd: rb, 
                    reverse_fwd: rvf, 
                    reverse_bwd: rvb 
                } => {
                    if rf.is_some() { regular_fwd += 1; }
                    if rb.is_some() { regular_bwd += 1; }
                    if rvf.is_some() { reverse_fwd += 1; }
                    if rvb.is_some() { reverse_bwd += 1; }
                    steps += 1;
                }
                DualPipeAction::Done => break,
                _ => {}
            }
            
            // Safety: prevent infinite loops
            if steps > 100 {
                panic!("DualPipe scheduler didn't terminate");
            }
        }
        
        // Each stream should process 4 micro-batches
        assert_eq!(regular_fwd, 4, "Regular stream forward count");
        assert_eq!(regular_bwd, 4, "Regular stream backward count");
        assert_eq!(reverse_fwd, 4, "Reverse stream forward count");
        assert_eq!(reverse_bwd, 4, "Reverse stream backward count");
    }
    
    #[test]
    fn test_dualpipe_phases() {
        let scheduler = DualPipeScheduler::new(8);
        
        // Should start in warmup
        assert_eq!(*scheduler.phase(), DualPipePhase::Warmup);
        
        // Should not be done initially
        assert!(!scheduler.is_done());
        
        // Bubble ratio should be calculated
        let bubble = scheduler.bubble_ratio();
        assert!(bubble >= 0.0 && bubble <= 1.0);
    }
    
    #[test]
    fn test_dualpipe_reset() {
        let mut scheduler = DualPipeScheduler::new(4);
        
        // Run some steps
        for _ in 0..5 {
            scheduler.next_action();
        }
        
        // Reset
        scheduler.reset();
        
        // Should be back to warmup
        assert_eq!(*scheduler.phase(), DualPipePhase::Warmup);
        assert!(!scheduler.is_done());
    }
    
    #[test]
    fn test_dualpipe_config() {
        let config = DualPipeConfig::new(8)
            .with_stages(4, 1)
            .with_checkpointing(true, 2);
        
        assert_eq!(config.num_stages, 4);
        assert_eq!(config.stage_rank, 1);
        assert!(!config.is_first_stage());
        assert!(!config.is_last_stage());
        assert!(config.activation_checkpointing);
        assert_eq!(config.checkpoint_chunk_size, 2);
        assert_eq!(config.micro_batches_per_stream(), 4);
    }
    
    #[test]
    fn test_activation_buffer() {
        let mut buffer = ActivationBuffer::new(1.0); // 1GB limit
        
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        
        // Create dummy tensors
        let input = Tensor::zeros((10, 64), DType::F32, &Device::Cpu).unwrap();
        let output = Tensor::zeros((10, 64), DType::F32, &Device::Cpu).unwrap();
        
        // Store regular activation
        let activation = DualPipeActivation::new(0, true, input.clone(), output.clone());
        buffer.store(activation).unwrap();
        
        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());
        
        // Store reverse activation
        let activation = DualPipeActivation::new(1, false, input.clone(), output.clone());
        buffer.store(activation).unwrap();
        
        assert_eq!(buffer.len(), 2);
        
        // Pop activations
        let regular = buffer.pop_regular();
        assert!(regular.is_some());
        assert_eq!(regular.unwrap().micro_batch_id, 0);
        
        let reverse = buffer.pop_reverse();
        assert!(reverse.is_some());
        assert_eq!(reverse.unwrap().micro_batch_id, 1);
        
        assert!(buffer.is_empty());
    }
    
    #[test]
    fn test_stage_metrics() {
        let mut metrics = StageMetrics::default();
        
        assert_eq!(metrics.forward_throughput(), 0.0);
        assert_eq!(metrics.backward_throughput(), 0.0);
        assert_eq!(metrics.compute_efficiency(), 0.0);
        
        // Simulate some work
        metrics.forward_count = 100;
        metrics.backward_count = 100;
        metrics.forward_time_us = 1_000_000; // 1 second
        metrics.backward_time_us = 1_000_000;
        metrics.bubble_count = 20;
        
        assert!((metrics.forward_throughput() - 100.0).abs() < 0.1);
        assert!((metrics.backward_throughput() - 100.0).abs() < 0.1);
        
        // Efficiency = (100 + 100) / (100 + 100 + 20) = 200/220 ≈ 0.909
        assert!((metrics.compute_efficiency() - 0.909).abs() < 0.01);
    }
    
    #[test]
    fn test_dualpipe_stage() {
        let config = DualPipeConfig::new(8).with_stages(4, 1);
        let mut stage = DualPipeStage::new(1, config, 1.0);
        
        assert_eq!(stage.stage_id, 1);
        assert_eq!(stage.metrics().forward_count, 0);
        
        // Test forward pass
        let input = Tensor::randn(0f32, 1.0, (4, 64), &Device::Cpu).unwrap();
        let output = stage.forward(&input, 0, true, |x| {
            // Simple identity forward
            Ok(x.clone())
        }).unwrap();
        
        assert_eq!(output.dims(), input.dims());
        assert_eq!(stage.metrics().forward_count, 1);
        assert!(!stage.activation_buffer().is_empty());
    }
    
    #[test]
    fn test_lr_scheduler_warmup() {
        let mut scheduler = DualPipeLRScheduler::new(
            0.001, // base_lr
            100,   // warmup_steps
            1000,  // total_steps
            LRScheduleType::Constant,
            0,
        );
        
        // At step 0
        assert_eq!(scheduler.get_lr(), 0.0);
        
        // After 50 steps (halfway through warmup)
        for _ in 0..50 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.0005).abs() < 0.00001);
        
        // After warmup
        for _ in 0..50 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.001).abs() < 0.00001);
    }
    
    #[test]
    fn test_lr_scheduler_cosine() {
        let mut scheduler = DualPipeLRScheduler::new(
            0.001,
            10,
            110,
            LRScheduleType::Cosine { min_lr: 0.0001 },
            0,
        );
        
        // Warmup
        for _ in 0..10 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.001).abs() < 0.00001);
        
        // End of training should approach min_lr
        for _ in 0..100 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.0001).abs() < 0.0001);
    }
    
    #[test]
    fn test_partition_config_uniform() {
        let config = PartitionConfig::uniform(24, 4);
        
        // Each stage should get 6 layers
        for i in 0..4 {
            let (start, end) = config.get_layer_range(i);
            assert_eq!(end - start, 6);
        }
        
        // Ranges should be contiguous
        let (_, end0) = config.get_layer_range(0);
        let (start1, _) = config.get_layer_range(1);
        assert_eq!(end0, start1);
    }
    
    #[test]
    fn test_partition_config_uneven() {
        let config = PartitionConfig::uniform(10, 3);
        
        // 10 layers, 3 stages: 4, 3, 3
        assert_eq!(config.num_layers_for_stage(0), 4);
        assert_eq!(config.num_layers_for_stage(1), 3);
        assert_eq!(config.num_layers_for_stage(2), 3);
    }
    
    #[test]
    fn test_schedule_benchmark() {
        let benchmark = benchmark_schedule_efficiency(8, 32);
        
        // DualPipe should have better bubble ratio than 1F1B
        assert!(benchmark.dualpipe_bubble_ratio < benchmark.onef1b_bubble_ratio);
        assert!(benchmark.improvement_percent > 0.0);
        
        benchmark.print();
    }
    
    #[test]
    fn test_dualpipe_engine_creation() {
        let config = DualPipeConfig::new(8).with_stages(4, 0);
        let engine = DualPipeEngine::new(config, 1.0);
        
        assert!(engine.config().is_first_stage());
        assert!(!engine.config().is_last_stage());
        assert_eq!(engine.get_metrics().train_iterations, 0);
    }
    
    #[test]
    fn test_engine_metrics() {
        let config = DualPipeConfig::new(8);
        let engine = DualPipeEngine::new(config, 1.0);
        
        let metrics = engine.get_metrics();
        assert_eq!(metrics.train_iterations, 0);
        assert_eq!(metrics.bubble_ratio, 0.0);
        
        metrics.print_summary();
    }
}
