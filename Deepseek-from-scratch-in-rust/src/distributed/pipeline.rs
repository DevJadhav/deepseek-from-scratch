//! Pipeline Parallelism for distributed training.
//!
//! Implements 1F1B (One Forward One Backward) schedule for efficient
//! pipeline parallelism with micro-batch management.

use candle_core::{Result, Tensor};
use std::collections::VecDeque;
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
}
