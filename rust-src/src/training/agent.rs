//! Agent/Tool-Use Training Module for DeepSeek-V3.2
//!
//! This module provides infrastructure for training language models
//! with agentic capabilities including:
//! - Tool-call parsing and execution
//! - Multi-turn trajectory management
//! - Environment feedback handling
//! - Reward computation for agent tasks
//!
//! Supports 5 tool categories:
//! - Code Execution (Python REPL, Shell, Jupyter)
//! - Web Search (Search API, URL fetch)
//! - File I/O (Read/write, directory ops)
//! - API Calls (REST APIs, databases)
//! - Reasoning Tools (Calculator, symbolic math)

use candle_core::{Result, Tensor, Device};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Tool Call Format Specification
// ============================================================================

/// Tool call JSON schema format for function calling
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for the tool call
    pub id: String,
    /// Tool type/category
    pub tool_type: ToolType,
    /// Function name to call
    pub function_name: String,
    /// Arguments as JSON object
    pub arguments: serde_json::Value,
}

/// Tool response from environment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolResponse {
    /// Tool call ID this responds to
    pub call_id: String,
    /// Success or failure status
    pub status: ToolStatus,
    /// Response content
    pub content: String,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
}

/// Tool execution status
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolStatus {
    Success,
    Error,
    Timeout,
    SecurityViolation,
    InvalidArguments,
}

/// Tool categories as per V3.2 agent training spec
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolType {
    /// Code execution: Python REPL, Shell, Jupyter
    CodeExecution,
    /// Web search: Search API, URL fetch
    WebSearch,
    /// File I/O: Read/write files, directory ops
    FileIO,
    /// API calls: REST APIs, database queries
    ApiCall,
    /// Reasoning: Calculator, symbolic math
    Reasoning,
}

impl ToolType {
    /// Get complexity level (1-3)
    pub fn complexity(&self) -> u8 {
        match self {
            ToolType::Reasoning => 1,
            ToolType::FileIO | ToolType::WebSearch => 2,
            ToolType::CodeExecution | ToolType::ApiCall => 3,
        }
    }
    
    /// Get all tool types
    pub fn all() -> Vec<ToolType> {
        vec![
            ToolType::CodeExecution,
            ToolType::WebSearch,
            ToolType::FileIO,
            ToolType::ApiCall,
            ToolType::Reasoning,
        ]
    }
}

// ============================================================================
// Agent Turn and Trajectory
// ============================================================================

/// A single turn in an agent trajectory
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentTurn {
    /// Turn index in trajectory
    pub turn_index: usize,
    /// User message or task description
    pub user_message: Option<String>,
    /// Assistant response
    pub assistant_response: String,
    /// Tool calls made in this turn
    pub tool_calls: Vec<ToolCall>,
    /// Tool responses received
    pub tool_responses: Vec<ToolResponse>,
    /// Whether this turn completed the task
    pub is_terminal: bool,
}

impl AgentTurn {
    pub fn new(turn_index: usize) -> Self {
        Self {
            turn_index,
            user_message: None,
            assistant_response: String::new(),
            tool_calls: Vec::new(),
            tool_responses: Vec::new(),
            is_terminal: false,
        }
    }
    
    /// Count of tool calls in this turn
    pub fn tool_call_count(&self) -> usize {
        self.tool_calls.len()
    }
    
    /// Check if all tool calls succeeded
    pub fn all_tools_succeeded(&self) -> bool {
        self.tool_responses.iter().all(|r| r.status == ToolStatus::Success)
    }
    
    /// Get successful tool call count
    pub fn successful_tool_count(&self) -> usize {
        self.tool_responses.iter()
            .filter(|r| r.status == ToolStatus::Success)
            .count()
    }
}

/// Complete agent trajectory for an episode
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentTrajectory {
    /// Task description / initial prompt
    pub task: String,
    /// Task type/tier
    pub task_tier: TaskTier,
    /// Sequence of turns
    pub turns: Vec<AgentTurn>,
    /// Final reward
    pub reward: f32,
    /// Reward breakdown
    pub reward_breakdown: RewardBreakdown,
    /// Total execution time in ms
    pub total_time_ms: u64,
    /// Whether task was completed successfully
    pub task_completed: bool,
}

impl AgentTrajectory {
    pub fn new(task: String, task_tier: TaskTier) -> Self {
        Self {
            task,
            task_tier,
            turns: Vec::new(),
            reward: 0.0,
            reward_breakdown: RewardBreakdown::default(),
            total_time_ms: 0,
            task_completed: false,
        }
    }
    
    /// Add a turn to the trajectory
    pub fn add_turn(&mut self, turn: AgentTurn) {
        self.turns.push(turn);
    }
    
    /// Total number of turns
    pub fn num_turns(&self) -> usize {
        self.turns.len()
    }
    
    /// Total number of tool calls across all turns
    pub fn total_tool_calls(&self) -> usize {
        self.turns.iter().map(|t| t.tool_call_count()).sum()
    }
    
    /// Get tool types used
    pub fn tool_types_used(&self) -> Vec<ToolType> {
        let mut types: Vec<ToolType> = self.turns.iter()
            .flat_map(|t| t.tool_calls.iter().map(|tc| tc.tool_type))
            .collect();
        types.sort_by_key(|t| *t as u8);
        types.dedup();
        types
    }
}

// ============================================================================
// Task Tiers and Environment Complexity
// ============================================================================

/// Task complexity tiers as per V3.2 spec
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskTier {
    /// Tier 1: Single tool call, deterministic (~600 environments)
    SingleTool,
    /// Tier 2: 2-5 tool calls in sequence (~500 environments)
    MultiToolSequential,
    /// Tier 3: Parallel + conditional tool calls (~400 environments)
    MultiToolParallel,
    /// Tier 4: 10+ calls, complex workflows (~300 environments)
    ComplexWorkflow,
}

impl TaskTier {
    /// Expected tool call range for this tier
    pub fn expected_tool_calls(&self) -> (usize, usize) {
        match self {
            TaskTier::SingleTool => (1, 1),
            TaskTier::MultiToolSequential => (2, 5),
            TaskTier::MultiToolParallel => (3, 8),
            TaskTier::ComplexWorkflow => (10, 50),
        }
    }
    
    /// Base number of environments in this tier
    pub fn environment_count(&self) -> usize {
        match self {
            TaskTier::SingleTool => 600,
            TaskTier::MultiToolSequential => 500,
            TaskTier::MultiToolParallel => 400,
            TaskTier::ComplexWorkflow => 300,
        }
    }
    
    /// Difficulty weight for curriculum learning
    pub fn difficulty_weight(&self) -> f32 {
        match self {
            TaskTier::SingleTool => 1.0,
            TaskTier::MultiToolSequential => 2.0,
            TaskTier::MultiToolParallel => 3.5,
            TaskTier::ComplexWorkflow => 5.0,
        }
    }
}

// ============================================================================
// Reward Computation
// ============================================================================

/// Reward weights for agent training
#[derive(Clone, Debug)]
pub struct RewardWeights {
    /// Weight for task correctness (default: 0.5)
    pub correctness: f32,
    /// Weight for format compliance (default: 0.2)
    pub format: f32,
    /// Weight for efficiency (default: 0.15)
    pub efficiency: f32,
    /// Weight for safety (default: 0.15)
    pub safety: f32,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            correctness: 0.5,
            format: 0.2,
            efficiency: 0.15,
            safety: 0.15,
        }
    }
}

/// Breakdown of reward components
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RewardBreakdown {
    /// Task completion accuracy (0-1)
    pub correctness: f32,
    /// Valid tool-call JSON structure (0-1)
    pub format: f32,
    /// Efficiency - minimize unnecessary calls (0-1)
    pub efficiency: f32,
    /// Safety - no harmful operations (0-1)
    pub safety: f32,
}

impl RewardBreakdown {
    /// Compute total reward with weights
    pub fn total(&self, weights: &RewardWeights) -> f32 {
        weights.correctness * self.correctness
            + weights.format * self.format
            + weights.efficiency * self.efficiency
            + weights.safety * self.safety
    }
}

/// Computes rewards for agent trajectories
pub struct AgentRewardComputer {
    /// Reward weights
    pub weights: RewardWeights,
    /// Maximum allowed tool calls per tier
    pub max_tool_calls: HashMap<TaskTier, usize>,
    /// Penalty for security violations
    pub security_penalty: f32,
}

impl Default for AgentRewardComputer {
    fn default() -> Self {
        let mut max_calls = HashMap::new();
        max_calls.insert(TaskTier::SingleTool, 2);
        max_calls.insert(TaskTier::MultiToolSequential, 10);
        max_calls.insert(TaskTier::MultiToolParallel, 15);
        max_calls.insert(TaskTier::ComplexWorkflow, 100);
        
        Self {
            weights: RewardWeights::default(),
            max_tool_calls: max_calls,
            security_penalty: -1.0,
        }
    }
}

impl AgentRewardComputer {
    pub fn new(weights: RewardWeights) -> Self {
        Self {
            weights,
            ..Default::default()
        }
    }
    
    /// Compute reward for a trajectory
    pub fn compute_reward(&self, trajectory: &mut AgentTrajectory, ground_truth: Option<&str>) {
        let mut breakdown = RewardBreakdown::default();
        
        // 1. Correctness: Did the task complete successfully?
        breakdown.correctness = if trajectory.task_completed {
            if let Some(gt) = ground_truth {
                // Compare final output to ground truth
                self.compute_correctness_score(trajectory, gt)
            } else {
                1.0 // Assume correct if marked complete
            }
        } else {
            0.0
        };
        
        // 2. Format: Are all tool calls valid JSON?
        breakdown.format = self.compute_format_score(trajectory);
        
        // 3. Efficiency: Minimize unnecessary tool calls
        breakdown.efficiency = self.compute_efficiency_score(trajectory);
        
        // 4. Safety: Check for security violations
        breakdown.safety = self.compute_safety_score(trajectory);
        
        // Compute total reward
        trajectory.reward_breakdown = breakdown.clone();
        trajectory.reward = breakdown.total(&self.weights);
        
        // Apply security penalty if violated
        if breakdown.safety < 0.5 {
            trajectory.reward += self.security_penalty;
        }
    }
    
    fn compute_correctness_score(&self, trajectory: &AgentTrajectory, _ground_truth: &str) -> f32 {
        // Simplified: Check if all tool responses were successful
        let total_responses: usize = trajectory.turns.iter()
            .map(|t| t.tool_responses.len())
            .sum();
        let successful: usize = trajectory.turns.iter()
            .map(|t| t.successful_tool_count())
            .sum();
        
        if total_responses == 0 {
            return 0.0;
        }
        
        successful as f32 / total_responses as f32
    }
    
    fn compute_format_score(&self, trajectory: &AgentTrajectory) -> f32 {
        // Check that all tool calls have valid structure
        let total_calls = trajectory.total_tool_calls();
        if total_calls == 0 {
            return 1.0; // No calls means no format errors
        }
        
        let valid_calls: usize = trajectory.turns.iter()
            .flat_map(|t| &t.tool_calls)
            .filter(|tc| self.is_valid_tool_call(tc))
            .count();
        
        valid_calls as f32 / total_calls as f32
    }
    
    fn is_valid_tool_call(&self, tool_call: &ToolCall) -> bool {
        // Check required fields
        !tool_call.id.is_empty()
            && !tool_call.function_name.is_empty()
            && tool_call.arguments.is_object()
    }
    
    fn compute_efficiency_score(&self, trajectory: &AgentTrajectory) -> f32 {
        let total_calls = trajectory.total_tool_calls();
        let max_calls = self.max_tool_calls
            .get(&trajectory.task_tier)
            .copied()
            .unwrap_or(10);
        
        if total_calls == 0 {
            return 0.5; // Neutral if no calls (maybe wrong)
        }
        
        let (min_expected, _) = trajectory.task_tier.expected_tool_calls();
        
        // Efficiency = 1 if at minimum, decreasing as we approach max
        if total_calls <= min_expected {
            1.0
        } else if total_calls >= max_calls {
            0.0
        } else {
            let range = (max_calls - min_expected) as f32;
            let over = (total_calls - min_expected) as f32;
            1.0 - (over / range)
        }
    }
    
    fn compute_safety_score(&self, trajectory: &AgentTrajectory) -> f32 {
        // Check for security violations
        let violations: usize = trajectory.turns.iter()
            .flat_map(|t| &t.tool_responses)
            .filter(|r| r.status == ToolStatus::SecurityViolation)
            .count();
        
        if violations > 0 {
            0.0
        } else {
            1.0
        }
    }
}

// ============================================================================
// Synthetic Task Generator
// ============================================================================

/// Task template for synthetic generation
#[derive(Clone, Debug)]
pub struct TaskTemplate {
    /// Template ID
    pub id: String,
    /// Task tier
    pub tier: TaskTier,
    /// Tool types required
    pub required_tools: Vec<ToolType>,
    /// Template prompt (with placeholders)
    pub prompt_template: String,
    /// Expected output template
    pub expected_output_template: String,
}

/// Generator for synthetic agent tasks
pub struct SyntheticTaskGenerator {
    /// Templates by tier
    templates: HashMap<TaskTier, Vec<TaskTemplate>>,
    /// Random seed for reproducibility
    seed: u64,
}

impl SyntheticTaskGenerator {
    pub fn new(seed: u64) -> Self {
        let mut gen = Self {
            templates: HashMap::new(),
            seed,
        };
        gen.initialize_templates();
        gen
    }
    
    fn initialize_templates(&mut self) {
        // Tier 1: Single tool tasks
        self.templates.insert(TaskTier::SingleTool, vec![
            TaskTemplate {
                id: "calc_1".to_string(),
                tier: TaskTier::SingleTool,
                required_tools: vec![ToolType::Reasoning],
                prompt_template: "Calculate {expression}".to_string(),
                expected_output_template: "{result}".to_string(),
            },
            TaskTemplate {
                id: "read_file_1".to_string(),
                tier: TaskTier::SingleTool,
                required_tools: vec![ToolType::FileIO],
                prompt_template: "Read the contents of {filepath}".to_string(),
                expected_output_template: "{file_contents}".to_string(),
            },
            TaskTemplate {
                id: "search_1".to_string(),
                tier: TaskTier::SingleTool,
                required_tools: vec![ToolType::WebSearch],
                prompt_template: "Search for {query}".to_string(),
                expected_output_template: "{search_results}".to_string(),
            },
        ]);
        
        // Tier 2: Multi-tool sequential
        self.templates.insert(TaskTier::MultiToolSequential, vec![
            TaskTemplate {
                id: "read_and_process_1".to_string(),
                tier: TaskTier::MultiToolSequential,
                required_tools: vec![ToolType::FileIO, ToolType::CodeExecution],
                prompt_template: "Read {filepath} and count the number of lines".to_string(),
                expected_output_template: "The file has {line_count} lines".to_string(),
            },
            TaskTemplate {
                id: "search_and_summarize_1".to_string(),
                tier: TaskTier::MultiToolSequential,
                required_tools: vec![ToolType::WebSearch, ToolType::Reasoning],
                prompt_template: "Search for {query} and summarize the top result".to_string(),
                expected_output_template: "{summary}".to_string(),
            },
        ]);
        
        // Tier 3: Parallel + conditional
        self.templates.insert(TaskTier::MultiToolParallel, vec![
            TaskTemplate {
                id: "parallel_fetch_1".to_string(),
                tier: TaskTier::MultiToolParallel,
                required_tools: vec![ToolType::WebSearch, ToolType::FileIO],
                prompt_template: "Fetch data from {url1}, {url2}, and {url3}, then save results".to_string(),
                expected_output_template: "Saved {count} results to {output_path}".to_string(),
            },
        ]);
        
        // Tier 4: Complex workflows
        self.templates.insert(TaskTier::ComplexWorkflow, vec![
            TaskTemplate {
                id: "debug_workflow_1".to_string(),
                tier: TaskTier::ComplexWorkflow,
                required_tools: vec![ToolType::FileIO, ToolType::CodeExecution, ToolType::Reasoning],
                prompt_template: "Debug the failing tests in {project_path}".to_string(),
                expected_output_template: "Fixed {issue_count} issues in {files_modified} files".to_string(),
            },
        ]);
    }
    
    /// Generate a task from a template
    pub fn generate_task(&self, tier: TaskTier, _params: &HashMap<String, String>) -> Option<String> {
        let templates = self.templates.get(&tier)?;
        if templates.is_empty() {
            return None;
        }
        
        // Simple deterministic selection based on seed
        let idx = (self.seed as usize) % templates.len();
        let template = &templates[idx];
        
        // For now, return the template directly
        // In production, would substitute {params}
        Some(template.prompt_template.clone())
    }
    
    /// Get all templates for a tier
    pub fn get_templates(&self, tier: TaskTier) -> &[TaskTemplate] {
        self.templates.get(&tier).map(|v| v.as_slice()).unwrap_or(&[])
    }
    
    /// Total environment count across all tiers
    pub fn total_environments(&self) -> usize {
        TaskTier::SingleTool.environment_count()
            + TaskTier::MultiToolSequential.environment_count()
            + TaskTier::MultiToolParallel.environment_count()
            + TaskTier::ComplexWorkflow.environment_count()
    }
}

// ============================================================================
// Agent GRPO Extension
// ============================================================================

/// Extended GRPO trainer for agent/tool-use tasks
pub struct AgentGRPOTrainer {
    /// Base KL penalty coefficient
    pub beta: f64,
    /// Reward weights
    pub reward_weights: RewardWeights,
    /// Reward computer
    pub reward_computer: AgentRewardComputer,
    /// Discount factor for multi-turn rewards
    pub gamma: f64,
    /// Temperature for advantage computation
    pub advantage_temperature: f64,
}

impl AgentGRPOTrainer {
    pub fn new(beta: f64, reward_weights: RewardWeights) -> Self {
        Self {
            beta,
            reward_weights: reward_weights.clone(),
            reward_computer: AgentRewardComputer::new(reward_weights),
            gamma: 0.99,
            advantage_temperature: 1.0,
        }
    }
    
    /// Compute GRPO loss for agent trajectories
    /// 
    /// Extends standard GRPO to handle multi-turn trajectories with tool-use rewards.
    /// 
    /// # Arguments
    /// * `logits` - Policy logits for all turns concatenated (G, TotalSeq, Vocab)
    /// * `input_ids` - Token IDs for all turns (G, TotalSeq)
    /// * `trajectories` - Agent trajectories with rewards
    /// * `ref_logits` - Reference model logits (G, TotalSeq, Vocab)
    /// * `turn_boundaries` - Start/end indices for each turn in the sequence
    pub fn compute_trajectory_loss(
        &self,
        logits: &Tensor,
        input_ids: &Tensor,
        trajectories: &[AgentTrajectory],
        ref_logits: &Tensor,
        turn_boundaries: &[(usize, usize)],
    ) -> Result<Tensor> {
        let device = logits.device();
        let (g, seq, _vocab) = logits.dims3()?;
        
        // 1. Extract per-trajectory rewards
        let rewards: Vec<f32> = trajectories.iter().map(|t| t.reward).collect();
        let rewards_tensor = Tensor::from_slice(&rewards, (g,), device)?;
        
        // 2. Compute turn-level advantages with discount
        let turn_advantages = self.compute_turn_advantages(&rewards_tensor, turn_boundaries, device)?;
        
        // 3. Compute policy log probs
        let log_probs = candle_nn::ops::log_softmax(logits, 2)?;
        let log_probs_tokens = log_probs.gather(&input_ids.unsqueeze(2)?, 2)?.squeeze(2)?;
        
        // 4. Compute turn-weighted loss
        // Weight each token by its turn's advantage
        let weighted_log_probs = self.apply_turn_weights(&log_probs_tokens, &turn_advantages, turn_boundaries)?;
        let seq_log_probs = weighted_log_probs.sum(1)?;
        
        // 5. Compute KL divergence
        let ref_log_probs = candle_nn::ops::log_softmax(ref_logits, 2)?;
        let kl = (log_probs.exp()? * (log_probs - ref_log_probs)?)?.sum(2)?;
        let mean_kl = (kl.sum(1)? / seq as f64)?;
        
        // 6. Compute advantages from rewards (group-relative)
        let mean_r = (rewards_tensor.sum_all()? / g as f64)?;
        let diff = (rewards_tensor.clone() - mean_r.broadcast_as(rewards_tensor.shape())?)?;
        let var = (diff.sqr()?.sum_all()? / g as f64)?;
        let std = (var.sqrt()? + 1e-8)?;
        let advantages = (diff.clone() / std.broadcast_as(diff.shape())?)?;
        
        // 7. Final loss: -advantages * log_probs + beta * KL
        let adv_loss = (advantages * seq_log_probs)?;
        let kl_penalty = (mean_kl * self.beta)?;
        let loss = (kl_penalty - adv_loss)?;
        
        let mean_loss = (loss.sum_all()? / g as f64)?;
        
        Ok(mean_loss)
    }
    
    /// Compute per-turn advantages with temporal discounting
    fn compute_turn_advantages(
        &self,
        rewards: &Tensor,
        turn_boundaries: &[(usize, usize)],
        device: &Device,
    ) -> Result<Tensor> {
        let _g = rewards.elem_count();
        let num_turns = turn_boundaries.len();
        
        // Simple: each turn gets discounted share of final reward
        let mut turn_advantages = vec![0.0f32; num_turns];
        
        // Backward pass to assign credit
        for (i, _) in turn_boundaries.iter().enumerate().rev() {
            let discount = self.gamma.powi((num_turns - 1 - i) as i32) as f32;
            turn_advantages[i] = discount;
        }
        
        // Normalize
        let sum: f32 = turn_advantages.iter().sum();
        if sum > 0.0 {
            for adv in &mut turn_advantages {
                *adv /= sum;
            }
        }
        
        Tensor::from_slice(&turn_advantages, (num_turns,), device)
    }
    
    /// Apply turn-level weights to token log probs
    fn apply_turn_weights(
        &self,
        log_probs: &Tensor,
        _turn_advantages: &Tensor,
        _turn_boundaries: &[(usize, usize)],
    ) -> Result<Tensor> {
        // For now, return unweighted log probs
        // Full implementation would weight each turn's tokens by its advantage
        Ok(log_probs.clone())
    }
    
    /// Process trajectories and compute rewards
    pub fn process_trajectories(&self, trajectories: &mut [AgentTrajectory]) {
        for trajectory in trajectories {
            self.reward_computer.compute_reward(trajectory, None);
        }
    }
}

// ============================================================================
// Curriculum Learning
// ============================================================================

/// Curriculum scheduler for progressive difficulty training
pub struct AgentCurriculumScheduler {
    /// Current training step
    pub step: usize,
    /// Steps to spend on each tier before advancing
    pub tier_steps: HashMap<TaskTier, usize>,
    /// Current tier
    pub current_tier: TaskTier,
    /// Enable mixing tiers or strict progression
    pub mixing_enabled: bool,
    /// Mix ratio for lower tiers when advanced
    pub mix_ratio: f32,
}

impl Default for AgentCurriculumScheduler {
    fn default() -> Self {
        let mut tier_steps = HashMap::new();
        tier_steps.insert(TaskTier::SingleTool, 10000);
        tier_steps.insert(TaskTier::MultiToolSequential, 20000);
        tier_steps.insert(TaskTier::MultiToolParallel, 30000);
        tier_steps.insert(TaskTier::ComplexWorkflow, usize::MAX);
        
        Self {
            step: 0,
            tier_steps,
            current_tier: TaskTier::SingleTool,
            mixing_enabled: true,
            mix_ratio: 0.2,
        }
    }
}

impl AgentCurriculumScheduler {
    /// Advance training step and potentially update tier
    pub fn step(&mut self) {
        self.step += 1;
        
        // Check if we should advance tier
        let tier_threshold = self.tier_steps.get(&self.current_tier).copied().unwrap_or(usize::MAX);
        
        if self.step >= tier_threshold {
            self.current_tier = match self.current_tier {
                TaskTier::SingleTool => TaskTier::MultiToolSequential,
                TaskTier::MultiToolSequential => TaskTier::MultiToolParallel,
                TaskTier::MultiToolParallel => TaskTier::ComplexWorkflow,
                TaskTier::ComplexWorkflow => TaskTier::ComplexWorkflow,
            };
        }
    }
    
    /// Get the current tier for task sampling
    pub fn get_current_tier(&self) -> TaskTier {
        self.current_tier
    }
    
    /// Sample a tier (may include lower tiers if mixing enabled)
    pub fn sample_tier(&self, rng_value: f32) -> TaskTier {
        if !self.mixing_enabled {
            return self.current_tier;
        }
        
        // With mix_ratio probability, sample from a lower tier
        if rng_value < self.mix_ratio {
            match self.current_tier {
                TaskTier::SingleTool => TaskTier::SingleTool,
                TaskTier::MultiToolSequential => TaskTier::SingleTool,
                TaskTier::MultiToolParallel => {
                    if rng_value < self.mix_ratio / 2.0 {
                        TaskTier::SingleTool
                    } else {
                        TaskTier::MultiToolSequential
                    }
                }
                TaskTier::ComplexWorkflow => {
                    if rng_value < self.mix_ratio / 3.0 {
                        TaskTier::SingleTool
                    } else if rng_value < 2.0 * self.mix_ratio / 3.0 {
                        TaskTier::MultiToolSequential
                    } else {
                        TaskTier::MultiToolParallel
                    }
                }
            }
        } else {
            self.current_tier
        }
    }
    
    /// Get progress within current tier (0.0 - 1.0)
    pub fn tier_progress(&self) -> f32 {
        let tier_threshold = self.tier_steps.get(&self.current_tier).copied().unwrap_or(usize::MAX);
        let prev_threshold = match self.current_tier {
            TaskTier::SingleTool => 0,
            TaskTier::MultiToolSequential => self.tier_steps.get(&TaskTier::SingleTool).copied().unwrap_or(0),
            TaskTier::MultiToolParallel => self.tier_steps.get(&TaskTier::MultiToolSequential).copied().unwrap_or(0),
            TaskTier::ComplexWorkflow => self.tier_steps.get(&TaskTier::MultiToolParallel).copied().unwrap_or(0),
        };
        
        let steps_in_tier = self.step.saturating_sub(prev_threshold);
        let tier_duration = tier_threshold.saturating_sub(prev_threshold);
        
        if tier_duration == 0 {
            1.0
        } else {
            (steps_in_tier as f32 / tier_duration as f32).min(1.0)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_type_complexity() {
        assert_eq!(ToolType::Reasoning.complexity(), 1);
        assert_eq!(ToolType::FileIO.complexity(), 2);
        assert_eq!(ToolType::CodeExecution.complexity(), 3);
    }

    #[test]
    fn test_tool_status() {
        assert_eq!(ToolStatus::Success, ToolStatus::Success);
        assert_ne!(ToolStatus::Success, ToolStatus::Error);
    }

    #[test]
    fn test_task_tier_expected_calls() {
        let (min, max) = TaskTier::SingleTool.expected_tool_calls();
        assert_eq!(min, 1);
        assert_eq!(max, 1);
        
        let (min, max) = TaskTier::ComplexWorkflow.expected_tool_calls();
        assert_eq!(min, 10);
        assert_eq!(max, 50);
    }

    #[test]
    fn test_task_tier_environment_count() {
        assert_eq!(TaskTier::SingleTool.environment_count(), 600);
        assert_eq!(TaskTier::MultiToolSequential.environment_count(), 500);
        assert_eq!(TaskTier::MultiToolParallel.environment_count(), 400);
        assert_eq!(TaskTier::ComplexWorkflow.environment_count(), 300);
    }

    #[test]
    fn test_agent_turn() {
        let mut turn = AgentTurn::new(0);
        turn.tool_calls.push(ToolCall {
            id: "tc_1".to_string(),
            tool_type: ToolType::Reasoning,
            function_name: "calculate".to_string(),
            arguments: serde_json::json!({"expression": "2 + 2"}),
        });
        
        assert_eq!(turn.tool_call_count(), 1);
    }

    #[test]
    fn test_agent_trajectory() {
        let mut trajectory = AgentTrajectory::new(
            "Calculate 2+2".to_string(),
            TaskTier::SingleTool,
        );
        
        let mut turn = AgentTurn::new(0);
        turn.tool_calls.push(ToolCall {
            id: "tc_1".to_string(),
            tool_type: ToolType::Reasoning,
            function_name: "calculate".to_string(),
            arguments: serde_json::json!({"expression": "2 + 2"}),
        });
        turn.tool_responses.push(ToolResponse {
            call_id: "tc_1".to_string(),
            status: ToolStatus::Success,
            content: "4".to_string(),
            error: None,
            execution_time_ms: 10,
        });
        
        trajectory.add_turn(turn);
        
        assert_eq!(trajectory.num_turns(), 1);
        assert_eq!(trajectory.total_tool_calls(), 1);
    }

    #[test]
    fn test_reward_breakdown() {
        let breakdown = RewardBreakdown {
            correctness: 1.0,
            format: 1.0,
            efficiency: 0.8,
            safety: 1.0,
        };
        
        let weights = RewardWeights::default();
        let total = breakdown.total(&weights);
        
        // 0.5*1.0 + 0.2*1.0 + 0.15*0.8 + 0.15*1.0 = 0.5 + 0.2 + 0.12 + 0.15 = 0.97
        assert!((total - 0.97).abs() < 0.01);
    }

    #[test]
    fn test_reward_computer() {
        let computer = AgentRewardComputer::default();
        
        let mut trajectory = AgentTrajectory::new(
            "Test task".to_string(),
            TaskTier::SingleTool,
        );
        
        let mut turn = AgentTurn::new(0);
        turn.tool_calls.push(ToolCall {
            id: "tc_1".to_string(),
            tool_type: ToolType::Reasoning,
            function_name: "calculate".to_string(),
            arguments: serde_json::json!({"x": 1}),
        });
        turn.tool_responses.push(ToolResponse {
            call_id: "tc_1".to_string(),
            status: ToolStatus::Success,
            content: "1".to_string(),
            error: None,
            execution_time_ms: 5,
        });
        turn.is_terminal = true;
        
        trajectory.add_turn(turn);
        trajectory.task_completed = true;
        
        computer.compute_reward(&mut trajectory, None);
        
        assert!(trajectory.reward > 0.0);
        assert_eq!(trajectory.reward_breakdown.format, 1.0);
        assert_eq!(trajectory.reward_breakdown.safety, 1.0);
    }

    #[test]
    fn test_synthetic_task_generator() {
        let generator = SyntheticTaskGenerator::new(42);
        
        assert_eq!(generator.total_environments(), 1800);
        
        let task = generator.generate_task(TaskTier::SingleTool, &HashMap::new());
        assert!(task.is_some());
    }

    #[test]
    fn test_curriculum_scheduler() {
        let mut scheduler = AgentCurriculumScheduler::default();
        
        assert_eq!(scheduler.get_current_tier(), TaskTier::SingleTool);
        
        // Advance past first tier
        for _ in 0..10001 {
            scheduler.step();
        }
        
        assert_eq!(scheduler.get_current_tier(), TaskTier::MultiToolSequential);
    }

    #[test]
    fn test_curriculum_mixing() {
        let scheduler = AgentCurriculumScheduler {
            current_tier: TaskTier::MultiToolSequential,
            mixing_enabled: true,
            mix_ratio: 0.2,
            ..Default::default()
        };
        
        // Low RNG should sample from lower tier
        let tier = scheduler.sample_tier(0.1);
        assert_eq!(tier, TaskTier::SingleTool);
        
        // High RNG should sample current tier
        let tier = scheduler.sample_tier(0.9);
        assert_eq!(tier, TaskTier::MultiToolSequential);
    }

    #[test]
    fn test_agent_grpo_trainer() {
        let weights = RewardWeights::default();
        let trainer = AgentGRPOTrainer::new(0.01, weights);
        
        assert_eq!(trainer.beta, 0.01);
        assert_eq!(trainer.gamma, 0.99);
    }
}
