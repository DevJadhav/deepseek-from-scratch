use deepseek_from_scratch_in_rust::model::mla::{MultiHeadLatentAttention, DeepSeekAttention};
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;

#[test]
fn test_mla_shapes() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let d_model = 64;
    let num_heads = 4;
    let d_latent = 32;
    
    let vars = VarBuilder::zeros(DType::F32, &device);
    let mla = MultiHeadLatentAttention::new(d_model, num_heads, d_latent, vars)?;

    let batch_size = 2;
    let seq_len = 10;
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;

    let output = mla.forward(&input, None)?;

    assert_eq!(output.dims3()?, (batch_size, seq_len, d_model));
    Ok(())
}

#[test]
fn test_deepseek_attention_shapes() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let d_model = 64;
    let num_heads = 4;
    let d_latent = 32;
    let d_rope = 16;
    
    let vars = VarBuilder::zeros(DType::F32, &device);
    let attn = DeepSeekAttention::new(d_model, num_heads, d_latent, d_rope, vars)?;

    let batch_size = 2;
    let seq_len = 10;
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;

    let output = attn.forward(&input, None)?;

    assert_eq!(output.dims3()?, (batch_size, seq_len, d_model));
    Ok(())
}

// ============================================================================
// Phase 5: Agent/Tool-Use Training Tests
// ============================================================================

use deepseek_from_scratch_in_rust::training::agent::{
    ToolType, ToolStatus, ToolCall, ToolResponse, AgentTurn, AgentTrajectory,
    TaskTier, RewardWeights, RewardBreakdown, AgentRewardComputer,
};

#[test]
fn test_tool_call_creation() {
    let call = ToolCall {
        id: "call_001".to_string(),
        tool_type: ToolType::CodeExecution,
        function_name: "run_python".to_string(),
        arguments: serde_json::json!({"code": "print('hello')"}),
    };
    
    assert_eq!(call.id, "call_001");
    assert!(matches!(call.tool_type, ToolType::CodeExecution));
    assert_eq!(call.function_name, "run_python");
}

#[test]
fn test_tool_response_creation() {
    let response = ToolResponse {
        call_id: "call_001".to_string(),
        status: ToolStatus::Success,
        content: "Execution complete".to_string(),
        error: None,
        execution_time_ms: 100,
    };
    
    assert_eq!(response.call_id, "call_001");
    assert!(matches!(response.status, ToolStatus::Success));
    assert_eq!(response.execution_time_ms, 100);
}

#[test]
fn test_agent_turn_creation() {
    let turn = AgentTurn::new(0);
    
    assert_eq!(turn.turn_index, 0);
    assert!(turn.tool_calls.is_empty());
    assert!(!turn.is_terminal);
}

#[test]
fn test_agent_trajectory_creation() {
    let trajectory = AgentTrajectory::new(
        "Calculate 2+2".to_string(),
        TaskTier::SingleTool
    );
    
    assert_eq!(trajectory.task, "Calculate 2+2");
    assert!(matches!(trajectory.task_tier, TaskTier::SingleTool));
    assert!(trajectory.turns.is_empty());
    assert!(!trajectory.task_completed);
}

#[test]
fn test_trajectory_with_turns() {
    let mut trajectory = AgentTrajectory::new(
        "Calculate 2+2".to_string(),
        TaskTier::SingleTool
    );
    
    // Create a turn with a tool call
    let mut turn = AgentTurn::new(0);
    turn.assistant_response = "I'll calculate that for you.".to_string();
    
    let tool_call = ToolCall {
        id: "call_1".to_string(),
        tool_type: ToolType::Reasoning,
        function_name: "calculate".to_string(),
        arguments: serde_json::json!({"expression": "2+2"}),
    };
    turn.tool_calls.push(tool_call);
    
    let tool_response = ToolResponse {
        call_id: "call_1".to_string(),
        status: ToolStatus::Success,
        content: "Result: 4".to_string(),
        error: None,
        execution_time_ms: 10,
    };
    turn.tool_responses.push(tool_response);
    turn.is_terminal = true;
    
    trajectory.add_turn(turn);
    trajectory.task_completed = true;
    
    assert_eq!(trajectory.num_turns(), 1);
    assert_eq!(trajectory.total_tool_calls(), 1);
    assert!(trajectory.task_completed);
}

#[test]
fn test_task_tier_expected_calls() {
    // Tier 1: Single tool
    let (min, max) = TaskTier::SingleTool.expected_tool_calls();
    assert_eq!(min, 1);
    assert_eq!(max, 1);
    
    // Tier 2: Multi-tool sequential
    let (min, max) = TaskTier::MultiToolSequential.expected_tool_calls();
    assert_eq!(min, 2);
    assert_eq!(max, 5);
    
    // Tier 3: Multi-tool parallel
    let (min, max) = TaskTier::MultiToolParallel.expected_tool_calls();
    assert_eq!(min, 3);
    assert_eq!(max, 8);
    
    // Tier 4: Complex workflow
    let (min, max) = TaskTier::ComplexWorkflow.expected_tool_calls();
    assert_eq!(min, 10);
    assert_eq!(max, 50);
}

#[test]
fn test_tier_environment_counts() {
    // Total should be ~1,800
    let total: usize = [
        TaskTier::SingleTool,
        TaskTier::MultiToolSequential,
        TaskTier::MultiToolParallel,
        TaskTier::ComplexWorkflow,
    ].iter().map(|t| t.environment_count()).sum();
    
    assert_eq!(total, 1800);
    
    assert_eq!(TaskTier::SingleTool.environment_count(), 600);
    assert_eq!(TaskTier::MultiToolSequential.environment_count(), 500);
    assert_eq!(TaskTier::MultiToolParallel.environment_count(), 400);
    assert_eq!(TaskTier::ComplexWorkflow.environment_count(), 300);
}

#[test]
fn test_reward_weights() {
    let weights = RewardWeights::default();
    
    // Default weights should sum to 1.0
    let total = weights.correctness + weights.format + weights.efficiency + weights.safety;
    assert!((total - 1.0).abs() < 1e-6);
    
    // R_total = 0.5*R_correct + 0.2*R_format + 0.15*R_efficiency + 0.15*R_safety
    assert_eq!(weights.correctness, 0.5);
    assert_eq!(weights.format, 0.2);
    assert_eq!(weights.efficiency, 0.15);
    assert_eq!(weights.safety, 0.15);
}

#[test]
fn test_reward_breakdown_default() {
    let breakdown = RewardBreakdown::default();
    
    assert_eq!(breakdown.correctness, 0.0);
    assert_eq!(breakdown.format, 0.0);
    assert_eq!(breakdown.efficiency, 0.0);
    assert_eq!(breakdown.safety, 0.0);
}

#[test]
fn test_tool_type_complexity() {
    // Reasoning tools are simplest
    assert_eq!(ToolType::Reasoning.complexity(), 1);
    
    // Code execution is most complex (level 3)
    assert_eq!(ToolType::CodeExecution.complexity(), 3);
    
    // All tool types should have valid complexity (1-3)
    for tool_type in ToolType::all() {
        assert!(tool_type.complexity() >= 1);
        assert!(tool_type.complexity() <= 3);
    }
}

#[test]
fn test_tool_type_all() {
    let all_types = ToolType::all();
    
    assert_eq!(all_types.len(), 5);
    assert!(all_types.contains(&ToolType::CodeExecution));
    assert!(all_types.contains(&ToolType::WebSearch));
    assert!(all_types.contains(&ToolType::FileIO));
    assert!(all_types.contains(&ToolType::ApiCall));
    assert!(all_types.contains(&ToolType::Reasoning));
}

#[test]
fn test_turn_tool_call_count() {
    let mut turn = AgentTurn::new(0);
    
    assert_eq!(turn.tool_call_count(), 0);
    
    turn.tool_calls.push(ToolCall {
        id: "1".to_string(),
        tool_type: ToolType::Reasoning,
        function_name: "calc".to_string(),
        arguments: serde_json::json!({}),
    });
    
    assert_eq!(turn.tool_call_count(), 1);
    
    turn.tool_calls.push(ToolCall {
        id: "2".to_string(),
        tool_type: ToolType::WebSearch,
        function_name: "search".to_string(),
        arguments: serde_json::json!({}),
    });
    
    assert_eq!(turn.tool_call_count(), 2);
}

#[test]
fn test_turn_successful_tool_count() {
    let mut turn = AgentTurn::new(0);
    
    // Add successful response
    turn.tool_responses.push(ToolResponse {
        call_id: "1".to_string(),
        status: ToolStatus::Success,
        content: "OK".to_string(),
        error: None,
        execution_time_ms: 10,
    });
    
    // Add failed response
    turn.tool_responses.push(ToolResponse {
        call_id: "2".to_string(),
        status: ToolStatus::Error,
        content: "".to_string(),
        error: Some("Failed".to_string()),
        execution_time_ms: 10,
    });
    
    assert_eq!(turn.successful_tool_count(), 1);
    assert!(!turn.all_tools_succeeded());
}

#[test]
fn test_trajectory_tool_types_used() {
    let mut trajectory = AgentTrajectory::new(
        "Test task".to_string(),
        TaskTier::MultiToolSequential
    );
    
    let mut turn = AgentTurn::new(0);
    turn.tool_calls.push(ToolCall {
        id: "1".to_string(),
        tool_type: ToolType::WebSearch,
        function_name: "search".to_string(),
        arguments: serde_json::json!({}),
    });
    turn.tool_calls.push(ToolCall {
        id: "2".to_string(),
        tool_type: ToolType::Reasoning,
        function_name: "calc".to_string(),
        arguments: serde_json::json!({}),
    });
    trajectory.add_turn(turn);
    
    let types = trajectory.tool_types_used();
    assert!(types.contains(&ToolType::WebSearch));
    assert!(types.contains(&ToolType::Reasoning));
}

#[test]
fn test_task_tier_difficulty_weight() {
    // Higher tiers should have higher weights
    assert!(TaskTier::SingleTool.difficulty_weight() < TaskTier::MultiToolSequential.difficulty_weight());
    assert!(TaskTier::MultiToolSequential.difficulty_weight() < TaskTier::MultiToolParallel.difficulty_weight());
    assert!(TaskTier::MultiToolParallel.difficulty_weight() < TaskTier::ComplexWorkflow.difficulty_weight());
}

