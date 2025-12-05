"""
Agent/Tool-Use Training Module for DeepSeek-V3.2

This module provides infrastructure for training language models
with agentic capabilities including:
- Tool-call parsing and execution
- Multi-turn trajectory management
- Environment feedback handling
- Reward computation for agent tasks

Supports 5 tool categories:
- Code Execution (Python REPL, Shell, Jupyter)
- Web Search (Search API, URL fetch)
- File I/O (Read/write, directory ops)
- API Calls (REST APIs, databases)
- Reasoning Tools (Calculator, symbolic math)
"""

import json
import random
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Tool Call Format Specification
# ============================================================================

class ToolType(Enum):
    """Tool categories as per V3.2 specification"""
    CODE_EXECUTION = auto()  # Python REPL, Shell, Jupyter
    WEB_SEARCH = auto()       # Search API, URL fetch
    FILE_IO = auto()          # Read/write, directory ops
    API_CALL = auto()         # REST APIs, databases
    REASONING = auto()        # Calculator, symbolic math


class ToolStatus(Enum):
    """Tool execution status"""
    SUCCESS = auto()
    ERROR = auto()
    TIMEOUT = auto()
    PERMISSION_DENIED = auto()


@dataclass
class ToolCall:
    """Tool call JSON schema format for function calling"""
    id: str
    tool_type: ToolType
    function_name: str
    arguments: Dict[str, Any]
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps({
            "id": self.id,
            "tool_type": self.tool_type.name,
            "function_name": self.function_name,
            "arguments": self.arguments
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> Optional["ToolCall"]:
        """Parse from JSON string"""
        try:
            data = json.loads(json_str)
            return cls(
                id=data["id"],
                tool_type=ToolType[data["tool_type"]],
                function_name=data["function_name"],
                arguments=data.get("arguments", {})
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None


@dataclass
class ToolResponse:
    """Tool response from environment"""
    call_id: str
    status: ToolStatus
    content: str
    error: Optional[str] = None
    execution_time_ms: int = 0
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps({
            "call_id": self.call_id,
            "status": self.status.name,
            "content": self.content,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms
        }, indent=2)


# ============================================================================
# Agent Actions and Trajectories
# ============================================================================

class AgentActionType(Enum):
    """Types of actions an agent can take"""
    THINK = auto()        # Internal reasoning step
    TOOL_CALL = auto()    # Execute a tool
    RESPOND = auto()      # Final response to user


@dataclass
class AgentAction:
    """Single action in a trajectory"""
    action_type: AgentActionType
    content: str
    tool_call: Optional[ToolCall] = None
    tool_response: Optional[ToolResponse] = None


@dataclass
class AgentStep:
    """Complete step: action + optional environment feedback"""
    action: AgentAction
    tokens: List[int] = field(default_factory=list)
    token_start: int = 0
    token_end: int = 0


@dataclass
class AgentTrajectory:
    """Complete trajectory of agent interacting with environment"""
    prompt: str
    steps: List[AgentStep] = field(default_factory=list)
    final_output: str = ""
    task_completed: bool = False
    task_tier: "TaskTier" = None
    ground_truth: Optional[str] = None
    
    @property
    def total_tokens(self) -> int:
        """Total tokens in trajectory"""
        return sum(len(step.tokens) for step in self.steps)
    
    @property
    def num_tool_calls(self) -> int:
        """Count of tool calls made"""
        return sum(
            1 for step in self.steps 
            if step.action.action_type == AgentActionType.TOOL_CALL
        )
    
    @property
    def successful_tool_calls(self) -> int:
        """Count of successful tool calls"""
        return sum(
            1 for step in self.steps
            if step.action.action_type == AgentActionType.TOOL_CALL
            and step.action.tool_response is not None
            and step.action.tool_response.status == ToolStatus.SUCCESS
        )


# ============================================================================
# Task Tiers and Environment Complexity
# ============================================================================

class TaskTier(Enum):
    """Task complexity tiers as per V3.2 spec"""
    SINGLE_TOOL = 1          # Tier 1: Single tool call (~600 environments)
    MULTI_TOOL_SEQ = 2       # Tier 2: 2-5 tool calls in sequence (~500 environments)
    MULTI_TOOL_PARALLEL = 3  # Tier 3: Parallel + conditional (~400 environments)
    COMPLEX_WORKFLOW = 4     # Tier 4: 10+ calls, complex workflows (~300 environments)
    
    def expected_tool_calls(self) -> Tuple[int, int]:
        """Expected tool call range for this tier"""
        return {
            TaskTier.SINGLE_TOOL: (1, 1),
            TaskTier.MULTI_TOOL_SEQ: (2, 5),
            TaskTier.MULTI_TOOL_PARALLEL: (3, 8),
            TaskTier.COMPLEX_WORKFLOW: (10, 50),
        }[self]
    
    def environment_count(self) -> int:
        """Base number of environments in this tier"""
        return {
            TaskTier.SINGLE_TOOL: 600,
            TaskTier.MULTI_TOOL_SEQ: 500,
            TaskTier.MULTI_TOOL_PARALLEL: 400,
            TaskTier.COMPLEX_WORKFLOW: 300,
        }[self]


# ============================================================================
# Reward Computation
# ============================================================================

@dataclass
class RewardWeights:
    """Weights for multi-objective reward function
    
    R_total = w_correct * R_correctness 
            + w_format * R_format 
            + w_efficiency * R_efficiency 
            + w_safety * R_safety
    """
    correctness: float = 0.5
    format: float = 0.2
    efficiency: float = 0.15
    safety: float = 0.15
    
    def __post_init__(self):
        total = self.correctness + self.format + self.efficiency + self.safety
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"


@dataclass
class RewardBreakdown:
    """Breakdown of reward components"""
    correctness: float = 0.0   # Task completion accuracy (0-1)
    format: float = 0.0        # Valid tool-call JSON structure (0-1)
    efficiency: float = 0.0    # Minimize unnecessary calls (0-1)
    safety: float = 0.0        # No harmful operations (0-1)
    total: float = 0.0         # Weighted sum


class AgentRewardComputer:
    """Compute rewards for agent trajectories"""
    
    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        max_calls_per_tier: Optional[Dict[TaskTier, int]] = None,
        device: str = "cpu"
    ):
        self.weights = weights or RewardWeights()
        self.max_calls = max_calls_per_tier or {
            TaskTier.SINGLE_TOOL: 2,
            TaskTier.MULTI_TOOL_SEQ: 10,
            TaskTier.MULTI_TOOL_PARALLEL: 15,
            TaskTier.COMPLEX_WORKFLOW: 100,
        }
        self.device = device
    
    def compute_reward(
        self,
        trajectory: AgentTrajectory,
        ground_truth: Optional[str] = None
    ) -> RewardBreakdown:
        """Compute reward breakdown for a trajectory"""
        breakdown = RewardBreakdown()
        
        # 1. Correctness reward
        if trajectory.task_completed:
            if ground_truth is not None:
                breakdown.correctness = self._compute_correctness(
                    trajectory.final_output, ground_truth
                )
            else:
                breakdown.correctness = 1.0
        
        # 2. Format reward - check tool call JSON validity
        breakdown.format = self._compute_format_score(trajectory)
        
        # 3. Efficiency reward - penalize excess tool calls
        breakdown.efficiency = self._compute_efficiency(trajectory)
        
        # 4. Safety reward - check for dangerous operations
        breakdown.safety = self._compute_safety(trajectory)
        
        # Weighted total
        breakdown.total = (
            self.weights.correctness * breakdown.correctness +
            self.weights.format * breakdown.format +
            self.weights.efficiency * breakdown.efficiency +
            self.weights.safety * breakdown.safety
        )
        
        return breakdown
    
    def _compute_correctness(self, output: str, ground_truth: str) -> float:
        """Compute task correctness score"""
        # Simple exact match for now
        # Could use fuzzy matching, ROUGE, or semantic similarity
        output_clean = output.strip().lower()
        truth_clean = ground_truth.strip().lower()
        
        if output_clean == truth_clean:
            return 1.0
        
        # Partial credit for substring match
        if truth_clean in output_clean or output_clean in truth_clean:
            return 0.5
        
        return 0.0
    
    def _compute_format_score(self, trajectory: AgentTrajectory) -> float:
        """Check tool-call JSON format validity"""
        if trajectory.num_tool_calls == 0:
            return 1.0  # No tool calls to validate
        
        valid_calls = 0
        for step in trajectory.steps:
            if step.action.action_type == AgentActionType.TOOL_CALL:
                if step.action.tool_call is not None:
                    # Check if tool call has valid structure
                    try:
                        json.loads(step.action.tool_call.to_json())
                        valid_calls += 1
                    except json.JSONDecodeError:
                        pass
        
        return valid_calls / trajectory.num_tool_calls
    
    def _compute_efficiency(self, trajectory: AgentTrajectory) -> float:
        """Compute efficiency score based on tool call count"""
        if trajectory.task_tier is None:
            return 1.0
        
        max_calls = self.max_calls.get(trajectory.task_tier, 100)
        expected_min, expected_max = trajectory.task_tier.expected_tool_calls()
        
        num_calls = trajectory.num_tool_calls
        
        # Perfect score if within expected range
        if expected_min <= num_calls <= expected_max:
            return 1.0
        
        # Penalize based on deviation
        if num_calls < expected_min:
            return max(0.0, num_calls / expected_min)
        elif num_calls > max_calls:
            return 0.0
        else:
            # Between expected_max and max_calls
            excess = num_calls - expected_max
            penalty_range = max_calls - expected_max
            return max(0.0, 1.0 - (excess / penalty_range))
    
    def _compute_safety(self, trajectory: AgentTrajectory) -> float:
        """Check for dangerous operations"""
        dangerous_patterns = [
            r"rm\s+-rf",
            r"DROP\s+TABLE",
            r"DELETE\s+FROM.*WHERE\s+1=1",
            r"sudo\s+rm",
            r"chmod\s+777",
            r"curl.*\|\s*bash",
            r"eval\s*\(",
            r"exec\s*\(",
        ]
        
        for step in trajectory.steps:
            if step.action.tool_call is not None:
                args_str = json.dumps(step.action.tool_call.arguments)
                for pattern in dangerous_patterns:
                    if re.search(pattern, args_str, re.IGNORECASE):
                        return 0.0
        
        return 1.0
    
    def compute_batch_rewards(
        self,
        trajectories: List[AgentTrajectory],
        ground_truths: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Compute rewards for a batch of trajectories"""
        if ground_truths is None:
            ground_truths = [None] * len(trajectories)
        
        rewards = []
        for traj, gt in zip(trajectories, ground_truths):
            breakdown = self.compute_reward(traj, gt)
            rewards.append(breakdown.total)
        
        return torch.tensor(rewards, device=self.device, dtype=torch.float32)


# ============================================================================
# Synthetic Task Generator
# ============================================================================

@dataclass
class TaskTemplate:
    """Template for generating synthetic tasks"""
    name: str
    tier: TaskTier
    tool_types: List[ToolType]
    prompt_template: str
    expected_steps: int
    ground_truth_fn: Optional[Callable[..., str]] = None


class TaskGenerator:
    """Generate synthetic agent tasks across all tiers (~1,800 environments)"""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.templates: Dict[TaskTier, List[TaskTemplate]] = {}
        self._init_templates()
    
    def _init_templates(self):
        """Initialize task templates for each tier"""
        
        # Tier 1: Single Tool Tasks (600 environments)
        self.templates[TaskTier.SINGLE_TOOL] = [
            TaskTemplate(
                name="simple_calculation",
                tier=TaskTier.SINGLE_TOOL,
                tool_types=[ToolType.REASONING],
                prompt_template="Calculate: {expr}",
                expected_steps=1,
            ),
            TaskTemplate(
                name="web_lookup",
                tier=TaskTier.SINGLE_TOOL,
                tool_types=[ToolType.WEB_SEARCH],
                prompt_template="Search for information about: {topic}",
                expected_steps=1,
            ),
            TaskTemplate(
                name="file_read",
                tier=TaskTier.SINGLE_TOOL,
                tool_types=[ToolType.FILE_IO],
                prompt_template="Read the contents of file: {filename}",
                expected_steps=1,
            ),
            TaskTemplate(
                name="code_execute",
                tier=TaskTier.SINGLE_TOOL,
                tool_types=[ToolType.CODE_EXECUTION],
                prompt_template="Execute this Python code: {code}",
                expected_steps=1,
            ),
        ]
        
        # Tier 2: Multi-Tool Sequential Tasks (500 environments)
        self.templates[TaskTier.MULTI_TOOL_SEQ] = [
            TaskTemplate(
                name="search_and_summarize",
                tier=TaskTier.MULTI_TOOL_SEQ,
                tool_types=[ToolType.WEB_SEARCH, ToolType.REASONING],
                prompt_template="Search for {topic} and summarize the key points",
                expected_steps=3,
            ),
            TaskTemplate(
                name="read_and_process",
                tier=TaskTier.MULTI_TOOL_SEQ,
                tool_types=[ToolType.FILE_IO, ToolType.CODE_EXECUTION],
                prompt_template="Read {filename} and run analysis code",
                expected_steps=2,
            ),
            TaskTemplate(
                name="api_then_compute",
                tier=TaskTier.MULTI_TOOL_SEQ,
                tool_types=[ToolType.API_CALL, ToolType.REASONING],
                prompt_template="Fetch data from {api} and compute statistics",
                expected_steps=3,
            ),
        ]
        
        # Tier 3: Multi-Tool Parallel + Conditional (400 environments)
        self.templates[TaskTier.MULTI_TOOL_PARALLEL] = [
            TaskTemplate(
                name="parallel_search",
                tier=TaskTier.MULTI_TOOL_PARALLEL,
                tool_types=[ToolType.WEB_SEARCH, ToolType.WEB_SEARCH, ToolType.REASONING],
                prompt_template="Compare information about {topic1} and {topic2}",
                expected_steps=5,
            ),
            TaskTemplate(
                name="conditional_workflow",
                tier=TaskTier.MULTI_TOOL_PARALLEL,
                tool_types=[ToolType.FILE_IO, ToolType.CODE_EXECUTION, ToolType.API_CALL],
                prompt_template="If file exists, process it; otherwise fetch from API",
                expected_steps=4,
            ),
        ]
        
        # Tier 4: Complex Workflows (300 environments)
        self.templates[TaskTier.COMPLEX_WORKFLOW] = [
            TaskTemplate(
                name="research_pipeline",
                tier=TaskTier.COMPLEX_WORKFLOW,
                tool_types=[
                    ToolType.WEB_SEARCH, ToolType.WEB_SEARCH, ToolType.FILE_IO,
                    ToolType.CODE_EXECUTION, ToolType.REASONING
                ],
                prompt_template="Research {topic}, analyze data, and generate report",
                expected_steps=12,
            ),
            TaskTemplate(
                name="data_pipeline",
                tier=TaskTier.COMPLEX_WORKFLOW,
                tool_types=[
                    ToolType.API_CALL, ToolType.FILE_IO, ToolType.CODE_EXECUTION,
                    ToolType.CODE_EXECUTION, ToolType.FILE_IO
                ],
                prompt_template="ETL pipeline: fetch, transform, validate, store",
                expected_steps=15,
            ),
        ]
    
    def generate_task(self, tier: Optional[TaskTier] = None) -> Tuple[str, TaskTier]:
        """Generate a random task from specified tier or random tier"""
        if tier is None:
            tier = self.rng.choice(list(TaskTier))
        
        templates = self.templates.get(tier, [])
        if not templates:
            raise ValueError(f"No templates for tier {tier}")
        
        template = self.rng.choice(templates)
        
        # Fill in template variables
        prompt = self._fill_template(template.prompt_template)
        
        return prompt, tier
    
    def _fill_template(self, template: str) -> str:
        """Fill in template placeholders with random values"""
        # Simple placeholder filling
        replacements = {
            "{expr}": f"{self.rng.randint(1, 100)} + {self.rng.randint(1, 100)} * {self.rng.randint(1, 10)}",
            "{topic}": self.rng.choice(["machine learning", "climate change", "quantum computing", "economics"]),
            "{topic1}": self.rng.choice(["Python", "JavaScript", "Rust"]),
            "{topic2}": self.rng.choice(["Go", "TypeScript", "C++"]),
            "{filename}": self.rng.choice(["data.csv", "config.json", "README.md"]),
            "{code}": "print('Hello, World!')",
            "{api}": self.rng.choice(["/api/users", "/api/data", "/api/stats"]),
        }
        
        result = template
        for key, value in replacements.items():
            result = result.replace(key, value)
        
        return result
    
    def generate_batch(
        self,
        batch_size: int,
        tier: Optional[TaskTier] = None
    ) -> List[Tuple[str, TaskTier]]:
        """Generate a batch of tasks"""
        return [self.generate_task(tier) for _ in range(batch_size)]
    
    def get_tier_distribution(self) -> Dict[TaskTier, int]:
        """Get distribution of environments across tiers"""
        return {tier: tier.environment_count() for tier in TaskTier}
    
    def get_templates(self, tier: TaskTier) -> List[TaskTemplate]:
        """Get templates for a specific tier"""
        return self.templates.get(tier, [])


# ============================================================================
# Agent Environment Simulation
# ============================================================================

class AgentEnvironment:
    """Simulated environment for agent training"""
    
    def __init__(
        self,
        tool_handlers: Optional[Dict[ToolType, Callable]] = None,
        max_steps: int = 50,
        device: str = "cpu"
    ):
        self.tool_handlers = tool_handlers or self._default_handlers()
        self.max_steps = max_steps
        self.device = device
    
    def _default_handlers(self) -> Dict[ToolType, Callable]:
        """Create default mock tool handlers"""
        return {
            ToolType.CODE_EXECUTION: self._mock_code_execution,
            ToolType.WEB_SEARCH: self._mock_web_search,
            ToolType.FILE_IO: self._mock_file_io,
            ToolType.API_CALL: self._mock_api_call,
            ToolType.REASONING: self._mock_reasoning,
        }
    
    def _mock_code_execution(self, args: Dict[str, Any]) -> ToolResponse:
        """Mock code execution"""
        code = args.get("code", "")
        return ToolResponse(
            call_id="",
            status=ToolStatus.SUCCESS,
            content=f"Executed code: output = {hash(code) % 100}",
            execution_time_ms=100
        )
    
    def _mock_web_search(self, args: Dict[str, Any]) -> ToolResponse:
        """Mock web search"""
        query = args.get("query", "")
        return ToolResponse(
            call_id="",
            status=ToolStatus.SUCCESS,
            content=f"Search results for '{query}': [Result 1, Result 2, Result 3]",
            execution_time_ms=500
        )
    
    def _mock_file_io(self, args: Dict[str, Any]) -> ToolResponse:
        """Mock file I/O"""
        action = args.get("action", "read")
        path = args.get("path", "")
        return ToolResponse(
            call_id="",
            status=ToolStatus.SUCCESS,
            content=f"File {action} on {path}: success",
            execution_time_ms=50
        )
    
    def _mock_api_call(self, args: Dict[str, Any]) -> ToolResponse:
        """Mock API call"""
        endpoint = args.get("endpoint", "")
        return ToolResponse(
            call_id="",
            status=ToolStatus.SUCCESS,
            content=f"API response from {endpoint}: {{'status': 'ok'}}",
            execution_time_ms=200
        )
    
    def _mock_reasoning(self, args: Dict[str, Any]) -> ToolResponse:
        """Mock reasoning tool"""
        expression = args.get("expression", "0")
        try:
            # Safe eval for simple math
            result = eval(expression, {"__builtins__": {}}, {})
            return ToolResponse(
                call_id="",
                status=ToolStatus.SUCCESS,
                content=f"Result: {result}",
                execution_time_ms=10
            )
        except Exception as e:
            return ToolResponse(
                call_id="",
                status=ToolStatus.ERROR,
                content="",
                error=str(e),
                execution_time_ms=10
            )
    
    def execute_tool_call(self, tool_call: ToolCall) -> ToolResponse:
        """Execute a tool call and return response"""
        handler = self.tool_handlers.get(tool_call.tool_type)
        
        if handler is None:
            return ToolResponse(
                call_id=tool_call.id,
                status=ToolStatus.ERROR,
                content="",
                error=f"Unknown tool type: {tool_call.tool_type}",
                execution_time_ms=0
            )
        
        response = handler(tool_call.arguments)
        response.call_id = tool_call.id
        return response


# ============================================================================
# Agent GRPO Trainer (Extended for Multi-Turn)
# ============================================================================

@dataclass
class AgentGRPOConfig:
    """Configuration for Agent GRPO training"""
    beta: float = 0.04            # KL penalty coefficient
    gamma: float = 0.99           # Discount factor for multi-turn
    group_size: int = 8           # GRPO group size
    max_trajectory_len: int = 4096
    turn_credit_method: str = "exponential"  # 'exponential', 'uniform', 'final_only'
    use_curriculum: bool = True   # Progressive tier curriculum
    device: str = "cpu"


class AgentGRPOTrainer:
    """Extended GRPO trainer for multi-turn agent trajectories"""
    
    def __init__(self, config: Optional[AgentGRPOConfig] = None):
        self.config = config or AgentGRPOConfig()
        self.beta = self.config.beta
        self.gamma = self.config.gamma
        self.device = self.config.device
        
        # Curriculum state
        self.current_tier = TaskTier.SINGLE_TOOL
        self.steps_completed = 0
        self.tier_thresholds = {
            TaskTier.SINGLE_TOOL: 10000,
            TaskTier.MULTI_TOOL_SEQ: 20000,
            TaskTier.MULTI_TOOL_PARALLEL: 30000,
            TaskTier.COMPLEX_WORKFLOW: float('inf'),
        }
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        rewards: torch.Tensor,
        ref_logits: torch.Tensor,
        turn_boundaries: Optional[List[List[Tuple[int, int]]]] = None
    ) -> torch.Tensor:
        """Compute GRPO loss with multi-turn credit assignment
        
        Args:
            logits: Policy logits (G, Seq, Vocab)
            input_ids: Token IDs (G, Seq)
            rewards: Trajectory rewards (G,)
            ref_logits: Reference policy logits (G, Seq, Vocab)
            turn_boundaries: Per-sample list of (start, end) token indices for each turn
        
        Returns:
            Scalar loss tensor
        """
        G, Seq, Vocab = logits.shape
        
        # 1. Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)  # (G, Seq)
        
        # 2. Compute per-turn advantages if boundaries provided
        if turn_boundaries is not None:
            turn_advantages = self._compute_turn_advantages(rewards, turn_boundaries)
            weighted_log_probs = self._apply_turn_weights(
                token_log_probs, turn_advantages, turn_boundaries
            )
            seq_log_probs = weighted_log_probs.sum(dim=1)  # (G,)
        else:
            seq_log_probs = token_log_probs.sum(dim=1)  # (G,)
        
        # 3. KL divergence
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1)  # (G, Seq)
        mean_kl = kl.mean(dim=1)  # (G,)
        
        # 4. Group-relative advantages
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r  # (G,)
        
        # 5. Final loss
        loss = -(advantages * seq_log_probs) + self.beta * mean_kl
        
        return loss.mean()
    
    def _compute_turn_advantages(
        self,
        rewards: torch.Tensor,
        turn_boundaries: List[List[Tuple[int, int]]]
    ) -> List[torch.Tensor]:
        """Compute per-turn advantages with temporal discounting"""
        turn_advantages = []
        
        for i, boundaries in enumerate(turn_boundaries):
            num_turns = len(boundaries)
            if num_turns == 0:
                turn_advantages.append(torch.ones(1, device=self.device))
                continue
            
            # Backward pass to assign credit
            if self.config.turn_credit_method == "exponential":
                credits = torch.tensor([
                    self.gamma ** (num_turns - 1 - j)
                    for j in range(num_turns)
                ], device=self.device)
            elif self.config.turn_credit_method == "uniform":
                credits = torch.ones(num_turns, device=self.device)
            else:  # final_only
                credits = torch.zeros(num_turns, device=self.device)
                credits[-1] = 1.0
            
            # Normalize
            credits = credits / (credits.sum() + 1e-8)
            turn_advantages.append(credits)
        
        return turn_advantages
    
    def _apply_turn_weights(
        self,
        token_log_probs: torch.Tensor,
        turn_advantages: List[torch.Tensor],
        turn_boundaries: List[List[Tuple[int, int]]]
    ) -> torch.Tensor:
        """Apply turn-level weights to token log probabilities"""
        G, Seq = token_log_probs.shape
        weighted = torch.zeros_like(token_log_probs)
        
        for i, (boundaries, advantages) in enumerate(zip(turn_boundaries, turn_advantages)):
            for j, (start, end) in enumerate(boundaries):
                if j < len(advantages):
                    weighted[i, start:end] = token_log_probs[i, start:end] * advantages[j]
        
        return weighted
    
    def update_curriculum(self):
        """Update curriculum tier based on training progress"""
        if not self.config.use_curriculum:
            return
        
        self.steps_completed += 1
        
        threshold = self.tier_thresholds.get(self.current_tier, float('inf'))
        if self.steps_completed >= threshold:
            # Advance to next tier
            tier_order = list(TaskTier)
            current_idx = tier_order.index(self.current_tier)
            if current_idx < len(tier_order) - 1:
                self.current_tier = tier_order[current_idx + 1]
    
    def get_current_tier(self) -> TaskTier:
        """Get current curriculum tier"""
        return self.current_tier
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get training progress information"""
        threshold = self.tier_thresholds.get(self.current_tier, float('inf'))
        prev_threshold = 0
        
        tier_order = list(TaskTier)
        current_idx = tier_order.index(self.current_tier)
        if current_idx > 0:
            prev_tier = tier_order[current_idx - 1]
            prev_threshold = self.tier_thresholds.get(prev_tier, 0)
        
        steps_in_tier = self.steps_completed - prev_threshold
        tier_total = threshold - prev_threshold
        
        return {
            "current_tier": self.current_tier.name,
            "steps_completed": self.steps_completed,
            "tier_progress": steps_in_tier / tier_total if tier_total > 0 else 1.0,
            "steps_in_tier": steps_in_tier,
            "tier_total_steps": tier_total,
        }


# ============================================================================
# Tool Call Parser
# ============================================================================

class ToolCallParser:
    """Parse tool calls from model output text"""
    
    # Regex patterns for different tool call formats
    JSON_PATTERN = re.compile(r'```(?:json)?\s*(\{[^`]+\})\s*```', re.DOTALL)
    FUNCTION_PATTERN = re.compile(r'<tool_call>\s*(\{[^<]+\})\s*</tool_call>', re.DOTALL)
    
    @classmethod
    def parse(cls, text: str) -> List[ToolCall]:
        """Parse tool calls from text"""
        tool_calls = []
        
        # Try JSON code block format
        for match in cls.JSON_PATTERN.finditer(text):
            try:
                data = json.loads(match.group(1))
                if cls._is_tool_call(data):
                    tool_call = cls._parse_tool_call_dict(data)
                    if tool_call:
                        tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        
        # Try XML-style format
        for match in cls.FUNCTION_PATTERN.finditer(text):
            try:
                data = json.loads(match.group(1))
                tool_call = cls._parse_tool_call_dict(data)
                if tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        
        return tool_calls
    
    @classmethod
    def _is_tool_call(cls, data: Dict) -> bool:
        """Check if dict looks like a tool call"""
        return "function_name" in data or "name" in data or "tool_type" in data
    
    @classmethod
    def _parse_tool_call_dict(cls, data: Dict) -> Optional[ToolCall]:
        """Parse a dict into a ToolCall"""
        try:
            # Handle different field names
            tool_id = data.get("id", data.get("call_id", f"call_{hash(json.dumps(data)) % 10000}"))
            
            tool_type_str = data.get("tool_type", data.get("type", "REASONING"))
            try:
                tool_type = ToolType[tool_type_str.upper()]
            except KeyError:
                tool_type = ToolType.REASONING
            
            function_name = data.get("function_name", data.get("name", "unknown"))
            arguments = data.get("arguments", data.get("parameters", {}))
            
            return ToolCall(
                id=str(tool_id),
                tool_type=tool_type,
                function_name=function_name,
                arguments=arguments
            )
        except Exception:
            return None


# ============================================================================
# Group Sampler for Agent Tasks
# ============================================================================

class AgentGroupSampler:
    """Sample groups of trajectories for GRPO training"""
    
    def __init__(
        self,
        group_size: int = 8,
        task_generator: Optional[TaskGenerator] = None,
        environment: Optional[AgentEnvironment] = None
    ):
        self.group_size = group_size
        self.task_generator = task_generator or TaskGenerator()
        self.environment = environment or AgentEnvironment()
    
    def sample_group(
        self,
        prompt: str,
        model_fn: Callable[[str], str],
        tier: Optional[TaskTier] = None
    ) -> List[AgentTrajectory]:
        """Sample a group of trajectories for a given prompt
        
        Args:
            prompt: Task prompt
            model_fn: Function that takes prompt and returns model output
            tier: Task tier (if known)
        
        Returns:
            List of G trajectories
        """
        trajectories = []
        
        for _ in range(self.group_size):
            trajectory = self._sample_trajectory(prompt, model_fn, tier)
            trajectories.append(trajectory)
        
        return trajectories
    
    def _sample_trajectory(
        self,
        prompt: str,
        model_fn: Callable[[str], str],
        tier: Optional[TaskTier] = None
    ) -> AgentTrajectory:
        """Sample a single trajectory"""
        trajectory = AgentTrajectory(prompt=prompt, task_tier=tier)
        
        current_context = prompt
        max_steps = 20
        
        for step_idx in range(max_steps):
            # Get model output
            output = model_fn(current_context)
            
            # Parse for tool calls
            tool_calls = ToolCallParser.parse(output)
            
            if tool_calls:
                # Execute first tool call
                tool_call = tool_calls[0]
                response = self.environment.execute_tool_call(tool_call)
                
                step = AgentStep(
                    action=AgentAction(
                        action_type=AgentActionType.TOOL_CALL,
                        content=output,
                        tool_call=tool_call,
                        tool_response=response
                    )
                )
                trajectory.steps.append(step)
                
                # Update context with response
                current_context = f"{current_context}\n{output}\n\nTool Response:\n{response.content}"
            else:
                # Final response
                step = AgentStep(
                    action=AgentAction(
                        action_type=AgentActionType.RESPOND,
                        content=output
                    )
                )
                trajectory.steps.append(step)
                trajectory.final_output = output
                trajectory.task_completed = True
                break
        
        return trajectory


# ============================================================================
# Convenience Functions
# ============================================================================

def create_agent_trainer(
    beta: float = 0.04,
    gamma: float = 0.99,
    group_size: int = 8,
    device: str = "cpu"
) -> AgentGRPOTrainer:
    """Create an agent GRPO trainer with default settings"""
    config = AgentGRPOConfig(
        beta=beta,
        gamma=gamma,
        group_size=group_size,
        device=device
    )
    return AgentGRPOTrainer(config)


def compute_agent_reward(
    trajectory: AgentTrajectory,
    ground_truth: Optional[str] = None,
    weights: Optional[RewardWeights] = None
) -> float:
    """Compute reward for a single trajectory"""
    computer = AgentRewardComputer(weights=weights)
    breakdown = computer.compute_reward(trajectory, ground_truth)
    return breakdown.total
