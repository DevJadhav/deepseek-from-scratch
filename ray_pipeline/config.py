"""
Unified Pipeline Configuration

Provides model size presets, backend selection, and training hyperparameters
for the Ray-based DeepSeek training pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import os


class ModelSize(Enum):
    """Pre-defined model size configurations."""
    TINY = "tiny"         # ~10M params - for testing
    SMALL = "small"       # ~100M params - for prototyping
    MEDIUM = "medium"     # ~1B params - for research
    LARGE = "large"       # ~7B params - for production
    XLARGE = "xlarge"     # ~13B+ params - for advanced use


class Backend(Enum):
    """Training backend options."""
    PYTORCH_CUDA = "pytorch_cuda"
    PYTORCH_MPS = "pytorch_mps"
    PYTORCH_CPU = "pytorch_cpu"
    MLX = "mlx"
    RUST = "rust"
    MODAL_GPU = "modal_gpu"  # Cloud GPU training on Modal
    AUTO = "auto"  # Auto-detect best available


class Stage(Enum):
    """Pipeline stages."""
    DATA_PREP = "data_prep"
    PRETRAIN = "pretrain"
    SFT = "sft"
    GRPO = "grpo"
    DISTILLATION = "distillation"
    EXPORT = "export"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Core dimensions
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    vocab_size: int = 32000
    max_seq_len: int = 2048
    
    # MLA (Multi-head Latent Attention)
    d_latent: int = 128
    d_rope: int = 64
    
    # MoE (Mixture of Experts)
    num_experts: int = 16
    num_shared_experts: int = 1
    top_k: int = 2
    num_expert_groups: int = 4
    moe_hidden_mult: float = 2.0
    
    # MTP (Multi-Token Prediction)
    mtp_k: int = 1  # Number of future tokens to predict
    
    # Attention type
    use_sparse_attention: bool = False
    sparse_window_size: int = 4096
    sparse_global_tokens: int = 512
    
    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    @classmethod
    def tiny(cls) -> "ModelConfig":
        """Tiny config (~10M params) for testing."""
        return cls(
            d_model=256,
            num_heads=4,
            num_layers=4,
            vocab_size=8000,
            max_seq_len=512,
            d_latent=64,
            d_rope=32,
            num_experts=4,
            num_shared_experts=1,
            top_k=1,
            num_expert_groups=2,
            mtp_k=0,
        )
    
    @classmethod
    def small(cls) -> "ModelConfig":
        """Small config (~100M params) for prototyping."""
        return cls(
            d_model=512,
            num_heads=8,
            num_layers=8,
            vocab_size=32000,
            max_seq_len=2048,
            d_latent=128,
            d_rope=64,
            num_experts=16,
            num_shared_experts=1,
            top_k=2,
            num_expert_groups=4,
            mtp_k=1,
        )
    
    @classmethod
    def medium(cls) -> "ModelConfig":
        """Medium config (~1B params) for research."""
        return cls(
            d_model=2048,
            num_heads=16,
            num_layers=24,
            vocab_size=64000,
            max_seq_len=4096,
            d_latent=256,
            d_rope=128,
            num_experts=64,
            num_shared_experts=2,
            top_k=4,
            num_expert_groups=8,
            mtp_k=2,
        )
    
    @classmethod
    def large(cls) -> "ModelConfig":
        """Large config (~7B params) for production."""
        return cls(
            d_model=4096,
            num_heads=32,
            num_layers=32,
            vocab_size=102400,
            max_seq_len=8192,
            d_latent=512,
            d_rope=128,
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            num_expert_groups=8,
            mtp_k=3,
            use_sparse_attention=True,
        )
    
    @classmethod
    def from_size(cls, size: ModelSize) -> "ModelConfig":
        """Create config from ModelSize enum."""
        configs = {
            ModelSize.TINY: cls.tiny,
            ModelSize.SMALL: cls.small,
            ModelSize.MEDIUM: cls.medium,
            ModelSize.LARGE: cls.large,
            ModelSize.XLARGE: cls.large,  # Same as large for now
        }
        return configs[size]()
    
    def estimate_params(self) -> int:
        """Estimate total parameters (approximate)."""
        # Embedding
        embed_params = self.vocab_size * self.d_model
        
        # Per layer
        attn_params = 4 * self.d_model * self.d_model  # Q, K, V, O projections
        
        # MoE FFN
        expert_params = 2 * self.d_model * int(self.d_model * self.moe_hidden_mult)
        total_expert_params = (
            self.num_shared_experts * expert_params * 2 +  # Shared experts (larger)
            self.num_experts * expert_params  # Routed experts
        )
        
        layer_params = attn_params + total_expert_params
        
        # Total
        total = embed_params + self.num_layers * layer_params + embed_params  # +output head
        return int(total)


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    # Paths
    data_dir: str = "./data"
    domain_paths: Dict[str, str] = field(default_factory=lambda: {
        "codeforces": "open-r1/codeforces",
        "math": "open-r1/OpenThoughts-114k-math",
    })
    cache_dir: str = "./cache"
    
    # Tokenizer
    tokenizer_name: str = "deepseek-ai/deepseek-llm-7b-base"
    tokenizer_path: Optional[str] = None
    
    # Domain mixing weights
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        "web": 0.60,
        "code": 0.20,
        "math": 0.10,
        "books": 0.05,
        "scientific": 0.05,
    })
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_start_seq_len: int = 512
    curriculum_end_seq_len: int = 4096
    curriculum_warmup_steps: int = 10000
    curriculum_total_steps: int = 50000
    
    # Processing
    num_workers: int = 4
    prefetch_batches: int = 2
    shuffle_buffer_size: int = 10000


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimizer
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    
    # Schedule
    warmup_steps: int = 2000
    max_steps: int = 20000  # Production: 20k steps for time-sliced waves
    scheduler: str = "cosine"  # cosine, wsd (warmup-stable-decay)
    
    # WSD scheduler params
    wsd_stable_ratio: float = 0.8
    wsd_decay_ratio: float = 0.1
    
    # Batch
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # float16, bfloat16
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_steps: int = 1000  # 1k checkpoint intervals
    keep_last_n_checkpoints: int = 5  # Keep more for wave comparison
    
    # Logging
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 500
    
    # Gradient checkpointing (memory optimization)
    gradient_checkpointing: bool = False


@dataclass
class SFTConfig:
    """Supervised Fine-Tuning configuration."""
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # NEFTune
    use_neftune: bool = True
    neftune_alpha: float = 5.0
    
    # Training
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.03


@dataclass
class GRPOConfig:
    """GRPO (Group Relative Policy Optimization) configuration."""
    # GRPO params
    beta: float = 0.01  # KL penalty coefficient
    group_size: int = 4  # Number of samples per prompt
    
    # Reward model
    reward_model_path: Optional[str] = None
    use_rule_based_reward: bool = True
    
    # Training
    learning_rate: float = 1e-6
    num_iterations: int = 1000
    kl_target: float = 0.1


@dataclass 
class DistillationConfig:
    """Knowledge Distillation configuration."""
    # Teacher
    teacher_model_path: str = ""
    
    # Distillation params
    temperature: float = 4.0
    alpha: float = 0.5  # Weight for KD loss vs CE loss
    loss_type: str = "kl"  # kl, mse, jsd
    
    # Hidden distillation
    use_hidden_distillation: bool = False
    hidden_weight: float = 0.1
    
    # Progressive distillation
    use_progressive: bool = False
    num_stages: int = 3
    intermediate_sizes: List[int] = field(default_factory=lambda: [4096, 2048, 1024])


@dataclass
class ExportConfig:
    """Model export configuration."""
    output_dir: str = "./exports"
    
    # Formats
    export_safetensors: bool = True
    export_gguf: bool = False
    export_coreml: bool = False
    export_onnx: bool = False
    
    # Quantization for export
    quantize: bool = False
    quantization_bits: int = 8  # 4, 8


class WaveBackend(Enum):
    """Backend for each wave in time-sliced execution."""
    RUST = "rust"
    PYTHON = "python"
    MLX = "mlx"


@dataclass
class WaveConfig:
    """Configuration for a single time-sliced wave."""
    wave_id: int
    backend: WaveBackend
    start_step: int
    end_step: int
    stages: List[str] = field(default_factory=list)
    checkpoint_from: Optional[str] = None  # Previous wave checkpoint path
    
    @property
    def num_steps(self) -> int:
        return self.end_step - self.start_step


@dataclass
class TimeSlicedConfig:
    """Configuration for time-sliced wave execution across backends."""
    enabled: bool = False
    num_waves: int = 4
    steps_per_wave: int = 5000  # 20k / 4 waves = 5k per wave
    
    # GPU assignment (PP=3)
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1, 2])
    pipeline_parallel_size: int = 3
    
    # Validation
    validation_after_each_wave: bool = True
    validation_split: str = "valid"
    
    # Wave definitions (generated from defaults or explicitly configured)
    waves: List[WaveConfig] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate default wave configurations if not provided."""
        if not self.waves and self.enabled:
            self.waves = self._generate_default_waves()
    
    def _generate_default_waves(self) -> List[WaveConfig]:
        """Generate the 4 default waves alternating Rust/Python."""
        return [
            # Wave 1: Rust - Attention components
            WaveConfig(
                wave_id=1,
                backend=WaveBackend.RUST,
                start_step=0,
                end_step=self.steps_per_wave,
                stages=["MQA", "GQA", "MLA", "DeepSeek Attention"],
            ),
            # Wave 2: Python - MoE components
            WaveConfig(
                wave_id=2,
                backend=WaveBackend.PYTHON,
                start_step=self.steps_per_wave,
                end_step=self.steps_per_wave * 2,
                stages=["Standard MOE", "DeepSeek MOE"],
                checkpoint_from="checkpoints/step_5000/",
            ),
            # Wave 3: Rust - RL/Reward components
            WaveConfig(
                wave_id=3,
                backend=WaveBackend.RUST,
                start_step=self.steps_per_wave * 2,
                end_step=self.steps_per_wave * 3,
                stages=["GRPO", "R1", "DPO", "Reward"],
                checkpoint_from="checkpoints/step_10000/",
            ),
            # Wave 4: Python - Advanced training
            WaveConfig(
                wave_id=4,
                backend=WaveBackend.PYTHON,
                start_step=self.steps_per_wave * 3,
                end_step=self.steps_per_wave * 4,
                stages=["MTP", "FP8", "Distillation", "5D Parallelism"],
                checkpoint_from="checkpoints/step_15000/",
            ),
        ]


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    # Ray cluster
    num_workers: int = 3
    use_gpu: bool = True
    gpus_per_worker: int = 1
    cpus_per_worker: int = 4
    
    # Parallelism strategy (5D: DP x TP x PP x EP x SP)
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 3  # PP=3 for 3-GPU setup
    expert_parallel_size: int = 1
    sequence_parallel_size: int = 1
    
    # Communication
    backend: str = "nccl"  # nccl, gloo
    
    # Ray specific
    ray_address: Optional[str] = None  # None = local, "auto" = existing cluster
    runtime_env: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.
    
    This is the main configuration class that holds all settings
    for the entire training pipeline.
    """
    # Identifiers
    run_name: str = "deepseek-run"
    project_name: str = "deepseek-from-scratch"
    
    # Backend
    backend: Backend = Backend.AUTO
    
    # Model
    model_size: ModelSize = ModelSize.SMALL
    model: ModelConfig = field(default_factory=ModelConfig.small)
    
    # Stages
    stages_to_run: List[Stage] = field(default_factory=lambda: [
        Stage.DATA_PREP,
        Stage.PRETRAIN,
    ])
    
    # Stage configs
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    time_sliced: TimeSlicedConfig = field(default_factory=TimeSlicedConfig)
    
    # Monitoring
    use_wandb: bool = False
    wandb_project: str = "deepseek-training"
    wandb_entity: Optional[str] = None
    
    # Paths
    output_dir: str = "./outputs"
    
    def __post_init__(self):
        """Sync model config with model size."""
        if isinstance(self.model_size, str):
            self.model_size = ModelSize(self.model_size)
        if isinstance(self.backend, str):
            self.backend = Backend(self.backend)
        
        # Update model config based on size if using default (not if loaded from JSON)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        elif self.model.d_model == 768:  # Default value - apply size preset
            self.model = ModelConfig.from_size(self.model_size)
        
        # Ensure nested configs are proper dataclasses
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
    
    @classmethod
    def from_size(cls, size: ModelSize, backend: Backend = Backend.AUTO) -> "PipelineConfig":
        """Create pipeline config from model size."""
        return cls(
            model_size=size,
            backend=backend,
            model=ModelConfig.from_size(size),
        )
    
    @classmethod
    def tiny_test(cls) -> "PipelineConfig":
        """Tiny config for quick testing."""
        config = cls.from_size(ModelSize.TINY)
        config.training.max_steps = 100
        config.training.save_every_n_steps = 50
        config.training.log_every_n_steps = 1
        return config
    
    @classmethod
    def small_prototype(cls) -> "PipelineConfig":
        """Small config for prototyping."""
        config = cls.from_size(ModelSize.SMALL)
        config.training.max_steps = 10000
        return config
    
    @classmethod
    def medium_research(cls) -> "PipelineConfig":
        """Medium config for research experiments."""
        config = cls.from_size(ModelSize.MEDIUM)
        config.training.max_steps = 50000
        config.distributed.num_workers = 4
        return config
    
    @classmethod
    def large_production(cls) -> "PipelineConfig":
        """Large config for production training."""
        config = cls.from_size(ModelSize.LARGE)
        config.training.max_steps = 200000
        config.distributed.num_workers = 8
        config.training.gradient_checkpointing = True
        return config
    
    @classmethod
    def production_3gpu_time_sliced(cls) -> "PipelineConfig":
        """
        Production config for 3-GPU time-sliced pipeline with PP=3.
        
        This configuration sets up:
        - 3 GPUs with pipeline_parallel_size=3
        - 20k total steps across 4 waves (5k each)
        - Alternating Rust/Python backends
        - 1k checkpoint intervals for wave comparison
        - Best model selection at completion
        
        Usage:
            config = PipelineConfig.production_3gpu_time_sliced()
            workflow = DeepSeekWorkflow(config)
            result = workflow.run_time_sliced_waves()
        """
        config = cls.from_size(ModelSize.MEDIUM)
        
        # Training config
        config.training.max_steps = 20000
        config.training.save_every_n_steps = 1000
        config.training.keep_last_n_checkpoints = 20  # Keep all for comparison
        config.training.gradient_checkpointing = True
        
        # Distributed config - 3 GPUs with PP=3
        config.distributed.num_workers = 3
        config.distributed.gpus_per_worker = 1
        config.distributed.pipeline_parallel_size = 3
        config.distributed.data_parallel_size = 1
        config.distributed.tensor_parallel_size = 1
        
        # Time-sliced wave config
        config.time_sliced = TimeSlicedConfig(
            enabled=True,
            num_waves=4,
            steps_per_wave=5000,
            gpu_ids=[0, 1, 2],
            pipeline_parallel_size=3,
            validation_after_each_wave=True,
        )
        
        # Data config with domain paths
        config.data.domain_paths = {
            "codeforces": "open-r1/codeforces",
            "math": "open-r1/OpenThoughts-114k-math",
        }
        
        return config
    
    def detect_backend(self) -> Backend:
        """Auto-detect the best available backend."""
        if self.backend != Backend.AUTO:
            return self.backend
        
        # Try CUDA first
        try:
            import torch
            if torch.cuda.is_available():
                return Backend.PYTORCH_CUDA
        except ImportError:
            pass
        
        # Try MPS (Apple Silicon)
        try:
            import torch
            if torch.backends.mps.is_available():
                return Backend.PYTORCH_MPS
        except (ImportError, AttributeError):
            pass
        
        # Try MLX
        try:
            import mlx.core as mx
            return Backend.MLX
        except ImportError:
            pass
        
        # Fall back to CPU
        return Backend.PYTORCH_CPU
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        
        def convert(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj
        
        result = asdict(self)
        # Convert enums to values
        result['backend'] = self.backend.value
        result['model_size'] = self.model_size.value
        result['stages_to_run'] = [s.value for s in self.stages_to_run]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary."""
        data = data.copy()
        
        # Convert enums
        if 'backend' in data:
            data['backend'] = Backend(data['backend']) if isinstance(data['backend'], str) else data['backend']
        if 'model_size' in data:
            data['model_size'] = ModelSize(data['model_size']) if isinstance(data['model_size'], str) else data['model_size']
        if 'stages_to_run' in data:
            data['stages_to_run'] = [Stage(s) if isinstance(s, str) else s for s in data['stages_to_run']]
        
        # Reconstruct nested dataclasses
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelConfig(**data['model'])
        if 'data' in data and isinstance(data['data'], dict):
            data['data'] = DataConfig(**data['data'])
        if 'training' in data and isinstance(data['training'], dict):
            data['training'] = TrainingConfig(**data['training'])
        if 'sft' in data and isinstance(data['sft'], dict):
            data['sft'] = SFTConfig(**data['sft'])
        if 'grpo' in data and isinstance(data['grpo'], dict):
            data['grpo'] = GRPOConfig(**data['grpo'])
        if 'distillation' in data and isinstance(data['distillation'], dict):
            data['distillation'] = DistillationConfig(**data['distillation'])
        if 'export' in data and isinstance(data['export'], dict):
            data['export'] = ExportConfig(**data['export'])
        if 'distributed' in data and isinstance(data['distributed'], dict):
            data['distributed'] = DistributedConfig(**data['distributed'])
        if 'time_sliced' in data and isinstance(data['time_sliced'], dict):
            ts_data = data['time_sliced'].copy()
            if 'waves' in ts_data and isinstance(ts_data['waves'], list):
                ts_data['waves'] = [
                    WaveConfig(
                        wave_id=w['wave_id'],
                        backend=WaveBackend(w['backend']) if isinstance(w['backend'], str) else w['backend'],
                        start_step=w['start_step'],
                        end_step=w['end_step'],
                        stages=w.get('stages', []),
                        checkpoint_from=w.get('checkpoint_from'),
                    ) for w in ts_data['waves']
                ]
            data['time_sliced'] = TimeSlicedConfig(**ts_data)
        
        # Create instance without calling __post_init__ side effects
        config = object.__new__(cls)
        for key, value in data.items():
            setattr(config, key, value)
        return config
    
    def save(self, path: str):
        """Save config to JSON file."""
        import json
        
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, Path):
                return str(obj)
            return obj
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=serialize)
    
    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert enums
        if 'backend' in data:
            data['backend'] = Backend(data['backend'])
        if 'model_size' in data:
            data['model_size'] = ModelSize(data['model_size'])
        if 'stages_to_run' in data:
            data['stages_to_run'] = [Stage(s) for s in data['stages_to_run']]
        
        # Convert nested configs to their dataclass types
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelConfig(**data['model'])
        if 'data' in data and isinstance(data['data'], dict):
            data['data'] = DataConfig(**data['data'])
        if 'training' in data and isinstance(data['training'], dict):
            data['training'] = TrainingConfig(**data['training'])
        if 'sft' in data and isinstance(data['sft'], dict):
            data['sft'] = SFTConfig(**data['sft'])
        if 'grpo' in data and isinstance(data['grpo'], dict):
            data['grpo'] = GRPOConfig(**data['grpo'])
        if 'distillation' in data and isinstance(data['distillation'], dict):
            data['distillation'] = DistillationConfig(**data['distillation'])
        if 'export' in data and isinstance(data['export'], dict):
            data['export'] = ExportConfig(**data['export'])
        if 'distributed' in data and isinstance(data['distributed'], dict):
            data['distributed'] = DistributedConfig(**data['distributed'])
        if 'time_sliced' in data and isinstance(data['time_sliced'], dict):
            ts_data = data['time_sliced'].copy()
            if 'waves' in ts_data and isinstance(ts_data['waves'], list):
                ts_data['waves'] = [
                    WaveConfig(
                        wave_id=w['wave_id'],
                        backend=WaveBackend(w['backend']) if isinstance(w['backend'], str) else w['backend'],
                        start_step=w['start_step'],
                        end_step=w['end_step'],
                        stages=w.get('stages', []),
                        checkpoint_from=w.get('checkpoint_from'),
                    ) for w in ts_data['waves']
                ]
            data['time_sliced'] = TimeSlicedConfig(**ts_data)
        
        return cls(**data)
    
    def summary(self) -> str:
        """Return a human-readable summary of the config."""
        lines = [
            f"═══════════════════════════════════════════════════════════",
            f"  DeepSeek Training Pipeline Configuration",
            f"═══════════════════════════════════════════════════════════",
            f"",
            f"  Run Name:     {self.run_name}",
            f"  Backend:      {self.detect_backend().value}",
            f"  Model Size:   {self.model_size.value}",
            f"",
            f"  Model Architecture:",
            f"    - d_model:      {self.model.d_model}",
            f"    - num_heads:    {self.model.num_heads}",
            f"    - num_layers:   {self.model.num_layers}",
            f"    - num_experts:  {self.model.num_experts}",
            f"    - top_k:        {self.model.top_k}",
            f"    - vocab_size:   {self.model.vocab_size}",
            f"    - max_seq_len:  {self.model.max_seq_len}",
            f"    - Est. params:  {self.model.estimate_params() / 1e6:.1f}M",
            f"",
            f"  Training:",
            f"    - batch_size:   {self.training.batch_size}",
            f"    - grad_accum:   {self.training.gradient_accumulation_steps}",
            f"    - learning_rate:{self.training.learning_rate}",
            f"    - max_steps:    {self.training.max_steps}",
            f"    - warmup_steps: {self.training.warmup_steps}",
            f"",
            f"  Distributed:",
            f"    - num_workers:  {self.distributed.num_workers}",
            f"    - gpus/worker:  {self.distributed.gpus_per_worker}",
            f"",
            f"  Stages: {[s.value for s in self.stages_to_run]}",
            f"═══════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)
