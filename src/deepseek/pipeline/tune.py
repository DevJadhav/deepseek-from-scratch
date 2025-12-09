"""
Ray Tune Hyperparameter Optimization Module
===========================================

This module provides hyperparameter tuning integration using Ray Tune with
Hydra config composition. Supports ASHA (default) and PBT schedulers across
PyTorch+GPU, MLX, and Rust backends.

Design Principles:
- Hydra handles config composition only (no multirun)
- Ray Tune handles trial parallelization and scheduling
- Backend-specific metric names for multi-backend optimization
- Integration with existing checkpoint structure

Usage:
    # Run ASHA tuning
    uv run python -m deepseek.pipeline.tune experiment=tune_asha
    
    # Run PBT tuning
    uv run python -m deepseek.pipeline.tune experiment=tune_pbt
    
    # Override settings
    uv run python -m deepseek.pipeline.tune experiment=tune_asha tune.num_samples=50

Architecture:
    HyperparameterSearch
    ├── build_search_space()    # Convert Hydra config to Ray Tune search space
    ├── get_scheduler()         # Return ASHA or PBT scheduler
    ├── create_trainable()      # Create backend-specific trainable function
    └── run()                   # Execute tuning with Ray Tune
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ray
from ray import tune
from ray.air.config import RunConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.schedulers.trial_scheduler import TrialScheduler

# Hydra imports for config composition
try:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict  # type: ignore[misc]

LOGGER = logging.getLogger(__name__)


class HyperparameterSearch:
    """
    Ray Tune integration for hyperparameter optimization.
    
    Uses Hydra for config composition and Ray Tune for trial parallelization.
    Supports ASHA (fast early stopping) and PBT (adaptive mutation) schedulers.
    
    Attributes:
        config: Hydra-composed configuration
        backend: Target backend (pytorch, mlx, rust)
        storage_path: Checkpoint directory from Hydra config
        
    Example:
        >>> search = HyperparameterSearch.from_hydra("experiment=tune_asha")
        >>> results = search.run(backend="pytorch")
        >>> best_config = results.get_best_result().config
    """
    
    def __init__(
        self,
        config: DictConfig | dict[str, Any],
        config_dir: Path | None = None,
    ):
        """
        Initialize hyperparameter search.
        
        Args:
            config: Hydra configuration (OmegaConf DictConfig or dict)
            config_dir: Path to Hydra config directory
        """
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        
        self.config = config
        self.config_dir = config_dir or Path(__file__).parents[4] / "config" / "hydra"
        
        # Extract tune config - check both root level and nested in experiment
        # When using experiment=tune_asha, Hydra nests the experiment config under 'experiment' key
        self.tune_config = config.get("tune", {})
        if not self.tune_config:
            # Check if tune is nested in experiment config
            exp_cfg = config.get("experiment", {})
            if isinstance(exp_cfg, (dict, DictConfig)) and exp_cfg.get("tune"):
                self.tune_config = exp_cfg.get("tune", {})
        
        if not self.tune_config.get("enabled", False):
            LOGGER.warning("Tune is not enabled in config. Set tune.enabled=true")
        
        # Determine storage path from Hydra checkpoint config or tune config
        # First check if checkpoint config has output_dir
        checkpoint_cfg = config.get("checkpoint", {})
        if checkpoint_cfg.get("output_dir"):
            storage_dir = checkpoint_cfg.get("output_dir")
        else:
            # Fall back to paths + tune subdir
            paths_cfg = config.get("paths", {})
            base_checkpoint_dir = paths_cfg.get("checkpoint_dir", "./checkpoints")
            tune_subdir = self.tune_config.get("checkpoint_subdir", "tune")
            storage_dir = f"{base_checkpoint_dir}/{tune_subdir}"
        
        # Make sure it's an absolute path (Ray Tune requires this)
        self.storage_path = Path(storage_dir).resolve()
        
    @classmethod
    def from_hydra(
        cls,
        overrides: list[str] | None = None,
        config_name: str = "config",
        config_dir: Path | None = None,
    ) -> HyperparameterSearch:
        """
        Create search from Hydra config composition.
        
        Args:
            overrides: Hydra CLI overrides (e.g., ["experiment=tune_asha"])
            config_name: Name of main config file
            config_dir: Path to config directory
            
        Returns:
            Initialized HyperparameterSearch instance
        """
        if not HYDRA_AVAILABLE:
            raise ImportError("Hydra is required for config composition. Install with: pip install hydra-core")
        
        config_dir = config_dir or Path(__file__).parents[4] / "config" / "hydra"
        
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Initialize Hydra and compose config
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name=config_name, overrides=overrides or [])
        
        return cls(config=cfg, config_dir=config_dir)
    
    def build_search_space(self) -> dict[str, Any]:
        """
        Convert Hydra search space config to Ray Tune search space.
        
        The search space is defined in tune.search_space with:
        - *_choices: list of discrete values -> tune.choice()
        - *_min/*_max: range values -> tune.loguniform() for lr, tune.uniform() otherwise
        
        Returns:
            Dict of parameter names to Ray Tune samplers
        """
        search_space = {}
        ss_config = self.tune_config.get("search_space", {})
        
        # Learning rate (log-uniform)
        if ss_config.get("learning_rate_choices"):
            search_space["learning_rate"] = tune.choice(
                list(ss_config["learning_rate_choices"])
            )
        else:
            search_space["learning_rate"] = tune.loguniform(
                ss_config.get("learning_rate_min", 1e-5),
                ss_config.get("learning_rate_max", 1e-3),
            )
        
        # Batch size (discrete choices)
        batch_choices = ss_config.get("batch_size_choices", [8, 16, 32, 64])
        search_space["batch_size"] = tune.choice(list(batch_choices))
        
        # Warmup steps (uniform int)
        search_space["warmup_steps"] = tune.randint(
            ss_config.get("warmup_steps_min", 100),
            ss_config.get("warmup_steps_max", 2000),
        )
        
        # Weight decay (log-uniform)
        search_space["weight_decay"] = tune.loguniform(
            ss_config.get("weight_decay_min", 0.001),
            ss_config.get("weight_decay_max", 0.1),
        )
        
        # MoE capacity factor (uniform)
        search_space["moe_capacity_factor"] = tune.uniform(
            ss_config.get("moe_capacity_factor_min", 1.0),
            ss_config.get("moe_capacity_factor_max", 2.0),
        )
        
        # GRPO beta (log-uniform, for alignment stage)
        search_space["grpo_beta"] = tune.loguniform(
            ss_config.get("grpo_beta_min", 0.01),
            ss_config.get("grpo_beta_max", 0.5),
        )
        
        return search_space
    
    def get_scheduler(self, backend: str | None = None) -> TrialScheduler:
        """
        Get Ray Tune scheduler based on config.
        
        Returns ASHA for fast early stopping (default) or PBT for adaptive mutation.
        
        Args:
            backend: Target backend for metric name resolution
            
        Returns:
            Configured TrialScheduler instance
        """
        scheduler_type = self.tune_config.get("scheduler_type", "asha").lower()
        mode = self.tune_config.get("mode", "min")
        metric_name = self._get_metric_name(backend)
        
        if scheduler_type == "pbt":
            # Population Based Training
            # Always use default mutations for PBT since Hydra config objects
            # need special conversion and may not be valid tune search spaces
            hyperparam_mutations = {
                "learning_rate": tune.loguniform(1e-5, 1e-3),
                "weight_decay": tune.loguniform(0.001, 0.1),
            }
            
            return PopulationBasedTraining(
                time_attr="training_iteration",
                metric=metric_name,
                mode=mode,
                perturbation_interval=self.tune_config.get("perturbation_interval", 100),
                hyperparam_mutations=hyperparam_mutations,
                # Quantiles for exploit (top 25% exploit from bottom 25%)
                quantile_fraction=0.25,
                resample_probability=0.25,
                # Synch to ensure checkpoints available
                synch=True,
            )
        else:
            # ASHA (default) - Async Successive Halving
            return ASHAScheduler(
                time_attr="training_iteration",
                metric=metric_name,
                mode=mode,
                max_t=self.tune_config.get("max_t", 10000),
                grace_period=self.tune_config.get("grace_period", 100),
                reduction_factor=self.tune_config.get("reduction_factor", 4),
            )
    
    def _get_metric_name(self, backend: str | None = None) -> str:
        """Get backend-specific metric name."""
        backend = backend or self._detect_backend()
        
        if backend in ("pytorch", "torch", "pytorch_mps", "pytorch_cuda"):
            return self.tune_config.get("torch_metric", "torch_val_loss")
        elif backend == "mlx":
            return self.tune_config.get("mlx_metric", "mlx_val_loss")
        elif backend == "rust":
            return self.tune_config.get("rust_metric", "rust_val_loss")
        else:
            return "val_loss"
    
    def _detect_backend(self) -> str:
        """Detect available backend."""
        import importlib.util

        # Check for MLX (Apple Silicon)
        if importlib.util.find_spec("mlx") is not None:
            return "mlx"

        # Check for PyTorch with CUDA
        if importlib.util.find_spec("torch") is not None:
            import torch
            if torch.cuda.is_available() or torch.backends.mps.is_available():
                return "pytorch"

        # Default to rust if available
        if importlib.util.find_spec("deepseek_rust") is not None:
            return "rust"

        return "pytorch"  # Fallback
    
    @staticmethod
    def _filter_model_config(model_cfg: dict) -> dict:
        """
        Filter model config to only include fields accepted by ModelConfig dataclass.
        
        The Hydra config has extra nested sections (moe, mla, attention) that need
        to be flattened or removed to match the dataclass structure.
        """
        # Fields that ModelConfig dataclass accepts
        valid_fields = {
            "d_model", "num_heads", "num_layers", "vocab_size", "max_seq_len",
            "d_latent", "d_rope", "num_experts", "num_shared_experts", "top_k",
            "num_expert_groups", "moe_hidden_mult", "mtp_k", "use_sparse_attention",
            "sparse_window_size", "sparse_global_tokens", "dropout", "attention_dropout",
        }
        
        filtered = {}
        for key, value in model_cfg.items():
            if key in valid_fields and not isinstance(value, dict):
                filtered[key] = value
        
        # Extract top_k from nested moe config if present
        if "moe" in model_cfg and isinstance(model_cfg["moe"], dict):
            moe_cfg = model_cfg["moe"]
            if "num_experts" in moe_cfg:
                filtered["num_experts"] = moe_cfg["num_experts"]
            if "num_shared_experts" in moe_cfg:
                filtered["num_shared_experts"] = moe_cfg["num_shared_experts"]
            if "top_k" in moe_cfg:
                filtered["top_k"] = moe_cfg["top_k"]
        
        return filtered
    
    @staticmethod
    def _filter_data_config(data_cfg: dict) -> dict:
        """Filter data config to only include fields accepted by DataConfig dataclass."""
        valid_fields = {
            "data_dir", "domain_paths", "cache_dir", "tokenizer_name", "tokenizer_path",
            "domain_weights", "use_curriculum", "curriculum_start_seq_len",
            "curriculum_end_seq_len", "curriculum_warmup_steps", "curriculum_total_steps",
            "num_workers", "prefetch_batches", "shuffle_buffer_size",
        }
        
        return {k: v for k, v in data_cfg.items() if k in valid_fields}
    
    @staticmethod
    def _filter_training_config(training_cfg: dict) -> dict:
        """Filter training config to only include fields accepted by TrainingConfig dataclass."""
        valid_fields = {
            "learning_rate", "min_learning_rate", "weight_decay", "beta1", "beta2",
            "max_grad_norm", "warmup_steps", "max_steps", "scheduler", "wsd_stable_ratio",
            "batch_size", "gradient_accumulation_steps", "eval_interval",
            "save_interval", "log_interval", "label_smoothing", "use_amp",
            "use_gradient_checkpointing", "use_flash_attention",
        }
        
        return {k: v for k, v in training_cfg.items() if k in valid_fields}
    
    def create_trainable(
        self,
        backend: str,
        stage: str = "pretrain",
    ) -> Callable:
        """
        Create a Ray Tune trainable function for the specified backend.
        
        Args:
            backend: Target backend ("pytorch", "pytorch_mps", "mlx", "rust")
            stage: Training stage (pretrain, sft, grpo)
            
        Returns:
            Trainable function compatible with Ray Tune
        """
        base_config = OmegaConf.to_container(self.config, resolve=True)
        metric_name = self._get_metric_name(backend)
        
        if backend in ("pytorch", "torch", "pytorch_mps", "pytorch_cuda"):
            return self._create_pytorch_trainable(base_config, stage, metric_name)
        elif backend == "mlx":
            return self._create_mlx_trainable(base_config, stage, metric_name)
        elif backend == "rust":
            return self._create_rust_trainable(base_config, stage, metric_name)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _create_pytorch_trainable(
        self,
        base_config: dict,
        stage: str,
        metric_name: str,
    ) -> Callable:
        """Create PyTorch trainable with tune.report() integration."""
        
        def pytorch_trainable(config: dict):
            """PyTorch training function for Ray Tune."""
            import copy
            from ray import tune

            from deepseek.pipeline.config import PipelineConfig
            from deepseek.pipeline.runners.pytorch_runner import PyTorchRunner

            # Deep copy to avoid modifying base config
            merged_config = copy.deepcopy(base_config)
            
            # Merge tune params into training config
            if "training" not in merged_config:
                merged_config["training"] = {}
            merged_config["training"]["learning_rate"] = config["learning_rate"]
            merged_config["training"]["batch_size"] = config["batch_size"]
            merged_config["training"]["warmup_steps"] = config["warmup_steps"]
            merged_config["training"]["weight_decay"] = config["weight_decay"]
            
            # Filter all config sections to only include valid dataclass fields
            if "model" in merged_config:
                merged_config["model"] = HyperparameterSearch._filter_model_config(
                    merged_config["model"]
                )
            if "data" in merged_config:
                merged_config["data"] = HyperparameterSearch._filter_data_config(
                    merged_config["data"]
                )
            if "training" in merged_config:
                merged_config["training"] = HyperparameterSearch._filter_training_config(
                    merged_config["training"]
                )

            # Create runner
            try:
                pipeline_config = PipelineConfig.from_dict(merged_config)
            except TypeError as e:
                LOGGER.warning(f"Config parsing warning: {e}, using defaults")
                pipeline_config = PipelineConfig()
                
            runner = PyTorchRunner(pipeline_config, stage=stage)
            
            # Run training and report metrics
            try:
                result = runner.run(
                    dataset_uri=merged_config.get("data", {}).get("dataset_uri", ""),
                    pad_token_id=merged_config.get("data", {}).get("pad_token_id", 0),
                    training_config=merged_config.get("training", {}),
                )
                
                tune.report({
                    metric_name: result.metrics.get("loss", float("inf")),
                    "training_iteration": result.metrics.get("step", 0),
                })
                
            except Exception as e:
                LOGGER.error(f"Training failed: {e}")
                tune.report({metric_name: float("inf"), "training_iteration": 0})
        
        return pytorch_trainable
    
    def _create_mlx_trainable(
        self,
        base_config: dict,
        stage: str,
        metric_name: str,
    ) -> Callable:
        """Create MLX trainable with tune.report() integration."""
        
        def mlx_trainable(config: dict):
            """MLX training function for Ray Tune."""
            import copy
            from ray import tune

            from deepseek.pipeline.config import PipelineConfig
            from deepseek.pipeline.runners.mlx_runner import MLXRunner

            # Deep copy to avoid modifying base config
            merged_config = copy.deepcopy(base_config)
            
            # Merge tune params into training config
            if "training" not in merged_config:
                merged_config["training"] = {}
            merged_config["training"]["learning_rate"] = config["learning_rate"]
            merged_config["training"]["batch_size"] = config["batch_size"]
            merged_config["training"]["warmup_steps"] = config["warmup_steps"]
            merged_config["training"]["weight_decay"] = config["weight_decay"]
            
            # Filter all config sections to only include valid dataclass fields
            if "model" in merged_config:
                merged_config["model"] = HyperparameterSearch._filter_model_config(
                    merged_config["model"]
                )
            if "data" in merged_config:
                merged_config["data"] = HyperparameterSearch._filter_data_config(
                    merged_config["data"]
                )
            if "training" in merged_config:
                merged_config["training"] = HyperparameterSearch._filter_training_config(
                    merged_config["training"]
                )

            try:
                pipeline_config = PipelineConfig.from_dict(merged_config)
            except TypeError as e:
                LOGGER.warning(f"Config parsing warning: {e}, using defaults")
                pipeline_config = PipelineConfig()
                
            runner = MLXRunner(pipeline_config, stage=stage)
            
            try:
                result = runner.run(
                    dataset_uri=merged_config.get("data", {}).get("dataset_uri"),
                    pad_token_id=merged_config.get("data", {}).get("pad_token_id", 0),
                    training_config=merged_config.get("training", {}),
                )
                
                tune.report({
                    metric_name: result.metrics.get("loss", float("inf")),
                    "training_iteration": result.metrics.get("step", 0),
                })
                
            except Exception as e:
                LOGGER.error(f"MLX training failed: {e}")
                tune.report({metric_name: float("inf"), "training_iteration": 0})
        
        return mlx_trainable
    
    def _create_rust_trainable(
        self,
        base_config: dict,
        stage: str,
        metric_name: str,
    ) -> Callable:
        """Create Rust trainable that parses JSON metrics from stdout."""
        
        def rust_trainable(config: dict):
            """Rust training function for Ray Tune with JSON metric parsing."""
            import copy
            from ray import tune

            from deepseek.pipeline.config import PipelineConfig
            from deepseek.pipeline.runners.rust_runner import RustRunner

            # Deep copy to avoid modifying base config
            merged_config = copy.deepcopy(base_config)
            
            # Merge tune params into training config
            if "training" not in merged_config:
                merged_config["training"] = {}
            merged_config["training"]["learning_rate"] = config["learning_rate"]
            merged_config["training"]["batch_size"] = config["batch_size"]
            merged_config["training"]["warmup_steps"] = config["warmup_steps"]
            merged_config["training"]["weight_decay"] = config["weight_decay"]
            
            # Filter all config sections to only include valid dataclass fields
            if "model" in merged_config:
                merged_config["model"] = HyperparameterSearch._filter_model_config(
                    merged_config["model"]
                )
            if "data" in merged_config:
                merged_config["data"] = HyperparameterSearch._filter_data_config(
                    merged_config["data"]
                )
            if "training" in merged_config:
                merged_config["training"] = HyperparameterSearch._filter_training_config(
                    merged_config["training"]
                )

            try:
                pipeline_config = PipelineConfig.from_dict(merged_config)
            except TypeError as e:
                LOGGER.warning(f"Config parsing warning: {e}, using defaults")
                pipeline_config = PipelineConfig()
                
            runner = RustRunner(pipeline_config, stage=stage)
            
            try:
                result = runner.run(
                    dataset_uri=merged_config.get("data", {}).get("dataset_uri"),
                    pad_token_id=merged_config.get("data", {}).get("pad_token_id", 0),
                    training_config=merged_config.get("training", {}),
                )
                
                # Parse metrics from result or stdout
                loss = result.metrics.get("loss", float("inf"))
                step = result.metrics.get("step", 0)
                
                tune.report({
                    metric_name: loss,
                    "training_iteration": step,
                })
                
            except Exception as e:
                LOGGER.error(f"Rust training failed: {e}")
                tune.report({metric_name: float("inf"), "training_iteration": 0})
        
        return rust_trainable
    
    def run(
        self,
        backend: str | None = None,
        stage: str = "pretrain",
        num_samples: int | None = None,
        resources_per_trial: dict | None = None,
    ) -> tune.ResultGrid:
        """
        Execute hyperparameter tuning.
        
        Args:
            backend: Target backend (auto-detected if None)
            stage: Training stage
            num_samples: Override number of trials
            resources_per_trial: Override resource allocation
            
        Returns:
            Ray Tune ResultGrid with all trial results
        """
        import sys
        
        backend = backend or self._detect_backend()
        num_samples = num_samples or self.tune_config.get("num_samples", 10)
        
        # Get current working directory for runtime env
        working_dir = Path.cwd().resolve()
        
        # Initialize Ray - let workers inherit the parent environment
        # Don't use py_modules or working_dir as it can cause packaging issues
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Build search space and scheduler
        search_space = self.build_search_space()
        scheduler = self.get_scheduler(backend)
        
        # Determine resources per trial
        # Note: MPS (Metal) doesn't expose as CUDA GPU to Ray, so use CPU only for pytorch_mps
        if resources_per_trial is None:
            if backend == "pytorch_mps":
                # MPS uses Metal acceleration, not CUDA GPUs visible to Ray
                resources_per_trial = {"cpu": 4}
            elif backend in ("pytorch", "torch", "pytorch_cuda"):
                resources_per_trial = {"cpu": 2, "gpu": 1}
            elif backend == "mlx":
                resources_per_trial = {"cpu": 4}  # MLX uses Metal, not CUDA
            else:
                resources_per_trial = {"cpu": 4}
        
        # Create storage path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Configure checkpointing for PBT
        checkpoint_config = CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute=self._get_metric_name(backend),
            checkpoint_score_order="min" if self.tune_config.get("mode", "min") == "min" else "max",
        )
        
        # Experiment name from config
        experiment_name = self.config.get("experiment", {}).get("name", "deepseek-tune")
        
        # Create tuner
        # Note: metric and mode are already set in the scheduler,
        # so we don't pass them to TuneConfig to avoid conflict
        tuner = tune.Tuner(
            tune.with_resources(
                self.create_trainable(backend, stage),
                resources=resources_per_trial,
            ),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=num_samples,
                max_concurrent_trials=self.tune_config.get("max_concurrent_trials", 4),
            ),
            run_config=RunConfig(
                name=experiment_name,
                storage_path=str(self.storage_path),
                checkpoint_config=checkpoint_config,
                verbose=1,
            ),
        )
        
        LOGGER.info(
            "Starting hyperparameter search: backend=%s, scheduler=%s, samples=%d",
            backend,
            self.tune_config.get("scheduler_type", "asha"),
            num_samples,
        )
        
        results = tuner.fit()
        
        # Log best result
        best_result = results.get_best_result()
        if best_result:
            LOGGER.info("Best trial config: %s", best_result.config)
            LOGGER.info("Best trial metric: %s", best_result.metrics)
        
        return results


def run_hyperparameter_search(
    overrides: list[str] | None = None,
    backend: str | None = None,
    stage: str = "pretrain",
) -> tune.ResultGrid:
    """
    Convenience function to run hyperparameter search.
    
    Args:
        overrides: Hydra config overrides (e.g., ["experiment=tune_asha"])
        backend: Target backend (auto-detected if None)
        stage: Training stage
        
    Returns:
        Ray Tune ResultGrid
    """
    search = HyperparameterSearch.from_hydra(overrides=overrides)
    return search.run(backend=backend, stage=stage)


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ray Tune Hyperparameter Search")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["pytorch", "mlx", "rust"],
        default=None,
        help="Target backend (auto-detected if not specified)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="pretrain",
        help="Training stage (pretrain, sft, grpo)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra config overrides (e.g., experiment=tune_asha tune.num_samples=50)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run search
    results = run_hyperparameter_search(
        overrides=args.overrides,
        backend=args.backend,
        stage=args.stage,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Hyperparameter Search Complete")
    print("=" * 60)
    
    best = results.get_best_result()
    if best:
        print(f"\nBest Config: {best.config}")
        print(f"Best Metrics: {best.metrics}")
        print(f"Best Checkpoint: {best.checkpoint}")
