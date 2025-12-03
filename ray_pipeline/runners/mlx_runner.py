"""
MLX Runner for Ray Pipeline
============================

This runner integrates with the actual MLX implementation from:

- ``mlx_impl/attention.py`` - MultiQueryAttention, GroupedQueryAttention, MultiHeadLatentAttention
- ``mlx_impl/moe.py`` - DeepSeekMoE, DeepSeekMoEV3
- ``mlx_impl/mtp.py`` - MTPModel (Multi-Token Prediction)
- ``mlx_impl/grpo.py`` - GRPOTrainer (Group Relative Policy Optimization)
- ``mlx_impl/sft.py`` - SFTTrainer, SFTConfig
- ``mlx_impl/distillation.py`` - compute_distillation_loss
- ``mlx_impl/pipeline.py`` - ScalingLaws, DataMixer, CurriculumScheduler

Usage
-----
The runner is invoked by pipeline stages when backend is MLX::

    from ray_pipeline.runners import MLXRunner
    from ray_pipeline.config import PipelineConfig, Backend

    config = PipelineConfig.from_size(ModelSize.SMALL)
    config.backend = Backend.MLX
    runner = MLXRunner(config, stage="pretrain")
    result = runner.run(dataset_uri="./data/train", pad_token_id=0)

Backend Requirements
--------------------
- Apple Silicon Mac (M1/M2/M3)
- MLX framework installed (``pip install mlx``)

Implementation Details
----------------------
Unlike PyTorch, MLX doesn't integrate with Ray Train directly. This runner:

1. Imports actual MLX modules from ``mlx_impl/``
2. Creates models using MLX's nn.Module classes
3. Runs training loop natively on Apple Silicon GPU
4. Reports results back to the pipeline

Imported Modules from Implementation
------------------------------------
- ``mlx_impl.attention.MultiHeadLatentAttention``
- ``mlx_impl.moe.DeepSeekMoE``
- ``mlx_impl.mtp.MTPModel``
- ``mlx_impl.grpo.GRPOTrainer``
- ``mlx_impl.sft.SFTTrainer``
- ``mlx_impl.pipeline.ScalingLaws, DataMixer``
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ray_pipeline.runners.base import BaseRunner, RunnerResult
from ray_pipeline.config import PipelineConfig


class MLXRunner(BaseRunner):
    """
    Runs MLX implementation for Apple Silicon.
    
    This runner directly imports and executes the MLX implementation modules
    from ``deepseek-from-scratch-python/mlx_impl/``.
    
    Attributes
    ----------
    name : str
        Runner identifier ("mlx")
    stage : str
        Current pipeline stage
    repo_root : Path
        Repository root directory
    mlx_impl_path : Path
        Path to mlx_impl directory
    
    Examples
    --------
    >>> config = PipelineConfig.from_size(ModelSize.SMALL)
    >>> config.backend = Backend.MLX
    >>> runner = MLXRunner(config, stage="pretrain")
    >>> result = runner.run(dataset_uri="./cache/train", pad_token_id=0)
    """

    name = "mlx"

    def __init__(self, config: PipelineConfig, stage: str):
        """
        Initialize the MLX runner.
        
        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration
        stage : str
            Pipeline stage name
        """
        super().__init__(config)
        self.stage = stage
        self.repo_root = Path(__file__).resolve().parents[2]
        self.mlx_impl_path = self.repo_root / "deepseek-from-scratch-python" / "mlx_impl"

    def run(
        self,
        dataset_uri: Optional[str] = None,
        pad_token_id: int = 0,
        training_config: Optional[Dict[str, Any]] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> RunnerResult:
        """
        Execute MLX training using actual implementation modules.
        
        This method imports and uses the following MLX modules:
        
        - ``mlx_impl.attention.MultiHeadLatentAttention`` - MLA attention
        - ``mlx_impl.moe.DeepSeekMoE`` - Mixture of Experts
        - ``mlx_impl.mtp.MTPModel`` - Multi-Token Prediction
        - ``mlx_impl.grpo.GRPOTrainer`` - GRPO alignment
        - ``mlx_impl.sft.SFTTrainer`` - Supervised fine-tuning
        
        Parameters
        ----------
        dataset_uri : str, optional
            Path to dataset directory
        pad_token_id : int
            Padding token ID
        training_config : Dict, optional
            Training hyperparameter overrides
        extra_config : Dict, optional
            Stage-specific configuration
            
        Returns
        -------
        RunnerResult
            Training metrics and artifacts
        """
        if not self.mlx_impl_path.exists():
            raise FileNotFoundError(
                f"MLX implementation not found at {self.mlx_impl_path}"
            )

        # Add mlx_impl to path for imports
        if str(self.mlx_impl_path) not in sys.path:
            sys.path.insert(0, str(self.mlx_impl_path))

        # ============================================================
        # IMPORT ACTUAL MLX IMPLEMENTATION MODULES
        # ============================================================
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
            
            # Attention mechanisms
            from attention import (
                MultiQueryAttention,
                GroupedQueryAttention, 
                MultiHeadLatentAttention,
            )
            # Mixture of Experts
            from moe import DeepSeekMoE, DeepSeekMoEV3Config
            # Multi-Token Prediction
            from mtp import MTPModel
            # Training utilities
            from grpo import GRPOTrainer
            from sft import SFTTrainer, SFTConfig
            from pipeline import ScalingLaws, DataMixer, CurriculumScheduler
            from distillation import compute_distillation_loss, KDLossType
            
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import MLX modules. Ensure MLX is installed: {e}"
            ) from e

        self.logger.info("MLX modules imported successfully")
        self.logger.info("Running stage: %s", self.stage)

        # Set MLX to use GPU
        mx.set_default_device(mx.gpu)
        
        model_cfg = training_config or {}
        d_model = model_cfg.get("d_model", self.config.model.d_model)
        num_heads = model_cfg.get("num_heads", self.config.model.num_heads)
        num_layers = model_cfg.get("num_layers", self.config.model.num_layers)
        vocab_size = model_cfg.get("vocab_size", self.config.model.vocab_size)
        
        metrics = {}
        checkpoint_path = None
        
        # ============================================================
        # STAGE-SPECIFIC EXECUTION USING ACTUAL MLX CLASSES
        # ============================================================
        
        if self.stage == "pretrain":
            # Use the new TinyMLXTrainer for actual training
            self.logger.info(
                "Creating TinyMTPModel: vocab=%d, d_model=%d, layers=%d, mtp_k=%d",
                vocab_size, d_model, num_layers, self.config.model.mtp_k
            )
            
            # Try to use the new tiny_trainer module for real training
            try:
                from tiny_trainer import (
                    TinyMTPModel, 
                    TinyModelConfig, 
                    TinyMLXTrainer,
                    DataLoader,
                )
                
                # Create model config
                tiny_config = TinyModelConfig(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    max_seq_len=self.config.model.max_seq_len,
                    d_latent=self.config.model.d_latent,
                    d_rope=self.config.model.d_rope,
                    num_experts=self.config.model.num_experts,
                    num_shared_experts=self.config.model.num_shared_experts,
                    top_k=self.config.model.top_k,
                    mtp_k=self.config.model.mtp_k,
                )
                
                # Create model
                model = TinyMTPModel(tiny_config)
                
                # Count parameters (recursive for nested dicts/lists)
                def count_params(params):
                    total = 0
                    if isinstance(params, dict):
                        for v in params.values():
                            total += count_params(v)
                    elif isinstance(params, list):
                        for v in params:
                            total += count_params(v)
                    elif hasattr(params, 'size'):
                        total += params.size
                    return total
                
                total_params = count_params(model.parameters())
                self.logger.info("Model parameters: %d (~%.2fM)", total_params, total_params/1e6)
                
                # Check if we have data
                if dataset_uri:
                    self.logger.info("Dataset URI: %s", dataset_uri)
                    
                    # Load tokenizer
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.config.data.tokenizer_name or "gpt2"
                    )
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    # Create data loader
                    train_loader = DataLoader(
                        data_path=dataset_uri,
                        tokenizer=tokenizer,
                        batch_size=self.config.training.batch_size,
                        max_seq_len=self.config.model.max_seq_len,
                        shuffle=True,
                    )
                    
                    # Create trainer
                    trainer = TinyMLXTrainer(
                        model=model,
                        config=tiny_config,
                        learning_rate=self.config.training.learning_rate,
                        weight_decay=self.config.training.weight_decay,
                        warmup_steps=self.config.training.warmup_steps,
                        max_steps=self.config.training.max_steps,
                        gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
                        checkpoint_dir=self.config.training.checkpoint_dir,
                        log_every=self.config.training.log_every_n_steps,
                        save_every=self.config.training.save_every_n_steps,
                        eval_every=self.config.training.eval_every_n_steps,
                    )
                    
                    # Train
                    trainer.train(
                        train_loader=train_loader,
                        valid_loader=None,
                        pad_token_id=pad_token_id,
                    )
                    
                    checkpoint_path = str(Path(self.config.training.checkpoint_dir) / "final")
                    metrics["training_completed"] = True
                    metrics["final_step"] = trainer.global_step
                else:
                    self.logger.warning("No dataset_uri provided, running demo forward pass only")
                    # Demo forward pass
                    batch_size, seq_len = 4, 64
                    input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
                    main_logits, mtp_logits, _ = model(input_ids)
                    mx.eval(main_logits)
                    
                    metrics["main_logits_shape"] = list(main_logits.shape)
                    metrics["num_mtp_heads"] = len(mtp_logits)
                    
            except ImportError as e:
                self.logger.warning("tiny_trainer not available, using legacy MTPModel: %s", e)
                # Fallback to original demo
                model = MTPModel(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    num_layers=num_layers,
                    k_predictions=self.config.model.mtp_k,
                )
                
                # Demo forward pass
                batch_size, seq_len = 4, 64
                input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
                main_logits, future_logits = model(input_ids)
                mx.eval(main_logits)
                
                metrics["main_logits_shape"] = list(main_logits.shape)
                metrics["num_mtp_heads"] = len(future_logits)
            
            self.logger.info("Pretrain stage completed")
            
            # Also test MoE
            moe = DeepSeekMoE(
                d_model=d_model,
                d_hidden=d_model * 2,
                num_experts=self.config.model.num_experts,
                num_shared=self.config.model.num_shared_experts,
                num_routed=self.config.model.num_experts,
                top_k=self.config.model.top_k,
            )
            x = mx.random.normal((4, 64, d_model))
            moe_out = moe(x)
            mx.eval(moe_out)
            metrics["moe_output_shape"] = list(moe_out.shape)
            self.logger.info("DeepSeekMoE forward pass successful")
            
        elif self.stage == "sft":
            # Use SFTTrainer from mlx_impl
            self.logger.info("Initializing SFTTrainer")
            sft_config = SFTConfig()
            sft_config.lora_r = self.config.sft.lora_r
            sft_config.lora_alpha = self.config.sft.lora_alpha
            sft_config.use_neftune = self.config.sft.use_neftune
            sft_config.neftune_alpha = self.config.sft.neftune_alpha
            
            # Create a simple model for SFT demo
            model = MTPModel(
                vocab_size=vocab_size,
                d_model=d_model,
                num_layers=num_layers,
                k_predictions=0,
            )
            trainer = SFTTrainer(model, sft_config)
            
            metrics["lora_r"] = sft_config.lora_r
            metrics["use_neftune"] = sft_config.use_neftune
            self.logger.info("SFTTrainer initialized")
            
        elif self.stage == "grpo":
            # Use GRPOTrainer from mlx_impl
            self.logger.info("Initializing GRPOTrainer with beta=%f", self.config.grpo.beta)
            grpo = GRPOTrainer(beta=self.config.grpo.beta)
            
            # Demo GRPO loss computation
            group_size = self.config.grpo.group_size
            seq_len = 10
            logits = mx.random.normal((group_size, seq_len, vocab_size))
            ref_logits = mx.random.normal((group_size, seq_len, vocab_size))
            input_ids = mx.random.randint(0, vocab_size, (group_size, seq_len))
            rewards = mx.array([1.0, 0.5, -0.5, 0.0][:group_size])
            
            loss = grpo.compute_loss(logits, input_ids, rewards, ref_logits)
            mx.eval(loss)
            
            metrics["grpo_loss"] = float(loss.item())
            metrics["group_size"] = group_size
            self.logger.info("GRPO loss computed: %f", metrics["grpo_loss"])
            
        elif self.stage == "distillation":
            # Use distillation loss from mlx_impl
            self.logger.info("Computing distillation loss")
            from distillation import DistillationConfig
            
            dist_config = DistillationConfig()
            dist_config.temperature = self.config.distillation.temperature
            dist_config.alpha = self.config.distillation.alpha
            
            student_logits = mx.random.normal((2, 10, vocab_size))
            teacher_logits = mx.random.normal((2, 10, vocab_size))
            
            loss = compute_distillation_loss(student_logits, teacher_logits, dist_config)
            mx.eval(loss)
            
            metrics["distillation_loss"] = float(loss.item())
            metrics["temperature"] = dist_config.temperature
            self.logger.info("Distillation loss: %f", metrics["distillation_loss"])
        
        else:
            # Default: run attention demo
            self.logger.info("Running attention demo for stage: %s", self.stage)
            mla = MultiHeadLatentAttention(
                d_model=d_model,
                num_heads=num_heads,
                d_latent=self.config.model.d_latent,
                d_rope=self.config.model.d_rope,
            )
            x = mx.random.normal((4, 64, d_model))
            out = mla(x)
            mx.eval(out)
            metrics["mla_output_shape"] = list(out.shape)

        metrics["stage"] = self.stage
        metrics["backend"] = "mlx"
        
        return RunnerResult(
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            artifacts={},
            extra={"stage": self.stage},
        )
