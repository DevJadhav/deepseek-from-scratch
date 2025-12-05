"""
Training Utilities for Modal GPU Infrastructure
================================================

Helper functions for model building, data loading, and DeepSpeed configuration.
Extracted from distributed_trainer.py for better modularity and testability.

This module contains:
- Model building utilities (_build_model)
- Data loading utilities (_load_data, ParquetDataset)
- DeepSpeed configuration builders
- Common training utilities
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepseek.cloud.modal.logging_utils import get_logger

_logger = get_logger(__name__)


def build_model(config: dict[str, Any]):
    """
    Build a model from configuration.
    
    Attempts to load the full DeepSeek model, falling back to a simplified
    transformer for testing when the full model is not available.
    
    Args:
        config: Model configuration dictionary containing:
            - vocab_size: Vocabulary size
            - hidden_size: Hidden dimension
            - num_layers: Number of transformer layers
            - num_attention_heads: Number of attention heads
            - num_kv_heads: Number of key-value heads (optional)
            - intermediate_size: FFN intermediate dimension
            - max_position_embeddings: Maximum sequence length (optional)
    
    Returns:
        PyTorch model instance
    """
    import torch
    import torch.nn as nn
    
    class SimpleTransformer(nn.Module):
        """
        Simplified transformer for testing when DeepSeek model not available.
        
        This provides a minimal transformer architecture that can be used for
        testing the training infrastructure without the full DeepSeek model.
        """
        
        def __init__(self, config: dict[str, Any]):
            super().__init__()
            
            self.embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=config["hidden_size"],
                    nhead=config["num_attention_heads"],
                    dim_feedforward=config["intermediate_size"],
                    batch_first=True,
                )
                for _ in range(config["num_layers"])
            ])
            self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"])
        
        def forward(self, input_ids, mask=None):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.lm_head(x)
    
    # Try to import actual DeepSeek model
    try:
        from deepseek.model.transformer import DeepSeekModel
        _logger.info("model_loaded", type="DeepSeekModel")
        return DeepSeekModel(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_heads=config["num_attention_heads"],
            num_kv_heads=config.get("num_kv_heads", config["num_attention_heads"]),
            intermediate_size=config["intermediate_size"],
            max_seq_len=config.get("max_position_embeddings", 512),
        )
    except ImportError:
        _logger.warning("deepseek_model_unavailable", fallback="SimpleTransformer")
        return SimpleTransformer(config)


class ParquetDataset:
    """
    Dataset for loading training data from parquet or JSONL files.
    
    Supports:
    - Single parquet/jsonl files
    - Directories containing multiple files
    - Nested directory structures
    - Both tokenized (input_ids) and raw text formats
    """
    
    def __init__(self, path: str | Path):
        """
        Initialize the dataset.
        
        Args:
            path: Path to a parquet/jsonl file or directory containing files
        """
        import torch
        
        self.path = Path(path)
        self.data: list[dict[str, Any]] = []
        
        _logger.debug("dataset_init_started", path=str(self.path))
        
        if not self.path.exists():
            _logger.warning("dataset_path_not_found", path=str(self.path))
            return
        
        self._load_files()
        _logger.info("dataset_loaded", num_samples=len(self.data))
    
    def _load_files(self) -> None:
        """Load data from files."""
        import pyarrow.parquet as pq
        
        if self.path.is_dir():
            self._load_directory()
        elif self.path.is_file():
            self._load_single_file()
    
    def _load_directory(self) -> None:
        """Load all parquet/jsonl files from directory."""
        import pyarrow.parquet as pq
        
        contents = list(self.path.iterdir())
        _logger.debug("directory_contents", count=len(contents))
        
        # Direct files
        for f in self.path.glob("*.parquet"):
            _logger.debug("loading_parquet", path=str(f))
            table = pq.read_table(f)
            self.data.extend(table.to_pylist())
        
        for f in self.path.glob("*.jsonl"):
            _logger.debug("loading_jsonl", path=str(f))
            self._load_jsonl(f)
        
        # Subdirectories
        for f in self.path.glob("**/*.parquet"):
            if f.parent != self.path:
                _logger.debug("loading_parquet_subdir", path=str(f))
                table = pq.read_table(f)
                self.data.extend(table.to_pylist())
        
        for f in self.path.glob("**/*.jsonl"):
            if f.parent != self.path:
                _logger.debug("loading_jsonl_subdir", path=str(f))
                self._load_jsonl(f)
    
    def _load_single_file(self) -> None:
        """Load data from a single file."""
        import pyarrow.parquet as pq
        
        if str(self.path).endswith('.parquet'):
            _logger.debug("loading_single_parquet", path=str(self.path))
            table = pq.read_table(self.path)
            self.data.extend(table.to_pylist())
        elif str(self.path).endswith('.jsonl'):
            _logger.debug("loading_single_jsonl", path=str(self.path))
            self._load_jsonl(self.path)
    
    def _load_jsonl(self, path: Path) -> None:
        """Load data from a JSONL file."""
        with open(path) as fp:
            for line in fp:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch
        
        item = self.data[idx]
        
        # Handle different data formats
        if "input_ids" in item:
            input_ids = torch.tensor(item["input_ids"][:512])  # Truncate
        elif "text" in item:
            # Simple byte-level tokenization (fallback for missing tokenizer)
            text = item["text"][:2048]
            input_ids = torch.tensor([ord(c) % 32000 for c in text[:512]])
        else:
            raise ValueError(f"Unknown data format: {list(item.keys())}")
        
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for batching sequences with padding.
    
    Args:
        batch: List of sample dictionaries with input_ids and attention_mask
        
    Returns:
        Batched and padded tensors
    """
    import torch
    
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, b in enumerate(batch):
        seq_len = len(b["input_ids"])
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = b["attention_mask"]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_training_data(data_path: str, batch_size: int):
    """
    Load training data and create a DataLoader.
    
    Args:
        data_path: Path to training data (parquet or jsonl)
        batch_size: Batch size for training
        
    Returns:
        PyTorch DataLoader
    """
    from torch.utils.data import DataLoader
    
    dataset = ParquetDataset(data_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,  # Modal doesn't support multiprocessing well
    )


def build_deepspeed_config(
    training_config: dict[str, Any],
    distributed_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Build DeepSpeed configuration for ZeRO optimization.
    
    Args:
        training_config: Training hyperparameters
        distributed_config: Distributed training configuration
        
    Returns:
        DeepSpeed configuration dictionary
    """
    zero_stage = distributed_config.get("zero_stage", 2)
    
    config = {
        "train_batch_size": (
            training_config["batch_size"] 
            * distributed_config.get("data_parallel_size", 1)
        ),
        "gradient_accumulation_steps": training_config.get(
            "gradient_accumulation_steps", 1
        ),
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_config["learning_rate"],
                "weight_decay": training_config.get("weight_decay", 0.01),
            }
        },
        "fp16": {
            "enabled": training_config.get("use_amp", True),
        },
        "zero_optimization": {
            "stage": zero_stage,
            "offload_optimizer": {
                "device": "cpu" if zero_stage >= 2 else "none",
            },
            "offload_param": {
                "device": "cpu" if zero_stage >= 3 else "none",
            },
        },
    }
    
    _logger.debug("deepspeed_config_built", zero_stage=zero_stage)
    return config


def get_model_size_config(size: str) -> dict[str, Any]:
    """
    Get model configuration for a given size preset.
    
    Args:
        size: Size preset ("tiny", "small", "medium", "large")
        
    Returns:
        Model configuration dictionary
    """
    configs = {
        "tiny": {
            "hidden_size": 256,
            "num_layers": 4,
            "num_attention_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 512,
            "vocab_size": 32000,
            "max_position_embeddings": 512,
        },
        "small": {
            "hidden_size": 512,
            "num_layers": 8,
            "num_attention_heads": 8,
            "num_kv_heads": 4,
            "intermediate_size": 1024,
            "vocab_size": 32000,
            "max_position_embeddings": 1024,
        },
        "medium": {
            "hidden_size": 1024,
            "num_layers": 16,
            "num_attention_heads": 16,
            "num_kv_heads": 8,
            "intermediate_size": 2048,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,
        },
        "large": {
            "hidden_size": 2048,
            "num_layers": 24,
            "num_attention_heads": 32,
            "num_kv_heads": 8,
            "intermediate_size": 4096,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
        },
    }
    
    return configs.get(size, configs["tiny"])


def get_default_training_config(max_steps: int = 1000) -> dict[str, Any]:
    """
    Get default training configuration.
    
    Args:
        max_steps: Maximum training steps
        
    Returns:
        Training configuration dictionary
    """
    return {
        "batch_size": 4,
        "learning_rate": 1e-4,
        "max_steps": max_steps,
        "warmup_steps": min(10, max_steps // 10),
        "gradient_accumulation_steps": 2,
        "use_amp": True,
        "save_steps": max(50, max_steps // 10),
        "log_steps": 10,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
    }
