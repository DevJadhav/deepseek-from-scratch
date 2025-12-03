"""Ray Data preparation stage."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import ray
from ray import worker
import ray.data as rd
from ray.data.dataset import Dataset

from ray_pipeline.config import PipelineConfig
from ray_pipeline.stages.base import BaseStage, StageContext

# Prefer PyTorch implementation of pipeline utilities (has full feature set)
from deepseek.training.pipeline import DataMixer, CurriculumScheduler  # type: ignore

try:
    from transformers import AutoTokenizer  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "transformers must be installed to run the data preparation stage. "
        "Install with `pip install transformers tokenizers`"
    ) from exc


class DataPrepStage(BaseStage):
    """Builds a Ray Dataset with tokenized, curriculum-aware samples."""

    stage_name = "data_prep"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.tokenizer = None

    def setup(self):
        """Initializes Ray runtime and tokenizer."""
        if not ray.is_initialized():
            self.logger.info("Initializing Ray runtime for data preparation stage")
            ray.init(address=self.config.distributed.ray_address or None, ignore_reinit_error=True)

        tokenizer_name = (
            self.config.data.tokenizer_path or self.config.data.tokenizer_name
        )
        self.logger.info("Loading tokenizer %s", tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def run(self, context: StageContext) -> StageContext:
        self.setup()
        data_dir = Path(self.config.data.data_dir)
        cache_dir = Path(self.config.data.cache_dir) / self.config.run_name / "pretrain"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Load per-domain datasets
        domain_datasets: Dict[str, Dataset] = {}
        for domain, weight in self.config.data.domain_weights.items():
            if weight <= 0:
                continue
            ds = self._read_domain_dataset(domain, data_dir)
            if ds is None:
                self.logger.warning("No data found for domain '%s'", domain)
                continue
            domain_datasets[domain] = ds

        if not domain_datasets:
            raise RuntimeError("No domain datasets could be loaded. Check data paths.")

        # Mix datasets according to weights
        mixer = DataMixer(self.config.data.domain_weights)
        mixed_ds = self._mix_datasets(domain_datasets, mixer)

        # Curriculum scheduling
        if self.config.data.use_curriculum:
            scheduler = CurriculumScheduler(
                start_seq_len=self.config.data.curriculum_start_seq_len,
                end_seq_len=self.config.data.curriculum_end_seq_len,
                warmup_steps=self.config.data.curriculum_warmup_steps,
                total_steps=self.config.data.curriculum_total_steps,
            )
            target_seq_len = scheduler.get_seq_length(0)
        else:
            scheduler = None
            target_seq_len = self.config.model.max_seq_len

        self.logger.info(
            "Tokenizing dataset with seq_len=%d using %s",
            target_seq_len,
            self.tokenizer.name_or_path,
        )

        tokenized = mixed_ds.map_batches(
            lambda batch: self._tokenize_batch(batch, target_seq_len),
            batch_format="pandas",
            compute=rd.ActorPoolStrategy(num_actors=self.config.data.num_workers),
        )

        num_rows = tokenized.count()
        # Persist dataset to parquet for reproducibility
        output_uri = str(cache_dir)
        self.logger.info("Writing tokenized dataset to %s", output_uri)
        tokenized.write_parquet(output_uri, try_create_dir=True)

        context.previous_output = output_uri
        context.metadata["dataset_path"] = output_uri
        context.metadata["num_rows"] = num_rows
        context.metadata["seq_len"] = target_seq_len
        context.metadata["domains"] = list(domain_datasets.keys())
        context.metadata["tokenizer"] = self.tokenizer.name_or_path
        context.metadata["pad_token_id"] = int(self.tokenizer.pad_token_id)

        self.teardown()
        return context

    def teardown(self):
        if ray.is_initialized() and not worker.global_worker.node.should_shutdown():
            self.logger.info("Ray runtime remains active for subsequent stages")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _read_domain_dataset(self, domain: str, base_dir: Path) -> Optional[Dataset]:
        """Load a domain dataset as a Ray Dataset."""
        domain_paths = self.config.data.domain_paths or {}
        path = Path(domain_paths.get(domain, base_dir / domain))

        if not path.exists():
            self.logger.warning("Domain path %s does not exist", path)
            return None

        if path.is_dir():
            files = list(path.glob("*.jsonl")) or list(path.glob("*.json"))
            if not files:
                files = list(path.glob("*.txt"))
            if not files:
                self.logger.warning("No supported files found in %s", path)
                return None
            datasets = [self._read_file(f) for f in files]
            datasets = [ds for ds in datasets if ds is not None]
            if not datasets:
                return None
            ds = datasets[0]
            for other in datasets[1:]:
                ds = ds.union(other)
            return ds
        else:
            return self._read_file(path)

    def _read_file(self, path: Path) -> Optional[Dataset]:
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            return ray.data.read_json(str(path))
        if suffix == ".parquet":
            return ray.data.read_parquet(str(path))
        if suffix in {".txt", ".text"}:
            return ray.data.read_text(str(path)).map(lambda row: {"text": row["text"]})
        self.logger.warning("Unsupported file type %s", path)
        return None

    def _mix_datasets(self, datasets: Dict[str, Dataset], mixer: DataMixer) -> Dataset:
        """Interleave datasets proportionally to domain weights."""
        weights = mixer.weights
        min_weight = min(weights[domain] for domain in datasets.keys())
        repeats: List[Dataset] = []
        for domain, ds in datasets.items():
            repeat_factor = max(1, int(round(weights[domain] / min_weight)))
            repeats.extend([ds] * repeat_factor)
            self.logger.info(
                "Domain %s weight %.3f repeat %d", domain, weights[domain], repeat_factor
            )
        mixed = repeats[0]
        for ds in repeats[1:]:
            mixed = mixed.union(ds)
        return mixed.randomize_block_order()

    def _tokenize_batch(self, batch, seq_len: int):
        texts = batch["text"].tolist()
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="np",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
