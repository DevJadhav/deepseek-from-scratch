"""Ray Data preparation stage with framework selection.

This stage implements data preparation (tokenization, mixing, curriculum)
with automatic framework selection for CPU-bound operations.

Framework Selection Strategy (default):
- Data Loading: Rust+CPU → Python+CPU
- Tokenization: Rust+CPU → Python+CPU
- Shuffling: Rust+CPU → Python+CPU

The actual data processing uses Ray for parallelization,
but framework selection determines optimal backends.

Can be configured via:
1. Environment variable: DEEPSEEK_FRAMEWORK_PRESET=rust_primary
2. PipelineConfig.framework_preset
3. Direct FrameworkSelector injection

Auto-download:
- If no data is found, automatically downloads FineWeb-Edu
- Set DEEPSEEK_AUTO_DOWNLOAD=false to disable
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import ray
import ray.data as rd
from ray.data.dataset import Dataset

from deepseek.pipeline.config import PipelineConfig
from deepseek.pipeline.framework_selector import (
    Framework,
    FrameworkSelector,
    PipelineFrameworkConfig,
    TaskType,
)
from deepseek.pipeline.stages.base import BaseStage, StageContext
from deepseek.pipeline.utils.data_downloader import DataDownloader

# Prefer PyTorch implementation of pipeline utilities (has full feature set)
from deepseek.torch.training.pipeline import CurriculumScheduler, DataMixer  # type: ignore

LOGGER = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "transformers must be installed to run the data preparation stage. "
        "Install with `pip install transformers tokenizers`"
    ) from exc


class DataPrepStage(BaseStage):
    """Builds a Ray Dataset with tokenized, curriculum-aware samples.

    Supports framework selection for data processing operations,
    though the actual execution uses Ray for distributed processing.

    Configuration Options:
    - framework_selector: Injected FrameworkSelector instance
    - framework_preset: String preset ("default", "rust_primary", etc.)
    """

    stage_name = "data_prep"

    def __init__(
        self,
        config: PipelineConfig,
        framework_selector: FrameworkSelector | None = None,
        framework_preset: str | None = None,
    ):
        """Initialize DataPrep stage.

        Args:
            config: Pipeline configuration
            framework_selector: Optional pre-configured selector
            framework_preset: Optional preset name for framework config
        """
        super().__init__(config)
        self.tokenizer = None

        # Initialize framework selector
        if framework_selector is not None:
            self._framework_selector = framework_selector
        elif framework_preset:
            fw_config = PipelineFrameworkConfig.from_preset(framework_preset)
            self._framework_selector = FrameworkSelector(fw_config)
        else:
            # Use default configuration
            self._framework_selector = FrameworkSelector()

        self._log_framework_selection()

    def _log_framework_selection(self) -> None:
        """Log selected frameworks for data processing tasks."""
        data_fw = self._framework_selector.select(TaskType.DATA_LOADING)
        tok_fw = self._framework_selector.select(TaskType.TOKENIZATION)
        shuffle_fw = self._framework_selector.select(TaskType.SHUFFLING)

        LOGGER.info(
            "DataPrep framework selection: data_loading=%s, tokenization=%s, shuffling=%s",
            data_fw.value,
            tok_fw.value,
            shuffle_fw.value,
        )

    def setup(self):
        """Initializes Ray runtime and tokenizer."""
        if not ray.is_initialized():
            self.logger.info("Initializing Ray runtime for data preparation stage")
            # Initialize Ray with proper env configuration for uv compatibility
            import sys
            ray.init(
                address=self.config.distributed.ray_address or None,
                ignore_reinit_error=True,
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": ":".join(sys.path),
                    },
                },
            )

        tokenizer_name = self.config.data.tokenizer_path or self.config.data.tokenizer_name
        self.logger.info("Loading tokenizer %s", tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def run(self, context: StageContext) -> StageContext:
        self.setup()
        data_dir = Path(self.config.data.data_dir)
        cache_dir = Path(self.config.data.cache_dir) / self.config.run_name / "pretrain"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check for auto-download
        auto_download = os.environ.get("DEEPSEEK_AUTO_DOWNLOAD", "true").lower() != "false"
        max_samples = int(os.environ.get("DEEPSEEK_DOWNLOAD_SAMPLES", "5000"))

        # Load per-domain datasets
        domain_datasets: dict[str, Dataset] = {}
        missing_domains: list[str] = []

        for domain, weight in self.config.data.domain_weights.items():
            if weight <= 0:
                continue
            ds = self._read_domain_dataset(domain, data_dir)
            if ds is None:
                self.logger.warning("No data found for domain '%s'", domain)
                missing_domains.append(domain)
                continue
            domain_datasets[domain] = ds

        # Auto-download missing domains
        if missing_domains and auto_download:
            self.logger.info("Auto-downloading missing domains: %s", missing_domains)
            downloader = DataDownloader(output_dir=data_dir)
            downloader.download_all_domains(
                max_samples_per_domain=max_samples,
                domains=missing_domains,
                skip_existing=True,
            )
            # Try loading again after download
            for domain in missing_domains:
                ds = self._read_domain_dataset(domain, data_dir)
                if ds is not None:
                    domain_datasets[domain] = ds
                    self.logger.info("Loaded domain '%s' after download", domain)

        if not domain_datasets:
            raise RuntimeError(
                "No domain datasets could be loaded. "
                "Set DEEPSEEK_AUTO_DOWNLOAD=true or download data manually. "
                f"Expected domains: {list(self.config.data.domain_weights.keys())}"
            )

        self.logger.info("Loaded domains: %s", list(domain_datasets.keys()))

        # Mix datasets according to weights
        mixer = DataMixer(self.config.data.domain_weights)
        mixed_ds = self._mix_datasets(domain_datasets, mixer)

        # Curriculum scheduling
        if self.config.data.use_curriculum:
            scheduler = CurriculumScheduler(
                start_seq_len=self.config.data.curriculum_start_seq_len,
                end_seq_len=self.config.data.curriculum_end_seq_len,
                seq_curriculum_steps=self.config.data.curriculum_warmup_steps,
                difficulty_warmup_steps=self.config.data.curriculum_total_steps,
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
            concurrency=self.config.data.num_workers,
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
        """Clean up resources, keeping Ray active for subsequent stages."""
        if ray.is_initialized():
            self.logger.info("Ray runtime remains active for subsequent stages")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _read_domain_dataset(self, domain: str, base_dir: Path) -> Dataset | None:
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

    def _read_file(self, path: Path) -> Dataset | None:
        # Always use absolute paths for Ray compatibility
        abs_path = str(path.resolve())
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            return ray.data.read_json(abs_path)
        if suffix == ".parquet":
            return ray.data.read_parquet(abs_path)
        if suffix in {".txt", ".text"}:
            return ray.data.read_text(abs_path).map(lambda row: {"text": row["text"]})
        self.logger.warning("Unsupported file type %s", path)
        return None

    def _mix_datasets(self, datasets: dict[str, Dataset], mixer: DataMixer) -> Dataset:
        """Interleave datasets proportionally to domain weights."""
        weights = mixer.weights
        min_weight = min(weights[domain] for domain in datasets.keys())
        repeats: list[Dataset] = []
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

    # ------------------------------------------------------------------
    # Framework Selection API
    # ------------------------------------------------------------------
    def get_framework_selector(self) -> FrameworkSelector:
        """Get the framework selector for inspection or modification.

        Returns:
            Current FrameworkSelector instance
        """
        return self._framework_selector

    def set_framework_preset(self, preset: str) -> None:
        """Change framework preset at runtime.

        Args:
            preset: Preset name ("default", "rust_primary", etc.)
        """
        fw_config = PipelineFrameworkConfig.from_preset(preset)
        self._framework_selector.reconfigure(fw_config)
        self._log_framework_selection()

    def get_selected_framework(self, task: TaskType) -> Framework:
        """Get selected framework for a specific task.

        Args:
            task: Task type to query

        Returns:
            Selected Framework
        """
        return self._framework_selector.select(task)
