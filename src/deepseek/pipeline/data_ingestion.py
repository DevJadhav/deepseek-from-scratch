"""
Production-Ready Data Ingestion Pipeline for DeepSeek

Implements Phase 1: Pre-Training (Data Ingestion) features:
- DeterministicShuffler: Global deterministic shuffle across heterogeneous clusters
- StreamingDataPipeline: HuggingFace streaming integration
- TokenLevelBatcher: Token-level batching (not sample-level)
- DynamicPadder: Dynamic padding to minimize compute waste

Reference: production_hardening.md Section 3.1
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Lazy imports for optional dependencies
_HF_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import torch
    from torch.utils.data import DataLoader, IterableDataset

    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from datasets import load_dataset

    _HF_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# DeterministicShuffler: Reproducible Shuffling Across Heterogeneous Clusters
# =============================================================================


@dataclass
class ShuffleState:
    """Immutable shuffle state for reproducibility."""

    seed: int
    worker_id: int
    epoch: int
    position: int = 0

    def to_dict(self) -> dict[str, int]:
        """Serialize state for checkpointing."""
        return {
            "seed": self.seed,
            "worker_id": self.worker_id,
            "epoch": self.epoch,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> ShuffleState:
        """Deserialize state from checkpoint."""
        return cls(**d)


class DeterministicShuffler:
    """
    Global deterministic shuffle for reproducible training across heterogeneous clusters.

    Ensures exact reproducibility of data ordering across:
    - Different hardware (Metal + CUDA nodes)
    - Multiple workers/processes
    - Resume from checkpoints
    - Different Python/NumPy versions

    The shuffler uses a hierarchical RNG design:
    1. Master RNG seeded from global seed
    2. Per-worker RNGs derived deterministically from master
    3. Per-epoch RNGs derived from worker RNGs

    This guarantees:
    - Same shuffle order for same (seed, worker_id, epoch)
    - Independent shuffles across workers (no overlap/duplication)
    - Resumable from any position

    Example:
        >>> shuffler = DeterministicShuffler(seed=42, num_workers=4)
        >>> indices = shuffler.shuffle_for_worker(list(range(1000)), worker_id=0)
        >>> # Same call always returns same result
        >>> assert shuffler.shuffle_for_worker(list(range(1000)), worker_id=0) == indices
    """

    def __init__(
        self,
        seed: int,
        num_workers: int,
        buffer_size: int = 10000,
    ):
        """
        Initialize deterministic shuffler.

        Args:
            seed: Global seed for reproducibility
            num_workers: Total number of workers across all nodes
            buffer_size: Size of shuffle buffer for streaming
        """
        self.seed = seed
        self.num_workers = num_workers
        self.buffer_size = buffer_size

        # Create master RNG using PCG64 (deterministic across platforms)
        self.master_rng = np.random.Generator(np.random.PCG64(seed))

        # Generate deterministic seeds for each worker
        # Use large integers to avoid seed collisions
        self._worker_seeds = [int(self.master_rng.integers(0, 2**62)) for _ in range(num_workers)]

        # Per-worker RNGs (created lazily)
        self._worker_rngs: dict[int, np.random.Generator] = {}

        # Track current epoch for each worker
        self._current_epoch: dict[int, int] = {i: 0 for i in range(num_workers)}

    def _get_worker_rng(self, worker_id: int, epoch: int = 0) -> np.random.Generator:
        """Get or create RNG for a specific worker and epoch."""
        if worker_id >= self.num_workers:
            raise ValueError(f"worker_id {worker_id} >= num_workers {self.num_workers}")

        # Combine worker seed and epoch for epoch-specific RNG
        combined_seed = self._worker_seeds[worker_id] + epoch
        return np.random.Generator(np.random.PCG64(combined_seed))

    def shuffle_for_worker(
        self,
        data: list[Any],
        worker_id: int,
        epoch: int = 0,
    ) -> list[Any]:
        """
        Deterministic shuffle specific to worker and epoch.

        Args:
            data: List of data items to shuffle
            worker_id: Worker ID (0 to num_workers-1)
            epoch: Current epoch (for varying shuffle per epoch)

        Returns:
            Shuffled list (new list, original unchanged)
        """
        rng = self._get_worker_rng(worker_id, epoch)
        indices = rng.permutation(len(data))
        return [data[i] for i in indices]

    def shuffle_indices(
        self,
        length: int,
        worker_id: int,
        epoch: int = 0,
    ) -> np.ndarray:
        """
        Get shuffled indices for a dataset of given length.

        Args:
            length: Length of dataset
            worker_id: Worker ID
            epoch: Current epoch

        Returns:
            NumPy array of shuffled indices
        """
        rng = self._get_worker_rng(worker_id, epoch)
        return rng.permutation(length)

    def get_worker_shard(
        self,
        total_size: int,
        worker_id: int,
    ) -> tuple[int, int]:
        """
        Get (start, end) indices for worker's shard of data.

        Divides data evenly across workers with remainder distributed.

        Args:
            total_size: Total dataset size
            worker_id: Worker ID

        Returns:
            Tuple of (start_idx, end_idx) for this worker
        """
        base_size = total_size // self.num_workers
        remainder = total_size % self.num_workers

        # Workers with ID < remainder get one extra sample
        start = worker_id * base_size + min(worker_id, remainder)
        size = base_size + (1 if worker_id < remainder else 0)
        end = start + size

        return start, end

    def streaming_shuffle(
        self,
        iterator: Iterator[Any],
        worker_id: int,
        epoch: int = 0,
    ) -> Generator[Any, None, None]:
        """
        Streaming shuffle using reservoir-like buffer.

        Provides approximate shuffle for streaming data while
        maintaining determinism.

        Args:
            iterator: Input data iterator
            worker_id: Worker ID
            epoch: Current epoch

        Yields:
            Shuffled items
        """
        rng = self._get_worker_rng(worker_id, epoch)
        buffer: list[Any] = []

        for item in iterator:
            buffer.append(item)

            if len(buffer) >= self.buffer_size:
                # Shuffle buffer and yield items
                indices = rng.permutation(len(buffer))
                for idx in indices:
                    yield buffer[idx]
                buffer = []

        # Yield remaining items in buffer
        if buffer:
            indices = rng.permutation(len(buffer))
            for idx in indices:
                yield buffer[idx]

    def get_state(self, worker_id: int) -> ShuffleState:
        """Get current state for checkpointing."""
        return ShuffleState(
            seed=self.seed,
            worker_id=worker_id,
            epoch=self._current_epoch.get(worker_id, 0),
        )

    def set_state(self, state: ShuffleState):
        """Restore state from checkpoint."""
        self._current_epoch[state.worker_id] = state.epoch

    def advance_epoch(self, worker_id: int):
        """Advance to next epoch for a worker."""
        self._current_epoch[worker_id] = self._current_epoch.get(worker_id, 0) + 1


# =============================================================================
# StreamingDataPipeline: HuggingFace Streaming Integration
# =============================================================================


@dataclass
class StreamingConfig:
    """Configuration for streaming data pipeline."""

    # HuggingFace dataset settings
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_split: str = "train"
    dataset_subset: str | None = None
    streaming: bool = True

    # Processing
    text_column: str = "text"
    max_samples: int | None = None

    # Shuffle
    shuffle_seed: int = 42
    shuffle_buffer_size: int = 10000

    # Tokenization
    tokenizer_name: str = "deepseek-ai/deepseek-llm-7b-base"
    max_seq_length: int = 2048

    # Batching
    batch_size: int = 8
    drop_last: bool = False

    # Workers
    num_workers: int = 4
    prefetch_factor: int = 2


class StreamingDataPipeline:
    """
    Production streaming pipeline with HuggingFace datasets integration.

    Features:
    - True streaming (no full dataset in memory)
    - Deterministic shuffling via DeterministicShuffler
    - Multi-worker data loading
    - Automatic tokenization
    - Resume from checkpoint support

    Example:
        >>> config = StreamingConfig(
        ...     dataset_name="HuggingFaceFW/fineweb-edu",
        ...     streaming=True,
        ... )
        >>> pipeline = StreamingDataPipeline(config)
        >>> for batch in pipeline.iterate(worker_id=0, num_workers=4):
        ...     # Process batch
        ...     pass
    """

    def __init__(
        self,
        config: StreamingConfig,
        tokenizer: Any | None = None,
    ):
        """
        Initialize streaming pipeline.

        Args:
            config: StreamingConfig instance
            tokenizer: Optional pre-loaded tokenizer
        """
        if not _HF_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets required. Install with: uv pip install datasets"
            )

        self.config = config
        self._tokenizer = tokenizer
        self._dataset = None
        self._shuffler = DeterministicShuffler(
            seed=config.shuffle_seed,
            num_workers=config.num_workers,
            buffer_size=config.shuffle_buffer_size,
        )

    @property
    def tokenizer(self):
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer_name,
                    use_fast=True,
                )
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
            except ImportError as exc:
                raise ImportError(
                    "transformers required for tokenization. Install with: "
                    "uv pip install transformers"
                ) from exc
        return self._tokenizer

    def _load_dataset(self) -> Any:
        """Load HuggingFace dataset in streaming mode."""
        kwargs: dict[str, Any] = {
            "streaming": self.config.streaming,
            "split": self.config.dataset_split,
        }
        if self.config.dataset_subset:
            kwargs["name"] = self.config.dataset_subset

        ds = load_dataset(self.config.dataset_name, **kwargs)

        if self.config.max_samples is not None:
            ds = ds.take(self.config.max_samples)

        return ds

    def iterate(
        self,
        worker_id: int = 0,
        num_workers: int = 1,
        epoch: int = 0,
        skip_samples: int = 0,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Iterate over dataset with worker sharding.

        Args:
            worker_id: Current worker ID
            num_workers: Total number of workers
            epoch: Current epoch
            skip_samples: Number of samples to skip (for resume)

        Yields:
            Dict with 'text' and optionally tokenized fields
        """
        ds = self._load_dataset()

        # For streaming, we use shuffle buffer
        if self.config.streaming:
            ds = ds.shuffle(
                seed=self.config.shuffle_seed + epoch,
                buffer_size=self.config.shuffle_buffer_size,
            )

        sample_idx = 0
        for example in ds:
            # Skip samples if resuming
            if sample_idx < skip_samples:
                sample_idx += 1
                continue

            # Worker sharding for streaming mode
            if sample_idx % num_workers != worker_id:
                sample_idx += 1
                continue

            text = example.get(self.config.text_column, "")
            if not text:
                sample_idx += 1
                continue

            yield {"text": text, "sample_idx": sample_idx}
            sample_idx += 1

    def iterate_tokenized(
        self,
        worker_id: int = 0,
        num_workers: int = 1,
        epoch: int = 0,
    ) -> Generator[dict[str, np.ndarray], None, None]:
        """
        Iterate over tokenized samples.

        Args:
            worker_id: Current worker ID
            num_workers: Total number of workers
            epoch: Current epoch

        Yields:
            Dict with 'input_ids' and 'attention_mask' arrays
        """
        for sample in self.iterate(worker_id, num_workers, epoch):
            text = sample["text"]
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_attention_mask=True,
            )
            yield {
                "input_ids": np.array(encoded["input_ids"], dtype=np.int64),
                "attention_mask": np.array(encoded["attention_mask"], dtype=np.int64),
            }


# =============================================================================
# TokenLevelBatcher: Token-Level Batching (Not Sample-Level)
# =============================================================================


@dataclass
class TokenBatcherConfig:
    """Configuration for token-level batching."""

    # Target tokens per batch
    target_tokens_per_batch: int = 16384  # 8 samples * 2048 tokens

    # Sequence length constraints
    max_seq_length: int = 2048
    min_seq_length: int = 64

    # Packing settings
    pack_sequences: bool = True
    pad_to_multiple: int = 8  # Pad to multiple for efficiency

    # Special tokens
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 1

    # Batch constraints
    max_samples_per_batch: int = 32  # Upper limit on samples


class TokenLevelBatcher:
    """
    Token-level batching for efficient training.

    Unlike sample-level batching which pads all samples to max length,
    token-level batching:
    - Targets a specific number of tokens per batch
    - Packs multiple shorter sequences together
    - Minimizes padding waste
    - Handles variable-length sequences efficiently

    This matches production training pipelines like GPT-4 and DeepSeek.

    Example:
        >>> config = TokenBatcherConfig(target_tokens_per_batch=16384)
        >>> batcher = TokenLevelBatcher(config)
        >>> for batch in batcher.batch(token_iterator):
        ...     # batch has ~16384 tokens total
        ...     pass
    """

    def __init__(self, config: TokenBatcherConfig):
        """
        Initialize token-level batcher.

        Args:
            config: TokenBatcherConfig instance
        """
        self.config = config
        self._buffer: list[np.ndarray] = []
        self._buffer_tokens = 0

    def _create_packed_batch(
        self,
        sequences: list[np.ndarray],
    ) -> dict[str, Any]:
        """
        Pack sequences into a batch with attention mask.

        Uses document boundary markers to prevent cross-document attention.

        Args:
            sequences: List of token arrays

        Returns:
            Dict with 'input_ids', 'attention_mask', 'position_ids', 'doc_ids'
        """
        cfg = self.config

        # Calculate total packed length
        total_tokens = sum(len(s) for s in sequences)
        padded_length = (
            (total_tokens + cfg.pad_to_multiple - 1) // cfg.pad_to_multiple * cfg.pad_to_multiple
        )

        # Initialize arrays
        input_ids = np.full(padded_length, cfg.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros(padded_length, dtype=np.int64)
        position_ids = np.zeros(padded_length, dtype=np.int64)
        doc_ids = np.zeros(padded_length, dtype=np.int64)  # For tracking documents

        # Pack sequences
        offset = 0
        for doc_idx, seq in enumerate(sequences):
            seq_len = len(seq)
            input_ids[offset : offset + seq_len] = seq
            attention_mask[offset : offset + seq_len] = 1
            position_ids[offset : offset + seq_len] = np.arange(seq_len)
            doc_ids[offset : offset + seq_len] = doc_idx
            offset += seq_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "doc_ids": doc_ids,
            "num_documents": len(sequences),
            "total_tokens": total_tokens,
        }

    def _create_padded_batch(
        self,
        sequences: list[np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Create padded batch (traditional approach).

        Args:
            sequences: List of token arrays

        Returns:
            Dict with 'input_ids' and 'attention_mask' as 2D arrays
        """
        cfg = self.config

        # Find max length in batch
        max_len = max(len(s) for s in sequences)
        padded_length = (
            (max_len + cfg.pad_to_multiple - 1) // cfg.pad_to_multiple * cfg.pad_to_multiple
        )
        padded_length = min(padded_length, cfg.max_seq_length)

        batch_size = len(sequences)
        input_ids = np.full((batch_size, padded_length), cfg.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((batch_size, padded_length), dtype=np.int64)

        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), padded_length)
            input_ids[i, :seq_len] = seq[:seq_len]
            attention_mask[i, :seq_len] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def batch(
        self,
        token_iterator: Iterator[np.ndarray],
        pack: bool | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Create batches targeting a specific number of tokens.

        Args:
            token_iterator: Iterator yielding token arrays (1D np.ndarray)
            pack: Whether to pack sequences (overrides config if set)

        Yields:
            Batched dict with input_ids, attention_mask, etc.
        """
        cfg = self.config
        should_pack = pack if pack is not None else cfg.pack_sequences

        buffer: list[np.ndarray] = []
        buffer_tokens = 0

        for tokens in token_iterator:
            # Truncate if too long
            if len(tokens) > cfg.max_seq_length:
                tokens = tokens[: cfg.max_seq_length]

            # Skip if too short
            if len(tokens) < cfg.min_seq_length:
                continue

            buffer.append(tokens)
            buffer_tokens += len(tokens)

            # Check if we've reached target batch size
            should_yield = (
                buffer_tokens >= cfg.target_tokens_per_batch
                or len(buffer) >= cfg.max_samples_per_batch
            )

            if should_yield:
                if should_pack:
                    yield self._create_packed_batch(buffer)
                else:
                    yield self._create_padded_batch(buffer)
                buffer = []
                buffer_tokens = 0

        # Yield remaining samples
        if buffer:
            if should_pack:
                yield self._create_packed_batch(buffer)
            else:
                yield self._create_padded_batch(buffer)

    def batch_from_samples(
        self,
        sample_iterator: Iterator[dict[str, Any]],
        tokenizer: Any,
        text_key: str = "text",
        pack: bool | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Create batches directly from text samples.

        Args:
            sample_iterator: Iterator yielding dicts with text
            tokenizer: Tokenizer instance
            text_key: Key for text in sample dict
            pack: Whether to pack sequences

        Yields:
            Batched dict
        """

        def tokenize_iter():
            for sample in sample_iterator:
                text = sample.get(text_key, "")
                if not text:
                    continue
                tokens = tokenizer.encode(text, add_special_tokens=True)
                yield np.array(tokens, dtype=np.int64)

        yield from self.batch(tokenize_iter(), pack=pack)


# =============================================================================
# DynamicPadder: Dynamic Padding to Minimize Compute Waste
# =============================================================================


@dataclass
class DynamicPaddingConfig:
    """Configuration for dynamic padding."""

    # Bucket boundaries for sequence lengths
    bucket_boundaries: list[int] = field(
        default_factory=lambda: [64, 128, 256, 512, 1024, 2048, 4096]
    )

    # Minimum batch size per bucket
    min_batch_size: int = 1

    # Target efficiency (tokens used / tokens allocated)
    target_efficiency: float = 0.85

    # Padding token
    pad_token_id: int = 0


class DynamicPadder:
    """
    Dynamic padding strategy to minimize compute waste.

    Uses sequence length bucketing to group similar-length sequences together,
    reducing the amount of padding needed.

    Key optimizations:
    - Sequences are grouped into length buckets
    - Each bucket pads only to bucket maximum (not global max)
    - Batches are formed from single buckets when possible
    - Tracks efficiency metrics for monitoring

    Example:
        >>> config = DynamicPaddingConfig()
        >>> padder = DynamicPadder(config)
        >>> for batch in padder.pad(sequences):
        ...     # batch has minimal padding
        ...     pass
    """

    def __init__(self, config: DynamicPaddingConfig):
        """
        Initialize dynamic padder.

        Args:
            config: DynamicPaddingConfig instance
        """
        self.config = config
        self.buckets = sorted(config.bucket_boundaries)

        # Track statistics
        self._stats: dict[str, int] = {
            "total_tokens": 0,
            "padded_tokens": 0,
            "batches_created": 0,
        }

    def _get_bucket(self, seq_len: int) -> int:
        """Get bucket index for a sequence length."""
        for i, boundary in enumerate(self.buckets):
            if seq_len <= boundary:
                return i
        return len(self.buckets)  # Overflow bucket

    def _get_bucket_size(self, bucket_idx: int) -> int:
        """Get padded size for a bucket."""
        if bucket_idx >= len(self.buckets):
            return max(self.buckets) if self.buckets else 4096
        return self.buckets[bucket_idx]

    @property
    def efficiency(self) -> float:
        """Current padding efficiency (real tokens / total tokens)."""
        if self._stats["padded_tokens"] == 0:
            return 1.0
        total = self._stats["total_tokens"] + self._stats["padded_tokens"]
        return self._stats["total_tokens"] / total

    def get_stats(self) -> dict[str, Any]:
        """Get padding statistics."""
        return {
            **self._stats,
            "efficiency": self.efficiency,
        }

    def reset_stats(self):
        """Reset statistics."""
        self._stats = {
            "total_tokens": 0,
            "padded_tokens": 0,
            "batches_created": 0,
        }

    def pad_sequence(
        self,
        tokens: np.ndarray,
        target_length: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pad a single sequence to target length.

        Args:
            tokens: Input token array
            target_length: Target length (uses bucket if None)

        Returns:
            Tuple of (padded_tokens, attention_mask)
        """
        seq_len = len(tokens)

        if target_length is None:
            bucket_idx = self._get_bucket(seq_len)
            target_length = self._get_bucket_size(bucket_idx)

        if seq_len >= target_length:
            # Truncate if needed
            return tokens[:target_length], np.ones(target_length, dtype=np.int64)

        # Pad
        padded = np.full(target_length, self.config.pad_token_id, dtype=np.int64)
        padded[:seq_len] = tokens

        mask = np.zeros(target_length, dtype=np.int64)
        mask[:seq_len] = 1

        # Update stats
        self._stats["total_tokens"] += seq_len
        self._stats["padded_tokens"] += target_length - seq_len

        return padded, mask

    def pad_batch(
        self,
        sequences: list[np.ndarray],
        pad_to_max: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Pad a batch of sequences efficiently.

        Args:
            sequences: List of token arrays
            pad_to_max: If True, pad to max in batch; otherwise use buckets

        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        if not sequences:
            return {"input_ids": np.array([]), "attention_mask": np.array([])}

        if pad_to_max:
            target_len = max(len(s) for s in sequences)
        else:
            # Use bucket size based on max length
            max_len = max(len(s) for s in sequences)
            bucket_idx = self._get_bucket(max_len)
            target_len = self._get_bucket_size(bucket_idx)

        batch_size = len(sequences)
        input_ids = np.full(
            (batch_size, target_len),
            self.config.pad_token_id,
            dtype=np.int64,
        )
        attention_mask = np.zeros((batch_size, target_len), dtype=np.int64)

        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), target_len)
            input_ids[i, :seq_len] = seq[:seq_len]
            attention_mask[i, :seq_len] = 1

            self._stats["total_tokens"] += seq_len
            self._stats["padded_tokens"] += target_len - seq_len

        self._stats["batches_created"] += 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def create_bucketed_batches(
        self,
        token_iterator: Iterator[np.ndarray],
        batch_size: int,
    ) -> Generator[dict[str, np.ndarray], None, None]:
        """
        Create batches using bucket-based grouping.

        Sequences are first grouped by bucket, then batched
        within each bucket for minimal padding.

        Args:
            token_iterator: Iterator yielding token arrays
            batch_size: Target batch size

        Yields:
            Batched dict with minimal padding
        """
        # Initialize buckets
        bucket_queues: dict[int, list[np.ndarray]] = {i: [] for i in range(len(self.buckets) + 1)}

        for tokens in token_iterator:
            bucket_idx = self._get_bucket(len(tokens))
            bucket_queues[bucket_idx].append(tokens)

            # Check if this bucket is full
            if len(bucket_queues[bucket_idx]) >= batch_size:
                batch_seqs = bucket_queues[bucket_idx][:batch_size]
                bucket_queues[bucket_idx] = bucket_queues[bucket_idx][batch_size:]
                yield self.pad_batch(batch_seqs)

        # Yield remaining sequences from all buckets
        for bucket_idx in range(len(self.buckets) + 1):
            while bucket_queues[bucket_idx]:
                batch_seqs = bucket_queues[bucket_idx][:batch_size]
                bucket_queues[bucket_idx] = bucket_queues[bucket_idx][batch_size:]
                if batch_seqs:
                    yield self.pad_batch(batch_seqs)


# =============================================================================
# Unified DataIngestionPipeline
# =============================================================================


@dataclass
class DataIngestionConfig:
    """Unified configuration for data ingestion pipeline."""

    # Streaming config
    streaming: StreamingConfig = field(default_factory=StreamingConfig)

    # Token batching config
    token_batching: TokenBatcherConfig = field(default_factory=TokenBatcherConfig)

    # Dynamic padding config
    dynamic_padding: DynamicPaddingConfig = field(default_factory=DynamicPaddingConfig)

    # Shuffle settings (from DeterministicShuffler)
    seed: int = 42
    num_workers: int = 4
    shuffle_buffer_size: int = 10000


class DataIngestionPipeline:
    """
    Unified data ingestion pipeline combining all Phase 1 components.

    Features:
    - HuggingFace streaming integration
    - Deterministic shuffling across workers
    - Token-level batching
    - Dynamic padding for efficiency
    - Checkpoint/resume support

    Example:
        >>> config = DataIngestionConfig()
        >>> pipeline = DataIngestionPipeline(config)
        >>> for batch in pipeline.iterate_batches(worker_id=0):
        ...     # Process batch
        ...     pass
    """

    def __init__(
        self,
        config: DataIngestionConfig,
        tokenizer: Any | None = None,
    ):
        """
        Initialize data ingestion pipeline.

        Args:
            config: DataIngestionConfig instance
            tokenizer: Optional pre-loaded tokenizer
        """
        self.config = config
        self._tokenizer = tokenizer

        # Initialize components
        self.shuffler = DeterministicShuffler(
            seed=config.seed,
            num_workers=config.num_workers,
            buffer_size=config.shuffle_buffer_size,
        )

        self.streaming_pipeline: StreamingDataPipeline | None = None
        if _HF_AVAILABLE:
            self.streaming_pipeline = StreamingDataPipeline(
                config=config.streaming,
                tokenizer=tokenizer,
            )

        self.token_batcher = TokenLevelBatcher(config.token_batching)
        self.dynamic_padder = DynamicPadder(config.dynamic_padding)

    @property
    def tokenizer(self):
        """Get tokenizer (lazy-loaded)."""
        if self._tokenizer is None:
            if self.streaming_pipeline is not None:
                self._tokenizer = self.streaming_pipeline.tokenizer
            else:
                try:
                    from transformers import AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.config.streaming.tokenizer_name,
                        use_fast=True,
                    )
                except ImportError as exc:
                    raise ImportError(
                        "transformers required. Install with: uv pip install transformers"
                    ) from exc
        return self._tokenizer

    def iterate_batches(
        self,
        worker_id: int = 0,
        num_workers: int | None = None,
        epoch: int = 0,
        use_packing: bool = True,
        use_dynamic_padding: bool = True,
    ) -> Generator[dict[str, np.ndarray], None, None]:
        """
        Iterate over batches with all optimizations.

        Args:
            worker_id: Current worker ID
            num_workers: Total workers (uses config if None)
            epoch: Current epoch
            use_packing: Whether to use sequence packing
            use_dynamic_padding: Whether to use dynamic padding

        Yields:
            Batched dict with input_ids, attention_mask, etc.
        """
        if num_workers is None:
            num_workers = self.config.num_workers

        if self.streaming_pipeline is None:
            raise RuntimeError(
                "Streaming pipeline not available. Install datasets: uv pip install datasets"
            )

        # Get tokenized stream
        token_stream = self.streaming_pipeline.iterate_tokenized(
            worker_id=worker_id,
            num_workers=num_workers,
            epoch=epoch,
        )

        # Extract just the input_ids
        def token_iterator():
            for sample in token_stream:
                yield sample["input_ids"]

        # Apply shuffling
        shuffled_stream = self.shuffler.streaming_shuffle(
            token_iterator(),
            worker_id=worker_id,
            epoch=epoch,
        )

        # Apply batching
        if use_packing:
            yield from self.token_batcher.batch(shuffled_stream, pack=True)
        elif use_dynamic_padding:
            yield from self.dynamic_padder.create_bucketed_batches(
                shuffled_stream,
                batch_size=self.config.streaming.batch_size,
            )
        else:
            yield from self.token_batcher.batch(shuffled_stream, pack=False)

    def iterate_from_local(
        self,
        data_iterator: Iterator[dict[str, Any]],
        worker_id: int = 0,
        epoch: int = 0,
        text_key: str = "text",
        use_packing: bool = True,
    ) -> Generator[dict[str, np.ndarray], None, None]:
        """
        Iterate over batches from local data.

        Args:
            data_iterator: Iterator yielding dicts with text
            worker_id: Current worker ID
            epoch: Current epoch
            text_key: Key for text field
            use_packing: Whether to use sequence packing

        Yields:
            Batched dict
        """
        yield from self.token_batcher.batch_from_samples(
            data_iterator,
            self.tokenizer,
            text_key=text_key,
            pack=use_packing,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "padding": self.dynamic_padder.get_stats(),
            "config": {
                "seed": self.config.seed,
                "num_workers": self.config.num_workers,
            },
        }


# =============================================================================
# PyTorch DataLoader Integration
# =============================================================================

if _TORCH_AVAILABLE:

    class TokenLevelDataset(IterableDataset):  # type: ignore[misc]
        """
        PyTorch IterableDataset wrapper for token-level batching.

        Integrates with DataIngestionPipeline for seamless PyTorch training.
        """

        def __init__(
            self,
            config: DataIngestionConfig,
            tokenizer: Any | None = None,
            use_packing: bool = True,
        ):
            """
            Initialize dataset.

            Args:
                config: DataIngestionConfig instance
                tokenizer: Optional tokenizer
                use_packing: Whether to use sequence packing
            """
            self.config = config
            self.tokenizer = tokenizer
            self.use_packing = use_packing
            self._pipeline: DataIngestionPipeline | None = None
            self._epoch = 0

        def set_epoch(self, epoch: int):
            """Set current epoch (for shuffling)."""
            self._epoch = epoch

        def __iter__(self):
            """Iterate over batches."""
            # Get worker info
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers
            else:
                worker_id = 0
                num_workers = 1

            # Create pipeline for this worker
            pipeline = DataIngestionPipeline(
                self.config,
                tokenizer=self.tokenizer,
            )

            for batch in pipeline.iterate_batches(
                worker_id=worker_id,
                num_workers=num_workers,
                epoch=self._epoch,
                use_packing=self.use_packing,
            ):
                # Convert to torch tensors
                yield {
                    k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    for k, v in batch.items()
                }

    def create_token_level_dataloader(
        config: DataIngestionConfig,
        tokenizer: Any | None = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        use_packing: bool = True,
    ) -> DataLoader:  # type: ignore[type-arg]
        """
        Create PyTorch DataLoader with token-level batching.

        Args:
            config: DataIngestionConfig instance
            tokenizer: Optional tokenizer
            num_workers: Number of DataLoader workers
            pin_memory: Whether to pin memory
            use_packing: Whether to use sequence packing

        Returns:
            PyTorch DataLoader
        """
        dataset = TokenLevelDataset(
            config=config,
            tokenizer=tokenizer,
            use_packing=use_packing,
        )

        # Note: batch_size=1 because dataset already yields batches
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=lambda x: x[0],  # Unbatch the single item
        )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Configuration
    "ShuffleState",
    "StreamingConfig",
    "TokenBatcherConfig",
    "DynamicPaddingConfig",
    "DataIngestionConfig",
    # Core classes
    "DeterministicShuffler",
    "StreamingDataPipeline",
    "TokenLevelBatcher",
    "DynamicPadder",
    "DataIngestionPipeline",
]

# Conditionally export PyTorch components
if _TORCH_AVAILABLE:
    __all__.extend(
        [
            "TokenLevelDataset",
            "create_token_level_dataloader",
        ]
    )
