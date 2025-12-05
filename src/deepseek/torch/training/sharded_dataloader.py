"""
Sharded Binary DataLoader
=========================

Efficient data loader for pre-tokenized binary shards. Supports:
- Memory-mapped file reading for low memory footprint
- Distributed sampling for multi-GPU training
- Sequence packing and curriculum learning
- Modal volume integration for cloud training

Binary Shard Format:
    Header (24 bytes):
        - Magic bytes: 8 bytes ("DSTKN001")
        - Version: 4 bytes (uint32)
        - Num sequences: 4 bytes (uint32)
        - Seq length: 4 bytes (uint32)
        - Dtype: 4 bytes (0=uint16, 1=uint32)

    Data:
        - Token IDs: num_sequences * seq_length * dtype_size
"""

from __future__ import annotations

import json
import mmap
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Magic bytes for binary shard format
SHARD_MAGIC = b"DSTKN001"
HEADER_SIZE = 24  # 8 + 4 + 4 + 4 + 4 bytes


@dataclass
class ShardMetadata:
    """Metadata for a single shard."""

    path: Path
    num_sequences: int
    seq_length: int
    dtype: np.dtype
    size_bytes: int
    hash: str = ""


@dataclass
class ManifestConfig:
    """Configuration from manifest.json."""

    dataset: str = ""
    version: str = ""
    total_shards: int = 0
    total_bytes: int = 0
    tokenizer: str = "gpt2"
    max_seq_len: int = 512
    shards: dict = field(default_factory=dict)


def read_shard_header(path: Path) -> tuple[int, int, np.dtype]:
    """
    Read the header of a binary shard file.

    Returns:
        Tuple of (num_sequences, seq_length, dtype)
    """
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != SHARD_MAGIC:
            raise ValueError(f"Invalid shard magic bytes in {path}: {magic}")

        _ = struct.unpack("<I", f.read(4))[0]  # version
        num_sequences = struct.unpack("<I", f.read(4))[0]
        seq_length = struct.unpack("<I", f.read(4))[0]
        dtype_code = struct.unpack("<I", f.read(4))[0]

        dtype = np.uint16 if dtype_code == 0 else np.uint32

    return num_sequences, seq_length, dtype


def load_manifest(manifest_path: Path) -> ManifestConfig:
    """Load manifest.json and return config."""
    with open(manifest_path) as f:
        data = json.load(f)

    config = ManifestConfig(
        dataset=data.get("dataset", ""),
        version=data.get("version", ""),
        total_shards=data.get("total_shards", 0),
        total_bytes=data.get("total_bytes", 0),
        tokenizer=data.get("metadata", {}).get("tokenizer", "gpt2"),
        max_seq_len=data.get("metadata", {}).get("max_seq_len", 512),
        shards=data.get("shards", {}),
    )
    return config


class ShardedBinaryDataset(Dataset):
    """
    PyTorch Dataset for reading pre-tokenized binary shards.

    Supports memory-mapped access for efficient reading of large datasets.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        manifest_name: str = "manifest.json",
        use_mmap: bool = True,
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the dataset.

        Args:
            shard_dir: Directory containing binary shards and manifest
            manifest_name: Name of the manifest file
            use_mmap: Whether to use memory-mapped files
            shuffle_shards: Whether to shuffle shard order
            seed: Random seed for shuffling
        """
        self.shard_dir = Path(shard_dir)
        self.use_mmap = use_mmap
        self.shuffle_shards = shuffle_shards
        self.seed = seed

        # Load manifest
        manifest_path = self.shard_dir / manifest_name
        if manifest_path.exists():
            self.manifest = load_manifest(manifest_path)
        else:
            # Auto-detect shards
            self.manifest = self._auto_detect_shards()

        # Load shard metadata
        self.shards: list[ShardMetadata] = []
        self._load_shard_metadata()

        # Build index mapping
        self._cumulative_sizes: list[int] = []
        self._build_index()

        # Cached mmap handles
        self._mmap_cache: dict[int, tuple[np.ndarray, int]] = {}

    def _auto_detect_shards(self) -> ManifestConfig:
        """Auto-detect shards when no manifest is present."""
        shard_files = sorted(self.shard_dir.glob("*.bin"))
        return ManifestConfig(
            dataset="auto-detected",
            total_shards=len(shard_files),
            shards={f.name: {"path": str(f)} for f in shard_files},
        )

    def _load_shard_metadata(self):
        """Load metadata for all shards."""
        shard_paths = []

        if self.manifest.shards:
            for name, info in self.manifest.shards.items():
                path = Path(info.get("path", self.shard_dir / name))
                if not path.is_absolute():
                    path = self.shard_dir / path
                if path.exists():
                    shard_paths.append((path, info.get("hash", "")))
        else:
            for p in sorted(self.shard_dir.glob("*.bin")):
                shard_paths.append((p, ""))

        # Shuffle shard order
        if self.shuffle_shards:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(shard_paths)

        for path, hash_val in shard_paths:
            try:
                num_seqs, seq_len, dtype = read_shard_header(path)
                self.shards.append(
                    ShardMetadata(
                        path=path,
                        num_sequences=num_seqs,
                        seq_length=seq_len,
                        dtype=dtype,
                        size_bytes=path.stat().st_size,
                        hash=hash_val,
                    )
                )
            except Exception as e:
                print(f"Warning: Could not read shard {path}: {e}")

    def _build_index(self):
        """Build cumulative index for fast lookup."""
        total = 0
        for shard in self.shards:
            total += shard.num_sequences
            self._cumulative_sizes.append(total)

    def _get_shard_and_offset(self, idx: int) -> tuple[int, int]:
        """Get shard index and offset within shard for global index."""
        shard_idx = np.searchsorted(self._cumulative_sizes, idx, side="right")
        if shard_idx == 0:
            offset = idx
        else:
            offset = idx - self._cumulative_sizes[shard_idx - 1]
        return shard_idx, offset

    def _get_mmap(self, shard_idx: int) -> tuple[np.ndarray, int]:
        """Get memory-mapped array for a shard."""
        if shard_idx in self._mmap_cache:
            return self._mmap_cache[shard_idx]

        shard = self.shards[shard_idx]

        if self.use_mmap:
            with open(shard.path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                # Create numpy array from mmap, skipping header
                data = np.frombuffer(
                    mm,
                    dtype=shard.dtype,
                    offset=HEADER_SIZE,
                ).reshape(shard.num_sequences, shard.seq_length)
        else:
            # Load entire shard into memory
            with open(shard.path, "rb") as f:
                f.seek(HEADER_SIZE)
                data = np.frombuffer(
                    f.read(),
                    dtype=shard.dtype,
                ).reshape(shard.num_sequences, shard.seq_length)

        self._mmap_cache[shard_idx] = (data, shard.seq_length)
        return data, shard.seq_length

    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        shard_idx, offset = self._get_shard_and_offset(idx)
        data, seq_len = self._get_mmap(shard_idx)

        tokens = data[offset].copy()  # Copy to avoid memory issues

        return {
            "input_ids": torch.from_numpy(tokens.astype(np.int64)),
            "labels": torch.from_numpy(tokens.astype(np.int64)),
        }

    @property
    def total_tokens(self) -> int:
        """Total number of tokens in the dataset."""
        return sum(s.num_sequences * s.seq_length for s in self.shards)

    @property
    def seq_length(self) -> int:
        """Sequence length (assumes all shards have same length)."""
        return self.shards[0].seq_length if self.shards else 0


def create_sharded_dataloader(
    shard_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for sharded binary data.

    Args:
        shard_dir: Directory containing shards and manifest
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Pin memory for faster GPU transfer
        distributed: Use distributed sampler
        rank: Process rank for distributed training
        world_size: Total number of processes
        seed: Random seed
        **kwargs: Additional DataLoader arguments

    Returns:
        DataLoader instance
    """
    dataset = ShardedBinaryDataset(
        shard_dir=shard_dir,
        shuffle_shards=shuffle,
        seed=seed,
    )

    if distributed and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **kwargs,
    )


class ModalVolumeDataset(ShardedBinaryDataset):
    """
    Dataset that reads from Modal volume paths.

    This is a thin wrapper that handles Modal volume path conventions.
    """

    def __init__(
        self,
        volume_path: str = "/data/fineweb-edu",
        manifest_name: str = "manifest.json",
        **kwargs,
    ):
        """
        Initialize from Modal volume.

        Args:
            volume_path: Path within Modal volume
            manifest_name: Name of manifest file
            **kwargs: Additional arguments for ShardedBinaryDataset
        """
        shard_dir = Path(volume_path) / "shards"
        manifest_path = Path(volume_path) / manifest_name

        # If manifest is at volume root, copy to shard dir expectation
        if manifest_path.exists() and not (shard_dir / manifest_name).exists():
            shard_dir.mkdir(parents=True, exist_ok=True)
            # Create symlink or use parent manifest

        super().__init__(
            shard_dir=shard_dir if shard_dir.exists() else volume_path,
            manifest_name=manifest_name,
            **kwargs,
        )


def iter_batches(
    shard_dir: str | Path,
    batch_size: int = 8,
    max_batches: int | None = None,
    device: str = "cpu",
) -> Iterator[dict[str, torch.Tensor]]:
    """
    Iterate over batches from sharded data (for simple use cases).

    Args:
        shard_dir: Directory containing shards
        batch_size: Batch size
        max_batches: Maximum number of batches (None = all)
        device: Device to move tensors to

    Yields:
        Batch dictionaries
    """
    dataloader = create_sharded_dataloader(
        shard_dir=shard_dir,
        batch_size=batch_size,
        num_workers=0,  # Simple iteration
        shuffle=False,
    )

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        yield {
            "input_ids": batch["input_ids"].to(device),
            "labels": batch["labels"].to(device),
        }
