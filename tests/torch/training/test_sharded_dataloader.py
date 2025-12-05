"""
Tests for Sharded Binary DataLoader.

Tests cover:
- Binary shard format reading/writing
- ShardedBinaryDataset functionality
- DataLoader creation and iteration
- Memory-mapped file access
- Multi-shard handling
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from deepseek.torch.training.sharded_dataloader import (
    HEADER_SIZE,
    SHARD_MAGIC,
    ManifestConfig,
    ShardedBinaryDataset,
    ShardMetadata,
    create_sharded_dataloader,
    iter_batches,
    load_manifest,
    read_shard_header,
)


def write_test_shard(
    path: Path,
    num_sequences: int,
    seq_length: int,
    dtype_code: int = 0,
    seed: int = 42,
) -> np.ndarray:
    """Write a test shard file and return the token data."""
    dtype = np.uint16 if dtype_code == 0 else np.uint32
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, 50000, size=(num_sequences, seq_length), dtype=dtype)

    with open(path, "wb") as f:
        # Write header
        f.write(SHARD_MAGIC)
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", num_sequences))
        f.write(struct.pack("<I", seq_length))
        f.write(struct.pack("<I", dtype_code))

        # Write data
        f.write(tokens.tobytes())

    return tokens


def create_test_manifest(shard_dir: Path, shard_info: dict) -> Path:
    """Create a test manifest.json file."""
    manifest = {
        "dataset": "test-dataset",
        "version": "1.0.0",
        "total_shards": len(shard_info),
        "total_bytes": sum(s.get("size", 0) for s in shard_info.values()),
        "metadata": {
            "tokenizer": "gpt2",
            "max_seq_len": 512,
        },
        "shards": shard_info,
    }
    manifest_path = shard_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path


class TestShardFormat:
    """Tests for binary shard format."""

    def test_read_shard_header(self, tmp_path):
        """Test reading shard header."""
        shard_path = tmp_path / "test.bin"
        num_seqs, seq_len = 10, 128
        write_test_shard(shard_path, num_seqs, seq_len)

        read_num_seqs, read_seq_len, dtype = read_shard_header(shard_path)

        assert read_num_seqs == num_seqs
        assert read_seq_len == seq_len
        assert dtype == np.uint16

    def test_read_shard_header_uint32(self, tmp_path):
        """Test reading shard with uint32 dtype."""
        shard_path = tmp_path / "test.bin"
        num_seqs, seq_len = 5, 64
        write_test_shard(shard_path, num_seqs, seq_len, dtype_code=1)

        _, _, dtype = read_shard_header(shard_path)
        assert dtype == np.uint32

    def test_invalid_magic_bytes(self, tmp_path):
        """Test that invalid magic bytes raise error."""
        shard_path = tmp_path / "bad.bin"
        with open(shard_path, "wb") as f:
            f.write(b"BADMAGIC")
            f.write(struct.pack("<IIII", 1, 10, 128, 0))

        with pytest.raises(ValueError, match="Invalid shard magic bytes"):
            read_shard_header(shard_path)

    def test_header_size_constant(self):
        """Test header size is correctly calculated."""
        assert HEADER_SIZE == 24  # 8 + 4 + 4 + 4 + 4

    def test_shard_data_integrity(self, tmp_path):
        """Test that shard data can be read back correctly."""
        shard_path = tmp_path / "test.bin"
        num_seqs, seq_len = 10, 128
        original_tokens = write_test_shard(shard_path, num_seqs, seq_len)

        # Read back
        with open(shard_path, "rb") as f:
            f.seek(HEADER_SIZE)
            data = np.frombuffer(f.read(), dtype=np.uint16)
            data = data.reshape(num_seqs, seq_len)

        np.testing.assert_array_equal(data, original_tokens)


class TestManifest:
    """Tests for manifest loading."""

    def test_load_manifest(self, tmp_path):
        """Test loading manifest.json."""
        shard_info = {
            "shard_0.bin": {"path": "shard_0.bin", "hash": "abc123"},
            "shard_1.bin": {"path": "shard_1.bin", "hash": "def456"},
        }
        create_test_manifest(tmp_path, shard_info)

        config = load_manifest(tmp_path / "manifest.json")

        assert config.dataset == "test-dataset"
        assert config.version == "1.0.0"
        assert config.total_shards == 2
        assert config.tokenizer == "gpt2"
        assert len(config.shards) == 2

    def test_manifest_config_defaults(self):
        """Test ManifestConfig default values."""
        config = ManifestConfig()

        assert config.dataset == ""
        assert config.version == ""
        assert config.total_shards == 0
        assert config.tokenizer == "gpt2"
        assert config.max_seq_len == 512


class TestShardMetadata:
    """Tests for ShardMetadata dataclass."""

    def test_shard_metadata_creation(self, tmp_path):
        """Test creating ShardMetadata."""
        path = tmp_path / "test.bin"
        metadata = ShardMetadata(
            path=path,
            num_sequences=100,
            seq_length=512,
            dtype=np.uint16,
            size_bytes=102400,
            hash="abc123",
        )

        assert metadata.path == path
        assert metadata.num_sequences == 100
        assert metadata.seq_length == 512
        assert metadata.dtype == np.uint16


class TestShardedBinaryDataset:
    """Tests for ShardedBinaryDataset."""

    def test_single_shard_dataset(self, tmp_path):
        """Test dataset with single shard."""
        shard_path = tmp_path / "shard_0.bin"
        num_seqs, seq_len = 10, 64
        write_test_shard(shard_path, num_seqs, seq_len)

        dataset = ShardedBinaryDataset(tmp_path, use_mmap=False, shuffle_shards=False)

        assert len(dataset) == num_seqs
        assert dataset.seq_length == seq_len

    def test_dataset_getitem(self, tmp_path):
        """Test __getitem__ returns correct format."""
        shard_path = tmp_path / "shard_0.bin"
        write_test_shard(shard_path, 10, 64)

        dataset = ShardedBinaryDataset(tmp_path, use_mmap=False, shuffle_shards=False)
        sample = dataset[0]

        assert "input_ids" in sample
        assert "labels" in sample
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert sample["input_ids"].shape == (64,)
        assert sample["input_ids"].dtype == torch.int64

    def test_multi_shard_dataset(self, tmp_path):
        """Test dataset with multiple shards."""
        for i in range(3):
            shard_path = tmp_path / f"shard_{i}.bin"
            write_test_shard(shard_path, 10, 64, seed=i)

        dataset = ShardedBinaryDataset(tmp_path, use_mmap=False, shuffle_shards=False)

        assert len(dataset) == 30  # 3 shards * 10 sequences each
        assert len(dataset.shards) == 3

    def test_cross_shard_indexing(self, tmp_path):
        """Test indexing across shard boundaries."""
        # Create 2 shards with 5 sequences each
        for i in range(2):
            shard_path = tmp_path / f"shard_{i}.bin"
            write_test_shard(shard_path, 5, 32, seed=i)

        dataset = ShardedBinaryDataset(tmp_path, use_mmap=False, shuffle_shards=False)

        # Access from first shard
        sample_0 = dataset[0]
        _sample_4 = dataset[4]

        # Access from second shard
        sample_5 = dataset[5]
        _sample_9 = dataset[9]

        assert sample_0["input_ids"].shape == (32,)
        assert sample_5["input_ids"].shape == (32,)

        # Verify different shards have different data
        assert not torch.equal(sample_0["input_ids"], sample_5["input_ids"])

    def test_mmap_mode(self, tmp_path):
        """Test memory-mapped file reading."""
        shard_path = tmp_path / "shard_0.bin"
        write_test_shard(shard_path, 10, 64)

        dataset = ShardedBinaryDataset(tmp_path, use_mmap=True, shuffle_shards=False)
        sample = dataset[0]

        assert sample["input_ids"].shape == (64,)

    def test_total_tokens_property(self, tmp_path):
        """Test total_tokens property."""
        for i in range(2):
            shard_path = tmp_path / f"shard_{i}.bin"
            write_test_shard(shard_path, 10, 64)

        dataset = ShardedBinaryDataset(tmp_path, use_mmap=False, shuffle_shards=False)

        assert dataset.total_tokens == 2 * 10 * 64  # 1280

    def test_with_manifest(self, tmp_path):
        """Test dataset loading with manifest."""
        shard_path = tmp_path / "shard_0.bin"
        write_test_shard(shard_path, 10, 64)

        shard_info = {
            "shard_0.bin": {"path": "shard_0.bin", "hash": "test123"},
        }
        create_test_manifest(tmp_path, shard_info)

        dataset = ShardedBinaryDataset(tmp_path, shuffle_shards=False)

        assert len(dataset) == 10
        assert dataset.manifest.dataset == "test-dataset"

    def test_shuffle_shards_deterministic(self, tmp_path):
        """Test that shard shuffling is deterministic with seed."""
        for i in range(3):
            shard_path = tmp_path / f"shard_{i}.bin"
            write_test_shard(shard_path, 5, 32, seed=i)

        dataset1 = ShardedBinaryDataset(tmp_path, shuffle_shards=True, seed=42)
        dataset2 = ShardedBinaryDataset(tmp_path, shuffle_shards=True, seed=42)

        # Same seed should give same order
        for i in range(len(dataset1)):
            assert torch.equal(dataset1[i]["input_ids"], dataset2[i]["input_ids"])


class TestCreateShardedDataloader:
    """Tests for create_sharded_dataloader function."""

    def test_basic_dataloader(self, tmp_path):
        """Test creating basic dataloader."""
        shard_path = tmp_path / "shard_0.bin"
        write_test_shard(shard_path, 20, 64)

        dataloader = create_sharded_dataloader(
            shard_dir=tmp_path,
            batch_size=4,
            num_workers=0,
            shuffle=False,
        )

        assert dataloader.batch_size == 4

        batch = next(iter(dataloader))
        assert batch["input_ids"].shape == (4, 64)

    def test_dataloader_iteration(self, tmp_path):
        """Test full iteration through dataloader."""
        shard_path = tmp_path / "shard_0.bin"
        write_test_shard(shard_path, 16, 32)

        dataloader = create_sharded_dataloader(
            shard_dir=tmp_path,
            batch_size=4,
            num_workers=0,
            shuffle=False,
        )

        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            assert batch["input_ids"].shape == (4, 32)

        assert batch_count == 4  # 16 samples / 4 batch_size

    def test_distributed_sampler(self, tmp_path):
        """Test dataloader with distributed sampler."""
        shard_path = tmp_path / "shard_0.bin"
        write_test_shard(shard_path, 20, 64)

        # Simulate 2 GPUs
        dl_rank0 = create_sharded_dataloader(
            shard_dir=tmp_path,
            batch_size=4,
            num_workers=0,
            distributed=True,
            rank=0,
            world_size=2,
        )

        dl_rank1 = create_sharded_dataloader(
            shard_dir=tmp_path,
            batch_size=4,
            num_workers=0,
            distributed=True,
            rank=1,
            world_size=2,
        )

        # Each should have same length
        assert len(dl_rank0.dataset) == len(dl_rank1.dataset)


class TestIterBatches:
    """Tests for iter_batches function."""

    def test_basic_iteration(self, tmp_path):
        """Test basic batch iteration."""
        shard_path = tmp_path / "shard_0.bin"
        write_test_shard(shard_path, 16, 32)

        batches = list(iter_batches(tmp_path, batch_size=4, max_batches=2))

        assert len(batches) == 2
        assert batches[0]["input_ids"].shape == (4, 32)

    def test_max_batches_limit(self, tmp_path):
        """Test max_batches limits iteration."""
        shard_path = tmp_path / "shard_0.bin"
        write_test_shard(shard_path, 100, 32)

        batches = list(iter_batches(tmp_path, batch_size=4, max_batches=5))

        assert len(batches) == 5

    def test_device_placement(self, tmp_path):
        """Test device placement (CPU)."""
        shard_path = tmp_path / "shard_0.bin"
        write_test_shard(shard_path, 8, 32)

        batches = list(iter_batches(tmp_path, batch_size=4, device="cpu"))

        assert batches[0]["input_ids"].device.type == "cpu"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_directory(self, tmp_path):
        """Test dataset with no shards."""
        dataset = ShardedBinaryDataset(tmp_path, shuffle_shards=False)

        assert len(dataset) == 0
        assert len(dataset.shards) == 0

    def test_missing_shard_in_manifest(self, tmp_path):
        """Test handling of missing shard referenced in manifest."""
        shard_info = {
            "missing_shard.bin": {"path": "missing_shard.bin"},
        }
        create_test_manifest(tmp_path, shard_info)

        dataset = ShardedBinaryDataset(tmp_path, shuffle_shards=False)

        # Should handle gracefully (shard won't be loaded)
        assert len(dataset.shards) == 0

    def test_corrupted_shard_header(self, tmp_path):
        """Test handling of corrupted shard header."""
        shard_path = tmp_path / "bad_shard.bin"
        with open(shard_path, "wb") as f:
            f.write(SHARD_MAGIC)  # Valid magic
            f.write(b"\x00" * 4)  # Truncated header

        dataset = ShardedBinaryDataset(tmp_path, shuffle_shards=False)

        # Should handle gracefully with warning
        assert len(dataset.shards) == 0

    def test_very_small_shard(self, tmp_path):
        """Test handling of very small shard."""
        shard_path = tmp_path / "small.bin"
        write_test_shard(shard_path, num_sequences=1, seq_length=8)

        dataset = ShardedBinaryDataset(tmp_path, shuffle_shards=False)

        assert len(dataset) == 1
        sample = dataset[0]
        assert sample["input_ids"].shape == (8,)
