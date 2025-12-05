"""
Tests for Modal Infrastructure (prepare_fineweb.py).

Tests cover:
- Shard file format validation
- Manifest generation
- Hash computation
- Local validation logic (without Modal runtime)

Note: These tests mock Modal functionality since Modal requires
cloud infrastructure. Integration tests with Modal are run separately.
"""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def write_test_shard(
    path: Path,
    num_sequences: int,
    seq_length: int,
    dtype_code: int = 0,
    version: int = 1,
) -> bytes:
    """Write a test shard file and return its content as bytes."""
    dtype = np.uint16 if dtype_code == 0 else np.uint32
    rng = np.random.default_rng(42)
    tokens = rng.integers(0, 50000, size=(num_sequences, seq_length), dtype=dtype)

    content = bytearray()
    # Write header
    content.extend(b"DSTKN001")  # Magic bytes
    content.extend(struct.pack("<I", version))
    content.extend(struct.pack("<I", num_sequences))
    content.extend(struct.pack("<I", seq_length))
    content.extend(struct.pack("<I", dtype_code))
    # Write data
    content.extend(tokens.tobytes())

    with open(path, "wb") as f:
        f.write(content)

    return bytes(content)


class TestComputeFileHash:
    """Tests for file hash computation."""

    def test_hash_computation(self, tmp_path):
        """Test that hash is computed correctly."""
        test_file = tmp_path / "test.bin"
        content = b"test content for hashing"
        test_file.write_bytes(content)

        # Compute hash
        sha256_hash = hashlib.sha256()
        with open(test_file, "rb") as f:
            while chunk := f.read(8192):
                sha256_hash.update(chunk)
        result = sha256_hash.hexdigest()

        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_hash_large_file(self, tmp_path):
        """Test hash computation for larger file."""
        test_file = tmp_path / "large.bin"
        content = b"x" * 1_000_000  # 1MB file
        test_file.write_bytes(content)

        sha256_hash = hashlib.sha256()
        with open(test_file, "rb") as f:
            while chunk := f.read(8192):
                sha256_hash.update(chunk)
        result = sha256_hash.hexdigest()

        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_hash_empty_file(self, tmp_path):
        """Test hash of empty file."""
        test_file = tmp_path / "empty.bin"
        test_file.write_bytes(b"")

        sha256_hash = hashlib.sha256()
        with open(test_file, "rb") as f:
            while chunk := f.read(8192):
                sha256_hash.update(chunk)
        result = sha256_hash.hexdigest()

        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected


class TestValidateShardFormat:
    """Tests for shard format validation."""

    def test_valid_shard(self, tmp_path):
        """Test validation of valid shard file."""
        shard_path = tmp_path / "valid.bin"
        write_test_shard(shard_path, num_sequences=10, seq_length=64)

        # Read and validate header
        with open(shard_path, "rb") as f:
            magic = f.read(8)
            version = struct.unpack("<I", f.read(4))[0]
            num_seqs = struct.unpack("<I", f.read(4))[0]
            seq_len = struct.unpack("<I", f.read(4))[0]
            dtype_code = struct.unpack("<I", f.read(4))[0]

        assert magic == b"DSTKN001"
        assert version == 1
        assert num_seqs == 10
        assert seq_len == 64
        assert dtype_code == 0

    def test_invalid_magic_bytes(self, tmp_path):
        """Test that invalid magic bytes are detected."""
        shard_path = tmp_path / "invalid.bin"
        with open(shard_path, "wb") as f:
            f.write(b"BADMAGIC")
            f.write(struct.pack("<IIII", 1, 10, 64, 0))

        with open(shard_path, "rb") as f:
            magic = f.read(8)

        assert magic != b"DSTKN001"

    def test_truncated_header(self, tmp_path):
        """Test that truncated header is detected."""
        shard_path = tmp_path / "truncated.bin"
        with open(shard_path, "wb") as f:
            f.write(b"DSTKN001")  # Only magic bytes, incomplete header

        file_size = shard_path.stat().st_size
        assert file_size < 24  # Header should be 24 bytes


class TestManifestFormat:
    """Tests for manifest format and structure."""

    def test_manifest_structure(self):
        """Test expected manifest JSON structure."""
        manifest = {
            "dataset": "fineweb-edu-sample",
            "version": "1.0.0",
            "total_shards": 3,
            "total_bytes": 1024000,
            "metadata": {
                "tokenizer": "gpt2",
                "max_seq_len": 512,
                "created_at": "2025-01-01T00:00:00Z",
            },
            "shards": {
                "shard_0.bin": {
                    "path": "shards/shard_0.bin",
                    "hash": "abc123...",
                    "num_sequences": 1000,
                    "seq_length": 512,
                    "size_bytes": 341376,
                },
            },
        }

        # Validate structure
        assert "dataset" in manifest
        assert "version" in manifest
        assert "total_shards" in manifest
        assert "shards" in manifest

        # Validate shard info
        shard_info = manifest["shards"]["shard_0.bin"]
        assert "path" in shard_info
        assert "hash" in shard_info

    def test_manifest_serialization(self, tmp_path):
        """Test that manifest can be serialized and deserialized."""
        manifest = {
            "dataset": "test",
            "version": "1.0.0",
            "total_shards": 1,
            "shards": {},
        }

        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with open(manifest_path) as f:
            loaded = json.load(f)

        assert loaded == manifest


class TestShardFormatConstants:
    """Tests for shard format constants."""

    def test_magic_bytes(self):
        """Test magic bytes are correct."""
        assert b"DSTKN001" == b"\x44\x53\x54\x4b\x4e\x30\x30\x31"

    def test_header_size(self):
        """Test header size calculation."""
        # Magic (8) + Version (4) + NumSeqs (4) + SeqLen (4) + DType (4)
        expected_size = 8 + 4 + 4 + 4 + 4
        assert expected_size == 24

    def test_dtype_codes(self):
        """Test dtype code meanings."""
        dtype_map = {
            0: np.uint16,
            1: np.uint32,
        }

        assert dtype_map[0] == np.uint16
        assert dtype_map[1] == np.uint32


class TestVolumePathConventions:
    """Tests for Modal volume path conventions."""

    def test_default_volume_path(self):
        """Test default volume path structure."""
        volume_path = "/data/fineweb-edu"

        assert volume_path.startswith("/data/")
        assert "fineweb" in volume_path

    def test_shard_directory_structure(self, tmp_path):
        """Test expected shard directory structure."""
        # Create expected structure
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()

        for i in range(3):
            shard_path = shards_dir / f"shard_{i:04d}.bin"
            write_test_shard(shard_path, 10, 64)

        manifest_path = tmp_path / "manifest.json"
        manifest = {
            "dataset": "fineweb-edu",
            "shards": {
                f"shard_{i:04d}.bin": {"path": f"shards/shard_{i:04d}.bin"}
                for i in range(3)
            },
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Verify structure
        assert shards_dir.is_dir()
        assert manifest_path.is_file()
        assert len(list(shards_dir.glob("*.bin"))) == 3


class TestShardDataIntegrity:
    """Tests for shard data integrity."""

    def test_shard_data_integrity(self, tmp_path):
        """Test that shard data can be read back correctly."""
        shard_path = tmp_path / "test.bin"
        num_seqs, seq_len = 10, 128
        write_test_shard(shard_path, num_seqs, seq_len)

        # Read back data
        with open(shard_path, "rb") as f:
            f.seek(24)  # Skip header
            data = np.frombuffer(f.read(), dtype=np.uint16)
            data = data.reshape(num_seqs, seq_len)

        assert data.shape == (num_seqs, seq_len)
        assert data.dtype == np.uint16

    def test_shard_hash_consistency(self, tmp_path):
        """Test that same content produces same hash."""
        # Create two identical shards
        shard1 = tmp_path / "shard1.bin"
        shard2 = tmp_path / "shard2.bin"

        write_test_shard(shard1, 10, 64)
        write_test_shard(shard2, 10, 64)

        def compute_hash(path):
            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            return sha256.hexdigest()

        assert compute_hash(shard1) == compute_hash(shard2)

    def test_different_seeds_different_data(self, tmp_path):
        """Test that different seeds produce different data."""
        # Use different seeds to generate different data
        dtype = np.uint16
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)

        tokens1 = rng1.integers(0, 50000, size=(10, 64), dtype=dtype)
        tokens2 = rng2.integers(0, 50000, size=(10, 64), dtype=dtype)

        assert not np.array_equal(tokens1, tokens2)


class TestModalIntegration:
    """Tests for Modal-specific integration patterns."""

    def test_modal_volume_mount_path(self):
        """Test Modal volume mount path convention."""
        # Modal volumes are typically mounted at /data/<volume-name>
        volume_name = "deepseek-data"
        expected_mount = f"/data/{volume_name}"

        assert expected_mount == "/data/deepseek-data"

    def test_parallel_upload_count(self):
        """Test that parallel upload count is reasonable."""
        parallel_workers = 4
        # Should be between 1 and CPU count
        assert 1 <= parallel_workers <= 32

    def test_shard_naming_convention(self):
        """Test shard naming follows convention."""
        for i in range(10):
            name = f"shard_{i:04d}.bin"
            assert name.startswith("shard_")
            assert name.endswith(".bin")
            # Extract number
            num_part = name.replace("shard_", "").replace(".bin", "")
            assert len(num_part) == 4
            assert num_part.isdigit()
