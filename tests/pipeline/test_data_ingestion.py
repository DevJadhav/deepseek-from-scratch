"""
Comprehensive tests for the Phase 1 Data Ingestion Pipeline.

Tests cover:
- DeterministicShuffler: Reproducibility, worker isolation, epoch handling
- StreamingDataPipeline: HuggingFace integration (mocked)
- TokenLevelBatcher: Token batching, packing, padding
- DynamicPadder: Bucket-based padding, efficiency
- DataIngestionPipeline: End-to-end integration

Reference: production_hardening.md Section 3.1
"""

from __future__ import annotations

import numpy as np
import pytest

from deepseek.pipeline.data_ingestion import (
    DeterministicShuffler,
    DynamicPadder,
    DynamicPaddingConfig,
    ShuffleState,
    TokenBatcherConfig,
    TokenLevelBatcher,
)


# =============================================================================
# DeterministicShuffler Tests
# =============================================================================


class TestDeterministicShuffler:
    """Tests for DeterministicShuffler class."""

    def test_initialization(self):
        """Test shuffler initialization."""
        shuffler = DeterministicShuffler(seed=42, num_workers=4)
        assert shuffler.seed == 42
        assert shuffler.num_workers == 4
        assert shuffler.buffer_size == 10000
        assert len(shuffler._worker_seeds) == 4

    def test_deterministic_same_seed_same_result(self):
        """Same seed should produce identical shuffle order."""
        shuffler1 = DeterministicShuffler(seed=42, num_workers=4)
        shuffler2 = DeterministicShuffler(seed=42, num_workers=4)

        data = list(range(1000))

        result1 = shuffler1.shuffle_for_worker(data, worker_id=0)
        result2 = shuffler2.shuffle_for_worker(data, worker_id=0)

        assert result1 == result2

    def test_different_seeds_different_results(self):
        """Different seeds should produce different shuffles."""
        shuffler1 = DeterministicShuffler(seed=42, num_workers=4)
        shuffler2 = DeterministicShuffler(seed=123, num_workers=4)

        data = list(range(1000))

        result1 = shuffler1.shuffle_for_worker(data, worker_id=0)
        result2 = shuffler2.shuffle_for_worker(data, worker_id=0)

        assert result1 != result2

    def test_different_workers_different_shuffles(self):
        """Different workers should get different shuffles."""
        shuffler = DeterministicShuffler(seed=42, num_workers=4)
        data = list(range(1000))

        result0 = shuffler.shuffle_for_worker(data, worker_id=0)
        result1 = shuffler.shuffle_for_worker(data, worker_id=1)
        result2 = shuffler.shuffle_for_worker(data, worker_id=2)

        assert result0 != result1
        assert result1 != result2
        assert result0 != result2

    def test_same_worker_same_epoch_reproducible(self):
        """Same worker and epoch should be reproducible across calls."""
        shuffler = DeterministicShuffler(seed=42, num_workers=4)
        data = list(range(1000))

        result1 = shuffler.shuffle_for_worker(data, worker_id=1, epoch=5)
        result2 = shuffler.shuffle_for_worker(data, worker_id=1, epoch=5)

        assert result1 == result2

    def test_different_epochs_different_shuffles(self):
        """Different epochs should produce different shuffles."""
        shuffler = DeterministicShuffler(seed=42, num_workers=4)
        data = list(range(1000))

        result_epoch0 = shuffler.shuffle_for_worker(data, worker_id=0, epoch=0)
        result_epoch1 = shuffler.shuffle_for_worker(data, worker_id=0, epoch=1)
        result_epoch2 = shuffler.shuffle_for_worker(data, worker_id=0, epoch=2)

        assert result_epoch0 != result_epoch1
        assert result_epoch1 != result_epoch2

    def test_shuffle_preserves_elements(self):
        """Shuffle should preserve all elements."""
        shuffler = DeterministicShuffler(seed=42, num_workers=4)
        data = list(range(100))

        result = shuffler.shuffle_for_worker(data, worker_id=0)

        assert sorted(result) == data
        assert len(result) == len(data)

    def test_shuffle_indices(self):
        """Test shuffle_indices method."""
        shuffler = DeterministicShuffler(seed=42, num_workers=4)

        indices = shuffler.shuffle_indices(length=100, worker_id=0)

        assert len(indices) == 100
        assert sorted(indices) == list(range(100))

    def test_get_worker_shard_even_division(self):
        """Test worker sharding with even division."""
        shuffler = DeterministicShuffler(seed=42, num_workers=4)

        # 100 items / 4 workers = 25 each
        start0, end0 = shuffler.get_worker_shard(100, worker_id=0)
        start1, end1 = shuffler.get_worker_shard(100, worker_id=1)
        start2, end2 = shuffler.get_worker_shard(100, worker_id=2)
        start3, end3 = shuffler.get_worker_shard(100, worker_id=3)

        assert (start0, end0) == (0, 25)
        assert (start1, end1) == (25, 50)
        assert (start2, end2) == (50, 75)
        assert (start3, end3) == (75, 100)

    def test_get_worker_shard_with_remainder(self):
        """Test worker sharding with remainder distribution."""
        shuffler = DeterministicShuffler(seed=42, num_workers=4)

        # 103 items / 4 workers = 25 each + 3 remainder
        # Workers 0, 1, 2 get 26, worker 3 gets 25
        start0, end0 = shuffler.get_worker_shard(103, worker_id=0)
        start1, end1 = shuffler.get_worker_shard(103, worker_id=1)
        start2, end2 = shuffler.get_worker_shard(103, worker_id=2)
        start3, end3 = shuffler.get_worker_shard(103, worker_id=3)

        assert end0 - start0 == 26  # Worker 0 gets extra
        assert end1 - start1 == 26  # Worker 1 gets extra
        assert end2 - start2 == 26  # Worker 2 gets extra
        assert end3 - start3 == 25  # Worker 3 gets base
        assert end3 == 103  # All covered

    def test_streaming_shuffle_deterministic(self):
        """Test streaming shuffle is deterministic."""
        shuffler = DeterministicShuffler(seed=42, num_workers=4, buffer_size=50)

        data = list(range(200))

        result1 = list(shuffler.streaming_shuffle(iter(data), worker_id=0, epoch=0))
        result2 = list(shuffler.streaming_shuffle(iter(data), worker_id=0, epoch=0))

        assert result1 == result2

    def test_streaming_shuffle_preserves_elements(self):
        """Test streaming shuffle preserves all elements."""
        shuffler = DeterministicShuffler(seed=42, num_workers=4, buffer_size=50)

        data = list(range(200))
        result = list(shuffler.streaming_shuffle(iter(data), worker_id=0))

        assert sorted(result) == data

    def test_invalid_worker_id_raises(self):
        """Test that invalid worker_id raises error."""
        shuffler = DeterministicShuffler(seed=42, num_workers=4)

        with pytest.raises(ValueError, match="worker_id 5 >= num_workers 4"):
            shuffler.shuffle_for_worker([1, 2, 3], worker_id=5)

    def test_state_serialization(self):
        """Test state can be serialized and restored."""
        shuffler = DeterministicShuffler(seed=42, num_workers=4)

        state = shuffler.get_state(worker_id=1)
        state_dict = state.to_dict()

        restored_state = ShuffleState.from_dict(state_dict)

        assert restored_state.seed == state.seed
        assert restored_state.worker_id == state.worker_id
        assert restored_state.epoch == state.epoch

    def test_advance_epoch(self):
        """Test epoch advancement."""
        shuffler = DeterministicShuffler(seed=42, num_workers=4)

        assert shuffler.get_state(worker_id=0).epoch == 0

        shuffler.advance_epoch(worker_id=0)
        assert shuffler.get_state(worker_id=0).epoch == 1

        shuffler.advance_epoch(worker_id=0)
        assert shuffler.get_state(worker_id=0).epoch == 2


# =============================================================================
# TokenLevelBatcher Tests
# =============================================================================


class TestTokenLevelBatcher:
    """Tests for TokenLevelBatcher class."""

    def test_initialization(self):
        """Test batcher initialization."""
        config = TokenBatcherConfig(target_tokens_per_batch=1000)
        batcher = TokenLevelBatcher(config)

        assert batcher.config.target_tokens_per_batch == 1000

    def test_packed_batch_structure(self):
        """Test packed batch has correct structure."""
        config = TokenBatcherConfig(
            target_tokens_per_batch=500,
            max_seq_length=100,
            min_seq_length=10,
            pad_to_multiple=8,
        )
        batcher = TokenLevelBatcher(config)

        sequences = [
            np.arange(50, dtype=np.int64),
            np.arange(30, dtype=np.int64),
            np.arange(40, dtype=np.int64),
        ]

        batches = list(batcher.batch(iter(sequences), pack=True))

        assert len(batches) == 1
        batch = batches[0]

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "position_ids" in batch
        assert "doc_ids" in batch
        assert "num_documents" in batch
        assert "total_tokens" in batch

        assert batch["num_documents"] == 3
        assert batch["total_tokens"] == 120  # 50 + 30 + 40

    def test_padded_batch_structure(self):
        """Test padded (non-packed) batch has correct structure."""
        config = TokenBatcherConfig(
            target_tokens_per_batch=500,
            max_seq_length=100,
            min_seq_length=10,
            pad_to_multiple=8,
        )
        batcher = TokenLevelBatcher(config)

        sequences = [
            np.arange(50, dtype=np.int64),
            np.arange(30, dtype=np.int64),
        ]

        batches = list(batcher.batch(iter(sequences), pack=False))

        assert len(batches) == 1
        batch = batches[0]

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == 2  # 2 sequences
        assert batch["input_ids"].shape[1] % 8 == 0  # Padded to multiple

    def test_target_tokens_per_batch(self):
        """Test batching respects target tokens per batch."""
        config = TokenBatcherConfig(
            target_tokens_per_batch=200,
            max_seq_length=100,
            min_seq_length=10,
            max_samples_per_batch=100,
        )
        batcher = TokenLevelBatcher(config)

        # Create sequences that would exceed target
        sequences = [np.arange(60, dtype=np.int64) for _ in range(10)]

        batches = list(batcher.batch(iter(sequences), pack=True))

        # Should create multiple batches
        assert len(batches) > 1

    def test_min_seq_length_filtering(self):
        """Test sequences shorter than min_seq_length are filtered."""
        config = TokenBatcherConfig(
            target_tokens_per_batch=1000,
            max_seq_length=100,
            min_seq_length=50,
        )
        batcher = TokenLevelBatcher(config)

        sequences = [
            np.arange(60, dtype=np.int64),  # Included
            np.arange(20, dtype=np.int64),  # Filtered (< 50)
            np.arange(70, dtype=np.int64),  # Included
        ]

        batches = list(batcher.batch(iter(sequences), pack=True))

        assert len(batches) == 1
        assert batches[0]["num_documents"] == 2  # Only 2 sequences included

    def test_max_seq_length_truncation(self):
        """Test sequences longer than max_seq_length are truncated."""
        config = TokenBatcherConfig(
            target_tokens_per_batch=1000,
            max_seq_length=50,
            min_seq_length=10,
        )
        batcher = TokenLevelBatcher(config)

        sequences = [np.arange(100, dtype=np.int64)]  # Too long

        batches = list(batcher.batch(iter(sequences), pack=True))

        assert batches[0]["total_tokens"] == 50  # Truncated

    def test_max_samples_per_batch(self):
        """Test max_samples_per_batch limit."""
        config = TokenBatcherConfig(
            target_tokens_per_batch=10000,  # High to not trigger token limit
            max_seq_length=100,
            min_seq_length=10,
            max_samples_per_batch=3,
        )
        batcher = TokenLevelBatcher(config)

        sequences = [np.arange(20, dtype=np.int64) for _ in range(10)]

        batches = list(batcher.batch(iter(sequences), pack=True))

        # Each batch should have at most 3 documents
        for batch in batches[:-1]:  # All but potentially last
            assert batch["num_documents"] <= 3

    def test_packed_position_ids_reset_per_document(self):
        """Test position_ids reset for each document in packed batch."""
        config = TokenBatcherConfig(
            target_tokens_per_batch=1000,
            min_seq_length=10,
        )
        batcher = TokenLevelBatcher(config)

        sequences = [
            np.arange(30, dtype=np.int64),
            np.arange(20, dtype=np.int64),
        ]

        batches = list(batcher.batch(iter(sequences), pack=True))
        batch = batches[0]

        # First doc: positions 0-29
        # Second doc: positions 0-19
        assert batch["position_ids"][0] == 0
        assert batch["position_ids"][29] == 29
        assert batch["position_ids"][30] == 0  # Reset for second doc
        assert batch["position_ids"][49] == 19

    def test_doc_ids_track_document_boundaries(self):
        """Test doc_ids correctly identify document boundaries."""
        config = TokenBatcherConfig(
            target_tokens_per_batch=1000,
            min_seq_length=10,
        )
        batcher = TokenLevelBatcher(config)

        sequences = [
            np.arange(30, dtype=np.int64),
            np.arange(20, dtype=np.int64),
        ]

        batches = list(batcher.batch(iter(sequences), pack=True))
        batch = batches[0]

        # First 30 tokens should have doc_id=0
        assert all(batch["doc_ids"][:30] == 0)
        # Next 20 tokens should have doc_id=1
        assert all(batch["doc_ids"][30:50] == 1)

    def test_empty_iterator(self):
        """Test handling of empty iterator."""
        config = TokenBatcherConfig()
        batcher = TokenLevelBatcher(config)

        batches = list(batcher.batch(iter([]), pack=True))

        assert len(batches) == 0


# =============================================================================
# DynamicPadder Tests
# =============================================================================


class TestDynamicPadder:
    """Tests for DynamicPadder class."""

    def test_initialization(self):
        """Test padder initialization."""
        config = DynamicPaddingConfig(bucket_boundaries=[64, 128, 256, 512])
        padder = DynamicPadder(config)

        assert padder.buckets == [64, 128, 256, 512]

    def test_bucket_assignment(self):
        """Test sequences are assigned to correct buckets."""
        config = DynamicPaddingConfig(bucket_boundaries=[64, 128, 256, 512])
        padder = DynamicPadder(config)

        assert padder._get_bucket(50) == 0  # <= 64
        assert padder._get_bucket(64) == 0  # == 64
        assert padder._get_bucket(65) == 1  # <= 128
        assert padder._get_bucket(128) == 1  # == 128
        assert padder._get_bucket(200) == 2  # <= 256
        assert padder._get_bucket(600) == 4  # > 512 (overflow)

    def test_bucket_size(self):
        """Test bucket size retrieval."""
        config = DynamicPaddingConfig(bucket_boundaries=[64, 128, 256, 512])
        padder = DynamicPadder(config)

        assert padder._get_bucket_size(0) == 64
        assert padder._get_bucket_size(1) == 128
        assert padder._get_bucket_size(2) == 256
        assert padder._get_bucket_size(3) == 512
        assert padder._get_bucket_size(4) == 512  # Overflow returns max

    def test_pad_sequence(self):
        """Test single sequence padding."""
        config = DynamicPaddingConfig(
            bucket_boundaries=[64, 128, 256],
            pad_token_id=0,
        )
        padder = DynamicPadder(config)

        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        padded, mask = padder.pad_sequence(tokens)

        assert len(padded) == 64  # Bucket for len=5
        assert padded[:5].tolist() == [1, 2, 3, 4, 5]
        assert all(padded[5:] == 0)
        assert mask[:5].tolist() == [1, 1, 1, 1, 1]
        assert all(mask[5:] == 0)

    def test_pad_sequence_with_target_length(self):
        """Test padding with explicit target length."""
        config = DynamicPaddingConfig(pad_token_id=0)
        padder = DynamicPadder(config)

        tokens = np.array([1, 2, 3], dtype=np.int64)
        padded, mask = padder.pad_sequence(tokens, target_length=10)

        assert len(padded) == 10
        assert padded[:3].tolist() == [1, 2, 3]
        assert all(padded[3:] == 0)

    def test_pad_sequence_truncation(self):
        """Test padding truncates when sequence exceeds target."""
        config = DynamicPaddingConfig(pad_token_id=0)
        padder = DynamicPadder(config)

        tokens = np.arange(100, dtype=np.int64)
        padded, mask = padder.pad_sequence(tokens, target_length=50)

        assert len(padded) == 50
        assert padded.tolist() == list(range(50))
        assert all(mask == 1)

    def test_pad_batch(self):
        """Test batch padding."""
        config = DynamicPaddingConfig(
            bucket_boundaries=[64, 128, 256],
            pad_token_id=0,
        )
        padder = DynamicPadder(config)

        sequences = [
            np.array([1, 2, 3], dtype=np.int64),
            np.array([4, 5, 6, 7, 8], dtype=np.int64),
            np.array([9, 10], dtype=np.int64),
        ]

        batch = padder.pad_batch(sequences)

        assert batch["input_ids"].shape == (3, 64)  # Batch size 3, bucket 64
        assert batch["attention_mask"].shape == (3, 64)

        # Check first sequence
        assert batch["input_ids"][0, :3].tolist() == [1, 2, 3]
        assert batch["attention_mask"][0, :3].tolist() == [1, 1, 1]
        assert all(batch["attention_mask"][0, 3:] == 0)

    def test_pad_batch_empty(self):
        """Test padding empty batch."""
        config = DynamicPaddingConfig()
        padder = DynamicPadder(config)

        batch = padder.pad_batch([])

        assert batch["input_ids"].size == 0
        assert batch["attention_mask"].size == 0

    def test_efficiency_tracking(self):
        """Test padding efficiency tracking."""
        config = DynamicPaddingConfig(
            bucket_boundaries=[64, 128, 256],
            pad_token_id=0,
        )
        padder = DynamicPadder(config)

        # Start with perfect efficiency
        assert padder.efficiency == 1.0

        # Pad a short sequence
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        padder.pad_sequence(tokens)

        stats = padder.get_stats()
        assert stats["total_tokens"] == 5
        assert stats["padded_tokens"] == 59  # 64 - 5
        assert stats["efficiency"] < 1.0

    def test_reset_stats(self):
        """Test stats reset."""
        config = DynamicPaddingConfig()
        padder = DynamicPadder(config)

        # Do some padding
        padder.pad_sequence(np.array([1, 2, 3], dtype=np.int64))

        # Reset
        padder.reset_stats()

        stats = padder.get_stats()
        assert stats["total_tokens"] == 0
        assert stats["padded_tokens"] == 0
        assert stats["batches_created"] == 0

    def test_bucketed_batches(self):
        """Test bucket-based batch creation."""
        config = DynamicPaddingConfig(
            bucket_boundaries=[64, 128, 256],
            pad_token_id=0,
        )
        padder = DynamicPadder(config)

        # Create sequences of varying lengths
        sequences = [
            np.arange(30, dtype=np.int64),  # Bucket 0 (<=64)
            np.arange(50, dtype=np.int64),  # Bucket 0 (<=64)
            np.arange(80, dtype=np.int64),  # Bucket 1 (<=128)
            np.arange(100, dtype=np.int64),  # Bucket 1 (<=128)
        ]

        batches = list(padder.create_bucketed_batches(iter(sequences), batch_size=2))

        # Should create 2 batches: one for bucket 0, one for bucket 1
        assert len(batches) == 2

    def test_bucketed_batches_respects_batch_size(self):
        """Test bucketed batches respect batch size."""
        config = DynamicPaddingConfig(
            bucket_boundaries=[64, 128, 256],
            pad_token_id=0,
        )
        padder = DynamicPadder(config)

        # Create many sequences in same bucket
        sequences = [np.arange(30, dtype=np.int64) for _ in range(10)]

        batches = list(padder.create_bucketed_batches(iter(sequences), batch_size=3))

        # First 3 batches should have 3 items, last batch has 1
        assert batches[0]["input_ids"].shape[0] == 3
        assert batches[1]["input_ids"].shape[0] == 3
        assert batches[2]["input_ids"].shape[0] == 3
        assert batches[3]["input_ids"].shape[0] == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestDataIngestionIntegration:
    """Integration tests for the full pipeline."""

    def test_shuffler_with_batcher(self):
        """Test DeterministicShuffler integrates with TokenLevelBatcher."""
        shuffler = DeterministicShuffler(seed=42, num_workers=2)
        batcher_config = TokenBatcherConfig(
            target_tokens_per_batch=500,
            min_seq_length=10,
        )
        batcher = TokenLevelBatcher(batcher_config)

        # Create data
        data = [np.arange(50 + i, dtype=np.int64) for i in range(20)]

        # Shuffle for worker 0
        shuffled = shuffler.shuffle_for_worker(data, worker_id=0)

        # Batch the shuffled data
        batches = list(batcher.batch(iter(shuffled), pack=True))

        assert len(batches) > 0
        # Verify total tokens are correct
        total = sum(b["total_tokens"] for b in batches)
        expected = sum(50 + i for i in range(20))
        assert total == expected

    def test_shuffler_epoch_variation_with_batcher(self):
        """Test different epochs produce different batch orders."""
        shuffler = DeterministicShuffler(seed=42, num_workers=2)
        batcher_config = TokenBatcherConfig(
            target_tokens_per_batch=300,
            min_seq_length=10,
        )
        batcher = TokenLevelBatcher(batcher_config)

        data = [np.arange(50 + i, dtype=np.int64) for i in range(10)]

        # Epoch 0
        shuffled_e0 = shuffler.shuffle_for_worker(data, worker_id=0, epoch=0)
        batches_e0 = list(batcher.batch(iter(shuffled_e0), pack=True))

        # Epoch 1
        shuffled_e1 = shuffler.shuffle_for_worker(data, worker_id=0, epoch=1)
        batches_e1 = list(batcher.batch(iter(shuffled_e1), pack=True))

        # Should have same number of batches but different order
        assert len(batches_e0) == len(batches_e1)

        # First batch content should differ
        first_ids_e0 = batches_e0[0]["input_ids"]
        first_ids_e1 = batches_e1[0]["input_ids"]
        assert not np.array_equal(first_ids_e0, first_ids_e1)


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_single_item_shuffle(self):
        """Test shuffling single item."""
        shuffler = DeterministicShuffler(seed=42, num_workers=1)
        data = [42]

        result = shuffler.shuffle_for_worker(data, worker_id=0)

        assert result == [42]

    def test_empty_data_shuffle(self):
        """Test shuffling empty list."""
        shuffler = DeterministicShuffler(seed=42, num_workers=1)

        result = shuffler.shuffle_for_worker([], worker_id=0)

        assert result == []

    def test_single_worker(self):
        """Test single worker configuration."""
        shuffler = DeterministicShuffler(seed=42, num_workers=1)
        data = list(range(100))

        result = shuffler.shuffle_for_worker(data, worker_id=0)

        assert len(result) == 100
        assert sorted(result) == data

    def test_many_workers(self):
        """Test many workers configuration."""
        shuffler = DeterministicShuffler(seed=42, num_workers=100)
        data = list(range(1000))

        # Each worker should get different shuffle
        results = [
            shuffler.shuffle_for_worker(data, worker_id=i)
            for i in range(10)  # Test first 10
        ]

        # All should be valid permutations
        for result in results:
            assert sorted(result) == data

        # All should be different
        for i, r1 in enumerate(results):
            for j, r2 in enumerate(results):
                if i != j:
                    assert r1 != r2

    def test_large_epoch_number(self):
        """Test large epoch numbers don't overflow."""
        shuffler = DeterministicShuffler(seed=42, num_workers=4)
        data = list(range(100))

        # Should not raise
        result = shuffler.shuffle_for_worker(data, worker_id=0, epoch=1000000)

        assert len(result) == 100

    def test_batcher_single_sequence(self):
        """Test batching single sequence."""
        config = TokenBatcherConfig(
            target_tokens_per_batch=1000,
            min_seq_length=10,
        )
        batcher = TokenLevelBatcher(config)

        sequences = [np.arange(50, dtype=np.int64)]

        batches = list(batcher.batch(iter(sequences), pack=True))

        assert len(batches) == 1
        assert batches[0]["num_documents"] == 1

    def test_padder_very_long_sequence(self):
        """Test padding very long sequence (overflow bucket)."""
        config = DynamicPaddingConfig(
            bucket_boundaries=[64, 128, 256],
            pad_token_id=0,
        )
        padder = DynamicPadder(config)

        tokens = np.arange(1000, dtype=np.int64)
        padded, mask = padder.pad_sequence(tokens, target_length=1000)

        assert len(padded) == 1000
        assert all(mask == 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
