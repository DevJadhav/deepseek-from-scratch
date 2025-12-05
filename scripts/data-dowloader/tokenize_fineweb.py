#!/usr/bin/env python3
"""
Tokenize FineWeb-Edu Dataset
============================

Tokenize the downloaded FineWeb-Edu dataset into binary shards for efficient
training. Uses tiktoken for fast tokenization with GPT-2 encoding.

Usage:
    python scripts/data/tokenize_fineweb.py --input ./data/fineweb-edu --output ./data/fineweb-tokenized
    python scripts/data/tokenize_fineweb.py --input ./data/fineweb-edu --output ./data/fineweb-tokenized --tokenizer gpt2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers import AutoTokenizer

console = Console()

# Magic bytes for binary shard format
SHARD_MAGIC = b"DSTKN001"  # DeepSeek Token v001


def tokenize_shard(
    shard_path: Path,
    tokenizer_name: str,
    max_seq_len: int,
) -> tuple[list[np.ndarray], int, int]:
    """
    Tokenize a single JSONL shard file.

    Args:
        shard_path: Path to the JSONL shard file
        tokenizer_name: Name of the tokenizer to use
        max_seq_len: Maximum sequence length for packing

    Returns:
        Tuple of (token_sequences, total_tokens, num_samples)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sequences = []
    total_tokens = 0
    num_samples = 0
    current_seq = []

    with open(shard_path) as f:
        for line in f:
            try:
                sample = json.loads(line)
                text = sample.get("text", "")
                if not text:
                    continue

                # Tokenize
                tokens = tokenizer.encode(text, add_special_tokens=False)
                num_samples += 1

                # Sequence packing: append tokens to current sequence
                current_seq.extend(tokens)
                current_seq.append(tokenizer.eos_token_id)

                # Split into max_seq_len chunks
                while len(current_seq) >= max_seq_len:
                    seq = current_seq[:max_seq_len]
                    sequences.append(np.array(seq, dtype=np.uint16))
                    total_tokens += len(seq)
                    current_seq = current_seq[max_seq_len:]

            except json.JSONDecodeError:
                continue

    # Handle remaining tokens
    if current_seq:
        # Pad last sequence if needed
        if len(current_seq) < max_seq_len:
            current_seq.extend([tokenizer.pad_token_id] * (max_seq_len - len(current_seq)))
        sequences.append(np.array(current_seq[:max_seq_len], dtype=np.uint16))
        total_tokens += len(current_seq[:max_seq_len])

    return sequences, total_tokens, num_samples


def write_binary_shard(
    output_path: Path,
    sequences: list[np.ndarray],
    tokenizer_name: str,
    max_seq_len: int,
) -> tuple[int, str]:
    """
    Write tokenized sequences to a binary shard file.

    Binary format:
    - 8 bytes: Magic bytes "DSTKN001"
    - 4 bytes: Max sequence length (uint32)
    - 4 bytes: Number of sequences (uint32)
    - 4 bytes: Tokenizer name length (uint32)
    - N bytes: Tokenizer name (utf-8)
    - Sequences: Each sequence is max_seq_len * 2 bytes (uint16 tokens)

    Returns:
        Tuple of (bytes_written, checksum)
    """
    tokenizer_bytes = tokenizer_name.encode("utf-8")

    with open(output_path, "wb") as f:
        # Write header
        f.write(SHARD_MAGIC)
        f.write(struct.pack("<I", max_seq_len))
        f.write(struct.pack("<I", len(sequences)))
        f.write(struct.pack("<I", len(tokenizer_bytes)))
        f.write(tokenizer_bytes)

        # Write sequences
        for seq in sequences:
            f.write(seq.tobytes())

    # Compute checksum
    sha256 = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)

    return output_path.stat().st_size, sha256.hexdigest()


def tokenize_dataset(
    input_dir: Path,
    output_dir: Path,
    tokenizer_name: str = "gpt2",
    max_seq_len: int = 512,
    num_workers: int = 4,
    tokens_per_shard: int = 100_000_000,  # 100M tokens per output shard
) -> dict:
    """
    Tokenize the entire FineWeb-Edu dataset.

    Args:
        input_dir: Directory containing downloaded JSONL shards
        output_dir: Directory for tokenized binary shards
        tokenizer_name: HuggingFace tokenizer name
        max_seq_len: Maximum sequence length
        num_workers: Number of parallel workers
        tokens_per_shard: Target tokens per output shard

    Returns:
        Dictionary with tokenization statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find input shards
    train_dir = input_dir / "train"
    if not train_dir.exists():
        train_dir = input_dir

    input_shards = sorted(train_dir.glob("shard_*.jsonl"))
    if not input_shards:
        console.print(f"[red]No shards found in {train_dir}[/red]")
        sys.exit(1)

    console.print(f"[bold blue]Tokenizing FineWeb-Edu[/bold blue]")
    console.print(f"Input: {input_dir}")
    console.print(f"Output: {output_dir}")
    console.print(f"Tokenizer: {tokenizer_name}")
    console.print(f"Max sequence length: {max_seq_len}")
    console.print(f"Input shards: {len(input_shards)}")

    stats = {
        "total_tokens": 0,
        "total_samples": 0,
        "output_shards": 0,
        "bytes_written": 0,
    }

    all_sequences = []
    output_shard_idx = 0
    shard_metadata = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Tokenizing shards...", total=len(input_shards))

        # Process shards in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(tokenize_shard, shard, tokenizer_name, max_seq_len): shard
                for shard in input_shards
            }

            for future in as_completed(futures):
                shard_path = futures[future]
                try:
                    sequences, tokens, samples = future.result()
                    all_sequences.extend(sequences)
                    stats["total_tokens"] += tokens
                    stats["total_samples"] += samples

                    # Write output shards when we have enough tokens
                    while len(all_sequences) * max_seq_len >= tokens_per_shard:
                        seqs_to_write = all_sequences[: tokens_per_shard // max_seq_len]
                        all_sequences = all_sequences[tokens_per_shard // max_seq_len :]

                        output_path = output_dir / f"train_{output_shard_idx:05d}.bin"
                        bytes_written, checksum = write_binary_shard(
                            output_path,
                            seqs_to_write,
                            tokenizer_name,
                            max_seq_len,
                        )

                        shard_metadata.append(
                            {
                                "path": output_path.name,
                                "sequences": len(seqs_to_write),
                                "tokens": len(seqs_to_write) * max_seq_len,
                                "bytes": bytes_written,
                                "checksum": checksum,
                            }
                        )

                        stats["output_shards"] += 1
                        stats["bytes_written"] += bytes_written
                        output_shard_idx += 1

                except Exception as e:
                    console.print(f"[red]Error processing {shard_path}: {e}[/red]")

                progress.update(task, advance=1)

    # Write remaining sequences
    if all_sequences:
        output_path = output_dir / f"train_{output_shard_idx:05d}.bin"
        bytes_written, checksum = write_binary_shard(
            output_path,
            all_sequences,
            tokenizer_name,
            max_seq_len,
        )

        shard_metadata.append(
            {
                "path": output_path.name,
                "sequences": len(all_sequences),
                "tokens": len(all_sequences) * max_seq_len,
                "bytes": bytes_written,
                "checksum": checksum,
            }
        )

        stats["output_shards"] += 1
        stats["bytes_written"] += bytes_written

    # Write manifest
    manifest = {
        "format": "deepseek-tokens-v1",
        "tokenizer": tokenizer_name,
        "max_seq_len": max_seq_len,
        "stats": stats,
        "shards": shard_metadata,
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    console.print("\n[bold green]Tokenization complete![/bold green]")
    console.print(f"  Total tokens: {stats['total_tokens']:,}")
    console.print(f"  Total samples: {stats['total_samples']:,}")
    console.print(f"  Output shards: {stats['output_shards']}")
    console.print(f"  Total size: {stats['bytes_written'] / 1e9:.2f} GB")

    return stats


class ShardedBinaryDataset:
    """
    Dataset for reading tokenized binary shards.

    Supports memory-mapped file reading for efficient I/O.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)

        # Load manifest
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.tokenizer_name = self.manifest["tokenizer"]
        self.max_seq_len = self.manifest["max_seq_len"]
        self.shards = [
            s for s in self.manifest["shards"] if s["path"].startswith(split)
        ]

        # Calculate total sequences
        self.total_sequences = sum(s["sequences"] for s in self.shards)

        # Memory map shards
        self._mmap_shards()

    def _mmap_shards(self):
        """Memory map all shard files for efficient access."""
        import mmap

        self.mmaps = []
        self.shard_offsets = []
        self.cumulative_seqs = [0]

        for shard_info in self.shards:
            shard_path = self.data_dir / shard_info["path"]

            with open(shard_path, "rb") as f:
                # Read header to get offset
                magic = f.read(8)
                if magic != SHARD_MAGIC:
                    raise ValueError(f"Invalid shard magic: {shard_path}")

                max_seq_len = struct.unpack("<I", f.read(4))[0]
                num_seqs = struct.unpack("<I", f.read(4))[0]
                tokenizer_len = struct.unpack("<I", f.read(4))[0]
                _tokenizer = f.read(tokenizer_len)

                header_size = 8 + 4 + 4 + 4 + tokenizer_len
                self.shard_offsets.append(header_size)

                # Memory map
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.mmaps.append(mm)

            self.cumulative_seqs.append(self.cumulative_seqs[-1] + shard_info["sequences"])

    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get a single sequence by index."""
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self.total_sequences})")

        # Find which shard contains this index
        shard_idx = 0
        for i, cum_seq in enumerate(self.cumulative_seqs[1:], 1):
            if idx < cum_seq:
                shard_idx = i - 1
                break

        # Get local index within shard
        local_idx = idx - self.cumulative_seqs[shard_idx]

        # Read from memory mapped file
        offset = self.shard_offsets[shard_idx] + local_idx * self.max_seq_len * 2
        data = self.mmaps[shard_idx][offset : offset + self.max_seq_len * 2]

        return np.frombuffer(data, dtype=np.uint16)

    def close(self):
        """Close memory mapped files."""
        for mm in self.mmaps:
            mm.close()


def main():
    parser = argparse.ArgumentParser(description="Tokenize FineWeb-Edu dataset")

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input directory with downloaded JSONL shards",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for tokenized binary shards",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="HuggingFace tokenizer name (default: gpt2)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--tokens-per-shard",
        type=int,
        default=100_000_000,
        help="Target tokens per output shard (default: 100M)",
    )

    args = parser.parse_args()

    tokenize_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        tokenizer_name=args.tokenizer,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        tokens_per_shard=args.tokens_per_shard,
    )


if __name__ == "__main__":
    main()
