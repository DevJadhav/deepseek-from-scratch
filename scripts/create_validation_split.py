#!/usr/bin/env python3
"""
Create Train/Validation Split (D10)
====================================

Creates a 95/5 train/validation split from tokenized data.

Usage:
    uv run python scripts/create_validation_split.py --data-dir ./data/tokenized --output-dir ./data/splits
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

SHARD_MAGIC = b"DSTKN001"
DOMAINS = ["web", "code", "math", "books", "scientific"]
VALIDATION_RATIO = 0.05  # 5% validation


def read_binary_shard(shard_path: Path) -> tuple[np.ndarray, str, int]:
    """Read sequences from a binary shard.
    
    Returns:
        Tuple of (sequences array, tokenizer name, max_seq_len)
    """
    with open(shard_path, "rb") as f:
        magic = f.read(8)
        if magic != SHARD_MAGIC:
            raise ValueError(f"Invalid magic bytes in {shard_path}")
        
        max_seq_len = struct.unpack("<I", f.read(4))[0]
        num_sequences = struct.unpack("<I", f.read(4))[0]
        tokenizer_len = struct.unpack("<I", f.read(4))[0]
        tokenizer_name = f.read(tokenizer_len).decode("utf-8")
        
        # Read all sequences
        data = np.frombuffer(f.read(), dtype=np.uint32)
        sequences = data.reshape(num_sequences, max_seq_len)
    
    return sequences, tokenizer_name, max_seq_len


def write_binary_shard(
    output_path: Path,
    sequences: np.ndarray,
    tokenizer_name: str,
    max_seq_len: int,
) -> int:
    """Write sequences to a binary shard.
    
    Returns:
        Number of bytes written
    """
    tokenizer_bytes = tokenizer_name.encode("utf-8")
    
    with open(output_path, "wb") as f:
        f.write(SHARD_MAGIC)
        f.write(struct.pack("<I", max_seq_len))
        f.write(struct.pack("<I", len(sequences)))
        f.write(struct.pack("<I", len(tokenizer_bytes)))
        f.write(tokenizer_bytes)
        f.write(sequences.astype(np.uint32).tobytes())
    
    return output_path.stat().st_size


def split_domain(
    domain_dir: Path,
    train_dir: Path,
    val_dir: Path,
    domain: str,
    val_ratio: float = VALIDATION_RATIO,
    seed: int = 42,
) -> dict:
    """Split a domain's data into train/val sets.
    
    Returns:
        Statistics dictionary
    """
    result = {
        "domain": domain,
        "train_sequences": 0,
        "val_sequences": 0,
        "train_tokens": 0,
        "val_tokens": 0,
    }
    
    # Load manifest
    manifest_path = domain_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"  ✗ No manifest found for {domain}")
        return result
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Collect all sequences
    all_sequences = []
    tokenizer_name = manifest.get("tokenizer", "unknown")
    max_seq_len = manifest.get("max_seq_len", 2048)
    
    for shard_info in manifest.get("shards", []):
        shard_path = domain_dir / shard_info["path"]
        if shard_path.exists():
            sequences, tokenizer_name, max_seq_len = read_binary_shard(shard_path)
            all_sequences.append(sequences)
    
    if not all_sequences:
        print(f"  ✗ No data found for {domain}")
        return result
    
    # Concatenate all sequences
    all_sequences = np.concatenate(all_sequences, axis=0)
    total_sequences = len(all_sequences)
    
    # Shuffle with fixed seed for reproducibility
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_sequences)
    
    # Split
    split_idx = int(total_sequences * (1 - val_ratio))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_sequences = all_sequences[train_indices]
    val_sequences = all_sequences[val_indices]
    
    # Write train shard
    train_domain_dir = train_dir / domain
    train_domain_dir.mkdir(parents=True, exist_ok=True)
    train_path = train_domain_dir / "train_00000.bin"
    write_binary_shard(train_path, train_sequences, tokenizer_name, max_seq_len)
    
    # Write val shard
    val_domain_dir = val_dir / domain
    val_domain_dir.mkdir(parents=True, exist_ok=True)
    val_path = val_domain_dir / "val_00000.bin"
    write_binary_shard(val_path, val_sequences, tokenizer_name, max_seq_len)
    
    # Write manifests
    train_manifest = {
        "format": "deepseek-tokens-v1",
        "tokenizer": tokenizer_name,
        "max_seq_len": max_seq_len,
        "split": "train",
        "stats": {
            "total_sequences": len(train_sequences),
            "total_tokens": len(train_sequences) * max_seq_len,
        },
        "shards": [{"path": "train_00000.bin", "sequences": len(train_sequences)}],
    }
    with open(train_domain_dir / "manifest.json", "w") as f:
        json.dump(train_manifest, f, indent=2)
    
    val_manifest = {
        "format": "deepseek-tokens-v1",
        "tokenizer": tokenizer_name,
        "max_seq_len": max_seq_len,
        "split": "val",
        "stats": {
            "total_sequences": len(val_sequences),
            "total_tokens": len(val_sequences) * max_seq_len,
        },
        "shards": [{"path": "val_00000.bin", "sequences": len(val_sequences)}],
    }
    with open(val_domain_dir / "manifest.json", "w") as f:
        json.dump(val_manifest, f, indent=2)
    
    result["train_sequences"] = len(train_sequences)
    result["val_sequences"] = len(val_sequences)
    result["train_tokens"] = len(train_sequences) * max_seq_len
    result["val_tokens"] = len(val_sequences) * max_seq_len
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Create train/validation split from tokenized data"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/tokenized"),
        help="Directory containing tokenized data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/splits"),
        help="Output directory for splits",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VALIDATION_RATIO,
        help=f"Validation ratio (default: {VALIDATION_RATIO})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=DOMAINS,
        help="Domains to process",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("D10: CREATING TRAIN/VALIDATION SPLIT")
    print("=" * 60)
    print(f"Input: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Validation ratio: {args.val_ratio:.0%}")
    print(f"Seed: {args.seed}")
    print()
    
    train_dir = args.output_dir / "train"
    val_dir = args.output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "domains": {},
        "summary": {
            "total_train_sequences": 0,
            "total_val_sequences": 0,
            "total_train_tokens": 0,
            "total_val_tokens": 0,
        },
    }
    
    for domain in args.domains:
        domain_dir = args.data_dir / domain
        if not domain_dir.exists():
            print(f"Skipping {domain}: not found")
            continue
        
        print(f"Splitting {domain}...")
        result = split_domain(
            domain_dir=domain_dir,
            train_dir=train_dir,
            val_dir=val_dir,
            domain=domain,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        
        results["domains"][domain] = result
        results["summary"]["total_train_sequences"] += result["train_sequences"]
        results["summary"]["total_val_sequences"] += result["val_sequences"]
        results["summary"]["total_train_tokens"] += result["train_tokens"]
        results["summary"]["total_val_tokens"] += result["val_tokens"]
        
        print(f"  ✓ {domain}: {result['train_sequences']:,} train, {result['val_sequences']:,} val")
    
    print()
    print("=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"Train sequences: {results['summary']['total_train_sequences']:,}")
    print(f"Val sequences: {results['summary']['total_val_sequences']:,}")
    print(f"Train tokens: {results['summary']['total_train_tokens']:,}")
    print(f"Val tokens: {results['summary']['total_val_tokens']:,}")
    print(f"Val ratio: {results['summary']['total_val_sequences'] / (results['summary']['total_train_sequences'] + results['summary']['total_val_sequences']):.2%}")
    
    # Save manifest
    with open(args.output_dir / "split_manifest.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nManifest saved to: {args.output_dir}/split_manifest.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
