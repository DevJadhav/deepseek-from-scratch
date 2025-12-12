#!/usr/bin/env python3
"""
Prepare Training Data for DeepSeek
===================================

Complete data preparation pipeline implementing:
- D1-D5: Download all domain datasets
- D6: Tokenize all datasets
- D7: Validate tokenized data integrity
- D8: Configure domain mixing ratios (30/30/30/5/5)
- D9: Setup streaming data pipeline
- D10: Create validation split (5%)

Usage:
    # Run all data preparation steps
    uv run python scripts/prepare_training_data.py --all

    # Download only
    uv run python scripts/prepare_training_data.py --download

    # Tokenize only
    uv run python scripts/prepare_training_data.py --tokenize

    # Create validation split
    uv run python scripts/prepare_training_data.py --split

    # Validate data integrity
    uv run python scripts/prepare_training_data.py --validate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Domain mixing configuration (30/30/30/5/5)
DOMAIN_WEIGHTS = {
    "web": 0.30,
    "code": 0.30,
    "math": 0.30,
    "books": 0.05,
    "scientific": 0.05,
}

VALIDATION_SPLIT = 0.05  # 5% validation


def download_datasets(
    output_dir: Path,
    max_samples: int = 10000,
    domains: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """D1-D5: Download all domain datasets."""
    from deepseek.pipeline.utils.data_downloader import DataDownloader
    
    logger.info("=" * 60)
    logger.info("D1-D5: Downloading domain datasets")
    logger.info("=" * 60)
    
    downloader = DataDownloader(output_dir=output_dir)
    results = downloader.download_all_domains(
        max_samples_per_domain=max_samples,
        skip_existing=not force,
        domains=domains,
    )
    
    logger.info("\nDownload Summary:")
    for domain, stats in results.items():
        if stats.get("skipped"):
            logger.info(f"  {domain}: SKIPPED (already exists)")
        elif stats.get("error"):
            logger.error(f"  {domain}: ERROR - {stats['error']}")
        else:
            logger.info(
                f"  {domain}: {stats['samples_downloaded']} samples "
                f"({stats['bytes_written'] / 1024 / 1024:.2f} MB)"
            )
    
    return results


def tokenize_datasets(
    input_dir: Path,
    output_dir: Path,
    tokenizer_name: str = "deepseek-ai/deepseek-llm-7b-base",
    max_seq_len: int = 2048,
    domains: list[str] | None = None,
) -> dict[str, Any]:
    """D6: Tokenize all datasets."""
    logger.info("=" * 60)
    logger.info("D6: Tokenizing datasets")
    logger.info("=" * 60)
    
    from transformers import AutoTokenizer
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.warning(f"Failed to load {tokenizer_name}, falling back to gpt2: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer_name = "gpt2"
    
    target_domains = domains or list(DOMAIN_WEIGHTS.keys())
    results = {}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for domain in target_domains:
        domain_input = input_dir / domain
        if not domain_input.exists():
            logger.warning(f"Skipping {domain}: directory not found")
            results[domain] = {"error": "not found"}
            continue
        
        domain_output = output_dir / domain
        domain_output.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nTokenizing {domain}...")
        
        stats = {
            "total_tokens": 0,
            "total_samples": 0,
            "shards_written": 0,
        }
        
        # Read JSONL files and tokenize
        sequences = []
        
        for shard_file in sorted(domain_input.glob("shard_*.jsonl")):
            with open(shard_file) as f:
                for line in f:
                    try:
                        sample = json.loads(line)
                        text = sample.get("text", "")
                        if not text:
                            continue
                        
                        # Tokenize
                        tokens = tokenizer.encode(
                            text, 
                            truncation=True,
                            max_length=max_seq_len,
                            add_special_tokens=True,
                        )
                        
                        if len(tokens) > 10:  # Skip very short sequences
                            sequences.append({
                                "input_ids": tokens,
                                "domain": domain,
                                "length": len(tokens),
                            })
                            stats["total_tokens"] += len(tokens)
                            stats["total_samples"] += 1
                    except json.JSONDecodeError:
                        continue
        
        # Write tokenized data
        output_file = domain_output / "tokenized.jsonl"
        with open(output_file, "w") as f:
            for seq in sequences:
                f.write(json.dumps(seq) + "\n")
        
        stats["shards_written"] = 1
        stats["output_file"] = str(output_file)
        
        # Write domain manifest
        manifest = {
            "domain": domain,
            "tokenizer": tokenizer_name,
            "max_seq_len": max_seq_len,
            "stats": stats,
        }
        with open(domain_output / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        results[domain] = stats
        logger.info(f"  {domain}: {stats['total_tokens']:,} tokens from {stats['total_samples']} samples")
    
    # Write combined manifest
    combined = {
        "format": "deepseek-tokenized-v1",
        "tokenizer": tokenizer_name,
        "max_seq_len": max_seq_len,
        "domains": results,
        "domain_weights": DOMAIN_WEIGHTS,
        "total_tokens": sum(r.get("total_tokens", 0) for r in results.values()),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(combined, f, indent=2)
    
    return results


def validate_data(data_dir: Path) -> dict[str, Any]:
    """D7: Validate tokenized data integrity."""
    logger.info("=" * 60)
    logger.info("D7: Validating data integrity")
    logger.info("=" * 60)
    
    results = {
        "valid": True,
        "domains": {},
        "checksums": {},
    }
    
    tokenized_dir = data_dir / "tokenized"
    if not tokenized_dir.exists():
        logger.error("Tokenized directory not found")
        results["valid"] = False
        return results
    
    for domain in DOMAIN_WEIGHTS.keys():
        domain_dir = tokenized_dir / domain
        if not domain_dir.exists():
            logger.warning(f"Domain {domain} not found")
            results["domains"][domain] = {"status": "missing"}
            continue
        
        # Check manifest
        manifest_path = domain_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning(f"Manifest not found for {domain}")
            results["domains"][domain] = {"status": "no manifest"}
            continue
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Verify tokenized file
        tokenized_file = domain_dir / "tokenized.jsonl"
        if not tokenized_file.exists():
            logger.warning(f"Tokenized file not found for {domain}")
            results["domains"][domain] = {"status": "no data"}
            continue
        
        # Compute checksum
        sha256 = hashlib.sha256()
        with open(tokenized_file, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        checksum = sha256.hexdigest()[:16]
        
        # Count samples
        sample_count = 0
        token_count = 0
        with open(tokenized_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    sample_count += 1
                    token_count += data.get("length", len(data.get("input_ids", [])))
                except json.JSONDecodeError:
                    pass
        
        results["domains"][domain] = {
            "status": "valid",
            "samples": sample_count,
            "tokens": token_count,
            "checksum": checksum,
        }
        results["checksums"][domain] = checksum
        
        # Verify against manifest
        expected_samples = manifest.get("stats", {}).get("total_samples", 0)
        if sample_count != expected_samples:
            logger.warning(
                f"{domain}: sample count mismatch "
                f"(expected {expected_samples}, got {sample_count})"
            )
            results["domains"][domain]["status"] = "count mismatch"
            results["valid"] = False
        else:
            logger.info(f"  {domain}: ✓ {sample_count} samples, checksum {checksum}")
    
    if results["valid"]:
        logger.info("\n✓ All data validated successfully")
    else:
        logger.warning("\n⚠ Some validation issues found")
    
    # Save validation results
    with open(data_dir / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def create_validation_split(
    data_dir: Path,
    validation_ratio: float = VALIDATION_SPLIT,
) -> dict[str, Any]:
    """D10: Create train/validation split (5%)."""
    logger.info("=" * 60)
    logger.info("D10: Creating validation split")
    logger.info("=" * 60)
    
    import numpy as np
    
    tokenized_dir = data_dir / "tokenized"
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "train_samples": 0,
        "val_samples": 0,
        "domains": {},
    }
    
    train_samples = []
    val_samples = []
    
    rng = np.random.Generator(np.random.PCG64(42))  # Deterministic seed
    
    for domain in DOMAIN_WEIGHTS.keys():
        domain_dir = tokenized_dir / domain
        tokenized_file = domain_dir / "tokenized.jsonl"
        
        if not tokenized_file.exists():
            logger.warning(f"Skipping {domain}: tokenized file not found")
            continue
        
        # Load samples
        samples = []
        with open(tokenized_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    samples.append(data)
                except json.JSONDecodeError:
                    continue
        
        # Shuffle deterministically
        indices = rng.permutation(len(samples))
        split_idx = int(len(samples) * (1 - validation_ratio))
        
        domain_train = [samples[i] for i in indices[:split_idx]]
        domain_val = [samples[i] for i in indices[split_idx:]]
        
        train_samples.extend(domain_train)
        val_samples.extend(domain_val)
        
        results["domains"][domain] = {
            "train": len(domain_train),
            "val": len(domain_val),
        }
        
        logger.info(f"  {domain}: {len(domain_train)} train, {len(domain_val)} val")
    
    # Write split files
    with open(splits_dir / "train.jsonl", "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    
    with open(splits_dir / "val.jsonl", "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")
    
    results["train_samples"] = len(train_samples)
    results["val_samples"] = len(val_samples)
    results["validation_ratio"] = validation_ratio
    
    # Write split manifest
    with open(splits_dir / "manifest.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Split complete: {len(train_samples)} train, {len(val_samples)} val")
    
    return results


def verify_streaming_pipeline(data_dir: Path) -> dict[str, Any]:
    """D9: Verify streaming data pipeline configuration."""
    logger.info("=" * 60)
    logger.info("D9: Verifying streaming data pipeline")
    logger.info("=" * 60)
    
    from deepseek.pipeline.data_ingestion import (
        DomainMixer,
        MultiDomainConfig,
        DOMAIN_MIXING_WEIGHTS,
    )
    
    results = {
        "status": "success",
        "domain_weights": DOMAIN_MIXING_WEIGHTS,
        "domains_found": [],
        "domains_missing": [],
    }
    
    # Check if all domain data exists
    for domain in DOMAIN_MIXING_WEIGHTS.keys():
        domain_path = data_dir / domain
        if domain_path.exists():
            results["domains_found"].append(domain)
        else:
            results["domains_missing"].append(domain)
    
    if results["domains_missing"]:
        logger.warning(f"Missing domains: {results['domains_missing']}")
        results["status"] = "partial"
    
    # Test DomainMixer initialization
    try:
        config = MultiDomainConfig.from_weights(
            DOMAIN_MIXING_WEIGHTS,
            data_root=data_dir,
            validation_split=VALIDATION_SPLIT,
        )
        mixer = DomainMixer(config)
        
        # Get domain statistics
        if results["domains_found"]:
            stats = mixer.get_domain_stats()
            results["domain_stats"] = stats
            
            logger.info("\nDomain Statistics:")
            for domain, stat in stats.items():
                logger.info(
                    f"  {domain}: {stat['samples']} samples, "
                    f"weight={stat['weight']:.0%}"
                )
        
        results["mixer_initialized"] = True
        logger.info("\n✓ DomainMixer initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize DomainMixer: {e}")
        results["status"] = "error"
        results["error"] = str(e)
    
    return results


def print_summary(data_dir: Path):
    """Print summary of all data preparation."""
    logger.info("=" * 60)
    logger.info("DATA PREPARATION SUMMARY")
    logger.info("=" * 60)
    
    # Check raw data
    logger.info("\nRaw Data:")
    for domain in DOMAIN_WEIGHTS.keys():
        domain_path = data_dir / domain
        if domain_path.exists():
            manifest_file = domain_path / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file) as f:
                    manifest = json.load(f)
                samples = manifest.get("samples_downloaded", 0)
                logger.info(f"  ✓ {domain}: {samples} samples")
            else:
                logger.info(f"  ✓ {domain}: exists (no manifest)")
        else:
            logger.info(f"  ✗ {domain}: missing")
    
    # Check tokenized data
    tokenized_dir = data_dir / "tokenized"
    if tokenized_dir.exists():
        logger.info("\nTokenized Data:")
        manifest_file = tokenized_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            logger.info(f"  Total tokens: {manifest.get('total_tokens', 0):,}")
            logger.info(f"  Tokenizer: {manifest.get('tokenizer', 'unknown')}")
    
    # Check splits
    splits_dir = data_dir / "splits"
    if splits_dir.exists():
        logger.info("\nTrain/Val Splits:")
        manifest_file = splits_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            logger.info(f"  Train: {manifest.get('train_samples', 0):,} samples")
            logger.info(f"  Val: {manifest.get('val_samples', 0):,} samples")
    
    logger.info("\nDomain Mixing Weights:")
    for domain, weight in DOMAIN_WEIGHTS.items():
        logger.info(f"  {domain}: {weight:.0%}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for DeepSeek",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Data directory (default: ./data)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Max samples per domain for download (default: 10000)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="deepseek-ai/deepseek-llm-7b-base",
        help="Tokenizer to use (default: deepseek-ai/deepseek-llm-7b-base)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    
    # Action flags
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--download", action="store_true", help="D1-D5: Download datasets")
    parser.add_argument("--tokenize", action="store_true", help="D6: Tokenize datasets")
    parser.add_argument("--validate", action="store_true", help="D7: Validate data")
    parser.add_argument("--split", action="store_true", help="D10: Create val split")
    parser.add_argument("--verify-pipeline", action="store_true", help="D9: Verify pipeline")
    parser.add_argument("--summary", action="store_true", help="Print summary")
    parser.add_argument("--force", action="store_true", help="Force re-download/re-process")
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Default to summary if no action specified
    if not any([args.all, args.download, args.tokenize, args.validate, 
                args.split, args.verify_pipeline, args.summary]):
        args.summary = True
    
    results = {}
    
    # D1-D5: Download
    if args.all or args.download:
        results["download"] = download_datasets(
            output_dir=data_dir,
            max_samples=args.max_samples,
            force=args.force,
        )
    
    # D6: Tokenize
    if args.all or args.tokenize:
        results["tokenize"] = tokenize_datasets(
            input_dir=data_dir,
            output_dir=data_dir / "tokenized",
            tokenizer_name=args.tokenizer,
            max_seq_len=args.max_seq_len,
        )
    
    # D7: Validate
    if args.all or args.validate:
        results["validate"] = validate_data(data_dir)
    
    # D9: Verify pipeline
    if args.all or args.verify_pipeline:
        results["pipeline"] = verify_streaming_pipeline(data_dir)
    
    # D10: Create split
    if args.all or args.split:
        results["split"] = create_validation_split(data_dir)
    
    # Summary
    if args.summary or args.all:
        print_summary(data_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
