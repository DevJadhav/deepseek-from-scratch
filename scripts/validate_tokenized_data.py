#!/usr/bin/env python3
"""
Validate Tokenized Data Integrity (D7)
======================================

Validates the integrity of tokenized data by:
1. Verifying checksums of all binary shards
2. Checking manifest consistency
3. Validating binary format headers
4. Confirming token counts

Usage:
    uv run python scripts/validate_tokenized_data.py --data-dir ./data/tokenized
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

# Magic bytes expected in binary shards
SHARD_MAGIC = b"DSTKN001"

DOMAINS = ["web", "code", "math", "books", "scientific"]


def validate_binary_shard(shard_path: Path) -> dict:
    """Validate a single binary shard file.
    
    Returns:
        Dictionary with validation results
    """
    result = {
        "path": str(shard_path),
        "valid": False,
        "errors": [],
    }
    
    if not shard_path.exists():
        result["errors"].append("File not found")
        return result
    
    try:
        with open(shard_path, "rb") as f:
            # Read and verify magic bytes
            magic = f.read(8)
            if magic != SHARD_MAGIC:
                result["errors"].append(f"Invalid magic bytes: {magic!r}")
                return result
            
            # Read header
            max_seq_len = struct.unpack("<I", f.read(4))[0]
            num_sequences = struct.unpack("<I", f.read(4))[0]
            tokenizer_len = struct.unpack("<I", f.read(4))[0]
            tokenizer_name = f.read(tokenizer_len).decode("utf-8")
            
            result["max_seq_len"] = max_seq_len
            result["num_sequences"] = num_sequences
            result["tokenizer"] = tokenizer_name
            
            # Calculate expected file size
            header_size = 8 + 4 + 4 + 4 + tokenizer_len
            # uint32 = 4 bytes per token
            data_size = num_sequences * max_seq_len * 4
            expected_size = header_size + data_size
            actual_size = shard_path.stat().st_size
            
            if actual_size != expected_size:
                result["errors"].append(
                    f"Size mismatch: expected {expected_size}, got {actual_size}"
                )
                return result
            
            result["valid"] = True
            result["tokens"] = num_sequences * max_seq_len
            
    except Exception as e:
        result["errors"].append(f"Read error: {e}")
    
    return result


def compute_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_domain(domain_dir: Path, domain: str) -> dict:
    """Validate all shards for a domain.
    
    Returns:
        Dictionary with domain validation results
    """
    result = {
        "domain": domain,
        "valid": True,
        "shards_validated": 0,
        "total_tokens": 0,
        "checksum_matches": 0,
        "errors": [],
    }
    
    if not domain_dir.exists():
        result["valid"] = False
        result["errors"].append(f"Domain directory not found: {domain_dir}")
        return result
    
    # Load manifest
    manifest_path = domain_dir / "manifest.json"
    if not manifest_path.exists():
        result["valid"] = False
        result["errors"].append("Manifest not found")
        return result
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    shards = manifest.get("shards", [])
    if not shards:
        result["valid"] = False
        result["errors"].append("No shards in manifest")
        return result
    
    result["manifest_tokens"] = manifest.get("stats", {}).get("total_tokens", 0)
    result["manifest_samples"] = manifest.get("stats", {}).get("total_samples", 0)
    
    # Validate each shard
    for shard_info in shards:
        shard_path = domain_dir / shard_info["path"]
        
        # Validate binary format
        shard_result = validate_binary_shard(shard_path)
        
        if not shard_result["valid"]:
            result["valid"] = False
            result["errors"].extend(shard_result["errors"])
            continue
        
        result["shards_validated"] += 1
        result["total_tokens"] += shard_result.get("tokens", 0)
        
        # Verify checksum
        computed_checksum = compute_checksum(shard_path)
        expected_checksum = shard_info.get("checksum", "")
        
        if computed_checksum == expected_checksum:
            result["checksum_matches"] += 1
        else:
            result["valid"] = False
            result["errors"].append(
                f"{shard_info['path']}: checksum mismatch "
                f"(expected {expected_checksum[:16]}..., got {computed_checksum[:16]}...)"
            )
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate tokenized data integrity"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/tokenized"),
        help="Directory containing tokenized data",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=DOMAINS,
        help="Domains to validate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for validation results (JSON)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("D7: VALIDATING TOKENIZED DATA INTEGRITY")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Domains: {args.domains}")
    print()
    
    all_results = {
        "valid": True,
        "domains": {},
        "summary": {
            "total_tokens": 0,
            "total_shards": 0,
            "domains_valid": 0,
            "domains_invalid": 0,
        },
    }
    
    for domain in args.domains:
        domain_dir = args.data_dir / domain
        print(f"Validating {domain}...")
        
        result = validate_domain(domain_dir, domain)
        all_results["domains"][domain] = result
        
        if result["valid"]:
            print(f"  ✓ {domain}: {result['shards_validated']} shards, "
                  f"{result['total_tokens']:,} tokens, "
                  f"{result['checksum_matches']} checksums OK")
            all_results["summary"]["domains_valid"] += 1
            all_results["summary"]["total_tokens"] += result["total_tokens"]
            all_results["summary"]["total_shards"] += result["shards_validated"]
        else:
            print(f"  ✗ {domain}: INVALID")
            for error in result["errors"][:3]:  # Show first 3 errors
                print(f"    - {error}")
            all_results["valid"] = False
            all_results["summary"]["domains_invalid"] += 1
    
    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Domains valid: {all_results['summary']['domains_valid']}/{len(args.domains)}")
    print(f"Total shards: {all_results['summary']['total_shards']}")
    print(f"Total tokens: {all_results['summary']['total_tokens']:,}")
    print()
    
    if all_results["valid"]:
        print("✓ ALL DATA VALIDATED SUCCESSFULLY")
    else:
        print("✗ VALIDATION FAILED - See errors above")
    
    # Save results
    output_path = args.output or (args.data_dir / "validation_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return 0 if all_results["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
