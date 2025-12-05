#!/usr/bin/env python3
"""
Download FineWeb-Edu Dataset
============================

Download and prepare the FineWeb-Edu dataset for training.
Uses HuggingFace datasets library with streaming for efficient downloads.

Usage:
    python scripts/data/download_fineweb_edu.py --output ./data/fineweb-edu
    python scripts/data/download_fineweb_edu.py --output ./data/fineweb-edu --subset sample-10BT --max-samples 100000
    python scripts/data/download_fineweb_edu.py --output ./data/fineweb-edu --resume
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


console = Console()


def check_disk_space(output_dir: Path, required_gb: float = 50.0) -> bool:
    """Check if there's enough disk space for the download."""
    try:
        stat = shutil.disk_usage(output_dir.parent if output_dir.exists() else output_dir.parent.parent)
        available_gb = stat.free / (1024**3)
        if available_gb < required_gb:
            console.print(
                f"[yellow]Warning: Only {available_gb:.1f}GB available, "
                f"recommended {required_gb:.1f}GB for FineWeb-Edu[/yellow]"
            )
            return False
        return True
    except Exception as e:
        console.print(f"[yellow]Could not check disk space: {e}[/yellow]")
        return True


def download_fineweb_edu(
    output_dir: Path,
    subset: str = "sample-10BT",
    max_samples: int | None = None,
    resume: bool = False,
    num_proc: int = 4,
    shard_size: int = 50000,  # Samples per shard
) -> dict:
    """
    Download FineWeb-Edu dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save the data
        subset: Dataset subset (sample-10BT, sample-100BT, or full)
        max_samples: Maximum number of samples to download (None for all)
        resume: Whether to resume interrupted download
        num_proc: Number of processes for parallel processing
        shard_size: Number of samples per output shard
        
    Returns:
        Dictionary with download statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check disk space
    check_disk_space(output_dir)
    
    # Track progress
    progress_file = output_dir / ".download_progress.json"
    if resume and progress_file.exists():
        with open(progress_file) as f:
            progress_state = json.load(f)
        start_idx = progress_state.get("samples_downloaded", 0)
        console.print(f"[green]Resuming from sample {start_idx}[/green]")
    else:
        start_idx = 0
    
    console.print(f"[bold blue]Downloading FineWeb-Edu ({subset})[/bold blue]")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Max samples: {max_samples or 'all'}")
    
    # Load dataset with streaming
    console.print("\n[dim]Loading dataset from HuggingFace...[/dim]")
    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=subset,
            split="train",
            streaming=True,
            trust_remote_code=False,
        )
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        console.print("[yellow]Tip: You may need to authenticate with HuggingFace:[/yellow]")
        console.print("  huggingface-cli login")
        sys.exit(1)
    
    # Process and save data
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "samples_downloaded": 0,
        "total_tokens_approx": 0,
        "shards_written": 0,
        "bytes_written": 0,
    }
    
    current_shard = []
    shard_idx = start_idx // shard_size
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Downloading...",
            total=max_samples if max_samples else None,
        )
        
        for idx, sample in enumerate(dataset):
            if idx < start_idx:
                continue
                
            if max_samples and idx >= max_samples + start_idx:
                break
            
            # Extract text
            text = sample.get("text", "")
            if not text:
                continue
            
            # Add to current shard
            current_shard.append({
                "text": text,
                "id": sample.get("id", str(idx)),
                "metadata": {
                    "url": sample.get("url", ""),
                    "dump": sample.get("dump", ""),
                    "score": sample.get("score", 0.0),
                }
            })
            
            # Approximate token count (rough estimate: 4 chars per token)
            stats["total_tokens_approx"] += len(text) // 4
            stats["samples_downloaded"] += 1
            
            # Write shard when full
            if len(current_shard) >= shard_size:
                shard_path = train_dir / f"shard_{shard_idx:05d}.jsonl"
                bytes_written = _write_shard(shard_path, current_shard)
                stats["bytes_written"] += bytes_written
                stats["shards_written"] += 1
                
                current_shard = []
                shard_idx += 1
                
                # Save progress
                with open(progress_file, "w") as f:
                    json.dump({
                        "samples_downloaded": stats["samples_downloaded"],
                        "shards_written": stats["shards_written"],
                        "shard_idx": shard_idx,
                    }, f)
            
            progress.update(task, advance=1)
    
    # Write remaining samples
    if current_shard:
        shard_path = train_dir / f"shard_{shard_idx:05d}.jsonl"
        bytes_written = _write_shard(shard_path, current_shard)
        stats["bytes_written"] += bytes_written
        stats["shards_written"] += 1
    
    # Write manifest
    manifest = {
        "dataset": "fineweb-edu",
        "subset": subset,
        "stats": stats,
        "shards": [
            str(p.name) for p in sorted(train_dir.glob("shard_*.jsonl"))
        ],
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Clean up progress file
    if progress_file.exists():
        progress_file.unlink()
    
    console.print("\n[bold green]Download complete![/bold green]")
    console.print(f"  Samples: {stats['samples_downloaded']:,}")
    console.print(f"  Shards: {stats['shards_written']}")
    console.print(f"  Approx tokens: {stats['total_tokens_approx']:,}")
    console.print(f"  Total size: {stats['bytes_written'] / 1e9:.2f} GB")
    
    return stats


def _write_shard(path: Path, samples: list) -> int:
    """Write samples to a JSONL shard file. Returns bytes written."""
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return path.stat().st_size


def main():
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu dataset")
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/fineweb-edu",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="sample-10BT",
        choices=["sample-10BT", "sample-100BT"],
        help="Dataset subset to download",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to download (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted download",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of processes for parallel processing",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=50000,
        help="Number of samples per output shard",
    )
    
    args = parser.parse_args()
    
    download_fineweb_edu(
        output_dir=Path(args.output),
        subset=args.subset,
        max_samples=args.max_samples,
        resume=args.resume,
        num_proc=args.num_proc,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
