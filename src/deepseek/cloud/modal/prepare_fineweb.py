#!/usr/bin/env python3
"""
Prepare FineWeb-Edu Data on Modal
=================================

Upload tokenized FineWeb-Edu shards to Modal persistent volume for distributed training.
This script handles parallel uploads, validation, and manifest synchronization.

Usage:
    # Upload shards from local directory to Modal volume
    uv run modal run modal_gpu/prepare_fineweb.py::upload_shards --shard-dir ./data/fineweb-tokenized

    # Validate uploaded data
    uv run modal run modal_gpu/prepare_fineweb.py::validate_shards

    # List available shards
    uv run modal run modal_gpu/prepare_fineweb.py::list_shards

    # Download manifest
    uv run modal run modal_gpu/prepare_fineweb.py::get_manifest
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

# Modal app configuration
app = modal.App("fineweb-data-prep")

# Use the same data volume as the training app
data_volume = modal.Volume.from_name(
    "deepseek-data",
    create_if_missing=True,
)

# Minimal image for data handling
data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24.0",
        "rich>=13.0.0",
        "tqdm>=4.65.0",
    )
)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
    timeout=3600,  # 1 hour timeout
)
def upload_single_shard(
    shard_content: bytes,
    shard_name: str,
    expected_hash: str,
) -> dict:
    """
    Upload a single shard to the Modal volume.

    Args:
        shard_content: Binary content of the shard
        shard_name: Name of the shard file
        expected_hash: Expected SHA256 hash for validation

    Returns:
        Dict with upload status and metadata
    """
    from pathlib import Path

    shard_dir = Path("/data/fineweb-edu/shards")
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_path = shard_dir / shard_name

    # Write shard
    with open(shard_path, "wb") as f:
        f.write(shard_content)

    # Verify hash
    sha256_hash = hashlib.sha256()
    with open(shard_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    actual_hash = sha256_hash.hexdigest()

    if actual_hash != expected_hash:
        shard_path.unlink()  # Remove corrupted file
        return {
            "shard": shard_name,
            "status": "failed",
            "error": f"Hash mismatch: expected {expected_hash[:16]}..., got {actual_hash[:16]}...",
        }

    # Commit the volume after writing
    data_volume.commit()

    return {
        "shard": shard_name,
        "status": "success",
        "size_bytes": len(shard_content),
        "hash": actual_hash,
        "path": str(shard_path),
    }


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
    timeout=7200,  # 2 hour timeout for large uploads
)
def update_manifest(shard_results: list[dict], metadata: dict) -> dict:
    """
    Update the manifest file with uploaded shard information.

    Args:
        shard_results: List of upload results from upload_single_shard
        metadata: Additional metadata (tokenizer, config, etc.)

    Returns:
        Dict with manifest update status
    """
    from datetime import datetime
    from pathlib import Path

    manifest_dir = Path("/data/fineweb-edu")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"

    # Load existing manifest or create new
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "dataset": "fineweb-edu",
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "shards": {},
            "metadata": {},
        }

    # Update with new shards
    successful_shards = [r for r in shard_results if r.get("status") == "success"]
    failed_shards = [r for r in shard_results if r.get("status") == "failed"]

    for result in successful_shards:
        manifest["shards"][result["shard"]] = {
            "path": result["path"],
            "size_bytes": result["size_bytes"],
            "hash": result["hash"],
            "uploaded_at": datetime.now().isoformat(),
        }

    # Update metadata
    manifest["metadata"].update(metadata)
    manifest["updated_at"] = datetime.now().isoformat()
    manifest["total_shards"] = len(manifest["shards"])
    manifest["total_bytes"] = sum(s["size_bytes"] for s in manifest["shards"].values())

    # Write manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Commit the volume
    data_volume.commit()

    return {
        "status": "success",
        "manifest_path": str(manifest_path),
        "total_shards": manifest["total_shards"],
        "successful_uploads": len(successful_shards),
        "failed_uploads": len(failed_shards),
        "failed_shards": [r["shard"] for r in failed_shards],
    }


@app.local_entrypoint()
def upload_shards(
    shard_dir: str = "./data/fineweb-tokenized",
    max_workers: int = 4,
    tokenizer: str = "gpt2",
    max_seq_len: int = 512,
):
    """
    Upload tokenized shards to Modal volume.

    Args:
        shard_dir: Local directory containing tokenized shards
        max_workers: Number of parallel upload workers
        tokenizer: Tokenizer name used for tokenization
        max_seq_len: Maximum sequence length used
    """
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    console = Console()
    shard_path = Path(shard_dir)

    if not shard_path.exists():
        console.print(f"[red]Error: Shard directory not found: {shard_dir}[/red]")
        return

    # Find all shard files
    shard_files = sorted(shard_path.glob("*.bin"))
    if not shard_files:
        console.print(f"[yellow]No .bin shard files found in {shard_dir}[/yellow]")
        return

    console.print(f"[green]Found {len(shard_files)} shards to upload[/green]")

    # Upload shards in parallel
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Uploading shards...", total=len(shard_files))

        for shard_file in shard_files:
            # Read shard content
            with open(shard_file, "rb") as f:
                content = f.read()

            # Compute hash
            file_hash = compute_file_hash(shard_file)

            # Upload to Modal (spawns remote function)
            result = upload_single_shard.remote(
                shard_content=content,
                shard_name=shard_file.name,
                expected_hash=file_hash,
            )
            results.append(result)
            progress.advance(task)

    console.print("[blue]Waiting for uploads to complete...[/blue]")

    # Collect results
    final_results = []
    for result in results:
        final_results.append(result)

    # Update manifest
    metadata = {
        "tokenizer": tokenizer,
        "max_seq_len": max_seq_len,
        "source_dir": str(shard_path.resolve()),
    }

    manifest_result = update_manifest.remote(final_results, metadata)

    console.print("\n[bold green]Upload Complete![/bold green]")
    console.print(f"  Total shards: {manifest_result['total_shards']}")
    console.print(f"  Successful: {manifest_result['successful_uploads']}")
    console.print(f"  Failed: {manifest_result['failed_uploads']}")

    if manifest_result["failed_shards"]:
        console.print(f"  [red]Failed shards: {manifest_result['failed_shards']}[/red]")


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
)
def validate_shards() -> dict:
    """
    Validate all uploaded shards by checking their hashes.

    Returns:
        Dict with validation results
    """
    from pathlib import Path

    manifest_path = Path("/data/fineweb-edu/manifest.json")
    if not manifest_path.exists():
        return {"status": "error", "message": "No manifest found"}

    with open(manifest_path) as f:
        manifest = json.load(f)

    valid_shards = []
    invalid_shards = []

    for shard_name, shard_info in manifest.get("shards", {}).items():
        shard_path = Path(shard_info["path"])
        if not shard_path.exists():
            invalid_shards.append({"shard": shard_name, "error": "File not found"})
            continue

        # Verify hash
        sha256_hash = hashlib.sha256()
        with open(shard_path, "rb") as f:
            while chunk := f.read(8192):
                sha256_hash.update(chunk)
        actual_hash = sha256_hash.hexdigest()

        if actual_hash == shard_info["hash"]:
            valid_shards.append(shard_name)
        else:
            invalid_shards.append({
                "shard": shard_name,
                "error": f"Hash mismatch: expected {shard_info['hash'][:16]}, got {actual_hash[:16]}",
            })

    return {
        "status": "success" if not invalid_shards else "partial",
        "valid_count": len(valid_shards),
        "invalid_count": len(invalid_shards),
        "invalid_shards": invalid_shards,
        "total_shards": len(manifest.get("shards", {})),
    }


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
)
def list_shards() -> dict:
    """
    List all available shards in the Modal volume.

    Returns:
        Dict with shard list and metadata
    """
    from pathlib import Path

    manifest_path = Path("/data/fineweb-edu/manifest.json")
    if not manifest_path.exists():
        return {"status": "error", "message": "No manifest found"}

    with open(manifest_path) as f:
        manifest = json.load(f)

    return {
        "status": "success",
        "dataset": manifest.get("dataset"),
        "version": manifest.get("version"),
        "total_shards": manifest.get("total_shards", 0),
        "total_bytes": manifest.get("total_bytes", 0),
        "shards": list(manifest.get("shards", {}).keys()),
        "metadata": manifest.get("metadata", {}),
    }


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
)
def get_manifest() -> dict:
    """
    Get the full manifest from Modal volume.

    Returns:
        Full manifest dict
    """
    from pathlib import Path

    manifest_path = Path("/data/fineweb-edu/manifest.json")
    if not manifest_path.exists():
        return {"status": "error", "message": "No manifest found"}

    with open(manifest_path) as f:
        return json.load(f)


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
    timeout=1800,
)
def cleanup_shards(shard_names: list[str] | None = None) -> dict:
    """
    Remove shards from the Modal volume.

    Args:
        shard_names: List of shard names to remove. If None, removes all.

    Returns:
        Dict with cleanup results
    """
    from pathlib import Path

    manifest_path = Path("/data/fineweb-edu/manifest.json")
    if not manifest_path.exists():
        return {"status": "error", "message": "No manifest found"}

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Determine which shards to remove
    if shard_names is None:
        shards_to_remove = list(manifest.get("shards", {}).keys())
    else:
        shards_to_remove = shard_names

    removed = []
    errors = []

    for shard_name in shards_to_remove:
        if shard_name not in manifest.get("shards", {}):
            errors.append({"shard": shard_name, "error": "Not in manifest"})
            continue

        shard_path = Path(manifest["shards"][shard_name]["path"])
        try:
            if shard_path.exists():
                shard_path.unlink()
            del manifest["shards"][shard_name]
            removed.append(shard_name)
        except Exception as e:
            errors.append({"shard": shard_name, "error": str(e)})

    # Update manifest
    manifest["total_shards"] = len(manifest.get("shards", {}))
    manifest["total_bytes"] = sum(
        s["size_bytes"] for s in manifest.get("shards", {}).values()
    )

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Commit the volume
    data_volume.commit()

    return {
        "status": "success",
        "removed_count": len(removed),
        "error_count": len(errors),
        "removed": removed,
        "errors": errors,
    }


if __name__ == "__main__":
    # For local testing, run with: python modal_gpu/prepare_fineweb.py
    print("Use 'modal run modal_gpu/prepare_fineweb.py::upload_shards' to upload shards")
    print("Use 'modal run modal_gpu/prepare_fineweb.py::validate_shards' to validate")
    print("Use 'modal run modal_gpu/prepare_fineweb.py::list_shards' to list shards")
