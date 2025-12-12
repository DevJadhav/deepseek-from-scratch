#!/usr/bin/env python3
"""
Export checkpoints from Modal volumes to local checkpoint directory.

This script downloads PyTorch and Rust checkpoints from the Modal
'deepseek-checkpoints' volume to the local checkpoints/ directory.

Usage:
    uv run python scripts/export_modal_checkpoints.py --list           # List available checkpoints
    uv run python scripts/export_modal_checkpoints.py --download       # Download final checkpoints only
    uv run python scripts/export_modal_checkpoints.py --download --all # Download all checkpoints
    uv run python scripts/export_modal_checkpoints.py --download --backend pytorch  # PyTorch only
    uv run python scripts/export_modal_checkpoints.py --download --backend rust     # Rust only
"""

import argparse
import sys
from pathlib import Path


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def list_checkpoints():
    """List all checkpoints in the Modal volume."""
    import modal

    print("🔍 Connecting to Modal volume 'deepseek-checkpoints'...")

    try:
        vol = modal.Volume.from_name("deepseek-checkpoints")
    except modal.exception.NotFoundError:
        print("❌ Volume 'deepseek-checkpoints' not found!")
        print("   Make sure you have run training on Modal first.")
        return []

    print("📁 Listing checkpoints in volume:\n")

    all_files = []
    total_size = 0

    # List all files recursively
    for entry in vol.listdir("/", recursive=True):
        path = entry.path
        # Skip directories
        if path.endswith("/"):
            continue

        size = getattr(entry, 'size', 0) or 0
        all_files.append((path, size))
        total_size += size

        # Pretty print with size
        size_str = format_size(size) if size else "(empty)"
        print(f"  {path} - {size_str}")

    if not all_files:
        print("  (no files found)")
    else:
        print(f"\n📊 Total: {len(all_files)} files, {format_size(total_size)}")

    return all_files


def download_checkpoints(
    backend: str = "all",
    local_dir: str = "checkpoints",
    final_only: bool = True
):
    """Download checkpoints from Modal volume to local directory."""
    import modal

    print("🔍 Connecting to Modal volume 'deepseek-checkpoints'...")

    try:
        vol = modal.Volume.from_name("deepseek-checkpoints")
    except modal.exception.NotFoundError:
        print("❌ Volume 'deepseek-checkpoints' not found!")
        return False

    # Determine which backends to download
    backends = []
    if backend == "all":
        backends = ["pytorch", "rust"]
    else:
        backends = [backend]

    local_base = Path(local_dir)
    local_base.mkdir(parents=True, exist_ok=True)

    print(f"📥 Download settings:")
    print(f"   Output directory: {local_base.absolute()}")
    print(f"   Backends: {', '.join(backends)}")
    print(f"   Final only: {final_only}\n")

    # Get list of files
    files_to_download = []
    for entry in vol.listdir("/", recursive=True):
        path = entry.path

        # Skip directories
        if path.endswith("/"):
            continue

        # Skip empty files
        size = getattr(entry, 'size', 0) or 0
        if size == 0:
            continue

        # Filter by backend
        should_download = False
        for b in backends:
            if f"/{b}/" in path or path.startswith(f"{b}/") or path.startswith(f"{b}_"):
                should_download = True
                break

        if not should_download:
            continue

        # Filter for final checkpoints only if requested
        if final_only:
            # Keep: latest.pt, step_10000.pt, step_5000.pt (final steps), final/
            is_final = (
                "latest" in path or
                "final" in path or
                "step_10000" in path or
                "step_5000" in path
            )
            if not is_final:
                continue

        files_to_download.append((path, size))

    if not files_to_download:
        print("❌ No files matched the criteria.")
        return False

    # Calculate total size
    total_size = sum(s for _, s in files_to_download)
    print(f"📦 Files to download: {len(files_to_download)}")
    print(f"   Total size: {format_size(total_size)}\n")

    # Confirm if large download
    if total_size > 10 * 1024 * 1024 * 1024:  # > 10GB
        print(f"⚠️  Warning: This is a large download ({format_size(total_size)})")
        response = input("   Continue? [y/N]: ")
        if response.lower() != 'y':
            print("   Cancelled.")
            return False

    downloaded_count = 0
    error_count = 0

    for path, size in files_to_download:
        # Create local path
        rel_path = path.lstrip("/")
        local_path = local_base / rel_path

        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download file
        print(f"  ⬇️  {path} ({format_size(size)})")
        try:
            # Read file from volume
            with vol.read_file(path) as f:
                content = f.read()

            # Write to local file
            with open(local_path, "wb") as f:
                f.write(content)

            downloaded_count += 1
        except Exception as e:
            print(f"     ❌ Error: {e}")
            error_count += 1

    print(f"\n✅ Downloaded {downloaded_count} files")
    if error_count > 0:
        print(f"⚠️  {error_count} errors occurred")

    return error_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Export checkpoints from Modal volumes"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available checkpoints"
    )
    parser.add_argument(
        "--download", "-d",
        action="store_true",
        help="Download checkpoints to local directory"
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["all", "pytorch", "rust"],
        default="all",
        help="Which backend checkpoints to download (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        default="checkpoints",
        help="Local output directory (default: checkpoints)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all checkpoints, not just final ones"
    )

    args = parser.parse_args()

    if not args.list and not args.download:
        parser.print_help()
        print("\n⚠️  Please specify --list or --download")
        sys.exit(1)

    if args.list:
        list_checkpoints()

    if args.download:
        success = download_checkpoints(
            backend=args.backend,
            local_dir=args.output,
            final_only=not args.all
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()