#!/usr/bin/env python3
"""
Export checkpoints from Modal volumes to SafeTensors format.

This script runs on Modal to:
1. Load PyTorch checkpoints from the Modal volume
2. Extract just the model weights (not optimizer state)
3. Save as SafeTensors format (much smaller than full checkpoints)
4. Download the exported files locally

Usage:
    # List available checkpoints
    uv run modal run scripts/export_safetensors_modal.py::list_checkpoints

    # Export final PyTorch 512M checkpoint to SafeTensors
    uv run modal run scripts/export_safetensors_modal.py::export_checkpoint \
        --checkpoint-path pytorch/512M/step_10000.pt

    # Export and download locally
    uv run python scripts/export_safetensors_modal.py --download
"""

import modal

# Create Modal app
app = modal.App("deepseek-checkpoint-export")

# Volume for checkpoints
checkpoint_volume = modal.Volume.from_name(
    "deepseek-checkpoints",
    create_if_missing=True,
)

# Export volume for SafeTensors output
export_volume = modal.Volume.from_name(
    "deepseek-exports",
    create_if_missing=True,
)

# Simple image with torch and safetensors
export_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "packaging>=21.0",
    )
)


@app.function(
    image=export_image,
    volumes={
        "/checkpoints": checkpoint_volume,
        "/exports": export_volume,
    },
    timeout=3600,
)
def list_checkpoints():
    """List all available checkpoints in the volume."""
    import os
    from pathlib import Path

    print("📁 Available checkpoints:\n")

    checkpoint_dir = Path("/checkpoints")
    total_size = 0
    files = []

    for root, dirs, filenames in os.walk(checkpoint_dir):
        for filename in filenames:
            filepath = Path(root) / filename
            rel_path = filepath.relative_to(checkpoint_dir)
            size = filepath.stat().st_size
            total_size += size
            files.append((str(rel_path), size))

    # Sort by path
    files.sort(key=lambda x: x[0])

    for path, size in files:
        size_mb = size / (1024 * 1024)
        size_gb = size / (1024 * 1024 * 1024)
        if size_gb >= 1:
            print(f"  {path} ({size_gb:.2f} GB)")
        else:
            print(f"  {path} ({size_mb:.2f} MB)")

    print(f"\n📊 Total: {len(files)} files, {total_size / (1024**3):.2f} GB")
    return files


@app.function(
    image=export_image,
    volumes={
        "/checkpoints": checkpoint_volume,
        "/exports": export_volume,
    },
    timeout=3600,
    memory=32768,  # 32GB memory for large checkpoints
)
def export_checkpoint(checkpoint_path: str, output_name: str = None):
    """Export a checkpoint to SafeTensors format (model weights only)."""
    import torch
    from pathlib import Path
    from safetensors.torch import save_file

    checkpoint_file = Path("/checkpoints") / checkpoint_path

    if not checkpoint_file.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None

    print(f"📂 Loading checkpoint: {checkpoint_path}")
    print(f"   Size: {checkpoint_file.stat().st_size / (1024**3):.2f} GB")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

    # Extract model state dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("   Found model_state_dict key")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("   Found state_dict key")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("   Found model key")
        else:
            # Assume the whole thing is a state dict
            state_dict = {k: v for k, v in checkpoint.items()
                         if isinstance(v, torch.Tensor)}
            print(f"   Extracted {len(state_dict)} tensors from checkpoint")
    else:
        print("❌ Unexpected checkpoint format")
        return None

    # Print checkpoint info
    if isinstance(checkpoint, dict):
        print(f"\n📋 Checkpoint info:")
        for key in checkpoint.keys():
            if key not in ["model_state_dict", "state_dict", "model", "optimizer_state_dict", "optimizer"]:
                value = checkpoint[key]
                if not isinstance(value, (dict, torch.Tensor)):
                    print(f"   {key}: {value}")

    # Generate output name
    if output_name is None:
        # Extract model size and step from path
        parts = checkpoint_path.replace(".pt", "").split("/")
        output_name = "_".join(parts) + ".safetensors"

    output_path = Path("/exports") / output_name

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving to: {output_name}")
    print(f"   Tensors: {len(state_dict)}")

    # Calculate total parameters
    total_params = sum(t.numel() for t in state_dict.values())
    print(f"   Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Save as SafeTensors
    save_file(state_dict, str(output_path))

    # Commit to volume
    export_volume.commit()

    output_size = output_path.stat().st_size
    print(f"   Output size: {output_size / (1024**3):.2f} GB")
    print(f"\n✅ Export complete: {output_name}")

    return str(output_name)


@app.function(
    image=export_image,
    volumes={
        "/checkpoints": checkpoint_volume,
        "/exports": export_volume,
    },
    timeout=7200,
    memory=32768,
)
def export_all_finals():
    """Export all final checkpoints to SafeTensors format."""
    import os
    from pathlib import Path

    checkpoint_dir = Path("/checkpoints")
    exported = []

    # Find all "final" checkpoints
    final_patterns = ["latest.pt", "step_10000.pt", "step_5000.pt"]

    for root, dirs, filenames in os.walk(checkpoint_dir):
        for filename in filenames:
            # Check if it's a final checkpoint
            is_final = any(p in filename for p in final_patterns)
            if not is_final:
                continue

            filepath = Path(root) / filename
            rel_path = filepath.relative_to(checkpoint_dir)

            print(f"\n{'='*60}")
            result = export_checkpoint.local(str(rel_path))
            if result:
                exported.append(result)

    print(f"\n{'='*60}")
    print(f"✅ Exported {len(exported)} checkpoints:")
    for name in exported:
        print(f"   - {name}")

    return exported


@app.function(
    image=export_image,
    volumes={
        "/exports": export_volume,
    },
    timeout=600,
)
def list_exports():
    """List all exported SafeTensors files."""
    import os
    from pathlib import Path

    print("📁 Exported SafeTensors files:\n")

    export_dir = Path("/exports")
    files = []

    if export_dir.exists():
        for root, dirs, filenames in os.walk(export_dir):
            for filename in filenames:
                if filename.endswith(".safetensors"):
                    filepath = Path(root) / filename
                    rel_path = filepath.relative_to(export_dir)
                    size = filepath.stat().st_size
                    files.append((str(rel_path), size))

    if not files:
        print("  (no exports found)")
        print("\n  Run 'export_checkpoint' or 'export_all_finals' first.")
    else:
        files.sort(key=lambda x: x[0])
        total_size = 0
        for path, size in files:
            size_gb = size / (1024 * 1024 * 1024)
            print(f"  {path} ({size_gb:.2f} GB)")
            total_size += size
        print(f"\n📊 Total: {len(files)} files, {total_size / (1024**3):.2f} GB")

    return files


@app.local_entrypoint()
def main(
    action: str = "list",
    checkpoint: str = None,
    download: bool = False,
):
    """
    Main entry point for checkpoint export.

    Args:
        action: One of 'list', 'export', 'export-all', 'list-exports'
        checkpoint: Path to specific checkpoint (for 'export' action)
        download: Whether to download exports locally
    """
    if action == "list":
        list_checkpoints.remote()
    elif action == "export" and checkpoint:
        export_checkpoint.remote(checkpoint)
    elif action == "export-all":
        export_all_finals.remote()
    elif action == "list-exports":
        list_exports.remote()
    else:
        print("Usage:")
        print("  modal run scripts/export_safetensors_modal.py --action list")
        print("  modal run scripts/export_safetensors_modal.py --action export --checkpoint pytorch/512M/step_10000.pt")
        print("  modal run scripts/export_safetensors_modal.py --action export-all")
        print("  modal run scripts/export_safetensors_modal.py --action list-exports")
