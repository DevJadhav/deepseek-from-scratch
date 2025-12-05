#!/usr/bin/env python3
"""
PyTorch+GPU CLI for DeepSeek Training
=====================================

Command-line interface for running DeepSeek training on Modal H100 GPUs.
Provides subcommands for train, evaluate, export, and inference similar to the Rust CLI.

Usage:
    python -m modal_gpu.cli train --config config.json --max-steps 1000
    python -m modal_gpu.cli evaluate --checkpoint ./checkpoints/final
    python -m modal_gpu.cli export --checkpoint ./checkpoints/final --format gguf
    python -m modal_gpu.cli infer --checkpoint ./checkpoints/final --prompt "Once upon a time"
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(
    name="deepseek-pytorch",
    help="DeepSeek PyTorch+GPU Training CLI",
    add_completion=False,
)


# =============================================================================
# Train Command
# =============================================================================


@app.command()
def train(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to training config JSON file",
    ),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Path to resume from checkpoint",
    ),
    output: Path = typer.Option(
        Path("./checkpoints"),
        "--output",
        "-o",
        help="Output directory for checkpoints",
    ),
    max_steps: int = typer.Option(
        10000,
        "--max-steps",
        help="Maximum training steps",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        help="Training batch size per GPU",
    ),
    learning_rate: float = typer.Option(
        1e-4,
        "--lr",
        help="Learning rate",
    ),
    warmup_steps: int = typer.Option(
        100,
        "--warmup-steps",
        help="Number of warmup steps",
    ),
    gradient_accumulation: int = typer.Option(
        4,
        "--grad-accum",
        help="Gradient accumulation steps",
    ),
    use_deepspeed: bool = typer.Option(
        True,
        "--deepspeed/--no-deepspeed",
        help="Use DeepSpeed for distributed training",
    ),
    zero_stage: int = typer.Option(
        2,
        "--zero-stage",
        help="DeepSpeed ZeRO optimization stage (0, 1, 2, or 3)",
    ),
    data_path: Path = typer.Option(
        Path("./data"),
        "--data",
        help="Path to training data",
    ),
    model_size: str = typer.Option(
        "tiny",
        "--model-size",
        help="Model size preset: tiny, small, medium, large",
    ),
    use_amp: bool = typer.Option(
        True,
        "--amp/--no-amp",
        help="Use automatic mixed precision",
    ),
    log_steps: int = typer.Option(
        10,
        "--log-steps",
        help="Log every N steps",
    ),
    save_steps: int = typer.Option(
        500,
        "--save-steps",
        help="Save checkpoint every N steps",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON (for pipeline integration)",
    ),
):
    """
    Train a DeepSeek model on GPU.

    Supports single GPU and distributed training with DeepSpeed ZeRO optimization.
    """
    # Build config
    model_configs = {
        "tiny": {
            "hidden_size": 256,
            "num_layers": 4,
            "num_attention_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 512,
            "vocab_size": 32000,
            "max_position_embeddings": 512,
        },
        "small": {
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
            "num_kv_heads": 4,
            "intermediate_size": 2048,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,
        },
        "medium": {
            "hidden_size": 1024,
            "num_layers": 24,
            "num_attention_heads": 16,
            "num_kv_heads": 4,
            "intermediate_size": 4096,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
        },
        "large": {
            "hidden_size": 2048,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_kv_heads": 8,
            "intermediate_size": 8192,
            "vocab_size": 100000,
            "max_position_embeddings": 8192,
        },
    }

    if config and config.exists():
        with open(config) as f:
            full_config = json.load(f)
        model_config = full_config.get(
            "model", model_configs.get(model_size, model_configs["tiny"])
        )
        training_config = full_config.get("training", {})
    else:
        model_config = model_configs.get(model_size, model_configs["tiny"])
        training_config = {}

    # Override with CLI args
    training_config.update(
        {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "gradient_accumulation_steps": gradient_accumulation,
            "use_amp": use_amp,
            "save_steps": save_steps,
            "log_steps": log_steps,
        }
    )

    distributed_config = {
        "use_deepspeed": use_deepspeed,
        "zero_stage": zero_stage,
    }

    result = {
        "command": "train",
        "model_config": model_config,
        "training_config": training_config,
        "distributed_config": distributed_config,
        "data_path": str(data_path),
        "output_dir": str(output),
        "resume_from": str(checkpoint) if checkpoint else None,
    }

    if json_output:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo("=" * 60)
        typer.echo("DeepSeek PyTorch Training")
        typer.echo("=" * 60)
        typer.echo(f"Model size: {model_size}")
        typer.echo(f"Max steps: {max_steps}")
        typer.echo(f"Batch size: {batch_size}")
        typer.echo(f"Learning rate: {learning_rate}")
        typer.echo(f"DeepSpeed: {use_deepspeed} (ZeRO Stage {zero_stage})")
        typer.echo(f"Output: {output}")
        typer.echo(f"Data: {data_path}")
        if checkpoint:
            typer.echo(f"Resuming from: {checkpoint}")
        typer.echo("=" * 60)

        # Check if we can import modal
        try:
            import deepseek.cloud.modal.distributed_trainer  # noqa: F401

            typer.echo("\nStarting training on Modal...")
            # This would run on Modal
            typer.echo("Note: Run with `modal run modal_gpu/cli.py` for actual GPU execution")
        except ImportError as e:
            typer.echo(f"\nModal not available: {e}")
            typer.echo("Run locally or use `modal run` for GPU execution")

    return result


# =============================================================================
# Evaluate Command
# =============================================================================


@app.command()
def evaluate(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to model checkpoint",
    ),
    data_path: Path | None = typer.Option(
        None,
        "--data",
        help="Path to evaluation data",
    ),
    max_batches: int = typer.Option(
        100,
        "--max-batches",
        help="Maximum number of batches to evaluate",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        help="Evaluation batch size",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
):
    """
    Evaluate a trained model checkpoint.

    Computes perplexity and other metrics on the evaluation dataset.
    """
    result = {
        "command": "evaluate",
        "checkpoint": str(checkpoint),
        "data_path": str(data_path) if data_path else None,
        "max_batches": max_batches,
        "batch_size": batch_size,
    }

    if json_output:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo("=" * 60)
        typer.echo("DeepSeek Model Evaluation")
        typer.echo("=" * 60)
        typer.echo(f"Checkpoint: {checkpoint}")
        typer.echo(f"Max batches: {max_batches}")
        typer.echo(f"Batch size: {batch_size}")
        if data_path:
            typer.echo(f"Data: {data_path}")
        typer.echo("=" * 60)

        if not checkpoint.exists():
            typer.echo(f"Error: Checkpoint not found at {checkpoint}", err=True)
            raise typer.Exit(1)

        typer.echo("\nEvaluation would run here...")
        typer.echo("Note: Run with `modal run` for actual GPU execution")

    return result


# =============================================================================
# Export Command
# =============================================================================


@app.command()
def export(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to model checkpoint",
    ),
    output: Path = typer.Option(
        Path("./exports"),
        "--output",
        "-o",
        help="Output directory for exported model",
    ),
    format: str = typer.Option(
        "safetensors",
        "--format",
        "-f",
        help="Export format: safetensors, gguf, onnx",
    ),
    quantize: str | None = typer.Option(
        None,
        "--quantize",
        "-q",
        help="Quantization: q4_0, q4_1, q5_0, q5_1, q8_0",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
):
    """
    Export a trained model to different formats.

    Supports safetensors, GGUF (for llama.cpp), and ONNX formats.
    """
    result = {
        "command": "export",
        "checkpoint": str(checkpoint),
        "output": str(output),
        "format": format,
        "quantize": quantize,
    }

    if json_output:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo("=" * 60)
        typer.echo("DeepSeek Model Export")
        typer.echo("=" * 60)
        typer.echo(f"Checkpoint: {checkpoint}")
        typer.echo(f"Output: {output}")
        typer.echo(f"Format: {format}")
        if quantize:
            typer.echo(f"Quantization: {quantize}")
        typer.echo("=" * 60)

        if not checkpoint.exists():
            typer.echo(f"Error: Checkpoint not found at {checkpoint}", err=True)
            raise typer.Exit(1)

        output.mkdir(parents=True, exist_ok=True)

        if format == "gguf":
            typer.echo("\nGGUF export would run here...")
            typer.echo("Uses scripts/export_gguf.py")
        elif format == "safetensors":
            typer.echo("\nSafetensors export would run here...")
        elif format == "onnx":
            typer.echo("\nONNX export would run here...")
        else:
            typer.echo(f"Unknown format: {format}", err=True)
            raise typer.Exit(1)

    return result


# =============================================================================
# Inference Command
# =============================================================================


@app.command()
def infer(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to model checkpoint",
    ),
    prompt: str = typer.Option(
        "Once upon a time",
        "--prompt",
        "-p",
        help="Input prompt for generation",
    ),
    max_tokens: int = typer.Option(
        100,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Sampling temperature",
    ),
    top_p: float = typer.Option(
        0.9,
        "--top-p",
        help="Top-p sampling parameter",
    ),
    top_k: int = typer.Option(
        50,
        "--top-k",
        help="Top-k sampling parameter",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
):
    """
    Run inference with a trained model.

    Generates text from a prompt using the specified model checkpoint.
    """
    result = {
        "command": "infer",
        "checkpoint": str(checkpoint),
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    if json_output:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo("=" * 60)
        typer.echo("DeepSeek Inference")
        typer.echo("=" * 60)
        typer.echo(f"Checkpoint: {checkpoint}")
        typer.echo(f"Prompt: {prompt}")
        typer.echo(f"Max tokens: {max_tokens}")
        typer.echo(f"Temperature: {temperature}")
        typer.echo(f"Top-p: {top_p}, Top-k: {top_k}")
        typer.echo("=" * 60)

        if not checkpoint.exists():
            typer.echo(f"Error: Checkpoint not found at {checkpoint}", err=True)
            raise typer.Exit(1)

        typer.echo("\nGeneration would run here...")
        typer.echo("Note: Run with `modal run` for actual GPU execution")

    return result


# =============================================================================
# Demo Command
# =============================================================================


@app.command()
def demo():
    """
    Run a quick demo to verify setup.

    Tests basic functionality without requiring a trained model.
    """
    typer.echo("=" * 60)
    typer.echo("DeepSeek PyTorch Demo")
    typer.echo("=" * 60)

    # Check dependencies
    typer.echo("\nChecking dependencies...")

    try:
        import torch

        typer.echo(f"  ✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            typer.echo(f"  ✓ CUDA available ({torch.cuda.device_count()} devices)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            typer.echo("  ✓ MPS (Apple Silicon) available")
        else:
            typer.echo("  ⚠ No GPU available (CPU only)")
    except ImportError:
        typer.echo("  ✗ PyTorch not installed")

    try:
        import modal

        typer.echo(f"  ✓ Modal {modal.__version__}")
    except ImportError:
        typer.echo("  ⚠ Modal not installed (needed for cloud GPU)")

    try:
        import transformers

        typer.echo(f"  ✓ Transformers {transformers.__version__}")
    except ImportError:
        typer.echo("  ⚠ Transformers not installed")

    try:
        import deepspeed

        typer.echo(f"  ✓ DeepSpeed {deepspeed.__version__}")
    except ImportError:
        typer.echo("  ⚠ DeepSpeed not installed (needed for distributed training)")

    typer.echo("\n" + "=" * 60)
    typer.echo("Demo complete!")
    typer.echo("=" * 60)


# =============================================================================
# Status Command
# =============================================================================


@app.command()
def status(
    checkpoint_dir: Path = typer.Option(
        Path("./checkpoints"),
        "--checkpoint-dir",
        help="Directory to check for checkpoints",
    ),
):
    """
    Show status of training checkpoints.
    """
    typer.echo("=" * 60)
    typer.echo("DeepSeek Training Status")
    typer.echo("=" * 60)

    if not checkpoint_dir.exists():
        typer.echo(f"Checkpoint directory not found: {checkpoint_dir}")
        return

    checkpoints = list(checkpoint_dir.glob("**/training_state.json"))
    if not checkpoints:
        checkpoints = list(checkpoint_dir.glob("**/model.safetensors"))

    if not checkpoints:
        typer.echo("No checkpoints found")
        return

    typer.echo(f"Found {len(checkpoints)} checkpoint(s):\n")

    for ckpt in sorted(checkpoints):
        ckpt_dir = ckpt.parent
        typer.echo(f"  {ckpt_dir.name}/")

        # Try to load training state
        state_file = ckpt_dir / "training_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            typer.echo(f"    Step: {state.get('global_step', 'N/A')}")
            typer.echo(f"    Loss: {state.get('best_loss', state.get('last_loss', 'N/A'))}")

    typer.echo("\n" + "=" * 60)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
