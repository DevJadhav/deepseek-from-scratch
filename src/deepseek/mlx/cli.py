#!/usr/bin/env python3
"""
MLX CLI for DeepSeek from Scratch.

This CLI provides the same interface as the Rust CLI for consistency:
- train: Train a model on Apple Silicon
- evaluate: Evaluate a model checkpoint
- export: Export model to various formats
- demo: Run component demonstrations

Usage:
    python -m mlx_impl.cli train --config config.json
    python -m mlx_impl.cli evaluate checkpoints/model.safetensors
    python -m mlx_impl.cli export model.safetensors --format gguf
    python -m mlx_impl.cli demo --component attention
"""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="deepseek-mlx",
    help="DeepSeek from Scratch - MLX Implementation for Apple Silicon",
    add_completion=False,
)


# =============================================================================
# Train Command
# =============================================================================


@app.command()
def train(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to training configuration file (JSON/YAML)",
        ),
    ] = None,
    data_path: Annotated[
        Path | None,
        typer.Option(
            "--data",
            help="Path to training data directory",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for checkpoints",
        ),
    ] = Path("checkpoints/mlx"),
    model_size: Annotated[
        str,
        typer.Option(
            "--model-size",
            help="Model size: tiny, small, base, large",
        ),
    ] = "tiny",
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Training batch size",
        ),
    ] = 4,
    learning_rate: Annotated[
        float,
        typer.Option(
            "--lr",
            help="Learning rate",
        ),
    ] = 1e-4,
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            help="Maximum training steps (0 = use config)",
        ),
    ] = 0,
    gradient_accumulation: Annotated[
        int,
        typer.Option(
            "--grad-accum",
            help="Gradient accumulation steps",
        ),
    ] = 1,
    use_moe: Annotated[
        bool,
        typer.Option(
            "--moe/--no-moe",
            help="Use Mixture of Experts architecture",
        ),
    ] = False,
    use_mla: Annotated[
        bool,
        typer.Option(
            "--mla/--no-mla",
            help="Use Multi-Head Latent Attention",
        ),
    ] = True,
    use_fp8: Annotated[
        bool,
        typer.Option(
            "--fp8/--no-fp8",
            help="Use FP8 quantization (simulated)",
        ),
    ] = False,
    resume: Annotated[
        Path | None,
        typer.Option(
            "--resume",
            help="Resume from checkpoint",
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for reproducibility",
        ),
    ] = 42,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ] = False,
) -> dict:
    """
    Train a DeepSeek model on Apple Silicon using MLX.

    Supports various architectures including MoE, MLA, and MTP.

    Examples:
        # Train tiny model with default settings
        python -m mlx_impl.cli train --model-size tiny

        # Train with custom config
        python -m mlx_impl.cli train --config configs/tiny_mlx_config.py

        # Train with MoE enabled
        python -m mlx_impl.cli train --model-size small --moe
    """
    import mlx.core as mx

    # Set random seed
    mx.random.seed(seed)

    typer.echo("=" * 60)
    typer.echo("DeepSeek MLX Training")
    typer.echo("=" * 60)

    # Build configuration
    training_config = {
        "model_size": model_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "gradient_accumulation": gradient_accumulation,
        "use_moe": use_moe,
        "use_mla": use_mla,
        "use_fp8": use_fp8,
        "seed": seed,
        "output_dir": str(output_dir),
    }

    # Model size configurations
    model_configs = {
        "tiny": {"d_model": 256, "n_layers": 4, "n_heads": 4, "vocab_size": 32000},
        "small": {"d_model": 512, "n_layers": 8, "n_heads": 8, "vocab_size": 32000},
        "base": {"d_model": 1024, "n_layers": 12, "n_heads": 16, "vocab_size": 32000},
        "large": {"d_model": 2048, "n_layers": 24, "n_heads": 32, "vocab_size": 32000},
    }

    if model_size not in model_configs:
        typer.echo(f"Error: Unknown model size '{model_size}'", err=True)
        typer.echo(f"Available: {list(model_configs.keys())}", err=True)
        raise typer.Exit(1)

    model_config = model_configs[model_size]
    training_config.update(model_config)

    if config:
        typer.echo(f"Config: {config}")
    typer.echo(f"Model size: {model_size}")
    typer.echo(f"  - d_model: {model_config['d_model']}")
    typer.echo(f"  - n_layers: {model_config['n_layers']}")
    typer.echo(f"  - n_heads: {model_config['n_heads']}")
    typer.echo(f"Batch size: {batch_size}")
    typer.echo(f"Learning rate: {learning_rate}")
    typer.echo(f"Use MoE: {use_moe}")
    typer.echo(f"Use MLA: {use_mla}")
    typer.echo(f"Output: {output_dir}")

    if data_path:
        typer.echo(f"Data: {data_path}")

    if resume:
        typer.echo(f"Resuming from: {resume}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("\n" + "-" * 60)
    typer.echo("Training would start here...")
    typer.echo("(Full implementation in mlx_impl/training.py)")
    typer.echo("-" * 60)

    if verbose:
        typer.echo("\nVerbose configuration:")
        for key, value in training_config.items():
            typer.echo(f"  {key}: {value}")

    return training_config


# =============================================================================
# Evaluate Command
# =============================================================================


@app.command()
def evaluate(
    checkpoint: Annotated[
        Path,
        typer.Argument(
            help="Path to model checkpoint (.safetensors or directory)",
        ),
    ],
    data_path: Annotated[
        Path | None,
        typer.Option(
            "--data",
            help="Path to evaluation data",
        ),
    ] = None,
    max_batches: Annotated[
        int,
        typer.Option(
            "--max-batches",
            help="Maximum number of batches to evaluate",
        ),
    ] = 100,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Evaluation batch size",
        ),
    ] = 8,
    metrics: Annotated[
        str,
        typer.Option(
            "--metrics",
            help="Comma-separated list of metrics: perplexity,accuracy,loss",
        ),
    ] = "perplexity,loss",
    output_file: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Save results to JSON file",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ] = False,
) -> dict:
    """
    Evaluate a trained model checkpoint.

    Computes metrics like perplexity and loss on evaluation data.

    Examples:
        python -m mlx_impl.cli evaluate checkpoints/model.safetensors

        python -m mlx_impl.cli evaluate model/ --metrics perplexity,accuracy
    """
    typer.echo("=" * 60)
    typer.echo("DeepSeek MLX Evaluation")
    typer.echo("=" * 60)

    typer.echo(f"\nCheckpoint: {checkpoint}")
    if data_path:
        typer.echo(f"Data: {data_path}")
    typer.echo(f"Batch size: {batch_size}")
    typer.echo(f"Max batches: {max_batches}")
    typer.echo(f"Metrics: {metrics}")

    # Parse metrics
    metric_list = [m.strip() for m in metrics.split(",")]

    results = {
        "checkpoint": str(checkpoint),
        "metrics": {},
        "config": {
            "batch_size": batch_size,
            "max_batches": max_batches,
        },
    }

    typer.echo("\n" + "-" * 60)
    typer.echo("Evaluation would run here...")
    typer.echo("(Full implementation in mlx_impl/evaluation.py)")
    typer.echo("-" * 60)

    # Placeholder results
    for metric in metric_list:
        if metric == "perplexity":
            results["metrics"]["perplexity"] = 15.5
        elif metric == "loss":
            results["metrics"]["loss"] = 2.74
        elif metric == "accuracy":
            results["metrics"]["accuracy"] = 0.65

    typer.echo("\nResults:")
    for metric, value in results["metrics"].items():
        typer.echo(f"  {metric}: {value}")

    if output_file:
        import json

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(results, indent=2))
        typer.echo(f"\nResults saved to: {output_file}")

    return results


# =============================================================================
# Export Command
# =============================================================================


@app.command()
def export(
    model_path: Annotated[
        Path,
        typer.Argument(
            help="Path to model checkpoint",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output path for exported model",
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Export format: gguf, safetensors, coreml, mlpackage",
        ),
    ] = "safetensors",
    quantization: Annotated[
        str | None,
        typer.Option(
            "--quantization",
            "-q",
            help="Quantization: q4, q8, fp16, fp32",
        ),
    ] = None,
    vocab_path: Annotated[
        Path | None,
        typer.Option(
            "--vocab",
            help="Path to vocabulary file (for GGUF)",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ] = False,
) -> dict:
    """
    Export a model to various formats.

    Supports GGUF (for llama.cpp), safetensors, and CoreML.

    Examples:
        # Export to GGUF for llama.cpp
        python -m mlx_impl.cli export model.safetensors --format gguf

        # Export with quantization
        python -m mlx_impl.cli export model/ -f gguf -q q4

        # Export to CoreML
        python -m mlx_impl.cli export model/ --format coreml
    """
    typer.echo("=" * 60)
    typer.echo("DeepSeek MLX Export")
    typer.echo("=" * 60)

    valid_formats = ["gguf", "safetensors", "coreml", "mlpackage"]
    if format not in valid_formats:
        typer.echo(f"Error: Invalid format '{format}'", err=True)
        typer.echo(f"Valid formats: {valid_formats}", err=True)
        raise typer.Exit(1)

    valid_quants = ["q4", "q8", "fp16", "fp32", None]
    if quantization not in valid_quants:
        typer.echo(f"Error: Invalid quantization '{quantization}'", err=True)
        typer.echo(f"Valid options: {[q for q in valid_quants if q]}", err=True)
        raise typer.Exit(1)

    # Default output path
    if output_path is None:
        suffix = f".{format}"
        if format == "coreml":
            suffix = ".mlpackage"
        output_path = model_path.with_suffix(suffix)

    typer.echo(f"\nModel: {model_path}")
    typer.echo(f"Format: {format}")
    typer.echo(f"Output: {output_path}")
    if quantization:
        typer.echo(f"Quantization: {quantization}")
    if vocab_path:
        typer.echo(f"Vocab: {vocab_path}")

    result = {
        "model_path": str(model_path),
        "output_path": str(output_path),
        "format": format,
        "quantization": quantization,
        "success": True,
    }

    typer.echo("\n" + "-" * 60)
    typer.echo("Export would run here...")
    typer.echo("(Full implementation in scripts/export_gguf.py)")
    typer.echo("-" * 60)

    return result


# =============================================================================
# Demo Command
# =============================================================================


@app.command()
def demo(
    component: Annotated[
        str | None,
        typer.Option(
            "--component",
            "-c",
            help="Component to demo: attention, moe, mtp, grpo, pipeline, sft, all",
        ),
    ] = "all",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ] = False,
) -> None:
    """
    Run component demonstrations.

    Shows each DeepSeek component in action.

    Examples:
        # Run all demos
        python -m mlx_impl.cli demo

        # Demo specific component
        python -m mlx_impl.cli demo --component attention
    """
    import mlx.core as mx

    typer.echo("=" * 60)
    typer.echo("DeepSeek MLX Component Demos")
    typer.echo("=" * 60)

    valid_components = ["attention", "moe", "mtp", "grpo", "pipeline", "sft", "all"]
    if component not in valid_components:
        typer.echo(f"Error: Invalid component '{component}'", err=True)
        typer.echo(f"Valid components: {valid_components}", err=True)
        raise typer.Exit(1)

    # Set device
    mx.set_default_device(mx.gpu)
    typer.echo(f"\nUsing device: {mx.default_device()}")

    if component in ["attention", "all"]:
        _demo_attention(verbose)

    if component in ["moe", "all"]:
        _demo_moe(verbose)

    if component in ["mtp", "all"]:
        _demo_mtp(verbose)

    if component in ["grpo", "all"]:
        _demo_grpo(verbose)

    if component in ["pipeline", "all"]:
        _demo_pipeline(verbose)

    if component in ["sft", "all"]:
        _demo_sft(verbose)

    typer.echo("\n" + "=" * 60)
    typer.echo("Demo Complete!")
    typer.echo("=" * 60)


def _demo_attention(verbose: bool = False) -> None:
    """Demo attention mechanisms."""
    import mlx.core as mx

    typer.echo("\n--- Attention Mechanisms ---")

    try:
        from attention import (
            GroupedQueryAttention,
            MultiHeadLatentAttention,
            MultiQueryAttention,
        )

        x = mx.random.normal((4, 64, 512))

        # MQA
        mqa = MultiQueryAttention(d_model=512, num_heads=8)
        out = mqa(x)
        typer.echo(f"MQA Output: {out.shape}")

        # GQA
        gqa = GroupedQueryAttention(d_model=512, num_heads=32, num_groups=4)
        out = gqa(x)
        typer.echo(f"GQA Output: {out.shape}")

        # MLA
        mla = MultiHeadLatentAttention(d_model=512, num_heads=8, d_latent=128, d_rope=64)
        out = mla(x)
        typer.echo(f"MLA Output: {out.shape}")

    except ImportError as e:
        typer.echo(f"  Skipped (import error): {e}")


def _demo_moe(verbose: bool = False) -> None:
    """Demo Mixture of Experts."""
    import mlx.core as mx

    typer.echo("\n--- Mixture of Experts ---")

    try:
        from moe import DeepSeekMoE

        x = mx.random.normal((4, 64, 512))
        moe = DeepSeekMoE(
            d_model=512,
            d_hidden=1024,
            num_experts=10,
            num_shared=2,
            num_routed=8,
            top_k=2,
        )
        out = moe(x)
        typer.echo(f"MoE Output: {out.shape}")

    except ImportError as e:
        typer.echo(f"  Skipped (import error): {e}")


def _demo_mtp(verbose: bool = False) -> None:
    """Demo Multi-Token Prediction."""
    import mlx.core as mx

    typer.echo("\n--- Multi-Token Prediction ---")

    try:
        from mtp import MTPModel

        mtp = MTPModel(vocab_size=1000, d_model=512, num_layers=2, k_predictions=1)
        input_ids = mx.random.randint(0, 1000, (4, 64))
        main_logits, future_logits = mtp(input_ids)
        typer.echo(f"Main Logits: {main_logits.shape}")
        typer.echo(f"Future Logits: {future_logits[0].shape}")

    except ImportError as e:
        typer.echo(f"  Skipped (import error): {e}")


def _demo_grpo(verbose: bool = False) -> None:
    """Demo GRPO training."""
    import mlx.core as mx

    typer.echo("\n--- GRPO Training ---")

    try:
        from grpo import GRPOTrainer

        grpo = GRPOTrainer(beta=0.01)
        logits = mx.random.normal((4, 10, 100))
        ref_logits = mx.random.normal((4, 10, 100))
        input_ids = mx.random.randint(0, 100, (4, 10))
        rewards = mx.array([1.0, 0.5, -0.5, 0.0])

        loss = grpo.compute_loss(logits, input_ids, rewards, ref_logits)
        typer.echo(f"GRPO Loss: {loss.item():.4f}")

    except ImportError as e:
        typer.echo(f"  Skipped (import error): {e}")


def _demo_pipeline(verbose: bool = False) -> None:
    """Demo training pipeline components."""
    typer.echo("\n--- Training Pipeline ---")

    try:
        from pipeline import CurriculumScheduler, DataMixer, PipelineConfig, ScalingLaws

        # Scaling Laws
        scaling = ScalingLaws()
        typer.echo(f"Predicted loss (7B, 2T): {scaling.predict_loss(7e9, 2e12):.4f}")

        # Data Mixing
        mixing = DataMixer(
            {
                "web": 0.60,
                "code": 0.20,
                "math": 0.10,
                "books": 0.05,
                "scientific": 0.05,
            }
        )
        probs = mixing.get_probs()
        typer.echo(f"Data mix: {len(probs)} domains")

        # Curriculum
        curriculum = CurriculumScheduler(512, 4096, 10000, 5000)
        typer.echo(f"Seq length at step 0: {curriculum.get_seq_length(0)}")
        typer.echo(f"Seq length at step 10000: {curriculum.get_seq_length(10000)}")

        # Pipeline config
        config = PipelineConfig()
        typer.echo(f"Model size: {config.model_size / 1e9:.1f}B")

    except ImportError as e:
        typer.echo(f"  Skipped (import error): {e}")


def _demo_sft(verbose: bool = False) -> None:
    """Demo supervised fine-tuning components."""
    typer.echo("\n--- Supervised Fine-Tuning ---")

    try:
        from sft import DeepSeekChatTemplate, SFTConfig

        template = DeepSeekChatTemplate()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        formatted = template.format_conversation(messages)
        typer.echo(f"Chat template formatted: {len(formatted)} chars")

        config = SFTConfig()
        typer.echo(f"LoRA rank: {config.lora_r}")
        typer.echo(f"NEFTune enabled: {config.use_neftune}")

    except ImportError as e:
        typer.echo(f"  Skipped (import error): {e}")


# =============================================================================
# Status Command
# =============================================================================


@app.command()
def status() -> None:
    """
    Show MLX environment status and available components.
    """
    typer.echo("=" * 60)
    typer.echo("DeepSeek MLX Status")
    typer.echo("=" * 60)

    # Check MLX
    typer.echo("\n--- MLX Environment ---")
    try:
        import mlx.core as mx

        typer.echo(f"MLX version: {mx.__version__}")
        typer.echo(f"Default device: {mx.default_device()}")
    except ImportError:
        typer.echo("MLX: NOT INSTALLED")
        raise typer.Exit(1) from None

    # Check components
    typer.echo("\n--- Components ---")
    components = {
        "attention": "attention",
        "moe": "moe",
        "mtp": "mtp",
        "grpo": "grpo",
        "pipeline": "pipeline",
        "sft": "sft",
        "reward_model": "reward_model",
        "distillation": "distillation",
    }

    for name, module in components.items():
        try:
            __import__(module)
            typer.echo(f"  {name}: ✓")
        except ImportError as e:
            typer.echo(f"  {name}: ✗ ({e})")


# =============================================================================
# Version Callback
# =============================================================================


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        typer.echo("DeepSeek MLX CLI v0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = None,
) -> None:
    """
    DeepSeek from Scratch - MLX Implementation.

    A command-line interface for training, evaluating, and exporting
    DeepSeek models on Apple Silicon using MLX.
    """


if __name__ == "__main__":
    app()
