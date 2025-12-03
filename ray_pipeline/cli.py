"""Command-line interface for the Ray pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from ray_pipeline.config import Backend, ModelSize, PipelineConfig, Stage
from ray_pipeline.workflow import run_pipeline, DeepSeekWorkflow

app = typer.Typer(add_completion=False, help="DeepSeek Ray Pipeline CLI")


def _load_config(
    config_path: Optional[Path],
    model_size: ModelSize,
    backend: Backend,
) -> PipelineConfig:
    if config_path:
        return PipelineConfig.load(str(config_path))
    return PipelineConfig.from_size(model_size, backend=backend)


@app.command()
def run(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to pipeline config JSON file",
    ),
    model_size: ModelSize = typer.Option(
        ModelSize.SMALL,
        "--model-size",
        help="Model size preset when no config file is provided",
    ),
    backend: Backend = typer.Option(
        Backend.AUTO,
        "--backend",
        help="Preferred backend (defaults to auto-detect)",
    ),
    stages: Optional[List[Stage]] = typer.Option(
        None,
        "--stage",
        help="Subset of stages to run (can be repeated)",
    ),
    no_workflow: bool = typer.Option(
        False,
        "--no-workflow",
        help="Run sequentially without Ray Workflow",
    ),
    time_sliced: bool = typer.Option(
        False,
        "--time-sliced",
        help="Run time-sliced wave execution (4 waves alternating Rust/Python)",
    ),
    gpus: int = typer.Option(
        3,
        "--gpus",
        help="Number of GPUs for time-sliced execution",
    ),
    pp_size: int = typer.Option(
        3,
        "--pp-size",
        help="Pipeline parallel size",
    ),
    max_steps: int = typer.Option(
        20000,
        "--max-steps",
        help="Maximum training steps",
    ),
):
    """Execute the configured pipeline."""
    if time_sliced:
        cfg = PipelineConfig.production_3gpu_time_sliced()
        cfg.time_sliced.gpu_ids = list(range(gpus))
        cfg.time_sliced.pipeline_parallel_size = pp_size
        cfg.training.max_steps = max_steps
        cfg.distributed.num_workers = gpus
        cfg.distributed.pipeline_parallel_size = pp_size
    else:
        cfg = _load_config(config, model_size, backend)
    
    if stages:
        cfg.stages_to_run = list(stages)
    
    typer.echo(cfg.summary())
    
    if time_sliced:
        workflow = DeepSeekWorkflow(cfg)
        context = workflow.run_time_sliced_waves()
    else:
        context = run_pipeline(cfg, use_ray=not no_workflow)
    
    typer.echo("Pipeline completed. Final metadata:")
    typer.echo(context.metadata)


@app.command()
def run_rust(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    gpus: int = typer.Option(3, "--gpus", help="Number of GPUs"),
    pp_size: int = typer.Option(3, "--pp-size", help="Pipeline parallel size"),
    max_steps: int = typer.Option(10000, "--max-steps", help="Maximum training steps"),
    start_step: int = typer.Option(0, "--start-step", help="Starting step"),
    checkpoint_from: Optional[Path] = typer.Option(None, "--checkpoint-from", help="Load checkpoint from path"),
):
    """
    Run Rust backend waves only (Waves 1 & 3).
    
    Wave 1: MQA/GQA/MLA/DeepSeek Attention
    Wave 3: GRPO/R1/DPO/Reward
    
    Example:
        python -m ray_pipeline.cli run-rust --gpus 3 --pp-size 3 --max-steps 10000
    """
    from ray_pipeline.config import WaveBackend, WaveConfig, TimeSlicedConfig
    from ray_pipeline.workflow import DeepSeekWorkflow
    
    cfg = PipelineConfig.production_3gpu_time_sliced()
    cfg.time_sliced.gpu_ids = list(range(gpus))
    cfg.time_sliced.pipeline_parallel_size = pp_size
    cfg.distributed.num_workers = gpus
    cfg.distributed.pipeline_parallel_size = pp_size
    
    # Configure Rust-only waves (1 & 3)
    steps_per_wave = max_steps // 2
    cfg.time_sliced.waves = [
        WaveConfig(
            wave_id=1,
            backend=WaveBackend.RUST,
            start_step=start_step,
            end_step=start_step + steps_per_wave,
            stages=["MQA", "GQA", "MLA", "DeepSeek Attention"],
            checkpoint_from=str(checkpoint_from) if checkpoint_from else None,
        ),
        WaveConfig(
            wave_id=3,
            backend=WaveBackend.RUST,
            start_step=start_step + steps_per_wave,
            end_step=start_step + max_steps,
            stages=["GRPO", "R1", "DPO", "Reward"],
            checkpoint_from=f"checkpoints/step_{start_step + steps_per_wave}/",
        ),
    ]
    cfg.time_sliced.num_waves = 2
    cfg.training.max_steps = start_step + max_steps
    
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo("  Rust Backend Execution (Waves 1 & 3)")
    typer.echo(f"  GPUs: {gpus} | PP: {pp_size} | Steps: {start_step}-{start_step + max_steps}")
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo(cfg.summary())
    
    workflow = DeepSeekWorkflow(cfg)
    context = workflow.run_time_sliced_waves()
    
    typer.echo("Rust waves completed. Metadata:")
    typer.echo(context.metadata)


@app.command()
def run_python(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    gpus: int = typer.Option(3, "--gpus", help="Number of GPUs"),
    pp_size: int = typer.Option(3, "--pp-size", help="Pipeline parallel size"),
    max_steps: int = typer.Option(10000, "--max-steps", help="Maximum training steps"),
    start_step: int = typer.Option(0, "--start-step", help="Starting step"),
    checkpoint_from: Optional[Path] = typer.Option(None, "--checkpoint-from", help="Load checkpoint from path"),
):
    """
    Run Python/PyTorch backend waves only (Waves 2 & 4).
    
    Wave 2: Standard MOE/DeepSeek MOE
    Wave 4: MTP/FP8/Distillation/5D Parallelism
    
    Example:
        python -m ray_pipeline.cli run-python --gpus 3 --pp-size 3 --max-steps 10000 --checkpoint-from checkpoints/step_5000
    """
    from ray_pipeline.config import WaveBackend, WaveConfig, TimeSlicedConfig
    from ray_pipeline.workflow import DeepSeekWorkflow
    
    cfg = PipelineConfig.production_3gpu_time_sliced()
    cfg.time_sliced.gpu_ids = list(range(gpus))
    cfg.time_sliced.pipeline_parallel_size = pp_size
    cfg.distributed.num_workers = gpus
    cfg.distributed.pipeline_parallel_size = pp_size
    
    # Configure Python-only waves (2 & 4)
    steps_per_wave = max_steps // 2
    cfg.time_sliced.waves = [
        WaveConfig(
            wave_id=2,
            backend=WaveBackend.PYTHON,
            start_step=start_step,
            end_step=start_step + steps_per_wave,
            stages=["Standard MOE", "DeepSeek MOE"],
            checkpoint_from=str(checkpoint_from) if checkpoint_from else None,
        ),
        WaveConfig(
            wave_id=4,
            backend=WaveBackend.PYTHON,
            start_step=start_step + steps_per_wave,
            end_step=start_step + max_steps,
            stages=["MTP", "FP8", "Distillation", "5D Parallelism"],
            checkpoint_from=f"checkpoints/step_{start_step + steps_per_wave}/",
        ),
    ]
    cfg.time_sliced.num_waves = 2
    cfg.training.max_steps = start_step + max_steps
    
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo("  Python/PyTorch Backend Execution (Waves 2 & 4)")
    typer.echo(f"  GPUs: {gpus} | PP: {pp_size} | Steps: {start_step}-{start_step + max_steps}")
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo(cfg.summary())
    
    workflow = DeepSeekWorkflow(cfg)
    context = workflow.run_time_sliced_waves()
    
    typer.echo("Python waves completed. Metadata:")
    typer.echo(context.metadata)


@app.command()
def run_mlx(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    max_steps: int = typer.Option(3000, "--max-steps", help="Maximum training steps"),
    start_step: int = typer.Option(0, "--start-step", help="Starting step"),
    checkpoint_from: Optional[Path] = typer.Option(None, "--checkpoint-from", help="Load checkpoint from path"),
    batch_size: int = typer.Option(2, "--batch-size", help="Batch size (lower = less memory)"),
    d_model: int = typer.Option(32, "--d-model", help="Model dimension (lower = less memory)"),
    num_layers: int = typer.Option(2, "--num-layers", help="Number of layers (lower = less memory)"),
):
    """
    Run Python/MLX backend waves on Apple Silicon.
    
    All 4 waves use MLX backend for unified Apple Silicon training.
    
    Example:
        python -m ray_pipeline.cli run-mlx --max-steps 3000
    """
    from ray_pipeline.config import WaveBackend, WaveConfig, TimeSlicedConfig
    from ray_pipeline.workflow import DeepSeekWorkflow
    
    cfg = PipelineConfig.production_3gpu_time_sliced()
    # MLX doesn't use GPU IDs the same way
    cfg.time_sliced.gpu_ids = [0]  # Apple Silicon unified memory
    cfg.time_sliced.pipeline_parallel_size = 1  # No PP for MLX
    cfg.distributed.num_workers = 1
    cfg.distributed.pipeline_parallel_size = 1
    
    # Memory-conscious settings for Apple Silicon
    cfg.training.batch_size = batch_size
    cfg.training.gradient_accumulation_steps = 8  # Compensate for small batch
    cfg.model.d_model = d_model
    cfg.model.num_layers = num_layers
    cfg.model.num_heads = max(2, d_model // 64)  # Scale heads with d_model
    
    # Configure MLX-only waves (all 4)
    steps_per_wave = max_steps // 4
    cfg.time_sliced.waves = [
        WaveConfig(
            wave_id=1,
            backend=WaveBackend.MLX,
            start_step=start_step,
            end_step=start_step + steps_per_wave,
            stages=["MQA", "GQA", "MLA", "DeepSeek Attention"],
            checkpoint_from=str(checkpoint_from) if checkpoint_from else None,
        ),
        WaveConfig(
            wave_id=2,
            backend=WaveBackend.MLX,
            start_step=start_step + steps_per_wave,
            end_step=start_step + steps_per_wave * 2,
            stages=["Standard MOE", "DeepSeek MOE"],
            checkpoint_from=f"checkpoints/step_{start_step + steps_per_wave}/",
        ),
        WaveConfig(
            wave_id=3,
            backend=WaveBackend.MLX,
            start_step=start_step + steps_per_wave * 2,
            end_step=start_step + steps_per_wave * 3,
            stages=["GRPO", "R1", "DPO", "Reward"],
            checkpoint_from=f"checkpoints/step_{start_step + steps_per_wave * 2}/",
        ),
        WaveConfig(
            wave_id=4,
            backend=WaveBackend.MLX,
            start_step=start_step + steps_per_wave * 3,
            end_step=start_step + max_steps,
            stages=["MTP", "FP8", "Distillation", "5D Parallelism"],
            checkpoint_from=f"checkpoints/step_{start_step + steps_per_wave * 3}/",
        ),
    ]
    cfg.time_sliced.num_waves = 4
    cfg.training.max_steps = start_step + max_steps
    
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo("  MLX Backend Execution (All Waves on Apple Silicon)")
    typer.echo(f"  Device: Apple Silicon GPU | Steps: {start_step}-{start_step + max_steps}")
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo(cfg.summary())
    
    workflow = DeepSeekWorkflow(cfg)
    context = workflow.run_time_sliced_waves()
    
    typer.echo("MLX waves completed. Metadata:")
    typer.echo(context.metadata)


@app.command()
def benchmark(
    max_steps: int = typer.Option(3000, "--max-steps", help="Maximum training steps per run"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file for results"),
    mlx_batch_size: int = typer.Option(2, "--mlx-batch-size", help="Batch size for MLX (memory control)"),
):
    """
    Run all three backends and benchmark their performance.
    
    Runs Rust+GPU, Python+GPU, and Python+MLX sequentially and compares:
    - Total training time
    - Steps per second
    - Final validation loss
    
    Example:
        python -m ray_pipeline.cli benchmark --max-steps 3000 --output benchmark_results.json
    """
    import time
    import json
    from ray_pipeline.config import WaveBackend, WaveConfig
    from ray_pipeline.workflow import DeepSeekWorkflow
    
    results = {}
    
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo("  BENCHMARK: Rust+GPU vs Python+GPU vs MLX")
    typer.echo(f"  Max Steps: {max_steps}")
    typer.echo("═══════════════════════════════════════════════════════════\n")
    
    # Helper to run a benchmark
    def run_bench(name: str, backend: WaveBackend, use_gpu: bool = True):
        typer.echo(f"\n▶ Starting {name} benchmark...")
        start_time = time.time()
        
        cfg = PipelineConfig.production_3gpu_time_sliced()
        steps_per_wave = max_steps // 4
        
        if backend == WaveBackend.MLX:
            cfg.time_sliced.gpu_ids = [0]
            cfg.time_sliced.pipeline_parallel_size = 1
            cfg.distributed.num_workers = 1
            # Memory-conscious settings for Apple Silicon
            cfg.training.batch_size = mlx_batch_size
            cfg.training.gradient_accumulation_steps = 8
            cfg.model.d_model = 128
            cfg.model.num_layers = 2
            cfg.model.num_heads = 2
        else:
            cfg.time_sliced.gpu_ids = [0, 1, 2]
            cfg.time_sliced.pipeline_parallel_size = 3
            cfg.distributed.num_workers = 3
        
        cfg.time_sliced.waves = [
            WaveConfig(wave_id=i+1, backend=backend, start_step=i*steps_per_wave, 
                      end_step=(i+1)*steps_per_wave, stages=["Training"])
            for i in range(4)
        ]
        cfg.time_sliced.num_waves = 4
        cfg.training.max_steps = max_steps
        
        try:
            workflow = DeepSeekWorkflow(cfg)
            context = workflow.run_time_sliced_waves()
            
            elapsed = time.time() - start_time
            wave_metrics = context.metadata.get("wave_metrics", {})
            final_loss = min(wave_metrics.values()) if wave_metrics else float("inf")
            
            return {
                "name": name,
                "backend": backend.value,
                "elapsed_seconds": elapsed,
                "steps_per_second": max_steps / elapsed,
                "final_loss": final_loss,
                "wave_metrics": wave_metrics,
                "success": True,
            }
        except Exception as e:
            typer.echo(f"  ✗ {name} failed: {e}", err=True)
            return {
                "name": name,
                "backend": backend.value,
                "success": False,
                "error": str(e),
            }
    
    # Run all benchmarks
    results["rust_gpu"] = run_bench("Rust+GPU (Modal H100)", WaveBackend.RUST)
    results["python_gpu"] = run_bench("Python+GPU (Modal H100)", WaveBackend.PYTHON)
    results["mlx"] = run_bench("MLX (Apple Silicon)", WaveBackend.MLX)
    
    # Print summary
    typer.echo("\n" + "═" * 60)
    typer.echo("  BENCHMARK RESULTS")
    typer.echo("═" * 60)
    
    for key, res in results.items():
        if res.get("success"):
            typer.echo(f"\n{res['name']}:")
            typer.echo(f"  Time: {res['elapsed_seconds']:.1f}s")
            typer.echo(f"  Speed: {res['steps_per_second']:.2f} steps/sec")
            typer.echo(f"  Final Loss: {res['final_loss']:.4f}")
        else:
            typer.echo(f"\n{res.get('name', key)}: FAILED - {res.get('error', 'Unknown error')}")
    
    # Determine winner
    successful = {k: v for k, v in results.items() if v.get("success")}
    if successful:
        fastest = min(successful.values(), key=lambda x: x["elapsed_seconds"])
        best_loss = min(successful.values(), key=lambda x: x["final_loss"])
        
        typer.echo(f"\n🏆 Fastest: {fastest['name']} ({fastest['elapsed_seconds']:.1f}s)")
        typer.echo(f"🏆 Best Loss: {best_loss['name']} ({best_loss['final_loss']:.4f})")
    
    # Save results
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        typer.echo(f"\nResults saved to: {output}")
    
    return results


@app.command()
def config_show(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    model_size: ModelSize = typer.Option(ModelSize.SMALL, "--model-size"),
    backend: Backend = typer.Option(Backend.AUTO, "--backend"),
):
    """Print the configuration summary."""
    cfg = _load_config(config, model_size, backend)
    typer.echo(cfg.summary())


if __name__ == "__main__":
    app()
