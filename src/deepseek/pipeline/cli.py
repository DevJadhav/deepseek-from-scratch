"""Command-line interface for the Ray pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from deepseek.pipeline.config import Backend, ModelSize, PipelineConfig, Stage
from deepseek.pipeline.workflow import run_pipeline, DeepSeekWorkflow

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
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Path to training data directory"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Path to output/checkpoint directory"),
    gpus: int = typer.Option(1, "--gpus", help="Number of GPUs"),
    pp_size: int = typer.Option(1, "--pp-size", help="Pipeline parallel size"),
    max_steps: int = typer.Option(100, "--max-steps", help="Maximum training steps"),
    start_step: int = typer.Option(0, "--start-step", help="Starting step"),
    checkpoint_from: Optional[Path] = typer.Option(None, "--checkpoint-from", help="Load checkpoint from path"),
    use_cuda: bool = typer.Option(False, "--cuda/--no-cuda", help="Use CUDA GPU (default: False, uses Metal on macOS)"),
    use_metal: bool = typer.Option(True, "--metal/--no-metal", help="Use Metal GPU on macOS (default: True)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run mode (don't execute Rust binary)"),
    verify_only: bool = typer.Option(False, "--verify-only", help="Quick verification run (dry run with tiny config)"),
):
    """
    Run Rust backend waves only (Waves 1 & 3).
    
    Wave 1: MQA/GQA/MLA/DeepSeek Attention
    Wave 3: GRPO/R1/DPO/Reward
    
    Example:
        python -m deepseek.pipeline.cli run-rust --verify-only  # Quick verification (dry run)
        python -m deepseek.pipeline.cli run-rust --max-steps 100 --data-dir ./data/fineweb-edu/train
        python -m deepseek.pipeline.cli run-rust --cuda --max-steps 1000  # Use CUDA
    """
    from deepseek.pipeline.config import WaveBackend, WaveConfig
    from deepseek.pipeline.workflow import DeepSeekWorkflow
    
    # Quick verification mode - dry run with tiny settings
    if verify_only:
        max_steps = 10
        dry_run = True
        gpus = 1
        pp_size = 1
        typer.echo("[VERIFY MODE] Using dry run with minimal settings")
    
    cfg = PipelineConfig.production_3gpu_time_sliced()
    cfg.time_sliced.gpu_ids = list(range(gpus))
    cfg.time_sliced.pipeline_parallel_size = pp_size
    cfg.distributed.num_workers = gpus
    cfg.distributed.pipeline_parallel_size = pp_size
    
    # Set data directory if provided
    if data_dir:
        cfg.data.data_dir = str(data_dir)
        typer.echo(f"Using data directory: {data_dir}")
    
    # Set output directory if provided
    if output_dir:
        cfg.training.checkpoint_dir = str(output_dir)
        typer.echo(f"Using output directory: {output_dir}")
    
    # Select Rust backend based on GPU options
    if use_cuda:
        rust_backend = WaveBackend.RUST_CUDA
        backend_name = "Rust+CUDA"
    elif use_metal:
        rust_backend = WaveBackend.RUST_METAL
        backend_name = "Rust+Metal"
    else:
        rust_backend = WaveBackend.RUST_CPU
        backend_name = "Rust+CPU"
    
    # Configure Rust-only waves (1 & 3) with all stages
    all_stages = ["data_prep", "pretrain", "sft", "grpo", "distillation", "export"]
    steps_per_wave = max_steps // 2
    cfg.time_sliced.waves = [
        WaveConfig(
            wave_id=1,
            backend=rust_backend,
            start_step=start_step,
            end_step=start_step + steps_per_wave,
            stages=all_stages,
            checkpoint_from=str(checkpoint_from) if checkpoint_from else None,
        ),
        WaveConfig(
            wave_id=3,
            backend=rust_backend,
            start_step=start_step + steps_per_wave,
            end_step=start_step + max_steps,
            stages=all_stages,
            checkpoint_from=f"checkpoints/step_{start_step + steps_per_wave}/",
        ),
    ]
    cfg.time_sliced.num_waves = 2
    cfg.training.max_steps = start_step + max_steps
    
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo(f"  {backend_name} Backend Execution (Waves 1 & 3)")
    typer.echo(f"  GPUs: {gpus} | PP: {pp_size} | Steps: {start_step}-{start_step + max_steps}")
    if dry_run:
        typer.echo("  Mode: DRY RUN (not executing Rust binary)")
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo(cfg.summary())
    
    workflow = DeepSeekWorkflow(cfg)
    context = workflow.run_time_sliced_waves(
        input_data=str(data_dir) if data_dir else None,
        output_dir=str(output_dir) if output_dir else None,
    )
    
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
    use_cuda: bool = typer.Option(True, "--cuda/--no-cuda", help="Use CUDA GPU (default: True)"),
):
    """
    Run Python/PyTorch backend waves only (Waves 2 & 4).
    
    Wave 2: Standard MOE/DeepSeek MOE
    Wave 4: MTP/FP8/Distillation/5D Parallelism
    
    Example:
        python -m deepseek.pipeline.cli run-python --gpus 3 --pp-size 3 --max-steps 10000 --checkpoint-from checkpoints/step_5000
    """
    from deepseek.pipeline.config import WaveBackend, WaveConfig, TimeSlicedConfig
    from deepseek.pipeline.workflow import DeepSeekWorkflow
    
    cfg = PipelineConfig.production_3gpu_time_sliced()
    cfg.time_sliced.gpu_ids = list(range(gpus))
    cfg.time_sliced.pipeline_parallel_size = pp_size
    cfg.distributed.num_workers = gpus
    cfg.distributed.pipeline_parallel_size = pp_size
    
    # Select PyTorch backend based on CUDA availability
    pytorch_backend = WaveBackend.PYTORCH_CUDA if use_cuda else WaveBackend.PYTORCH_CPU
    
    # Configure Python-only waves (2 & 4)
    all_stages = ["data_prep", "pretrain", "sft", "grpo", "distillation", "export"]
    steps_per_wave = max_steps // 2
    cfg.time_sliced.waves = [
        WaveConfig(
            wave_id=2,
            backend=pytorch_backend,
            start_step=start_step,
            end_step=start_step + steps_per_wave,
            stages=all_stages,
            checkpoint_from=str(checkpoint_from) if checkpoint_from else None,
        ),
        WaveConfig(
            wave_id=4,
            backend=pytorch_backend,
            start_step=start_step + steps_per_wave,
            end_step=start_step + max_steps,
            stages=all_stages,
            checkpoint_from=f"checkpoints/step_{start_step + steps_per_wave}/",
        ),
    ]
    cfg.time_sliced.num_waves = 2
    cfg.training.max_steps = start_step + max_steps
    
    backend_name = "PyTorch+CUDA" if use_cuda else "PyTorch+CPU"
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo(f"  {backend_name} Backend Execution (Waves 2 & 4)")
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
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Path to training data directory"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Path to output/checkpoint directory"),
    max_steps: int = typer.Option(100, "--max-steps", help="Maximum training steps"),
    start_step: int = typer.Option(0, "--start-step", help="Starting step"),
    checkpoint_from: Optional[Path] = typer.Option(None, "--checkpoint-from", help="Load checkpoint from path"),
    batch_size: int = typer.Option(2, "--batch-size", help="Batch size (lower = less memory)"),
    d_model: int = typer.Option(64, "--d-model", help="Model dimension (lower = less memory)"),
    num_layers: int = typer.Option(2, "--num-layers", help="Number of layers (lower = less memory)"),
    save_every: int = typer.Option(25, "--save-every", help="Save checkpoint every N steps"),
    verify_only: bool = typer.Option(False, "--verify-only", help="Quick verification run (10 steps, tiny model)"),
):
    """
    Run Python/MLX backend waves on Apple Silicon.
    
    All 4 waves use MLX backend for unified Apple Silicon training.
    Runs all pipeline stages: data_prep, pretrain, sft, grpo, distillation, export.
    
    Example:
        python -m deepseek.pipeline.cli run-mlx --verify-only  # Quick verification
        python -m deepseek.pipeline.cli run-mlx --max-steps 100 --data-dir ./data/fineweb-edu/train
        python -m deepseek.pipeline.cli run-mlx --max-steps 500 --batch-size 4 --d-model 128
    """
    from deepseek.pipeline.config import WaveBackend, WaveConfig, TimeSlicedConfig
    from deepseek.pipeline.workflow import DeepSeekWorkflow
    
    # Quick verification mode - tiny model, few steps
    if verify_only:
        max_steps = 10
        batch_size = 1
        d_model = 32
        num_layers = 1
        save_every = 5
        typer.echo("[VERIFY MODE] Using minimal settings for quick verification")
    
    cfg = PipelineConfig.production_3gpu_time_sliced()
    # MLX doesn't use GPU IDs the same way
    cfg.time_sliced.gpu_ids = [0]  # Apple Silicon unified memory
    cfg.time_sliced.pipeline_parallel_size = 1  # No PP for MLX
    cfg.distributed.num_workers = 1
    cfg.distributed.pipeline_parallel_size = 1
    
    # Set data directory if provided
    if data_dir:
        cfg.data.data_dir = str(data_dir)
        typer.echo(f"Using data directory: {data_dir}")
    
    # Set output directory if provided
    if output_dir:
        cfg.training.checkpoint_dir = str(output_dir)
        typer.echo(f"Using output directory: {output_dir}")
    
    # Memory-conscious settings for Apple Silicon
    cfg.training.batch_size = batch_size
    cfg.training.gradient_accumulation_steps = 8  # Compensate for small batch
    cfg.training.save_every_n_steps = save_every
    cfg.model.d_model = d_model
    cfg.model.num_layers = num_layers
    cfg.model.num_heads = max(2, d_model // 64)  # Scale heads with d_model
    
    # Configure MLX-only waves (all 4) with full pipeline stages
    all_stages = ["data_prep", "pretrain", "sft", "grpo", "distillation", "export"]
    steps_per_wave = max_steps // 4
    cfg.time_sliced.waves = [
        WaveConfig(
            wave_id=1,
            backend=WaveBackend.MLX,
            start_step=start_step,
            end_step=start_step + steps_per_wave,
            stages=all_stages,
            checkpoint_from=str(checkpoint_from) if checkpoint_from else None,
        ),
        WaveConfig(
            wave_id=2,
            backend=WaveBackend.MLX,
            start_step=start_step + steps_per_wave,
            end_step=start_step + steps_per_wave * 2,
            stages=all_stages,
            checkpoint_from=f"checkpoints/step_{start_step + steps_per_wave}/",
        ),
        WaveConfig(
            wave_id=3,
            backend=WaveBackend.MLX,
            start_step=start_step + steps_per_wave * 2,
            end_step=start_step + steps_per_wave * 3,
            stages=all_stages,
            checkpoint_from=f"checkpoints/step_{start_step + steps_per_wave * 2}/",
        ),
        WaveConfig(
            wave_id=4,
            backend=WaveBackend.MLX,
            start_step=start_step + steps_per_wave * 3,
            end_step=start_step + max_steps,
            stages=all_stages,
            checkpoint_from=f"checkpoints/step_{start_step + steps_per_wave * 3}/",
        ),
    ]
    cfg.time_sliced.num_waves = 4
    cfg.training.max_steps = start_step + max_steps
    
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo("  MLX Backend Execution (All Waves on Apple Silicon)")
    typer.echo(f"  Device: Apple Silicon GPU | Steps: {start_step}-{start_step + max_steps}")
    typer.echo(f"  Model: d_model={d_model}, layers={num_layers}, batch={batch_size}")
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo(cfg.summary())
    
    workflow = DeepSeekWorkflow(cfg)
    context = workflow.run_time_sliced_waves(
        input_data=str(data_dir) if data_dir else None,
        output_dir=str(output_dir) if output_dir else None,
    )
    
    typer.echo("MLX waves completed. Metadata:")
    typer.echo(context.metadata)


@app.command()
def run_cpu(
    max_steps: int = typer.Option(1000, "--max-steps", help="Maximum training steps"),
    batch_size: int = typer.Option(1, "--batch-size", help="Batch size (keep low for CPU)"),
    d_model: int = typer.Option(64, "--d-model", help="Model dimension (keep small for CPU)"),
    num_layers: int = typer.Option(2, "--num-layers", help="Number of layers (keep small for CPU)"),
    start_step: int = typer.Option(0, "--start-step", help="Starting step number"),
    checkpoint_from: Optional[Path] = typer.Option(None, "--checkpoint", help="Resume from checkpoint"),
):
    """
    Run CPU-only backend waves (for testing without GPU/Metal).
    
    All 4 waves use CPU backend for portable execution.
    Uses PyTorch CPU backend with small model for testing/debugging.
    
    Example:
        python -m deepseek.pipeline.cli run-cpu --max-steps 500
        python -m deepseek.pipeline.cli run-cpu --max-steps 1000 --batch-size 2 --d-model 64
    """
    from deepseek.pipeline.config import WaveBackend, WaveConfig, TimeSlicedConfig
    from deepseek.pipeline.workflow import DeepSeekWorkflow
    
    cfg = PipelineConfig.production_3gpu_time_sliced()
    cfg.time_sliced.gpu_ids = []  # No GPUs
    cfg.time_sliced.pipeline_parallel_size = 1
    cfg.distributed.num_workers = 1
    cfg.distributed.pipeline_parallel_size = 1
    
    # CPU-friendly settings (small model)
    cfg.training.batch_size = batch_size
    cfg.training.gradient_accumulation_steps = 4
    cfg.model.d_model = d_model
    cfg.model.num_layers = num_layers
    cfg.model.num_heads = max(1, d_model // 64)
    
    # Configure CPU-only waves (all 4) with full pipeline stages
    all_stages = ["data_prep", "pretrain", "sft", "grpo", "distillation", "export"]
    steps_per_wave = max_steps // 4
    cfg.time_sliced.waves = [
        WaveConfig(
            wave_id=1,
            backend=WaveBackend.CPU_ONLY,
            start_step=start_step,
            end_step=start_step + steps_per_wave,
            stages=all_stages,
            checkpoint_from=str(checkpoint_from) if checkpoint_from else None,
        ),
        WaveConfig(
            wave_id=2,
            backend=WaveBackend.CPU_ONLY,
            start_step=start_step + steps_per_wave,
            end_step=start_step + steps_per_wave * 2,
            stages=all_stages,
            checkpoint_from=f"checkpoints/step_{start_step + steps_per_wave}/",
        ),
        WaveConfig(
            wave_id=3,
            backend=WaveBackend.CPU_ONLY,
            start_step=start_step + steps_per_wave * 2,
            end_step=start_step + steps_per_wave * 3,
            stages=all_stages,
            checkpoint_from=f"checkpoints/step_{start_step + steps_per_wave * 2}/",
        ),
        WaveConfig(
            wave_id=4,
            backend=WaveBackend.CPU_ONLY,
            start_step=start_step + steps_per_wave * 3,
            end_step=start_step + max_steps,
            stages=all_stages,
            checkpoint_from=f"checkpoints/step_{start_step + steps_per_wave * 3}/",
        ),
    ]
    cfg.time_sliced.num_waves = 4
    cfg.training.max_steps = start_step + max_steps
    
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo("  CPU-Only Backend Execution (No GPU/Metal Required)")
    typer.echo(f"  Device: CPU | Steps: {start_step}-{start_step + max_steps}")
    typer.echo(f"  Model: d_model={d_model}, layers={num_layers}, batch={batch_size}")
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo(cfg.summary())
    
    workflow = DeepSeekWorkflow(cfg)
    context = workflow.run_time_sliced_waves()
    
    typer.echo("CPU-only waves completed. Metadata:")
    typer.echo(context.metadata)


@app.command()
def benchmark(
    max_steps: int = typer.Option(3000, "--max-steps", help="Maximum training steps per run"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file for results"),
    mlx_batch_size: int = typer.Option(2, "--mlx-batch-size", help="Batch size for MLX (memory control)"),
):
    """
    Run all three backends and benchmark their performance.
    
    Runs Rust+Metal, PyTorch+CUDA, and MLX sequentially and compares:
    - Total training time
    - Steps per second
    - Final validation loss
    
    Example:
        python -m deepseek.pipeline.cli benchmark --max-steps 3000 --output benchmark_results.json
    """
    import time
    import json
    from deepseek.pipeline.config import WaveBackend, WaveConfig
    from deepseek.pipeline.workflow import DeepSeekWorkflow
    
    results = {}
    
    typer.echo("═══════════════════════════════════════════════════════════")
    typer.echo("  BENCHMARK: Rust+Metal vs PyTorch+CUDA vs MLX")
    typer.echo(f"  Max Steps: {max_steps}")
    typer.echo("═══════════════════════════════════════════════════════════\n")
    
    # Helper to run a benchmark
    def run_bench(name: str, backend: WaveBackend, use_gpu: bool = True):
        typer.echo(f"\n▶ Starting {name} benchmark...")
        start_time = time.time()
        
        cfg = PipelineConfig.production_3gpu_time_sliced()
        all_stages = ["data_prep", "pretrain", "sft", "grpo", "distillation", "export"]
        steps_per_wave = max_steps // 4
        
        if backend == WaveBackend.MLX:
            cfg.time_sliced.gpu_ids = [0]
            cfg.time_sliced.pipeline_parallel_size = 1
            cfg.distributed.num_workers = 1
            # Memory-conscious settings for Apple Silicon
            cfg.training.batch_size = mlx_batch_size
            cfg.training.gradient_accumulation_steps = 8
            cfg.model.d_model = 128
            cfg.model.num_layers = 4
            cfg.model.num_heads = 2
        else:
            cfg.time_sliced.gpu_ids = [0, 1, 2]
            cfg.time_sliced.pipeline_parallel_size = 3
            cfg.distributed.num_workers = 3
        
        cfg.time_sliced.waves = [
            WaveConfig(wave_id=i+1, backend=backend, start_step=i*steps_per_wave, 
                      end_step=(i+1)*steps_per_wave, stages=all_stages)
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
    results["rust_metal"] = run_bench("Rust+Metal", WaveBackend.RUST_METAL)
    results["pytorch_cuda"] = run_bench("PyTorch+CUDA", WaveBackend.PYTORCH_CUDA)
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
