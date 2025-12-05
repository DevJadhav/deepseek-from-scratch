#!/usr/bin/env python3
"""
Training Dashboard for DeepSeek Training Pipeline.

Rich terminal UI for displaying live training progress,
GPU costs, and pipeline status.

Features:
- Live training metrics display
- Cost tracking visualization
- Progress bars for training stages
- Budget alerts and warnings
- ETA calculations

Usage:
    from monitoring.dashboard import TrainingDashboard

    dashboard = TrainingDashboard(cost_tracker)
    dashboard.start()
    # ... update during training ...
    dashboard.update(step=100, loss=2.5)
    dashboard.stop()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from monitoring.cost_tracker import CostTracker


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    step: int = 0
    total_steps: int = 10000
    loss: float = 0.0
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    batch_size: int = 8
    gradient_norm: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 80.0  # H100 has 80GB
    stage: str = "pretrain"
    epoch: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_steps <= 0:
            return 0.0
        return (self.step / self.total_steps) * 100

    @property
    def gpu_memory_percent(self) -> float:
        """Calculate GPU memory usage percentage."""
        if self.gpu_memory_total <= 0:
            return 0.0
        return (self.gpu_memory_used / self.gpu_memory_total) * 100


class TrainingDashboard:
    """
    Rich terminal dashboard for training monitoring.

    Displays:
    - Training progress (step, loss, learning rate)
    - Cost tracking (GPU hours, cost, budget)
    - GPU metrics (memory, throughput)
    - Stage progress and ETA

    Example:
        tracker = CostTracker(budget_limit=1000.0)
        dashboard = TrainingDashboard(tracker)

        with dashboard:
            for step in range(1000):
                # Train...
                dashboard.update(step=step, loss=loss)
    """

    def __init__(
        self,
        cost_tracker: CostTracker | None = None,
        refresh_rate: int = 4,
        title: str = "DeepSeek Training Pipeline",
    ):
        """
        Initialize dashboard.

        Args:
            cost_tracker: Optional CostTracker for cost display
            refresh_rate: Refresh rate in Hz
            title: Dashboard title
        """
        self.cost_tracker = cost_tracker
        self.refresh_rate = refresh_rate
        self.title = title
        self.console = Console()
        self.metrics = TrainingMetrics()
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._start_time: datetime | None = None
        self._loss_history: list[float] = []

    def start(self, total_steps: int = 10000, stage: str = "pretrain") -> None:
        """
        Start the dashboard.

        Args:
            total_steps: Total training steps
            stage: Current training stage
        """
        self.metrics.total_steps = total_steps
        self.metrics.stage = stage
        self._start_time = datetime.now()
        self._loss_history = []

        # Create progress bar
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=self.refresh_rate,
        )
        self._task_id = self._progress.add_task(f"[{stage}]", total=total_steps, completed=0)

        # Start live display
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=self.refresh_rate,
            screen=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the dashboard."""
        if self._live:
            self._live.stop()
            self._live = None
        self._progress = None
        self._task_id = None

    def update(
        self,
        step: int | None = None,
        loss: float | None = None,
        learning_rate: float | None = None,
        tokens_per_second: float | None = None,
        batch_size: int | None = None,
        gradient_norm: float | None = None,
        gpu_memory_used: float | None = None,
        stage: str | None = None,
        epoch: int | None = None,
        **kwargs,
    ) -> None:
        """
        Update dashboard with new metrics.

        Args:
            step: Current training step
            loss: Current loss value
            learning_rate: Current learning rate
            tokens_per_second: Training throughput
            batch_size: Batch size
            gradient_norm: Gradient norm
            gpu_memory_used: GPU memory used in GB
            stage: Training stage
            epoch: Current epoch
            **kwargs: Additional metrics (ignored)
        """
        if step is not None:
            self.metrics.step = step
        if loss is not None:
            self.metrics.loss = loss
            self._loss_history.append(loss)
            # Keep last 100 values for smoothing
            if len(self._loss_history) > 100:
                self._loss_history = self._loss_history[-100:]
        if learning_rate is not None:
            self.metrics.learning_rate = learning_rate
        if tokens_per_second is not None:
            self.metrics.tokens_per_second = tokens_per_second
        if batch_size is not None:
            self.metrics.batch_size = batch_size
        if gradient_norm is not None:
            self.metrics.gradient_norm = gradient_norm
        if gpu_memory_used is not None:
            self.metrics.gpu_memory_used = gpu_memory_used
        if stage is not None:
            self.metrics.stage = stage
        if epoch is not None:
            self.metrics.epoch = epoch

        self.metrics.timestamp = datetime.now()

        # Update progress bar
        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=self.metrics.step,
                description=f"[{self.metrics.stage}]",
            )

        # Update live display
        if self._live:
            self._live.update(self._render())

    def _render(self) -> Panel:
        """Render the dashboard."""
        layout = Layout()

        # Create main sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        # Header
        header_text = Text(self.title, style="bold cyan", justify="center")
        layout["header"].update(Panel(header_text, style="bold"))

        # Body - split into left and right
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        # Left side - Training Metrics
        layout["left"].update(self._render_training_panel())

        # Right side - Cost Panel
        layout["right"].update(self._render_cost_panel())

        # Footer - Progress bar
        if self._progress:
            layout["footer"].update(Panel(self._progress))
        else:
            layout["footer"].update(Panel(Text("Not started", justify="center", style="dim")))

        return Panel(layout, title="[bold]Dashboard[/bold]", border_style="blue")

    def _render_training_panel(self) -> Panel:
        """Render training metrics panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Add metrics
        table.add_row("Stage", self.metrics.stage)
        table.add_row("Step", f"{self.metrics.step:,} / {self.metrics.total_steps:,}")
        table.add_row("Progress", f"{self.metrics.progress_percent:.1f}%")
        table.add_row("Loss", f"{self.metrics.loss:.4f}")

        # Smoothed loss
        if self._loss_history:
            smoothed = sum(self._loss_history[-10:]) / min(10, len(self._loss_history))
            table.add_row("Loss (smooth)", f"{smoothed:.4f}")

        table.add_row("Learning Rate", f"{self.metrics.learning_rate:.2e}")
        table.add_row("Tokens/sec", f"{self.metrics.tokens_per_second:,.0f}")
        table.add_row("Batch Size", str(self.metrics.batch_size))
        table.add_row("Grad Norm", f"{self.metrics.gradient_norm:.4f}")

        # GPU Memory
        mem_style = "green"
        if self.metrics.gpu_memory_percent > 90:
            mem_style = "red"
        elif self.metrics.gpu_memory_percent > 75:
            mem_style = "yellow"

        table.add_row(
            "GPU Memory",
            f"[{mem_style}]{self.metrics.gpu_memory_used:.1f}/{self.metrics.gpu_memory_total:.1f} GB "
            f"({self.metrics.gpu_memory_percent:.1f}%)[/{mem_style}]",
        )

        # ETA
        if self._start_time and self.metrics.step > 0:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            steps_remaining = self.metrics.total_steps - self.metrics.step
            if self.metrics.step > 0:
                time_per_step = elapsed / self.metrics.step
                eta_seconds = steps_remaining * time_per_step
                eta = timedelta(seconds=int(eta_seconds))
                table.add_row("ETA", str(eta))

        return Panel(table, title="[bold]Training Metrics[/bold]", border_style="green")

    def _render_cost_panel(self) -> Panel:
        """Render cost tracking panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        if self.cost_tracker:
            tracker = self.cost_tracker

            # Cost metrics
            table.add_row("GPU Type", tracker.gpu_type.name)
            table.add_row("Hourly Rate", f"${tracker.gpu_type.hourly_rate:.2f}/hr")
            table.add_row("GPU Hours", f"{tracker.total_gpu_hours:.2f}")
            table.add_row("Total Cost", f"${tracker.total_cost:.2f}")
            table.add_row("Budget", f"${tracker.budget_limit:.2f}")
            table.add_row("Remaining", f"${tracker.remaining_budget:.2f}")

            # Budget percentage with color
            percent = tracker.budget_percent_used
            if percent >= 95:
                style = "bold red"
            elif percent >= 75:
                style = "yellow"
            elif percent >= 50:
                style = "cyan"
            else:
                style = "green"

            table.add_row("Budget Used", f"[{style}]{percent:.1f}%[/{style}]")
            table.add_row("Hours Remaining", f"{tracker.estimated_hours_remaining:.1f} hrs")

            # Stage breakdown
            stage_summary = tracker.get_stage_summary()
            if stage_summary:
                table.add_row("", "")  # Spacer
                table.add_row("[bold]By Stage[/bold]", "")
                for stage, data in stage_summary.items():
                    table.add_row(f"  {stage}", f"${data['cost']:.2f} ({data['gpu_hours']:.1f}h)")

            # Alerts
            if tracker.alerts:
                latest_alert = tracker.alerts[-1]
                alert_style = {
                    "info": "blue",
                    "warning": "yellow",
                    "critical": "red",
                    "exceeded": "bold red",
                }.get(latest_alert.level.value, "white")
                table.add_row("", "")  # Spacer
                table.add_row(
                    "[bold]Alert[/bold]",
                    f"[{alert_style}]{latest_alert.level.value.upper()}[/{alert_style}]",
                )
        else:
            table.add_row("Status", "No cost tracker configured")

        return Panel(table, title="[bold]Cost Tracking[/bold]", border_style="yellow")

    def __enter__(self) -> TrainingDashboard:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

    def print_summary(self) -> None:
        """Print final training summary."""
        self.console.print("\n")
        self.console.rule("[bold blue]Training Summary[/bold blue]")

        # Training metrics
        table = Table(title="Final Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Steps", f"{self.metrics.step:,}")
        table.add_row("Final Loss", f"{self.metrics.loss:.4f}")
        if self._loss_history:
            table.add_row(
                "Avg Loss (last 100)", f"{sum(self._loss_history) / len(self._loss_history):.4f}"
            )

        if self._start_time:
            elapsed = datetime.now() - self._start_time
            table.add_row("Total Time", str(elapsed).split(".")[0])

        self.console.print(table)

        # Cost summary
        if self.cost_tracker:
            self.console.print("\n")
            cost_table = Table(title="Cost Summary")
            cost_table.add_column("Metric", style="cyan")
            cost_table.add_column("Value", style="green")

            cost_table.add_row("Total GPU Hours", f"{self.cost_tracker.total_gpu_hours:.2f}")
            cost_table.add_row("Total Cost", f"${self.cost_tracker.total_cost:.2f}")
            cost_table.add_row("Budget Used", f"{self.cost_tracker.budget_percent_used:.1f}%")

            self.console.print(cost_table)


def create_dashboard(
    cost_tracker: CostTracker | None = None,
    title: str = "DeepSeek Training Pipeline",
) -> TrainingDashboard:
    """
    Factory function to create a TrainingDashboard.

    Args:
        cost_tracker: Optional CostTracker instance
        title: Dashboard title

    Returns:
        Configured TrainingDashboard instance
    """
    return TrainingDashboard(cost_tracker=cost_tracker, title=title)
