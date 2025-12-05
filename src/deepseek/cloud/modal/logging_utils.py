"""
Structured Logging Utilities for Modal GPU Infrastructure
=========================================================

Provides centralized logging configuration with:
- Structlog integration with JSON output for production
- Console output with rich formatting for development
- Correlation IDs for distributed tracing
- Context binding for worker identification

Usage:
    from deepseek.cloud.modal.logging_utils import get_logger, configure_logging
    
    # Configure once at module entry
    configure_logging(env="production")
    
    # Get logger with context
    logger = get_logger(__name__)
    logger = logger.bind(rank=0, worker_id="gpu-worker-0")
    
    # Use structured logging
    logger.info("training_started", step=0, batch_size=32)
    logger.error("checkpoint_failed", path="/checkpoints", exc_info=True)
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import structlog

# Context variable for correlation ID (thread-safe)
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracing."""
    return f"trace-{uuid.uuid4().hex[:12]}"


def get_correlation_id() -> str:
    """Get the current correlation ID or generate a new one."""
    cid = _correlation_id.get()
    if not cid:
        cid = generate_correlation_id()
        _correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    _correlation_id.set(cid)


def add_correlation_id(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor to add correlation ID to all log entries."""
    event_dict["correlation_id"] = get_correlation_id()
    return event_dict


def add_timestamp(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add ISO timestamp to log entries."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_service_info(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add service metadata to log entries."""
    event_dict["service"] = "deepseek-training"
    event_dict["version"] = os.getenv("APP_VERSION", "0.1.0")
    return event_dict


@lru_cache(maxsize=1)
def configure_logging(
    env: str = "development",
    log_level: str | None = None,
    json_logs: bool | None = None,
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        env: Environment name ("development", "production", "testing")
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR)
        json_logs: Force JSON output (default: True for production)
    
    Call this once at the start of your application.
    """
    # Determine settings based on environment
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO" if env == "production" else "DEBUG")
    
    if json_logs is None:
        json_logs = env == "production" or os.getenv("JSON_LOGS", "").lower() == "true"
    
    # Shared processors
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_timestamp,
        add_correlation_id,
        add_service_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_logs:
        # JSON output for production (machine-parseable)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Console output for development (human-readable)
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("modal").setLevel(logging.INFO)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog BoundLogger
        
    Example:
        logger = get_logger(__name__)
        logger = logger.bind(rank=0)
        logger.info("training_started", step=0)
    """
    return structlog.get_logger(name)


class TrainingLogger:
    """
    Specialized logger for training loops with metric tracking.
    
    Provides convenience methods for common training events and
    automatic metric aggregation.
    
    Example:
        logger = TrainingLogger("pretrain", rank=0, world_size=4)
        logger.log_step(step=100, loss=2.5, lr=1e-4)
        logger.log_checkpoint_saved("/checkpoints/step_100")
        logger.log_epoch_complete(epoch=1, avg_loss=2.3)
    """
    
    def __init__(
        self,
        stage: str,
        rank: int = 0,
        world_size: int = 1,
        logger_name: str | None = None,
    ):
        self.stage = stage
        self.rank = rank
        self.world_size = world_size
        self._logger = get_logger(logger_name or f"training.{stage}")
        self._logger = self._logger.bind(
            stage=stage,
            rank=rank,
            world_size=world_size,
        )
        self._step_metrics: list[dict[str, Any]] = []
    
    def log_step(
        self,
        step: int,
        loss: float,
        lr: float | None = None,
        throughput: float | None = None,
        memory_gb: float | None = None,
        **extra: Any,
    ) -> None:
        """Log a training step."""
        metrics = {
            "step": step,
            "loss": loss,
        }
        if lr is not None:
            metrics["learning_rate"] = lr
        if throughput is not None:
            metrics["throughput_samples_per_sec"] = throughput
        if memory_gb is not None:
            metrics["gpu_memory_gb"] = memory_gb
        metrics.update(extra)
        
        self._step_metrics.append(metrics)
        self._logger.info("training_step", **metrics)
    
    def log_checkpoint_saved(
        self,
        path: str,
        step: int | None = None,
        **extra: Any,
    ) -> None:
        """Log checkpoint save event."""
        self._logger.info(
            "checkpoint_saved",
            checkpoint_path=path,
            step=step,
            **extra,
        )
    
    def log_checkpoint_loaded(
        self,
        path: str,
        step: int | None = None,
        **extra: Any,
    ) -> None:
        """Log checkpoint load event."""
        self._logger.info(
            "checkpoint_loaded",
            checkpoint_path=path,
            step=step,
            **extra,
        )
    
    def log_epoch_complete(
        self,
        epoch: int,
        avg_loss: float | None = None,
        **extra: Any,
    ) -> None:
        """Log epoch completion."""
        self._logger.info(
            "epoch_complete",
            epoch=epoch,
            avg_loss=avg_loss,
            **extra,
        )
    
    def log_training_started(
        self,
        model_params: int,
        max_steps: int,
        batch_size: int,
        **extra: Any,
    ) -> None:
        """Log training initialization."""
        self._logger.info(
            "training_started",
            model_params=model_params,
            max_steps=max_steps,
            batch_size=batch_size,
            **extra,
        )
    
    def log_training_complete(
        self,
        final_step: int,
        total_time_seconds: float,
        final_loss: float | None = None,
        **extra: Any,
    ) -> None:
        """Log training completion."""
        self._logger.info(
            "training_complete",
            final_step=final_step,
            total_time_seconds=total_time_seconds,
            final_loss=final_loss,
            **extra,
        )
    
    def log_error(
        self,
        event: str,
        error: Exception | str,
        **extra: Any,
    ) -> None:
        """Log an error event."""
        self._logger.error(
            event,
            error=str(error),
            error_type=type(error).__name__ if isinstance(error, Exception) else "str",
            exc_info=isinstance(error, Exception),
            **extra,
        )
    
    def log_warning(
        self,
        event: str,
        message: str,
        **extra: Any,
    ) -> None:
        """Log a warning event."""
        self._logger.warning(event, message=message, **extra)
    
    def get_aggregated_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics from all logged steps."""
        if not self._step_metrics:
            return {}
        
        losses = [m["loss"] for m in self._step_metrics]
        return {
            "total_steps": len(self._step_metrics),
            "avg_loss": sum(losses) / len(losses),
            "min_loss": min(losses),
            "max_loss": max(losses),
            "final_loss": losses[-1] if losses else None,
        }


# Initialize logging on import if not already configured
def _auto_configure():
    """Auto-configure logging based on environment."""
    if not structlog.is_configured():
        env = os.getenv("ENVIRONMENT", "development")
        configure_logging(env=env)


_auto_configure()
