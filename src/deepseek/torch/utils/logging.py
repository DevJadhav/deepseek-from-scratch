import logging
import sys
import structlog
from prometheus_client import Counter, Histogram, start_http_server

# Configure Prometheus metrics
TRAINING_LOSS = Histogram(
    "deepseek_training_loss",
    "Training loss",
    ["model_name", "phase"]
)

TOKENS_PROCESSED = Counter(
    "deepseek_tokens_processed_total",
    "Total number of tokens processed",
    ["model_name"]
)

def configure_logging(log_level: str = "INFO", json_format: bool = True):
    """
    Configure structured logging.
    
    Args:
        log_level: Logging level (INFO, DEBUG, etc.)
        json_format: Whether to output logs in JSON format
    """
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
        
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str):
    """Get a structured logger."""
    return structlog.get_logger(name)

def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics server."""
    try:
        start_http_server(port)
        get_logger("metrics").info("Prometheus metrics server started", port=port)
    except Exception as e:
        get_logger("metrics").error("Failed to start metrics server", error=str(e))
