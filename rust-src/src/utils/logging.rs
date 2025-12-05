//! Logging utilities with structured JSON output support.
//!
//! Supports both human-readable and JSON log formats for log aggregation.

use tracing_subscriber::{fmt, EnvFilter};

/// Logging format options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogFormat {
    /// Human-readable format (default)
    Pretty,
    /// JSON format for log aggregation (matches Python's structlog)
    Json,
    /// Compact single-line format
    Compact,
}

/// Initialize logging with the default pretty format.
pub fn init_logging() {
    init_logging_with_format(LogFormat::Pretty);
}

/// Initialize logging with the specified format.
/// 
/// Set `RUST_LOG` environment variable to control log level.
/// Set `DEEPSEEK_LOG_FORMAT=json` for JSON output.
pub fn init_logging_with_format(format: LogFormat) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    match format {
        LogFormat::Pretty => {
            fmt()
                .with_env_filter(filter)
                .with_target(true)
                .with_thread_ids(false)
                .with_file(false)
                .with_line_number(false)
                .init();
        }
        LogFormat::Json => {
            fmt()
                .with_env_filter(filter)
                .json()
                .with_current_span(true)
                .with_span_list(true)
                .flatten_event(true)
                .init();
        }
        LogFormat::Compact => {
            fmt()
                .with_env_filter(filter)
                .compact()
                .with_target(false)
                .init();
        }
    }
}

/// Initialize logging based on environment variables.
/// 
/// Checks `DEEPSEEK_LOG_FORMAT` for format selection:
/// - "json" -> JSON format
/// - "compact" -> Compact format  
/// - anything else -> Pretty format (default)
pub fn init_logging_from_env() {
    let format = std::env::var("DEEPSEEK_LOG_FORMAT")
        .map(|s| match s.to_lowercase().as_str() {
            "json" => LogFormat::Json,
            "compact" => LogFormat::Compact,
            _ => LogFormat::Pretty,
        })
        .unwrap_or(LogFormat::Pretty);
    
    init_logging_with_format(format);
}

