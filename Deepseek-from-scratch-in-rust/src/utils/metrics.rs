//! Prometheus metrics for training observability.
//!
//! Provides training loss, throughput, and memory metrics.

use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec,
    Opts, Registry, TextEncoder, Encoder,
};
use std::sync::OnceLock;
use tracing::info;

/// Global metrics registry
static REGISTRY: OnceLock<MetricsRegistry> = OnceLock::new();

/// Collection of all DeepSeek training metrics
pub struct MetricsRegistry {
    pub registry: Registry,
    
    // Training metrics
    pub training_loss: Histogram,
    pub training_step_duration: Histogram,
    pub tokens_processed: Counter,
    pub steps_completed: Counter,
    
    // Throughput metrics
    pub tokens_per_second: Gauge,
    pub samples_per_second: Gauge,
    
    // Memory metrics
    pub gpu_memory_used: Gauge,
    pub gpu_memory_allocated: Gauge,
    
    // Gradient metrics
    pub gradient_norm: Histogram,
    pub learning_rate: Gauge,
    
    // Distributed metrics
    pub communication_time: Histogram,
    pub sync_time: Histogram,
    
    // Per-layer metrics (optional, can be expensive)
    pub layer_forward_time: HistogramVec,
    pub layer_backward_time: HistogramVec,
}

impl MetricsRegistry {
    /// Create a new metrics registry with all metrics registered.
    pub fn new() -> Self {
        let registry = Registry::new();
        
        // Training loss histogram with buckets for typical loss ranges
        let training_loss = Histogram::with_opts(
            HistogramOpts::new("deepseek_training_loss", "Training loss value")
                .buckets(vec![0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0])
        ).unwrap();
        registry.register(Box::new(training_loss.clone())).unwrap();
        
        // Step duration in seconds
        let training_step_duration = Histogram::with_opts(
            HistogramOpts::new("deepseek_step_duration_seconds", "Time per training step")
                .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
        ).unwrap();
        registry.register(Box::new(training_step_duration.clone())).unwrap();
        
        // Token counters
        let tokens_processed = Counter::with_opts(
            Opts::new("deepseek_tokens_processed_total", "Total tokens processed")
        ).unwrap();
        registry.register(Box::new(tokens_processed.clone())).unwrap();
        
        let steps_completed = Counter::with_opts(
            Opts::new("deepseek_steps_completed_total", "Total training steps completed")
        ).unwrap();
        registry.register(Box::new(steps_completed.clone())).unwrap();
        
        // Throughput gauges
        let tokens_per_second = Gauge::with_opts(
            Opts::new("deepseek_tokens_per_second", "Current tokens/second throughput")
        ).unwrap();
        registry.register(Box::new(tokens_per_second.clone())).unwrap();
        
        let samples_per_second = Gauge::with_opts(
            Opts::new("deepseek_samples_per_second", "Current samples/second throughput")
        ).unwrap();
        registry.register(Box::new(samples_per_second.clone())).unwrap();
        
        // Memory gauges (in bytes)
        let gpu_memory_used = Gauge::with_opts(
            Opts::new("deepseek_gpu_memory_used_bytes", "GPU memory currently in use")
        ).unwrap();
        registry.register(Box::new(gpu_memory_used.clone())).unwrap();
        
        let gpu_memory_allocated = Gauge::with_opts(
            Opts::new("deepseek_gpu_memory_allocated_bytes", "GPU memory allocated")
        ).unwrap();
        registry.register(Box::new(gpu_memory_allocated.clone())).unwrap();
        
        // Gradient metrics
        let gradient_norm = Histogram::with_opts(
            HistogramOpts::new("deepseek_gradient_norm", "Gradient L2 norm")
                .buckets(vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        ).unwrap();
        registry.register(Box::new(gradient_norm.clone())).unwrap();
        
        let learning_rate = Gauge::with_opts(
            Opts::new("deepseek_learning_rate", "Current learning rate")
        ).unwrap();
        registry.register(Box::new(learning_rate.clone())).unwrap();
        
        // Communication metrics
        let communication_time = Histogram::with_opts(
            HistogramOpts::new("deepseek_communication_seconds", "Time spent in collective communications")
                .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
        ).unwrap();
        registry.register(Box::new(communication_time.clone())).unwrap();
        
        let sync_time = Histogram::with_opts(
            HistogramOpts::new("deepseek_sync_seconds", "Time spent in synchronization barriers")
                .buckets(vec![0.0001, 0.001, 0.01, 0.1, 1.0])
        ).unwrap();
        registry.register(Box::new(sync_time.clone())).unwrap();
        
        // Per-layer metrics
        let layer_forward_time = HistogramVec::new(
            HistogramOpts::new("deepseek_layer_forward_seconds", "Forward pass time per layer")
                .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1]),
            &["layer"]
        ).unwrap();
        registry.register(Box::new(layer_forward_time.clone())).unwrap();
        
        let layer_backward_time = HistogramVec::new(
            HistogramOpts::new("deepseek_layer_backward_seconds", "Backward pass time per layer")
                .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1]),
            &["layer"]
        ).unwrap();
        registry.register(Box::new(layer_backward_time.clone())).unwrap();
        
        Self {
            registry,
            training_loss,
            training_step_duration,
            tokens_processed,
            steps_completed,
            tokens_per_second,
            samples_per_second,
            gpu_memory_used,
            gpu_memory_allocated,
            gradient_norm,
            learning_rate,
            communication_time,
            sync_time,
            layer_forward_time,
            layer_backward_time,
        }
    }
    
    /// Gather all metrics as Prometheus text format.
    pub fn gather(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}

/// Get the global metrics registry.
pub fn get_metrics() -> &'static MetricsRegistry {
    REGISTRY.get_or_init(MetricsRegistry::new)
}

/// Record a training step with all associated metrics.
pub fn record_training_step(
    loss: f64,
    duration_secs: f64,
    tokens: u64,
    lr: f64,
    grad_norm: Option<f64>,
) {
    let m = get_metrics();
    m.training_loss.observe(loss);
    m.training_step_duration.observe(duration_secs);
    m.tokens_processed.inc_by(tokens as f64);
    m.steps_completed.inc();
    m.learning_rate.set(lr);
    
    if duration_secs > 0.0 {
        m.tokens_per_second.set(tokens as f64 / duration_secs);
    }
    
    if let Some(gn) = grad_norm {
        m.gradient_norm.observe(gn);
    }
}

/// Record GPU memory usage.
pub fn record_memory_usage(used_bytes: u64, allocated_bytes: u64) {
    let m = get_metrics();
    m.gpu_memory_used.set(used_bytes as f64);
    m.gpu_memory_allocated.set(allocated_bytes as f64);
}

/// Record communication time.
pub fn record_communication_time(duration_secs: f64) {
    get_metrics().communication_time.observe(duration_secs);
}

/// Record synchronization time.
pub fn record_sync_time(duration_secs: f64) {
    get_metrics().sync_time.observe(duration_secs);
}

/// Start a simple HTTP server to expose metrics on the given port.
/// 
/// This is a basic implementation. For production, consider using
/// a proper web framework with graceful shutdown.
#[cfg(feature = "metrics-server")]
pub async fn start_metrics_server(port: u16) -> std::io::Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    info!(port = port, "Metrics server started");
    
    loop {
        let (mut socket, _) = listener.accept().await?;
        
        tokio::spawn(async move {
            let mut buf = [0; 1024];
            let _ = socket.read(&mut buf).await;
            
            let metrics = get_metrics().gather();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
                metrics.len(),
                metrics
            );
            
            let _ = socket.write_all(response.as_bytes()).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_creation() {
        let metrics = MetricsRegistry::new();
        metrics.training_loss.observe(2.5);
        metrics.tokens_processed.inc_by(1000.0);
        
        let output = metrics.gather();
        assert!(output.contains("deepseek_training_loss"));
        assert!(output.contains("deepseek_tokens_processed_total"));
    }
    
    #[test]
    fn test_record_training_step() {
        record_training_step(2.0, 0.5, 1024, 1e-4, Some(1.5));
        
        let output = get_metrics().gather();
        assert!(output.contains("deepseek_training_loss"));
    }
}
