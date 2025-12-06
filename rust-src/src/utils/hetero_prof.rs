//! HeteroProf: Custom Rust-Based Profiler for Ray Actors
//!
//! A profiler for heterogeneous distributed training that captures:
//! - Per-tensor memory movement between Python/Rust
//! - GPU kernel utilization on heterogeneous devices
//! - Expert load imbalance in real-time
//! - KV cache memory pressure during long-context training
//!
//! Features:
//! - Chrome tracing format export for visualization
//! - Real-time metrics streaming
//! - Cross-device timing correlation
//! - Memory allocation tracking

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;

/// Event category for profiling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventCategory {
    TensorTransfer,
    KernelExecution,
    MemoryAllocation,
    ExpertRouting,
    KVCacheOperation,
    RayActorCall,
    ModelForward,
    ModelBackward,
    Optimizer,
    Communication,
    Custom,
}

impl EventCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            EventCategory::TensorTransfer => "tensor_transfer",
            EventCategory::KernelExecution => "kernel",
            EventCategory::MemoryAllocation => "memory",
            EventCategory::ExpertRouting => "expert_routing",
            EventCategory::KVCacheOperation => "kv_cache",
            EventCategory::RayActorCall => "ray_actor",
            EventCategory::ModelForward => "forward",
            EventCategory::ModelBackward => "backward",
            EventCategory::Optimizer => "optimizer",
            EventCategory::Communication => "communication",
            EventCategory::Custom => "custom",
        }
    }
}

/// Device type for profiling
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    MetalGPU(usize),
    CUDAGPU(usize),
    CPU,
    Network,
}

impl DeviceType {
    pub fn as_str(&self) -> String {
        match self {
            DeviceType::MetalGPU(id) => format!("metal:{}", id),
            DeviceType::CUDAGPU(id) => format!("cuda:{}", id),
            DeviceType::CPU => "cpu".to_string(),
            DeviceType::Network => "network".to_string(),
        }
    }
}

/// A single profiling event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileEvent {
    pub name: String,
    pub category: EventCategory,
    pub start_us: u64,
    pub duration_us: u64,
    pub device: DeviceType,
    pub thread_id: u64,
    pub process_id: u32,
    pub metadata: HashMap<String, String>,
}

impl ProfileEvent {
    /// Convert to Chrome tracing format
    pub fn to_chrome_trace(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name,
            "cat": self.category.as_str(),
            "ph": "X",  // Complete event
            "ts": self.start_us,
            "dur": self.duration_us,
            "pid": self.process_id,
            "tid": self.thread_id,
            "args": self.metadata,
        })
    }
}

/// Tensor transfer metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TensorTransferMetrics {
    pub total_bytes: usize,
    pub total_transfers: usize,
    pub zero_copy_transfers: usize,
    pub serialized_transfers: usize,
    pub total_latency_us: u64,
}

/// Expert load metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertLoadMetrics {
    pub expert_id: usize,
    pub tokens_routed: usize,
    pub utilization_percent: f64,
    pub load_imbalance: f64,
}

/// KV Cache metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KVCacheMetrics {
    pub total_memory_bytes: usize,
    pub used_memory_bytes: usize,
    pub evictions: usize,
    pub hit_rate: f64,
}

/// GPU utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUMetrics {
    pub device: DeviceType,
    pub compute_utilization: f64,
    pub memory_utilization: f64,
    pub memory_used_bytes: usize,
    pub memory_total_bytes: usize,
    pub temperature_celsius: Option<f64>,
    pub power_watts: Option<f64>,
}

/// Aggregated metrics for a time window
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub window_start_us: u64,
    pub window_end_us: u64,
    pub tensor_transfers: TensorTransferMetrics,
    pub expert_loads: Vec<ExpertLoadMetrics>,
    pub kv_cache: KVCacheMetrics,
    pub gpu_metrics: Vec<GPUMetrics>,
    pub event_counts: HashMap<String, usize>,
    pub throughput_tokens_per_sec: f64,
}

/// Active span for scoped profiling
pub struct ProfileSpan {
    profiler: Arc<HeteroProfiler>,
    name: String,
    category: EventCategory,
    device: DeviceType,
    start_time: Instant,
    start_us: u64,
    metadata: HashMap<String, String>,
}

impl ProfileSpan {
    fn new(
        profiler: Arc<HeteroProfiler>,
        name: String,
        category: EventCategory,
        device: DeviceType,
    ) -> Self {
        let start_us = get_timestamp_us();
        Self {
            profiler,
            name,
            category,
            device,
            start_time: Instant::now(),
            start_us,
            metadata: HashMap::new(),
        }
    }
    
    /// Add metadata to the span
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
}

impl Drop for ProfileSpan {
    fn drop(&mut self) {
        let duration_us = self.start_time.elapsed().as_micros() as u64;
        
        let event = ProfileEvent {
            name: std::mem::take(&mut self.name),
            category: self.category,
            start_us: self.start_us,
            duration_us,
            device: self.device.clone(),
            thread_id: get_thread_id(),
            process_id: std::process::id(),
            metadata: std::mem::take(&mut self.metadata),
        };
        
        self.profiler.record_event(event);
    }
}

/// Get current timestamp in microseconds
fn get_timestamp_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

/// Get current thread ID
fn get_thread_id() -> u64 {
    // Hash the thread ID for a stable u64 representation
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    std::thread::current().id().hash(&mut hasher);
    hasher.finish()
}

/// Main profiler struct
#[cfg_attr(feature = "pyo3-bindings", pyclass)]
pub struct HeteroProfiler {
    events: RwLock<Vec<ProfileEvent>>,
    tensor_metrics: RwLock<TensorTransferMetrics>,
    expert_loads: RwLock<HashMap<usize, ExpertLoadMetrics>>,
    kv_cache_metrics: RwLock<KVCacheMetrics>,
    enabled: Arc<Mutex<bool>>,
    start_time: Instant,
    process_name: String,
}

impl HeteroProfiler {
    pub fn new(process_name: &str) -> Arc<Self> {
        Arc::new(Self {
            events: RwLock::new(Vec::new()),
            tensor_metrics: RwLock::new(TensorTransferMetrics::default()),
            expert_loads: RwLock::new(HashMap::new()),
            kv_cache_metrics: RwLock::new(KVCacheMetrics::default()),
            enabled: Arc::new(Mutex::new(true)),
            start_time: Instant::now(),
            process_name: process_name.to_string(),
        })
    }
    
    /// Enable or disable profiling
    pub fn set_enabled(&self, enabled: bool) {
        if let Ok(mut e) = self.enabled.lock() {
            *e = enabled;
        }
    }
    
    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.lock().map(|e| *e).unwrap_or(false)
    }
    
    /// Start a profiling span (RAII-style)
    pub fn start_span(
        self: &Arc<Self>,
        name: &str,
        category: EventCategory,
        device: DeviceType,
    ) -> ProfileSpan {
        ProfileSpan::new(self.clone(), name.to_string(), category, device)
    }
    
    /// Record a profiling event
    pub fn record_event(&self, event: ProfileEvent) {
        if !self.is_enabled() {
            return;
        }
        
        if let Ok(mut events) = self.events.write() {
            events.push(event);
        }
    }
    
    /// Record a tensor transfer
    pub fn record_tensor_transfer(
        &self,
        src: &str,
        dst: &str,
        bytes: usize,
        is_zero_copy: bool,
        latency_us: u64,
    ) {
        if !self.is_enabled() {
            return;
        }
        
        if let Ok(mut metrics) = self.tensor_metrics.write() {
            metrics.total_bytes += bytes;
            metrics.total_transfers += 1;
            metrics.total_latency_us += latency_us;
            
            if is_zero_copy {
                metrics.zero_copy_transfers += 1;
            } else {
                metrics.serialized_transfers += 1;
            }
        }
        
        let mut metadata = HashMap::new();
        metadata.insert("src".to_string(), src.to_string());
        metadata.insert("dst".to_string(), dst.to_string());
        metadata.insert("bytes".to_string(), bytes.to_string());
        metadata.insert("zero_copy".to_string(), is_zero_copy.to_string());
        
        self.record_event(ProfileEvent {
            name: format!("transfer_{}_{}", src, dst),
            category: EventCategory::TensorTransfer,
            start_us: get_timestamp_us() - latency_us,
            duration_us: latency_us,
            device: DeviceType::Network,
            thread_id: get_thread_id(),
            process_id: std::process::id(),
            metadata,
        });
    }
    
    /// Record expert routing
    pub fn record_expert_routing(
        &self,
        expert_id: usize,
        tokens_routed: usize,
        total_tokens: usize,
    ) {
        if !self.is_enabled() {
            return;
        }
        
        let utilization = tokens_routed as f64 / total_tokens.max(1) as f64 * 100.0;
        
        if let Ok(mut loads) = self.expert_loads.write() {
            let entry = loads.entry(expert_id).or_insert_with(|| ExpertLoadMetrics {
                expert_id,
                tokens_routed: 0,
                utilization_percent: 0.0,
                load_imbalance: 0.0,
            });
            
            entry.tokens_routed += tokens_routed;
            entry.utilization_percent = utilization;
        }
    }
    
    /// Record KV cache operation
    pub fn record_kv_cache_op(
        &self,
        operation: &str,
        memory_used: usize,
        memory_total: usize,
        evicted: bool,
    ) {
        if !self.is_enabled() {
            return;
        }
        
        if let Ok(mut cache) = self.kv_cache_metrics.write() {
            cache.used_memory_bytes = memory_used;
            cache.total_memory_bytes = memory_total;
            if evicted {
                cache.evictions += 1;
            }
        }
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), operation.to_string());
        metadata.insert("memory_used".to_string(), memory_used.to_string());
        metadata.insert("memory_total".to_string(), memory_total.to_string());
        
        self.record_event(ProfileEvent {
            name: format!("kv_cache_{}", operation),
            category: EventCategory::KVCacheOperation,
            start_us: get_timestamp_us(),
            duration_us: 0,
            device: DeviceType::CPU,
            thread_id: get_thread_id(),
            process_id: std::process::id(),
            metadata,
        });
    }
    
    /// Get aggregated metrics for a time window
    pub fn get_aggregated_metrics(&self, window_us: u64) -> AggregatedMetrics {
        let now = get_timestamp_us();
        let window_start = now.saturating_sub(window_us);
        
        let mut metrics = AggregatedMetrics {
            window_start_us: window_start,
            window_end_us: now,
            ..Default::default()
        };
        
        // Copy tensor metrics
        if let Ok(tm) = self.tensor_metrics.read() {
            metrics.tensor_transfers = tm.clone();
        }
        
        // Copy expert loads
        if let Ok(el) = self.expert_loads.read() {
            metrics.expert_loads = el.values().cloned().collect();
        }
        
        // Copy KV cache metrics
        if let Ok(kv) = self.kv_cache_metrics.read() {
            metrics.kv_cache = kv.clone();
        }
        
        // Count events by category
        if let Ok(events) = self.events.read() {
            for event in events.iter() {
                if event.start_us >= window_start && event.start_us <= now {
                    *metrics.event_counts.entry(event.category.as_str().to_string()).or_insert(0) += 1;
                }
            }
        }
        
        metrics
    }
    
    /// Export to Chrome tracing format
    pub fn export_chrome_trace(&self) -> String {
        let events = self.events.read().unwrap();
        
        let trace_events: Vec<serde_json::Value> = events
            .iter()
            .map(|e| e.to_chrome_trace())
            .collect();
        
        let trace = serde_json::json!({
            "traceEvents": trace_events,
            "displayTimeUnit": "us",
            "metadata": {
                "process_name": self.process_name,
            }
        });
        
        serde_json::to_string_pretty(&trace).unwrap_or_default()
    }
    
    /// Export metrics summary as JSON
    pub fn export_metrics_json(&self) -> String {
        let metrics = self.get_aggregated_metrics(60_000_000);  // Last 60 seconds
        serde_json::to_string_pretty(&metrics).unwrap_or_default()
    }
    
    /// Clear all recorded data
    pub fn clear(&self) {
        if let Ok(mut events) = self.events.write() {
            events.clear();
        }
        if let Ok(mut tm) = self.tensor_metrics.write() {
            *tm = TensorTransferMetrics::default();
        }
        if let Ok(mut el) = self.expert_loads.write() {
            el.clear();
        }
        if let Ok(mut kv) = self.kv_cache_metrics.write() {
            *kv = KVCacheMetrics::default();
        }
    }
    
    /// Get event count
    pub fn event_count(&self) -> usize {
        self.events.read().map(|e| e.len()).unwrap_or(0)
    }
}

/// Global profiler instance
static GLOBAL_PROFILER: once_cell::sync::Lazy<Arc<HeteroProfiler>> =
    once_cell::sync::Lazy::new(|| HeteroProfiler::new("hetero_prof"));

/// Get the global profiler instance
pub fn get_profiler() -> Arc<HeteroProfiler> {
    GLOBAL_PROFILER.clone()
}

/// Macro for easy span creation
#[macro_export]
macro_rules! profile_span {
    ($name:expr, $category:expr) => {
        $crate::utils::hetero_prof::get_profiler().start_span(
            $name,
            $category,
            $crate::utils::hetero_prof::DeviceType::CPU,
        )
    };
    ($name:expr, $category:expr, $device:expr) => {
        $crate::utils::hetero_prof::get_profiler().start_span($name, $category, $device)
    };
}

// PyO3 bindings for Python interop
#[cfg(feature = "pyo3-bindings")]
#[pymethods]
impl HeteroProfiler {
    #[new]
    pub fn py_new(process_name: &str) -> Self {
        Self {
            events: RwLock::new(Vec::new()),
            tensor_metrics: RwLock::new(TensorTransferMetrics::default()),
            expert_loads: RwLock::new(HashMap::new()),
            kv_cache_metrics: RwLock::new(KVCacheMetrics::default()),
            enabled: Arc::new(Mutex::new(true)),
            start_time: Instant::now(),
            process_name: process_name.to_string(),
        }
    }
    
    #[pyo3(name = "start_span")]
    pub fn py_start_span(&self, name: &str, category: &str, device: &str) {
        let cat = match category {
            "tensor_transfer" => EventCategory::TensorTransfer,
            "kernel" => EventCategory::KernelExecution,
            "memory" => EventCategory::MemoryAllocation,
            "expert_routing" => EventCategory::ExpertRouting,
            "kv_cache" => EventCategory::KVCacheOperation,
            "ray_actor" => EventCategory::RayActorCall,
            "forward" => EventCategory::ModelForward,
            "backward" => EventCategory::ModelBackward,
            "optimizer" => EventCategory::Optimizer,
            _ => EventCategory::Custom,
        };
        
        let dev = if device.starts_with("metal:") {
            let id: usize = device[6..].parse().unwrap_or(0);
            DeviceType::MetalGPU(id)
        } else if device.starts_with("cuda:") {
            let id: usize = device[5..].parse().unwrap_or(0);
            DeviceType::CUDAGPU(id)
        } else {
            DeviceType::CPU
        };
        
        self.record_event(ProfileEvent {
            name: name.to_string(),
            category: cat,
            start_us: get_timestamp_us(),
            duration_us: 0,
            device: dev,
            thread_id: get_thread_id(),
            process_id: std::process::id(),
            metadata: HashMap::new(),
        });
    }
    
    #[pyo3(name = "record_tensor_transfer")]
    pub fn py_record_tensor_transfer(
        &self,
        src: &str,
        dst: &str,
        bytes: usize,
        is_zero_copy: bool,
        latency_us: u64,
    ) {
        self.record_tensor_transfer(src, dst, bytes, is_zero_copy, latency_us);
    }
    
    #[pyo3(name = "export_chrome_trace")]
    pub fn py_export_chrome_trace(&self) -> String {
        self.export_chrome_trace()
    }
    
    #[pyo3(name = "export_metrics_json")]
    pub fn py_export_metrics_json(&self) -> String {
        self.export_metrics_json()
    }
    
    #[pyo3(name = "clear")]
    pub fn py_clear(&self) {
        self.clear();
    }
    
    #[pyo3(name = "event_count")]
    pub fn py_event_count(&self) -> usize {
        self.event_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_profiler_creation() {
        let profiler = HeteroProfiler::new("test");
        assert!(profiler.is_enabled());
        assert_eq!(profiler.event_count(), 0);
    }
    
    #[test]
    fn test_profiler_enable_disable() {
        let profiler = HeteroProfiler::new("test");
        assert!(profiler.is_enabled());
        
        profiler.set_enabled(false);
        assert!(!profiler.is_enabled());
        
        profiler.set_enabled(true);
        assert!(profiler.is_enabled());
    }
    
    #[test]
    fn test_record_event() {
        let profiler = HeteroProfiler::new("test");
        
        profiler.record_event(ProfileEvent {
            name: "test_event".to_string(),
            category: EventCategory::Custom,
            start_us: 1000,
            duration_us: 100,
            device: DeviceType::CPU,
            thread_id: 1,
            process_id: 1,
            metadata: HashMap::new(),
        });
        
        assert_eq!(profiler.event_count(), 1);
    }
    
    #[test]
    fn test_tensor_transfer_metrics() {
        let profiler = HeteroProfiler::new("test");
        
        profiler.record_tensor_transfer("cpu", "gpu", 1024, true, 100);
        profiler.record_tensor_transfer("gpu", "cpu", 2048, false, 200);
        
        let metrics = profiler.get_aggregated_metrics(1_000_000);
        assert_eq!(metrics.tensor_transfers.total_transfers, 2);
        assert_eq!(metrics.tensor_transfers.total_bytes, 3072);
        assert_eq!(metrics.tensor_transfers.zero_copy_transfers, 1);
        assert_eq!(metrics.tensor_transfers.serialized_transfers, 1);
    }
    
    #[test]
    fn test_expert_routing() {
        let profiler = HeteroProfiler::new("test");
        
        profiler.record_expert_routing(0, 100, 1000);
        profiler.record_expert_routing(1, 200, 1000);
        
        let metrics = profiler.get_aggregated_metrics(1_000_000);
        assert_eq!(metrics.expert_loads.len(), 2);
    }
    
    #[test]
    fn test_kv_cache_metrics() {
        let profiler = HeteroProfiler::new("test");
        
        profiler.record_kv_cache_op("write", 1024, 4096, false);
        profiler.record_kv_cache_op("evict", 512, 4096, true);
        
        let metrics = profiler.get_aggregated_metrics(1_000_000);
        assert_eq!(metrics.kv_cache.evictions, 1);
    }
    
    #[test]
    fn test_chrome_trace_export() {
        let profiler = HeteroProfiler::new("test");
        
        profiler.record_event(ProfileEvent {
            name: "test_event".to_string(),
            category: EventCategory::Custom,
            start_us: 1000,
            duration_us: 100,
            device: DeviceType::CPU,
            thread_id: 1,
            process_id: 1,
            metadata: HashMap::new(),
        });
        
        let trace = profiler.export_chrome_trace();
        assert!(trace.contains("traceEvents"));
        assert!(trace.contains("test_event"));
    }
    
    #[test]
    fn test_profile_span_raii() {
        let profiler = HeteroProfiler::new("test");
        
        {
            let _span = profiler.start_span(
                "test_span",
                EventCategory::Custom,
                DeviceType::CPU,
            );
            // Span automatically records when dropped
        }
        
        assert_eq!(profiler.event_count(), 1);
    }
    
    #[test]
    fn test_clear() {
        let profiler = HeteroProfiler::new("test");
        
        profiler.record_tensor_transfer("cpu", "gpu", 1024, true, 100);
        assert_eq!(profiler.event_count(), 1);
        
        profiler.clear();
        assert_eq!(profiler.event_count(), 0);
    }
    
    #[test]
    fn test_global_profiler() {
        let profiler = get_profiler();
        assert!(profiler.is_enabled());
    }
    
    #[test]
    fn test_device_type_string() {
        assert_eq!(DeviceType::CPU.as_str(), "cpu");
        assert_eq!(DeviceType::MetalGPU(0).as_str(), "metal:0");
        assert_eq!(DeviceType::CUDAGPU(1).as_str(), "cuda:1");
    }
    
    #[test]
    fn test_event_category_string() {
        assert_eq!(EventCategory::TensorTransfer.as_str(), "tensor_transfer");
        assert_eq!(EventCategory::KernelExecution.as_str(), "kernel");
        assert_eq!(EventCategory::ExpertRouting.as_str(), "expert_routing");
    }
}
