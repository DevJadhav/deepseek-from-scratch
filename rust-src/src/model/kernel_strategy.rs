//! JIT vs AOT Kernel Strategy Analysis
//!
//! This module provides a comprehensive analysis and implementation of
//! Just-In-Time (JIT) vs Ahead-Of-Time (AOT) kernel compilation strategies
//! for different backends.
//!
//! # Strategy Summary
//!
//! | Backend | Strategy | Implementation | Reason |
//! |---------|----------|----------------|--------|
//! | Metal   | AOT      | Candle Metal   | Metal shaders are pre-compiled |
//! | CUDA    | AOT      | Candle CUDA    | NVRTC compilation is expensive |
//! | Triton  | JIT      | Python Triton  | Specializes to input shapes |
//! | MLX     | JIT      | MLX compile    | Dynamic graph optimization |
//!
//! # Trade-offs
//!
//! ## AOT (Ahead-Of-Time)
//! - Pros: No runtime compilation overhead, deterministic performance
//! - Cons: Generic kernels, no shape specialization, larger binary
//!
//! ## JIT (Just-In-Time)
//! - Pros: Shape specialization, optimal register usage, fusion opportunities
//! - Cons: First-run compilation, cache management, non-determinism

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Kernel compilation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationStrategy {
    /// Ahead-of-time compilation (pre-compiled kernels)
    AOT,
    /// Just-in-time compilation (runtime compilation)
    JIT,
    /// Hybrid: AOT for common shapes, JIT for uncommon
    Hybrid,
}

/// Backend type for kernel dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Metal backend (macOS/iOS)
    Metal,
    /// CUDA backend (NVIDIA GPUs)
    Cuda,
    /// Triton backend (JIT compilation)
    Triton,
    /// MLX backend (Apple Silicon)
    Mlx,
    /// CPU fallback
    Cpu,
}

impl Backend {
    /// Get the default compilation strategy for this backend
    pub fn default_strategy(&self) -> CompilationStrategy {
        match self {
            Self::Metal => CompilationStrategy::AOT,
            Self::Cuda => CompilationStrategy::AOT,
            Self::Triton => CompilationStrategy::JIT,
            Self::Mlx => CompilationStrategy::JIT,
            Self::Cpu => CompilationStrategy::AOT,
        }
    }

    /// Check if this backend supports JIT compilation
    pub fn supports_jit(&self) -> bool {
        matches!(self, Self::Triton | Self::Mlx)
    }

    /// Check if this backend supports shape specialization
    pub fn supports_shape_specialization(&self) -> bool {
        matches!(self, Self::Triton | Self::Mlx)
    }
}

/// Kernel shape key for caching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelKey {
    /// Kernel name/type
    pub name: String,
    /// Input shapes
    pub shapes: Vec<Vec<usize>>,
    /// Data type
    pub dtype: String,
    /// Backend-specific options
    pub options: String,
}

impl KernelKey {
    /// Create a new kernel key
    pub fn new(name: &str, shapes: Vec<Vec<usize>>, dtype: &str) -> Self {
        Self {
            name: name.to_string(),
            shapes,
            dtype: dtype.to_string(),
            options: String::new(),
        }
    }

    /// Create key with options
    pub fn with_options(mut self, options: &str) -> Self {
        self.options = options.to_string();
        self
    }
}

/// Compiled kernel information
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    /// Kernel key
    pub key: KernelKey,
    /// Compilation time
    pub compile_time: Duration,
    /// Whether this was JIT compiled
    pub is_jit: bool,
    /// Number of invocations
    pub invocation_count: u64,
    /// Total execution time
    pub total_exec_time: Duration,
}

impl CompiledKernel {
    /// Create a new compiled kernel entry
    pub fn new(key: KernelKey, compile_time: Duration, is_jit: bool) -> Self {
        Self {
            key,
            compile_time,
            is_jit,
            invocation_count: 0,
            total_exec_time: Duration::ZERO,
        }
    }

    /// Record an invocation
    pub fn record_invocation(&mut self, exec_time: Duration) {
        self.invocation_count += 1;
        self.total_exec_time += exec_time;
    }

    /// Get average execution time
    pub fn avg_exec_time(&self) -> Duration {
        if self.invocation_count > 0 {
            self.total_exec_time / self.invocation_count as u32
        } else {
            Duration::ZERO
        }
    }

    /// Check if JIT compilation has amortized
    pub fn is_amortized(&self, threshold: u64) -> bool {
        self.invocation_count >= threshold
    }
}

/// Kernel cache for JIT-compiled kernels
pub struct KernelCache {
    /// Cached kernels by key
    kernels: Arc<Mutex<HashMap<KernelKey, CompiledKernel>>>,
    /// Maximum cache size
    max_size: usize,
    /// Cache hits
    hits: Arc<Mutex<u64>>,
    /// Cache misses
    misses: Arc<Mutex<u64>>,
}

impl Default for KernelCache {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl KernelCache {
    /// Create a new kernel cache
    pub fn new(max_size: usize) -> Self {
        Self {
            kernels: Arc::new(Mutex::new(HashMap::new())),
            max_size,
            hits: Arc::new(Mutex::new(0)),
            misses: Arc::new(Mutex::new(0)),
        }
    }

    /// Get a cached kernel
    pub fn get(&self, key: &KernelKey) -> Option<CompiledKernel> {
        let kernels = self.kernels.lock().unwrap();
        if let Some(kernel) = kernels.get(key) {
            *self.hits.lock().unwrap() += 1;
            Some(kernel.clone())
        } else {
            *self.misses.lock().unwrap() += 1;
            None
        }
    }

    /// Insert a compiled kernel
    pub fn insert(&self, kernel: CompiledKernel) {
        let mut kernels = self.kernels.lock().unwrap();
        
        // Evict if at capacity (simple LRU could be improved)
        if kernels.len() >= self.max_size {
            // Remove least-invoked kernel
            if let Some(key_to_remove) = kernels
                .iter()
                .min_by_key(|(_, k)| k.invocation_count)
                .map(|(k, _)| k.clone())
            {
                kernels.remove(&key_to_remove);
            }
        }
        
        kernels.insert(kernel.key.clone(), kernel);
    }

    /// Update kernel statistics
    pub fn update_stats(&self, key: &KernelKey, exec_time: Duration) {
        let mut kernels = self.kernels.lock().unwrap();
        if let Some(kernel) = kernels.get_mut(key) {
            kernel.record_invocation(exec_time);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let kernels = self.kernels.lock().unwrap();
        let hits = *self.hits.lock().unwrap();
        let misses = *self.misses.lock().unwrap();
        
        CacheStats {
            size: kernels.len(),
            max_size: self.max_size,
            hits,
            misses,
            hit_rate: if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.kernels.lock().unwrap().clear();
        *self.hits.lock().unwrap() = 0;
        *self.misses.lock().unwrap() = 0;
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current cache size
    pub size: usize,
    /// Maximum cache size
    pub max_size: usize,
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Hit rate (0.0 - 1.0)
    pub hit_rate: f64,
}

/// Metal-specific AOT configuration
#[derive(Debug, Clone)]
pub struct MetalAOTConfig {
    /// Enable pipeline state caching
    pub enable_caching: bool,
    /// Precompile common kernel variants
    pub precompile_variants: bool,
    /// Common tile sizes to precompile
    pub precompile_tile_sizes: Vec<(usize, usize, usize)>,
}

impl Default for MetalAOTConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            precompile_variants: true,
            precompile_tile_sizes: vec![
                (16, 16, 16),
                (32, 32, 32),
                (64, 64, 32),
            ],
        }
    }
}

/// CUDA-specific AOT configuration
#[derive(Debug, Clone)]
pub struct CudaAOTConfig {
    /// Enable kernel caching
    pub enable_caching: bool,
    /// Use cubin cache
    pub use_cubin_cache: bool,
    /// Precompile for specific SM versions
    pub target_sm_versions: Vec<(u32, u32)>,
    /// Use TMA when available (Hopper+)
    pub use_tma: bool,
    /// Use Warpgroup when available (Hopper+)
    pub use_warpgroup: bool,
}

impl Default for CudaAOTConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            use_cubin_cache: true,
            target_sm_versions: vec![
                (7, 0),  // Volta
                (8, 0),  // Ampere
                (9, 0),  // Hopper
            ],
            use_tma: true,
            use_warpgroup: true,
        }
    }
}

/// Triton-specific JIT configuration
#[derive(Debug, Clone)]
pub struct TritonJITConfig {
    /// Enable kernel caching
    pub enable_caching: bool,
    /// Cache directory
    pub cache_dir: Option<String>,
    /// Enable auto-tuning
    pub enable_autotune: bool,
    /// Number of warmup iterations for auto-tuning
    pub autotune_warmup: usize,
    /// Number of benchmark iterations for auto-tuning
    pub autotune_reps: usize,
}

impl Default for TritonJITConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_dir: None,
            enable_autotune: true,
            autotune_warmup: 25,
            autotune_reps: 100,
        }
    }
}

/// MLX-specific JIT configuration
#[derive(Debug, Clone)]
pub struct MlxJITConfig {
    /// Enable graph caching
    pub enable_caching: bool,
    /// Enable fusion optimization
    pub enable_fusion: bool,
    /// Enable constant folding
    pub enable_constant_folding: bool,
    /// Compile mode (lazy vs eager)
    pub compile_mode: MlxCompileMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxCompileMode {
    /// Lazy compilation (compile on first use)
    Lazy,
    /// Eager compilation (compile immediately)
    Eager,
}

impl Default for MlxJITConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            enable_fusion: true,
            enable_constant_folding: true,
            compile_mode: MlxCompileMode::Lazy,
        }
    }
}

/// Unified kernel strategy configuration
#[derive(Debug, Clone)]
pub struct KernelStrategyConfig {
    /// Metal AOT configuration
    pub metal: MetalAOTConfig,
    /// CUDA AOT configuration  
    pub cuda: CudaAOTConfig,
    /// Triton JIT configuration
    pub triton: TritonJITConfig,
    /// MLX JIT configuration
    pub mlx: MlxJITConfig,
    /// Enable hybrid strategy (AOT + JIT)
    pub enable_hybrid: bool,
    /// Threshold for JIT amortization (number of invocations)
    pub jit_amortization_threshold: u64,
}

impl Default for KernelStrategyConfig {
    fn default() -> Self {
        Self {
            metal: MetalAOTConfig::default(),
            cuda: CudaAOTConfig::default(),
            triton: TritonJITConfig::default(),
            mlx: MlxJITConfig::default(),
            enable_hybrid: true,
            jit_amortization_threshold: 10,
        }
    }
}

/// Kernel strategy manager
pub struct KernelStrategyManager {
    /// Configuration
    config: KernelStrategyConfig,
    /// Kernel cache
    cache: KernelCache,
    /// Backend-specific caches
    backend_caches: HashMap<Backend, KernelCache>,
}

impl Default for KernelStrategyManager {
    fn default() -> Self {
        Self::new(KernelStrategyConfig::default())
    }
}

impl KernelStrategyManager {
    /// Create a new strategy manager
    pub fn new(config: KernelStrategyConfig) -> Self {
        let mut backend_caches = HashMap::new();
        backend_caches.insert(Backend::Metal, KernelCache::new(256));
        backend_caches.insert(Backend::Cuda, KernelCache::new(512));
        backend_caches.insert(Backend::Triton, KernelCache::new(1024));
        backend_caches.insert(Backend::Mlx, KernelCache::new(512));
        
        Self {
            config,
            cache: KernelCache::new(2048),
            backend_caches,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &KernelStrategyConfig {
        &self.config
    }

    /// Decide compilation strategy for a kernel
    pub fn decide_strategy(&self, backend: Backend, key: &KernelKey) -> CompilationStrategy {
        // Check if hybrid is enabled
        if !self.config.enable_hybrid {
            return backend.default_strategy();
        }

        // Check cache for existing compiled kernel
        if let Some(kernel) = self.cache.get(key) {
            if kernel.is_amortized(self.config.jit_amortization_threshold) {
                // JIT has amortized, use JIT
                return CompilationStrategy::JIT;
            }
        }

        // For backends that support JIT, use hybrid strategy
        if backend.supports_jit() {
            // Check if this is a common shape that should use AOT
            if self.is_common_shape(&key.shapes) {
                CompilationStrategy::AOT
            } else {
                CompilationStrategy::JIT
            }
        } else {
            CompilationStrategy::AOT
        }
    }

    /// Check if shapes are common (suitable for AOT)
    fn is_common_shape(&self, shapes: &[Vec<usize>]) -> bool {
        // Common shapes are powers of 2 and standard sizes
        for shape in shapes {
            for &dim in shape {
                if dim > 8192 || !is_nice_number(dim) {
                    return false;
                }
            }
        }
        true
    }

    /// Get or compile a kernel
    pub fn get_kernel(&self, backend: Backend, key: KernelKey) -> CompiledKernel {
        // Check cache first
        if let Some(kernel) = self.cache.get(&key) {
            return kernel;
        }

        // Compile new kernel
        let strategy = self.decide_strategy(backend, &key);
        let compile_start = Instant::now();
        
        // Simulate compilation (in real implementation, this would call backend-specific compilers)
        let compile_time = compile_start.elapsed();
        
        let kernel = CompiledKernel::new(
            key.clone(),
            compile_time,
            strategy == CompilationStrategy::JIT,
        );
        
        self.cache.insert(kernel.clone());
        kernel
    }

    /// Record kernel execution
    pub fn record_execution(&self, key: &KernelKey, exec_time: Duration) {
        self.cache.update_stats(key, exec_time);
    }

    /// Get cache statistics
    pub fn stats(&self) -> HashMap<String, CacheStats> {
        let mut stats = HashMap::new();
        stats.insert("global".to_string(), self.cache.stats());
        
        for (backend, cache) in &self.backend_caches {
            stats.insert(format!("{:?}", backend).to_lowercase(), cache.stats());
        }
        
        stats
    }
}

/// Check if a number is "nice" for kernel dimensions (power of 2, multiple of 32, etc.)
fn is_nice_number(n: usize) -> bool {
    if n == 0 {
        return false;
    }
    // Power of 2
    if n.count_ones() == 1 {
        return true;
    }
    // Multiple of 32 or 64
    if n % 64 == 0 || n % 32 == 0 {
        return true;
    }
    // Common attention dimensions
    if n == 128 || n == 256 || n == 512 || n == 1024 || n == 2048 || n == 4096 {
        return true;
    }
    false
}

/// Analysis result for kernel strategy selection
#[derive(Debug, Clone)]
pub struct StrategyAnalysis {
    /// Recommended strategy
    pub recommended: CompilationStrategy,
    /// Reasoning
    pub reasoning: Vec<String>,
    /// Expected compile time (if known)
    pub expected_compile_time: Option<Duration>,
    /// Expected speedup from JIT specialization
    pub expected_jit_speedup: Option<f64>,
    /// Break-even invocations for JIT
    pub jit_break_even_invocations: Option<u64>,
}

impl StrategyAnalysis {
    /// Analyze strategy for a kernel
    pub fn analyze(backend: Backend, key: &KernelKey) -> Self {
        let mut reasoning = Vec::new();
        let mut recommended = backend.default_strategy();
        let mut expected_jit_speedup = None;
        let mut jit_break_even = None;

        match backend {
            Backend::Metal => {
                reasoning.push("Metal uses AOT compilation via MTLComputePipelineState".to_string());
                reasoning.push("Metal shaders are pre-compiled into metallib".to_string());
                reasoning.push("Pipeline state caching eliminates recompilation".to_string());
            }
            Backend::Cuda => {
                reasoning.push("CUDA uses AOT via NVCC or cubin caching".to_string());
                reasoning.push("NVRTC JIT compilation is expensive (100s of ms)".to_string());
                if key.shapes.iter().any(|s| s.iter().any(|&d| d > 4096)) {
                    reasoning.push("Large tensors benefit from shape-specialized tiling".to_string());
                    expected_jit_speedup = Some(1.1); // 10% speedup
                    jit_break_even = Some(100); // Need 100 invocations to amortize
                }
            }
            Backend::Triton => {
                reasoning.push("Triton is designed for JIT compilation".to_string());
                reasoning.push("Shape specialization can yield 2-3x speedup".to_string());
                reasoning.push("Auto-tuning selects optimal tile sizes".to_string());
                recommended = CompilationStrategy::JIT;
                expected_jit_speedup = Some(2.0);
                jit_break_even = Some(50);
            }
            Backend::Mlx => {
                reasoning.push("MLX uses lazy JIT compilation".to_string());
                reasoning.push("Graph fusion optimizes memory bandwidth".to_string());
                reasoning.push("Compilation is fast (~10ms)".to_string());
                recommended = CompilationStrategy::JIT;
                expected_jit_speedup = Some(1.5);
                jit_break_even = Some(10);
            }
            Backend::Cpu => {
                reasoning.push("CPU uses pre-compiled BLAS/LAPACK kernels".to_string());
                recommended = CompilationStrategy::AOT;
            }
        }

        Self {
            recommended,
            reasoning,
            expected_compile_time: None,
            expected_jit_speedup,
            jit_break_even_invocations: jit_break_even,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_default_strategy() {
        assert_eq!(Backend::Metal.default_strategy(), CompilationStrategy::AOT);
        assert_eq!(Backend::Cuda.default_strategy(), CompilationStrategy::AOT);
        assert_eq!(Backend::Triton.default_strategy(), CompilationStrategy::JIT);
        assert_eq!(Backend::Mlx.default_strategy(), CompilationStrategy::JIT);
    }

    #[test]
    fn test_kernel_key() {
        let key = KernelKey::new(
            "matmul",
            vec![vec![1024, 1024], vec![1024, 1024]],
            "f32",
        );
        assert_eq!(key.name, "matmul");
        assert_eq!(key.shapes.len(), 2);
    }

    #[test]
    fn test_kernel_cache() {
        let cache = KernelCache::new(10);
        let key = KernelKey::new("test", vec![vec![32, 32]], "f32");
        
        // Initially empty
        assert!(cache.get(&key).is_none());
        
        // Insert
        let kernel = CompiledKernel::new(key.clone(), Duration::from_millis(10), true);
        cache.insert(kernel);
        
        // Now found
        assert!(cache.get(&key).is_some());
        
        let stats = cache.stats();
        assert_eq!(stats.size, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_compiled_kernel_amortization() {
        let key = KernelKey::new("test", vec![vec![32]], "f32");
        let mut kernel = CompiledKernel::new(key, Duration::from_millis(100), true);
        
        assert!(!kernel.is_amortized(10));
        
        for _ in 0..10 {
            kernel.record_invocation(Duration::from_millis(1));
        }
        
        assert!(kernel.is_amortized(10));
    }

    #[test]
    fn test_is_nice_number() {
        assert!(is_nice_number(32));
        assert!(is_nice_number(64));
        assert!(is_nice_number(128));
        assert!(is_nice_number(256));
        assert!(is_nice_number(1024));
        assert!(is_nice_number(4096));
        assert!(!is_nice_number(0));
        assert!(!is_nice_number(37));
    }

    #[test]
    fn test_strategy_analysis() {
        let key = KernelKey::new("attention", vec![vec![2048, 64]], "f32");
        
        let analysis = StrategyAnalysis::analyze(Backend::Triton, &key);
        assert_eq!(analysis.recommended, CompilationStrategy::JIT);
        assert!(analysis.expected_jit_speedup.is_some());
        
        let analysis = StrategyAnalysis::analyze(Backend::Metal, &key);
        assert_eq!(analysis.recommended, CompilationStrategy::AOT);
    }

    #[test]
    fn test_kernel_strategy_manager() {
        let manager = KernelStrategyManager::default();
        
        let key = KernelKey::new("matmul", vec![vec![1024, 1024]], "f32");
        let kernel = manager.get_kernel(Backend::Cuda, key.clone());
        
        assert_eq!(kernel.key.name, "matmul");
        
        // Record some executions
        for _ in 0..5 {
            manager.record_execution(&key, Duration::from_millis(1));
        }
        
        let stats = manager.stats();
        assert!(stats.contains_key("global"));
    }
}
