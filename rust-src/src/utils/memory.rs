//! Memory Profiling and Management for DeepSeek Rust
//!
//! This module provides memory profiling utilities:
//! - Memory statistics collection
//! - Peak memory tracking
//! - Memory pool management hints
//! - NVTX-style profiling markers
//!
//! # Example
//! ```rust,ignore
//! use deepseek::utils::memory::{MemoryProfiler, MemoryStats};
//!
//! let mut profiler = MemoryProfiler::new();
//! profiler.start_region("forward_pass");
//! // ... forward pass ...
//! profiler.end_region();
//! let stats = profiler.get_stats();
//! ```

use candle_core::Device;
use std::time::{Duration, Instant};

/// Memory statistics container
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Currently allocated memory in bytes
    pub allocated_bytes: u64,
    /// Peak allocated memory in bytes
    pub peak_allocated_bytes: u64,
    /// Number of allocations
    pub num_allocations: u64,
    /// Number of deallocations
    pub num_deallocations: u64,
    /// Reserved memory (may be higher than allocated)
    pub reserved_bytes: u64,
}

impl MemoryStats {
    /// Get allocated memory in MB
    pub fn allocated_mb(&self) -> f64 {
        self.allocated_bytes as f64 / 1024.0 / 1024.0
    }

    /// Get peak allocated memory in MB
    pub fn peak_allocated_mb(&self) -> f64 {
        self.peak_allocated_bytes as f64 / 1024.0 / 1024.0
    }

    /// Get reserved memory in MB
    pub fn reserved_mb(&self) -> f64 {
        self.reserved_bytes as f64 / 1024.0 / 1024.0
    }
}

/// Profiling region for timing
#[derive(Debug, Clone)]
pub struct ProfileRegion {
    /// Region name
    pub name: String,
    /// Start time
    pub start: Instant,
    /// Duration (set when region ends)
    pub duration: Option<Duration>,
    /// Memory at start
    pub start_memory: u64,
    /// Memory at end
    pub end_memory: Option<u64>,
}

impl ProfileRegion {
    /// Create a new region
    pub fn new(name: &str, start_memory: u64) -> Self {
        Self {
            name: name.to_string(),
            start: Instant::now(),
            duration: None,
            start_memory,
            end_memory: None,
        }
    }

    /// End the region
    pub fn end(&mut self, end_memory: u64) {
        self.duration = Some(self.start.elapsed());
        self.end_memory = Some(end_memory);
    }

    /// Get memory delta
    pub fn memory_delta(&self) -> Option<i64> {
        self.end_memory.map(|end| end as i64 - self.start_memory as i64)
    }
}

/// Memory profiler for tracking GPU/CPU memory usage
#[derive(Debug)]
pub struct MemoryProfiler {
    /// Device being profiled
    device: Device,
    /// Current statistics
    stats: MemoryStats,
    /// Active profiling regions (stack)
    active_regions: Vec<ProfileRegion>,
    /// Completed regions
    completed_regions: Vec<ProfileRegion>,
    /// Log interval (in steps)
    log_interval: usize,
    /// Current step
    current_step: usize,
    /// Peak tracking enabled
    track_peak: bool,
}

impl MemoryProfiler {
    /// Create a new profiler for the given device
    pub fn new(device: Device) -> Self {
        Self {
            device,
            stats: MemoryStats::default(),
            active_regions: Vec::new(),
            completed_regions: Vec::new(),
            log_interval: 100,
            current_step: 0,
            track_peak: true,
        }
    }

    /// Set logging interval
    pub fn with_log_interval(mut self, interval: usize) -> Self {
        self.log_interval = interval;
        self
    }

    /// Start a profiling region
    pub fn start_region(&mut self, name: &str) {
        let current_memory = self.get_current_memory();
        let region = ProfileRegion::new(name, current_memory);
        self.active_regions.push(region);
    }

    /// End the most recent profiling region
    pub fn end_region(&mut self) {
        if let Some(mut region) = self.active_regions.pop() {
            let current_memory = self.get_current_memory();
            region.end(current_memory);
            self.completed_regions.push(region);
        }
    }

    /// Get current memory usage (device-specific)
    fn get_current_memory(&self) -> u64 {
        // Note: Candle doesn't expose direct memory APIs
        // In a real implementation, we'd use CUDA/Metal APIs
        // For now, return placeholder
        match &self.device {
            Device::Cuda(_) => {
                // Would call cudaMemGetInfo here
                0
            }
            Device::Metal(_) => {
                // Would use Metal APIs
                0
            }
            Device::Cpu => {
                // Would use system memory APIs
                0
            }
        }
    }

    /// Update statistics (call after each training step)
    pub fn step(&mut self) -> Option<MemoryStats> {
        self.current_step += 1;

        // Update current stats
        let current = self.get_current_memory();
        self.stats.allocated_bytes = current;
        if self.track_peak && current > self.stats.peak_allocated_bytes {
            self.stats.peak_allocated_bytes = current;
        }

        // Log at intervals
        if self.current_step % self.log_interval == 0 {
            Some(self.stats.clone())
        } else {
            None
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Get completed profiling regions
    pub fn get_completed_regions(&self) -> &[ProfileRegion] {
        &self.completed_regions
    }

    /// Reset peak memory tracking
    pub fn reset_peak(&mut self) {
        self.stats.peak_allocated_bytes = self.stats.allocated_bytes;
    }

    /// Clear completed regions
    pub fn clear_regions(&mut self) {
        self.completed_regions.clear();
    }

    /// Generate profiling report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Memory Profiling Report ===\n\n");

        report.push_str(&format!(
            "Current Memory: {:.2} MB\n",
            self.stats.allocated_mb()
        ));
        report.push_str(&format!(
            "Peak Memory: {:.2} MB\n",
            self.stats.peak_allocated_mb()
        ));
        report.push_str(&format!(
            "Reserved Memory: {:.2} MB\n\n",
            self.stats.reserved_mb()
        ));

        if !self.completed_regions.is_empty() {
            report.push_str("Profiling Regions:\n");
            for region in &self.completed_regions {
                let duration = region
                    .duration
                    .map(|d| format!("{:.2}ms", d.as_secs_f64() * 1000.0))
                    .unwrap_or_else(|| "N/A".to_string());
                let memory_delta = region
                    .memory_delta()
                    .map(|d| format!("{:+.2} MB", d as f64 / 1024.0 / 1024.0))
                    .unwrap_or_else(|| "N/A".to_string());
                report.push_str(&format!(
                    "  {} - Time: {}, Memory: {}\n",
                    region.name, duration, memory_delta
                ));
            }
        }

        report
    }
}

/// Memory budget configuration
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Maximum memory budget in bytes
    pub max_memory_bytes: u64,
    /// Memory fraction to use (0.0 - 1.0)
    pub memory_fraction: f64,
    /// Enable automatic batch size adjustment
    pub auto_adjust_batch: bool,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            max_memory_bytes: 0, // Will be auto-detected
            memory_fraction: 0.9,
            auto_adjust_batch: true,
            min_batch_size: 1,
            max_batch_size: 256,
        }
    }
}

impl MemoryBudget {
    /// Create budget with specific memory limit
    pub fn with_max_memory(max_memory_mb: u64) -> Self {
        Self {
            max_memory_bytes: max_memory_mb * 1024 * 1024,
            ..Default::default()
        }
    }

    /// Get budget in MB
    pub fn budget_mb(&self) -> f64 {
        (self.max_memory_bytes as f64 * self.memory_fraction) / 1024.0 / 1024.0
    }

    /// Check if memory usage is within budget
    pub fn within_budget(&self, current_bytes: u64) -> bool {
        current_bytes <= (self.max_memory_bytes as f64 * self.memory_fraction) as u64
    }

    /// Suggest batch size based on memory usage (linear adjustment)
    pub fn suggest_batch_size(&self, current_batch: usize, current_memory_bytes: u64) -> usize {
        if !self.auto_adjust_batch {
            return current_batch;
        }

        let budget = (self.max_memory_bytes as f64 * self.memory_fraction) as u64;

        if current_memory_bytes > budget {
            // Reduce batch size
            let reduction_factor = budget as f64 / current_memory_bytes as f64;
            let new_batch = (current_batch as f64 * reduction_factor * 0.9) as usize;
            new_batch.max(self.min_batch_size)
        } else if current_memory_bytes < budget / 2 {
            // Can increase batch size
            let increase_factor = budget as f64 / current_memory_bytes as f64 / 2.0;
            let new_batch = (current_batch as f64 * increase_factor.min(1.5)) as usize;
            new_batch.min(self.max_batch_size)
        } else {
            current_batch
        }
    }

    /// Find optimal batch size using binary search (recommended)
    ///
    /// Given the memory required per sample, finds the maximum batch size
    /// that fits within the memory budget using binary search for faster
    /// convergence than linear adjustment.
    ///
    /// # Arguments
    /// * `memory_per_sample_bytes` - Memory required per sample in bytes
    ///
    /// # Returns
    /// The optimal batch size within configured bounds
    pub fn find_optimal_batch_size_binary(&self, memory_per_sample_bytes: u64) -> usize {
        if !self.auto_adjust_batch {
            return self.max_batch_size;
        }

        let budget = (self.max_memory_bytes as f64 * self.memory_fraction) as u64;
        
        // Binary search for optimal batch size
        let mut low = self.min_batch_size;
        let mut high = self.max_batch_size;
        let mut best = self.min_batch_size;

        while low <= high {
            let mid = low + (high - low) / 2;
            let total_memory = (mid as u64) * memory_per_sample_bytes;

            if total_memory <= budget {
                best = mid;
                low = mid + 1;
            } else {
                if mid == 0 {
                    break;
                }
                high = mid - 1;
            }
        }

        best.max(self.min_batch_size).min(self.max_batch_size)
    }

    /// Find optimal batch size with a default fallback when auto-adjust is disabled
    ///
    /// # Arguments
    /// * `memory_per_sample_bytes` - Memory required per sample in bytes
    /// * `default_batch_size` - Batch size to return if auto-adjust is disabled
    ///
    /// # Returns
    /// The optimal batch size or the default if auto-adjust is disabled
    pub fn find_optimal_batch_size_binary_with_default(
        &self,
        memory_per_sample_bytes: u64,
        default_batch_size: usize,
    ) -> usize {
        if !self.auto_adjust_batch {
            return default_batch_size;
        }
        self.find_optimal_batch_size_binary(memory_per_sample_bytes)
    }
}

/// RAII guard for profiling regions
pub struct ProfileGuard<'a> {
    profiler: &'a mut MemoryProfiler,
}

impl<'a> ProfileGuard<'a> {
    /// Create a new guard that starts a region
    pub fn new(profiler: &'a mut MemoryProfiler, name: &str) -> Self {
        profiler.start_region(name);
        Self { profiler }
    }
}

impl<'a> Drop for ProfileGuard<'a> {
    fn drop(&mut self) {
        self.profiler.end_region();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats {
            allocated_bytes: 1024 * 1024 * 100, // 100 MB
            peak_allocated_bytes: 1024 * 1024 * 150,
            num_allocations: 100,
            num_deallocations: 50,
            reserved_bytes: 1024 * 1024 * 200,
        };

        assert!((stats.allocated_mb() - 100.0).abs() < 0.01);
        assert!((stats.peak_allocated_mb() - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_budget() {
        let budget = MemoryBudget::with_max_memory(1000); // 1000 MB

        assert!(budget.within_budget(800 * 1024 * 1024)); // 800 MB
        assert!(!budget.within_budget(950 * 1024 * 1024)); // 950 MB exceeds 90%
    }

    #[test]
    fn test_batch_size_suggestion() {
        let budget = MemoryBudget {
            max_memory_bytes: 1024 * 1024 * 1024, // 1 GB
            memory_fraction: 0.9,
            auto_adjust_batch: true,
            min_batch_size: 1,
            max_batch_size: 256,
        };

        // When over budget, should reduce
        let new_batch = budget.suggest_batch_size(32, 1024 * 1024 * 1000); // 1000 MB
        assert!(new_batch < 32);

        // When well under budget, should increase
        let new_batch = budget.suggest_batch_size(32, 1024 * 1024 * 200); // 200 MB
        assert!(new_batch > 32);
    }
}
