//! Heterogeneous Ray Scheduling for Multi-Architecture Clusters
//!
//! This module provides resource tagging, auto-detection, and scheduling
//! support for heterogeneous clusters (Apple Silicon, NVIDIA GPUs, etc.)
//!
//! Features:
//! - Custom resource types: `metal`, `cuda_compute_cap`, `memory_gb`
//! - Resource auto-detection for each node
//! - Placement groups for pipeline parallelism across architectures
//! - Node health monitoring for heterogeneous clusters

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Custom resource types for heterogeneous scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// Apple Metal GPU (Apple Silicon)
    Metal,
    /// NVIDIA CUDA GPU
    Cuda,
    /// CUDA compute capability (e.g., 8.0 for A100, 9.0 for H100)
    CudaComputeCap,
    /// System memory in GB
    MemoryGb,
    /// GPU memory in GB
    GpuMemoryGb,
    /// Number of CPU cores
    CpuCores,
    /// Whether the node has high-bandwidth memory (HBM)
    HasHbm,
    /// Network bandwidth in Gbps
    NetworkBandwidthGbps,
    /// Custom resource type
    Custom(u32),
}

impl ResourceType {
    /// Get the string identifier for this resource type
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceType::Metal => "metal",
            ResourceType::Cuda => "cuda",
            ResourceType::CudaComputeCap => "cuda_compute_cap",
            ResourceType::MemoryGb => "memory_gb",
            ResourceType::GpuMemoryGb => "gpu_memory_gb",
            ResourceType::CpuCores => "cpu_cores",
            ResourceType::HasHbm => "has_hbm",
            ResourceType::NetworkBandwidthGbps => "network_bandwidth_gbps",
            ResourceType::Custom(_) => "custom",
        }
    }

    /// Parse resource type from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "metal" => Some(ResourceType::Metal),
            "cuda" => Some(ResourceType::Cuda),
            "cuda_compute_cap" => Some(ResourceType::CudaComputeCap),
            "memory_gb" => Some(ResourceType::MemoryGb),
            "gpu_memory_gb" => Some(ResourceType::GpuMemoryGb),
            "cpu_cores" => Some(ResourceType::CpuCores),
            "has_hbm" => Some(ResourceType::HasHbm),
            "network_bandwidth_gbps" => Some(ResourceType::NetworkBandwidthGbps),
            _ => None,
        }
    }
}

/// Resource requirements for a task or actor
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Required resources (type -> quantity)
    pub resources: HashMap<String, f64>,
    /// Soft preferences (not strictly required)
    pub preferences: HashMap<String, f64>,
    /// Node affinity labels
    pub node_affinity: Vec<String>,
    /// Anti-affinity labels (avoid nodes with these labels)
    pub node_anti_affinity: Vec<String>,
}

impl ResourceRequirements {
    /// Create new empty resource requirements
    pub fn new() -> Self {
        Self::default()
    }

    /// Require a specific resource
    pub fn require(mut self, resource_type: &str, quantity: f64) -> Self {
        self.resources.insert(resource_type.to_string(), quantity);
        self
    }

    /// Add a soft preference for a resource
    pub fn prefer(mut self, resource_type: &str, quantity: f64) -> Self {
        self.preferences.insert(resource_type.to_string(), quantity);
        self
    }

    /// Require Metal GPU
    pub fn require_metal(self, count: u32) -> Self {
        self.require("metal", count as f64)
    }

    /// Require CUDA GPU
    pub fn require_cuda(self, count: u32) -> Self {
        self.require("cuda", count as f64)
    }

    /// Require minimum CUDA compute capability
    pub fn require_cuda_compute_cap(self, major: u32, minor: u32) -> Self {
        let cap = major as f64 + minor as f64 / 10.0;
        self.require("cuda_compute_cap", cap)
    }

    /// Require minimum memory in GB
    pub fn require_memory_gb(self, gb: u32) -> Self {
        self.require("memory_gb", gb as f64)
    }

    /// Require minimum GPU memory in GB
    pub fn require_gpu_memory_gb(self, gb: u32) -> Self {
        self.require("gpu_memory_gb", gb as f64)
    }

    /// Add node affinity
    pub fn with_node_affinity(mut self, label: &str) -> Self {
        self.node_affinity.push(label.to_string());
        self
    }

    /// Add node anti-affinity
    pub fn with_node_anti_affinity(mut self, label: &str) -> Self {
        self.node_anti_affinity.push(label.to_string());
        self
    }

    /// Create requirements for Apple Silicon node
    pub fn apple_silicon(memory_gb: u32) -> Self {
        Self::new()
            .require_metal(1)
            .require_memory_gb(memory_gb)
    }

    /// Create requirements for NVIDIA GPU node (H100)
    pub fn nvidia_h100() -> Self {
        Self::new()
            .require_cuda(1)
            .require_cuda_compute_cap(9, 0)
            .require_gpu_memory_gb(80)
    }

    /// Create requirements for NVIDIA GPU node (A100)
    pub fn nvidia_a100() -> Self {
        Self::new()
            .require_cuda(1)
            .require_cuda_compute_cap(8, 0)
            .require_gpu_memory_gb(40)
    }

    /// Check if requirements can be satisfied by given resources
    pub fn can_satisfy(&self, available: &NodeResources) -> bool {
        for (resource, required) in &self.resources {
            let available_qty = available.resources.get(resource).copied().unwrap_or(0.0);
            if available_qty < *required {
                return false;
            }
        }
        true
    }

    /// Convert to HashMap for Ray resource specification
    pub fn to_ray_resources(&self) -> HashMap<String, f64> {
        self.resources.clone()
    }
}

/// Detected resources on a node
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeResources {
    /// Node identifier
    pub node_id: String,
    /// Node hostname
    pub hostname: String,
    /// Available resources
    pub resources: HashMap<String, f64>,
    /// Node labels
    pub labels: Vec<String>,
    /// Node architecture
    pub architecture: NodeArchitecture,
    /// Whether the node is healthy
    pub is_healthy: bool,
    /// Last health check timestamp (Unix epoch seconds)
    pub last_health_check: u64,
}

/// Node architecture type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NodeArchitecture {
    #[default]
    Unknown,
    AppleSilicon,
    NvidiaGpu,
    AmdGpu,
    CpuOnly,
}

impl NodeResources {
    /// Create new node resources with auto-detection
    pub fn detect() -> Self {
        let mut resources = HashMap::new();
        let mut labels = Vec::new();
        let architecture;

        // Detect CPU cores
        let cpu_cores = num_cpus::get();
        resources.insert("cpu_cores".to_string(), cpu_cores as f64);

        // Detect system memory
        if let Ok(mem_info) = sys_info::mem_info() {
            let memory_gb = mem_info.total as f64 / 1024.0 / 1024.0;
            resources.insert("memory_gb".to_string(), memory_gb);
        }

        // Detect GPU type
        #[cfg(target_os = "macos")]
        {
            // Check for Apple Silicon Metal support
            if Self::has_metal_support() {
                resources.insert("metal".to_string(), 1.0);
                labels.push("apple_silicon".to_string());
                architecture = NodeArchitecture::AppleSilicon;

                // Estimate unified memory as GPU memory
                if let Ok(mem_info) = sys_info::mem_info() {
                    let memory_gb = mem_info.total as f64 / 1024.0 / 1024.0;
                    // Apple Silicon uses unified memory
                    resources.insert("gpu_memory_gb".to_string(), memory_gb * 0.75);
                }
            } else {
                architecture = NodeArchitecture::CpuOnly;
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Check for NVIDIA CUDA
            if Self::has_cuda_support() {
                let (cuda_count, compute_cap, gpu_memory) = Self::detect_cuda_gpus();
                resources.insert("cuda".to_string(), cuda_count as f64);
                resources.insert("cuda_compute_cap".to_string(), compute_cap);
                resources.insert("gpu_memory_gb".to_string(), gpu_memory);
                
                if compute_cap >= 9.0 {
                    labels.push("h100".to_string());
                    resources.insert("has_hbm".to_string(), 1.0);
                } else if compute_cap >= 8.0 {
                    labels.push("a100".to_string());
                    resources.insert("has_hbm".to_string(), 1.0);
                }
                architecture = NodeArchitecture::NvidiaGpu;
            } else {
                architecture = NodeArchitecture::CpuOnly;
            }
        }

        // Get hostname
        let hostname = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        // Generate node ID
        let node_id = format!("{}-{}", hostname, uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("0000"));

        Self {
            node_id,
            hostname,
            resources,
            labels,
            architecture,
            is_healthy: true,
            last_health_check: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Check if Metal is supported (macOS)
    #[cfg(target_os = "macos")]
    fn has_metal_support() -> bool {
        // Use metal-rs or check for Metal device availability
        // For now, assume all macOS systems have Metal
        true
    }

    #[cfg(not(target_os = "macos"))]
    fn has_metal_support() -> bool {
        false
    }

    /// Check if CUDA is supported
    #[cfg(not(target_os = "macos"))]
    fn has_cuda_support() -> bool {
        // Check for NVIDIA driver
        std::path::Path::new("/dev/nvidia0").exists()
            || std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
    }

    #[cfg(target_os = "macos")]
    fn has_cuda_support() -> bool {
        false
    }

    /// Detect CUDA GPUs and their properties
    #[cfg(not(target_os = "macos"))]
    fn detect_cuda_gpus() -> (u32, f64, f64) {
        // Try to read from nvidia-smi or CUDA runtime
        // Default values for now
        let cuda_count = std::env::var("CUDA_VISIBLE_DEVICES")
            .map(|v| v.split(',').count() as u32)
            .unwrap_or(1);
        
        // Default to A100 specs
        (cuda_count, 8.0, 40.0)
    }

    #[cfg(target_os = "macos")]
    fn detect_cuda_gpus() -> (u32, f64, f64) {
        (0, 0.0, 0.0)
    }

    /// Check if this node can satisfy given requirements
    pub fn satisfies(&self, requirements: &ResourceRequirements) -> bool {
        if !self.is_healthy {
            return false;
        }
        requirements.can_satisfy(self)
    }

    /// Get resource value by type
    pub fn get_resource(&self, resource_type: &str) -> f64 {
        self.resources.get(resource_type).copied().unwrap_or(0.0)
    }
}

/// Placement strategy for distributed tasks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlacementStrategy {
    /// Spread tasks across different nodes
    Spread,
    /// Pack tasks onto as few nodes as possible
    Pack,
    /// Strict placement based on resource requirements
    Strict,
    /// Custom placement based on scoring function
    Custom {
        /// Scoring weights for different resource types
        weights: HashMap<String, f64>,
    },
}

/// Placement group for coordinated scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementGroup {
    /// Unique group identifier
    pub group_id: String,
    /// Name of the placement group
    pub name: String,
    /// Resource bundles for each slot in the group
    pub bundles: Vec<ResourceRequirements>,
    /// Placement strategy
    pub strategy: PlacementStrategy,
    /// Current assignments (slot index -> node ID)
    pub assignments: HashMap<usize, String>,
    /// Whether the group is fully scheduled
    pub is_ready: bool,
}

impl PlacementGroup {
    /// Create a new placement group
    pub fn new(name: &str, strategy: PlacementStrategy) -> Self {
        Self {
            group_id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            bundles: Vec::new(),
            strategy,
            assignments: HashMap::new(),
            is_ready: false,
        }
    }

    /// Add a resource bundle to the group
    pub fn add_bundle(&mut self, requirements: ResourceRequirements) {
        self.bundles.push(requirements);
    }

    /// Create a pipeline parallel placement group
    ///
    /// Creates a group with `pp_size` slots, where:
    /// - Slots can be assigned to different architectures
    /// - Each slot has the specified resource requirements
    pub fn pipeline_parallel(
        name: &str,
        pp_size: usize,
        requirements_per_rank: ResourceRequirements,
    ) -> Self {
        let mut group = Self::new(name, PlacementStrategy::Spread);
        for _ in 0..pp_size {
            group.add_bundle(requirements_per_rank.clone());
        }
        group
    }

    /// Create a heterogeneous pipeline parallel placement group
    ///
    /// Allows different resource requirements for each pipeline stage
    pub fn heterogeneous_pipeline(
        name: &str,
        requirements_per_stage: Vec<ResourceRequirements>,
    ) -> Self {
        let mut group = Self::new(name, PlacementStrategy::Strict);
        for req in requirements_per_stage {
            group.add_bundle(req);
        }
        group
    }

    /// Get the number of slots in this group
    pub fn num_slots(&self) -> usize {
        self.bundles.len()
    }

    /// Check if a slot is assigned
    pub fn is_slot_assigned(&self, slot: usize) -> bool {
        self.assignments.contains_key(&slot)
    }

    /// Assign a node to a slot
    pub fn assign(&mut self, slot: usize, node_id: &str) -> bool {
        if slot >= self.bundles.len() {
            return false;
        }
        self.assignments.insert(slot, node_id.to_string());
        self.is_ready = self.assignments.len() == self.bundles.len();
        true
    }

    /// Get the node assigned to a slot
    pub fn get_assignment(&self, slot: usize) -> Option<&str> {
        self.assignments.get(&slot).map(|s| s.as_str())
    }
}

/// Node health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHealthStatus {
    /// Node identifier
    pub node_id: String,
    /// Whether the node is reachable
    pub is_reachable: bool,
    /// CPU utilization (0.0 - 1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0 - 1.0)
    pub memory_utilization: f64,
    /// GPU utilization (0.0 - 1.0), if applicable
    pub gpu_utilization: Option<f64>,
    /// GPU memory utilization (0.0 - 1.0), if applicable
    pub gpu_memory_utilization: Option<f64>,
    /// Network status
    pub network_status: NetworkStatus,
    /// Last error message, if any
    pub last_error: Option<String>,
    /// Timestamp of this health check
    pub timestamp: u64,
}

/// Network status for a node
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkStatus {
    /// Whether the node can reach other nodes
    pub can_reach_peers: bool,
    /// Latency to coordinator in ms
    pub coordinator_latency_ms: Option<f64>,
    /// Average latency to peer nodes in ms
    pub avg_peer_latency_ms: Option<f64>,
}

/// Health monitor for heterogeneous clusters
pub struct ClusterHealthMonitor {
    /// Known nodes and their resources
    nodes: Arc<RwLock<HashMap<String, NodeResources>>>,
    /// Node health status
    health: Arc<RwLock<HashMap<String, NodeHealthStatus>>>,
    /// Health check interval in seconds
    check_interval_secs: u64,
    /// Failure threshold (consecutive failures before marking unhealthy)
    failure_threshold: u32,
    /// Per-node failure counts
    failure_counts: Arc<RwLock<HashMap<String, u32>>>,
}

impl ClusterHealthMonitor {
    /// Create a new health monitor
    pub fn new(check_interval_secs: u64, failure_threshold: u32) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            health: Arc::new(RwLock::new(HashMap::new())),
            check_interval_secs,
            failure_threshold,
            failure_counts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a node with the monitor
    pub fn register_node(&self, resources: NodeResources) {
        let node_id = resources.node_id.clone();
        if let Ok(mut nodes) = self.nodes.write() {
            nodes.insert(node_id.clone(), resources);
        }
        if let Ok(mut counts) = self.failure_counts.write() {
            counts.insert(node_id, 0);
        }
    }

    /// Update health status for a node
    pub fn update_health(&self, status: NodeHealthStatus) {
        let node_id = status.node_id.clone();
        let is_healthy = status.is_reachable;

        // Update health status
        if let Ok(mut health) = self.health.write() {
            health.insert(node_id.clone(), status);
        }

        // Update failure count and node health
        if let Ok(mut counts) = self.failure_counts.write() {
            let count = counts.entry(node_id.clone()).or_insert(0);
            if is_healthy {
                *count = 0;
            } else {
                *count += 1;
            }

            // Mark node unhealthy if threshold exceeded
            if *count >= self.failure_threshold {
                if let Ok(mut nodes) = self.nodes.write() {
                    if let Some(node) = nodes.get_mut(&node_id) {
                        node.is_healthy = false;
                    }
                }
            }
        }
    }

    /// Get all healthy nodes
    pub fn get_healthy_nodes(&self) -> Vec<NodeResources> {
        if let Ok(nodes) = self.nodes.read() {
            nodes.values().filter(|n| n.is_healthy).cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Get nodes that can satisfy given requirements
    pub fn find_matching_nodes(&self, requirements: &ResourceRequirements) -> Vec<NodeResources> {
        self.get_healthy_nodes()
            .into_iter()
            .filter(|node| node.satisfies(requirements))
            .collect()
    }

    /// Get current health status for a node
    pub fn get_node_health(&self, node_id: &str) -> Option<NodeHealthStatus> {
        if let Ok(health) = self.health.read() {
            health.get(node_id).cloned()
        } else {
            None
        }
    }

    /// Get summary of cluster health
    pub fn get_cluster_summary(&self) -> ClusterHealthSummary {
        let nodes = if let Ok(nodes) = self.nodes.read() {
            nodes.clone()
        } else {
            HashMap::new()
        };

        let total_nodes = nodes.len();
        let healthy_nodes = nodes.values().filter(|n| n.is_healthy).count();

        let mut total_resources: HashMap<String, f64> = HashMap::new();
        let mut available_resources: HashMap<String, f64> = HashMap::new();

        for node in nodes.values() {
            for (key, value) in &node.resources {
                *total_resources.entry(key.clone()).or_insert(0.0) += value;
                if node.is_healthy {
                    *available_resources.entry(key.clone()).or_insert(0.0) += value;
                }
            }
        }

        // Count by architecture
        let mut architecture_counts: HashMap<String, usize> = HashMap::new();
        for node in nodes.values() {
            let arch_name = format!("{:?}", node.architecture);
            *architecture_counts.entry(arch_name).or_insert(0) += 1;
        }

        ClusterHealthSummary {
            total_nodes,
            healthy_nodes,
            unhealthy_nodes: total_nodes - healthy_nodes,
            total_resources,
            available_resources,
            architecture_counts,
        }
    }
}

/// Summary of cluster health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterHealthSummary {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of healthy nodes
    pub healthy_nodes: usize,
    /// Number of unhealthy nodes
    pub unhealthy_nodes: usize,
    /// Total resources across all nodes
    pub total_resources: HashMap<String, f64>,
    /// Available resources (healthy nodes only)
    pub available_resources: HashMap<String, f64>,
    /// Count of nodes by architecture
    pub architecture_counts: HashMap<String, usize>,
}

/// Heterogeneous scheduler for placing tasks on appropriate nodes
pub struct HeterogeneousScheduler {
    /// Health monitor for tracking node status
    health_monitor: Arc<ClusterHealthMonitor>,
    /// Active placement groups
    placement_groups: Arc<RwLock<HashMap<String, PlacementGroup>>>,
}

impl HeterogeneousScheduler {
    /// Create a new scheduler with the given health monitor
    pub fn new(health_monitor: Arc<ClusterHealthMonitor>) -> Self {
        Self {
            health_monitor,
            placement_groups: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create and register a placement group
    pub fn create_placement_group(&self, group: PlacementGroup) -> String {
        let group_id = group.group_id.clone();
        if let Ok(mut groups) = self.placement_groups.write() {
            groups.insert(group_id.clone(), group);
        }
        group_id
    }

    /// Schedule a placement group across available nodes
    pub fn schedule_placement_group(&self, group_id: &str) -> Result<(), SchedulingError> {
        let mut group = if let Ok(groups) = self.placement_groups.read() {
            groups.get(group_id).cloned()
        } else {
            None
        }.ok_or(SchedulingError::GroupNotFound)?;

        // Get healthy nodes
        let available_nodes = self.health_monitor.get_healthy_nodes();
        if available_nodes.is_empty() {
            return Err(SchedulingError::NoHealthyNodes);
        }

        // Try to assign each bundle
        let mut used_nodes: Vec<String> = Vec::new();
        let mut assignments: Vec<(usize, String)> = Vec::new();
        
        let bundles_clone = group.bundles.clone();
        let strategy = group.strategy.clone();
        
        for (slot_idx, requirements) in bundles_clone.iter().enumerate() {
            let matching_nodes: Vec<_> = available_nodes
                .iter()
                .filter(|n| {
                    n.satisfies(requirements) && 
                    // For Spread strategy, avoid reusing nodes
                    (matches!(strategy, PlacementStrategy::Pack) || 
                     !used_nodes.contains(&n.node_id))
                })
                .collect();

            if matching_nodes.is_empty() {
                return Err(SchedulingError::InsufficientResources {
                    slot: slot_idx,
                    requirements: requirements.clone(),
                });
            }

            // Select best node based on strategy
            let selected_node = match &strategy {
                PlacementStrategy::Spread => {
                    // Pick node with most resources
                    matching_nodes.into_iter()
                        .max_by(|a, b| {
                            let a_score: f64 = a.resources.values().sum();
                            let b_score: f64 = b.resources.values().sum();
                            a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
                        })
                }
                PlacementStrategy::Pack => {
                    // Pick node with least resources (pack tightly)
                    matching_nodes.into_iter()
                        .min_by(|a, b| {
                            let a_score: f64 = a.resources.values().sum();
                            let b_score: f64 = b.resources.values().sum();
                            a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
                        })
                }
                PlacementStrategy::Strict | PlacementStrategy::Custom { .. } => {
                    // Just pick first matching node
                    matching_nodes.into_iter().next()
                }
            };

            if let Some(node) = selected_node {
                assignments.push((slot_idx, node.node_id.clone()));
                used_nodes.push(node.node_id.clone());
            } else {
                return Err(SchedulingError::InsufficientResources {
                    slot: slot_idx,
                    requirements: requirements.clone(),
                });
            }
        }

        // Apply assignments to the group
        for (slot_idx, node_id) in assignments {
            group.assign(slot_idx, &node_id);
        }

        // Update stored group
        if let Ok(mut groups) = self.placement_groups.write() {
            groups.insert(group_id.to_string(), group);
        }

        Ok(())
    }

    /// Get a placement group by ID
    pub fn get_placement_group(&self, group_id: &str) -> Option<PlacementGroup> {
        if let Ok(groups) = self.placement_groups.read() {
            groups.get(group_id).cloned()
        } else {
            None
        }
    }

    /// Find best node for given requirements
    pub fn find_best_node(&self, requirements: &ResourceRequirements) -> Option<NodeResources> {
        self.health_monitor
            .find_matching_nodes(requirements)
            .into_iter()
            .max_by(|a, b| {
                let a_score: f64 = a.resources.values().sum();
                let b_score: f64 = b.resources.values().sum();
                a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// Scheduling errors
#[derive(Debug, Clone)]
pub enum SchedulingError {
    /// Placement group not found
    GroupNotFound,
    /// No healthy nodes available
    NoHealthyNodes,
    /// Insufficient resources for a slot
    InsufficientResources {
        slot: usize,
        requirements: ResourceRequirements,
    },
    /// Node not found
    NodeNotFound(String),
}

impl std::fmt::Display for SchedulingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulingError::GroupNotFound => write!(f, "Placement group not found"),
            SchedulingError::NoHealthyNodes => write!(f, "No healthy nodes available"),
            SchedulingError::InsufficientResources { slot, requirements: _ } => {
                write!(f, "Insufficient resources for slot {}", slot)
            }
            SchedulingError::NodeNotFound(id) => write!(f, "Node not found: {}", id),
        }
    }
}

impl std::error::Error for SchedulingError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_requirements_apple_silicon() {
        let req = ResourceRequirements::apple_silicon(36);
        assert!(req.resources.contains_key("metal"));
        assert!(req.resources.contains_key("memory_gb"));
        assert_eq!(req.resources.get("metal"), Some(&1.0));
        assert_eq!(req.resources.get("memory_gb"), Some(&36.0));
    }

    #[test]
    fn test_resource_requirements_nvidia_h100() {
        let req = ResourceRequirements::nvidia_h100();
        assert!(req.resources.contains_key("cuda"));
        assert!(req.resources.contains_key("cuda_compute_cap"));
        assert!(req.resources.contains_key("gpu_memory_gb"));
        assert_eq!(req.resources.get("cuda"), Some(&1.0));
        assert_eq!(req.resources.get("cuda_compute_cap"), Some(&9.0));
        assert_eq!(req.resources.get("gpu_memory_gb"), Some(&80.0));
    }

    #[test]
    fn test_resource_requirements_can_satisfy() {
        let req = ResourceRequirements::new()
            .require_metal(1)
            .require_memory_gb(32);

        // Node with sufficient resources
        let mut node = NodeResources::default();
        node.resources.insert("metal".to_string(), 1.0);
        node.resources.insert("memory_gb".to_string(), 64.0);
        node.is_healthy = true;
        assert!(req.can_satisfy(&node));

        // Node with insufficient memory
        node.resources.insert("memory_gb".to_string(), 16.0);
        assert!(!req.can_satisfy(&node));
    }

    #[test]
    fn test_placement_group_pipeline_parallel() {
        let req = ResourceRequirements::apple_silicon(36);
        let group = PlacementGroup::pipeline_parallel("pp_test", 3, req);
        
        assert_eq!(group.num_slots(), 3);
        assert_eq!(group.strategy, PlacementStrategy::Spread);
        assert!(!group.is_ready);
    }

    #[test]
    fn test_placement_group_heterogeneous() {
        let requirements = vec![
            ResourceRequirements::apple_silicon(36),
            ResourceRequirements::nvidia_h100(),
            ResourceRequirements::apple_silicon(36),
        ];
        let group = PlacementGroup::heterogeneous_pipeline("hetero_test", requirements);
        
        assert_eq!(group.num_slots(), 3);
        assert_eq!(group.strategy, PlacementStrategy::Strict);
    }

    #[test]
    fn test_placement_group_assignment() {
        let req = ResourceRequirements::new();
        let mut group = PlacementGroup::pipeline_parallel("assign_test", 2, req);
        
        assert!(!group.is_ready);
        assert!(!group.is_slot_assigned(0));
        
        group.assign(0, "node-1");
        assert!(group.is_slot_assigned(0));
        assert_eq!(group.get_assignment(0), Some("node-1"));
        assert!(!group.is_ready);
        
        group.assign(1, "node-2");
        assert!(group.is_ready);
    }

    #[test]
    fn test_cluster_health_monitor() {
        let monitor = ClusterHealthMonitor::new(30, 3);
        
        // Register nodes
        let mut node1 = NodeResources::default();
        node1.node_id = "node-1".to_string();
        node1.resources.insert("metal".to_string(), 1.0);
        node1.resources.insert("memory_gb".to_string(), 64.0);
        node1.is_healthy = true;
        
        let mut node2 = NodeResources::default();
        node2.node_id = "node-2".to_string();
        node2.resources.insert("cuda".to_string(), 1.0);
        node2.resources.insert("gpu_memory_gb".to_string(), 80.0);
        node2.is_healthy = true;
        
        monitor.register_node(node1);
        monitor.register_node(node2);
        
        let healthy = monitor.get_healthy_nodes();
        assert_eq!(healthy.len(), 2);
        
        // Find nodes matching Metal requirements
        let req = ResourceRequirements::new().require_metal(1);
        let matching = monitor.find_matching_nodes(&req);
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].node_id, "node-1");
    }

    #[test]
    fn test_cluster_health_summary() {
        let monitor = ClusterHealthMonitor::new(30, 3);
        
        let mut node1 = NodeResources::default();
        node1.node_id = "node-1".to_string();
        node1.architecture = NodeArchitecture::AppleSilicon;
        node1.resources.insert("metal".to_string(), 1.0);
        node1.is_healthy = true;
        
        let mut node2 = NodeResources::default();
        node2.node_id = "node-2".to_string();
        node2.architecture = NodeArchitecture::NvidiaGpu;
        node2.resources.insert("cuda".to_string(), 1.0);
        node2.is_healthy = false;
        
        monitor.register_node(node1);
        monitor.register_node(node2);
        
        let summary = monitor.get_cluster_summary();
        assert_eq!(summary.total_nodes, 2);
        assert_eq!(summary.healthy_nodes, 1);
        assert_eq!(summary.unhealthy_nodes, 1);
    }

    #[test]
    fn test_heterogeneous_scheduler() {
        let monitor = Arc::new(ClusterHealthMonitor::new(30, 3));
        
        // Register nodes
        let mut node1 = NodeResources::default();
        node1.node_id = "apple-1".to_string();
        node1.resources.insert("metal".to_string(), 1.0);
        node1.resources.insert("memory_gb".to_string(), 64.0);
        node1.is_healthy = true;
        
        let mut node2 = NodeResources::default();
        node2.node_id = "nvidia-1".to_string();
        node2.resources.insert("cuda".to_string(), 1.0);
        node2.resources.insert("cuda_compute_cap".to_string(), 9.0);
        node2.resources.insert("gpu_memory_gb".to_string(), 80.0);
        node2.is_healthy = true;
        
        monitor.register_node(node1);
        monitor.register_node(node2);
        
        let scheduler = HeterogeneousScheduler::new(monitor);
        
        // Create heterogeneous placement group
        let requirements = vec![
            ResourceRequirements::apple_silicon(32),
            ResourceRequirements::nvidia_h100(),
        ];
        let group = PlacementGroup::heterogeneous_pipeline("hetero", requirements);
        let group_id = scheduler.create_placement_group(group);
        
        // Schedule the group
        let result = scheduler.schedule_placement_group(&group_id);
        assert!(result.is_ok());
        
        // Verify assignments
        let scheduled_group = scheduler.get_placement_group(&group_id).unwrap();
        assert!(scheduled_group.is_ready);
        assert_eq!(scheduled_group.get_assignment(0), Some("apple-1"));
        assert_eq!(scheduled_group.get_assignment(1), Some("nvidia-1"));
    }

    #[test]
    fn test_scheduler_insufficient_resources() {
        let monitor = Arc::new(ClusterHealthMonitor::new(30, 3));
        
        // Register only Apple Silicon node
        let mut node1 = NodeResources::default();
        node1.node_id = "apple-1".to_string();
        node1.resources.insert("metal".to_string(), 1.0);
        node1.is_healthy = true;
        
        monitor.register_node(node1);
        
        let scheduler = HeterogeneousScheduler::new(monitor);
        
        // Try to schedule group that needs H100
        let requirements = vec![
            ResourceRequirements::nvidia_h100(),
        ];
        let group = PlacementGroup::heterogeneous_pipeline("need_h100", requirements);
        let group_id = scheduler.create_placement_group(group);
        
        let result = scheduler.schedule_placement_group(&group_id);
        assert!(matches!(result, Err(SchedulingError::InsufficientResources { .. })));
    }
}
