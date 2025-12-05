//! Distributed process group management.
//!
//! Manages global process groups for different parallelism strategies:
//! - Data Parallel (DP) group
//! - Tensor Parallel (TP) group  
//! - Pipeline Parallel (PP) group
//! - Expert Parallel (EP) group
//! - Sequence Parallel (SP) group (shares with TP)

use crate::utils::error::{DeepSeekError, Result};
use super::CollectiveCommunicator;
use super::backend::LocalCommunicator;
use std::sync::{Arc, OnceLock, RwLock};
use tracing::{info, warn};

/// Global distributed state
static DISTRIBUTED_STATE: OnceLock<RwLock<DistributedState>> = OnceLock::new();

/// Process group handle
#[derive(Clone)]
pub struct ProcessGroup {
    /// Communicator for this group
    pub communicator: Arc<dyn CollectiveCommunicator>,
    /// Ranks in this group (global rank -> local rank mapping)
    pub ranks: Vec<usize>,
    /// Size of this group
    pub size: usize,
}

impl ProcessGroup {
    pub fn new(communicator: Arc<dyn CollectiveCommunicator>, ranks: Vec<usize>) -> Self {
        let size = ranks.len();
        Self { communicator, ranks, size }
    }
    
    /// Get local rank within this group
    pub fn local_rank(&self, global_rank: usize) -> Option<usize> {
        self.ranks.iter().position(|&r| r == global_rank)
    }
}

/// Parallelism configuration
#[derive(Debug, Clone)]
pub struct ParallelismConfig {
    pub world_size: usize,
    pub dp_size: usize,
    pub tp_size: usize,
    pub pp_size: usize,
    pub ep_size: usize,
    pub sp_size: usize,
}

impl ParallelismConfig {
    pub fn new(world_size: usize, dp_size: usize, tp_size: usize, pp_size: usize, ep_size: usize, sp_size: usize) -> Result<Self> {
        // Validate dimensions
        let computed = dp_size * tp_size * pp_size;
        if computed != world_size {
            return Err(DeepSeekError::Distributed(format!(
                "DP({}) x TP({}) x PP({}) = {} != world_size({})",
                dp_size, tp_size, pp_size, computed, world_size
            )));
        }
        
        Ok(Self {
            world_size,
            dp_size,
            tp_size,
            pp_size,
            ep_size,
            sp_size,
        })
    }
    
    pub fn single_device() -> Self {
        Self {
            world_size: 1,
            dp_size: 1,
            tp_size: 1,
            pp_size: 1,
            ep_size: 1,
            sp_size: 1,
        }
    }
}

/// Global distributed state holding all process groups
pub struct DistributedState {
    pub initialized: bool,
    pub rank: usize,
    pub world_size: usize,
    pub config: ParallelismConfig,
    
    // Process groups
    pub world_group: Option<ProcessGroup>,
    pub dp_group: Option<ProcessGroup>,
    pub tp_group: Option<ProcessGroup>,
    pub pp_group: Option<ProcessGroup>,
    pub ep_group: Option<ProcessGroup>,
    pub sp_group: Option<ProcessGroup>,
    
    // Backend type
    pub backend: String,
}

impl Default for DistributedState {
    fn default() -> Self {
        Self {
            initialized: false,
            rank: 0,
            world_size: 1,
            config: ParallelismConfig::single_device(),
            world_group: None,
            dp_group: None,
            tp_group: None,
            pp_group: None,
            ep_group: None,
            sp_group: None,
            backend: "local".to_string(),
        }
    }
}

/// Initialize distributed training.
///
/// This sets up all process groups based on the parallelism configuration.
/// Must be called before any distributed operations.
pub fn initialize_distributed(
    rank: usize,
    world_size: usize,
    config: ParallelismConfig,
    backend: &str,
) -> Result<()> {
    let state = DISTRIBUTED_STATE.get_or_init(|| RwLock::new(DistributedState::default()));
    let mut state = state.write().map_err(|e| DeepSeekError::Distributed(e.to_string()))?;
    
    if state.initialized {
        warn!("Distributed already initialized, skipping");
        return Ok(());
    }
    
    info!(
        rank = rank,
        world_size = world_size,
        dp = config.dp_size,
        tp = config.tp_size,
        pp = config.pp_size,
        ep = config.ep_size,
        sp = config.sp_size,
        backend = backend,
        "Initializing distributed training"
    );
    
    state.rank = rank;
    state.world_size = world_size;
    state.config = config.clone();
    state.backend = backend.to_string();
    
    // Create process groups based on backend
    match backend {
        "local" => {
            // For local testing, create simulated groups
            initialize_local_groups(&mut state, &config)?;
        }
        "nccl" => {
            // NCCL backend - requires actual GPU cluster
            #[cfg(feature = "cuda")]
            {
                initialize_nccl_groups(&mut state, &config)?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(DeepSeekError::Distributed(
                    "NCCL backend requires CUDA feature".to_string()
                ));
            }
        }
        "gloo" => {
            return Err(DeepSeekError::NotImplemented(
                "Gloo backend not yet implemented".to_string()
            ));
        }
        _ => {
            return Err(DeepSeekError::Distributed(
                format!("Unknown backend: {}", backend)
            ));
        }
    }
    
    state.initialized = true;
    Ok(())
}

/// Initialize NCCL groups (requires CUDA)
#[cfg(feature = "cuda")]
fn initialize_nccl_groups(state: &mut DistributedState, config: &ParallelismConfig) -> Result<()> {
    use super::nccl_backend::create_nccl_communicators;
    use super::nccl_sys::NcclUniqueId;
    use candle_core::Device;
    use std::env;

    let rank = state.rank;
    let world_size = state.world_size;
    
    // Get unique ID from env var (assumed to be set by launcher)
    // In a real implementation, we would need a proper store (TCP/File)
    let unique_id_str = env::var("DEEPSEEK_NCCL_ID").unwrap_or_default();
    let unique_id = if unique_id_str.is_empty() {
        if rank == 0 {
            warn!("DEEPSEEK_NCCL_ID not set, generating new one (only works for single process)");
            super::NcclCommunicator::generate_unique_id()
                .map_err(|e| DeepSeekError::Distributed(e))?
        } else {
            return Err(DeepSeekError::Distributed(
                "DEEPSEEK_NCCL_ID not set and rank > 0".to_string()
            ));
        }
    } else {
        // Parse hex string to NcclUniqueId
        // This is a placeholder - actual parsing would be needed
        // For now, we assume single node/process for testing if ID not provided
        warn!("Parsing NCCL ID from string not fully implemented");
        NcclUniqueId::default() 
    };

    let device = Device::new_cuda(rank)?;
    let comm = create_nccl_communicators(rank, world_size, unique_id, device)
        .map_err(|e| DeepSeekError::Distributed(e))?;
        
    state.world_group = Some(ProcessGroup::new(comm.clone(), (0..world_size).collect()));
    
    // Helper to create subgroup
    // Since we lack ncclCommSplit, we only support:
    // 1. Group size == World size (reuse world comm)
    // 2. Group size == 1 (create local/dummy comm? or just reuse world if rank matches?)
    //    Actually, for size 1, we don't need NCCL, we can use LocalCommunicator?
    //    But we need consistent types.
    //    We'll use world comm but check ranks? No, collective ops would hang.
    
    // Strategy:
    // If group_size == world_size, use world_group.
    // If group_size == 1, use a LocalCommunicator (it implements CollectiveCommunicator).
    // If 1 < group_size < world_size, error out (not supported without split).
    
    let create_group = |size: usize, name: &str| -> Result<Option<ProcessGroup>> {
        if size == world_size {
            Ok(state.world_group.clone())
        } else if size == 1 {
            // Create a local communicator for this single rank
            let comms = LocalCommunicator::new_group(1);
            let comm = Arc::new(comms.into_iter().next().unwrap());
            Ok(Some(ProcessGroup::new(comm, vec![rank])))
        } else {
            Err(DeepSeekError::Distributed(format!(
                "{} size {} != world size {} not supported (ncclCommSplit missing)",
                name, size, world_size
            )))
        }
    };
    
    state.dp_group = create_group(config.dp_size, "DP")?;
    state.tp_group = create_group(config.tp_size, "TP")?;
    state.pp_group = create_group(config.pp_size, "PP")?;
    state.ep_group = create_group(config.ep_size, "EP")?;
    state.sp_group = create_group(config.sp_size, "SP")?;
    
    Ok(())
}

/// Initialize local (in-process) groups for testing
fn initialize_local_groups(state: &mut DistributedState, config: &ParallelismConfig) -> Result<()> {
    // For local testing with world_size=1, create single-rank groups
    if config.world_size == 1 {
        let comms = LocalCommunicator::new_group(1);
        let comm = Arc::new(comms.into_iter().next().unwrap());
        
        state.world_group = Some(ProcessGroup::new(comm.clone(), vec![0]));
        state.dp_group = Some(ProcessGroup::new(comm.clone(), vec![0]));
        state.tp_group = Some(ProcessGroup::new(comm.clone(), vec![0]));
        state.pp_group = Some(ProcessGroup::new(comm.clone(), vec![0]));
        state.ep_group = Some(ProcessGroup::new(comm.clone(), vec![0]));
        state.sp_group = Some(ProcessGroup::new(comm.clone(), vec![0]));
        
        return Ok(());
    }
    
    // For multi-rank local testing, we'd need thread-based simulation
    // This is primarily for unit tests
    Err(DeepSeekError::NotImplemented(
        "Multi-rank local groups require thread-based simulation".to_string()
    ))
}

/// Check if distributed is initialized
pub fn is_initialized() -> bool {
    DISTRIBUTED_STATE
        .get()
        .and_then(|s| s.read().ok())
        .map(|s| s.initialized)
        .unwrap_or(false)
}

/// Get current rank
pub fn get_rank() -> usize {
    DISTRIBUTED_STATE
        .get()
        .and_then(|s| s.read().ok())
        .map(|s| s.rank)
        .unwrap_or(0)
}

/// Get world size
pub fn get_world_size() -> usize {
    DISTRIBUTED_STATE
        .get()
        .and_then(|s| s.read().ok())
        .map(|s| s.world_size)
        .unwrap_or(1)
}

/// Get data parallel group
pub fn get_dp_group() -> Option<ProcessGroup> {
    DISTRIBUTED_STATE
        .get()
        .and_then(|s| s.read().ok())
        .and_then(|s| s.dp_group.clone())
}

/// Get tensor parallel group
pub fn get_tp_group() -> Option<ProcessGroup> {
    DISTRIBUTED_STATE
        .get()
        .and_then(|s| s.read().ok())
        .and_then(|s| s.tp_group.clone())
}

/// Get pipeline parallel group
pub fn get_pp_group() -> Option<ProcessGroup> {
    DISTRIBUTED_STATE
        .get()
        .and_then(|s| s.read().ok())
        .and_then(|s| s.pp_group.clone())
}

/// Get expert parallel group
pub fn get_ep_group() -> Option<ProcessGroup> {
    DISTRIBUTED_STATE
        .get()
        .and_then(|s| s.read().ok())
        .and_then(|s| s.ep_group.clone())
}

/// Get tensor parallel size
pub fn get_tp_size() -> usize {
    DISTRIBUTED_STATE
        .get()
        .and_then(|s| s.read().ok())
        .map(|s| s.config.tp_size)
        .unwrap_or(1)
}

/// Get tensor parallel rank
pub fn get_tp_rank() -> usize {
    let global_rank = get_rank();
    get_tp_group()
        .and_then(|g| g.local_rank(global_rank))
        .unwrap_or(0)
}

/// Get data parallel size
pub fn get_dp_size() -> usize {
    DISTRIBUTED_STATE
        .get()
        .and_then(|s| s.read().ok())
        .map(|s| s.config.dp_size)
        .unwrap_or(1)
}

/// Get expert parallel size
pub fn get_ep_size() -> usize {
    DISTRIBUTED_STATE
        .get()
        .and_then(|s| s.read().ok())
        .map(|s| s.config.ep_size)
        .unwrap_or(1)
}

/// Get pipeline parallel size
pub fn get_pp_size() -> usize {
    DISTRIBUTED_STATE
        .get()
        .and_then(|s| s.read().ok())
        .map(|s| s.config.pp_size)
        .unwrap_or(1)
}

/// Get pipeline parallel rank
pub fn get_pp_rank() -> usize {
    let global_rank = get_rank();
    get_pp_group()
        .and_then(|g| g.local_rank(global_rank))
        .unwrap_or(0)
}

/// Get sequence parallel size (same as TP size, since SP shares TP group)
pub fn get_sp_size() -> usize {
    DISTRIBUTED_STATE
        .get()
        .and_then(|s| s.read().ok())
        .map(|s| s.config.sp_size)
        .unwrap_or(1)
}

/// Get sequence parallel rank (same as TP rank)
pub fn get_sp_rank() -> usize {
    let global_rank = get_rank();
    get_sp_group()
        .and_then(|g| g.local_rank(global_rank))
        .unwrap_or(0)
}

/// Get sequence parallel group (same as TP group)
pub fn get_sp_group() -> Option<ProcessGroup> {
    DISTRIBUTED_STATE
        .get()
        .and_then(|s| s.read().ok())
        .and_then(|s| s.sp_group.clone())
}

/// Cleanup distributed state
pub fn cleanup_distributed() {
    if let Some(state) = DISTRIBUTED_STATE.get() {
        if let Ok(mut s) = state.write() {
            s.initialized = false;
            s.world_group = None;
            s.dp_group = None;
            s.tp_group = None;
            s.pp_group = None;
            s.ep_group = None;
            info!("Distributed state cleaned up");
        }
    }
}

/// Barrier synchronization across all ranks
pub fn barrier() -> Result<()> {
    // For now, this is a no-op in local mode
    // In NCCL mode, this would call ncclGroupEnd or similar
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_single_device_init() -> Result<()> {
        let config = ParallelismConfig::single_device();
        initialize_distributed(0, 1, config, "local")?;
        
        assert!(is_initialized());
        assert_eq!(get_rank(), 0);
        assert_eq!(get_world_size(), 1);
        assert_eq!(get_tp_size(), 1);
        
        cleanup_distributed();
        Ok(())
    }
    
    #[test]
    fn test_parallelism_config_validation() {
        // Valid config
        let config = ParallelismConfig::new(8, 2, 2, 2, 1, 1);
        assert!(config.is_ok());
        
        // Invalid config (dimensions don't match)
        let config = ParallelismConfig::new(8, 2, 2, 4, 1, 1);
        assert!(config.is_err());
    }
}
