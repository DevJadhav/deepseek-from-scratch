//! NCCL FFI bindings for distributed GPU communication.
//!
//! This module provides low-level bindings to NVIDIA's NCCL library.
//! When the `cuda` feature is enabled, it uses real NCCL.
//! Otherwise, it provides a simulation layer for testing.

use std::ffi::c_void;

/// NCCL result codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclResult {
    Success = 0,
    UnhandledCudaError = 1,
    SystemError = 2,
    InternalError = 3,
    InvalidArgument = 4,
    InvalidUsage = 5,
    RemoteError = 6,
    InProgress = 7,
    NumResults = 8,
}

/// NCCL data types (matches ncclDataType_t)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclDataType {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    Bfloat16 = 9,
    NumTypes = 10,
}

impl NcclDataType {
    /// Alias: Char is the same as Int8
    pub const CHAR: Self = Self::Int8;
    /// Alias: Int is the same as Int32
    pub const INT: Self = Self::Int32;
    /// Alias: Half is the same as Float16
    pub const HALF: Self = Self::Float16;
    /// Alias: Float is the same as Float32
    pub const FLOAT: Self = Self::Float32;
    /// Alias: Double is the same as Float64
    pub const DOUBLE: Self = Self::Float64;
}

/// NCCL reduction operations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclRedOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
    NumOps = 5,
}

/// Opaque NCCL communicator handle
#[repr(C)]
pub struct NcclComm {
    _private: [u8; 0],
}

/// NCCL unique ID for initialization
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NcclUniqueId {
    pub internal: [u8; 128],
}

impl Default for NcclUniqueId {
    fn default() -> Self {
        Self { internal: [0u8; 128] }
    }
}

/// CUDA stream handle (opaque pointer)
pub type CudaStream = *mut c_void;

// FFI declarations for NCCL functions
#[cfg(feature = "cuda")]
mod ffi {
    use super::*;
    
    #[link(name = "nccl")]
    extern "C" {
        pub fn ncclGetUniqueId(uniqueId: *mut NcclUniqueId) -> NcclResult;
        
        pub fn ncclCommInitRank(
            comm: *mut *mut NcclComm,
            nranks: i32,
            commId: NcclUniqueId,
            rank: i32,
        ) -> NcclResult;
        
        pub fn ncclCommDestroy(comm: *mut NcclComm) -> NcclResult;
        
        pub fn ncclCommCount(comm: *mut NcclComm, count: *mut i32) -> NcclResult;
        
        pub fn ncclCommUserRank(comm: *mut NcclComm, rank: *mut i32) -> NcclResult;
        
        pub fn ncclAllReduce(
            sendbuff: *const c_void,
            recvbuff: *mut c_void,
            count: usize,
            datatype: NcclDataType,
            op: NcclRedOp,
            comm: *mut NcclComm,
            stream: CudaStream,
        ) -> NcclResult;
        
        pub fn ncclBroadcast(
            sendbuff: *const c_void,
            recvbuff: *mut c_void,
            count: usize,
            datatype: NcclDataType,
            root: i32,
            comm: *mut NcclComm,
            stream: CudaStream,
        ) -> NcclResult;
        
        pub fn ncclReduce(
            sendbuff: *const c_void,
            recvbuff: *mut c_void,
            count: usize,
            datatype: NcclDataType,
            op: NcclRedOp,
            root: i32,
            comm: *mut NcclComm,
            stream: CudaStream,
        ) -> NcclResult;
        
        pub fn ncclAllGather(
            sendbuff: *const c_void,
            recvbuff: *mut c_void,
            sendcount: usize,
            datatype: NcclDataType,
            comm: *mut NcclComm,
            stream: CudaStream,
        ) -> NcclResult;
        
        pub fn ncclReduceScatter(
            sendbuff: *const c_void,
            recvbuff: *mut c_void,
            recvcount: usize,
            datatype: NcclDataType,
            op: NcclRedOp,
            comm: *mut NcclComm,
            stream: CudaStream,
        ) -> NcclResult;
        
        pub fn ncclSend(
            sendbuff: *const c_void,
            count: usize,
            datatype: NcclDataType,
            peer: i32,
            comm: *mut NcclComm,
            stream: CudaStream,
        ) -> NcclResult;
        
        pub fn ncclRecv(
            recvbuff: *mut c_void,
            count: usize,
            datatype: NcclDataType,
            peer: i32,
            comm: *mut NcclComm,
            stream: CudaStream,
        ) -> NcclResult;
        
        pub fn ncclGroupStart() -> NcclResult;
        
        pub fn ncclGroupEnd() -> NcclResult;
        
        pub fn ncclGetErrorString(result: NcclResult) -> *const i8;
        
        pub fn ncclGetVersion(version: *mut i32) -> NcclResult;
    }
}

// Re-export FFI functions when CUDA is available
#[cfg(feature = "cuda")]
pub use ffi::*;

// Simulation layer for testing without CUDA
#[cfg(not(feature = "cuda"))]
pub mod simulation {
    use super::*;
    use std::ptr;
    
    /// Simulated NCCL communicator state
    pub struct SimulatedComm {
        pub rank: i32,
        pub size: i32,
    }
    
    thread_local! {
        static SIMULATED_COMMS: std::cell::RefCell<Vec<Box<SimulatedComm>>> = 
            std::cell::RefCell::new(Vec::new());
    }
    
    pub unsafe fn ncclGetUniqueId(unique_id: *mut NcclUniqueId) -> NcclResult {
        if unique_id.is_null() {
            return NcclResult::InvalidArgument;
        }
        // Generate a fake unique ID
        (*unique_id).internal = [0u8; 128];
        (*unique_id).internal[0] = 0xDE;
        (*unique_id).internal[1] = 0xAD;
        (*unique_id).internal[2] = 0xBE;
        (*unique_id).internal[3] = 0xEF;
        NcclResult::Success
    }
    
    pub unsafe fn ncclCommInitRank(
        comm: *mut *mut NcclComm,
        nranks: i32,
        _comm_id: NcclUniqueId,
        rank: i32,
    ) -> NcclResult {
        if comm.is_null() || rank < 0 || rank >= nranks {
            return NcclResult::InvalidArgument;
        }
        
        let sim_comm = Box::new(SimulatedComm {
            rank,
            size: nranks,
        });
        
        // Store and return pointer
        let ptr = Box::into_raw(sim_comm) as *mut NcclComm;
        *comm = ptr;
        
        NcclResult::Success
    }
    
    pub unsafe fn ncclCommDestroy(comm: *mut NcclComm) -> NcclResult {
        if comm.is_null() {
            return NcclResult::InvalidArgument;
        }
        // Free the simulated comm
        let _ = Box::from_raw(comm as *mut SimulatedComm);
        NcclResult::Success
    }
    
    pub unsafe fn ncclCommCount(comm: *mut NcclComm, count: *mut i32) -> NcclResult {
        if comm.is_null() || count.is_null() {
            return NcclResult::InvalidArgument;
        }
        let sim = &*(comm as *const SimulatedComm);
        *count = sim.size;
        NcclResult::Success
    }
    
    pub unsafe fn ncclCommUserRank(comm: *mut NcclComm, rank: *mut i32) -> NcclResult {
        if comm.is_null() || rank.is_null() {
            return NcclResult::InvalidArgument;
        }
        let sim = &*(comm as *const SimulatedComm);
        *rank = sim.rank;
        NcclResult::Success
    }
    
    pub unsafe fn ncclAllReduce(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        _op: NcclRedOp,
        _comm: *mut NcclComm,
        _stream: CudaStream,
    ) -> NcclResult {
        if sendbuff.is_null() || recvbuff.is_null() {
            return NcclResult::InvalidArgument;
        }
        // Simulation: just copy data (single rank behavior)
        let size = count * dtype_size(datatype);
        ptr::copy_nonoverlapping(sendbuff as *const u8, recvbuff as *mut u8, size);
        NcclResult::Success
    }
    
    pub unsafe fn ncclBroadcast(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        _root: i32,
        _comm: *mut NcclComm,
        _stream: CudaStream,
    ) -> NcclResult {
        if sendbuff.is_null() || recvbuff.is_null() {
            return NcclResult::InvalidArgument;
        }
        let size = count * dtype_size(datatype);
        ptr::copy_nonoverlapping(sendbuff as *const u8, recvbuff as *mut u8, size);
        NcclResult::Success
    }
    
    pub unsafe fn ncclReduce(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        _op: NcclRedOp,
        _root: i32,
        _comm: *mut NcclComm,
        _stream: CudaStream,
    ) -> NcclResult {
        if sendbuff.is_null() || recvbuff.is_null() {
            return NcclResult::InvalidArgument;
        }
        let size = count * dtype_size(datatype);
        ptr::copy_nonoverlapping(sendbuff as *const u8, recvbuff as *mut u8, size);
        NcclResult::Success
    }
    
    pub unsafe fn ncclAllGather(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        sendcount: usize,
        datatype: NcclDataType,
        _comm: *mut NcclComm,
        _stream: CudaStream,
    ) -> NcclResult {
        if sendbuff.is_null() || recvbuff.is_null() {
            return NcclResult::InvalidArgument;
        }
        // Single rank: just copy
        let size = sendcount * dtype_size(datatype);
        ptr::copy_nonoverlapping(sendbuff as *const u8, recvbuff as *mut u8, size);
        NcclResult::Success
    }
    
    pub unsafe fn ncclReduceScatter(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        recvcount: usize,
        datatype: NcclDataType,
        _op: NcclRedOp,
        _comm: *mut NcclComm,
        _stream: CudaStream,
    ) -> NcclResult {
        if sendbuff.is_null() || recvbuff.is_null() {
            return NcclResult::InvalidArgument;
        }
        let size = recvcount * dtype_size(datatype);
        ptr::copy_nonoverlapping(sendbuff as *const u8, recvbuff as *mut u8, size);
        NcclResult::Success
    }
    
    pub unsafe fn ncclSend(
        _sendbuff: *const c_void,
        _count: usize,
        _datatype: NcclDataType,
        _peer: i32,
        _comm: *mut NcclComm,
        _stream: CudaStream,
    ) -> NcclResult {
        // Simulation: no-op for single process
        NcclResult::Success
    }
    
    pub unsafe fn ncclRecv(
        recvbuff: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        _peer: i32,
        _comm: *mut NcclComm,
        _stream: CudaStream,
    ) -> NcclResult {
        if recvbuff.is_null() {
            return NcclResult::InvalidArgument;
        }
        // Simulation: zero-fill the buffer
        let size = count * dtype_size(datatype);
        ptr::write_bytes(recvbuff as *mut u8, 0, size);
        NcclResult::Success
    }
    
    pub unsafe fn ncclGroupStart() -> NcclResult {
        NcclResult::Success
    }
    
    pub unsafe fn ncclGroupEnd() -> NcclResult {
        NcclResult::Success
    }
    
    pub fn ncclGetErrorString(result: NcclResult) -> *const i8 {
        match result {
            NcclResult::Success => b"Success\0".as_ptr() as *const i8,
            NcclResult::UnhandledCudaError => b"Unhandled CUDA error\0".as_ptr() as *const i8,
            NcclResult::SystemError => b"System error\0".as_ptr() as *const i8,
            NcclResult::InternalError => b"Internal error\0".as_ptr() as *const i8,
            NcclResult::InvalidArgument => b"Invalid argument\0".as_ptr() as *const i8,
            NcclResult::InvalidUsage => b"Invalid usage\0".as_ptr() as *const i8,
            NcclResult::RemoteError => b"Remote error\0".as_ptr() as *const i8,
            NcclResult::InProgress => b"In progress\0".as_ptr() as *const i8,
            NcclResult::NumResults => b"Unknown\0".as_ptr() as *const i8,
        }
    }
    
    pub unsafe fn ncclGetVersion(version: *mut i32) -> NcclResult {
        if version.is_null() {
            return NcclResult::InvalidArgument;
        }
        // Simulate NCCL 2.18.0
        *version = 21800;
        NcclResult::Success
    }
    
    /// Get size of NCCL data type in bytes
    pub fn dtype_size(dtype: NcclDataType) -> usize {
        match dtype {
            NcclDataType::Int8 | NcclDataType::Uint8 => 1,
            NcclDataType::Float16 | NcclDataType::Bfloat16 => 2,
            NcclDataType::Int32 | NcclDataType::Uint32 | NcclDataType::Float32 => 4,
            NcclDataType::Int64 | NcclDataType::Uint64 | NcclDataType::Float64 => 8,
            _ => 4,
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub use simulation::*;

/// Safe wrapper for NCCL result checking
pub fn check_nccl_result(result: NcclResult) -> Result<(), String> {
    if result == NcclResult::Success {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = ncclGetErrorString(result);
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        };
        Err(format!("NCCL error: {}", msg))
    }
}

/// Get NCCL version string
pub fn get_nccl_version() -> Result<String, String> {
    let mut version: i32 = 0;
    let result = unsafe { ncclGetVersion(&mut version) };
    check_nccl_result(result)?;
    
    let major = version / 10000;
    let minor = (version % 10000) / 100;
    let patch = version % 100;
    
    Ok(format!("{}.{}.{}", major, minor, patch))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nccl_version() {
        let version = get_nccl_version().unwrap();
        assert!(!version.is_empty());
        // Simulation returns 2.18.0
        #[cfg(not(feature = "cuda"))]
        assert_eq!(version, "2.18.0");
    }
    
    #[test]
    fn test_unique_id_generation() {
        let mut id = NcclUniqueId::default();
        let result = unsafe { ncclGetUniqueId(&mut id) };
        assert_eq!(result, NcclResult::Success);
        
        #[cfg(not(feature = "cuda"))]
        {
            assert_eq!(id.internal[0], 0xDE);
            assert_eq!(id.internal[1], 0xAD);
        }
    }
    
    #[test]
    fn test_comm_init_destroy() {
        let mut id = NcclUniqueId::default();
        unsafe { ncclGetUniqueId(&mut id) };
        
        let mut comm: *mut NcclComm = std::ptr::null_mut();
        let result = unsafe { ncclCommInitRank(&mut comm, 1, id, 0) };
        assert_eq!(result, NcclResult::Success);
        assert!(!comm.is_null());
        
        let mut rank: i32 = -1;
        let result = unsafe { ncclCommUserRank(comm, &mut rank) };
        assert_eq!(result, NcclResult::Success);
        assert_eq!(rank, 0);
        
        let mut count: i32 = -1;
        let result = unsafe { ncclCommCount(comm, &mut count) };
        assert_eq!(result, NcclResult::Success);
        assert_eq!(count, 1);
        
        let result = unsafe { ncclCommDestroy(comm) };
        assert_eq!(result, NcclResult::Success);
    }
    
    #[test]
    fn test_dtype_size() {
        #[cfg(not(feature = "cuda"))]
        {
            assert_eq!(dtype_size(NcclDataType::Float32), 4);
            assert_eq!(dtype_size(NcclDataType::Float16), 2);
            assert_eq!(dtype_size(NcclDataType::Float64), 8);
            assert_eq!(dtype_size(NcclDataType::Int8), 1);
        }
    }
}
