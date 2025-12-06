//! Shared Memory Arena for Ray Actor Communication
//!
//! This module provides mmap-based shared memory for efficient tensor transfer
//! between Ray actors without serialization overhead.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use memmap2::{MmapMut, MmapOptions};
use parking_lot::RwLock;
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::tensor_view::CandleTensorView;

/// Error type for shared memory operations
#[derive(Debug, thiserror::Error)]
pub enum SharedMemoryError {
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
    
    #[error("Arena full: requested {requested} bytes, available {available}")]
    ArenaFull { requested: usize, available: usize },
    
    #[error("Invalid handle: {0}")]
    InvalidHandle(String),
    
    #[error("Allocation too large: {size} bytes exceeds max {max}")]
    AllocationTooLarge { size: usize, max: usize },
    
    #[error("Shape mismatch")]
    ShapeMismatch,
    
    #[error("Arena not initialized")]
    NotInitialized,
}

impl From<SharedMemoryError> for PyErr {
    fn from(err: SharedMemoryError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", err))
    }
}

/// Header stored at the beginning of each tensor allocation
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct TensorHeader {
    /// Magic number for validation
    magic: u32,
    /// Version of the header format
    version: u8,
    /// Data type (DType encoded as u8)
    dtype: u8,
    /// Number of dimensions
    ndim: u8,
    /// Reserved for future use
    reserved: u8,
    /// Total size in bytes (including header)
    total_size: u64,
    /// Offset to data start
    data_offset: u64,
    /// Shape dimensions (max 8 dims)
    shape: [u64; 8],
}

const TENSOR_HEADER_MAGIC: u32 = 0x54454E53; // "TENS" in ASCII
const TENSOR_HEADER_VERSION: u8 = 1;
const TENSOR_HEADER_SIZE: usize = std::mem::size_of::<TensorHeader>();

impl TensorHeader {
    fn new(dtype: DType, shape: &[usize]) -> Self {
        let mut header = Self {
            magic: TENSOR_HEADER_MAGIC,
            version: TENSOR_HEADER_VERSION,
            dtype: dtype_to_u8(dtype),
            ndim: shape.len() as u8,
            reserved: 0,
            total_size: 0,
            data_offset: TENSOR_HEADER_SIZE as u64,
            shape: [0; 8],
        };
        
        for (i, &dim) in shape.iter().take(8).enumerate() {
            header.shape[i] = dim as u64;
        }
        
        let elem_size = dtype.size_in_bytes();
        let num_elements: usize = shape.iter().product();
        header.total_size = (TENSOR_HEADER_SIZE + num_elements * elem_size) as u64;
        
        header
    }
    
    fn validate(&self) -> Result<(), SharedMemoryError> {
        if self.magic != TENSOR_HEADER_MAGIC {
            return Err(SharedMemoryError::InvalidHandle("Invalid magic number".to_string()));
        }
        if self.version != TENSOR_HEADER_VERSION {
            return Err(SharedMemoryError::InvalidHandle("Version mismatch".to_string()));
        }
        Ok(())
    }
    
    fn get_shape(&self) -> Vec<usize> {
        self.shape[..self.ndim as usize]
            .iter()
            .map(|&d| d as usize)
            .collect()
    }
    
    fn get_dtype(&self) -> DType {
        u8_to_dtype(self.dtype)
    }
}

fn dtype_to_u8(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F64 => 1,
        DType::F16 => 2,
        DType::BF16 => 3,
        DType::I64 => 4,
        DType::U32 => 6,
        DType::U8 => 7,
    }
}

fn u8_to_dtype(v: u8) -> DType {
    match v {
        0 => DType::F32,
        1 => DType::F64,
        2 => DType::F16,
        3 => DType::BF16,
        4 => DType::I64,
        6 => DType::U32,
        7 => DType::U8,
        _ => DType::F32,
    }
}

/// Allocation metadata for tracking arena usage
#[derive(Clone, Debug)]
struct Allocation {
    offset: usize,
    size: usize,
    #[allow(dead_code)]
    name: String,
}

/// Inner state of the shared memory arena
struct ArenaInner {
    mmap: MmapMut,
    allocations: HashMap<String, Allocation>,
    free_offset: usize,
    capacity: usize,
}

/// Shared memory arena for zero-copy tensor transfer
///
/// Uses mmap-based shared memory to allow multiple Ray actors
/// to share tensor data without serialization.
#[pyclass(name = "SharedMemoryArena")]
pub struct SharedMemoryArena {
    name: String,
    path: PathBuf,
    inner: Arc<RwLock<ArenaInner>>,
    allocation_counter: AtomicU64,
}

#[pymethods]
impl SharedMemoryArena {
    /// Create a new shared memory arena
    #[new]
    pub fn new(name: &str, size_bytes: usize) -> PyResult<Self> {
        Self::create(name, size_bytes).map_err(|e| e.into())
    }
    
    /// Get the arena name
    #[getter]
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the arena capacity in bytes
    #[getter]
    pub fn capacity(&self) -> usize {
        self.inner.read().capacity
    }
    
    /// Get the current used space in bytes
    #[getter]
    pub fn used(&self) -> usize {
        self.inner.read().free_offset
    }
    
    /// Get the available space in bytes
    #[getter]
    pub fn available(&self) -> usize {
        let inner = self.inner.read();
        inner.capacity - inner.free_offset
    }
    
    /// Get the number of active allocations
    #[getter]
    pub fn num_allocations(&self) -> usize {
        self.inner.read().allocations.len()
    }
    
    /// Allocate space for a tensor and return a handle
    pub fn allocate(&self, tensor: &CandleTensorView) -> PyResult<SharedTensorHandle> {
        self.allocate_tensor(tensor.tensor())
            .map_err(|e| e.into())
    }
    
    /// Allocate with a specific name for later retrieval
    pub fn allocate_named(&self, name: &str, tensor: &CandleTensorView) -> PyResult<SharedTensorHandle> {
        self.allocate_tensor_named(name, tensor.tensor())
            .map_err(|e| e.into())
    }
    
    /// Retrieve a tensor by name
    pub fn get(&self, name: &str) -> PyResult<CandleTensorView> {
        let tensor = self.get_tensor(name)?;
        Ok(CandleTensorView::from_tensor(tensor))
    }
    
    /// Read a tensor from a handle
    pub fn read(&self, handle: &SharedTensorHandle) -> PyResult<CandleTensorView> {
        let tensor = self.read_tensor(handle)?;
        Ok(CandleTensorView::from_tensor(tensor))
    }
    
    /// Free a named allocation
    pub fn free(&self, name: &str) -> PyResult<()> {
        self.free_allocation(name).map_err(|e| e.into())
    }
    
    /// Reset the arena, freeing all allocations
    pub fn reset(&self) {
        let mut inner = self.inner.write();
        inner.allocations.clear();
        inner.free_offset = 0;
    }
    
    /// List all current allocations
    pub fn list_allocations(&self) -> Vec<(String, usize, usize)> {
        self.inner.read()
            .allocations
            .iter()
            .map(|(name, alloc)| (name.clone(), alloc.offset, alloc.size))
            .collect()
    }
    
    /// Get the file path of the arena
    #[getter]
    pub fn path(&self) -> String {
        self.path.to_string_lossy().to_string()
    }
    
    fn __repr__(&self) -> String {
        format!(
            "SharedMemoryArena(name='{}', capacity={}, used={}, allocations={})",
            self.name,
            self.capacity(),
            self.used(),
            self.num_allocations()
        )
    }
}

impl SharedMemoryArena {
    /// Create a new shared memory arena (internal)
    pub fn create(name: &str, size_bytes: usize) -> Result<Self, SharedMemoryError> {
        // Use temporary directory for the mmap file
        let path = std::env::temp_dir().join(format!("deepseek_arena_{}.mmap", name));
        
        // Create or truncate the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        
        // Set the file size
        file.set_len(size_bytes as u64)?;
        
        // Memory map the file
        let mmap = unsafe { MmapOptions::new().len(size_bytes).map_mut(&file)? };
        
        let inner = ArenaInner {
            mmap,
            allocations: HashMap::new(),
            free_offset: 0,
            capacity: size_bytes,
        };
        
        Ok(Self {
            name: name.to_string(),
            path,
            inner: Arc::new(RwLock::new(inner)),
            allocation_counter: AtomicU64::new(0),
        })
    }
    
    /// Attach to an existing shared memory arena
    #[allow(dead_code)]
    pub fn attach(name: &str) -> Result<Self, SharedMemoryError> {
        let path = std::env::temp_dir().join(format!("deepseek_arena_{}.mmap", name));
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;
        
        let metadata = file.metadata()?;
        let size_bytes = metadata.len() as usize;
        
        let mmap = unsafe { MmapOptions::new().len(size_bytes).map_mut(&file)? };
        
        let inner = ArenaInner {
            mmap,
            allocations: HashMap::new(),
            free_offset: 0, // Note: We lose track of existing allocations
            capacity: size_bytes,
        };
        
        Ok(Self {
            name: name.to_string(),
            path,
            inner: Arc::new(RwLock::new(inner)),
            allocation_counter: AtomicU64::new(0),
        })
    }
    
    /// Allocate space for a tensor with auto-generated name
    pub fn allocate_tensor(&self, tensor: &Tensor) -> Result<SharedTensorHandle, SharedMemoryError> {
        let counter = self.allocation_counter.fetch_add(1, Ordering::SeqCst);
        let name = format!("tensor_{}", counter);
        self.allocate_tensor_named(&name, tensor)
    }
    
    /// Allocate space for a tensor with a specific name
    pub fn allocate_tensor_named(&self, name: &str, tensor: &Tensor) -> Result<SharedTensorHandle, SharedMemoryError> {
        let tensor = tensor.contiguous()?;
        let header = TensorHeader::new(tensor.dtype(), tensor.dims());
        let total_size = header.total_size as usize;
        
        let mut inner = self.inner.write();
        
        // Check available space
        let available = inner.capacity - inner.free_offset;
        if total_size > available {
            return Err(SharedMemoryError::ArenaFull {
                requested: total_size,
                available,
            });
        }
        
        let offset = inner.free_offset;
        
        // Write header
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const TensorHeader as *const u8,
                TENSOR_HEADER_SIZE,
            )
        };
        inner.mmap[offset..offset + TENSOR_HEADER_SIZE].copy_from_slice(header_bytes);
        
        // Write tensor data
        let data_offset = offset + TENSOR_HEADER_SIZE;
        let _data_size = total_size - TENSOR_HEADER_SIZE;
        
        match tensor.dtype() {
            DType::F32 => {
                let data = tensor.flatten_all()?.to_vec1::<f32>()?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                inner.mmap[data_offset..data_offset + bytes.len()].copy_from_slice(bytes);
            }
            DType::F64 => {
                let data = tensor.flatten_all()?.to_vec1::<f64>()?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)
                };
                inner.mmap[data_offset..data_offset + bytes.len()].copy_from_slice(bytes);
            }
            DType::I64 => {
                let data = tensor.flatten_all()?.to_vec1::<i64>()?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)
                };
                inner.mmap[data_offset..data_offset + bytes.len()].copy_from_slice(bytes);
            }
            DType::U32 => {
                let data = tensor.flatten_all()?.to_vec1::<u32>()?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                inner.mmap[data_offset..data_offset + bytes.len()].copy_from_slice(bytes);
            }
            DType::U8 => {
                let data = tensor.flatten_all()?.to_vec1::<u8>()?;
                inner.mmap[data_offset..data_offset + data.len()].copy_from_slice(&data);
            }
            _ => {
                // For F16/BF16, convert to F32 first
                let f32_tensor = tensor.to_dtype(DType::F32)?;
                let data = f32_tensor.flatten_all()?.to_vec1::<f32>()?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                inner.mmap[data_offset..data_offset + bytes.len()].copy_from_slice(bytes);
            }
        }
        
        // Record allocation
        inner.allocations.insert(name.to_string(), Allocation {
            offset,
            size: total_size,
            name: name.to_string(),
        });
        inner.free_offset = offset + total_size;
        
        // Flush to ensure data is written
        inner.mmap.flush()?;
        
        Ok(SharedTensorHandle {
            arena_name: self.name.clone(),
            name: name.to_string(),
            offset,
            size: total_size,
            shape: tensor.dims().to_vec(),
            dtype: format!("{:?}", tensor.dtype()),
        })
    }
    
    /// Get a tensor by name
    pub fn get_tensor(&self, name: &str) -> Result<Tensor, SharedMemoryError> {
        let inner = self.inner.read();
        
        let alloc = inner.allocations.get(name)
            .ok_or_else(|| SharedMemoryError::InvalidHandle(format!("No allocation named '{}'", name)))?;
        
        self.read_at_offset(&inner, alloc.offset)
    }
    
    /// Read a tensor from a handle
    pub fn read_tensor(&self, handle: &SharedTensorHandle) -> Result<Tensor, SharedMemoryError> {
        let inner = self.inner.read();
        self.read_at_offset(&inner, handle.offset)
    }
    
    /// Read tensor data at a specific offset
    fn read_at_offset(&self, inner: &ArenaInner, offset: usize) -> Result<Tensor, SharedMemoryError> {
        // Read header
        let header_bytes = &inner.mmap[offset..offset + TENSOR_HEADER_SIZE];
        let header: TensorHeader = unsafe {
            std::ptr::read(header_bytes.as_ptr() as *const TensorHeader)
        };
        header.validate()?;
        
        let shape = header.get_shape();
        let dtype = header.get_dtype();
        let data_offset = offset + TENSOR_HEADER_SIZE;
        let num_elements: usize = shape.iter().product();
        
        // Read tensor data based on dtype
        let tensor = match dtype {
            DType::F32 => {
                let byte_len = num_elements * 4;
                let bytes = &inner.mmap[data_offset..data_offset + byte_len];
                let data: Vec<f32> = bytes.chunks_exact(4)
                    .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
            }
            DType::F64 => {
                let byte_len = num_elements * 8;
                let bytes = &inner.mmap[data_offset..data_offset + byte_len];
                let data: Vec<f64> = bytes.chunks_exact(8)
                    .map(|chunk| f64::from_ne_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7]
                    ]))
                    .collect();
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
            }
            DType::I64 => {
                let byte_len = num_elements * 8;
                let bytes = &inner.mmap[data_offset..data_offset + byte_len];
                let data: Vec<i64> = bytes.chunks_exact(8)
                    .map(|chunk| i64::from_ne_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7]
                    ]))
                    .collect();
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
            }
            DType::U32 => {
                let byte_len = num_elements * 4;
                let bytes = &inner.mmap[data_offset..data_offset + byte_len];
                let data: Vec<u32> = bytes.chunks_exact(4)
                    .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
            }
            DType::U8 => {
                let bytes = &inner.mmap[data_offset..data_offset + num_elements];
                Tensor::from_vec(bytes.to_vec(), shape.as_slice(), &Device::Cpu)?
            }
            _ => {
                // For F16/BF16, read as F32 and convert
                let byte_len = num_elements * 4;
                let bytes = &inner.mmap[data_offset..data_offset + byte_len];
                let data: Vec<f32> = bytes.chunks_exact(4)
                    .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                let tensor = Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?;
                tensor.to_dtype(dtype)?
            }
        };
        
        Ok(tensor)
    }
    
    /// Free a named allocation (note: doesn't actually reclaim space, just removes tracking)
    pub fn free_allocation(&self, name: &str) -> Result<(), SharedMemoryError> {
        let mut inner = self.inner.write();
        inner.allocations.remove(name)
            .ok_or_else(|| SharedMemoryError::InvalidHandle(format!("No allocation named '{}'", name)))?;
        Ok(())
    }
}

impl Drop for SharedMemoryArena {
    fn drop(&mut self) {
        // Clean up the mmap file
        if let Err(e) = std::fs::remove_file(&self.path) {
            tracing::warn!("Failed to remove arena file: {}", e);
        }
    }
}

/// Handle to a tensor in shared memory
///
/// Can be passed between processes to reference the same tensor data.
#[pyclass(name = "SharedTensorHandle")]
#[derive(Clone, Debug)]
pub struct SharedTensorHandle {
    #[pyo3(get)]
    pub arena_name: String,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub offset: usize,
    #[pyo3(get)]
    pub size: usize,
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub dtype: String,
}

#[pymethods]
impl SharedTensorHandle {
    /// Serialize the handle to bytes for IPC
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let json = serde_json::to_vec(&HandleData {
            arena_name: self.arena_name.clone(),
            name: self.name.clone(),
            offset: self.offset,
            size: self.size,
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
        }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Serialization error: {}", e)))?;
        
        Ok(PyBytes::new_bound(py, &json))
    }
    
    /// Deserialize a handle from bytes
    #[staticmethod]
    pub fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let handle_data: HandleData = serde_json::from_slice(data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Deserialization error: {}", e)))?;
        
        Ok(Self {
            arena_name: handle_data.arena_name,
            name: handle_data.name,
            offset: handle_data.offset,
            size: handle_data.size,
            shape: handle_data.shape,
            dtype: handle_data.dtype,
        })
    }
    
    fn __repr__(&self) -> String {
        format!(
            "SharedTensorHandle(arena='{}', name='{}', shape={:?}, dtype={})",
            self.arena_name, self.name, self.shape, self.dtype
        )
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct HandleData {
    arena_name: String,
    name: String,
    offset: usize,
    size: usize,
    shape: Vec<usize>,
    dtype: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_creation() {
        let arena = SharedMemoryArena::create("test_create", 1024 * 1024).unwrap();
        assert_eq!(arena.name(), "test_create");
        assert_eq!(arena.capacity(), 1024 * 1024);
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_tensor_allocation() {
        let arena = SharedMemoryArena::create("test_alloc", 1024 * 1024).unwrap();
        
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data.clone(), &[2, 3], &Device::Cpu).unwrap();
        
        let handle = arena.allocate_tensor_named("test_tensor", &tensor).unwrap();
        assert_eq!(handle.name, "test_tensor");
        assert_eq!(handle.shape, vec![2, 3]);
        
        // Read it back
        let restored = arena.get_tensor("test_tensor").unwrap();
        let restored_data = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(restored_data, data);
    }

    #[test]
    fn test_multiple_allocations() {
        let arena = SharedMemoryArena::create("test_multi", 1024 * 1024).unwrap();
        
        let t1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        let t2 = Tensor::from_vec(vec![4.0f32, 5.0, 6.0, 7.0], &[2, 2], &Device::Cpu).unwrap();
        let t3 = Tensor::from_vec(vec![8i64, 9, 10, 11, 12], &[5], &Device::Cpu).unwrap();
        
        arena.allocate_tensor_named("tensor1", &t1).unwrap();
        arena.allocate_tensor_named("tensor2", &t2).unwrap();
        arena.allocate_tensor_named("tensor3", &t3).unwrap();
        
        assert_eq!(arena.num_allocations(), 3);
        
        // Verify each tensor
        let r1 = arena.get_tensor("tensor1").unwrap();
        let r2 = arena.get_tensor("tensor2").unwrap();
        let r3 = arena.get_tensor("tensor3").unwrap();
        
        assert_eq!(r1.dims(), &[3]);
        assert_eq!(r2.dims(), &[2, 2]);
        assert_eq!(r3.dims(), &[5]);
    }

    #[test]
    fn test_arena_reset() {
        let arena = SharedMemoryArena::create("test_reset", 1024 * 1024).unwrap();
        
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        arena.allocate_tensor_named("test", &tensor).unwrap();
        
        assert!(arena.used() > 0);
        assert_eq!(arena.num_allocations(), 1);
        
        arena.reset();
        
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.num_allocations(), 0);
    }

    #[test]
    fn test_handle_serialization() {
        let handle = SharedTensorHandle {
            arena_name: "test".to_string(),
            name: "tensor".to_string(),
            offset: 100,
            size: 200,
            shape: vec![2, 3],
            dtype: "F32".to_string(),
        };
        
        let data = serde_json::to_vec(&HandleData {
            arena_name: handle.arena_name.clone(),
            name: handle.name.clone(),
            offset: handle.offset,
            size: handle.size,
            shape: handle.shape.clone(),
            dtype: handle.dtype.clone(),
        }).unwrap();
        
        let restored: HandleData = serde_json::from_slice(&data).unwrap();
        assert_eq!(restored.arena_name, "test");
        assert_eq!(restored.shape, vec![2, 3]);
    }
    
    #[test]
    fn test_tensor_header() {
        let header = TensorHeader::new(DType::F32, &[2, 3, 4]);
        assert_eq!(header.magic, TENSOR_HEADER_MAGIC);
        assert_eq!(header.version, TENSOR_HEADER_VERSION);
        assert_eq!(header.ndim, 3);
        assert_eq!(header.get_shape(), vec![2, 3, 4]);
        assert!(matches!(header.get_dtype(), DType::F32));
    }
}
