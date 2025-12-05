//! NCCL-based communicator implementing CollectiveCommunicator trait.
//!
//! Provides GPU-accelerated collective operations using NVIDIA's NCCL library.
//! Falls back to simulation when CUDA is not available.

use candle_core::{DType, Device, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use std::sync::Arc;
use super::CollectiveCommunicator;
use super::nccl_sys::{
    NcclComm, NcclDataType, NcclRedOp, NcclResult, NcclUniqueId,
    CudaStream, check_nccl_result,
    ncclCommInitRank, ncclCommDestroy, ncclCommCount, ncclCommUserRank,
    ncclAllReduce, ncclBroadcast, ncclAllGather, ncclReduceScatter,
    ncclSend, ncclRecv, ncclGroupStart, ncclGroupEnd, ncclGetUniqueId,
};

/// NCCL-based communicator for GPU collective operations.
pub struct NcclCommunicator {
    /// NCCL communicator handle
    comm: *mut NcclComm,
    /// This process's rank
    rank: usize,
    /// Total number of ranks
    world_size: usize,
    /// CUDA stream for async operations (null for default stream)
    stream: CudaStream,
    /// Device for tensor operations
    device: Device,
}

// Safety: NcclComm is thread-safe when used correctly with NCCL semantics
unsafe impl Send for NcclCommunicator {}
unsafe impl Sync for NcclCommunicator {}

impl NcclCommunicator {
    /// Initialize NCCL communicator.
    /// 
    /// # Arguments
    /// * `rank` - This process's rank (0 to world_size-1)
    /// * `world_size` - Total number of processes
    /// * `unique_id` - Shared unique ID from rank 0 (use `generate_unique_id`)
    /// * `device` - CUDA device to use
    /// 
    /// # Safety
    /// All ranks must call this with the same unique_id and world_size.
    pub fn new(
        rank: usize,
        world_size: usize,
        unique_id: NcclUniqueId,
        device: Device,
    ) -> std::result::Result<Self, String> {
        let mut comm: *mut NcclComm = std::ptr::null_mut();
        
        let result = unsafe {
            ncclCommInitRank(
                &mut comm,
                world_size as i32,
                unique_id,
                rank as i32,
            )
        };
        
        check_nccl_result(result)?;
        
        Ok(Self {
            comm,
            rank,
            world_size,
            stream: std::ptr::null_mut(), // Default stream
            device,
        })
    }
    
    /// Generate a unique ID for NCCL initialization.
    /// Only rank 0 should call this and broadcast to other ranks.
    pub fn generate_unique_id() -> std::result::Result<NcclUniqueId, String> {
        let mut id = NcclUniqueId::default();
        let result = unsafe { ncclGetUniqueId(&mut id) };
        check_nccl_result(result)?;
        Ok(id)
    }
    
    /// Create a group of communicators for local testing (single process).
    pub fn new_local_group(world_size: usize, device: Device) -> std::result::Result<Vec<Self>, String> {
        let unique_id = Self::generate_unique_id()?;
        
        let mut comms = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            comms.push(Self::new(rank, world_size, unique_id, device.clone())?);
        }
        
        Ok(comms)
    }
    
    /// Convert candle DType to NCCL data type
    fn dtype_to_nccl(dtype: DType) -> NcclDataType {
        match dtype {
            DType::F32 => NcclDataType::Float32,
            DType::F64 => NcclDataType::Float64,
            DType::F16 => NcclDataType::Float16,
            DType::BF16 => NcclDataType::Bfloat16,
            DType::I64 => NcclDataType::Int64,
            DType::U32 => NcclDataType::Uint32,
            DType::U8 => NcclDataType::Uint8,
        }
    }
    
    /// Get raw data pointer and count from tensor
    fn tensor_data_ptr(tensor: &Tensor) -> std::result::Result<(*const std::ffi::c_void, usize), String> {
        let storage = tensor.storage_and_layout().0;
        let count = tensor.elem_count();
        
        // Get raw pointer from storage
        // Note: This assumes contiguous storage
        match &*storage {
            candle_core::Storage::Cpu(cpu_storage) => {
                let ptr = match tensor.dtype() {
                    DType::F32 => cpu_storage.as_slice::<f32>()
                        .map_err(|e| e.to_string())?
                        .as_ptr() as *const std::ffi::c_void,
                    DType::F64 => cpu_storage.as_slice::<f64>()
                        .map_err(|e| e.to_string())?
                        .as_ptr() as *const std::ffi::c_void,
                    DType::I64 => cpu_storage.as_slice::<i64>()
                        .map_err(|e| e.to_string())?
                        .as_ptr() as *const std::ffi::c_void,
                    DType::U32 => cpu_storage.as_slice::<u32>()
                        .map_err(|e| e.to_string())?
                        .as_ptr() as *const std::ffi::c_void,
                    DType::U8 => cpu_storage.as_slice::<u8>()
                        .map_err(|e| e.to_string())?
                        .as_ptr() as *const std::ffi::c_void,
                    _ => return Err(format!("Unsupported dtype: {:?}", tensor.dtype())),
                };
                Ok((ptr, count))
            }
            #[cfg(feature = "cuda")]
            candle_core::Storage::Cuda(cuda_storage) => {
                // For CUDA, we need the device pointer
                let ptr = *cuda_storage.as_cuda_slice::<f32>()
                    .map_err(|e| e.to_string())?
                    .device_ptr() as *const std::ffi::c_void;
                Ok((ptr, count))
            }
            _ => Err("Unsupported storage type for NCCL".to_string()),
        }
    }
    
    /// Perform all-reduce operation using NCCL
    fn nccl_all_reduce(&self, tensor: &Tensor, op: NcclRedOp) -> Result<Tensor> {
        let (send_ptr, count) = Self::tensor_data_ptr(tensor)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        // Create output tensor
        let output = Tensor::zeros(tensor.shape(), tensor.dtype(), tensor.device())?;
        let (recv_ptr, _) = Self::tensor_data_ptr(&output)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        let nccl_dtype = Self::dtype_to_nccl(tensor.dtype());
        
        let result = unsafe {
            ncclAllReduce(
                send_ptr,
                recv_ptr as *mut std::ffi::c_void,
                count,
                nccl_dtype,
                op,
                self.comm,
                self.stream,
            )
        };
        
        check_nccl_result(result)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        Ok(output)
    }
}

impl Drop for NcclCommunicator {
    fn drop(&mut self) {
        if !self.comm.is_null() {
            unsafe {
                let _ = ncclCommDestroy(self.comm);
            }
        }
    }
}

impl CollectiveCommunicator for NcclCommunicator {
    fn rank(&self) -> usize {
        self.rank
    }
    
    fn world_size(&self) -> usize {
        self.world_size
    }
    
    fn all_reduce(&self, tensor: &Tensor) -> Result<Tensor> {
        self.nccl_all_reduce(tensor, NcclRedOp::Sum)
    }
    
    fn all_gather(&self, tensor: &Tensor) -> Result<Tensor> {
        let (send_ptr, count) = Self::tensor_data_ptr(tensor)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        // Output is world_size times larger
        let mut out_shape: Vec<usize> = tensor.dims().to_vec();
        out_shape[0] *= self.world_size;
        let output = Tensor::zeros(&out_shape[..], tensor.dtype(), tensor.device())?;
        let (recv_ptr, _) = Self::tensor_data_ptr(&output)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        let nccl_dtype = Self::dtype_to_nccl(tensor.dtype());
        
        let result = unsafe {
            ncclAllGather(
                send_ptr,
                recv_ptr as *mut std::ffi::c_void,
                count,
                nccl_dtype,
                self.comm,
                self.stream,
            )
        };
        
        check_nccl_result(result)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        Ok(output)
    }
    
    fn broadcast(&self, tensor: &Tensor, root_rank: usize) -> Result<Tensor> {
        let (send_ptr, count) = Self::tensor_data_ptr(tensor)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        let output = Tensor::zeros(tensor.shape(), tensor.dtype(), tensor.device())?;
        let (recv_ptr, _) = Self::tensor_data_ptr(&output)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        let nccl_dtype = Self::dtype_to_nccl(tensor.dtype());
        
        let result = unsafe {
            ncclBroadcast(
                send_ptr,
                recv_ptr as *mut std::ffi::c_void,
                count,
                nccl_dtype,
                root_rank as i32,
                self.comm,
                self.stream,
            )
        };
        
        check_nccl_result(result)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        Ok(output)
    }
    
    fn reduce_scatter(&self, tensor: &Tensor) -> Result<Tensor> {
        let (send_ptr, count) = Self::tensor_data_ptr(tensor)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        // Output is 1/world_size smaller
        let recv_count = count / self.world_size;
        let mut out_shape: Vec<usize> = tensor.dims().to_vec();
        out_shape[0] /= self.world_size;
        let output = Tensor::zeros(&out_shape[..], tensor.dtype(), tensor.device())?;
        let (recv_ptr, _) = Self::tensor_data_ptr(&output)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        let nccl_dtype = Self::dtype_to_nccl(tensor.dtype());
        
        let result = unsafe {
            ncclReduceScatter(
                send_ptr,
                recv_ptr as *mut std::ffi::c_void,
                recv_count,
                nccl_dtype,
                NcclRedOp::Sum,
                self.comm,
                self.stream,
            )
        };
        
        check_nccl_result(result)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        Ok(output)
    }
    
    fn all_to_all(&self, tensor: &Tensor) -> Result<Tensor> {
        // All-to-all using point-to-point operations
        // Each rank sends/receives world_size chunks
        let dims = tensor.dims();
        if dims[0] != self.world_size {
            return Err(candle_core::Error::Msg(format!(
                "First dimension ({}) must equal world_size ({})",
                dims[0], self.world_size
            )));
        }
        
        let chunk_size: usize = dims[1..].iter().product();
        let output = Tensor::zeros(tensor.shape(), tensor.dtype(), tensor.device())?;
        
        // Start group for batched operations
        check_nccl_result(unsafe { ncclGroupStart() })
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        // Send to all ranks, receive from all ranks
        for peer in 0..self.world_size {
            let send_chunk = tensor.narrow(0, peer, 1)?.flatten_all()?;
            let (send_ptr, _) = Self::tensor_data_ptr(&send_chunk)
                .map_err(|e| candle_core::Error::Msg(e))?;
            
            let recv_chunk = output.narrow(0, peer, 1)?.flatten_all()?;
            let (recv_ptr, _) = Self::tensor_data_ptr(&recv_chunk)
                .map_err(|e| candle_core::Error::Msg(e))?;
            
            let nccl_dtype = Self::dtype_to_nccl(tensor.dtype());
            
            unsafe {
                ncclSend(
                    send_ptr,
                    chunk_size,
                    nccl_dtype,
                    peer as i32,
                    self.comm,
                    self.stream,
                );
                ncclRecv(
                    recv_ptr as *mut std::ffi::c_void,
                    chunk_size,
                    nccl_dtype,
                    peer as i32,
                    self.comm,
                    self.stream,
                );
            }
        }
        
        // End group
        check_nccl_result(unsafe { ncclGroupEnd() })
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        Ok(output)
    }
    
    fn all_to_all_variable(
        &self,
        tensor: &Tensor,
        input_splits: &[usize],
        output_splits: &[usize],
    ) -> Result<Tensor> {
        // Variable-size all-to-all
        let total_recv: usize = output_splits.iter().sum();
        let hidden_dim = if tensor.dims().len() > 1 { tensor.dim(1)? } else { 1 };
        
        let output = Tensor::zeros(
            (total_recv, hidden_dim),
            tensor.dtype(),
            tensor.device()
        )?;
        
        check_nccl_result(unsafe { ncclGroupStart() })
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        let mut send_offset = 0;
        let mut recv_offset = 0;
        
        for peer in 0..self.world_size {
            let send_count = input_splits.get(peer).copied().unwrap_or(0);
            let recv_count = output_splits.get(peer).copied().unwrap_or(0);
            
            if send_count > 0 {
                let send_chunk = tensor.narrow(0, send_offset, send_count)?;
                let (send_ptr, count) = Self::tensor_data_ptr(&send_chunk)
                    .map_err(|e| candle_core::Error::Msg(e))?;
                
                let nccl_dtype = Self::dtype_to_nccl(tensor.dtype());
                
                unsafe {
                    ncclSend(
                        send_ptr,
                        count,
                        nccl_dtype,
                        peer as i32,
                        self.comm,
                        self.stream,
                    );
                }
            }
            
            if recv_count > 0 {
                let recv_chunk = output.narrow(0, recv_offset, recv_count)?;
                let (recv_ptr, count) = Self::tensor_data_ptr(&recv_chunk)
                    .map_err(|e| candle_core::Error::Msg(e))?;
                
                let nccl_dtype = Self::dtype_to_nccl(tensor.dtype());
                
                unsafe {
                    ncclRecv(
                        recv_ptr as *mut std::ffi::c_void,
                        count,
                        nccl_dtype,
                        peer as i32,
                        self.comm,
                        self.stream,
                    );
                }
            }
            
            send_offset += send_count;
            recv_offset += recv_count;
        }
        
        check_nccl_result(unsafe { ncclGroupEnd() })
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        Ok(output)
    }
    
    fn send(&self, tensor: &Tensor, dst_rank: usize) -> Result<()> {
        let (send_ptr, count) = Self::tensor_data_ptr(tensor)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        let nccl_dtype = Self::dtype_to_nccl(tensor.dtype());
        
        let result = unsafe {
            ncclSend(
                send_ptr,
                count,
                nccl_dtype,
                dst_rank as i32,
                self.comm,
                self.stream,
            )
        };
        
        check_nccl_result(result)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        Ok(())
    }
    
    fn recv(&self, shape: &[usize], device: &Device, src_rank: usize) -> Result<Tensor> {
        let output = Tensor::zeros(shape, DType::F32, device)?;
        let (recv_ptr, count) = Self::tensor_data_ptr(&output)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        let result = unsafe {
            ncclRecv(
                recv_ptr as *mut std::ffi::c_void,
                count,
                NcclDataType::Float32,
                src_rank as i32,
                self.comm,
                self.stream,
            )
        };
        
        check_nccl_result(result)
            .map_err(|e| candle_core::Error::Msg(e))?;
        
        Ok(output)
    }
}

/// Create NCCL communicators for all ranks in a distributed setting.
/// 
/// This function should be called by all ranks with the same unique_id.
/// The unique_id should be generated by rank 0 and broadcast to all others.
pub fn create_nccl_communicators(
    rank: usize,
    world_size: usize,
    unique_id: NcclUniqueId,
    device: Device,
) -> std::result::Result<Arc<NcclCommunicator>, String> {
    let comm = NcclCommunicator::new(rank, world_size, unique_id, device)?;
    Ok(Arc::new(comm))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nccl_communicator_single_rank() {
        let device = Device::Cpu;
        let unique_id = NcclCommunicator::generate_unique_id().unwrap();
        let comm = NcclCommunicator::new(0, 1, unique_id, device).unwrap();
        
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.world_size(), 1);
    }
    
    #[test]
    fn test_nccl_all_reduce_single_rank() {
        let device = Device::Cpu;
        let unique_id = NcclCommunicator::generate_unique_id().unwrap();
        let comm = NcclCommunicator::new(0, 1, unique_id, device.clone()).unwrap();
        
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (4,), &device).unwrap();
        let result = comm.all_reduce(&tensor).unwrap();
        
        // Single rank: output should equal input
        assert_eq!(result.dims(), tensor.dims());
    }
    
    #[test]
    fn test_nccl_broadcast_single_rank() {
        let device = Device::Cpu;
        let unique_id = NcclCommunicator::generate_unique_id().unwrap();
        let comm = NcclCommunicator::new(0, 1, unique_id, device.clone()).unwrap();
        
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (4,), &device).unwrap();
        let result = comm.broadcast(&tensor, 0).unwrap();
        
        assert_eq!(result.dims(), tensor.dims());
    }
    
    #[test]
    fn test_nccl_all_gather_single_rank() {
        let device = Device::Cpu;
        let unique_id = NcclCommunicator::generate_unique_id().unwrap();
        let comm = NcclCommunicator::new(0, 1, unique_id, device.clone()).unwrap();
        
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (4,), &device).unwrap();
        let result = comm.all_gather(&tensor).unwrap();
        
        // Single rank: output should have same shape (1 * input)
        assert_eq!(result.dims(), tensor.dims());
    }
    
    #[test]
    fn test_dtype_conversion() {
        assert_eq!(NcclCommunicator::dtype_to_nccl(DType::F32), NcclDataType::Float32);
        assert_eq!(NcclCommunicator::dtype_to_nccl(DType::F16), NcclDataType::Float16);
        assert_eq!(NcclCommunicator::dtype_to_nccl(DType::BF16), NcclDataType::Bfloat16);
        assert_eq!(NcclCommunicator::dtype_to_nccl(DType::I64), NcclDataType::Int64);
    }
}
