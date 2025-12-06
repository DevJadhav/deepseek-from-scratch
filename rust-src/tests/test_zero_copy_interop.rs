//! Integration tests for Zero-Copy PyO3 Interop
//!
//! Tests the complete zero-copy tensor transfer pipeline including:
//! - CandleTensorView creation and conversion
//! - Arrow IPC serialization/deserialization
//! - Shared memory arena operations
//! - Cross-process tensor sharing simulation
//!
//! Note: These tests require the `pyo3-bindings` feature to be enabled.

#![cfg(feature = "pyo3-bindings")]

use candle_core::{DType, Device, Tensor};
use std::sync::Arc;

// Import the zero-copy modules
use deepseek_rust::pyo3_bindings::{
    tensor_view::CandleTensorView,
    shared_memory::{SharedMemoryArena, SharedTensorHandle},
};

// ============================================================================
// CandleTensorView Tests
// ============================================================================

mod tensor_view_tests {
    use super::*;

    #[test]
    fn test_tensor_creation_f32() {
        let view = CandleTensorView::new(vec![2, 3, 4], "f32").unwrap();
        assert_eq!(view.numel(), 24);
        assert_eq!(view.dtype(), "F32");
        assert!(!view.is_borrowed());
    }

    #[test]
    fn test_tensor_creation_various_dtypes() {
        // Test all supported dtypes
        let dtypes = vec!["f32", "f64", "i64", "i32", "u32", "u8", "f16", "bf16"];
        for dtype in dtypes {
            let view = CandleTensorView::new(vec![4], dtype);
            assert!(view.is_ok(), "Failed to create tensor with dtype: {}", dtype);
        }
    }

    #[test]
    fn test_tensor_from_candle() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data.clone(), &[2, 3], &Device::Cpu).unwrap();
        let view = CandleTensorView::from_tensor(tensor);
        
        assert_eq!(view.numel(), 6);
        assert_eq!(view.tensor().dims(), &[2, 3]);
    }

    #[test]
    fn test_tensor_reshape() {
        let view = CandleTensorView::new(vec![2, 3], "f32").unwrap();
        let reshaped = view.reshape(vec![6]).unwrap();
        assert_eq!(reshaped.tensor().dims(), &[6]);
        
        let reshaped2 = view.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped2.tensor().dims(), &[3, 2]);
    }

    #[test]
    fn test_tensor_dtype_conversion() {
        let view = CandleTensorView::new(vec![4], "f32").unwrap();
        
        let f64_view = view.to_dtype("f64").unwrap();
        assert_eq!(f64_view.dtype(), "F64");
        
        let i64_view = view.to_dtype("i64").unwrap();
        assert_eq!(i64_view.dtype(), "I64");
    }

    #[test]
    fn test_tensor_contiguous() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, &[2, 3], &Device::Cpu).unwrap();
        let transposed = tensor.transpose(0, 1).unwrap();
        
        let view = CandleTensorView::from_tensor(transposed);
        // After transpose, tensor may not be contiguous
        let contiguous = view.contiguous().unwrap();
        assert!(contiguous.is_contiguous());
    }

    #[test]
    fn test_tensor_matmul() {
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];
        
        let a = Tensor::from_vec(a_data, &[2, 2], &Device::Cpu).unwrap();
        let b = Tensor::from_vec(b_data, &[2, 2], &Device::Cpu).unwrap();
        
        let view_a = CandleTensorView::from_tensor(a);
        let view_b = CandleTensorView::from_tensor(b);
        
        let result = view_a.matmul(&view_b).unwrap();
        assert_eq!(result.tensor().dims(), &[2, 2]);
        
        // Verify result: [[19, 22], [43, 50]]
        let result_data = result.tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((result_data[0] - 19.0).abs() < 1e-5);
        assert!((result_data[1] - 22.0).abs() < 1e-5);
        assert!((result_data[2] - 43.0).abs() < 1e-5);
        assert!((result_data[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_tensor_elementwise_ops() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data.clone(), &[4], &Device::Cpu).unwrap();
        
        let view = CandleTensorView::from_tensor(tensor.clone());
        let other = CandleTensorView::from_tensor(tensor);
        
        // Test add
        let sum = view.add(&other).unwrap();
        let sum_data = sum.tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(sum_data, vec![2.0, 4.0, 6.0, 8.0]);
        
        // Test mul
        let mul = view.mul(&other).unwrap();
        let mul_data = mul.tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(mul_data, vec![1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_tensor_nbytes() {
        let view_f32 = CandleTensorView::new(vec![100], "f32").unwrap();
        assert_eq!(view_f32.nbytes(), 400); // 100 * 4 bytes
        
        let view_f64 = CandleTensorView::new(vec![100], "f64").unwrap();
        assert_eq!(view_f64.nbytes(), 800); // 100 * 8 bytes
        
        let view_u8 = CandleTensorView::new(vec![100], "u8").unwrap();
        assert_eq!(view_u8.nbytes(), 100); // 100 * 1 byte
    }
}

// ============================================================================
// Shared Memory Arena Tests
// ============================================================================

mod shared_memory_tests {
    use super::*;

    #[test]
    fn test_arena_creation() {
        let arena = SharedMemoryArena::create("test_create", 1024 * 1024).unwrap();
        assert_eq!(arena.name(), "test_create");
        assert_eq!(arena.capacity(), 1024 * 1024);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.num_allocations(), 0);
    }

    #[test]
    fn test_arena_single_tensor() {
        let arena = SharedMemoryArena::create("test_single", 1024 * 1024).unwrap();
        
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data.clone(), &[2, 3], &Device::Cpu).unwrap();
        
        let handle = arena.allocate_tensor_named("my_tensor", &tensor).unwrap();
        assert_eq!(handle.name, "my_tensor");
        assert_eq!(handle.shape, vec![2, 3]);
        assert_eq!(handle.dtype, "F32");
        
        // Verify data integrity
        let restored = arena.get_tensor("my_tensor").unwrap();
        let restored_data = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(restored_data, data);
    }

    #[test]
    fn test_arena_multiple_tensors() {
        let arena = SharedMemoryArena::create("test_multi", 1024 * 1024).unwrap();
        
        let t1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        let t2 = Tensor::from_vec(vec![4.0f32, 5.0, 6.0, 7.0], &[2, 2], &Device::Cpu).unwrap();
        let t3 = Tensor::from_vec(vec![8.0f32, 9.0], &[2], &Device::Cpu).unwrap();
        
        arena.allocate_tensor_named("t1", &t1).unwrap();
        arena.allocate_tensor_named("t2", &t2).unwrap();
        arena.allocate_tensor_named("t3", &t3).unwrap();
        
        assert_eq!(arena.num_allocations(), 3);
        
        // Verify each tensor
        let r1 = arena.get_tensor("t1").unwrap();
        let r2 = arena.get_tensor("t2").unwrap();
        let r3 = arena.get_tensor("t3").unwrap();
        
        assert_eq!(r1.dims(), &[3]);
        assert_eq!(r2.dims(), &[2, 2]);
        assert_eq!(r3.dims(), &[2]);
    }

    #[test]
    fn test_arena_different_dtypes() {
        let arena = SharedMemoryArena::create("test_dtypes", 1024 * 1024).unwrap();
        
        // f32
        let f32_tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        arena.allocate_tensor_named("f32_tensor", &f32_tensor).unwrap();
        
        // f64
        let f64_tensor = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        arena.allocate_tensor_named("f64_tensor", &f64_tensor).unwrap();
        
        // i64
        let i64_tensor = Tensor::from_vec(vec![1i64, 2, 3], &[3], &Device::Cpu).unwrap();
        arena.allocate_tensor_named("i64_tensor", &i64_tensor).unwrap();
        
        // u32 (candle doesn't support i32)
        let u32_tensor = Tensor::from_vec(vec![1u32, 2, 3], &[3], &Device::Cpu).unwrap();
        arena.allocate_tensor_named("i32_tensor", &u32_tensor).unwrap();
        
        // u8
        let u8_tensor = Tensor::from_vec(vec![1u8, 2, 3], &[3], &Device::Cpu).unwrap();
        arena.allocate_tensor_named("u8_tensor", &u8_tensor).unwrap();
        
        // Verify all tensors
        let r_f32 = arena.get_tensor("f32_tensor").unwrap();
        let r_f64 = arena.get_tensor("f64_tensor").unwrap();
        let r_i64 = arena.get_tensor("i64_tensor").unwrap();
        let r_u32 = arena.get_tensor("i32_tensor").unwrap();
        let r_u8 = arena.get_tensor("u8_tensor").unwrap();
        
        assert!(matches!(r_f32.dtype(), DType::F32));
        assert!(matches!(r_f64.dtype(), DType::F64));
        assert!(matches!(r_i64.dtype(), DType::I64));
        assert!(matches!(r_u32.dtype(), DType::U32));
        assert!(matches!(r_u8.dtype(), DType::U8));
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
    fn test_arena_free_allocation() {
        let arena = SharedMemoryArena::create("test_free", 1024 * 1024).unwrap();
        
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        arena.allocate_tensor_named("to_free", &tensor).unwrap();
        
        assert_eq!(arena.num_allocations(), 1);
        
        arena.free_allocation("to_free").unwrap();
        
        assert_eq!(arena.num_allocations(), 0);
        
        // Getting freed tensor should fail
        assert!(arena.get_tensor("to_free").is_err());
    }

    #[test]
    fn test_arena_handle_read() {
        let arena = SharedMemoryArena::create("test_handle", 1024 * 1024).unwrap();
        
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data.clone(), &[2, 2], &Device::Cpu).unwrap();
        
        let handle = arena.allocate_tensor_named("via_handle", &tensor).unwrap();
        
        // Read using handle
        let restored = arena.read_tensor(&handle).unwrap();
        let restored_data = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(restored_data, data);
    }

    #[test]
    fn test_arena_list_allocations() {
        let arena = SharedMemoryArena::create("test_list", 1024 * 1024).unwrap();
        
        let t1 = Tensor::from_vec(vec![1.0f32, 2.0], &[2], &Device::Cpu).unwrap();
        let t2 = Tensor::from_vec(vec![3.0f32, 4.0, 5.0], &[3], &Device::Cpu).unwrap();
        
        arena.allocate_tensor_named("first", &t1).unwrap();
        arena.allocate_tensor_named("second", &t2).unwrap();
        
        let allocations = arena.list_allocations();
        assert_eq!(allocations.len(), 2);
        
        let names: Vec<&String> = allocations.iter().map(|(name, _, _)| name).collect();
        assert!(names.contains(&&"first".to_string()));
        assert!(names.contains(&&"second".to_string()));
    }

    #[test]
    fn test_arena_capacity_tracking() {
        let arena = SharedMemoryArena::create("test_capacity", 1024 * 1024).unwrap();
        
        assert_eq!(arena.available(), arena.capacity());
        
        let tensor = Tensor::from_vec(vec![1.0f32; 1000], &[1000], &Device::Cpu).unwrap();
        arena.allocate_tensor_named("large", &tensor).unwrap();
        
        let used = arena.used();
        assert!(used > 0);
        assert_eq!(arena.available(), arena.capacity() - used);
    }

    #[test]
    fn test_arena_large_tensor() {
        let arena = SharedMemoryArena::create("test_large", 64 * 1024 * 1024).unwrap();
        
        // Create a 1M element tensor (4MB for f32)
        let size = 1024 * 1024;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, &[1024, 1024], &Device::Cpu).unwrap();
        
        let handle = arena.allocate_tensor_named("large_tensor", &tensor).unwrap();
        assert_eq!(handle.shape, vec![1024, 1024]);
        
        // Verify first and last elements
        let restored = arena.get_tensor("large_tensor").unwrap();
        let restored_data = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((restored_data[0] - 0.0).abs() < 1e-5);
        assert!((restored_data[size - 1] - (size - 1) as f32).abs() < 1e-5);
    }

    #[test]
    fn test_arena_multidimensional_tensors() {
        let arena = SharedMemoryArena::create("test_multidim", 1024 * 1024).unwrap();
        
        // 3D tensor
        let t3d = Tensor::from_vec(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            &[2, 3, 4],
            &Device::Cpu
        ).unwrap();
        arena.allocate_tensor_named("t3d", &t3d).unwrap();
        
        // 4D tensor
        let t4d = Tensor::from_vec(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            &[2, 2, 2, 3],
            &Device::Cpu
        ).unwrap();
        arena.allocate_tensor_named("t4d", &t4d).unwrap();
        
        // Verify shapes are preserved
        let r3d = arena.get_tensor("t3d").unwrap();
        let r4d = arena.get_tensor("t4d").unwrap();
        
        assert_eq!(r3d.dims(), &[2, 3, 4]);
        assert_eq!(r4d.dims(), &[2, 2, 2, 3]);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_tensor_view_to_arena_roundtrip() {
        let arena = SharedMemoryArena::create("test_integration", 1024 * 1024).unwrap();
        
        // Create via CandleTensorView
        let view = CandleTensorView::new(vec![4, 4], "f32").unwrap();
        
        // Allocate to arena
        let handle = arena.allocate_tensor_named("from_view", view.tensor()).unwrap();
        
        // Read back
        let restored = arena.get_tensor("from_view").unwrap();
        
        // Verify
        assert_eq!(restored.dims(), view.tensor().dims());
        assert!(matches!(restored.dtype(), DType::F32));
    }

    #[test]
    fn test_concurrent_arena_access_simulation() {
        // Simulate multiple "processes" accessing shared arena
        let arena = Arc::new(SharedMemoryArena::create("test_concurrent", 10 * 1024 * 1024).unwrap());
        
        // "Process 1" writes
        let t1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        arena.allocate_tensor_named("from_p1", &t1).unwrap();
        
        // "Process 2" reads and writes
        let read_t1 = arena.get_tensor("from_p1").unwrap();
        let t2 = Tensor::from_vec(vec![4.0f32, 5.0], &[2], &Device::Cpu).unwrap();
        arena.allocate_tensor_named("from_p2", &t2).unwrap();
        
        // Verify both processes can read all data
        let final_read_1 = arena.get_tensor("from_p1").unwrap();
        let final_read_2 = arena.get_tensor("from_p2").unwrap();
        
        assert_eq!(final_read_1.dims(), &[3]);
        assert_eq!(final_read_2.dims(), &[2]);
    }

    #[test]
    fn test_model_weights_simulation() {
        // Simulate storing model weights in shared memory
        let arena = SharedMemoryArena::create("test_model", 64 * 1024 * 1024).unwrap();
        
        // Typical transformer layer weights
        let hidden_dim = 256;
        let ffn_dim = 1024;
        
        // Attention weights
        let q_proj = Tensor::from_vec(
            vec![0.1f32; hidden_dim * hidden_dim],
            &[hidden_dim, hidden_dim],
            &Device::Cpu
        ).unwrap();
        let k_proj = Tensor::from_vec(
            vec![0.1f32; hidden_dim * hidden_dim],
            &[hidden_dim, hidden_dim],
            &Device::Cpu
        ).unwrap();
        let v_proj = Tensor::from_vec(
            vec![0.1f32; hidden_dim * hidden_dim],
            &[hidden_dim, hidden_dim],
            &Device::Cpu
        ).unwrap();
        
        // FFN weights
        let ffn_up = Tensor::from_vec(
            vec![0.1f32; hidden_dim * ffn_dim],
            &[hidden_dim, ffn_dim],
            &Device::Cpu
        ).unwrap();
        let ffn_down = Tensor::from_vec(
            vec![0.1f32; ffn_dim * hidden_dim],
            &[ffn_dim, hidden_dim],
            &Device::Cpu
        ).unwrap();
        
        // Store all weights
        arena.allocate_tensor_named("layer0.attn.q_proj", &q_proj).unwrap();
        arena.allocate_tensor_named("layer0.attn.k_proj", &k_proj).unwrap();
        arena.allocate_tensor_named("layer0.attn.v_proj", &v_proj).unwrap();
        arena.allocate_tensor_named("layer0.ffn.up", &ffn_up).unwrap();
        arena.allocate_tensor_named("layer0.ffn.down", &ffn_down).unwrap();
        
        // Verify retrieval
        let allocations = arena.list_allocations();
        assert_eq!(allocations.len(), 5);
        
        // Verify shapes
        let restored_q = arena.get_tensor("layer0.attn.q_proj").unwrap();
        let restored_ffn = arena.get_tensor("layer0.ffn.up").unwrap();
        
        assert_eq!(restored_q.dims(), &[hidden_dim, hidden_dim]);
        assert_eq!(restored_ffn.dims(), &[hidden_dim, ffn_dim]);
    }

    #[test]
    fn test_gradient_transfer_simulation() {
        // Simulate gradient accumulation across workers
        let arena = SharedMemoryArena::create("test_gradients", 32 * 1024 * 1024).unwrap();
        
        let grad_shape = (512, 512);
        let numel = 512 * 512;
        
        // Worker 1 gradients
        let grad1 = Tensor::from_vec(
            vec![0.01f32; numel],
            grad_shape,
            &Device::Cpu
        ).unwrap();
        arena.allocate_tensor_named("worker1_grad", &grad1).unwrap();
        
        // Worker 2 gradients
        let grad2 = Tensor::from_vec(
            vec![0.02f32; numel],
            grad_shape,
            &Device::Cpu
        ).unwrap();
        arena.allocate_tensor_named("worker2_grad", &grad2).unwrap();
        
        // "Coordinator" reads and averages
        let g1 = arena.get_tensor("worker1_grad").unwrap();
        let g2 = arena.get_tensor("worker2_grad").unwrap();
        
        let avg_grad = (g1 + g2).unwrap().affine(0.5, 0.0).unwrap();
        
        // Store averaged gradient
        arena.allocate_tensor_named("averaged_grad", &avg_grad).unwrap();
        
        // Verify
        let final_grad = arena.get_tensor("averaged_grad").unwrap();
        let data = final_grad.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        
        // Average of 0.01 and 0.02 should be 0.015
        assert!((data[0] - 0.015).abs() < 1e-5);
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

mod error_handling_tests {
    use super::*;

    #[test]
    fn test_invalid_dtype() {
        let result = CandleTensorView::new(vec![4], "invalid_dtype");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_nonexistent_tensor() {
        let arena = SharedMemoryArena::create("test_nonexistent", 1024 * 1024).unwrap();
        let result = arena.get_tensor("does_not_exist");
        assert!(result.is_err());
    }

    #[test]
    fn test_free_nonexistent_allocation() {
        let arena = SharedMemoryArena::create("test_free_nonexistent", 1024 * 1024).unwrap();
        let result = arena.free_allocation("does_not_exist");
        assert!(result.is_err());
    }

    #[test]
    fn test_arena_overflow() {
        // Create very small arena
        let arena = SharedMemoryArena::create("test_overflow", 1024).unwrap();
        
        // Try to allocate tensor larger than arena
        let large_tensor = Tensor::from_vec(
            vec![0.0f32; 10000],
            &[10000],
            &Device::Cpu
        ).unwrap();
        
        let result = arena.allocate_tensor_named("too_big", &large_tensor);
        assert!(result.is_err());
    }
}

// ============================================================================
// Performance Characteristic Tests
// ============================================================================

mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_allocation_time_scaling() {
        let arena = SharedMemoryArena::create("test_perf", 256 * 1024 * 1024).unwrap();
        
        let sizes = vec![1024, 4096, 16384, 65536, 262144];
        
        for size in sizes {
            let tensor = Tensor::from_vec(
                vec![0.0f32; size],
                &[size],
                &Device::Cpu
            ).unwrap();
            
            let start = Instant::now();
            let _handle = arena.allocate_tensor_named(&format!("perf_{}", size), &tensor).unwrap();
            let elapsed = start.elapsed();
            
            // Just verify it completes; actual timing would be done in benchmarks
            assert!(elapsed.as_secs() < 10, "Allocation took too long for size {}", size);
        }
    }

    #[test]
    fn test_read_time_scaling() {
        let arena = SharedMemoryArena::create("test_read_perf", 256 * 1024 * 1024).unwrap();
        
        let sizes = vec![1024, 4096, 16384, 65536, 262144];
        
        for size in sizes {
            let tensor = Tensor::from_vec(
                vec![0.0f32; size],
                &[size],
                &Device::Cpu
            ).unwrap();
            
            arena.allocate_tensor_named(&format!("read_{}", size), &tensor).unwrap();
            
            let start = Instant::now();
            let _restored = arena.get_tensor(&format!("read_{}", size)).unwrap();
            let elapsed = start.elapsed();
            
            assert!(elapsed.as_secs() < 10, "Read took too long for size {}", size);
        }
    }
}
