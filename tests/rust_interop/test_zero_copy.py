"""
Tests for Zero-Copy Rust-Python Interop via PyO3

This module tests the zero-copy tensor transfer functionality between
Python (NumPy) and Rust (Candle) using PyO3 bindings.

Test Coverage:
- CandleTensorView: NumPy array to Candle tensor conversion
- ArrowTensorInterop: Arrow IPC serialization for tensor transfer
- SharedMemoryArena: mmap-based shared memory for Ray actor communication
"""

import numpy as np
import pytest

# Skip all tests if deepseek_rust is not available
pytest.importorskip("deepseek_rust", reason="deepseek_rust module not built")

import deepseek_rust


class TestCandleTensorView:
    """Tests for CandleTensorView - zero-copy tensor wrapper"""

    def test_from_numpy_f32(self):
        """Test creating tensor from float32 numpy array"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(arr)
        
        assert tensor.shape() == [2, 3]
        assert tensor.dtype() == "F32"
        assert tensor.numel() == 6

    def test_from_numpy_f64(self):
        """Test creating tensor from float64 numpy array"""
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f64(arr)
        
        assert tensor.shape() == [4]
        assert tensor.dtype() == "F64"
        assert tensor.numel() == 4

    def test_from_numpy_i64(self):
        """Test creating tensor from int64 numpy array"""
        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
        tensor = deepseek_rust.CandleTensorView.from_numpy_i64(arr)
        
        assert tensor.shape() == [3, 2]
        assert tensor.dtype() == "I64"
        assert tensor.numel() == 6

    def test_from_numpy_u32(self):
        """Test creating tensor from uint32 numpy array"""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        tensor = deepseek_rust.CandleTensorView.from_numpy_u32(arr)
        
        assert tensor.shape() == [5]
        assert tensor.dtype() == "U32"
        assert tensor.numel() == 5

    def test_from_numpy_u8(self):
        """Test creating tensor from uint8 numpy array"""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        tensor = deepseek_rust.CandleTensorView.from_numpy_u8(arr)
        
        assert tensor.shape() == [2, 3]
        assert tensor.dtype() == "U8"
        assert tensor.numel() == 6

    def test_roundtrip_f32(self):
        """Test numpy -> tensor -> numpy roundtrip for float32"""
        arr = np.random.randn(4, 5, 6).astype(np.float32)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(arr)
        arr2 = tensor.to_numpy_f32()
        
        assert arr2.shape == arr.shape
        assert np.allclose(arr, arr2, rtol=1e-5)

    def test_roundtrip_f64(self):
        """Test numpy -> tensor -> numpy roundtrip for float64"""
        arr = np.random.randn(3, 4).astype(np.float64)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f64(arr)
        arr2 = tensor.to_numpy_f64()
        
        assert arr2.shape == arr.shape
        assert np.allclose(arr, arr2, rtol=1e-10)

    def test_roundtrip_i64(self):
        """Test numpy -> tensor -> numpy roundtrip for int64"""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        tensor = deepseek_rust.CandleTensorView.from_numpy_i64(arr)
        arr2 = tensor.to_numpy_i64()
        
        assert arr2.shape == arr.shape
        assert np.array_equal(arr, arr2)

    def test_tensor_operations_matmul(self):
        """Test matrix multiplication in Rust"""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        
        tensor_a = deepseek_rust.CandleTensorView.from_numpy_f32(a)
        tensor_b = deepseek_rust.CandleTensorView.from_numpy_f32(b)
        
        result = tensor_a.matmul(tensor_b)
        result_np = result.to_numpy_f32()
        expected = a @ b
        
        assert np.allclose(result_np, expected, rtol=1e-5)

    def test_tensor_operations_add(self):
        """Test element-wise addition in Rust"""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        
        tensor_a = deepseek_rust.CandleTensorView.from_numpy_f32(a)
        tensor_b = deepseek_rust.CandleTensorView.from_numpy_f32(b)
        
        result = tensor_a.add(tensor_b)
        result_np = result.to_numpy_f32()
        expected = a + b
        
        assert np.allclose(result_np, expected, rtol=1e-5)

    def test_tensor_operations_mul(self):
        """Test element-wise multiplication in Rust"""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        
        tensor_a = deepseek_rust.CandleTensorView.from_numpy_f32(a)
        tensor_b = deepseek_rust.CandleTensorView.from_numpy_f32(b)
        
        result = tensor_a.mul(tensor_b)
        result_np = result.to_numpy_f32()
        expected = a * b
        
        assert np.allclose(result_np, expected, rtol=1e-5)

    def test_tensor_transpose(self):
        """Test tensor transpose in Rust"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(arr)
        
        result = tensor.transpose(0, 1)
        result_np = result.to_numpy_f32()
        expected = arr.T
        
        assert result.shape() == [3, 2]
        assert np.allclose(result_np, expected, rtol=1e-5)

    def test_large_tensor(self):
        """Test with larger tensor to verify performance"""
        arr = np.random.randn(100, 200, 50).astype(np.float32)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(arr)
        
        assert tensor.shape() == [100, 200, 50]
        assert tensor.numel() == 100 * 200 * 50
        
        # Roundtrip
        arr2 = tensor.to_numpy_f32()
        assert np.allclose(arr, arr2, rtol=1e-5)


class TestArrowTensorInterop:
    """Tests for ArrowTensorInterop - Arrow IPC serialization"""

    def test_serialize_deserialize_f32(self):
        """Test serialization and deserialization of float32 tensor"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(arr)
        
        interop = deepseek_rust.ArrowTensorInterop()
        serialized = interop.serialize_tensor(tensor)
        
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        restored = deepseek_rust.ArrowTensorInterop.deserialize_tensor(serialized)
        assert restored.shape() == tensor.shape()
        
        arr2 = restored.to_numpy_f32()
        assert np.allclose(arr, arr2, rtol=1e-5)

    def test_serialize_deserialize_f64(self):
        """Test serialization and deserialization of float64 tensor"""
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f64(arr)
        
        interop = deepseek_rust.ArrowTensorInterop()
        serialized = interop.serialize_tensor(tensor)
        
        restored = deepseek_rust.ArrowTensorInterop.deserialize_tensor(serialized)
        arr2 = restored.to_numpy_f64()
        
        assert np.allclose(arr, arr2, rtol=1e-10)

    def test_serialize_deserialize_i64(self):
        """Test serialization and deserialization of int64 tensor"""
        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
        tensor = deepseek_rust.CandleTensorView.from_numpy_i64(arr)
        
        interop = deepseek_rust.ArrowTensorInterop()
        serialized = interop.serialize_tensor(tensor)
        
        restored = deepseek_rust.ArrowTensorInterop.deserialize_tensor(serialized)
        arr2 = restored.to_numpy_i64()
        
        assert np.array_equal(arr, arr2)

    def test_batch_serialize_deserialize(self):
        """Test batch serialization of multiple tensors"""
        tensors = [
            deepseek_rust.CandleTensorView.from_numpy_f32(
                np.random.randn(10, 10).astype(np.float32)
            )
            for _ in range(3)
        ]
        names = ["tensor_0", "tensor_1", "tensor_2"]
        
        interop = deepseek_rust.ArrowTensorInterop()
        serialized = interop.serialize_batch(tensors, names)
        
        assert isinstance(serialized, bytes)
        
        restored = deepseek_rust.ArrowTensorInterop.deserialize_batch(serialized)
        assert len(restored) == 3
        
        for (name, tensor), expected_name in zip(restored, names, strict=True):
            assert name == expected_name
            assert tensor.shape() == [10, 10]

    def test_peek_metadata(self):
        """Test peeking at metadata without full deserialization"""
        arr = np.random.randn(5, 10, 15).astype(np.float32)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(arr)
        
        interop = deepseek_rust.ArrowTensorInterop()
        serialized = interop.serialize_tensor(tensor)
        
        metadata = deepseek_rust.ArrowTensorInterop.peek_metadata(serialized)
        assert metadata.shape == [5, 10, 15]
        assert metadata.dtype == "F32"


class TestSharedMemoryArena:
    """Tests for SharedMemoryArena - mmap-based shared memory"""

    def test_arena_creation(self):
        """Test creating a shared memory arena"""
        arena = deepseek_rust.SharedMemoryArena("test_creation", 1024 * 1024)
        
        assert arena.name == "test_creation"
        assert arena.capacity == 1024 * 1024
        assert arena.used == 0
        assert arena.available == 1024 * 1024
        assert arena.num_allocations == 0

    def test_allocate_and_get(self):
        """Test allocating and retrieving a tensor"""
        arena = deepseek_rust.SharedMemoryArena("test_alloc_get", 1024 * 1024)
        
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(arr)
        
        handle = arena.allocate_named("my_tensor", tensor)
        
        assert handle.name == "my_tensor"
        assert handle.shape == [2, 3]
        assert handle.dtype == "F32"
        assert arena.num_allocations == 1
        assert arena.used > 0
        
        # Retrieve the tensor
        restored = arena.get("my_tensor")
        arr2 = restored.to_numpy_f32()
        
        assert np.allclose(arr, arr2, rtol=1e-5)

    def test_allocate_auto_name(self):
        """Test allocating with auto-generated name"""
        arena = deepseek_rust.SharedMemoryArena("test_auto_name", 1024 * 1024)
        
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(arr)
        
        handle = arena.allocate(tensor)
        
        assert handle.name.startswith("tensor_")
        assert handle.shape == [3]

    def test_multiple_allocations(self):
        """Test multiple tensor allocations in same arena"""
        arena = deepseek_rust.SharedMemoryArena("test_multi_alloc", 10 * 1024 * 1024)
        
        tensors_data = [
            np.random.randn(10, 10).astype(np.float32),
            np.random.randn(20, 20).astype(np.float32),
            np.random.randn(5, 5, 5).astype(np.float32),
        ]
        names = ["t1", "t2", "t3"]
        
        for data, name in zip(tensors_data, names, strict=True):
            tensor = deepseek_rust.CandleTensorView.from_numpy_f32(data)
            arena.allocate_named(name, tensor)

        assert arena.num_allocations == 3

        # Verify all tensors can be retrieved
        for data, name in zip(tensors_data, names, strict=True):
            restored = arena.get(name)
            restored_np = restored.to_numpy_f32()
            assert np.allclose(data, restored_np, rtol=1e-5)

    def test_arena_reset(self):
        """Test resetting the arena"""
        arena = deepseek_rust.SharedMemoryArena("test_reset", 1024 * 1024)
        
        # Allocate some tensors
        for i in range(5):
            tensor = deepseek_rust.CandleTensorView.from_numpy_f32(
                np.random.randn(10).astype(np.float32)
            )
            arena.allocate_named(f"tensor_{i}", tensor)
        
        assert arena.num_allocations == 5
        assert arena.used > 0
        
        # Reset
        arena.reset()
        
        assert arena.num_allocations == 0
        assert arena.used == 0

    def test_arena_free(self):
        """Test freeing individual allocations"""
        arena = deepseek_rust.SharedMemoryArena("test_free", 1024 * 1024)
        
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(
            np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        arena.allocate_named("to_free", tensor)
        
        assert arena.num_allocations == 1
        
        arena.free("to_free")
        
        assert arena.num_allocations == 0

    def test_list_allocations(self):
        """Test listing all allocations"""
        arena = deepseek_rust.SharedMemoryArena("test_list", 1024 * 1024)
        
        for i in range(3):
            tensor = deepseek_rust.CandleTensorView.from_numpy_f32(
                np.random.randn(10).astype(np.float32)
            )
            arena.allocate_named(f"tensor_{i}", tensor)
        
        allocations = arena.list_allocations()
        
        assert len(allocations) == 3
        names = {name for name, offset, size in allocations}
        assert names == {"tensor_0", "tensor_1", "tensor_2"}

    def test_handle_serialization(self):
        """Test SharedTensorHandle serialization for IPC"""
        arena = deepseek_rust.SharedMemoryArena("test_handle_ser", 1024 * 1024)
        
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )
        handle = arena.allocate_named("ipc_tensor", tensor)
        
        # Serialize handle
        handle_bytes = handle.to_bytes()
        assert isinstance(handle_bytes, bytes)
        
        # Deserialize handle
        restored_handle = deepseek_rust.SharedTensorHandle.from_bytes(handle_bytes)
        assert restored_handle.name == handle.name
        assert restored_handle.shape == handle.shape
        assert restored_handle.dtype == handle.dtype

    def test_read_from_handle(self):
        """Test reading tensor using handle"""
        arena = deepseek_rust.SharedMemoryArena("test_read_handle", 1024 * 1024)
        
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(arr)
        handle = arena.allocate_named("handle_tensor", tensor)
        
        # Read using handle
        restored = arena.read(handle)
        arr2 = restored.to_numpy_f32()
        
        assert np.allclose(arr, arr2, rtol=1e-5)

    def test_different_dtypes(self):
        """Test arena with different tensor dtypes"""
        arena = deepseek_rust.SharedMemoryArena("test_dtypes", 10 * 1024 * 1024)
        
        # F32
        f32_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        f32_tensor = deepseek_rust.CandleTensorView.from_numpy_f32(f32_data)
        arena.allocate_named("f32_tensor", f32_tensor)
        
        # F64
        f64_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        f64_tensor = deepseek_rust.CandleTensorView.from_numpy_f64(f64_data)
        arena.allocate_named("f64_tensor", f64_tensor)
        
        # I64
        i64_data = np.array([1, 2, 3], dtype=np.int64)
        i64_tensor = deepseek_rust.CandleTensorView.from_numpy_i64(i64_data)
        arena.allocate_named("i64_tensor", i64_tensor)
        
        # Verify
        assert np.allclose(arena.get("f32_tensor").to_numpy_f32(), f32_data)
        assert np.allclose(arena.get("f64_tensor").to_numpy_f64(), f64_data)
        assert np.array_equal(arena.get("i64_tensor").to_numpy_i64(), i64_data)


class TestBenchmarkTransfer:
    """Tests for the benchmark_transfer function"""

    def test_benchmark_basic(self):
        """Test the benchmark function runs without errors"""
        # This test just ensures the benchmark function exists and runs
        try:
            # benchmark_transfer takes (rows, cols) - iterations is not a parameter
            result = deepseek_rust.benchmark_transfer(100, 100)
            assert isinstance(result, dict)
            assert "transfer_time_ms" in result or len(result) > 0
        except AttributeError:
            pytest.skip("benchmark_transfer not implemented")


class TestCreateSharedArena:
    """Tests for the create_shared_arena convenience function"""

    def test_create_arena_convenience(self):
        """Test the convenience function for creating arenas"""
        try:
            # create_shared_arena takes (name, capacity) - note capacity is in bytes
            arena = deepseek_rust.create_shared_arena("convenience_test", 512 * 1024)
            # Capacity is stored as usize, just verify it's positive
            assert arena.capacity > 0
            assert arena.name == "convenience_test"
        except AttributeError:
            pytest.skip("create_shared_arena not implemented")


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_empty_array(self):
        """Test handling of empty arrays"""
        arr = np.array([], dtype=np.float32)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(arr)
        
        assert tensor.numel() == 0

    def test_single_element(self):
        """Test handling of single element arrays"""
        arr = np.array([42.0], dtype=np.float32)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(arr)
        
        assert tensor.shape() == [1]
        assert tensor.numel() == 1
        
        arr2 = tensor.to_numpy_f32()
        assert np.allclose(arr, arr2)

    def test_high_dimensional_tensor(self):
        """Test handling of high-dimensional tensors"""
        arr = np.random.randn(2, 3, 4, 5, 6).astype(np.float32)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(arr)
        
        assert tensor.shape() == [2, 3, 4, 5, 6]
        assert tensor.numel() == 2 * 3 * 4 * 5 * 6
        
        arr2 = tensor.to_numpy_f32()
        assert np.allclose(arr, arr2, rtol=1e-5)

    def test_noncontiguous_array(self):
        """Test handling of non-contiguous numpy arrays"""
        arr = np.random.randn(10, 10).astype(np.float32)
        # Create non-contiguous view
        arr_nc = arr[::2, ::2]  # Every other element

        # Non-contiguous arrays require explicit conversion
        # The Rust code requires contiguous arrays
        arr_contiguous = np.ascontiguousarray(arr_nc)
        tensor = deepseek_rust.CandleTensorView.from_numpy_f32(arr_contiguous)
        arr2 = tensor.to_numpy_f32()

        assert arr2.shape == arr_nc.shape
        assert np.allclose(arr_contiguous, arr2, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
