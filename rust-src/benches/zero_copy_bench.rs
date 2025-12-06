//! Benchmarks for Zero-Copy vs Copy Tensor Transfer
//!
//! This benchmark suite compares different tensor transfer strategies:
//! 1. Standard copy transfer (serialize/deserialize)
//! 2. Arrow IPC transfer (columnar format)
//! 3. Shared memory transfer (mmap-based)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use candle_core::{Device, Tensor};
use std::time::Duration;

// Import our zero-copy modules
use deepseek_rust::pyo3_bindings::{
    arrow_interop::ArrowTensorInterop,
    shared_memory::SharedMemoryArena,
    tensor_view::CandleTensorView,
};

/// Create a random tensor of given size
fn create_test_tensor(size: usize) -> Tensor {
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
    Tensor::from_vec(data, &[size], &Device::Cpu).unwrap()
}

/// Create a 2D tensor for matrix operations
fn create_2d_tensor(rows: usize, cols: usize) -> Tensor {
    let size = rows * cols;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
    Tensor::from_vec(data, &[rows, cols], &Device::Cpu).unwrap()
}

/// Benchmark standard copy transfer (simulates serialization)
fn bench_copy_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy_transfer");
    
    for size in [1024, 4096, 16384, 65536, 262144, 1048576].iter() {
        let tensor = create_test_tensor(*size);
        let bytes = size * std::mem::size_of::<f32>();
        
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(BenchmarkId::new("copy", size), size, |b, _| {
            b.iter(|| {
                // Simulate copy: extract data and recreate tensor
                let data = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let shape = tensor.dims().to_vec();
                let restored = Tensor::from_vec(data, &shape, &Device::Cpu).unwrap();
                black_box(restored)
            })
        });
    }
    
    group.finish();
}

/// Benchmark Arrow IPC transfer
fn bench_arrow_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("arrow_transfer");
    
    for size in [1024, 4096, 16384, 65536, 262144, 1048576].iter() {
        let tensor = create_test_tensor(*size);
        let view = CandleTensorView::from_tensor(tensor.clone());
        let bytes = size * std::mem::size_of::<f32>();
        
        group.throughput(Throughput::Bytes(bytes as u64));
        
        // Benchmark serialization
        group.bench_with_input(BenchmarkId::new("serialize", size), size, |b, _| {
            b.iter(|| {
                let serialized = deepseek_rust::pyo3_bindings::arrow_interop::serialize_tensor_to_arrow_internal(&tensor).unwrap();
                black_box(serialized)
            })
        });
        
        // Pre-serialize for deserialization benchmark
        let serialized = deepseek_rust::pyo3_bindings::arrow_interop::serialize_tensor_to_arrow_internal(&tensor).unwrap();
        
        // Benchmark deserialization
        group.bench_with_input(BenchmarkId::new("deserialize", size), size, |b, _| {
            b.iter(|| {
                let restored = deepseek_rust::pyo3_bindings::arrow_interop::deserialize_tensor_from_arrow_internal(&serialized).unwrap();
                black_box(restored)
            })
        });
        
        // Benchmark round-trip
        group.bench_with_input(BenchmarkId::new("roundtrip", size), size, |b, _| {
            b.iter(|| {
                let serialized = deepseek_rust::pyo3_bindings::arrow_interop::serialize_tensor_to_arrow_internal(&tensor).unwrap();
                let restored = deepseek_rust::pyo3_bindings::arrow_interop::deserialize_tensor_from_arrow_internal(&serialized).unwrap();
                black_box(restored)
            })
        });
    }
    
    group.finish();
}

/// Benchmark shared memory transfer
fn bench_shared_memory_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("shared_memory_transfer");
    
    // Create arena large enough for largest tensor
    let arena = SharedMemoryArena::create("bench_arena", 64 * 1024 * 1024).unwrap();
    
    for size in [1024, 4096, 16384, 65536, 262144, 1048576].iter() {
        let tensor = create_test_tensor(*size);
        let bytes = size * std::mem::size_of::<f32>();
        
        group.throughput(Throughput::Bytes(bytes as u64));
        
        // Benchmark write
        group.bench_with_input(BenchmarkId::new("write", size), size, |b, _| {
            arena.reset();
            b.iter(|| {
                let handle = arena.allocate_tensor_named(&format!("t_{}", size), &tensor).unwrap();
                black_box(handle)
            })
        });
        
        // Setup for read benchmark
        arena.reset();
        let handle = arena.allocate_tensor_named(&format!("read_test_{}", size), &tensor).unwrap();
        
        // Benchmark read
        group.bench_with_input(BenchmarkId::new("read", size), size, |b, _| {
            b.iter(|| {
                let restored = arena.read_tensor(&handle).unwrap();
                black_box(restored)
            })
        });
        
        // Benchmark round-trip
        group.bench_with_input(BenchmarkId::new("roundtrip", size), size, |b, _| {
            let name = format!("rt_{}", rand::random::<u32>());
            arena.reset();
            b.iter(|| {
                let handle = arena.allocate_tensor_named(&name, &tensor).unwrap();
                let restored = arena.read_tensor(&handle).unwrap();
                black_box(restored)
            })
        });
    }
    
    group.finish();
}

/// Benchmark batch tensor transfer
fn bench_batch_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_transfer");
    
    // Create a batch of tensors (simulating model weights)
    let batch_sizes = [4, 16, 64];
    let tensor_size = 4096;
    
    for batch_size in batch_sizes {
        let tensors: Vec<Tensor> = (0..batch_size)
            .map(|_| create_test_tensor(tensor_size))
            .collect();
        
        let views: Vec<CandleTensorView> = tensors.iter()
            .map(|t| CandleTensorView::from_tensor(t.clone()))
            .collect();
        
        let names: Vec<String> = (0..batch_size)
            .map(|i| format!("weight_{}", i))
            .collect();
        
        let bytes = batch_size * tensor_size * std::mem::size_of::<f32>();
        
        group.throughput(Throughput::Bytes(bytes as u64));
        
        // Copy transfer (individual tensors)
        group.bench_with_input(BenchmarkId::new("copy_individual", batch_size), &tensors, |b, tensors| {
            b.iter(|| {
                let restored: Vec<Tensor> = tensors.iter()
                    .map(|t| {
                        let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                        let shape = t.dims().to_vec();
                        Tensor::from_vec(data, &shape, &Device::Cpu).unwrap()
                    })
                    .collect();
                black_box(restored)
            })
        });
        
        // Arrow batch transfer
        group.bench_with_input(BenchmarkId::new("arrow_batch", batch_size), &(&views, &names), |b, (views, names)| {
            b.iter(|| {
                let serialized = deepseek_rust::pyo3_bindings::arrow_interop::serialize_tensor_batch_internal(views, names).unwrap();
                let restored = deepseek_rust::pyo3_bindings::arrow_interop::deserialize_tensor_batch_internal(&serialized).unwrap();
                black_box(restored)
            })
        });
    }
    
    group.finish();
}

/// Benchmark different data types
fn bench_dtype_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_transfer");
    let size = 65536;
    
    // f32 tensor
    let f32_tensor = create_test_tensor(size);
    let f32_bytes = size * 4;
    
    group.throughput(Throughput::Bytes(f32_bytes as u64));
    group.bench_function("f32", |b| {
        b.iter(|| {
            let data = f32_tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let shape = f32_tensor.dims().to_vec();
            let restored = Tensor::from_vec(data, &shape, &Device::Cpu).unwrap();
            black_box(restored)
        })
    });
    
    // f64 tensor
    let f64_data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.001).collect();
    let f64_tensor = Tensor::from_vec(f64_data, &[size], &Device::Cpu).unwrap();
    let f64_bytes = size * 8;
    
    group.throughput(Throughput::Bytes(f64_bytes as u64));
    group.bench_function("f64", |b| {
        b.iter(|| {
            let data = f64_tensor.flatten_all().unwrap().to_vec1::<f64>().unwrap();
            let shape = f64_tensor.dims().to_vec();
            let restored = Tensor::from_vec(data, &shape, &Device::Cpu).unwrap();
            black_box(restored)
        })
    });
    
    // i64 tensor
    let i64_data: Vec<i64> = (0..size).map(|i| i as i64).collect();
    let i64_tensor = Tensor::from_vec(i64_data, &[size], &Device::Cpu).unwrap();
    let i64_bytes = size * 8;
    
    group.throughput(Throughput::Bytes(i64_bytes as u64));
    group.bench_function("i64", |b| {
        b.iter(|| {
            let data = i64_tensor.flatten_all().unwrap().to_vec1::<i64>().unwrap();
            let shape = i64_tensor.dims().to_vec();
            let restored = Tensor::from_vec(data, &shape, &Device::Cpu).unwrap();
            black_box(restored)
        })
    });
    
    group.finish();
}

/// Benchmark matrix operations with transfer
fn bench_matmul_with_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_with_transfer");
    
    let sizes = [(64, 64), (256, 256), (512, 512), (1024, 1024)];
    
    for (m, n) in sizes {
        let a = create_2d_tensor(m, n);
        let b = create_2d_tensor(n, m);
        let bytes = 2 * m * n * std::mem::size_of::<f32>();
        
        group.throughput(Throughput::Bytes(bytes as u64));
        
        // Pure matmul (no transfer overhead)
        group.bench_with_input(BenchmarkId::new("pure_matmul", m), &(m, n), |bench, _| {
            bench.iter(|| {
                let result = a.matmul(&b).unwrap();
                black_box(result)
            })
        });
        
        // Matmul with copy transfer simulation
        group.bench_with_input(BenchmarkId::new("matmul_with_copy", m), &(m, n), |bench, _| {
            bench.iter(|| {
                // Simulate receiving tensors
                let a_data = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let b_data = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let a_restored = Tensor::from_vec(a_data, &[m, n], &Device::Cpu).unwrap();
                let b_restored = Tensor::from_vec(b_data, &[n, m], &Device::Cpu).unwrap();
                
                // Compute
                let result = a_restored.matmul(&b_restored).unwrap();
                
                // Simulate sending result
                let result_data = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                black_box(result_data)
            })
        });
    }
    
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(2));
    targets = 
        bench_copy_transfer,
        bench_dtype_transfer,
        bench_matmul_with_transfer
}

// Note: Arrow and shared memory benchmarks require internal functions to be public
// which would need additional module changes. These are included for completeness.
// To run full benchmarks, uncomment after exposing internal functions.
// criterion_group!(
//     full_benches,
//     bench_copy_transfer,
//     bench_arrow_transfer,
//     bench_shared_memory_transfer,
//     bench_batch_transfer,
//     bench_dtype_transfer,
//     bench_matmul_with_transfer
// );

criterion_main!(benches);
