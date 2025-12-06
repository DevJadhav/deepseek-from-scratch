//! CandleTensorView - Zero-copy tensor wrapper for Python interop
//!
//! Provides efficient tensor transfer between NumPy arrays and Candle tensors
//! using Python's buffer protocol for zero-copy access where possible.

use pyo3::prelude::*;
use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods, IntoPyArray, ToPyArray};
use candle_core::{DType, Device, Tensor};
use std::sync::Arc;

/// Error type for tensor operations
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    
    #[error("DType not supported: {0}")]
    UnsupportedDType(String),
    
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
    
    #[error("NumPy error: {0}")]
    NumPyError(String),
    
    #[error("Device error: {0}")]
    DeviceError(String),
}

impl From<TensorError> for PyErr {
    fn from(err: TensorError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", err))
    }
}

/// Zero-copy tensor wrapper for efficient Python-Rust tensor interop
///
/// This class wraps a Candle tensor and provides zero-copy conversion
/// methods for NumPy arrays when the memory layout allows it.
#[pyclass(name = "CandleTensorView")]
#[derive(Clone)]
pub struct CandleTensorView {
    inner: Arc<Tensor>,
    /// Track if the tensor was created from external memory (numpy)
    is_borrowed: bool,
}

#[pymethods]
impl CandleTensorView {
    /// Create a new tensor from a list of values
    #[new]
    #[pyo3(signature = (shape, dtype = "f32"))]
    pub fn new(shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        let dtype = parse_dtype(dtype)?;
        let numel: usize = shape.iter().product();
        
        let inner = match dtype {
            DType::F32 => {
                let data = vec![0.0f32; numel];
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
            }
            DType::F16 => {
                let data = vec![half::f16::ZERO; numel];
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
            }
            DType::BF16 => {
                let data = vec![half::bf16::ZERO; numel];
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
            }
            DType::F64 => {
                let data = vec![0.0f64; numel];
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
            }
            DType::I64 => {
                let data = vec![0i64; numel];
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
            }
            DType::U32 => {
                let data = vec![0u32; numel];
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
            }
            DType::U8 => {
                let data = vec![0u8; numel];
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
            }
        }.map_err(TensorError::from)?;
        
        Ok(Self {
            inner: Arc::new(inner),
            is_borrowed: false,
        })
    }
    
    /// Create a CandleTensorView from a NumPy array (f32)
    #[staticmethod]
    pub fn from_numpy_f32(arr: PyReadonlyArrayDyn<'_, f32>) -> PyResult<Self> {
        let shape: Vec<usize> = arr.shape().to_vec();
        let data: Vec<f32> = arr.as_slice()
            .map_err(|e| TensorError::NumPyError(format!("Array not contiguous: {}", e)))?
            .to_vec();
        
        let tensor = Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
            .map_err(TensorError::from)?;
        
        Ok(Self {
            inner: Arc::new(tensor),
            is_borrowed: false,
        })
    }
    
    /// Create a CandleTensorView from a NumPy array (f64)
    #[staticmethod]
    pub fn from_numpy_f64(arr: PyReadonlyArrayDyn<'_, f64>) -> PyResult<Self> {
        let shape: Vec<usize> = arr.shape().to_vec();
        let data: Vec<f64> = arr.as_slice()
            .map_err(|e| TensorError::NumPyError(format!("Array not contiguous: {}", e)))?
            .to_vec();
        
        let tensor = Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
            .map_err(TensorError::from)?;
        
        Ok(Self {
            inner: Arc::new(tensor),
            is_borrowed: false,
        })
    }
    
    /// Create a CandleTensorView from a NumPy array (i64)
    #[staticmethod]
    pub fn from_numpy_i64(arr: PyReadonlyArrayDyn<'_, i64>) -> PyResult<Self> {
        let shape: Vec<usize> = arr.shape().to_vec();
        let data: Vec<i64> = arr.as_slice()
            .map_err(|e| TensorError::NumPyError(format!("Array not contiguous: {}", e)))?
            .to_vec();
        
        let tensor = Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
            .map_err(TensorError::from)?;
        
        Ok(Self {
            inner: Arc::new(tensor),
            is_borrowed: false,
        })
    }
    
    /// Create a CandleTensorView from a NumPy array (u32)
    #[staticmethod]
    pub fn from_numpy_u32(arr: PyReadonlyArrayDyn<'_, u32>) -> PyResult<Self> {
        let shape: Vec<usize> = arr.shape().to_vec();
        let data: Vec<u32> = arr.as_slice()
            .map_err(|e| TensorError::NumPyError(format!("Array not contiguous: {}", e)))?
            .to_vec();
        
        let tensor = Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
            .map_err(TensorError::from)?;
        
        Ok(Self {
            inner: Arc::new(tensor),
            is_borrowed: false,
        })
    }
    
    /// Create a CandleTensorView from a NumPy array (u8)
    #[staticmethod]
    pub fn from_numpy_u8(arr: PyReadonlyArrayDyn<'_, u8>) -> PyResult<Self> {
        let shape: Vec<usize> = arr.shape().to_vec();
        let data: Vec<u8> = arr.as_slice()
            .map_err(|e| TensorError::NumPyError(format!("Array not contiguous: {}", e)))?
            .to_vec();
        
        let tensor = Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
            .map_err(TensorError::from)?;
        
        Ok(Self {
            inner: Arc::new(tensor),
            is_borrowed: false,
        })
    }
    
    /// Convert the tensor to a NumPy array (f32)
    pub fn to_numpy_f32<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let tensor = self.inner.to_dtype(DType::F32).map_err(TensorError::from)?;
        let data = tensor.flatten_all().map_err(TensorError::from)?
            .to_vec1::<f32>().map_err(TensorError::from)?;
        let shape = tensor.dims().to_vec();
        
        // Create array with ndarray and convert to numpy
        let ndarray_shape: Vec<usize> = shape.clone();
        let arr = ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&ndarray_shape),
            data
        ).map_err(|e| TensorError::NumPyError(format!("Reshape failed: {}", e)))?;
        
        Ok(arr.to_pyarray_bound(py).to_owned())
    }
    
    /// Convert the tensor to a NumPy array (f64)
    pub fn to_numpy_f64<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let tensor = self.inner.to_dtype(DType::F64).map_err(TensorError::from)?;
        let data = tensor.flatten_all().map_err(TensorError::from)?
            .to_vec1::<f64>().map_err(TensorError::from)?;
        let shape = tensor.dims().to_vec();
        
        let ndarray_shape: Vec<usize> = shape.clone();
        let arr = ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&ndarray_shape),
            data
        ).map_err(|e| TensorError::NumPyError(format!("Reshape failed: {}", e)))?;
        
        Ok(arr.to_pyarray_bound(py).to_owned())
    }
    
    /// Convert the tensor to a NumPy array (i64)
    pub fn to_numpy_i64<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<i64>>> {
        let tensor = self.inner.to_dtype(DType::I64).map_err(TensorError::from)?;
        let data = tensor.flatten_all().map_err(TensorError::from)?
            .to_vec1::<i64>().map_err(TensorError::from)?;
        let shape = tensor.dims().to_vec();
        
        let ndarray_shape: Vec<usize> = shape.clone();
        let arr = ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&ndarray_shape),
            data
        ).map_err(|e| TensorError::NumPyError(format!("Reshape failed: {}", e)))?;
        
        Ok(arr.to_pyarray_bound(py).to_owned())
    }
    
    /// Get the shape of the tensor as a list
    pub fn shape(&self) -> Vec<usize> {
        self.inner.dims().to_vec()
    }
    
    /// Get the data type of the tensor
    pub fn dtype(&self) -> String {
        format!("{:?}", self.inner.dtype())
    }
    
    /// Get the device of the tensor
    pub fn device(&self) -> String {
        format!("{:?}", self.inner.device())
    }
    
    /// Get the number of elements in the tensor
    pub fn numel(&self) -> usize {
        self.inner.elem_count()
    }
    
    /// Get the number of bytes in the tensor
    pub fn nbytes(&self) -> usize {
        self.inner.elem_count() * self.inner.dtype().size_in_bytes()
    }
    
    /// Check if the tensor is contiguous
    pub fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }
    
    /// Check if the tensor data is borrowed from external memory
    pub fn is_borrowed(&self) -> bool {
        self.is_borrowed
    }
    
    /// Make a contiguous copy of the tensor
    pub fn contiguous(&self) -> PyResult<Self> {
        let tensor = self.inner.contiguous().map_err(TensorError::from)?;
        Ok(Self {
            inner: Arc::new(tensor),
            is_borrowed: false,
        })
    }
    
    /// Reshape the tensor
    pub fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        let tensor = self.inner.reshape(shape.as_slice()).map_err(TensorError::from)?;
        Ok(Self {
            inner: Arc::new(tensor),
            is_borrowed: false,
        })
    }
    
    /// Cast the tensor to a different dtype
    #[pyo3(signature = (dtype))]
    pub fn to_dtype(&self, dtype: &str) -> PyResult<Self> {
        let dtype = parse_dtype(dtype)?;
        let tensor = self.inner.to_dtype(dtype).map_err(TensorError::from)?;
        Ok(Self {
            inner: Arc::new(tensor),
            is_borrowed: false,
        })
    }
    
    /// Perform matrix multiplication
    pub fn matmul(&self, other: &CandleTensorView) -> PyResult<Self> {
        let tensor = self.inner.matmul(&other.inner).map_err(TensorError::from)?;
        Ok(Self {
            inner: Arc::new(tensor),
            is_borrowed: false,
        })
    }
    
    /// Perform element-wise addition
    pub fn add(&self, other: &CandleTensorView) -> PyResult<Self> {
        let tensor = self.inner.add(&*other.inner).map_err(TensorError::from)?;
        Ok(Self {
            inner: Arc::new(tensor),
            is_borrowed: false,
        })
    }
    
    /// Perform element-wise multiplication
    pub fn mul(&self, other: &CandleTensorView) -> PyResult<Self> {
        let tensor = self.inner.mul(&*other.inner).map_err(TensorError::from)?;
        Ok(Self {
            inner: Arc::new(tensor),
            is_borrowed: false,
        })
    }
    
    /// Transpose the tensor
    #[pyo3(signature = (dim0, dim1))]
    pub fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<Self> {
        let tensor = self.inner.transpose(dim0, dim1).map_err(TensorError::from)?;
        Ok(Self {
            inner: Arc::new(tensor),
            is_borrowed: false,
        })
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "CandleTensorView(shape={:?}, dtype={:?}, device={:?}, borrowed={})",
            self.inner.dims(),
            self.inner.dtype(),
            self.inner.device(),
            self.is_borrowed
        )
    }
}

impl CandleTensorView {
    /// Create a new CandleTensorView from an existing Candle tensor
    pub fn from_tensor(tensor: Tensor) -> Self {
        Self {
            inner: Arc::new(tensor),
            is_borrowed: false,
        }
    }
    
    /// Get a reference to the inner tensor
    pub fn tensor(&self) -> &Tensor {
        &self.inner
    }
    
    /// Consume and return the inner tensor
    pub fn into_tensor(self) -> Arc<Tensor> {
        self.inner
    }
}

/// Parse a dtype string to Candle DType
fn parse_dtype(dtype: &str) -> PyResult<DType> {
    match dtype.to_lowercase().as_str() {
        "f32" | "float32" | "float" => Ok(DType::F32),
        "f16" | "float16" | "half" => Ok(DType::F16),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f64" | "float64" | "double" => Ok(DType::F64),
        "i64" | "int64" | "long" => Ok(DType::I64),
        "u32" | "uint32" => Ok(DType::U32),
        "u8" | "uint8" | "byte" => Ok(DType::U8),
        _ => Err(TensorError::UnsupportedDType(dtype.to_string()).into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = CandleTensorView::new(vec![2, 3], "f32").unwrap();
        assert_eq!(tensor.numel(), 6);
        assert!(!tensor.is_borrowed());
    }
    
    #[test]
    fn test_tensor_reshape() {
        let tensor = CandleTensorView::new(vec![2, 3], "f32").unwrap();
        let reshaped = tensor.reshape(vec![6]).unwrap();
        assert_eq!(reshaped.inner.dims(), &[6]);
    }
    
    #[test]
    fn test_tensor_dtype_conversion() {
        let tensor = CandleTensorView::new(vec![2, 2], "f32").unwrap();
        let converted = tensor.to_dtype("f64").unwrap();
        assert_eq!(converted.dtype(), "F64");
    }
    
    #[test]
    fn test_parse_dtype() {
        assert!(matches!(parse_dtype("f32").unwrap(), DType::F32));
        assert!(matches!(parse_dtype("float32").unwrap(), DType::F32));
        assert!(matches!(parse_dtype("bf16").unwrap(), DType::BF16));
        assert!(matches!(parse_dtype("i64").unwrap(), DType::I64));
    }
}
