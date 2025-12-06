//! Arrow IPC Interop for efficient tensor serialization
//!
//! This module provides efficient tensor serialization using Apache Arrow's
//! IPC format, enabling zero-copy reads and efficient batch transfers.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use arrow::array::{Float32Array, Float64Array, Int64Array, UInt32Array, UInt8Array, ArrayRef};
use arrow::datatypes::{Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::ipc::reader::StreamReader;
use arrow::record_batch::RecordBatch;
use candle_core::{DType, Device, Tensor};
use std::io::Cursor;
use std::sync::Arc;

use super::tensor_view::CandleTensorView;

/// Error type for Arrow operations
#[derive(Debug, thiserror::Error)]
pub enum ArrowError {
    #[error("Arrow error: {0}")]
    ArrowInternal(#[from] arrow::error::ArrowError),
    
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
    
    #[error("Unsupported dtype: {0}")]
    UnsupportedDType(String),
    
    #[error("Shape metadata missing")]
    MissingShapeMetadata,
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl From<ArrowError> for PyErr {
    fn from(err: ArrowError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", err))
    }
}

/// Arrow-based tensor serialization for efficient IPC
///
/// Provides efficient tensor serialization using Apache Arrow's IPC format,
/// which supports zero-copy reads and is optimized for columnar data.
#[pyclass(name = "ArrowTensorInterop")]
#[derive(Clone)]
pub struct ArrowTensorInterop {
    /// Cached schema for repeated serialization of same-shaped tensors
    #[allow(dead_code)]
    cached_schema: Option<Arc<Schema>>,
}

#[pymethods]
impl ArrowTensorInterop {
    /// Create a new Arrow tensor interop handler
    #[new]
    pub fn new() -> Self {
        Self { cached_schema: None }
    }
    
    /// Serialize a tensor to Arrow IPC format
    ///
    /// Returns bytes that can be efficiently transferred between processes
    /// and deserialized with zero-copy reads.
    pub fn serialize_tensor<'py>(&self, py: Python<'py>, tensor: &CandleTensorView) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = serialize_tensor_to_arrow(tensor.tensor())?;
        Ok(PyBytes::new_bound(py, &bytes))
    }
    
    /// Deserialize a tensor from Arrow IPC format
    ///
    /// Performs zero-copy read of the Arrow data and creates a new Candle tensor.
    #[staticmethod]
    pub fn deserialize_tensor(data: &[u8]) -> PyResult<CandleTensorView> {
        let tensor = deserialize_tensor_from_arrow(data)?;
        Ok(CandleTensorView::from_tensor(tensor))
    }
    
    /// Serialize multiple tensors as a batch
    ///
    /// More efficient than serializing tensors individually when transferring
    /// multiple tensors (e.g., model weights, gradient batches).
    pub fn serialize_batch<'py>(
        &self,
        py: Python<'py>,
        tensors: Vec<CandleTensorView>,
        names: Vec<String>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if tensors.len() != names.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "tensors and names must have the same length"
            ));
        }
        
        let bytes = serialize_tensor_batch(&tensors, &names)?;
        Ok(PyBytes::new_bound(py, &bytes))
    }
    
    /// Deserialize a batch of tensors
    #[staticmethod]
    pub fn deserialize_batch(data: &[u8]) -> PyResult<Vec<(String, CandleTensorView)>> {
        let result = deserialize_tensor_batch(data)?;
        Ok(result)
    }
    
    /// Get metadata about serialized tensor without fully deserializing
    #[staticmethod]
    pub fn peek_metadata(data: &[u8]) -> PyResult<TensorMetadata> {
        peek_tensor_metadata(data)
    }
}

/// Metadata about a serialized tensor
#[pyclass]
#[derive(Clone)]
pub struct TensorMetadata {
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub dtype: String,
    #[pyo3(get)]
    pub size_bytes: usize,
}

#[pymethods]
impl TensorMetadata {
    fn __repr__(&self) -> String {
        format!(
            "TensorMetadata(shape={:?}, dtype={}, size_bytes={})",
            self.shape, self.dtype, self.size_bytes
        )
    }
}

/// Serialize a Candle tensor to Arrow IPC bytes
pub fn serialize_tensor_to_arrow(tensor: &Tensor) -> Result<Vec<u8>, ArrowError> {
    let tensor = tensor.contiguous()?;
    let shape = tensor.dims().to_vec();
    let dtype = tensor.dtype();
    
    // Create Arrow array from tensor data
    let array: ArrayRef = match dtype {
        DType::F32 => {
            let data = tensor.flatten_all()?.to_vec1::<f32>()?;
            Arc::new(Float32Array::from(data))
        }
        DType::F64 => {
            let data = tensor.flatten_all()?.to_vec1::<f64>()?;
            Arc::new(Float64Array::from(data))
        }
        DType::I64 => {
            let data = tensor.flatten_all()?.to_vec1::<i64>()?;
            Arc::new(Int64Array::from(data))
        }
        DType::U32 => {
            let data = tensor.flatten_all()?.to_vec1::<u32>()?;
            Arc::new(UInt32Array::from(data))
        }
        DType::U8 => {
            let data = tensor.flatten_all()?.to_vec1::<u8>()?;
            Arc::new(UInt8Array::from(data))
        }
        _ => return Err(ArrowError::UnsupportedDType(format!("{:?}", dtype))),
    };
    
    // Encode shape and dtype in metadata
    let shape_str = shape.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let dtype_str = format!("{:?}", dtype);
    
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("tensor_shape".to_string(), shape_str);
    metadata.insert("tensor_dtype".to_string(), dtype_str);
    
    let field = Field::new("data", array.data_type().clone(), false);
    let schema = Schema::new_with_metadata(vec![field], metadata);
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![array])?;
    
    // Write to IPC stream
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &schema)?;
        writer.write(&batch)?;
        writer.finish()?;
    }
    
    Ok(buffer)
}

/// Deserialize Arrow IPC bytes to a Candle tensor
pub fn deserialize_tensor_from_arrow(data: &[u8]) -> Result<Tensor, ArrowError> {
    let cursor = Cursor::new(data);
    let reader = StreamReader::try_new(cursor, None)?;
    
    // Get schema metadata
    let schema = reader.schema();
    let metadata = schema.metadata();
    
    let shape_str = metadata.get("tensor_shape")
        .ok_or(ArrowError::MissingShapeMetadata)?;
    let dtype_str = metadata.get("tensor_dtype")
        .ok_or(ArrowError::MissingShapeMetadata)?;
    
    let shape: Vec<usize> = shape_str.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap_or(1))
        .collect();
    
    // Read the batch
    let mut batches = Vec::new();
    for batch_result in reader {
        batches.push(batch_result?);
    }
    
    if batches.is_empty() {
        return Err(ArrowError::MissingShapeMetadata);
    }
    
    let batch = &batches[0];
    let array = batch.column(0);
    
    // Convert based on dtype
    let tensor = match dtype_str.as_str() {
        "F32" => {
            let arr = array.as_any().downcast_ref::<Float32Array>()
                .ok_or_else(|| ArrowError::UnsupportedDType("F32 downcast failed".to_string()))?;
            let data: Vec<f32> = arr.values().to_vec();
            Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
        }
        "F64" => {
            let arr = array.as_any().downcast_ref::<Float64Array>()
                .ok_or_else(|| ArrowError::UnsupportedDType("F64 downcast failed".to_string()))?;
            let data: Vec<f64> = arr.values().to_vec();
            Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
        }
        "I64" => {
            let arr = array.as_any().downcast_ref::<Int64Array>()
                .ok_or_else(|| ArrowError::UnsupportedDType("I64 downcast failed".to_string()))?;
            let data: Vec<i64> = arr.values().to_vec();
            Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
        }
        "U32" => {
            let arr = array.as_any().downcast_ref::<UInt32Array>()
                .ok_or_else(|| ArrowError::UnsupportedDType("U32 downcast failed".to_string()))?;
            let data: Vec<u32> = arr.values().to_vec();
            Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
        }
        "U8" => {
            let arr = array.as_any().downcast_ref::<UInt8Array>()
                .ok_or_else(|| ArrowError::UnsupportedDType("U8 downcast failed".to_string()))?;
            let data: Vec<u8> = arr.values().to_vec();
            Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
        }
        _ => return Err(ArrowError::UnsupportedDType(dtype_str.clone())),
    };
    
    Ok(tensor)
}

/// Serialize multiple tensors as a batch
pub fn serialize_tensor_batch(tensors: &[CandleTensorView], names: &[String]) -> Result<Vec<u8>, ArrowError> {
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    let mut metadata = std::collections::HashMap::new();
    
    for (tensor, name) in tensors.iter().zip(names.iter()) {
        let t = tensor.tensor().contiguous()?;
        let shape = t.dims().to_vec();
        let dtype = t.dtype();
        
        // Store shape and dtype in metadata
        let shape_str = shape.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        metadata.insert(format!("{}_shape", name), shape_str);
        metadata.insert(format!("{}_dtype", name), format!("{:?}", dtype));
        
        let array: ArrayRef = match dtype {
            DType::F32 => {
                let data = t.flatten_all()?.to_vec1::<f32>()?;
                Arc::new(Float32Array::from(data))
            }
            DType::F64 => {
                let data = t.flatten_all()?.to_vec1::<f64>()?;
                Arc::new(Float64Array::from(data))
            }
            DType::I64 => {
                let data = t.flatten_all()?.to_vec1::<i64>()?;
                Arc::new(Int64Array::from(data))
            }
            DType::U32 => {
                let data = t.flatten_all()?.to_vec1::<u32>()?;
                Arc::new(UInt32Array::from(data))
            }
            DType::U8 => {
                let data = t.flatten_all()?.to_vec1::<u8>()?;
                Arc::new(UInt8Array::from(data))
            }
            _ => return Err(ArrowError::UnsupportedDType(format!("{:?}", dtype))),
        };
        
        let field = Field::new(name.clone(), array.data_type().clone(), false);
        fields.push(field);
        arrays.push(array);
    }
    
    let schema = Schema::new_with_metadata(fields, metadata);
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), arrays)?;
    
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &schema)?;
        writer.write(&batch)?;
        writer.finish()?;
    }
    
    Ok(buffer)
}

/// Deserialize a batch of tensors
pub fn deserialize_tensor_batch(data: &[u8]) -> Result<Vec<(String, CandleTensorView)>, ArrowError> {
    let cursor = Cursor::new(data);
    let reader = StreamReader::try_new(cursor, None)?;
    
    let schema = reader.schema();
    let metadata = schema.metadata();
    
    let mut batches = Vec::new();
    for batch_result in reader {
        batches.push(batch_result?);
    }
    
    if batches.is_empty() {
        return Err(ArrowError::MissingShapeMetadata);
    }
    
    let batch = &batches[0];
    let mut result = Vec::new();
    
    for (i, field) in schema.fields().iter().enumerate() {
        let name = field.name().clone();
        let array = batch.column(i);
        
        let shape_key = format!("{}_shape", name);
        let dtype_key = format!("{}_dtype", name);
        
        let shape_str = metadata.get(&shape_key)
            .ok_or(ArrowError::MissingShapeMetadata)?;
        let dtype_str = metadata.get(&dtype_key)
            .ok_or(ArrowError::MissingShapeMetadata)?;
        
        let shape: Vec<usize> = shape_str.split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap_or(1))
            .collect();
        
        let tensor = match dtype_str.as_str() {
            "F32" => {
                let arr = array.as_any().downcast_ref::<Float32Array>()
                    .ok_or_else(|| ArrowError::UnsupportedDType("F32 downcast failed".to_string()))?;
                let data: Vec<f32> = arr.values().to_vec();
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
            }
            "F64" => {
                let arr = array.as_any().downcast_ref::<Float64Array>()
                    .ok_or_else(|| ArrowError::UnsupportedDType("F64 downcast failed".to_string()))?;
                let data: Vec<f64> = arr.values().to_vec();
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
            }
            "I64" => {
                let arr = array.as_any().downcast_ref::<Int64Array>()
                    .ok_or_else(|| ArrowError::UnsupportedDType("I64 downcast failed".to_string()))?;
                let data: Vec<i64> = arr.values().to_vec();
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
            }
            "U32" => {
                let arr = array.as_any().downcast_ref::<UInt32Array>()
                    .ok_or_else(|| ArrowError::UnsupportedDType("U32 downcast failed".to_string()))?;
                let data: Vec<u32> = arr.values().to_vec();
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
            }
            "U8" => {
                let arr = array.as_any().downcast_ref::<UInt8Array>()
                    .ok_or_else(|| ArrowError::UnsupportedDType("U8 downcast failed".to_string()))?;
                let data: Vec<u8> = arr.values().to_vec();
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
            }
            _ => return Err(ArrowError::UnsupportedDType(dtype_str.clone())),
        };
        
        result.push((name, CandleTensorView::from_tensor(tensor)));
    }
    
    Ok(result)
}

/// Peek at tensor metadata without fully deserializing
fn peek_tensor_metadata(data: &[u8]) -> PyResult<TensorMetadata> {
    let cursor = Cursor::new(data);
    let reader = StreamReader::try_new(cursor, None)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Arrow read error: {}", e)))?;
    
    let schema = reader.schema();
    let metadata = schema.metadata();
    
    let shape_str = metadata.get("tensor_shape")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing shape metadata"))?;
    let dtype_str = metadata.get("tensor_dtype")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing dtype metadata"))?;
    
    let shape: Vec<usize> = shape_str.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap_or(1))
        .collect();
    
    let elem_size = match dtype_str.as_str() {
        "F32" | "U32" => 4,
        "F64" | "I64" => 8,
        "F16" | "BF16" => 2,
        "U8" => 1,
        _ => 4,
    };
    
    let numel: usize = shape.iter().product();
    
    Ok(TensorMetadata {
        shape,
        dtype: dtype_str.clone(),
        size_bytes: numel * elem_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data.clone(), &[2, 3], &Device::Cpu).unwrap();
        
        let bytes = serialize_tensor_to_arrow(&tensor).unwrap();
        let restored = deserialize_tensor_from_arrow(&bytes).unwrap();
        
        assert_eq!(restored.dims(), &[2, 3]);
        let restored_data = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(restored_data, data);
    }
    
    #[test]
    fn test_serialize_deserialize_i64() {
        let data = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
        let tensor = Tensor::from_vec(data.clone(), &[2, 4], &Device::Cpu).unwrap();
        
        let bytes = serialize_tensor_to_arrow(&tensor).unwrap();
        let restored = deserialize_tensor_from_arrow(&bytes).unwrap();
        
        assert_eq!(restored.dims(), &[2, 4]);
        let restored_data = restored.flatten_all().unwrap().to_vec1::<i64>().unwrap();
        assert_eq!(restored_data, data);
    }
    
    #[test]
    fn test_batch_serialization() {
        let t1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        let t2 = Tensor::from_vec(vec![4.0f32, 5.0, 6.0, 7.0], &[2, 2], &Device::Cpu).unwrap();
        
        let tensors = vec![
            CandleTensorView::from_tensor(t1),
            CandleTensorView::from_tensor(t2),
        ];
        let names = vec!["tensor1".to_string(), "tensor2".to_string()];
        
        let bytes = serialize_tensor_batch(&tensors, &names).unwrap();
        let restored = deserialize_tensor_batch(&bytes).unwrap();
        
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].0, "tensor1");
        assert_eq!(restored[0].1.tensor().dims(), &[3]);
        assert_eq!(restored[1].0, "tensor2");
        assert_eq!(restored[1].1.tensor().dims(), &[2, 2]);
    }
}
