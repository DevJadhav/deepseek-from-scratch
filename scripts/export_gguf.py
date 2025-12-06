#!/usr/bin/env python3
"""Export trained DeepSeek model to GGUF format.

GGUF (GPT-Generated Unified Format) is the successor to GGML and is used by
llama.cpp, ollama, and other inference engines for efficient CPU/GPU inference.

Usage:
    # Basic export with Q8_0 quantization (recommended)
    uv run python scripts/export_gguf.py \
        --checkpoint ./checkpoints/tiny-mlx/final \
        --output ./exports/tiny-deepseek.gguf

    # Export with specific quantization
    uv run python scripts/export_gguf.py \
        --checkpoint ./checkpoints/modal/final \
        --output ./exports/tiny-deepseek-q4_k_m.gguf \
        --quantization q4_k_m

    # Export with F16 (no quantization loss)
    uv run python scripts/export_gguf.py \
        --checkpoint ./checkpoints/modal/final \
        --output ./exports/tiny-deepseek-f16.gguf \
        --quantization f16
"""

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np


class GGMLType(IntEnum):
    """GGML tensor data types."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23


@dataclass
class GGUFMetadata:
    """GGUF file metadata."""

    architecture: str = "deepseek"
    name: str = "tiny-deepseek"
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5


class GGUFWriter:
    """Write GGUF format files."""

    GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
    GGUF_VERSION = 3

    def __init__(self, output_path: Path, metadata: GGUFMetadata):
        self.output_path = output_path
        self.metadata = metadata
        self.tensors: list[tuple[str, np.ndarray, GGMLType]] = []
        self.kv_data: dict[str, Any] = {}

    def add_metadata(self) -> None:
        """Add standard GGUF metadata."""
        m = self.metadata

        # General metadata
        self.kv_data["general.architecture"] = m.architecture
        self.kv_data["general.name"] = m.name
        self.kv_data["general.file_type"] = 1  # F16

        # Model architecture
        self.kv_data[f"{m.architecture}.vocab_size"] = m.vocab_size
        self.kv_data[f"{m.architecture}.embedding_length"] = m.hidden_size
        self.kv_data[f"{m.architecture}.block_count"] = m.num_layers
        self.kv_data[f"{m.architecture}.attention.head_count"] = m.num_heads
        self.kv_data[f"{m.architecture}.attention.head_count_kv"] = m.num_kv_heads
        self.kv_data[
            f"{m.architecture}.context_length"
        ] = m.max_position_embeddings
        self.kv_data[f"{m.architecture}.rope.freq_base"] = m.rope_theta
        self.kv_data[
            f"{m.architecture}.attention.layer_norm_rms_epsilon"
        ] = m.layer_norm_eps

    def add_tensor(
        self, name: str, data: np.ndarray, dtype: GGMLType = GGMLType.F16
    ) -> None:
        """Add a tensor to the GGUF file."""
        self.tensors.append((name, data, dtype))

    def _get_quantized_size(self, n_elements: int, dtype: GGMLType) -> int:
        """Calculate the size of quantized data in bytes."""
        if dtype == GGMLType.F32:
            return n_elements * 4
        elif dtype == GGMLType.F16:
            return n_elements * 2
        elif dtype == GGMLType.Q8_0:
            # Q8_0: 32 elements per block, 2 bytes scale + 32 bytes data = 34 bytes per block
            n_blocks = (n_elements + 31) // 32
            return n_blocks * 34
        elif dtype == GGMLType.Q4_0:
            # Q4_0: 32 elements per block, 2 bytes scale + 16 bytes data = 18 bytes per block
            n_blocks = (n_elements + 31) // 32
            return n_blocks * 18
        elif dtype == GGMLType.Q4_K:
            # Q4_K: 256 elements per superblock
            # 2 (d) + 2 (dmin) + 12 (scales_mins) + 128 (data) = 144 bytes
            n_superblocks = (n_elements + 255) // 256
            return n_superblocks * 144
        elif dtype == GGMLType.Q5_K:
            # Q5_K: 256 elements per superblock
            # 2 (d) + 2 (dmin) + 12 (scales_mins) + 128 (low) + 32 (high) = 176 bytes
            n_superblocks = (n_elements + 255) // 256
            return n_superblocks * 176
        elif dtype == GGMLType.Q6_K:
            # Q6_K: 256 elements per superblock
            # 2 (d) + 16 (scales) + 128 (low) + 64 (high) = 210 bytes
            n_superblocks = (n_elements + 255) // 256
            return n_superblocks * 210
        else:
            # Default to F16 size
            return n_elements * 2

    def _write_string(self, f, s: str) -> None:
        """Write a GGUF string (length + bytes)."""
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))  # uint64 length
        f.write(encoded)

    def _write_kv_pair(self, f, key: str, value: Any) -> None:
        """Write a key-value pair."""
        self._write_string(f, key)

        if isinstance(value, str):
            f.write(struct.pack("<I", 8))  # GGUF_TYPE_STRING
            self._write_string(f, value)
        elif isinstance(value, int):
            if value < 0:
                f.write(struct.pack("<I", 5))  # GGUF_TYPE_INT32
                f.write(struct.pack("<i", value))
            else:
                f.write(struct.pack("<I", 4))  # GGUF_TYPE_UINT32
                f.write(struct.pack("<I", value))
        elif isinstance(value, float):
            f.write(struct.pack("<I", 6))  # GGUF_TYPE_FLOAT32
            f.write(struct.pack("<f", value))
        elif isinstance(value, bool):
            f.write(struct.pack("<I", 7))  # GGUF_TYPE_BOOL
            f.write(struct.pack("<?", value))
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")

    def _write_quantized_q8_0(self, f, data: np.ndarray) -> int:
        """Write Q8_0 quantized data and return bytes written."""
        quantized, scales = quantize_q8_0(data)
        n_blocks = quantized.shape[0]

        bytes_written = 0
        for i in range(n_blocks):
            # Write scale (fp16)
            f.write(scales[i].tobytes())
            bytes_written += 2
            # Write 32 int8 values
            f.write(quantized[i].tobytes())
            bytes_written += 32

        return bytes_written

    def _write_quantized_q4_0(self, f, data: np.ndarray) -> int:
        """Write Q4_0 quantized data and return bytes written."""
        packed, scales = quantize_q4_0(data)
        n_blocks = packed.shape[0]

        bytes_written = 0
        for i in range(n_blocks):
            # Write scale (fp16)
            f.write(scales[i].tobytes())
            bytes_written += 2
            # Write 16 packed bytes (32 values as nibbles)
            f.write(packed[i].tobytes())
            bytes_written += 16

        return bytes_written

    def _write_quantized_q4_k(self, f, data: np.ndarray) -> int:
        """Write Q4_K quantized data and return bytes written."""
        packed_quants, d, dmin, scales_mins = quantize_q4_k(data)
        n_superblocks = packed_quants.shape[0]

        bytes_written = 0
        for i in range(n_superblocks):
            # Write d (fp16)
            f.write(d[i].tobytes())
            bytes_written += 2
            # Write dmin (fp16)
            f.write(dmin[i].tobytes())
            bytes_written += 2
            # Write scales_mins (12 bytes)
            f.write(scales_mins[i].tobytes())
            bytes_written += 12
            # Write packed quants (128 bytes)
            f.write(packed_quants[i].tobytes())
            bytes_written += 128

        return bytes_written

    def _write_quantized_q5_k(self, f, data: np.ndarray) -> int:
        """Write Q5_K quantized data and return bytes written."""
        packed_low, packed_high, d, dmin, scales_mins = quantize_q5_k(data)
        n_superblocks = packed_low.shape[0]

        bytes_written = 0
        for i in range(n_superblocks):
            # Write d (fp16)
            f.write(d[i].tobytes())
            bytes_written += 2
            # Write dmin (fp16)
            f.write(dmin[i].tobytes())
            bytes_written += 2
            # Write scales_mins (12 bytes)
            f.write(scales_mins[i].tobytes())
            bytes_written += 12
            # Write packed low bits (128 bytes)
            f.write(packed_low[i].tobytes())
            bytes_written += 128
            # Write packed high bits (32 bytes)
            f.write(packed_high[i].tobytes())
            bytes_written += 32

        return bytes_written

    def _write_quantized_q6_k(self, f, data: np.ndarray) -> int:
        """Write Q6_K quantized data and return bytes written."""
        packed_low, packed_high, d, scales = quantize_q6_k(data)
        n_superblocks = packed_low.shape[0]

        bytes_written = 0
        for i in range(n_superblocks):
            # Write d (fp16)
            f.write(d[i].tobytes())
            bytes_written += 2
            # Write scales (16 int8)
            f.write(scales[i].tobytes())
            bytes_written += 16
            # Write packed low bits (128 bytes)
            f.write(packed_low[i].tobytes())
            bytes_written += 128
            # Write packed high bits (64 bytes)
            f.write(packed_high[i].tobytes())
            bytes_written += 64

        return bytes_written

    def write(self) -> None:
        """Write the GGUF file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.add_metadata()

        with open(self.output_path, "wb") as f:
            # Header
            f.write(struct.pack("<I", self.GGUF_MAGIC))
            f.write(struct.pack("<I", self.GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))  # tensor_count
            f.write(struct.pack("<Q", len(self.kv_data)))  # metadata_kv_count

            # Metadata key-value pairs
            for key, value in self.kv_data.items():
                self._write_kv_pair(f, key, value)

            # Tensor info (before tensor data)
            tensor_data_offset = 0
            tensor_infos = []

            for name, data, dtype in self.tensors:
                # Calculate quantized data size
                n_elements = data.size
                data_size = self._get_quantized_size(n_elements, dtype)

                # Align to 32 bytes
                padding = (32 - (data_size % 32)) % 32

                tensor_infos.append(
                    {
                        "name": name,
                        "n_dims": len(data.shape),
                        "dims": data.shape,
                        "dtype": dtype,
                        "offset": tensor_data_offset,
                    }
                )
                tensor_data_offset += data_size + padding

            # Write tensor info
            for info in tensor_infos:
                self._write_string(f, info["name"])
                f.write(struct.pack("<I", info["n_dims"]))  # n_dims
                for dim in info["dims"]:
                    f.write(struct.pack("<Q", dim))  # dims
                f.write(struct.pack("<I", info["dtype"]))  # type
                f.write(struct.pack("<Q", info["offset"]))  # offset

            # Align to 32 bytes before tensor data
            current_pos = f.tell()
            padding = (32 - (current_pos % 32)) % 32
            f.write(b"\x00" * padding)

            # Write tensor data with proper quantization
            for _name, data, dtype in self.tensors:
                if dtype == GGMLType.F16:
                    data_f16 = data.astype(np.float16)
                    f.write(data_f16.tobytes())
                    data_size = data_f16.nbytes
                elif dtype == GGMLType.F32:
                    data_f32 = data.astype(np.float32)
                    f.write(data_f32.tobytes())
                    data_size = data_f32.nbytes
                elif dtype == GGMLType.Q8_0:
                    data_size = self._write_quantized_q8_0(f, data)
                elif dtype == GGMLType.Q4_0:
                    data_size = self._write_quantized_q4_0(f, data)
                elif dtype == GGMLType.Q4_K:
                    data_size = self._write_quantized_q4_k(f, data)
                elif dtype == GGMLType.Q5_K:
                    data_size = self._write_quantized_q5_k(f, data)
                elif dtype == GGMLType.Q6_K:
                    data_size = self._write_quantized_q6_k(f, data)
                else:
                    # Unknown quantized types fall back to F16
                    data_f16 = data.astype(np.float16)
                    f.write(data_f16.tobytes())
                    data_size = data_f16.nbytes

                # Align to 32 bytes
                padding = (32 - (data_size % 32)) % 32
                f.write(b"\x00" * padding)


def quantize_q8_0(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to Q8_0 format (8-bit with block scaling).

    Q8_0 uses 32-element blocks with a single FP16 scale factor.
    Each element is quantized to int8.

    Returns:
        Tuple of (quantized_data, scales)
    """
    # Reshape to blocks of 32
    flat = data.flatten().astype(np.float32)

    # Pad to multiple of 32
    pad_size = (32 - (len(flat) % 32)) % 32
    if pad_size > 0:
        flat = np.pad(flat, (0, pad_size), mode="constant", constant_values=0)

    blocks = flat.reshape(-1, 32)

    # Compute scales (max abs value per block)
    # For all-zero blocks, use 1.0 to avoid division by zero and indicate "no data"
    scales_raw = np.max(np.abs(blocks), axis=1)
    scales = np.where(scales_raw == 0, np.float32(1.0), scales_raw).astype(np.float16)

    # Quantize to int8
    quantized = np.round(blocks / scales[:, np.newaxis].astype(np.float32) * 127).clip(-128, 127).astype(np.int8)

    return quantized, scales


def quantize_q4_0(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to Q4_0 format (4-bit with block scaling).

    Q4_0 uses 32-element blocks with a single FP16 scale factor.
    Each element is quantized to 4 bits (packed 2 per byte).

    Returns:
        Tuple of (quantized_data, scales)
    """
    flat = data.flatten().astype(np.float32)

    # Pad to multiple of 32
    pad_size = (32 - (len(flat) % 32)) % 32
    if pad_size > 0:
        flat = np.pad(flat, (0, pad_size), mode="constant", constant_values=0)

    blocks = flat.reshape(-1, 32)
    n_blocks = blocks.shape[0]

    # Compute scales (max abs value per block)
    scales = np.max(np.abs(blocks), axis=1).astype(np.float16)
    scales_safe = np.where(scales == 0, np.float16(1.0), scales)

    # Quantize to 4 bits (-8 to 7)
    quantized = np.round(blocks / scales_safe[:, np.newaxis] * 7).clip(-8, 7).astype(np.int8)

    # Pack nibbles: 2 values per byte
    # Low nibble = first value + 8, High nibble = second value + 8
    packed = np.zeros((n_blocks, 16), dtype=np.uint8)
    for i in range(16):
        low = (quantized[:, 2 * i] + 8).astype(np.uint8)
        high = (quantized[:, 2 * i + 1] + 8).astype(np.uint8)
        packed[:, i] = low | (high << 4)

    return packed, scales


def quantize_q4_k(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantize to Q4_K format (4-bit K-quant with superblock scaling).

    Q4_K uses 256-element superblocks with:
    - 1 FP16 d (superblock scale)
    - 1 FP16 dmin (superblock min)
    - 12 bytes of scales/mins for 8 sub-blocks (6 bits each, packed)
    - 128 bytes of quantized data (4-bit packed)

    Each superblock contains 8 sub-blocks of 32 elements.

    Returns:
        Tuple of (packed_quants, d, dmin, scales_mins)
    """
    flat = data.flatten().astype(np.float32)

    # Pad to multiple of 256 (superblock size)
    pad_size = (256 - (len(flat) % 256)) % 256
    if pad_size > 0:
        flat = np.pad(flat, (0, pad_size), mode="constant", constant_values=0)

    superblocks = flat.reshape(-1, 256)
    n_superblocks = superblocks.shape[0]

    # Reshape each superblock into 8 sub-blocks of 32 elements
    sub_blocks = superblocks.reshape(n_superblocks, 8, 32)

    # Compute per-sub-block scales and mins
    sub_maxes = np.max(sub_blocks, axis=2)  # (n_superblocks, 8)
    sub_mins = np.min(sub_blocks, axis=2)   # (n_superblocks, 8)

    # Superblock d and dmin: max of sub-block ranges
    d = np.max(sub_maxes - sub_mins, axis=1).astype(np.float16)  # (n_superblocks,)
    dmin = np.max(-sub_mins, axis=1).astype(np.float16)  # (n_superblocks,)

    d_safe = np.where(d == 0, np.float16(1.0), d)
    dmin_safe = np.where(dmin == 0, np.float16(1.0), dmin)

    # Compute 6-bit scales and mins for each sub-block
    scales = np.zeros((n_superblocks, 8), dtype=np.uint8)
    mins = np.zeros((n_superblocks, 8), dtype=np.uint8)

    for i in range(8):
        sub_range = sub_maxes[:, i] - sub_mins[:, i]
        scales[:, i] = np.round(sub_range / d_safe * 63).clip(0, 63).astype(np.uint8)
        mins[:, i] = np.round(-sub_mins[:, i] / dmin_safe * 63).clip(0, 63).astype(np.uint8)

    # Pack scales and mins into 12 bytes per superblock
    # Format: scales[0-3] low 4 bits, scales[0-3] high 2 bits, ...
    scales_mins = np.zeros((n_superblocks, 12), dtype=np.uint8)
    for sb in range(n_superblocks):
        # Pack first 4 scales (lower 4 bits)
        scales_mins[sb, 0] = (scales[sb, 0] & 0xF) | ((scales[sb, 1] & 0xF) << 4)
        scales_mins[sb, 1] = (scales[sb, 2] & 0xF) | ((scales[sb, 3] & 0xF) << 4)
        # Pack last 4 scales (lower 4 bits)
        scales_mins[sb, 2] = (scales[sb, 4] & 0xF) | ((scales[sb, 5] & 0xF) << 4)
        scales_mins[sb, 3] = (scales[sb, 6] & 0xF) | ((scales[sb, 7] & 0xF) << 4)
        # Pack first 4 mins (lower 4 bits)
        scales_mins[sb, 4] = (mins[sb, 0] & 0xF) | ((mins[sb, 1] & 0xF) << 4)
        scales_mins[sb, 5] = (mins[sb, 2] & 0xF) | ((mins[sb, 3] & 0xF) << 4)
        # Pack last 4 mins (lower 4 bits)
        scales_mins[sb, 6] = (mins[sb, 4] & 0xF) | ((mins[sb, 5] & 0xF) << 4)
        scales_mins[sb, 7] = (mins[sb, 6] & 0xF) | ((mins[sb, 7] & 0xF) << 4)
        # Pack high 2 bits of scales
        scales_mins[sb, 8] = ((scales[sb, 0] >> 4) & 0x3) | (((scales[sb, 1] >> 4) & 0x3) << 2) | \
                            (((scales[sb, 2] >> 4) & 0x3) << 4) | (((scales[sb, 3] >> 4) & 0x3) << 6)
        scales_mins[sb, 9] = ((scales[sb, 4] >> 4) & 0x3) | (((scales[sb, 5] >> 4) & 0x3) << 2) | \
                            (((scales[sb, 6] >> 4) & 0x3) << 4) | (((scales[sb, 7] >> 4) & 0x3) << 6)
        # Pack high 2 bits of mins
        scales_mins[sb, 10] = ((mins[sb, 0] >> 4) & 0x3) | (((mins[sb, 1] >> 4) & 0x3) << 2) | \
                             (((mins[sb, 2] >> 4) & 0x3) << 4) | (((mins[sb, 3] >> 4) & 0x3) << 6)
        scales_mins[sb, 11] = ((mins[sb, 4] >> 4) & 0x3) | (((mins[sb, 5] >> 4) & 0x3) << 2) | \
                             (((mins[sb, 6] >> 4) & 0x3) << 4) | (((mins[sb, 7] >> 4) & 0x3) << 6)

    # Quantize the actual values to 4 bits and pack
    packed_quants = np.zeros((n_superblocks, 128), dtype=np.uint8)

    for sb in range(n_superblocks):
        for sub_idx in range(8):
            sub_block = sub_blocks[sb, sub_idx]
            sc = (scales[sb, sub_idx] / 63.0) * d_safe[sb] if d_safe[sb] != 0 else 1.0
            mi = (mins[sb, sub_idx] / 63.0) * dmin_safe[sb] if dmin_safe[sb] != 0 else 0.0

            sc_safe = sc if sc != 0 else 1.0
            quantized = np.round((sub_block + mi) / sc_safe * 15).clip(0, 15).astype(np.uint8)

            # Pack into nibbles
            base = sub_idx * 16
            for i in range(16):
                low = quantized[2 * i]
                high = quantized[2 * i + 1]
                packed_quants[sb, base + i] = low | (high << 4)

    return packed_quants, d, dmin, scales_mins


def quantize_q5_k(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantize to Q5_K format (5-bit K-quant with superblock scaling).

    Q5_K uses 256-element superblocks with:
    - 1 FP16 d (superblock scale)
    - 1 FP16 dmin (superblock min)
    - 12 bytes of scales/mins for 8 sub-blocks
    - 128 bytes of low 4-bit quantized data
    - 32 bytes of high bits (bit 5 for each element)

    Returns:
        Tuple of (packed_quants_low, packed_quants_high, d, dmin, scales_mins)
    """
    flat = data.flatten().astype(np.float32)

    # Pad to multiple of 256
    pad_size = (256 - (len(flat) % 256)) % 256
    if pad_size > 0:
        flat = np.pad(flat, (0, pad_size), mode="constant", constant_values=0)

    superblocks = flat.reshape(-1, 256)
    n_superblocks = superblocks.shape[0]
    sub_blocks = superblocks.reshape(n_superblocks, 8, 32)

    # Compute per-sub-block scales and mins
    sub_maxes = np.max(sub_blocks, axis=2)
    sub_mins = np.min(sub_blocks, axis=2)

    d = np.max(sub_maxes - sub_mins, axis=1).astype(np.float16)
    dmin = np.max(-sub_mins, axis=1).astype(np.float16)

    d_safe = np.where(d == 0, np.float16(1.0), d)
    dmin_safe = np.where(dmin == 0, np.float16(1.0), dmin)

    # 6-bit scales and mins
    scales = np.zeros((n_superblocks, 8), dtype=np.uint8)
    mins = np.zeros((n_superblocks, 8), dtype=np.uint8)

    for i in range(8):
        sub_range = sub_maxes[:, i] - sub_mins[:, i]
        scales[:, i] = np.round(sub_range / d_safe * 63).clip(0, 63).astype(np.uint8)
        mins[:, i] = np.round(-sub_mins[:, i] / dmin_safe * 63).clip(0, 63).astype(np.uint8)

    # Pack scales and mins (same as Q4_K)
    scales_mins = np.zeros((n_superblocks, 12), dtype=np.uint8)
    for sb in range(n_superblocks):
        scales_mins[sb, 0] = (scales[sb, 0] & 0xF) | ((scales[sb, 1] & 0xF) << 4)
        scales_mins[sb, 1] = (scales[sb, 2] & 0xF) | ((scales[sb, 3] & 0xF) << 4)
        scales_mins[sb, 2] = (scales[sb, 4] & 0xF) | ((scales[sb, 5] & 0xF) << 4)
        scales_mins[sb, 3] = (scales[sb, 6] & 0xF) | ((scales[sb, 7] & 0xF) << 4)
        scales_mins[sb, 4] = (mins[sb, 0] & 0xF) | ((mins[sb, 1] & 0xF) << 4)
        scales_mins[sb, 5] = (mins[sb, 2] & 0xF) | ((mins[sb, 3] & 0xF) << 4)
        scales_mins[sb, 6] = (mins[sb, 4] & 0xF) | ((mins[sb, 5] & 0xF) << 4)
        scales_mins[sb, 7] = (mins[sb, 6] & 0xF) | ((mins[sb, 7] & 0xF) << 4)
        scales_mins[sb, 8] = ((scales[sb, 0] >> 4) & 0x3) | (((scales[sb, 1] >> 4) & 0x3) << 2) | \
                            (((scales[sb, 2] >> 4) & 0x3) << 4) | (((scales[sb, 3] >> 4) & 0x3) << 6)
        scales_mins[sb, 9] = ((scales[sb, 4] >> 4) & 0x3) | (((scales[sb, 5] >> 4) & 0x3) << 2) | \
                            (((scales[sb, 6] >> 4) & 0x3) << 4) | (((scales[sb, 7] >> 4) & 0x3) << 6)
        scales_mins[sb, 10] = ((mins[sb, 0] >> 4) & 0x3) | (((mins[sb, 1] >> 4) & 0x3) << 2) | \
                             (((mins[sb, 2] >> 4) & 0x3) << 4) | (((mins[sb, 3] >> 4) & 0x3) << 6)
        scales_mins[sb, 11] = ((mins[sb, 4] >> 4) & 0x3) | (((mins[sb, 5] >> 4) & 0x3) << 2) | \
                             (((mins[sb, 6] >> 4) & 0x3) << 4) | (((mins[sb, 7] >> 4) & 0x3) << 6)

    # Quantize to 5 bits and pack
    packed_quants_low = np.zeros((n_superblocks, 128), dtype=np.uint8)  # Low 4 bits
    packed_quants_high = np.zeros((n_superblocks, 32), dtype=np.uint8)  # High bit

    for sb in range(n_superblocks):
        for sub_idx in range(8):
            sub_block = sub_blocks[sb, sub_idx]
            sc = (scales[sb, sub_idx] / 63.0) * d_safe[sb] if d_safe[sb] != 0 else 1.0
            mi = (mins[sb, sub_idx] / 63.0) * dmin_safe[sb] if dmin_safe[sb] != 0 else 0.0

            sc_safe = sc if sc != 0 else 1.0
            quantized = np.round((sub_block + mi) / sc_safe * 31).clip(0, 31).astype(np.uint8)

            # Pack low 4 bits
            base = sub_idx * 16
            for i in range(16):
                low = quantized[2 * i] & 0xF
                high = quantized[2 * i + 1] & 0xF
                packed_quants_low[sb, base + i] = low | (high << 4)

            # Pack high bits (bit 5) - 32 elements per sub-block, 8 per byte
            high_base = sub_idx * 4
            for i in range(4):
                byte_val = 0
                for j in range(8):
                    if quantized[i * 8 + j] & 0x10:
                        byte_val |= (1 << j)
                packed_quants_high[sb, high_base + i] = byte_val

    return packed_quants_low, packed_quants_high, d, dmin, scales_mins


def quantize_q6_k(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantize to Q6_K format (6-bit K-quant with superblock scaling).

    Q6_K uses 256-element superblocks with:
    - 1 FP16 d (superblock scale)
    - 16 int8 scales (one per 16-element sub-block)
    - 128 bytes of low 4-bit quantized data
    - 64 bytes of high 2-bit quantized data

    Returns:
        Tuple of (packed_quants_low, packed_quants_high, d, scales)
    """
    flat = data.flatten().astype(np.float32)

    # Pad to multiple of 256
    pad_size = (256 - (len(flat) % 256)) % 256
    if pad_size > 0:
        flat = np.pad(flat, (0, pad_size), mode="constant", constant_values=0)

    superblocks = flat.reshape(-1, 256)
    n_superblocks = superblocks.shape[0]

    # Q6_K has 16 sub-blocks of 16 elements each
    sub_blocks = superblocks.reshape(n_superblocks, 16, 16)

    # Compute per-sub-block scales
    sub_maxes = np.max(np.abs(sub_blocks), axis=2)  # (n_superblocks, 16)

    # Superblock d: max of sub-block maxes
    d = np.max(sub_maxes, axis=1).astype(np.float16)  # (n_superblocks,)
    d_safe = np.where(d == 0, np.float16(1.0), d)

    # Compute int8 scales for each sub-block
    scales = np.zeros((n_superblocks, 16), dtype=np.int8)
    for i in range(16):
        scales[:, i] = np.round(sub_maxes[:, i] / d_safe * 127).clip(-128, 127).astype(np.int8)

    # Quantize to 6 bits and pack
    packed_quants_low = np.zeros((n_superblocks, 128), dtype=np.uint8)  # Low 4 bits
    packed_quants_high = np.zeros((n_superblocks, 64), dtype=np.uint8)  # High 2 bits

    for sb in range(n_superblocks):
        for sub_idx in range(16):
            sub_block = sub_blocks[sb, sub_idx]
            sc = (scales[sb, sub_idx] / 127.0) * d_safe[sb] if scales[sb, sub_idx] != 0 else 1.0

            quantized = np.round(sub_block / sc * 31).clip(-32, 31).astype(np.int8)
            # Shift to unsigned: -32..31 -> 0..63
            quantized_unsigned = (quantized + 32).astype(np.uint8)

            # Pack low 4 bits (2 values per byte)
            base = sub_idx * 8
            for i in range(8):
                low = quantized_unsigned[2 * i] & 0xF
                high = quantized_unsigned[2 * i + 1] & 0xF
                packed_quants_low[sb, base + i] = low | (high << 4)

            # Pack high 2 bits (4 values per byte)
            high_base = sub_idx * 4
            for i in range(4):
                byte_val = 0
                for j in range(4):
                    elem_idx = i * 4 + j
                    high_bits = (quantized_unsigned[elem_idx] >> 4) & 0x3
                    byte_val |= (high_bits << (j * 2))
                packed_quants_high[sb, high_base + i] = byte_val

    return packed_quants_low, packed_quants_high, d, scales


def load_checkpoint(checkpoint_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load model checkpoint and config."""
    # Try different checkpoint formats
    weights = {}
    config = {}

    # Check for config
    config_path = checkpoint_path / "training_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Check for MLX format (npz)
    npz_files = list(checkpoint_path.glob("*.npz"))
    if npz_files:
        for npz_file in npz_files:
            data = np.load(npz_file, allow_pickle=True)
            for key in data.files:
                weights[key] = data[key]
        return weights, config

    # Check for PyTorch format
    pt_files = list(checkpoint_path.glob("*.pt")) + list(
        checkpoint_path.glob("*.pth")
    )
    if pt_files:
        try:
            import torch

            for pt_file in pt_files:
                state_dict = torch.load(pt_file, map_location="cpu", weights_only=True)
                for key, value in state_dict.items():
                    weights[key] = value.numpy()
            return weights, config
        except ImportError:
            print("Warning: PyTorch not available, cannot load .pt files")

    # Check for safetensors format
    safetensors_files = list(checkpoint_path.glob("*.safetensors"))
    if safetensors_files:
        try:
            from safetensors import safe_open

            for st_file in safetensors_files:
                with safe_open(st_file, framework="numpy") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
            return weights, config
        except ImportError:
            print("Warning: safetensors not available")

    # Check for JSON weights (used by some configs)
    json_weights = checkpoint_path / "model_weights.json"
    if json_weights.exists():
        with open(json_weights) as f:
            weights_data = json.load(f)
            for key, value in weights_data.items():
                weights[key] = np.array(value)
        return weights, config

    return weights, config


def convert_weight_names(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert weight names to GGUF-compatible format."""
    converted = {}

    # Mapping from common formats to GGUF names
    name_mapping = {
        "embed_tokens": "token_embd",
        "lm_head": "output",
        "input_layernorm": "attn_norm",
        "post_attention_layernorm": "ffn_norm",
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
        "norm": "output_norm",
        "model.norm": "output_norm",
        # MoE-specific
        "moe.gate": "ffn_gate_inp",
        "moe.experts": "ffn_gate_exp",
    }

    for original_name, tensor in weights.items():
        new_name = original_name

        # Apply mappings
        for old, new in name_mapping.items():
            if old in new_name:
                new_name = new_name.replace(old, new)

        # Convert layer indices
        # "layers.0." -> "blk.0."
        if "layers." in new_name:
            new_name = new_name.replace("layers.", "blk.")
        if "model." in new_name and "blk." in new_name:
            new_name = new_name.replace("model.", "")

        converted[new_name] = tensor

    return converted


def get_model_metadata(
    config: dict, weights: dict[str, np.ndarray]
) -> GGUFMetadata:
    """Extract model metadata from config and weights."""
    metadata = GGUFMetadata()

    # Try to infer from config
    if config:
        metadata.vocab_size = config.get("vocab_size", 32000)
        metadata.hidden_size = config.get("hidden_size", config.get("d_model", 512))
        metadata.num_layers = config.get(
            "num_layers", config.get("n_layers", 8)
        )
        metadata.num_heads = config.get(
            "num_attention_heads", config.get("n_heads", 8)
        )
        metadata.num_kv_heads = config.get(
            "num_key_value_heads", config.get("n_kv_heads", 4)
        )
        metadata.max_position_embeddings = config.get(
            "max_position_embeddings", config.get("max_seq_len", 2048)
        )

    # Infer from weights if possible
    for name, tensor in weights.items():
        if "embed" in name.lower() and "token" in name.lower():
            if len(tensor.shape) == 2:
                metadata.vocab_size = tensor.shape[0]
                metadata.hidden_size = tensor.shape[1]
                break

    return metadata


def export_gguf(
    checkpoint_path: Path,
    output_path: Path,
    quantization: str = "f16",
) -> None:
    """Export model checkpoint to GGUF format."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    weights, config = load_checkpoint(checkpoint_path)

    if not weights:
        print(f"Error: No weights found in {checkpoint_path}")
        print("Supported formats: .npz, .pt, .pth, .safetensors")
        sys.exit(1)

    print(f"Loaded {len(weights)} tensors")

    # Get metadata
    metadata = get_model_metadata(config, weights)
    print(f"Model: {metadata.name}")
    print(f"  Vocab size: {metadata.vocab_size}")
    print(f"  Hidden size: {metadata.hidden_size}")
    print(f"  Layers: {metadata.num_layers}")
    print(f"  Heads: {metadata.num_heads}")

    # Convert weight names
    weights = convert_weight_names(weights)

    # Determine GGML type
    ggml_type_map = {
        "f32": GGMLType.F32,
        "f16": GGMLType.F16,
        "q8_0": GGMLType.Q8_0,
        "q4_0": GGMLType.Q4_0,
        "q4_k": GGMLType.Q4_K,
        "q4_k_m": GGMLType.Q4_K,  # Q4_K medium
        "q5_k": GGMLType.Q5_K,
        "q6_k": GGMLType.Q6_K,
    }

    ggml_type = ggml_type_map.get(quantization.lower(), GGMLType.F16)
    print(f"Quantization: {quantization} (GGML type: {ggml_type.name})")

    # Create GGUF writer
    writer = GGUFWriter(output_path, metadata)

    # Add tensors - quantization happens during write()
    total_size = 0
    for name, tensor in weights.items():
        writer.add_tensor(name, tensor, ggml_type)
        total_size += tensor.nbytes

    print(f"Original tensor size: {total_size / 1024 / 1024:.2f} MB")

    # Write GGUF file
    print(f"Writing GGUF to {output_path}...")
    writer.write()

    # Report final size
    final_size = output_path.stat().st_size
    print(f"Done! Output size: {final_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {total_size / final_size:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Export trained DeepSeek model to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export with Q8_0 quantization (recommended)
    python scripts/export_gguf.py \\
        --checkpoint ./checkpoints/tiny-mlx/final \\
        --output ./exports/tiny-deepseek.gguf

    # Export with F16 (no quantization loss)
    python scripts/export_gguf.py \\
        --checkpoint ./checkpoints/modal/final \\
        --output ./exports/tiny-deepseek-f16.gguf \\
        --quantization f16

    # Export with Q4_K_M (smallest size)
    python scripts/export_gguf.py \\
        --checkpoint ./checkpoints/modal/final \\
        --output ./exports/tiny-deepseek-q4_k_m.gguf \\
        --quantization q4_k_m

Quantization Options:
    f32     - Full precision (largest, best quality)
    f16     - Half precision (recommended for GPU)
    q8_0    - 8-bit quantization (recommended for CPU)
    q4_0    - 4-bit quantization (smallest)
    q4_k_m  - 4-bit K-quant medium (good balance)
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output GGUF file path",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q8_0",
        choices=["f32", "f16", "q8_0", "q4_0", "q4_k_m", "q5_k", "q6_k"],
        help="Quantization format (default: q8_0)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="tiny-deepseek",
        help="Model name for metadata",
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint path does not exist: {args.checkpoint}")
        sys.exit(1)

    export_gguf(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        quantization=args.quantization,
    )


if __name__ == "__main__":
    main()
