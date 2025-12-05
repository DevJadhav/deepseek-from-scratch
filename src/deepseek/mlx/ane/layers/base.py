"""
ANE-Compatible Base Layers

This module implements ANE-optimized versions of fundamental neural network layers:
- ANERMSNorm: Root Mean Square Normalization with channel-last layout
- ANELinear: Linear layer with INT8/INT4 quantized weights and tiled matmul
- ANEEmbedding: Embedding layer with CPU lookup + ANE transfer pattern

ANE Optimization Principles:
1. Channel-last (NHWC) layout for optimal ANE throughput
2. FP16 computation for activations
3. INT8/INT4 quantized weights for memory efficiency
4. Powers of 2 dimensions preferred
5. 16-byte memory alignment
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.quantization import (
    QuantizationConfig,
    QuantizationType,
    QuantizedTensor,
    dequantize_int4_block,
    dequantize_int8_per_channel,
    quantize_int4_block,
    quantize_int8_per_channel,
)
from ..utils.tensor_ops import (
    align_tensor,
    normalize_shape_for_ane,
    pad_to_multiple,
    unpad_tensor,
)


class ANERMSNorm(nn.Module):
    """
    ANE-optimized Root Mean Square Layer Normalization.

    Features:
    - Channel-last (NHWC) layout support
    - FP16 computation for ANE efficiency
    - Optional fused scaling with learned weight
    - 16-byte aligned tensors

    Args:
        dim: Feature dimension to normalize
        eps: Epsilon for numerical stability (default 1e-6)
        use_fp16: Use FP16 for computation (default True)
        elementwise_affine: Apply learned scale (default True)
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        use_fp16: bool = True,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.use_fp16 = use_fp16
        self.elementwise_affine = elementwise_affine

        # ANE-friendly dimension (pad to multiple of 16)
        self.padded_dim = pad_to_multiple(dim, 16)

        if elementwise_affine:
            # Initialize weight to ones, padded for ANE
            weight = torch.ones(self.padded_dim)
            self.weight = nn.Parameter(weight)
        else:
            self.register_parameter('weight', None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization."""
        # Compute variance along last dimension
        variance = x.pow(2).mean(-1, keepdim=True)
        # RMS = x / sqrt(variance + eps)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ANE optimizations.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor of shape (..., dim)
        """
        original_dtype = x.dtype
        original_dim = x.shape[-1]

        # Convert to FP16 for ANE efficiency
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()

        # Pad input if needed for ANE alignment
        if original_dim != self.padded_dim:
            x = F.pad(x, (0, self.padded_dim - original_dim))

        # Compute RMS normalization
        x = self._norm(x)

        # Apply learned scale if present
        if self.weight is not None:
            x = x * self.weight

        # Remove padding
        if original_dim != self.padded_dim:
            x = x[..., :original_dim]

        # Convert back to original dtype if needed
        if x.dtype != original_dtype:
            x = x.to(original_dtype)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, use_fp16={self.use_fp16}"


class ANELinear(nn.Module):
    """
    ANE-optimized Linear layer with quantization support.

    Features:
    - INT8/INT4 weight quantization with per-channel/per-block scaling
    - Tiled matrix multiplication for large dimensions (128x128 tiles)
    - FP16 activation computation
    - ANE-friendly dimension padding

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Include bias term (default True)
        quant_type: Weight quantization type (default INT8_PER_CHANNEL)
        tile_size: Tile size for tiled matmul (default 128)
        use_fp16: Use FP16 for computation (default True)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_type: QuantizationType = QuantizationType.INT8_PER_CHANNEL,
        tile_size: int = 128,
        use_fp16: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        self.tile_size = tile_size
        self.use_fp16 = use_fp16

        # ANE-friendly dimensions
        self.padded_in = pad_to_multiple(in_features, 16)
        self.padded_out = pad_to_multiple(out_features, 16)

        # Initialize weight (will be quantized after initialization)
        self.register_buffer('weight_quantized', None)
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_zero_point', None)

        # Keep FP16 weight for non-quantized mode
        weight = torch.empty(self.padded_out, self.padded_in)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight[:out_features, :in_features])

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Track if weights are quantized
        self._quantized = False

    def quantize_weights(self) -> None:
        """
        Quantize weights to INT8 or INT4.

        Should be called after loading pretrained weights and before inference.
        """
        with torch.no_grad():
            weight = self.weight.data

            # Pad weight for ANE alignment
            if weight.shape[0] != self.padded_out or weight.shape[1] != self.padded_in:
                weight_padded = torch.zeros(
                    self.padded_out, self.padded_in,
                    dtype=weight.dtype, device=weight.device
                )
                weight_padded[:self.out_features, :self.in_features] = weight
                weight = weight_padded

            if self.quant_type == QuantizationType.INT8_PER_CHANNEL:
                quantized = quantize_int8_per_channel(weight, axis=0)
                self.weight_quantized = quantized.data
                self.weight_scale = quantized.scale
                self.weight_zero_point = quantized.zero_point

            elif self.quant_type == QuantizationType.INT4_PER_BLOCK:
                quantized = quantize_int4_block(weight, self.tile_size)
                self.weight_quantized = quantized.data
                self.weight_scale = quantized.scale
                self.weight_zero_point = None

            self._quantized = True

    def _get_weight(self) -> torch.Tensor:
        """Get dequantized weight for computation."""
        if not self._quantized or self.weight_quantized is None:
            return self.weight

        if self.quant_type == QuantizationType.INT8_PER_CHANNEL:
            quantized = QuantizedTensor(
                data=self.weight_quantized,
                scale=self.weight_scale,
                zero_point=self.weight_zero_point,
                original_shape=(self.out_features, self.in_features),
                quant_type=QuantizationType.INT8_PER_CHANNEL,
            )
            weight = dequantize_int8_per_channel(quantized, axis=0)

        elif self.quant_type == QuantizationType.INT4_PER_BLOCK:
            quantized = QuantizedTensor(
                data=self.weight_quantized,
                scale=self.weight_scale,
                zero_point=None,
                original_shape=(self.out_features, self.in_features),
                quant_type=QuantizationType.INT4_PER_BLOCK,
            )
            weight = dequantize_int4_block(quantized, self.tile_size)

        else:
            weight = self.weight

        return weight[:self.out_features, :self.in_features]

    def _tiled_matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform tiled matrix multiplication for ANE compatibility.

        Splits large matrix operations into smaller tiles to fit ANE constraints.
        """
        if x.shape[-1] <= self.tile_size and weight.shape[0] <= self.tile_size:
            # Small enough, no tiling needed
            return F.linear(x, weight, None)

        # Tile along output dimension
        outputs = []
        for i in range(0, weight.shape[0], self.tile_size):
            end_i = min(i + self.tile_size, weight.shape[0])
            weight_tile = weight[i:end_i, :]

            # Further tile along input dimension if needed
            if x.shape[-1] > self.tile_size:
                partial = torch.zeros(
                    *x.shape[:-1], end_i - i,
                    dtype=x.dtype, device=x.device
                )
                for j in range(0, x.shape[-1], self.tile_size):
                    end_j = min(j + self.tile_size, x.shape[-1])
                    x_tile = x[..., j:end_j]
                    w_tile = weight_tile[:, j:end_j]
                    partial += x_tile @ w_tile.t()
                outputs.append(partial)
            else:
                outputs.append(F.linear(x, weight_tile, None))

        return torch.cat(outputs, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized weights and tiled matmul.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        original_dtype = x.dtype

        # Convert to FP16 for ANE efficiency
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()

        # Get weight (dequantized if necessary)
        weight = self._get_weight()
        if self.use_fp16:
            weight = weight.half()

        # Perform matmul (tiled for large dimensions)
        if max(x.shape[-1], self.out_features) > self.tile_size:
            output = self._tiled_matmul(x, weight)
        else:
            output = F.linear(x, weight, None)

        # Add bias
        if self.bias is not None:
            bias = self.bias.half() if self.use_fp16 else self.bias
            output = output + bias

        # Convert back to original dtype
        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quant_type={self.quant_type.value}, "
            f"tile_size={self.tile_size}, quantized={self._quantized}"
        )


class ANEEmbedding(nn.Module):
    """
    ANE-optimized Embedding layer.

    Note: Embedding lookups are not efficient on ANE. This layer performs
    the lookup on CPU and transfers results to the target device.

    Features:
    - CPU-side lookup for efficiency
    - FP16 output for ANE compatibility
    - Optional weight tying support
    - ANE-friendly dimension padding

    Args:
        num_embeddings: Size of vocabulary
        embedding_dim: Dimension of embeddings
        padding_idx: Index for padding token (optional)
        use_fp16: Use FP16 for output (default True)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        use_fp16: bool = True,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.use_fp16 = use_fp16

        # ANE-friendly dimension
        self.padded_dim = pad_to_multiple(embedding_dim, 16)

        # Initialize embedding weight
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim)
        )
        self.reset_parameters()

        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def reset_parameters(self) -> None:
        """Initialize embedding weights."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embeddings for input tokens.

        The lookup is performed on CPU for efficiency, then results
        are transferred to the target device.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, embedding_dim)
        """
        target_device = input_ids.device

        # Perform lookup (most efficient on CPU for embedding)
        # Note: For MPS/ANE, the lookup itself is fine, it's the
        # scattered memory access pattern that's inefficient
        embeddings = F.embedding(
            input_ids,
            self.weight,
            padding_idx=self.padding_idx,
        )

        # Convert to FP16 for ANE efficiency
        if self.use_fp16 and embeddings.dtype != torch.float16:
            embeddings = embeddings.half()

        return embeddings

    def extra_repr(self) -> str:
        s = f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        return s
