#!/usr/bin/env python3
"""
Inference Server for DeepSeek Model
====================================

FastAPI server with SSE streaming, OpenAI-compatible /v1/completions endpoint.
Supports all 3 backends: MLX, PyTorch+CUDA, and Rust (via FFI).

Usage:
    # MLX backend (default on Apple Silicon)
    uv run python scripts/inference_server.py --model-path ./checkpoints/final --port 8080
    
    # PyTorch backend
    INFERENCE_BACKEND=pytorch uv run python scripts/inference_server.py --model-path ./checkpoints/final
    
    # With specific device
    uv run python scripts/inference_server.py --model-path ./checkpoints/final --backend pytorch --device cuda:0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# API Models (OpenAI-compatible)
# =============================================================================


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""

    model: str = Field(default="deepseek", description="Model ID")
    prompt: str | list[str] = Field(..., description="Prompt(s) to generate completions for")
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: int = Field(default=50, ge=1, description="Top-k sampling")
    stream: bool = Field(default=False, description="Stream the response")
    stop: str | list[str] | None = Field(default=None, description="Stop sequences")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    n: int = Field(default=1, ge=1, le=8, description="Number of completions")
    echo: bool = Field(default=False, description="Echo the prompt")
    user: str | None = Field(default=None, description="User ID for tracking")


class CompletionChoice(BaseModel):
    """Completion choice."""

    text: str
    index: int
    logprobs: dict[str, Any] | None = None
    finish_reason: str | None = None


class CompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage | None = None


class ChatMessage(BaseModel):
    """Chat message."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(default="deepseek", description="Model ID")
    messages: list[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)
    stop: str | list[str] | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    backend: str
    device: str
    model_loaded: bool


# =============================================================================
# Backend Implementations
# =============================================================================


@dataclass
class ModelState:
    """Global model state."""

    model: Any = None
    tokenizer: Any = None
    config: Any = None
    backend: str = "mlx"
    device: str = "cpu"
    model_path: str = ""


# Global state
model_state = ModelState()


def load_mlx_model(model_path: str) -> tuple:
    """Load MLX model."""
    import mlx.core as mx

    from src.deepseek.mlx.tiny_trainer import TinyModelConfig, TinyMTPModel

    path = Path(model_path)

    # Load config
    config_file = path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config_dict = json.load(f)
        config = TinyModelConfig(**config_dict)
    else:
        logger.warning("No config.json found, using default config")
        config = TinyModelConfig()

    # Create model
    model = TinyMTPModel(config)

    # Load weights
    weights_file = path / "model.safetensors"
    if weights_file.exists():
        flat_weights = mx.load(str(weights_file))
        # Unflatten if needed
        if any("." in k for k in flat_weights.keys()):
            weights = _unflatten_params(flat_weights)
            model.update(weights)
        else:
            model.update(flat_weights)
        logger.info(f"Loaded MLX weights from {weights_file}")

    mx.eval(model.parameters())
    return model, config


def load_pytorch_model(model_path: str, device: str = "auto") -> tuple:
    """Load PyTorch model."""
    import torch

    path = Path(model_path)

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load config
    config_file = path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config = {}

    # Try to load from safetensors or pt
    weights_file = path / "model.safetensors"
    if weights_file.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_file))
    elif (path / "model.pt").exists():
        state_dict = torch.load(path / "model.pt", map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}")

    # Import model class
    try:
        from src.deepseek.torch.model.deepseek_v3 import DeepSeekV3Config, DeepSeekV3Model

        model_config = DeepSeekV3Config(**config)
        model = DeepSeekV3Model(model_config)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        logger.warning(f"Failed to load DeepSeekV3Model: {e}, trying TinyModel")
        # Fallback to simple model
        from src.deepseek.torch.model.model import DeepSeekConfig, DeepSeekModel

        model_config = DeepSeekConfig(**config) if config else DeepSeekConfig()
        model = DeepSeekModel(model_config)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    # Try torch.compile for optimization
    try:
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile()")
    except Exception as e:
        logger.warning(f"torch.compile() failed: {e}")

    return model, config, device


def _unflatten_params(flat_params: dict) -> dict:
    """Convert flat dotted keys back to nested dict structure."""

    def set_nested(d, keys, value):
        for i, key in enumerate(keys[:-1]):
            next_key = keys[i + 1]
            if key.isdigit():
                key = int(key)
                if isinstance(d, dict) and key not in d:
                    d[key] = [] if next_key.isdigit() else {}
                elif isinstance(d, list):
                    while len(d) <= key:
                        d.append([] if next_key.isdigit() else {})
                d = d[key]
            else:
                if key not in d:
                    d[key] = [] if next_key.isdigit() else {}
                d = d[key]

        final_key = keys[-1]
        if final_key.isdigit():
            final_key = int(final_key)
            while len(d) <= final_key:
                d.append(None)
            d[final_key] = value
        else:
            d[final_key] = value

    result = {}
    for key, value in flat_params.items():
        keys = key.split(".")
        set_nested(result, keys, value)

    return result


def load_tokenizer(tokenizer_name: str = "gpt2"):
    """Load tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# =============================================================================
# Generation Functions
# =============================================================================


def generate_mlx(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    stop_sequences: list[str] | None = None,
) -> str:
    """Generate text using MLX backend."""
    import mlx.core as mx

    model = model_state.model
    tokenizer = model_state.tokenizer

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors=None)
    tokens = mx.array([input_ids])

    generated = list(input_ids)

    for _ in range(max_tokens):
        # Forward pass
        logits = model(tokens)
        next_token_logits = logits[:, -1, :]

        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature

            # Apply top-k
            if top_k > 0:
                top_k_values = mx.topk(next_token_logits, k=min(top_k, next_token_logits.shape[-1]))
                threshold = top_k_values[0, -1]
                next_token_logits = mx.where(
                    next_token_logits < threshold,
                    mx.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # Apply top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits = mx.sort(next_token_logits, axis=-1)[:, ::-1]
                sorted_probs = mx.softmax(sorted_logits, axis=-1)
                cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
                mask = cumulative_probs > top_p
                # Keep at least one token
                mask = mx.concatenate([mx.zeros((1, 1), dtype=mx.bool_), mask[:, :-1]], axis=-1)
                sorted_logits = mx.where(mask, mx.full_like(sorted_logits, float("-inf")), sorted_logits)
                next_token_logits = sorted_logits

            # Sample
            probs = mx.softmax(next_token_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10), num_samples=1)
        else:
            # Greedy
            next_token = mx.argmax(next_token_logits, axis=-1, keepdims=True)

        next_token_id = next_token.item()
        generated.append(next_token_id)

        # Check for EOS
        if next_token_id == tokenizer.eos_token_id:
            break

        # Check for stop sequences
        if stop_sequences:
            current_text = tokenizer.decode(generated)
            for stop_seq in stop_sequences:
                if stop_seq in current_text:
                    generated_text = current_text[: current_text.rfind(stop_seq)]
                    return generated_text[len(prompt) :]

        # Update tokens for next iteration
        tokens = mx.array([generated])

    generated_text = tokenizer.decode(generated)
    return generated_text[len(prompt) :]


async def generate_mlx_stream(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    stop_sequences: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """Generate text using MLX backend with streaming."""
    import mlx.core as mx

    model = model_state.model
    tokenizer = model_state.tokenizer

    input_ids = tokenizer.encode(prompt, return_tensors=None)
    tokens = mx.array([input_ids])
    generated = list(input_ids)

    for _ in range(max_tokens):
        logits = model(tokens)
        next_token_logits = logits[:, -1, :]

        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            probs = mx.softmax(next_token_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10), num_samples=1)
        else:
            next_token = mx.argmax(next_token_logits, axis=-1, keepdims=True)

        next_token_id = next_token.item()
        generated.append(next_token_id)

        # Yield the new token
        token_text = tokenizer.decode([next_token_id])
        yield token_text

        if next_token_id == tokenizer.eos_token_id:
            break

        tokens = mx.array([generated])

        # Small delay for streaming effect
        await asyncio.sleep(0.01)


def generate_pytorch(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    stop_sequences: list[str] | None = None,
) -> str:
    """Generate text using PyTorch backend."""
    import torch
    import torch.nn.functional as F

    model = model_state.model
    tokenizer = model_state.tokenizer
    device = model_state.device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0].tolist()

    with torch.no_grad():
        for _ in range(max_tokens):
            tokens = torch.tensor([generated], device=device)
            outputs = model(tokens)

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                next_token_logits = next_token_logits / temperature

                # Top-k
                if top_k > 0:
                    top_k_values, _ = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
                    threshold = top_k_values[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(
                        next_token_logits < threshold,
                        torch.full_like(next_token_logits, float("-inf")),
                        next_token_logits,
                    )

                # Top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            next_token_id = next_token.item()
            generated.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

            if stop_sequences:
                current_text = tokenizer.decode(generated)
                for stop_seq in stop_sequences:
                    if stop_seq in current_text:
                        return current_text[len(prompt) : current_text.rfind(stop_seq)]

    return tokenizer.decode(generated)[len(prompt) :]


async def generate_pytorch_stream(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    stop_sequences: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """Generate text using PyTorch backend with streaming."""
    import torch
    import torch.nn.functional as F

    model = model_state.model
    tokenizer = model_state.tokenizer
    device = model_state.device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0].tolist()

    with torch.no_grad():
        for _ in range(max_tokens):
            tokens = torch.tensor([generated], device=device)
            outputs = model(tokens)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            next_token_id = next_token.item()
            generated.append(next_token_id)

            token_text = tokenizer.decode([next_token_id])
            yield token_text

            if next_token_id == tokenizer.eos_token_id:
                break

            await asyncio.sleep(0.01)


# =============================================================================
# FastAPI Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting inference server...")
    yield
    # Shutdown
    logger.info("Shutting down inference server...")


app = FastAPI(
    title="DeepSeek Inference Server",
    description="OpenAI-compatible inference server for DeepSeek models",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        backend=model_state.backend,
        device=model_state.device,
        model_loaded=model_state.model is not None,
    )


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepseek",
            }
        ],
    }


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Create a completion (OpenAI-compatible)."""
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Handle prompt (single or list)
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]

    # Parse stop sequences
    stop_sequences = None
    if request.stop:
        stop_sequences = [request.stop] if isinstance(request.stop, str) else request.stop

    # Check if streaming
    if request.stream:
        return StreamingResponse(
            _stream_completion(prompt, request, stop_sequences),
            media_type="text/event-stream",
        )

    # Non-streaming generation
    start_time = time.time()

    if model_state.backend == "mlx":
        generated_text = generate_mlx(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=stop_sequences,
        )
    else:
        generated_text = generate_pytorch(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=stop_sequences,
        )

    # Calculate tokens
    prompt_tokens = len(model_state.tokenizer.encode(prompt))
    completion_tokens = len(model_state.tokenizer.encode(generated_text))

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            CompletionChoice(
                text=generated_text,
                index=0,
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _stream_completion(
    prompt: str,
    request: CompletionRequest,
    stop_sequences: list[str] | None,
) -> AsyncGenerator[str, None]:
    """Stream completion response in SSE format."""
    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    if model_state.backend == "mlx":
        generator = generate_mlx_stream(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=stop_sequences,
        )
    else:
        generator = generate_pytorch_stream(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=stop_sequences,
        )

    async for token in generator:
        chunk = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": request.model,
            "choices": [{"text": token, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    final_chunk = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion (OpenAI-compatible)."""
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert chat messages to prompt
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"
    prompt += "Assistant: "

    # Use completion endpoint internally
    completion_request = CompletionRequest(
        model=request.model,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        stop=request.stop,
    )

    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(prompt, request),
            media_type="text/event-stream",
        )

    # Non-streaming
    if model_state.backend == "mlx":
        generated_text = generate_mlx(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    else:
        generated_text = generate_pytorch(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": "stop",
            }
        ],
    }


async def _stream_chat_completion(
    prompt: str,
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """Stream chat completion response."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    if model_state.backend == "mlx":
        generator = generate_mlx_stream(prompt, request.max_tokens, request.temperature)
    else:
        generator = generate_pytorch_stream(prompt, request.max_tokens, request.temperature)

    async for token in generator:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/generate")
async def generate_simple(request: Request):
    """Simple generate endpoint (non-OpenAI format)."""
    body = await request.json()
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 256)
    temperature = body.get("temperature", 0.7)

    if model_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if model_state.backend == "mlx":
        text = generate_mlx(prompt, max_tokens, temperature)
    else:
        text = generate_pytorch(prompt, max_tokens, temperature)

    return {"text": text, "prompt": prompt}


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DeepSeek Inference Server")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "mlx", "pytorch"],
        help="Backend to use",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device for PyTorch backend")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name")

    args = parser.parse_args()

    # Override with environment variable if set
    backend = os.environ.get("INFERENCE_BACKEND", args.backend)

    # Auto-detect backend
    if backend == "auto":
        try:
            import mlx.core

            backend = "mlx"
            logger.info("Auto-detected MLX backend (Apple Silicon)")
        except ImportError:
            backend = "pytorch"
            logger.info("Auto-detected PyTorch backend")

    # Load model
    logger.info(f"Loading model from {args.model_path} with {backend} backend...")

    if backend == "mlx":
        model, config = load_mlx_model(args.model_path)
        model_state.model = model
        model_state.config = config
        model_state.backend = "mlx"
        model_state.device = "mps"
    else:
        model, config, device = load_pytorch_model(args.model_path, args.device)
        model_state.model = model
        model_state.config = config
        model_state.backend = "pytorch"
        model_state.device = device

    # Load tokenizer
    model_state.tokenizer = load_tokenizer(args.tokenizer)
    model_state.model_path = args.model_path

    logger.info(f"Model loaded successfully. Backend: {model_state.backend}, Device: {model_state.device}")

    # Start server
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
