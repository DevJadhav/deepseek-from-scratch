#!/bin/bash
#
# Train Tiny DeepSeek Model (~10M parameters)
# ==========================================
#
# This script runs the complete training pipeline:
#   1. Setup environment
#   2. Download TinyStories dataset
#   3. Train model with MLX backend
#   4. Run inference
#
# Usage:
#   ./scripts/run_tiny_training.sh              # Full training
#   ./scripts/run_tiny_training.sh --quick      # Quick test (100 steps)
#   ./scripts/run_tiny_training.sh --sample     # Use sample data only
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    Tiny DeepSeek Training Pipeline (~10M parameters)${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Parse arguments
QUICK_TEST=false
SAMPLE_ONLY=false
SKIP_DATA=false
SKIP_TRAIN=false
MAX_STEPS=10000

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_TEST=true
            MAX_STEPS=100
            shift
            ;;
        --sample)
            SAMPLE_ONLY=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Configuration
DATA_DIR="./data/stories"
OUTPUT_DIR="./checkpoints/tiny-mlx"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Project root:  $PROJECT_ROOT"
echo "  Data dir:      $DATA_DIR"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Max steps:     $MAX_STEPS"
echo "  Quick test:    $QUICK_TEST"
echo "  Sample only:   $SAMPLE_ONLY"
echo ""

# Step 1: Check uv and sync dependencies
echo -e "${BLUE}[Step 1/5] Checking uv and syncing dependencies...${NC}"

if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

UV_VERSION=$(uv --version)
echo "  UV version: $UV_VERSION"

# Sync dependencies
echo "  Syncing project dependencies..."
uv sync

echo -e "${GREEN}  ✓ Dependencies synced${NC}"
echo ""

# Step 2: Prepare data
echo -e "${BLUE}[Step 2/5] Preparing data...${NC}"

if [ "$SKIP_DATA" = true ]; then
    echo "  Skipping data preparation (--skip-data)"
elif [ -f "$DATA_DIR/train/stories.jsonl" ]; then
    TRAIN_LINES=$(wc -l < "$DATA_DIR/train/stories.jsonl" | tr -d ' ')
    echo "  Training data already exists: $TRAIN_LINES samples"
else
    echo "  Downloading TinyStories dataset..."
    
    if [ "$SAMPLE_ONLY" = true ]; then
        echo "  (Using sample data only)"
        uv run python scripts/download_tinystories.py --sample-only --num-samples 5000
    else
        uv run python scripts/download_tinystories.py
    fi
fi

echo -e "${GREEN}  ✓ Data ready${NC}"
echo ""

# Step 3: Train model
echo -e "${BLUE}[Step 3/5] Training model...${NC}"

if [ "$SKIP_TRAIN" = true ]; then
    echo "  Skipping training (--skip-train)"
else
    TRAIN_ARGS="--data-dir $DATA_DIR --output-dir $OUTPUT_DIR --max-steps $MAX_STEPS"
    
    if [ "$QUICK_TEST" = true ]; then
        TRAIN_ARGS="$TRAIN_ARGS --quick-test"
    fi
    
    echo "  Running: uv run python scripts/train_tiny.py $TRAIN_ARGS"
    echo ""
    
    uv run python scripts/train_tiny.py $TRAIN_ARGS
fi

echo -e "${GREEN}  ✓ Training complete${NC}"
echo ""

# Step 4: Verify checkpoint
echo -e "${BLUE}[Step 4/5] Verifying checkpoint...${NC}"

CHECKPOINT_DIR="$OUTPUT_DIR/final"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "  Checkpoint found at: $CHECKPOINT_DIR"
    ls -la "$CHECKPOINT_DIR"
else
    echo -e "${YELLOW}  Warning: Final checkpoint not found${NC}"
    # Try to find any checkpoint
    CHECKPOINT_DIR=$(find "$OUTPUT_DIR" -name "config.json" -exec dirname {} \; 2>/dev/null | head -1)
    if [ -n "$CHECKPOINT_DIR" ]; then
        echo "  Using checkpoint: $CHECKPOINT_DIR"
    fi
fi

echo -e "${GREEN}  ✓ Checkpoint verified${NC}"
echo ""

# Step 5: Test inference
echo -e "${BLUE}[Step 5/5] Testing inference...${NC}"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "  Running sample inference..."
    echo ""
    
    uv run python scripts/inference.py \
        --checkpoint "$CHECKPOINT_DIR" \
        --prompt "Once upon a time, there was a little" \
        --max-tokens 50 \
        --temperature 0.7
else
    echo -e "${YELLOW}  Skipping inference (no checkpoint found)${NC}"
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}    Training Pipeline Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Run inference:"
echo "     uv run python scripts/inference.py --checkpoint $OUTPUT_DIR/final --interactive"
echo ""
echo "  2. Continue training:"
echo "     uv run python scripts/train_tiny.py --resume $OUTPUT_DIR/final --max-steps 20000"
echo ""
echo "  3. Use with ray_pipeline:"
echo "     uv run python -m ray_pipeline.cli run --config configs/tiny_mlx_full.json"
echo ""
