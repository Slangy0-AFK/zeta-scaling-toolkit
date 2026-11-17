#!/bin/bash
# Run the full 100,000-token contiguous context experiment
# This script installs dependencies, verifies the environment, and executes the experiment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "==============================================="
echo "Zeta Scaling Toolkit: 100K Contiguous Token Run"
echo "==============================================="
echo ""

# Step 1: Check dependencies
echo "[1/5] Verifying Python environment..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Step 2: Check model cache
echo ""
echo "[2/5] Verifying SmolLM2 model cache..."
MODEL_CACHE="$HOME/.cache/zeta-smollm-hf/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951"
if [ ! -f "$MODEL_CACHE/model.safetensors" ]; then
    echo "ERROR: SmolLM2 model not found at $MODEL_CACHE"
    echo "The model should have been cached during earlier setup."
    exit 1
fi
echo "✓ Model cache found and verified"

# Step 3: Check corpus
echo ""
echo "[3/5] Verifying corpus..."
CORPUS="$HOME/.cache/zeta-scaling-toolkit/tinyshakespeare.txt"
if [ ! -f "$CORPUS" ]; then
    echo "ERROR: Corpus not found at $CORPUS"
    exit 1
fi
echo "✓ Corpus available ($(wc -c < "$CORPUS") bytes)"

# Step 4: Set environment
echo ""
echo "[4/5] Setting environment variables..."
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "✓ Environment configured for offline operation"

# Step 5: Run experiment
echo ""
echo "[5/5] Starting 100,000-token contiguous experiment..."
echo "Output directory: $SCRIPT_DIR"
echo "Max runtime: 12 hours (43200 seconds)"
echo ""

cd "$SCRIPT_DIR"
python run_experiment.py \
  --snapshot "$MODEL_CACHE" \
  --output-dir . \
  --max-seconds 43200 \
  --min-available-mib 1024 \
  --resume

echo ""
echo "==============================================="
echo "✓ Experiment execution completed"
echo "==============================================="
echo ""
echo "Results location:"
echo "  - Gate status: $SCRIPT_DIR/CORRECTNESS_RESULTS.json"
echo "  - Run progress: $SCRIPT_DIR/progress.jsonl"
echo "  - Retrieval results: $SCRIPT_DIR/retention-results.jsonl"
echo "  - Resource usage: $SCRIPT_DIR/resource-results.json"
echo "  - Final report: $SCRIPT_DIR/FINAL_REPORT.md"
