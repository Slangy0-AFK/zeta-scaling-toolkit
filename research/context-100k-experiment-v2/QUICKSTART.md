# 100,000-Token Contiguous Experiment: Complete Setup & Execution Guide

**Status:** Ready to run. Preflight passed. Contiguous run awaiting execution.

## What's Already Done ✓

- ✓ Preflight correctness gates all passed
  - Native model loading verified
  - Position adapter passes numerical parity tests
  - Sequential cache continuation validated
  - Future-token invariance confirmed
  - Bounded vs full cache: tolerance within spec
  - Masked attention probability correct
  - Cache shapes and positions validated
  - RoPE at high positions passes FP64 reference check
- ✓ SmolLM2 model (360M) cached and verified
  - Revision: a10cc1512eabd3dde888204e902eca88bddb4951
  - Weights validated with SHA-256
  - Tokenizer in place
- ✓ Experiment protocol defined
  - 100,000 contiguous tokens with KV retention (64 sinks + 64 window)
  - Fact insertion at positions 128, 2048, 8000, 16000, 25000, 50000, 75000, 90000
  - Checkpoints at 2048-token intervals and key milestones
  - Deterministic seeding (20261117) for reproducibility
- ✓ Corpus available (1.1M tiny Shakespeare)
- ✓ Correctness gates defined with tolerances
- ✓ Resource tracking instrumented

## What's Missing (and how to add it)

### 1. ✓ Run Experiment Script
**Status:** ADDED
**File:** `RUN_100K_EXPERIMENT.sh`
**What it does:**
- Verifies Python environment and PyTorch
- Checks model cache and corpus availability
- Sets offline environment variables
- Runs the experiment with proper timeouts and memory monitoring
- Generates final report and results

### 2. ✓ Dependencies Verification
**Status:** READY
**File:** `requirements-lock.txt` (already exists with hashed dependencies)
**How to verify:**
```bash
pip list | grep -E "torch|transformers|safetensors"
```

### 3. ✓ Experiment State Tracking
**Status:** READY
**Files:** 
- `progress.jsonl` (created on first run)
- `checkpoints/` directory (created on first run)
- `CORRECTNESS_RESULTS.json` (tracks gate status)

### 4. ✓ Retrieval Evaluation Framework
**Status:** INCLUDED in `run_experiment.py`
**What it does:**
- Inserts facts at deterministic positions
- After 100K tokens, queries each fact
- Records exact-match accuracy
- Reports which positions were retained/evicted
- Compares against native-only baseline

### 5. ✓ Resource Monitoring
**Status:** INCLUDED in `run_experiment.py`
**Tracks:**
- Wall-clock time
- CPU time
- Throughput (tokens/second)
- Peak resident memory
- Peak GPU memory (if applicable)
- KV-cache size and evolution

### 6. ✓ Resume & Checkpoint Recovery
**Status:** INCLUDED in `run_experiment.py`
**Features:**
- Atomic checkpoint writes (temp file + replace)
- Provenance verification before resume
- Cache position validation
- Deterministic stream reconstruction
- Can resume from any checkpoint

## Quick Start: Run 100K Contiguous Experiment

### Prerequisites (verify all are present)

```bash
# 1. Check Python and PyTorch
python --version  # Should be 3.9+
python -c "import torch; print(torch.__version__)"

# 2. Check SmolLM2 model
ls ~/.cache/zeta-smollm-hf/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951/model.safetensors

# 3. Check corpus
ls ~/.cache/zeta-scaling-toolkit/tinyshakespeare.txt  # Should exist

# 4. Check transformers compatibility
python -c "import transformers; print(transformers.__version__)"
```

### Option A: Automated Script (Recommended)

```bash
cd /workspaces/zeta-scaling-toolkit
chmod +x research/context-100k-experiment-v2/RUN_100K_EXPERIMENT.sh
research/context-100k-experiment-v2/RUN_100K_EXPERIMENT.sh
```

### Option B: Manual Execution

```bash
cd /workspaces/zeta-scaling-toolkit/research/context-100k-experiment-v2

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python run_experiment.py \
  --snapshot ~/.cache/zeta-smollm-hf/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951 \
  --output-dir . \
  --max-seconds 43200 \
  --min-available-mib 1024 \
  --resume
```

### Option C: Resume Interrupted Run

Same command as Option B — the `--resume` flag automatically:
- Checks for existing progress
- Resumes from latest checkpoint if incomplete
- Skips repeated runs if already completed

## Expected Runtime & Resource Usage

- **Wall time:** ~2-6 hours (depending on CPU speed)
- **CPU cores used:** All available (100% CPU)
- **Memory:** ~2-4 GiB peak
- **Environment:** CPU-only (no GPU required)

## Monitoring Progress

While running, check progress file:

```bash
# Real-time progress
tail -f research/context-100k-experiment-v2/progress.jsonl | python -m json.tool

# Check gate status
cat research/context-100k-experiment-v2/CORRECTNESS_RESULTS.json | python -m json.tool

# Verify checkpoints
ls -lh research/context-100k-experiment-v2/checkpoints/
```

## Expected Outputs

After successful completion:

```
research/context-100k-experiment-v2/
├── CORRECTNESS_RESULTS.json        # Updated with contiguous_run_gate status
├── progress.jsonl                   # Token-by-token progress log
├── checkpoints/                     # Snapshots at key positions
│   ├── checkpoint-8193.pt
│   ├── checkpoint-25000.pt
│   ├── checkpoint-50000.pt
│   ├── checkpoint-65537.pt          # Critical: Tests > native 8192
│   ├── checkpoint-100000.pt
│   └── ...
├── retention-results.jsonl           # Fact retrieval results
├── native-baseline.jsonl             # 8064-token baseline queries
├── resource-results.json             # Timing, memory, throughput
└── FINAL_REPORT.md                   # Human-readable summary
```

## Validation Criteria

The experiment passes if:
1. ✓ Preflight gates all pass (already confirmed)
2. ✓ Contiguous run reaches position 65,537 without numerical instability
3. ✓ Checkpoints are bit-reproducible from same input
4. ✓ Retrieved facts at least partially match inserted values
5. ✓ Cache retention behavior is consistent with KV window specification

## Failure Scenarios (Handled)

- **Interrupted run:** Use `--resume` to continue from checkpoint
- **Out of memory:** Adjust `--min-available-mib` to require more free RAM
- **Stale checkpoint:** Rename `checkpoints/` directory to restart fresh
- **Network error:** Experiment runs offline; no network access needed

## Next Steps After Completion

1. Review `CORRECTNESS_RESULTS.json` for pass/fail status
2. Examine `retention-results.jsonl` for retrieval accuracy by position
3. Check `resource-results.json` for throughput and memory insights
4. Read `FINAL_REPORT.md` for human-readable interpretation
5. Compare against `native-baseline.jsonl` for context-retention claims

## Questions?

Refer to:
- `EXPERIMENT_PROTOCOL.md` — Protocol details and correctness gates
- `ENVIRONMENT.md` — Dependency versions and environment setup
- `PROVENANCE.md` — Model revision, corpus hash, artifact provenance
- `COMPATIBILITY_REPORT.md` — Security and compatibility considerations
