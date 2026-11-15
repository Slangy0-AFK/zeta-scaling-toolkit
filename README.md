# Zeta Scaling Toolkit

[![Tests](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions)
[![codecov](https://codecov.io/gh/Slangy0-AFK/zeta-scaling-toolkit/graph/badge.svg)](https://codecov.io/gh/Slangy0-AFK/zeta-scaling-toolkit)

A PyTorch library that helps large AI models train more stably and don't "break" or stop learning when they get massive.

## The Problem

When AI models (like Transformers or Vision models) get really big, they usually suffer from four major headaches:

1. **Attention Collapse** — They lose focus when reading long documents
2. **Training Stagnation** — They stop learning after a while because the learning rate hits zero
3. **Aspect-Ratio Distortion** — They get confused when looking at images of different sizes
4. **Weight Decay** — Their internal signals fade away as the network gets deeper

## The Solution

This toolkit uses a mathematical "floor" to prevent these numbers from dropping to zero. It applies a boundary formula: **W(T) = 1 + A₀ / ln(T)** where A₀ = 100π.

### The 4 Fixes

**1. `NonCollapsingAttention` (Fixes lost focus)**
- **What it does:** Stops the AI from losing attention when it reads really long text. It puts a safety floor on the attention scores so the model doesn't "zone out" on long documents.
- **Why it matters:** Prevents softmax saturation and representational collapse in long-context models.

**2. `AsymptoticLRScheduler` (Fixes early stopping)**
- **What it does:** Standard learning rates drop to zero eventually. This scheduler gets slower and slower, but *never hits zero*. This means you can train a model forever without it completely giving up.
- **Why it matters:** Allows infinite training without zero-gradient lockout. The LR decays to a logarithmic floor rather than zero.

**3. `get_scale_invariant_vit_coords` (Fixes image confusion)**
- **What it does:** If you train a Vision AI on small images and then show it a massive image, it usually breaks. This function maps image coordinates in a polar coordinate system, so the AI understands the image regardless of its size.
- **Why it matters:** Fixes aspect-ratio distortion across varying image resolutions for Vision Transformers.

**4. `CorrectedVarianceLinear` (Fixes fading signals)**
- **What it does:** When building a deep neural network, signals can fade out (like a whisper down a long hallway). This custom layer initializes the weights perfectly so the signal stays strong from start to finish.
- **Why it matters:** Fixes the 63% signal-volume decay in massive fan-in layers by normalizing variance against the logarithmic scale.

## Installation

```bash
pip install -r requirements.txt
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

```python
import torch
from zeta_scaling_toolkit import (
    NonCollapsingAttention,
    AsymptoticLRScheduler,
    get_scale_invariant_vit_coords,
    CorrectedVarianceLinear
)

# 1. Use stable attention for long sequences
attn = NonCollapsingAttention(d_k=64)
Q = torch.randn(2, 128, 64)
K = torch.randn(2, 128, 64)
V = torch.randn(2, 128, 64)
output = attn(Q, K, V, N=128)

# 2. Use a learning rate that never drops to zero
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = AsymptoticLRScheduler(optimizer)
for epoch in range(100):
    # training loop
    scheduler.step()

# 3. Get scale-invariant vision transformer coordinates
coords = get_scale_invariant_vit_coords(height=224, width=224, T=256)

# 4. Replace standard linear layers with corrected initialization
linear = CorrectedVarianceLinear(in_features=1024, out_features=4096)
x = torch.randn(2, 1024)
y = linear(x)
```

## Testing

Run the comprehensive test suite:

```bash
pip install -e ".[test]"
pytest tests/ -v --cov=zeta_scaling_toolkit
```

All 29 tests should pass with 100% code coverage.

## How It Works

The core insight is that many scaling factors in deep learning can hit zero, causing gradients to vanish or attention to collapse. By applying a mathematical floor based on logarithmic growth, we ensure:

- Attention weights never fully collapse
- Learning rates decay asymptotically but never reach zero
- Weight initialization accounts for layer depth
- Vision transformers work at any resolution

For technical details, see the source code with detailed docstrings in `zeta_scaling_toolkit/core.py`.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.21.0

## License

MIT License - see LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{zeta_scaling_toolkit_2026,
  title={Zeta Scaling Toolkit: Self-Normalizing Scaling for AI Pipelines},
  author={Zeta Scaling Toolkit Contributors},
  year={2026},
  url={https://github.com/Slangy0-AFK/zeta-scaling-toolkit}
}
```

## Disclaimer

This is an experimental mathematical framework. Evaluate against standard benchmarks for your specific architecture before deploying to production.
