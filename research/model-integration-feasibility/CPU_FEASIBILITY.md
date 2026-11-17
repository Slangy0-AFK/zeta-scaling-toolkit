# CPU feasibility

## Observed host, not model measurements

At inspection on 2026-11-16: two available logical CPUs, AMD EPYC 7763;
7.8 GiB total RAM, approximately 4.2 GiB available at that instant, no swap;
Python 3.14.2, PyTorch 2.15.0+cu132, CUDA unavailable, and Transformers not
installed. Host availability varies with other users and processes; cgroup
memory and CPU quota files reported no explicit limit at inspection.
Compatibility of a chosen Transformers/quantization stack with this Python,
PyTorch and CPU has not been tested.

## Memory arithmetic, not peak measurements

Weight-only lower bounds use nominal parameter count times bytes per weight;
they omit quantization metadata, transient copies, runtime, activations,
tokenizer and allocator overhead. BF16 is an estimate, not a claim that this
CPU executes BF16 efficiently. Q4 is an ideal 0.5-byte-per-weight lower bound;
actual supported formats and memory use differ.

| Model | FP32 weights | BF16 weights | Ideal Q4 weights | BF16 KV at published context |
|---|---:|---:|---:|---:|
| SmolLM2 360M | ~1.34 GiB | ~0.67 GiB | ~0.17 GiB | 320 MiB at 8,192 |
| Qwen 0.49B | ~1.83 GiB | ~0.91 GiB | ~0.23 GiB | 384 MiB at 32,768 |
| Qwen 1.54B | ~5.74 GiB | ~2.87 GiB | ~0.72 GiB | 896 MiB at 32,768 |

KV estimates assume batch 1, no sharing, two bytes per key/value element,
native KV-head storage, and no overhead:
`2 * layers * kv_heads * head_dim * cached_tokens * 2` bytes.
SmolLM2 needs 40 KiB/token (640 MiB at 8,192 in FP32); Qwen 0.5B
12 KiB/token (96 MiB at 8,192); Qwen 1.5B 28 KiB/token. A SmolLM2 Zeta
cache with 1,024 recent positions and 4 non-overlapping sinks has an ideal
BF16 KV size of about 40.2 MiB. GQA expansion, concatenation, cache copies,
other state and different dtypes can increase peak usage.

Full attention may need much more than KV storage. For example, an eager
FP32 attention score tensor for one SmolLM2 layer at 8,192 tokens is
`15 * 8192^2 * 4` bytes = 3.75 GiB; an FP32 full-sequence vocabulary-logit
tensor is another 1.5 GiB. Optimized SDPA, chunked prefill or last-token-only
logits may avoid these particular allocations, but their availability and
correctness must be verified, and Zeta's single-token-first adapter may be
slow. These are illustrative allocations, not peak measurements.

## Feasibility by stage

- **Loading:** SmolLM2 and Qwen 0.5B weight lower bounds fit the observed
  available RAM, but successful loading is unverified. Qwen 1.5B FP32 exceeds
  the observed available RAM before overhead; BF16 may fit only with
  substantial risk. Memory mapping and load-time copies vary.
- **Short inference:** SmolLM2 and Qwen 0.5B are plausible on two cores but no
  latency or numerical result has been measured. The 1.5B candidate is
  riskier; quantization may help but requires a compatible CPU backend.
- **Long context:** 8K or 32K prefill may be slow or run out of memory;
  efficient kernels, workspace allocations and a bounded adapter must be
  measured. No inference-time estimate can be responsibly extrapolated yet.
- **Full study:** many paired multi-turn trajectories, independent samples and
  beyond-boundary points are not established as practical on this host.
  Measure a separately approved short pilot before scheduling it; use a larger
  machine if resource limits dominate.

If quantization becomes necessary, use the same pinned quantized checkpoint,
format, dtype and backend for both conditions. A GGUF/ONNX runtime does not
automatically expose the Python Zeta cache; choosing it may require a separate
adapter. Quantization changes the model representation and can affect quality,
so compare like with like and retain a distinct provenance record. Do not
assume quantization, a particular memory saving, or working integration.
