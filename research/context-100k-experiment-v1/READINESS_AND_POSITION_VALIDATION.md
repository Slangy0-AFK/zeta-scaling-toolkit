# 100K experiment readiness and position validation

**Decision: BLOCKED. Do not run the 100,000-token experiment.** This review
found a security-versus-compatibility blocker in the recorded inference stack,
and the available prototype's 65,537-token maximum has not been validated
(the historical contiguous run reached 8,193 tokens). No packages were
installed, no model was downloaded, no position-extension code was added, and
no experiment was run.

## Scope and active branch

- Active checkout: `portfolio/research-lab-presentation`, HEAD
  `e7f4dd3f8c706157b91198436d3c368da9f99def`; it was clean at inspection.
- The requested source commit is
  `ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed`, titled “Add SmolLM2 position
  extension feasibility.” It is on
  `copilot/experiment-infrastructure-phase-2`, not in the active checkout's
  local object database or available local refs. Its metadata and files were
  read through GitHub without merging, resetting, or changing branches.
- The active branch contains the preflight/result artifacts under this
  directory, including `EXPERIMENT_PROTOCOL.md`, `reproduction.md`, and
  `resource-measurements.json`. They were preserved.
- The smallest relevant prototype source set is the original
  `research/smolLM2-integration-v1/adapter.py`, plus
  `research/smollm2-integration-v1/position_extension.py`,
  `validate_positions.py`, and `validate_contiguous.py`. The corresponding
  `tests/test_smollm2_adapter.py` and `tests/test_smollm2_positions.py` are
  needed to retain its contract checks. The commit also contains reports and
  result records; do not replace the active branch's existing research
  artifacts with those files. These sources were not copied into the active
  branch because their dependency contract is incompatible with the
  currently available patched Transformers candidate.

## Dependency security and compatibility

### Recorded environment

The prior prototype's result records identify:

| Component | Recorded identity |
|---|---|
| Python | `3.14.2` |
| PyTorch | `2.15.0+cu132` |
| Transformers | `4.52.2` |
| Model | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| Model and tokenizer revision | `a10cc1512eabd3dde888204e902eca88bddb4951` |
| Inference | FP32, CPU, eager attention, two CPU threads |

The model and tokenizer hashes recorded by the prior run are respectively
`e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e` and
`9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.
These hashes were not recomputed here. Both identities refer to the pinned
model revision above; no alternative model or tokenizer is proposed.

### Security findings

Transformers `4.52.2` is not a suitable security-approved pin for a new run.
The advisory review found:

| Advisory | Impact and version finding |
|---|---|
| [GHSA-29pf-2h5f-8g72 / CVE-2026-4372](https://github.com/advisories/GHSA-29pf-2h5f-8g72) | Critical model-loading RCE; versions before 5.3.0 are affected. |
| [GHSA-x9r9-c232-4q39 / CVE-2026-80047](https://github.com/advisories/GHSA-x9r9-c232-4q39) | Custom-generation-code loading can write remote Python before consent; the reviewed range includes 4.49.0–5.8.1, with no fix identified in the review. |
| [GHSA-xrqw-3rrv-vx5w / CVE-2026-9856](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w) | Processor `save_pretrained` path traversal; fixed in 5.10.0. |
| [GHSA-fgcw-684q-jj6r / CVE-2026-5241](https://github.com/advisories/GHSA-fgcw-684q-jj6r) | LightGlue model-loading remote-code execution; fixed in 5.5.0. |

These are package-wide findings, not claims that each vulnerable path is
reachable in the intended SmolLM2 inference flow. The unpatched custom
generation path must not be used, and `trust_remote_code` must remain
disabled.

Transformers `5.23.0` was identified in the research review as a current
candidate whose metadata supports Python 3.14 and requires PyTorch `>=2.6`.
The recorded PyTorch version meets that stated minimum. This is a candidate,
not an approved runtime: metadata compatibility does not establish model,
checkpoint, adapter, or inference correctness. Avoid the known custom
generation path regardless of version.

### Why there is no compatible dependency lock

The original adapter explicitly requires `transformers==4.52.2`, while
`4.52.2` is not acceptable under the reviewed security findings. The newer
Transformers Cache API is not a drop-in replacement:

- The prototype requires `Cache.update(..., cache_kwargs=...)` to check the
  explicit `cache_position`; the inspected current Llama implementation no
  longer supplies those cache kwargs.
- The prototype uses `get_max_cache_shape()`, which is absent from the newer
  API; cache layers and mask-size handling have also changed.
- Newer cache layers derive positions internally. The old adapter's explicit
  position/eviction assumptions have not been ported or validated against
  that behavior.
- Llama RoPE configuration moved from legacy `rope_theta`/`rope_scaling`
  fields to `rope_parameters`. Legacy migration exists in the newer code,
  but the pinned checkpoint load and resulting RoPE values were not tested.
- SmolLM2 checkpoint tensor conversion/loading under the newer model
  implementation has not been smoke-tested.

Thus a Transformers version bump would break the adapter contract, while
keeping the old version leaves a security blocker. The exact transitive
versions and hashes for `huggingface_hub`, `tokenizers`, `safetensors`, and
other required packages were not resolved. A published top-level wheel hash
is not a complete lock. No lock file is supplied because claiming a
reproducible, compatible, hash-locked environment without a complete
resolution and smoke test would be misleading.

**Required next step:** port the cache adapter to a selected patched
Transformers release, resolve and hash-lock the full dependency graph, then
test loading the exact pinned model and tokenizer with `trust_remote_code`
disabled. Preserve native RoPE settings and verify the loaded configuration
before any position-extension work.

## 65,537-position ceiling

The ceiling in the prototype is an explicit software limit, not a discovered
65,537-entry model buffer:

1. The default adapter copies the checkpoint's
   `max_position_embeddings=8192` into its own limit and rejects an input
   whose next absolute position exceeds it. This original 8,192 safety guard
   must remain unchanged.
2. `ExperimentalPositionAdapter` separately requires
   `max_positions <= 65537`. Its limit is an exclusive count: from position
   zero it permits token positions `0..65536` (65,537 tokens), but not a token
   at position 65,537.
3. `validate_contiguous.py` independently rejects lengths above 65,537 and
   constructs the adapter with `max_positions=65537`.
4. The prototype uses `torch.long` absolute position IDs, dynamically sized
   single-token attention masks, and the model's native unscaled RoPE. The
   inspected Transformers 4.52.2 path did not reveal a fixed 65,537-position
   RoPE table or attention-mask allocation. This is source inspection of that
   version, not validation of a newer runtime.

Accordingly, merely changing a maximum would remove two guards but would not
establish correctness in the patched Transformers API. No 100,000-position
configuration was implemented. The checkpoint's native `rope_theta` and
unscaled RoPE were not changed.

## Staged position validation

No staged run was performed for this task. The local checkout lacks the
required inference packages and model snapshot, and the dependency/API
compatibility blocker was not resolved. Results below distinguish previous
bounded evidence from requested tests:

| Requested stage | Evidence/status |
|---:|---|
| 65,537 tokens | Prior record reports one contiguous 8,193-token run only; no 65,537-token run. |
| 65,538 tokens | Not run. |
| 70,000 tokens | Not run. |
| 75,000 tokens | Not run. |
| 90,000 tokens | Not run. |
| 100,000 tokens | Not run; explicitly prohibited until prerequisites pass. |

Historical evidence, not rerun here:

- A five-token original-range parity test reported bit-identical output
  between the original and experimental adapters and maximum absolute
  difference `8.5115e-05` against native multi-token prefill
  (`atol=2e-4`, `rtol=1e-4`).
- Synthetic empty-history offset probes ended at position 65,536. Their
  independent FP64 RoPE reference had maximum absolute error `0.0012452`
  under a `0.003` tolerance. These are isolated probes with no preceding
  history, not contiguous long-context tests.
- The historical 8,193-token contiguous run reportedly processed actual
  tokenizer-produced IDs from position zero, maintained the expected
  4-sink/64-recent cache positions across chunks, and produced finite logits.
  It did not establish information retention or native failure boundary.

For the requested high positions, position-ID progression, independent RoPE
reference, causal masking and future-token invariance, continuation across
repeated eviction, finite logits and numerical stability all remain
unvalidated. A 100K opt-in must not be exposed until these checks pass
progressively on the exact patched and locked runtime.

## Resource feasibility and stop limits

The prior preflight record, from a different run, captured 4 logical CPUs,
about 14.2 GiB available RAM, and 86.8 GB free root-disk space. Its historical
8,193-token inference record reported a peak process RSS of about 2,812.8
MiB, 1,290.82 seconds wall time, and 5,587,968 bytes of bounded cache tensors.
These are not measurements from this task.

For the recorded 32-layer, 5-KV-head, 64-dimension FP32 model, a batch-one
100,000-token full-history K/V cache is approximately 8.19 GB decimal
(about 7.63 GiB), before model weights, activations, allocator overhead, and
other process memory. This is a calculation, not a measured native-cache
peak. The bounded cache's historical size does not estimate runtime or
demonstrate recall.

The prior runner stopped chunks at 1,024 MiB available RAM and required at
least 3,584 MiB available before model loading. Those are existing guard
values, not a validated safe threshold for a 100K run. No throughput,
prefill/decode rate, cache-update overhead, or 100K runtime was measured.
The previously mentioned 4.38-hour linear estimate is not a reliable runtime
prediction and must not be used as a budget. A future run needs a fresh
resource benchmark and explicit RAM and wall-clock abort thresholds before
progressing beyond bounded validation.

The recorded model footprint and prior free disk suggest provisioning may be
possible, but this was not tested in the current environment. At inspection,
the current sandbox was Python 3.12.3, without `torch`, `transformers`,
`tokenizers`, or `safetensors`, and without a local model snapshot. No
installation or model download was attempted.

## Experiment prerequisites that remain

Position acceptance alone is not evidence of usable context. The primary
experiment still requires a measured native failure boundary and a
pre-specified, matched native-versus-Zeta evaluation: unpredictable facts
distributed across the conversation, queries about evicted early facts,
exact-match scoring, and independent evaluation sets. No such experiment
was run or authorized by this readiness check.

## Command and release decision

**There is no safe 100K command to provide or run yet.** The current contiguous
runner has a hard maximum of 65,537 tokens and the adapter is incompatible
with the reviewed patched Transformers cache contract. Do not raise the
runner limit or invoke it with 100,000. First complete the dependency port,
full hash lock, model/tokenizer smoke test, and staged position validations;
then design and approve the separate information-retention experiment.

Final status: **BLOCKED**.
