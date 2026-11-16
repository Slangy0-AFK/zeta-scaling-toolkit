# Query-dependent associative retrieval protocol

## Status and scope

This protocol adds an independent synthetic retrieval task. It does not alter
historical successor-prediction results, existing candidate settings, model
architecture, or eligibility gates. The included smoke command only checks
generation, oracle labels, evaluator plumbing, and model forward shapes; it
does not train a model and is not experiment evidence.

The task can establish whether a model trained on 256-token examples can use
full causal attention to emit a randomly assigned value selected by a late
query at longer sequence lengths. It cannot establish natural-language recall,
semantic understanding, multi-hop reasoning, or general-purpose context
retention.

The existing full-attention `rope-log-full`, `rope-full`, and ALiBi
implementations can in principle support query-dependent retrieval: the final
query representation can attend to earlier key/value tokens. This is an
architectural capability, not evidence that the current models learn the
behavior. The 64-token-window candidate is deliberately retained only as a
local diagnostic. No trained retrieval result is claimed by this
implementation or its smoke artifact.

## Example construction

The tokenizer is a controlled integer vocabulary with disjoint classes:

- 32 unrelated filler tokens
- 64 key tokens
- 64 value tokens
- one stored-pair marker and one query marker

Each example contains 16 stored pairs for the primary length curve. Keys and
values are sampled without replacement within an example, then paired and
ordered independently for every example. Pair markers and the disjoint token
classes make pair boundaries explicit. Filler is sampled only from the noise
class. The query marker and its key occupy the final two context positions;
the prediction target is the next token, namely the paired value.

The queried key occurs exactly once among stored keys and once at the query.
Each value occurs once in the input, in its pair. The deterministic oracle
parses the stored pairs and looks up the final query key; it validates labels
but is never used as a model prediction. The generated examples record pair
order, target and query positions, target distance, and position bucket in a
separate example-ID registry.

Evaluation examples are stratified equally across early, middle, and late
target positions (the default 300 examples provide 100 examples per bucket).
Target distance is also recorded and binned. Because the query is at the end,
absolute target position and distance are correlated; both are reported, and
their marginal scores must not be interpreted as independent causal effects.

## Training, tuning, and reporting

Training context length is fixed at 256. The primary context-length curve uses
the same 16 stored pairs at 256, 512, 1,024, 2,048, and 4,096 tokens. The
secondary distractor-count evaluation holds context length at 4,096 and uses
8, 32, and 64 stored pairs. Its tuning selection remains based only on the
16-pair, 4,096-token tuning set.

Train, sanity, tuning, and report examples use separate deterministic split
namespaces and seed ranges. A content-derived ID registry rejects an example
appearing in more than one split. The same generated examples are evaluated
for every method, and each per-seed result includes the evaluation fingerprint
and denominators.

The initial method set is `rope-log-full`, `rope-full`, ALiBi, and
`rope-log-window`. Candidate indices 0 and 1 are resolved through the existing
candidate function; their settings are not changed. The windowed model is a
local-window diagnostic, not a full-context retrieval method: its 64-token
window and four initial sinks cannot expose a pair thousands of tokens away.

The sanity phase evaluates each method/candidate on 300 independent
256-token examples for 10 seeds. The three confirmatory methods
(`rope-log-full`, `rope-full`, and ALiBi) must each have an eligible candidate:
at least 8/10 seeds must achieve at least 0.95 exact-match accuracy. If any
confirmatory method fails this gate, tuning/report phases stop. The
`rope-log-window` diagnostic is evaluated but is not required to pass a
learnability gate that its local attention cannot satisfy; candidate 0 is
fixed for this diagnostic. Tuning evaluates eligible confirmatory candidates
on separate 4,096-token, 16-pair examples and selects by mean exact-match
accuracy. Reporting retrains the frozen candidates on new reporting seeds and
evaluates all prescribed context and distractor settings. Passing the
256-token gate establishes learnability only, not long-context retrieval.

Controls are a deterministic uniform answer from the 64 value tokens and an
input-ablation diagnostic that replaces stored pairs with filler while
preserving the query and context length. The latter is not a separately
trained model and may be out-of-distribution; interpret it as a reliance
diagnostic, not as the primary baseline.

The summary reports per-seed records, exact-match accuracy, per-position,
per-distance, and per-distractor counts, paired differences, win counts, and
95% seed-bootstrap intervals. Intervals use the existing deterministic
bootstrap helper and 10 reporting seeds; they remain uncertain at this sample
size, so per-seed values and denominators must accompany every aggregate.

## Limitations

- The token grammar is synthetic and fixed; success would show associative
  retrieval on this task, not natural-language recall or general reasoning.
- Target position and target-to-query distance are correlated by construction.
- The pair-ablation control is an input distribution shift, so its score is
  diagnostic rather than a standalone causal estimate.
- Ten reporting seeds provide limited precision; include per-seed scores and
  sample denominators with aggregate confidence intervals.
- The runner checkpoints after a full training fit, not mid-training.
  An interrupted unfinished fit restarts from its initial seed; completed
  checkpoints with matching manifest sidecars can resume evaluation. If
  interruption leaves only one of a checkpoint and sidecar, resume fails
  closed and requires manual investigation.

## Records and provenance

Output defaults to `context-retrieval-v1.jsonl`, but that prefix is already
occupied by the interrupted CPU attempt in this workspace. Choose a fresh
prefix. Completed model checkpoints are atomically written under a matching
checkpoint directory; each JSONL row includes the checkpoint
SHA-256, source/environment manifest and hash, protocol/configuration, method,
candidate settings, seed, split fingerprints, and per-length metrics.
The matching `.examples.jsonl` file records example IDs, split, pair order,
and positions. Each completed fit has a manifest sidecar binding checkpoint
hash, configuration hash, trial identity and source/environment fingerprint.
Result rows are written atomically after each trial and include a row hash.
Hashes detect accidental corruption, not malicious tampering.

New runs require a previously unused result path, registry, checkpoint
directory and experiment-status file. A POSIX advisory lock is held for the
entire phase; cooperating processes using the same output prefix cannot run
concurrently. External programs that ignore locks and network filesystems with
unreliable locking are not protected. Do not share a checkpoint directory
between experiments. The status file records phase `running`, `completed`, or
`incomplete`; `running` after a process crash is incomplete, not a result.
`--resume` is required for subsequent invocations, including tune and report,
and rejects mismatched configuration, source, environment or paths. Existing
unmanaged historical results cannot be resumed or rewritten; use a new
prefix. Preserve lock files; the operating system releases their locks when
the process exits.
`completed` means the phase executed, not that any eligibility gate passed.

## Commands

Run the new unit tests and the non-training smoke check:

```bash
python -m pytest -q tests/test_context_retrieval.py
python benchmarks/context_retrieval.py --phase smoke \
  --output benchmarks/context-retrieval-smoke.json
```

The smoke artifact is explicitly labeled non-research and refuses to overwrite
an existing file.

The following full phases use an illustrative fresh prefix and are documented
for a separately approved run only; they were not run as part of implementation.
First verify all output paths are unused:

```bash
python benchmarks/context_retrieval.py --phase sanity \
  --steps 3000 --batch-size 16 --eval-examples 300 \
  --output benchmarks/context-retrieval-example-v2.jsonl
python benchmarks/context_retrieval.py --phase tune \
  --steps 3000 --batch-size 16 --eval-examples 300 \
  --output benchmarks/context-retrieval-example-v2.jsonl --resume
python benchmarks/context_retrieval.py --phase report \
  --steps 3000 --batch-size 16 --eval-examples 300 \
  --output benchmarks/context-retrieval-example-v2.jsonl --resume
python benchmarks/context_retrieval.py --phase summary \
  --steps 3000 --batch-size 16 --eval-examples 300 \
  --output benchmarks/context-retrieval-example-v2.jsonl --resume
```

Do not start tune/report unless the sanity phase completes and the 0.95,
8-of-10 eligibility gate passes. No 2,000-step run is part of this protocol.

## GPU preparation and preflight

The runner defaults to CPU. `--device cuda` explicitly selects CUDA device 0;
it fails if CUDA is unavailable, rather than falling back to CPU. Model and
training batches are moved to that device; the training pool and seeded batch
indices remain on CPU. Evaluation constructs inputs on the model device.
Completed model checkpoints contain CPU tensors and can be loaded with
`torch.load(path, map_location="cpu", weights_only=True)` or loaded into a
model on CUDA. These are model-only checkpoints, not resumable training state.

Examples and batch indices use deterministic CPU generators and model
initialization uses `torch.manual_seed`. Identical seeds do not guarantee
bitwise-identical results across CPU/GPU, PyTorch versions, CUDA kernels or
hardware. The runner records device, dtype, TF32 and deterministic-algorithm
settings; it does not force deterministic algorithms. GPU compatibility and
numerical reproducibility must be checked on the actual hardware before the
scientific run; do not combine CPU and GPU records.

Run this diagnostic without training:

```bash
python benchmarks/context_retrieval_hardware.py
```

It reports GPU model, compute capability and free memory if CUDA is available,
plus a parameter/gradient/optimizer-state memory lower bound. This is not a
peak estimate: activations, attention workspaces, allocator reserve and models
held for the other candidates also consume GPU memory.

Only on an available CUDA device, run a separate non-research 20-step
performance smoke test:

```bash
python benchmarks/context_retrieval_hardware.py \
  --smoke-prefix benchmarks/context-retrieval-gpu-smoke --smoke-steps 20
```

It saves a separate smoke checkpoint and JSON, refuses to overwrite either,
checks CPU and CUDA checkpoint round trips, and reports steps/second, process
peak RSS and peak allocated/reserved GPU memory. Its runtime extrapolation is
for one method only and excludes evaluation and I/O; check that each method
fits and treat the full-run estimate as approximate.

The prior CPU attempt's `sanity-rope-log-full-0-200.pt` is an orphan without
a result record or manifest sidecar. The runner refuses to treat it as a
completed trial. Preserve that checkpoint, its example registry and
incomplete-status file. For a separately approved GPU gate, use a fresh,
unused output prefix and checkpoint directory:

```bash
python benchmarks/context_retrieval.py --phase sanity --device cuda \
  --steps 3000 --batch-size 16 --eval-examples 300 --eval-batch-size 8 \
  --methods rope-log-full rope-full alibi rope-log-window \
  --output benchmarks/context-retrieval-gpu-v1.jsonl \
  --checkpoint-dir benchmarks/context-retrieval-gpu-v1-checkpoints
```

This command requires explicit approval; the diagnostic and smoke test do not
constitute permission to launch it.
