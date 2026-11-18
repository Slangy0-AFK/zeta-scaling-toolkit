# Zeta Evidence Checklist

## Scope and result

Audit of fresh run `zeta100k-fresh-20261117-5bc2985bbe99`, completed
2026-11-18. The runner reached 100,000 tokens and exited with code 0. No new
run or long benchmark was started. Final checkpoint validation, log review,
and provenance hashing were read-only; the final verification is recorded in
[monitoring-20261117.jsonl](research/zeta100k-fresh-20261117-5bc2985bbe99/qualification/monitoring-20261117.jsonl).

Status applies only to this model, configuration, and run. The earlier
[100K pipeline report](research/zeta100k-fresh-20261117-5bc2985bbe99/100K_PIPELINE_INTEGRITY_REPORT.md),
[run protocol](research/zeta100k-fresh-20261117-5bc2985bbe99/FRESH_RUN_PROTOCOL.md),
and
[qualification recheck](research/zeta100k-fresh-20261117-5bc2985bbe99/QUALIFICATION_RECHECK_20261117.md)
contain pre-completion snapshots. Use the final result files and monitoring log
linked below for completion status.

## 1. Long-stream processing

| Requirement | Status | Evidence |
|---|---|---|
| 100,000 contiguous tokens completed | **PASS** | [resource-results.json](research/zeta100k-fresh-20261117-5bc2985bbe99/resource-results.json), [FINAL_REPORT.md](research/zeta100k-fresh-20261117-5bc2985bbe99/FINAL_REPORT.md), and [progress.jsonl](research/zeta100k-fresh-20261117-5bc2985bbe99/progress.jsonl) report 100,000 tokens, last position 99,999, completed status, and runner exit code 0 in the final monitoring record. |
| Final runner exit status | **PASS** | The runner's shell session exited with code 0 after reporting `100K PROCESSING PASS`; final state is also recorded in [monitoring-20261117.jsonl](research/zeta100k-fresh-20261117-5bc2985bbe99/qualification/monitoring-20261117.jsonl). |
| Position continuity | **PASS** | All 56 progress records were checked for monotonically increasing counts, position 0 as the start, and `last_position == processed_tokens - 1`. All corresponding checkpoint positions were validated. See [progress.jsonl](research/zeta100k-fresh-20261117-5bc2985bbe99/progress.jsonl) and [monitoring-20261117.jsonl](research/zeta100k-fresh-20261117-5bc2985bbe99/qualification/monitoring-20261117.jsonl). |
| Checkpoint integrity and pinned metadata | **PASS** | All 56 checkpoints passed hash, stream/model/revision metadata, retained-position, tensor-shape, and finite-cache checks. Final checkpoint path: `research/zeta100k-fresh-20261117-5bc2985bbe99/checkpoints/position-100000.safetensors` (large checkpoints are kept locally and ignored by Git); SHA-256 `43e7eee27e0287a01f631025f3d11b8311632a58de6c1faaf1943d2097e7a076`. |
| Pinned model, tokenizer, corpus, and stream provenance | **PASS** | Hashes were independently recomputed and matched [PROVENANCE.md](research/zeta100k-fresh-20261117-5bc2985bbe99/PROVENANCE.md), [resource-results.json](research/zeta100k-fresh-20261117-5bc2985bbe99/resource-results.json), and stream metadata: weights `e6bffe...bf86e`, tokenizer `9ca9ac...8e47c`, corpus `86c4e6...565ed`, stream `c9c43f...da322`. Model revision: `a10cc1512eabd3dde888204e902eca88bddb4951`. |
| Bounded 128-position KV cache | **PASS** | Final cache is 10,518,528 bytes, with 64 sink and 64 recent positions. Every checkpoint's 64 key/value tensors had shape `[1, 5, 128, 64]`; final retained positions were 0–63 and 99,936–99,999. See [resource-results.json](research/zeta100k-fresh-20261117-5bc2985bbe99/resource-results.json), the local checkpoint path above, and [run_experiment.py](research/context-100k-experiment-v2/run_experiment.py). |
| Finite logits and cache values throughout the stream | **PASS** | The runner checks stream logits and cache state during processing; it completed without a numerical exception. All saved checkpoint cache tensors were also checked for finite values. See [run_experiment.py](research/context-100k-experiment-v2/run_experiment.py), [CORRECTNESS_RESULTS.json](research/zeta100k-fresh-20261117-5bc2985bbe99/CORRECTNESS_RESULTS.json), and [monitoring-20261117.jsonl](research/zeta100k-fresh-20261117-5bc2985bbe99/qualification/monitoring-20261117.jsonl). |
| Checkpoint save/restore mechanism | **PASS** | The qualification smoke restored at position 4, resumed through position 8, and produced bit-identical logits. See [checkpoint-telemetry-isolation-results.json](research/zeta100k-fresh-20261117-5bc2985bbe99/qualification/post-resize-20261117/checkpoint-smoke-20261117-02/checkpoint-telemetry-isolation-results.json). |
| Restore the final 100K checkpoint into a new runner | **NOT YET TESTED** | The run completed normally and the final checkpoint was verified without restarting or resuming from it. |

## 2. Performance

| Requirement | Status | Evidence |
|---|---|---|
| Overall throughput and execution time documented | **PASS** | [resource-results.json](research/zeta100k-fresh-20261117-5bc2985bbe99/resource-results.json) reports 10.7199 tokens/s, 9,328.4 seconds of stream prefill, and 9,636.7 seconds total wall time. |
| Peak memory documented | **PASS** | Peak runner RSS was 3,400.3 MiB in [resource-results.json](research/zeta100k-fresh-20261117-5bc2985bbe99/resource-results.json). This is process RSS, not a claim about total host memory use. |
| CPU usage documented | **PASS** | The run recorded 18,748.1 process CPU-seconds over 9,636.7 wall-seconds, about 194.5% average process CPU (approximately 1.95 logical CPUs). This derived average is not whole-host utilization. See [resource-results.json](research/zeta100k-fresh-20261117-5bc2985bbe99/resource-results.json). |
| Performance repeatability established | **INCONCLUSIVE** | Recorded checkpoint intervals ranged from 10.356 to 10.806 tokens/s, but this is one full run, not repeated full-run trials. See [progress.jsonl](research/zeta100k-fresh-20261117-5bc2985bbe99/progress.jsonl) and [resource-results.json](research/zeta100k-fresh-20261117-5bc2985bbe99/resource-results.json). |
| Existing baseline comparison reviewed | **PASS** | The two-pair, 4,096-token attention microbenchmark reports 7.33× telemetry-off and 7.67× telemetry-on ratios for one configuration. The paired change is inconclusive; this is not full-model throughput. See [THERMAL_BENCHMARK_RESULTS.md](research/zeta100k-fresh-20261117-5bc2985bbe99/THERMAL_BENCHMARK_RESULTS.md). |
| Full-run speedup over a fair baseline established | **INCONCLUSIVE** | There is no matched 100,000-token native/full-history baseline. The native run is an 8,064-token prefix, with 3 of 8 queries completed; five facts were outside that prefix. See [native-baseline.jsonl](research/zeta100k-fresh-20261117-5bc2985bbe99/native-baseline.jsonl) and [FINAL_REPORT.md](research/zeta100k-fresh-20261117-5bc2985bbe99/FINAL_REPORT.md). The historical 8,192-token 9× result is not verified by this run. |

## 3. Integration

| Requirement | Status | Evidence |
|---|---|---|
| MathMesh telemetry delivered through completion | **PASS** | The asynchronous observer delivered 56 records through token 100,000, with zero failed requests, drops, or queued records at the final check. See [telemetry-status.json](research/zeta100k-fresh-20261117-5bc2985bbe99/telemetry-status.json) and [monitoring-20261117.jsonl](research/zeta100k-fresh-20261117-5bc2985bbe99/qualification/monitoring-20261117.jsonl). |
| Stability recommendations remained advisory | **PASS** | 56 records reached token 100,000; all have `applied_to_zeta: false`. See [stability-recommendations.jsonl](research/zeta100k-fresh-20261117-5bc2985bbe99/stability-recommendations.jsonl). |
| Deterministic RAG outside mathematical execution | **PASS** | Run-level and checkpoint audit calls recorded metadata only; the architecture report states RAG is not called from the token-processing path and receives no tensors, logits, or cache. See [MULTI_REPOSITORY_ARCHITECTURE.md](research/zeta100k-fresh-20261117-5bc2985bbe99/MULTI_REPOSITORY_ARCHITECTURE.md), [rag_audit_adapter.py](research/zeta100k-fresh-20261117-5bc2985bbe99/rag_audit_adapter.py), and [rag-checkpoint-6144.json](research/zeta100k-fresh-20261117-5bc2985bbe99/qualification/post-resize-20261117/rag-checkpoint-6144.json). |
| Observer failure isolation at a bounded smoke-test boundary | **PASS** | Injected telemetry failures during a short checkpoint smoke still produced finite logits and valid cache state. See [checkpoint-telemetry-isolation-results.json](research/zeta100k-fresh-20261117-5bc2985bbe99/qualification/post-resize-20261117/checkpoint-smoke-20261117-02/checkpoint-telemetry-isolation-results.json). |
| Physical support-service outage during the full run | **NOT YET TESTED** | Full-run observers remained available. The MathMesh/RAG/Stability outage records are admission-boundary failures, not outages during full inference. See [outage-mathmesh/integration-failure.json](research/zeta100k-fresh-20261117-5bc2985bbe99/qualification/outage-mathmesh/integration-failure.json), [outage-rag/integration-failure.json](research/zeta100k-fresh-20261117-5bc2985bbe99/qualification/outage-rag/integration-failure.json), and [outage-stability/integration-failure.json](research/zeta100k-fresh-20261117-5bc2985bbe99/qualification/outage-stability/integration-failure.json). |
| Integration overhead documented | **PASS** | One-time run-level service coordination took 0.3266 seconds. The 4,096-token telemetry off/on microbenchmark is variable and does not establish a stable throughput cost. See [integration-state.json](research/zeta100k-fresh-20261117-5bc2985bbe99/integration-state.json) and [THERMAL_BENCHMARK_RESULTS.md](research/zeta100k-fresh-20261117-5bc2985bbe99/THERMAL_BENCHMARK_RESULTS.md). |

## 4. Information retention

| Requirement | Status | Evidence |
|---|---|---|
| Existing retrieval evidence reviewed | **PASS** | This run's synthetic fact-retrieval records and the repository's [context retrieval protocol](benchmarks/CONTEXT_RETRIEVAL_PROTOCOL.md) and [benchmark results](benchmarks/RESULTS.md) were reviewed. |
| Long-distance retrieval on the run's synthetic facts | **FAIL** | The 100K evaluation returned 0 exact matches for 8 facts placed from token 128 through 90,000; all 8 short-prompt controls passed. This is a failure on this synthetic test, not a general language-quality judgment. See [retention-results.jsonl](research/zeta100k-fresh-20261117-5bc2985bbe99/retention-results.jsonl) and [FINAL_REPORT.md](research/zeta100k-fresh-20261117-5bc2985bbe99/FINAL_REPORT.md). |
| Information retention distinguished from stream processing | **PASS** | The 100K token pipeline passed while the separate retrieval test failed; the results do not equate processing with remembering. See [FINAL_REPORT.md](research/zeta100k-fresh-20261117-5bc2985bbe99/FINAL_REPORT.md) and [retention-results.jsonl](research/zeta100k-fresh-20261117-5bc2985bbe99/retention-results.jsonl). |
| Natural-language or general retrieval capability established | **NOT YET TESTED** | The completed retrieval evaluation used eight synthetic randomly assigned facts; no natural-language or broad retrieval evaluation was performed. |

## 5. Scaling

| Requirement | Status | Evidence |
|---|---|---|
| Larger-model compatibility | **NOT YET TESTED** | Only SmolLM2-360M-Instruct was run. |
| Different-architecture compatibility | **NOT YET TESTED** | The tested adapter requires Transformers 5.23.0, eager Llama attention, and native default RoPE; no other architecture was evaluated. See [compat_adapter.py](research/context-100k-experiment-v2/compat_adapter.py). |
| Memory scaling beyond this model/configuration | **NOT YET TESTED** | The run measured one 360M model at one CPU FP32 configuration; it is not a scaling curve. See [resource-results.json](research/zeta100k-fresh-20261117-5bc2985bbe99/resource-results.json). |
| Unverified scaling claims identified | **PASS** | This checklist separates the measured single-model result from larger-model, other-architecture, and memory-scaling claims. |

## 6. Research readiness

| Requirement | Status | Evidence |
|---|---|---|
| Existing research compared systematically with this exact architecture | **INCONCLUSIVE** | Internal baselines and benchmarks exist, but no systematic literature comparison specific to this Llama/default-RoPE/sink-window configuration was established. See [compat_adapter.py](research/context-100k-experiment-v2/compat_adapter.py), [benchmark results](benchmarks/RESULTS.md), and [THERMAL_BENCHMARK_RESULTS.md](research/zeta100k-fresh-20261117-5bc2985bbe99/THERMAL_BENCHMARK_RESULTS.md). |
| Novelty of the implementation established | **NOT YET TESTED** | The experiment validates an engineering implementation; it does not establish that its mechanisms are novel. |
| Implemented mechanisms distinguished from new claims | **PASS** | The implementation uses native unscaled RoPE with a bounded sink-plus-recent KV cache; the run demonstrates this configuration's operation, not a novel algorithm or broad model capability. See [compat_adapter.py](research/context-100k-experiment-v2/compat_adapter.py) and [MULTI_REPOSITORY_ARCHITECTURE.md](research/zeta100k-fresh-20261117-5bc2985bbe99/MULTI_REPOSITORY_ARCHITECTURE.md). |
| Reproducibility records reviewed | **PASS** | Run ID, starting commit, support commits, runtime versions, fixed seed/configuration, and model/tokenizer/corpus/stream hashes are recorded in [PROVENANCE.md](research/zeta100k-fresh-20261117-5bc2985bbe99/PROVENANCE.md) and [FRESH_RUN_PROTOCOL.md](research/zeta100k-fresh-20261117-5bc2985bbe99/FRESH_RUN_PROTOCOL.md). |
| Independent replication requirements identified | **PASS** | Replication should use the pinned commits/assets and identical seed, CPU FP32 settings, stream, chunk/cache sizes, and checkpoint checks; include a matched baseline and independently verify retrieval on preregistered data. Provenance inputs are in [PROVENANCE.md](research/zeta100k-fresh-20261117-5bc2985bbe99/PROVENANCE.md). |
| Limitations documented | **PASS** | The final report records retrieval failures, the unmatched native baseline, cache limits, and the distinction between processing and retention. See [FINAL_REPORT.md](research/zeta100k-fresh-20261117-5bc2985bbe99/FINAL_REPORT.md). |

## Experiment provenance

- **Run:** `zeta100k-fresh-20261117-5bc2985bbe99`; primary starting commit `cc2b8dfb45dd887f6f202ca69b8c874410fc4645`; runner source commit `ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed`.
- **Configuration:** `HuggingFaceTB/SmolLM2-360M-Instruct`, revision `a10cc1512eabd3dde888204e902eca88bddb4951`; seed `20261117`; 100,000 tokenizer-produced tokens; chunk size 64; 64 sink + 64 recent cache positions; CPU FP32.
- **Pinned assets:** weights SHA-256 `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`; tokenizer `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`; config `224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee`; corpus `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`; token stream `c9c43fb35050bc31f8d60e2a9bd600ae960f96d7a7ac1618b41505b2955da322`.
- **Support commits:** MathMesh `4ef77424b7ef4c5b131f558e85fa1eae24d890ab`; System-Stability-Toolkit `9e7a254ee9607b23ae35b756110d3713ad0dac45`; Deterministic RAG Governor `7eee42bc77757e8a23138b9b65c8a1072c88ca9d`.
- **Runtime:** Python 3.14.2, PyTorch 2.15.0+cpu, Transformers 5.23.0, Tokenizers 0.23.3, Safetensors 0.8.0, Hugging Face Hub 2.4.0.

Full repository, dependency, and support-service provenance is in
[PROVENANCE.md](research/zeta100k-fresh-20261117-5bc2985bbe99/PROVENANCE.md).

## Conclusions

1. **Demonstrated:** one CPU FP32 run processed 100,000 contiguous tokens with finite logits, a verified 128-position cache, 56 verified checkpoints, pinned provenance, and non-blocking telemetry/advisory observers.
2. **Unproven:** long-range factual retention (the tested synthetic task failed), full-run speed superiority over a fair baseline, repeatability across runs, other models or architectures, full-run physical outage isolation, final-checkpoint restoration, and research novelty.
3. **Assessment:** the evidence supports an engineering demonstration for this exact configuration. It does not establish a general research contribution or superior end-to-end performance.

## Direct answers

1. **Does the pipeline hold together?** Yes, for the measured 100K processing run: it completed with code 0 and passed the documented continuity, checkpoint, numerical, cache, and observer checks. This does not imply successful information retrieval.
2. **Can it beat a fair baseline without losing numerical integrity or continuity?** Not established. Integrity and continuity passed, but there is no matched 100K baseline. The short attention-only comparison is not an end-to-end speedup result, and its telemetry difference is inconclusive.
