# Research overview

## Purpose and scope

Zeta Scaling Toolkit is an experimental Python/PyTorch codebase for studying
implementation choices around attention temperature, positional information,
streaming attention, learning-rate schedules, coordinate systems, and weight
initialization. It is intended for inspection and controlled evaluation by
researchers and research engineers.

The repository contains two distinct kinds of work:

- **Reusable software components:** modules with input validation, numerical
  behavior, property tests, regression tests, and compatibility coverage.
- **Exploratory research experiments:** small controlled comparisons intended
  to test specific hypotheses. These experiments are not equivalent to
  validation on large models or production workloads.

## Current research finding

**Stage 1 long-context decision: NO-GO.** The documented held-out successor
evaluation did not meet its acceptance gate for the tested cosine-plus-log
variants. The planted-needle sanity gate also failed (6 of 10 seeds reached
the required 0.9 threshold; the criterion was at least 8 of 10). Under the
predefined gate, the retrieval comparison is not considered validated.

The measured result is useful as a negative finding: under the specified
small-model, CPU-only setup, the tested configurations did not demonstrate
the intended held-out long-context capability. It is not evidence that the
approach is universally ineffective, nor evidence of an alternative method
being superior outside this setup.

See [benchmarks/RESULTS.md](../benchmarks/RESULTS.md) for the numerical
comparisons, confidence intervals, seed definitions, commands, hardware
notes, and raw-record references. See [REPORT.md](REPORT.md) for the
experimental design and interpretation.

## Evidence standards

When interpreting results in this repository:

1. Distinguish unit/property correctness from task-level model performance.
2. Treat a benchmark gate as passed only when every predefined criterion is
   satisfied on the specified held-out data and seeds.
3. Keep tuning and reporting seeds separate; disclose when data ranges are
   shared between tuning and reporting.
4. Report incomplete sweeps as incomplete. Do not infer outcomes for missing
   records or treat interrupted runs as failures of the method.
5. Keep CPU and GPU results separate, and record software versions and
   hardware when comparing performance.
6. Treat negative and null results as first-class results.

## Reproducibility

- Start with the instructions in [TESTING.md](../TESTING.md).
- Review the exact commands and environment notes in
  [benchmarks/RESULTS.md](../benchmarks/RESULTS.md).
- Preserve seed lists, output JSONL files, checkpoint provenance, and
  configuration metadata when extending an experiment.
- Do not combine local and hosted-runner measurements unless the protocol
  explicitly justifies doing so.
- For a new claim, define the acceptance gate before examining the reporting
  set and document deviations from the protocol.

## Software quality and limitations

The test suite covers deterministic properties, edge cases, numerical
regressions, and compatibility behavior. The README describes component-level
contracts and known limitations. A passing software test suite does not
establish that a component improves model quality, scales efficiently, or
generalizes to a new dataset.

CUDA-specific behavior and performance should be treated as unverified unless
the relevant tests and benchmarks are run on a CUDA-enabled system. Small
synthetic and character-level tasks are useful diagnostics, not substitutes
for evaluation on the intended scientific workload.

## Release and development status

- Published GitHub release: [v0.1.0](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/releases/tag/v0.1.0).
- The current default branch reports `0.5.1` in package metadata. That
  development state is not represented by the published `v0.1.0` tag.
- The default branch CI has a recent successful scheduled run for mutation
  checking and nightly Hypothesis, but the latest push-triggered run skipped
  the standard test and quality jobs. An earlier matrix run failed on a
  checkpoint-loading test because PyTorch emitted a `TypedStorage`
  deprecation warning that the test configuration treats as an error.
- A separate context-extrapolation sweep was reported as interrupted before
  completion. Its partial records are not a completed benchmark result.

Resolve version consistency and obtain a complete green CI matrix before
publishing a new release or describing the development branch as release-ready.

## Research conduct

This project aims to make implementation details, test coverage, benchmark
protocols, and negative findings inspectable. Claims should remain scoped to
the exact code revision, data, hardware, model, seeds, and evaluation budget
that produced the evidence.