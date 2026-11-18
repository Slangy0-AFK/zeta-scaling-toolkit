# Thermal Benchmark Results

## Status

Short Zeta attention microbenchmarks ran successfully with telemetry off/on.
Thermal condition was **not measurable**: this environment exposed no
CPU-temperature or hardware thermal-throttling sensor. The full model and
100K stream were not loaded or run.

Raw data: `qualification/thermal/matched-attention-4096-pairs2-v2.json`.
Harness: `thermal_benchmark.py`. The benchmark reads progress/system counters
outside model execution and does not change Zeta parameters, cache, precision,
clock, fan, voltage, or power limits.

## Environment and Telemetry

- Date: 2026-11-17; Linux 6.8.0-1070-azure, Python 3.14.2, CPU-only.
- Processor string: AMD EPYC 7763, hypervisor vendor Microsoft; two visible
  logical CPUs.
- Input: deterministic synthetic random float32 Q/K/V; no model, tokenizer,
  corpus, or checkpoints loaded.
- Benchmark: existing `window_speed.py` / `NonCollapsingAttention`, length
  4,096, window 64 or 256, four heads, head dimension 8, four sinks, one CPU
  thread, three measured repetitions. Each pair used the same deterministic
  seed/configuration; telemetry order was alternated.
- Temperature: unavailable (zero thermal zones and hwmon temperature readings).
- CPU frequency: sampled `/proc/cpuinfo` `cpu MHz`, about 3.20–3.29 GHz in
  telemetry-on trials; virtualized and not a hardware-verified clock.
- CPU utilization: about 69–72% host/VM-wide; benchmark process about 94–97%
  of one CPU. These short measurements cannot attribute other host/tenant work.
- Thermal throttling: hardware indicator unavailable; cgroup
  `nr_throttled`/`throttled_usec` deltas were zero.
- Memory: benchmark child sampled peak RSS about 301.5–301.9 MiB. Available
  memory before the four trials ranged from 2,492 to 2,536 MiB; after trials
  it ranged from 2,492 to 2,684 MiB. These values exclude a loaded 360M model.

## Paired Attention Ratios

Ratio is full-causal log-attention median time divided by windowed-attention
median time. Values below are medians across the two off/on pairs. Attention
tokens/s is query length divided by one attention-operation time; it is **not**
full-model token throughput.

| Window | Scaling | Measurement | Off ratio | On ratio | Off window attention tokens/s | On window attention tokens/s |
|---:|---|---|---:|---:|---:|---:|
| 64 | none | Forward | 6.05× | 7.85× | 250,397 | 295,820 |
| 64 | log | Forward | 7.33× | 7.67× | 309,002 | 287,262 |
| 64 | none | Forward+backward | 6.82× | 8.01× | 75,372 | 80,800 |
| 64 | log | Forward+backward | 5.83× | 7.51× | 64,212 | 75,611 |
| 256 | none | Forward | 3.72× | 4.00× | 159,996 | 151,592 |
| 256 | log | Forward | 4.58× | 4.88× | 193,476 | 180,735 |
| 256 | none | Forward+backward | 4.41× | 4.89× | 48,776 | 49,004 |
| 256 | log | Forward+backward | 4.08× | 5.01× | 44,979 | 50,273 |

No measured 4,096 configuration reached the 9× ratio. Ratios differed across
pairs and should be treated as exploratory CPU microbenchmark results, not
stable speedups.

## Telemetry Overhead

| Pair | Telemetry off wall time | Telemetry on wall time | On minus off | Parent CPU off / on |
|---:|---:|---:|---:|---:|
| 1 | 5.718 s | 5.920 s | +0.202 s | 0.007 / 0.111 s |
| 2 | 5.613 s | 5.934 s | +0.321 s* | 0.005 / 0.106 s |

*The second pair ran telemetry-on first, then telemetry-off; the raw mode
overhead estimate. The observer parent consumed about 0.10 additional CPU
sequence is retained in the JSON file. Both v2 pairs were slower with telemetry
on, by 0.202 and 0.321 seconds. An earlier one-pair pilot had telemetry-on
faster, so this small, variable sample is not a stable overhead estimate. The
observer parent consumed about 0.10 additional CPU-seconds per run, while
sampled child RSS remained about 302 MiB. Telemetry-on used 56–59 samples at
100 ms intervals. No temperature sensor was available to correlate with the
throughput variation.

## 9× Target and Thermal Finding

The historical 9× attention-only comparison first exceeded 9× at length 8,192;
its 4,096 ratios were 7.90× forward and 6.93× forward+backward. This current
short benchmark ran only 4,096 and also stayed below 9×. It does not reproduce
the historical 8,192 point and does not compare the complete Zeta model.

No stable-versus-constrained thermal comparison was possible. Temperature and
hardware throttle signals were not exposed, and no power or cooling control was
changed. The available evidence cannot determine whether thermal constraints
limit Zeta throughput or whether a cooling intervention would improve it.
