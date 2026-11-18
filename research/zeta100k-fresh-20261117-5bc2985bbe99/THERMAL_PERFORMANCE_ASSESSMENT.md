# Thermal Performance Assessment

## Conclusion

**Thermal limiting is not established.** This Linux Codespace does not expose a
CPU temperature sensor or hardware thermal-throttle counter. The available CPU
frequency field is virtualized and cannot confirm the host's physical clock or
thermal behavior. No thermally stable versus constrained comparison was
possible, and no cooling improvement is claimed.

The full SmolLM2 Zeta runner was not loaded. At the latest recorded launch
check it had about 2,469 MiB available against its unchanged 3,584 MiB minimum.
The short benchmark below uses only Zeta's `NonCollapsingAttention` with
synthetic Q/K/V at length 4,096; it does not measure full-model throughput.

## Existing Toolkit Capabilities

The pinned System-Stability-Toolkit revision is
`9e7a254ee9607b23ae35b756110d3713ad0dac45`. Its `crash_fix.py` detects usable
CPU-core count and `/proc/meminfo` `MemAvailable`, then calculates advisory
buffer, concurrency, and task-granularity values. It does **not** read
hardware temperature, frequency, CPU utilization, thermal-throttling counters,
or fan state. It does not control fans, clocks, voltage, or power limits.

The recorded recommendation for workload size 100,000 was buffer 2,823,
parallel tasks 2, task size 186. Those values were not applied to Zeta. The
recommendation's concurrency is not a measurement of useful parallelism for
Zeta's one-thread attention benchmark.

## Available Linux Telemetry

- Hardware/VM: two visible CPUs; `lscpu` identifies an Azure-virtualized AMD
  EPYC 7763. The host CPU model and thermal policy are outside this container.
- Temperature: no `/sys/class/thermal/thermal_zone*` entries and no
  `/sys/class/hwmon/hwmon*/temp*_input` readings.
- Cooling devices: two entries named `Processor`, both `cur_state=0` and
  `max_state=0`; these expose no actionable cooling range or temperature.
- Frequency: `/proc/cpuinfo` reports `cpu MHz`; during the monitored sample it
  ranged roughly 3.20–3.29 GHz. This is a hypervisor-provided field, not a
  validated hardware frequency sensor or policy limit.
- Utilization: `/proc/stat` provides host/VM-wide utilization. The benchmark
  child's `/proc/PID/stat` CPU time was sampled separately. This does not
  attribute other VM tenants' activity.
- Throttling: `/sys/fs/cgroup/cpu.stat` reported zero `nr_throttled` and zero
  `throttled_usec` change during these short trials. The AMD
  `thermal_throttle` sysfs directory is absent. Zero cgroup throttling does not
  rule out host-level CPU throttling.
- Memory: the attention child sampled peak RSS was about 302 MiB. Available
  memory around these trials was roughly 2.49–2.68 GiB. This is not the memory
  footprint of the 360M model; model weights and tokenizer were not loaded.

## Benchmark Scope and Interpretation

`thermal_benchmark.py` runs the existing `benchmarks/window_speed.py` with
batch 1, four heads, head dimension 8, float32, one PyTorch thread, length
4,096, windows 64 and 256, and three measured repetitions. It alternates
telemetry-off and telemetry-on runs. Sampling is read-only and outside the
Zeta model execution path. Raw samples are in
`qualification/thermal/matched-attention-4096-pairs2-v2.json`.

In the v2 monitored runs, whole-host sampled CPU utilization was 70.3–72.1%;
The benchmark process used about 94–95% of one CPU. The telemetry parent used
about 0.106–0.111 CPU-seconds per 5.9-second benchmark, versus 0.005–0.007
CPU-seconds in the unmonitored control. In the two v2 pairs, telemetry-on wall
time was 0.202 s and 0.321 s longer. An earlier one-pair pilot showed the
opposite direction, so the small, variable sample does not establish a stable
wall-throughput penalty or that telemetry is overhead-free.

The historical 9× result is an attention-operation ratio, not a 9× end-to-end
Zeta or thermal target. The original measurements first exceeded 9× at length
8,192 for both forward and forward+backward, with window 64. This investigation
deliberately capped the new comparison at length 4,096; an 8,192 microbenchmark
was not attempted. The length-4,096 ratios do not meet 9×. See
[THERMAL_BENCHMARK_RESULTS.md](THERMAL_BENCHMARK_RESULTS.md).

## Recommendation

Do not infer thermal throttling from the virtual MHz variation or current
throughput. To assess it, repeat on a host exposing validated temperature,
frequency, and hardware throttle sensors, with sufficient model-load headroom.
Use matched randomized trials under normal operation and an independently
observed constrained condition. Do not induce a constrained condition by
changing fan, voltage, clock, or power controls automatically. Keep thermal
monitoring external to Zeta and do not alter its mathematical configuration.
