# Host Capacity Recheck

## Outcome: current host blocked; larger host requires approval

For verified machine-tier specifications, pricing, exact resize command, and
the approval boundary, see
[PROVISIONING_APPROVAL_REQUEST.md](PROVISIONING_APPROVAL_REQUEST.md).

Run ID: `zeta100k-fresh-20261117-5bc2985bbe99`. No model loading or 100K
processing occurred in this recheck. The original experiment and checkpoints
were not modified.

## Current execution environment

- `systemd-detect-virt --vm --container` identifies Docker; PID 1 is
  `docker-init`. The physical host is not visible from this dev container.
- GitHub Codespaces metadata identifies the active repository environment as
  `basicLinux32gb`, displayed as **2 cores, 8 GB RAM, 32 GB storage**. The
  `32gb` suffix refers to the displayed machine/storage class name, not 32 GiB
  RAM.
- Guest `/proc/meminfo`: 7,942.3 MiB total. At 2026-11-17 21:27:50 UTC,
  2,506.7 MiB was available; the later 21:33:58 UTC SSH snapshot showed
  2,428.6 MiB available, 1,155.4 MiB below the launch gate. Swap was zero.
  This is host-visible `MemAvailable`, the exact field the runner checks, not
  RSS or peak memory.
- Container cgroup: `memory.max=max`; the container itself has no lower memory
  cap. The latest SSH snapshot had a cgroup current charge of 3,897,901,056
  bytes and lifetime peak 5,931,794,432 bytes. Neither value is available
  memory.
- CPU: two logical CPUs. No experiment runner or support service was active.
- The full-run guard requires `MemAvailable >= 1,024 + 2,560 = 3,584 MiB`.
  A fresh isolated preflight was refused before model loading with
  `RuntimeError: insufficient available memory to load the model safely`.

## Process ownership and reclaim

The model, tokenizer, MathMesh API/database, RAG API, Stability Toolkit, and
telemetry process were all absent at the resource check. The largest resident
processes were VS Code extension hosts (approximately 1.17 GiB and 0.72 GiB
RSS), followed by TypeScript language servers (approximately 0.33 GiB and
0.14 GiB RSS). RSS includes shared mappings and must not be summed as unique
physical memory. These belong to active editor sessions, not this experiment;
none were terminated. Docker/containerd are system services and were left
running. Stopped experiment containers are not consuming active process RSS.
No owned experiment process or cache could be safely stopped to recover the
required 1,077 MiB headroom.

The pinned model weights occupy 723,674,912 bytes on disk and tokenizer JSON
2,104,556 bytes. The fresh standalone preflight did not preserve a process peak
RSS measurement. The 3,584 MiB pre-load guard is a configured reserve plus
headroom, not a measured model peak; it remains unchanged because the project
has not established a lower safe threshold.

## Larger-host options

A read-only Codespaces inventory found the current 2-core/8-GiB/32-GiB-storage
machine and a `standardLinux32gb` option with 4 CPUs, 16 GiB RAM, and 32 GiB
storage. Codespaces billing lists $0.36 per active hour for 4 cores and
$0.07/GB-month for storage; whether included quota covers this account's use is
unknown. A gross 12-hour compute estimate is $4.32, before any included quota
or billed storage. Resizing the existing Codespace or creating another host
may incur charges and changes account/resource configuration. The existing
Codespace also has uncommitted/unpushed repository changes. No resize or new
host was performed without approval.

The repository's workflows use standard `ubuntu-22.04` runners; they do not
provide a dedicated long-context benchmark host. The GitHub self-hosted-runner
inventory request returned HTTP 403, so no self-hosted alternative could be
verified. No Actions job was dispatched.

## Required next step

To proceed, approve either (a) resizing the existing Codespace to a tier whose
RAM is confirmed to exceed the 3,584 MiB available-memory gate with safe
headroom, after preserving its uncommitted work, or (b) creating a separate
CPU Linux host with at least 16 GiB RAM (32 GiB preferred) and permission to
run the experiment. After provisioning, verify live `MemAvailable`, CPU, and
cgroup limits, reproduce the locked environment and hashes, complete all
integrated numerical/outage/telemetry gates, then launch the fresh run only if
every gate passes. No resource purchase, resize, or account change was made.
