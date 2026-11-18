# Provisioning Approval Request

## Recommendation

Approve resizing the existing Codespace `fictional-spoon-p7r5xq4xq5wgf45w` from
`basicLinux32gb` (2 CPUs, 8 GiB RAM, 32 GiB storage) to
`standardLinux32gb` (4 CPUs, 16 GiB RAM, 32 GiB storage). The GitHub Codespaces
machine API returned these exact CPU/RAM/storage values. This is the quickest
reproducible option because the current Codespace already contains the fresh
run package, pinned model/tokenizer/corpus, locked Python environment, and
uncommitted repository state. The original experiment stays untouched.

**No resize or resource purchase has been made.** Changing the current machine
may disconnect/restart the active Codespace and may incur billing after any
included quota. Please approve this specific resource change before it is run.

## Capacity and Cost

| Option | CPUs | RAM | Storage | Persistence | Cost / suitability |
|---|---:|---:|---:|---|---|
| Current `basicLinux32gb` | 2 | 8 GiB | 32 GiB | Existing Codespace storage | Actual `MemAvailable` 2,428.6 MiB; blocked. |
| Existing `standardLinux32gb` tier | 4 | 16 GiB | 32 GiB | Resize existing Codespace; workspace remains in the same Codespace volume, but snapshot the uncommitted state first | Published price $0.36 per active hour, plus $0.07/GB-month storage. 12-hour runner budget costs at most $4.32 gross compute; 32 GB-month storage is $2.24 if billed, but this is the existing storage size. Included personal quota may offset charges; account balance was not accessible. |
| Standard hosted GitHub Actions runner (`ubuntu-22.04`) | 4 | 16 GB | 14 GB SSD | Temporary VM; artifacts must be uploaded for persistence | This repository is public; standard hosted runners are free for public repositories. A workflow job defaults to 360 minutes, and the 100K protocol allows 12 hours. The existing interim throughput estimate implies about 8.2 hours, so a single hosted job is not a suitable completion path; checkpointed multi-job recovery has not been qualified. |
| Separate Codespace or cloud VM | Not currently provisioned | Must be verified as >=16 GiB (32 GiB preferred) | Depends | New host; explicit transfer and durable storage required | No separate host is configured. Provisioning it may create billable use and requires separate approval. |

A rough startup-availability projection for the 16 GiB Codespace is about
10.6 GiB if the current approximately 5.4 GiB non-available memory footprint
remains unchanged. This is only an arithmetic projection, **not a measurement or
guarantee**. The first action after approval must be a live `MemAvailable`
measurement on the resized Codespace; launch remains blocked unless it is at
least 3,584 MiB with the support/test processes in the intended configuration.

The repository's GitHub Actions runner-inventory API returned HTTP 403, so no
self-hosted runner capacity could be verified. No Actions job was dispatched.

## Existing Reproducible State

Read-only SSH inspection confirmed that the existing Codespace has:

- Fresh run directory `research/zeta100k-fresh-20261117-5bc2985bbe99/`.
- Zeta source commit `cc2b8dfb45dd887f6f202ca69b8c874410fc4645`.
- Model `HuggingFaceTB/SmolLM2-360M-Instruct`, revision
  `a10cc1512eabd3dde888204e902eca88bddb4951`.
- Model, tokenizer, config, corpus hashes matching `PROVENANCE.md`.
- Dependency lock SHA-256
  `2fb39bb9cfde274d733425f50002b69d21b37693a6302976d510fd6e9d28095c`.
- The original experiment's checkpoint files still hash-match their recorded
  values.

No artifact transfer is needed for an in-place resize. Preserve the current
uncommitted workspace and user benchmark edits; do not reset or clean the
worktree. Re-create the `/tmp` venv from the lock if the resize restarts the
container or clears `/tmp`.

## Exact Action After Approval

The following command is prepared, not executed:

```bash
gh codespace edit \
  --codespace fictional-spoon-p7r5xq4xq5wgf45w \
  --machine standardLinux32gb
```

After GitHub applies the machine change and the Codespace reconnects:

```bash
gh codespace ssh --codespace fictional-spoon-p7r5xq4xq5wgf45w -- \
  'free -m; grep -E "^(MemAvailable|MemTotal|SwapTotal):" /proc/meminfo; getconf _NPROCESSORS_ONLN; cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current'
```

If `/tmp/zeta100k-fresh-5bc2985bbe99` is absent, recreate it and install the
exact lock without altering the project configuration:

```bash
python -m venv /tmp/zeta100k-fresh-5bc2985bbe99
/tmp/zeta100k-fresh-5bc2985bbe99/bin/python -m pip install --require-hashes \
  -r /workspaces/zeta-scaling-toolkit/research/context-100k-experiment-v2/requirements-lock.txt
/tmp/zeta100k-fresh-5bc2985bbe99/bin/python -m pip check
```

Then rerun standalone, each single-service, all-service numerical
qualification, active-run outage/recovery tests, and final telemetry concurrency
checks. Do not start the 100K runner unless every mandatory gate passes and the
live memory guard is satisfied. Keep the existing run ID and output directory;
never add `--resume` to initialize this fresh run.

## Approval Needed

Please approve resizing the existing Codespace to `standardLinux32gb` with the
listed gross rates, or decline and choose another funded/persistent Linux host.
Until approval, no resource settings are changed and the 100K run remains
unstarted.
