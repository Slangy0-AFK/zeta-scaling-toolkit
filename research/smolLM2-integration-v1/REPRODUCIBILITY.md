# Reproducibility

The recorded run used two CPU threads on a two-core AMD EPYC 7763 allocation,
Python 3.14.2, PyTorch 2.15.0+cu132, Transformers 4.52.2 and FP32 native Llama
eager attention. CUDA was unavailable. Before downloading, this host had
approximately 9.6 GiB free disk and 4.2 GiB available RAM; the public
safetensors file is 723,674,912 bytes. Resource availability can change.
The repository revision was `64f9afe03c9fe39d81e819c58bb844fe04603548`;
the used `zeta_scaling_toolkit/positions.py` SHA-256 was
`288ad1c6777d46b5e6dc08f34cc9ad13e968545441331bbfb71b2b69ed99f096`.

The downloaded snapshot and isolated virtual environment used for these tests
are under the session workspace, not in this repository. For a fresh run,
first check disk and RAM, ensure an existing compatible PyTorch installation,
and install the explicitly required runtime in a separate environment:

```bash
python -m venv --system-site-packages "$HOME/.cache/zeta-smollm-venv"
"$HOME/.cache/zeta-smollm-venv/bin/python" -m pip install --only-binary=:all: 'transformers==4.52.2'
```

The following downloads only the pinned model and tokenizer files. Obtain
permission and recheck resources before running it:

```bash
export HF_HOME="$HOME/.cache/zeta-smollm-hf"
export HF_HUB_DISABLE_XET=1
PY="$HOME/.cache/zeta-smollm-venv/bin/python"
SNAP="$("$PY" -c 'from huggingface_hub import snapshot_download; print(snapshot_download(repo_id="HuggingFaceTB/SmolLM2-360M-Instruct", revision="a10cc1512eabd3dde888204e902eca88bddb4951", allow_patterns=["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]))')"
```

From the repository root, use a **new empty output directory**. The second
phase refuses to run unless the first passed with the same weights and adapter
and validation source hashes. Existing result files are never overwritten.

```bash
OUT="$(mktemp -d)"
PYTHONPATH="$PWD" "$PY" research/smolLM2-integration-v1/validate.py --snapshot "$SNAP" --output-dir "$OUT" --phase parity
PYTHONPATH="$PWD" "$PY" research/smolLM2-integration-v1/validate.py --snapshot "$SNAP" --output-dir "$OUT" --phase bounded
PYTHONPATH="$PWD" "$PY" -m pytest -q tests/test_smollm2_adapter.py tests/test_properties.py -k 'smollm2_adapter or streaming_kv_cache'
ruff check research/smolLM2-integration-v1/adapter.py research/smolLM2-integration-v1/validate.py tests/test_smollm2_adapter.py
ruff format --check research/smolLM2-integration-v1/adapter.py research/smolLM2-integration-v1/validate.py tests/test_smollm2_adapter.py
```

The JSON records include weight, tokenizer, config, adapter and validation
hashes, tolerances, device, timing, memory and CPU time. The runner checks the
weight file's known SHA-256 before model loading and uses `local_files_only`;
it never downloads during validation. Its result files report only completed
successful phases: an error terminates the phase rather than producing a
success-shaped fallback. Precise results and timings may differ with hardware,
runtime, contention or math kernels. Do not equate this short engineering
validation with the proposed context-boundary experiment.
