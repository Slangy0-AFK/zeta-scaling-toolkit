# Provenance

## Source and model

- Source prototype commit: `ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed`.
- Model: `HuggingFaceTB/SmolLM2-360M-Instruct`.
- Immutable model revision: `a10cc1512eabd3dde888204e902eca88bddb4951`.
- Model weights SHA-256:
  `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`.
- Tokenizer SHA-256:
  `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.
- Model config SHA-256:
  `224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee`.

The snapshot was downloaded by exact revision and then loaded with offline mode,
`local_files_only=True`, `trust_remote_code=False`, and `use_safetensors=True`.
The runner checks the model, tokenizer, and config hashes before executing.

## Input stream

- Corpus SHA-256:
  `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
- Token count: `100000`.
- Token-stream SHA-256:
  `c9c43fb35050bc31f8d60e2a9bd600ae960f96d7a7ac1618b41505b2955da322`.
- Fact generation seed: `20261117`.
- Actual fact starts: `128`, `2047`, `8002`, `15998`, `24999`, `49999`,
  `75001`, and `90000`.

`CORRECTNESS_RESULTS.json`, the stream safetensors metadata, checkpoints, and
`resource-results.json` retain additional machine-readable provenance.
