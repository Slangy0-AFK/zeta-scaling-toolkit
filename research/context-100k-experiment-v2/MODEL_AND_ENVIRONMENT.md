# Model and environment

## Required model identity

- Repository: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Required immutable revision:
  `a10cc1512eabd3dde888204e902eca88bddb4951`
- Previously recorded weights SHA-256:
  `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`
- Previously recorded tokenizer SHA-256:
  `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`

The checkpoint and tokenizer were not present in the accessible Hugging Face
cache paths or `/models`. The recorded hashes originate in the prior
prototype's results and were not recomputed. Attempts to resolve the pinned
tokenizer URL failed for both `huggingface.co` and `hf.co`; no model files were
downloaded or substituted.

## Host and Python

- Checkout: `portfolio/research-lab-presentation`,
  `b2d441bdbf496b01a3a3538816925a006f357fd4`
- Python: 3.12.3 (system interpreter; not modified)
- Isolated venv: `/home/runner/.cache/zeta-100k-venv`
- PyTorch / Transformers / Tokenizers / Safetensors: not installed in the
  isolated venv
- CUDA: no usable `nvidia-smi` command/device detected; CPU execution was
  required if the experiment could run
- Exact validated prototype source:
  `ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed`
- Compatibility-port source commit: none; port not implemented

The uninstalled Transformers 5.23.0 candidate supports Python 3.12 per
package metadata and declares PyTorch `>=2.6`. Source inspection showed a
Cache API incompatible with the restored 4.52.2 adapter contract; therefore
metadata compatibility does not establish a working model runtime.
