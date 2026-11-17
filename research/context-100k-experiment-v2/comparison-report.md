# Native versus Zeta comparison

**Status: NOT RUN — no valid matched comparison is available.**

Neither native nor Zeta inference ran in this attempt. There is no 100K token
stream, no verified tokenizer count, no planted-fact answer set, no accuracy
score, and no matched baseline. Therefore this report makes no statement
about native model failure, Zeta context extension, useful information
retention, or compute savings.

The previously recorded 8,193-token Zeta engineering run belongs to source
commit `ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed` and Transformers 4.52.2. It
used a repeated sentence and a 64-token recent window plus four sinks. It was
not a retention experiment and is not comparable to the requested patched
runtime.

The immediate blocker is unavailable model provisioning: the pinned model
revision is not cached, and both Hugging Face hostnames failed DNS resolution.
In addition, the CPU PyTorch wheel endpoint could not be resolved, and the
restored adapter has not been ported to Transformers 5.23.0's Cache API.
