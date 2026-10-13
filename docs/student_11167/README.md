# Completed 11,167-example student

Qwen3-4B-Instruct-2507 completed three epochs of 4-bit QDoRA training on the certified 11,167-row release. Leonardo job `59924945` completed successfully after 2,094 optimizer steps. The saved adapter hash and source dataset hash were independently checked before delivery.

[Download the adapter and tokenizer](https://github.com/pparitoshh/alembic/releases/tag/student-11167-20261011). The archive also includes the exact run config and training evidence. Verify its SHA-256 against `artifact_manifest.json`, then extract it; `adapter/` is a standard PEFT adapter directory for `Qwen/Qwen3-4B-Instruct-2507`. It requires the base model and is not a standalone GGUF model.

Training used rank 16, alpha 32, learning rate 0.0001, effective batch size 16, BF16 compute, and maximum length 2,048. All 11,167 records passed the real training loader; assistant labels, supervised EOS and masked padding were checked, and no record reached the truncation boundary. Full loss logs are in `training_evidence.json`.

This proves training execution and artifact creation. Held-out evaluation and comparison with the base model remain pending; training loss does not establish generalization or improved tool calling. The separate 12,092-row training run is job `59940941`; its artifact will be delivered only after successful completion.
