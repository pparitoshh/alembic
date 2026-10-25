# Completed 12,092-example student

Qwen3-4B-Instruct-2507 completed full-snapshot 4-bit QDoRA training on 12,092 certified Q&A, including the 925 examples added since the previous trained snapshot. Job `59940941` completed 2,268 optimizer steps. The earlier adapter remains available for comparison.

[Download the adapter and tokenizer](https://github.com/pparitoshh/alembic/releases/tag/student-12092-20261011). Verify the archive checksum in `artifact_manifest.json`. Extract the archive and load `adapter/` with PEFT and the Qwen3-4B-Instruct-2507 base model. Exact configuration and training logs accompany the artifact.

Training execution and adapter hashes passed verification. Held-out evaluation remains pending; no improvement over the base model is claimed. Dataset production continues toward the published 50,000-example milestone.
