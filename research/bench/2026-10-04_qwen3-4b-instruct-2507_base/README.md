# Qwen3-4B-Instruct-2507 (base, not distilled) on the demo laptop, 2026-10-04

Machine: [Lenovo 81SX, i7-9750H 6c/12t, 15.5 GB DDR4](../../laptop/lenovo-81SX_i7-9750H.md) ·
llama.cpp b11392 · CPU only, 6 threads · context 8,192 · 3 interleaved passes with the laptop in normal
use (Zoom, Cursor; 1-min load average 12.6 / 6.9 / 6.2). Raw data: `bench.json`, table: `bench.md`.

Reproduce: `distillkit export` + `distillkit bench -c configs/tools_pilot.yaml --set run_dir=runs/bench_4b_base
--set student.model=Qwen/Qwen3-4B-Instruct-2507 --set export.merge=false
--set export.quants=[Q4_K_M,Q8_0,Q6_K,Q5_K_M,Q3_K_M] --set export.ollama_name=null`

## Results (mean ± std over 3 passes; RAM in GiB)

| Quant | File GB | Peak RAM GiB, default | Peak RAM GiB, `-lm none` | pp512 tok/s | Gen tok/s, empty ctx | Gen tok/s, 2,048 in ctx |
|---|---|---|---|---|---|---|
| Q3_K_M | 2.08 | 3.81 | **3.13** | 49.4 ± 4.1 | 12.6 ± 0.9 | **7.6 ± 0.4** |
| Q4_K_M | 2.50 | 5.15 | **3.52** | 50.9 ± 2.5 | 11.3 ± 0.7 | **7.2 ± 0.4** |
| Q5_K_M | 2.89 | 3.88 | — | 47.6 ± 2.6 | 9.6 ± 0.6 | 6.5 ± 0.5 |
| Q6_K | 3.31 | 4.27 | — | 48.9 ± 3.7 | 8.8 ± 0.7 | 6.1 ± 0.5 |
| Q8_0 | 4.28 | 5.18 | 5.17 | 38.4 ± 1.9 | 7.0 ± 0.5 | 5.2 ± 0.3 |

Peak RAM is the max over passes of a llama-completion run at context 8,192 (KV cache ~1.1 GiB included);
it was identical in every pass. `-lm none` column: one run each, after the bench.

## Findings

1. **Q4_K_M's 5.15 GiB is double-counted weights, not a real need.** On AVX2 CPUs llama.cpp repacks
   Q4_K/Q3_K weights into an interleaved layout; with the default memory-mapped load the file pages
   stay resident too. Loading without mmap (`--load-mode none`) frees them: **Q4_K_M 3.52 GiB**, under
   the < 4 GB target. `--no-repack` gives the same RAM; its speed cost is not measured yet.
2. **Speed: no quant meets "≥ 8 tok/s at 2,048 tokens of context" on this laptop** (Q3_K_M 7.6,
   Q4_K_M 7.2), all are above 8 with an empty context. Generation speed follows file size (memory
   bandwidth); prompt processing is flat ~50 tok/s, except Q8_0.
3. **Time to first token is the bigger UX issue:** system prompt + 8 tool schemas is ~1,500 tokens,
   ~30 s at ~50 tok/s before the first token. The `bench` TTFT check (cold first turn vs. follow-up
   with the prompt reused) is implemented but not yet run on these files.
4. **Q4_K_M is the provisional pick:** fits with `-lm none`, within ~6% of Q3_K_M's speed, lower
   quality risk. Final choice waits for the GGUF eval (accuracy per quant) on the trained student.

## Open

- Speed of `--no-repack` and `--load-mode none` (llama-bench `--repack 0,1 -lm mmap,none`).
- Whether Ollama's runner shows the same double counting, and how to set its load mode.
- TTFT on these files; the 0.6B pilot bench with 6 threads; GGUF eval live run.
