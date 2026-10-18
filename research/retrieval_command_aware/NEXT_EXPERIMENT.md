# Prepared retrieval variant — not submitted

The completed 2048-token experiment scored 0.67125 on v3 and 0.59848 on v2 under GPT-OSS paired judging. These are development-set preference scores, not factual accuracy. The original outputs and scores are frozen.

Two inspected failures asked for sbatch resource directives, but the three retrieved chunks came from general GPU, srun, salloc or heterogeneous-job documentation. Answers then invented options such as --mem-per-node. On Leonardo, non-submitting option-parser probes rejected --mem-per-node, --min-nodes, --start-time, --gres-per-task and --cpus-per-node. Other options rejected by the old checker are accepted by the local parser. Do not turn this into a universal command-independent allowlist.

## Prepared change

`retrieve.py` reserves up to two of the existing three context slots for the explicitly named Slurm command's manual page. For multiple named commands, it selects one page chunk per command. It fills remaining slots with the existing BM25 query, keeps cluster boundaries, and does not change site-specific queries. No evaluation answers or references enter retrieval.

The local preparation uses the same SQLite corpus SHA256 as the completed benchmark: `5513283331ad05bf42b40aee94ff387546caa097b75f556842612caa03414577`. Outputs are in `runs/retrieval_command_aware_20261014/`, separate from training and the original experiment.

Offline page coverage for generic questions naming a command:

- v2: original9/14; prepared variant14/14.
- v3: original126/177; prepared variant177/177.

This is page coverage only. It is not evidence that the needed option or fact appears in the selected chunk. In v3-162, neither selected sbatch chunk contains --mem, so option-level coverage still needs attention before spending GPU time. Two focused tests cover command detection, source-slot selection, uniqueness and the LUMI site boundary.

## Before submission

1. Inspect option-level coverage on development examples and improve generic selection if needed; do not insert reference answers or hand-written evaluation-specific corrections.
2. Use CommonMark-aware diagnostics to distinguish quoted shell blocks from literal shell redirects and output tables. Keep frozen metrics, and report changed checker definitions and denominators separately. The prototype audit does not change production verification gates.
3. Wait for the independent Gemma check of the original266 answer pairs. Do not treat judge agreement as truth or missing verdicts as ties.
4. Run a separate comparison using the same model, system prompt, answer budget2048, temperature0, corpus and questions. Change retrieval only. Preserve original caches and freeze the new inputs.
5. Evaluate command and source support alongside preferences. An untouched final evaluation is still needed because v2/v3 are development sets used during iteration.

No new GPU comparison, training data admission, publication or release is authorized by this preparation file itself. Continue only under the user's existing project scope and the validated prerequisites.
