# ymmis / Simeon — issue #37 implementation status

Source: https://github.com/pparitoshh/alembic/issues/37, including the 13 October measured-speed correction. Rechecked 14 October local time. One issue by this author exists in the repository.

| Recommendation | Current action and evidence |
|---|---|
| Teacher vs base first | Team report already has teacher v3 0.585 and v2 0.689. Do not confuse these win rates with accuracy or rerun an already answered question unnecessarily. |
| Complete supported answers | New generation prompt removes the artificial short-answer limit. Source support, prerequisites, site identification and uncertainty remain required. More words alone are not a quality win. |
| Fine-tune behaviour; retrieve facts | Running independent base-vs-base+documents comparison; training examples bind the actual source context. Tool-call/clarify/no-call mix is gated. Student+RAG remains a follow-up, not a completed result. |
| NF4 training vs BF16 evaluation mismatch | New pilot uses BF16 LoRA for both. Legacy B7 NF4-vs-BF16 ablation remains unmeasured by this experiment. |
| Paired question statistics | Per-seed and combined reports bootstrap question IDs jointly across seeds. Seed standard deviation is separate. Cross-round paired differences still require matched archived answers. |
| Calibrate actual gpt-oss | Every new evaluation judge runs the existing synthetic calibration first; failure blocks interpreting scores. This is not a real-pair human calibration claim. |
| Real-pair Gemma agreement | Additional dependent GPU job will calibrate Gemma and score all 266 actual student/base pairs for each of 3 seeds. Reports ordered-verdict agreement, position consistency, incomplete verdicts and Gemma win rates separately. |
| About 30 human labels | The same job prepares 30 randomized, anonymized real answer pairs and a separate model key. Human labels remain null until actually provided; no message has been sent to the mentor. |
| Absolute valid-flag metrics | Postprocessor reports both all-answer and flag-containing-answer rates; answers with no flags are not misrepresented as demonstrating flag knowledge. |
| sbatch --test-only | Conservative audit supports complete Leonardo-specific scripts only, without execution/submission. Unknown site, snippets and placeholders are explicitly untested. If frozen v2/v3 provides none, scheduler pass rate is null. |
| Plain LoRA / BF16 | Three seeds queued with the same reviewed input bytes; per-seed differences are not different source splits. |
| Packing | Skipped in line with the mentor's later measured correction for SDPA. |
| Liger | Not enabled without a runtime compatibility/performance check. No change to shared environments; future measured optimization. |
| Reduce validation overhead | Small pilot reserves whole source pages ahead of generation and trains one epoch with no repeated internal validation. A fixed validation subset is still needed for long production training. |
| Mixed-in general/base data | Deferred to an explicit retention ablation after pilot results. No return to the mostly off-topic 14k corpus to fill quotas. |
| DPO | Team's later D5 result was worse; do not repeat blindly or claim the original suggestion has proven a win. |
| Freeze v2 | Existing v2/v3 files remain unchanged; all fresh examples pass source exclusion and question-overlap audits. |

No 99%/100% quality claim, no publication, and no issue closure is made by these preparations. Goal: measurable improvement with non-regression and honest limitations.
