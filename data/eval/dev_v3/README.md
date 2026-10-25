# DistillKit development evaluation RC v3

Dataset owners: Vaibhav Mangroliya and Shraddha Jadhav. This is the existing,
development-exposed evaluation set, corrected and versioned. It is not a blind
final test or a complete HPC benchmark. No new model answers were generated.

| Slice | Ready | Draft/pending | Scope |
| --- | ---: | ---: | --- |
| `normal.jsonl` | 66 | 0 | Source-grounded normal development questions/references |
| `tools.jsonl` | 16 | 0 | First-response tool decision, schema, arguments and isolated mock execution |
| Existing judge calibration | 24 pairs, separate | — | Judge calibration only |
| Campaign diagnostics | Separate, excluded | — | Exposed application/reviewer diagnostics, never training or final test |
| Fresh final test | 0 | Not yet constructed | Must use new held-out document families |

“Ready” means ready for the stated **local development measurement scope** after
model review and automated checks. It does not resolve source licensing, authorize
external publication, or establish multi-turn task completion. All 18 unknown
URL/version/license fields across the original six sources remain unknown.

The normal slice contains the original 10 core + 16 requeue rows, including
req-016 once, plus the previously reviewed 40-row supplement. The older 26-row
`eval_all.jsonl` is not appended again. Original bytes and decisions are retained.

Only three references change; all 66 normal questions remain identical:

- arr-003: remove the unsupported log-directory prerequisite. `%A` and `%a` remain;
  `%x` appears only in the source's example pattern, with no inferred definition.
- req-002: remove undocumented preemption/node-failure triggers; retain the
  documented permission, same job ID and manual requeue command.
- req-011: retain the documented memory state and supply the documented `sacct`
  inspection command; remove unsupported memory-request options/remedy. The source
  scope note is judging guidance, not a sentence an answer must reproduce.

`corrections.json` preserves before/after values and exact local source quotes.
`source_support.json` binds every reference to its assigned document hash. The
inherited supplemental/overlap review is preserved, not represented as a new blind
review of 66 independent capabilities. No external source facts were injected.

## Tool review and scoring

All 16 drafts received coordinator model review and independent model review of
schemas, mock behavior, user intent and labels, followed by production-path CPU
tests. No human review is claimed or required; model review can be wrong.
`tool_review.json` preserves every before/after item and rationale.

| Items | Disposition / accepted first response |
| --- | --- |
| tool-001 | Ready: `job_status`, explicit ID |
| tool-002 | Ready: `read_job_log`, explicit ID, 50 lines, stderr (or documented default) |
| tool-003 | Ready: `gpu_availability`, named partition |
| tool-004 | Ready: explicitly requested `cancel_job`, identified ID, mock only |
| tool-005 | Ready: `partition_info`, named partition |
| tool-006 | Ready: `list_queue`, PENDING filter, no invented partition |
| tool-007 | Ready: `job_accounting`, explicit ID |
| tool-008 | Ready: `job_status` or `job_accounting`, explicit ID |
| tool-009 | Corrected/ready: `submit_job`, exact supplied script, `test_only=true` |
| tool-010 | Ready: unfiltered `gpu_availability` |
| tool-011, tool-012, tool-013 | Corrected/ready: no call; clarify missing job identity under visible policy |
| tool-014, tool-015, tool-016 | Ready: no call for general explanation/script drafting |

`tool-first-response-v2` uses the deployment-visible `hpc-tools-clarify-first-v1`
policy, appended to every tool question's system prompt. It limits this
first-response measurement to at most one call; no hidden expected label enters
the prompt. Normal questions receive their original system prompt and no tools.
The same policy is applied by Hugging Face and GGUF answer paths.

The scorer checks the whole emitted call list and malformed delimiters; multiple
calls, guessed IDs, invalid schema and incorrect arguments fail. Script comparison
preserves case/content (only CRLF and one terminal newline are normalized), replacing
the old any-script wildcard. Each score uses a fresh mock session. Bash syntax or
mock execution does not establish real Slurm execution or complete script validity.

A separately selectable `hpc-tools-readonly-discovery-v1` policy permits exactly an
argument-free `list_queue` discovery as an alternative where a matching dataset
explicitly allows it. It has a different visible prompt and cache binding. This
release uses clarify-first and has no such exceptions. Regression tests establish
that permitted discovery is excluded from the false-call numerator, while raw call
rate still counts it. No action or guessed filter is treated as discovery.

First-response pass is **not** factual answer quality, actual clarification quality,
multi-turn result grounding or delivered task success. In particular, a nonempty
no-call response may satisfy the decision metric while answering badly. These
unmeasured outcomes are labelled “not assessed,” not inferred as successes.

## Comparison compatibility

All 82 original baseline answers remain preserved. Normal 66 answering inputs are
unchanged: those cached answers can be rescored under normal-v3 references, applying
the same version to every compared system. No new judge calls/rescoring occurred.
All 16 tool answering prompts change, so their original answers are **incompatible**
with this tool-policy version; they are not silently reused or regenerated here.

Opt-in `config_overrides.json` selects the versioned files/scorer and a new run
directory. Existing configs/default scorer remain v1. V2 caches have new filenames
and bind questions, schemas, system/policy text, model name, token cap and backend
settings. Frozen model/tokenizer snapshots, runtime builds and weight content hashes
must still be recorded per comparison: the cache is not a full run manifest.
Legacy cache limitations remain unchanged. The judge protocol is unchanged.

## Coverage and use

| Coverage | Items / gap |
| --- | --- |
| Arrays, task identity, log patterns, control/dependencies | 30 |
| Requeue, signals/traps, state/hold/checkpoint fragments | 36 |
| Normal tasks | 18 concept, 20 how-to, 14 debug, 14 script |
| Mock tools | 10 call, 3 missing-ID/no-call, 3 general/no-call |
| CUDA, MPI, profiling, broad Linux/HPC debugging | Missing |
| Multi-turn tool result interpretation/task completion | Not measured by this slice |
| New source-held-out blind final test | Missing |

Run `python data/eval/dev_v3/validate_release.py` for exact schema, hash, reference
change and duplicate checks. `tests/test_eval_dev_v3.py` covers production scorer,
cache and prompt-path regressions using fake inference and isolated mocks.
`validation.json` records checks already run. Exact duplicate checks do not detect
all semantic overlap; related applications retain their documented inherited status.

Keep evaluation questions, references, paraphrases, both held-out source families,
calibration pairs and diagnostic fixtures out of training, gold prompts and retrieval.
This RC is a local dataset handoff; it is not an external release authorization.
