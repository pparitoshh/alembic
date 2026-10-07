"""Stage 2: dedup -> deterministic checks (answer + tool calls) -> decontamination against the eval set."""

import re
import json
from collections import Counter
from pathlib import Path

from .checks import check_answer, load_flags
from .config import Config
from .io import read_jsonl, write_jsonl
from .records import final_answer, messages
from .toolcheck import check_trace
from .tools import SCHEMAS, set_valid_flags
from .job_status_guard import WORKFLOW_VERSION, JobStatusPolicy
from .support import check_support, source_for_record
from .source_registry import admitted_sources, record_source_binding


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def run(cfg: Config) -> Path:
    vcfg, run_dir = cfg.verify, cfg.run_dir
    admitted_sources(cfg)  # fail before writing outputs if the configured source basis is invalid
    # generate appends rows in completion order; sort so "first copy wins" in dedup is reproducible
    rows = sorted(read_jsonl(run_dir / "generated.jsonl"), key=lambda r: r.get("id", ""))
    valid_flags = load_flags(vcfg.flag_list)
    set_valid_flags(valid_flags)  # tool checks run submit_job on the mock sbatch
    eval_shingles = [_shingles(r["question"]) for r in read_jsonl(cfg.eval.file)]

    kept, rejected, pending_review = [], [], []
    reviews, review_errors = {}, {}
    if vcfg.grounding_reviews:
        try:
            for line in vcfg.grounding_reviews.read_text().splitlines():
                if not line.strip():
                    continue
                report = json.loads(line)
                if not isinstance(report, dict) or not isinstance(report.get('id'), str):
                    raise ValueError('review needs an id')
                rid = report['id']
                if rid in reviews:
                    review_errors[rid] = 'duplicate support-review IDs'
                reviews[rid] = report
        except (OSError, ValueError) as exc:
            review_errors['*'] = f'cannot read support reviews: {exc}'
    kept_questions: dict[str, set] = {}  # question text -> shingles (N answers share one question)

    for r in rows:
        reason = None
        q, a = r["question"], final_answer(r)
        try:
            record_source_binding(cfg, r)
        except (OSError, ValueError) as exc:
            pending_review.append({**r, "pending_reason":"source_admission_unresolved",
                                   "source_admission_error":str(exc)})
            continue
        if not q or not a or len(a) > vcfg.max_answer_chars:
            reason = "empty_or_too_long"
        if reason is None and q not in kept_questions:
            sh = _shingles(q)
            if any(_jaccard(sh, e) >= vcfg.dedup_threshold for e in eval_shingles):
                reason = "eval_contamination"
            elif any(_jaccard(sh, k) >= vcfg.dedup_threshold for k in kept_questions.values()):
                reason = "near_duplicate"
        if reason is None:
            checks = check_answer(a, valid_flags)
            r = {**r, "checks": checks}
            if checks["bad_flags"]:
                reason = "bad_flags"
            elif not checks["bash_ok"]:
                reason = "bash_syntax"
        transcript = messages(r)
        tool_shaped = (r.get('tools') is not None or r.get('workflow_version') is not None or
                       r.get('tool_policy') is not None or
                       any(isinstance(m, dict) and (m.get('role') == 'tool' or m.get('tool_calls') or
                                                   m.get('origin') == 'runtime_guard')
                           for m in (transcript if isinstance(transcript, list) else [])))
        if reason is None and tool_shaped and r.get('mode') not in ('call', 'ask', 'none'):
            reason = 'tool_mode_mismatch'  # a label cannot bypass replay of tool evidence
        if reason is None and r.get("mode") in ("call", "ask", "none"):
            tool = check_trace(r, valid_flags)
            r = {**r, "tool_checks": tool}
            if cfg.generate.job_status_discovery and r.get('tool_policy') != JobStatusPolicy(cfg.generate.job_status_discovery).as_dict():
                reason = 'application_policy_mismatch'
            elif tool.get('application_policy_errors'):
                reason = 'application_policy_violation'  # blocked attempts are not model compliance
            elif tool["call_errors"]:
                reason = "tool_call_invalid"  # unknown tool, bad arguments, or the mock rejected it
            elif tool["trace_errors"]:
                reason = "tool_trace_incomplete_or_out_of_order"
            elif tool["result_errors"]:
                reason = "tool_result_mismatch"
            elif tool["ungrounded_ids"]:
                reason = "tool_ungrounded_id"  # a job id not taken from the question or a tool result
            elif tool["ungrounded_partitions"]:
                reason = "tool_ungrounded_partition"
            elif tool.get("workflow_errors"):
                reason = "workflow_unsupported_response"
            elif not tool["decision_ok"]:
                reason = "tool_decision"  # called when it should have answered/asked, or the reverse
        requires_support = vcfg.require_grounding_review or r.get('workflow_version') == WORKFLOW_VERSION
        if reason is None and requires_support:
            try:
                if r.get('doc_id') in cfg.seeds.eval_docs:
                    reason = 'heldout_source'
                elif r.get('tools') is not None and r['tools'] != SCHEMAS:
                    raise ValueError('record schemas differ from the configured tool implementation')
                else:
                    source = source_for_record(cfg, r)
                    issue = review_errors.get('*') or review_errors.get(r.get('id'))
                    support = ({'status':'uncertain','reason':issue} if issue else
                               check_support(r, source, reviews.get(r.get('id')),
                                             protocol=vcfg.grounding_review_protocol))
                    r = {**r, 'grounding_checks':support}
                    if support['status'] == 'unsupported':
                        reason = 'unsupported_claim'
                    elif support['status'] != 'supported':
                        pending_review.append({**r, 'pending_reason':'grounding_review_unresolved'})
                        continue
            except (OSError, ValueError, TypeError, KeyError) as exc:
                pending_review.append({**r, 'pending_reason':'grounding_evidence_unresolved',
                                       'grounding_checks':{'status':'uncertain','reason':str(exc)}})
                continue
        if reason:
            rejected.append({**r, "reject_reason": reason})
        else:
            kept.append(r)
            kept_questions[q] = _shingles(q)  # pending/rejected rows do not reserve the accepted copy

    write_jsonl(run_dir / "verified.jsonl", kept)
    write_jsonl(run_dir / "rejected.jsonl", rejected)
    write_jsonl(run_dir / "pending_review.jsonl", pending_review)
    reasons = Counter(r["reject_reason"] for r in rejected)
    print(f"[verify] kept {len(kept)}/{len(rows)}; rejected: {dict(reasons)}; pending review: {len(pending_review)}")
    return run_dir / "verified.jsonl"
