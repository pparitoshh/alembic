"""Stage 4: base vs. distilled student on the held-out eval set.

Metrics: flag hallucination + bash syntax (deterministic) and pairwise LLM-judge win rate
(judged in both orders to cancel position bias).
"""

import json
import re
from pathlib import Path

from .checks import check_answer, load_flags
from .io import read_jsonl, write_jsonl
from .teacher import Teacher

JUDGE_SYSTEM = "You are a strict expert judge of answers about HPC clusters (Slurm, CUDA, MPI). Reply with exactly one character: A, B, or T (tie)."

JUDGE_PROMPT = """Question: {question}

Reference answer (ground truth): {reference}

Answer A:
{a}

Answer B:
{b}

Which answer is more correct and helpful given the reference? Penalise invented options/commands and wrong facts heavily; do not reward length. Reply A, B, or T."""


def generate_answers(cfg: dict, questions: list[str]) -> dict[str, list[str]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .prompting import render_prompt

    scfg = cfg["student"]
    adapter = Path(cfg["run_dir"]) / "adapter"
    tok = AutoTokenizer.from_pretrained(scfg["model"])
    tok.padding_side = "left"
    dtype = torch.float16 if cfg["train"]["fp16"] else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(scfg["model"], dtype=dtype, device_map="auto")
    has_adapter = adapter.exists()
    if has_adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()

    def run_batch(prompts: list[str]) -> list[str]:
        outs = []
        for i in range(0, len(prompts), 8):
            enc = tok(prompts[i : i + 8], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=cfg["eval"]["max_new_tokens"], do_sample=False, pad_token_id=tok.pad_token_id
                )
            outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return [o.strip() for o in outs]

    prompts = [render_prompt(tok, scfg["system_prompt"], q) for q in questions]
    results = {}
    if has_adapter:
        with model.disable_adapter():
            results["base"] = run_batch(prompts)
        results["student"] = run_batch(prompts)
    else:
        results["base"] = run_batch(prompts)
    return results


VERDICT = re.compile(r"\b([ABT])\b")


def parse_verdict(text: str) -> str | None:
    """Last standalone A/B/T in the reply (the judge may reason before its verdict); None if absent."""
    found = VERDICT.findall(text.upper())
    return found[-1] if found else None


def judge(teacher: Teacher, q: dict, a: str, b: str) -> tuple[str, str]:
    raw = teacher.chat(JUDGE_SYSTEM, JUDGE_PROMPT.format(question=q["question"], reference=q["reference"], a=a, b=b), temperature=0.0)
    return parse_verdict(raw) or "T", raw


def run(cfg: dict) -> dict:
    run_dir = Path(cfg["run_dir"])
    eval_rows = read_jsonl(cfg["eval"]["file"])
    valid_flags = load_flags(cfg["verify"]["flag_list"])
    answers = generate_answers(cfg, [r["question"] for r in eval_rows])

    summary = {}
    for name, outs in answers.items():
        checks = [check_answer(o, valid_flags) for o in outs]
        n_flags = sum(c["n_flags"] for c in checks)
        n_bad = sum(len(c["bad_flags"]) for c in checks)
        summary[name] = {
            "check_pass_rate": sum(c["passed"] for c in checks) / len(checks),
            "bad_flag_rate": n_bad / n_flags if n_flags else 0.0,
            "answers_with_bad_flags": sum(bool(c["bad_flags"]) for c in checks),
        }

    per_q = [{**r, **{f"answer_{k}": v[i] for k, v in answers.items()}} for i, r in enumerate(eval_rows)]

    if "student" in answers:
        teacher = Teacher(cfg, section="judge")
        # every question in both orders, judged in parallel: [q0 student=A, q0 student=B, q1 ...]
        pairs = [(row, a, b) for row in per_q for a, b in ((row["answer_student"], row["answer_base"]), (row["answer_base"], row["answer_student"]))]
        verdicts = teacher.map(lambda p: judge(teacher, *p), pairs)
        score, unparsed = 0.0, 0
        for i, row in enumerate(per_q):
            (v1, raw1), (v2, raw2) = verdicts[2 * i], verdicts[2 * i + 1]  # student is A, then B
            pts = {"A": 1.0, "T": 0.5, "B": 0.0}[v1] + {"B": 1.0, "T": 0.5, "A": 0.0}[v2]
            row["judge"] = [v1, v2]
            row["judge_raw"] = [raw1, raw2]
            unparsed += (parse_verdict(raw1) is None) + (parse_verdict(raw2) is None)
            score += pts / 2
        summary["student_vs_base_win_rate"] = score / len(per_q)
        summary["judge_unparsed"] = unparsed  # counted as ties; should be 0

    write_jsonl(run_dir / "eval_outputs.jsonl", per_q)
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary
