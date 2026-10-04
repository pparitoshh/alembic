"""Stage 4: base vs. distilled student on the held-out eval set.

Metrics: flag hallucination + bash syntax (deterministic) and pairwise LLM-judge win rate
(judged in both orders to cancel position bias). The optional tool slice (`eval.tool_file`) is
answered with the tool schemas in the prompt and scored by `toolcheck`: when-to-call accuracy,
call validity, BFCL-style AST match and execution on the mock cluster.
"""

import json

from .checks import check_answer, load_flags
from .config import Config
from .io import read_jsonl, write_jsonl
from .schemas import JudgeVerdict
from .teacher import Teacher
from .toolcheck import score_tool_item
from .tools import SCHEMAS

JUDGE_SYSTEM = "You are a strict expert judge of answers about HPC clusters (Slurm, CUDA, MPI). Reply only with JSON: {\"reasoning\": \"<short comparison>\", \"verdict\": \"A\" | \"B\" | \"T\"} (T = tie)."

JUDGE_PROMPT = """Question: {question}

Reference answer (ground truth): {reference}

Answer A:
{a}

Answer B:
{b}

Which answer is more correct and helpful given the reference? Penalise invented options/commands and wrong facts heavily; do not reward length. Reply with the JSON verdict."""


def generate_answers(cfg: Config, questions: list[str], tools: list[list[dict] | None] | None = None) -> dict[str, list[str]]:
    """Greedy answers from the base model and, if an adapter exists, the student. `tools[i]` are
    the tool schemas offered with question i (None = plain question)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .prompting import render_prompt

    scfg = cfg.student
    adapter = cfg.run_dir / "adapter"
    tok = AutoTokenizer.from_pretrained(scfg.model)
    tok.padding_side = "left"
    # (Q)DoRA adapters are applied to the unquantized base, as in the merged model we export
    model = AutoModelForCausalLM.from_pretrained(scfg.model, dtype=getattr(torch, cfg.train.dtype), device_map="auto")
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
                    **enc, max_new_tokens=cfg.eval.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
                )
            outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return [o.strip() for o in outs]

    tools = tools or [None] * len(questions)
    prompts = [render_prompt(tok, scfg.system_prompt, q, t) for q, t in zip(questions, tools)]
    results = {}
    if has_adapter:
        with model.disable_adapter():
            results["base"] = run_batch(prompts)
        results["student"] = run_batch(prompts)
    else:
        results["base"] = run_batch(prompts)
    return results


def tool_summary(scores: list[dict], rows: list[dict]) -> dict:
    """Rates over the tool slice; validity/AST/execution only over items where a call was expected."""
    expected = [(s, r) for s, r in zip(scores, rows) if r["expect"] == "call"]
    called = [s for s, _ in expected if s["decision"] == "call"]
    return {
        "tool_decision_acc": sum(s["decision_ok"] for s in scores) / len(scores),
        "tool_false_call_rate": sum(s["decision"] == "call" for s, r in zip(scores, rows) if r["expect"] == "no_call") / max(1, sum(r["expect"] == "no_call" for r in rows)),
        "tool_valid_rate": sum(s["valid"] for s in called) / max(1, len(called)),
        "tool_ast_acc": sum(s.get("ast_ok", False) for s, _ in expected) / max(1, len(expected)),
        "tool_exec_ok_rate": sum(s["exec_ok"] for s in called) / max(1, len(called)),
        # over every item where the model called anything: no invented job ids
        "tool_grounded_rate": sum(s["grounded"] for s in scores if "grounded" in s) / max(1, sum("grounded" in s for s in scores)),
    }


def judge(teacher: Teacher, q: dict, a: str, b: str) -> tuple[str | None, str]:
    """(verdict, raw reply); verdict is None if the judge never returned valid JSON."""
    prompt = JUDGE_PROMPT.format(question=q["question"], reference=q["reference"], a=a, b=b)
    v, raw = teacher.chat_json(JUDGE_SYSTEM, prompt, JudgeVerdict, temperature=0.0)
    return (v.verdict if v else None), raw


def run(cfg: Config) -> dict:
    run_dir = cfg.run_dir
    eval_rows = read_jsonl(cfg.eval.file)
    tool_rows = read_jsonl(cfg.eval.tool_file) if cfg.eval.tool_file else []
    valid_flags = load_flags(cfg.verify.flag_list)
    # one model load for both slices; tool questions are asked with the tool schemas in the prompt
    all_answers = generate_answers(cfg, [r["question"] for r in eval_rows + tool_rows], [None] * len(eval_rows) + [SCHEMAS] * len(tool_rows))
    answers = {k: v[: len(eval_rows)] for k, v in all_answers.items()}
    tool_answers = {k: v[len(eval_rows) :] for k, v in all_answers.items()}

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

    if tool_rows:
        per_tool = [{**r, **{f"answer_{k}": v[i] for k, v in tool_answers.items()}} for i, r in enumerate(tool_rows)]
        for name in tool_answers:
            scores = [score_tool_item(row[f"answer_{name}"], row) for row in per_tool]
            for row, sc in zip(per_tool, scores):
                row[f"tool_score_{name}"] = sc
            summary[name] |= tool_summary(scores, tool_rows)
        write_jsonl(run_dir / "eval_tool_outputs.jsonl", per_tool)

    if "student" in answers:
        teacher = Teacher(cfg.judge, name="judge")
        # every question in both orders, judged in parallel: [q0 student=A, q0 student=B, q1 ...]
        pairs = [(row, a, b) for row in per_q for a, b in ((row["answer_student"], row["answer_base"]), (row["answer_base"], row["answer_student"]))]
        verdicts = teacher.map(lambda p: judge(teacher, *p), pairs)
        score, unparsed = 0.0, 0
        for i, row in enumerate(per_q):
            (v1, raw1), (v2, raw2) = verdicts[2 * i], verdicts[2 * i + 1]  # student is A, then B
            unparsed += (v1 is None) + (v2 is None)
            v1, v2 = v1 or "T", v2 or "T"
            pts = {"A": 1.0, "T": 0.5, "B": 0.0}[v1] + {"B": 1.0, "T": 0.5, "A": 0.0}[v2]
            row["judge"] = [v1, v2]
            row["judge_raw"] = [raw1, raw2]
            score += pts / 2
        summary["student_vs_base_win_rate"] = score / len(per_q)
        summary["judge_unparsed"] = unparsed  # counted as ties; should be 0

    write_jsonl(run_dir / "eval_outputs.jsonl", per_q)
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary
