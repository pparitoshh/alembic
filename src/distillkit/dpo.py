"""Plan D5: DPO pairs against invented and missing Slurm flags, and DPO training with a new LoRA adapter.

mine:  sample the start model (an OpenAI-compatible server, e.g. `vllm serve <merged model>`) on prose training
       questions. chosen = the verified teacher answer, rejected = a sample with an invented flag, or one that
       misses at least half of the teacher answer flags (teacher answer with >= 2 flags).
train: TRL DPOTrainer on the pairs. The reference model is the start model with the new adapter off.

Usage:
  python -m distillkit.dpo mine  --url http://127.0.0.1:8000/v1 --model M --train verified.jsonl --out pairs.jsonl
  python -m distillkit.dpo train --model M --pairs pairs.jsonl --out runs/d5_dpo/seed_42 --seed 42
"""

import argparse
import json
import random
from pathlib import Path

from .checks import check_answer, load_flags, slurm_flags_used

SYSTEM_PROMPT_FROM = "configs/qwen3_4b_dora_bf16.yaml"


def prose_rows(rows: list[dict]) -> list[dict]:
    """Rows with 1 user question and 1 assistant answer, no tools."""
    out = []
    for r in rows:
        m = r.get("messages") or []
        if len(m) == 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant" and not m[1].get("tool_calls"):
            out.append(r)
    return out


def rejection_reason(chosen: str, sample: str, valid: set[str]) -> str | None:
    """'bad_flags' if the sample invents a flag, 'missed_flags' if it misses >= half of the teacher flags."""
    if check_answer(sample, valid)["bad_flags"]:
        return "bad_flags"
    tf = set(slurm_flags_used(chosen)) & valid
    if len(tf) >= 2 and len(tf - set(slurm_flags_used(sample))) >= len(tf) / 2:
        return "missed_flags"
    return None


def mine_pairs(rows: list[dict], samples: list[list[str]], system_prompt: str, valid: set[str]):
    """At most 1 pair per question: the first rejected sample. Skips a question whose teacher answer is bad."""
    pairs, why = [], {"bad_flags": 0, "missed_flags": 0, "skipped_bad_chosen": 0}
    for r, ss in zip(rows, samples):
        q, chosen = r["messages"][0]["content"], r["messages"][1]["content"]
        if check_answer(chosen, valid)["bad_flags"]:
            why["skipped_bad_chosen"] += 1
            continue
        for s in ss:
            reason = rejection_reason(chosen, s, valid)
            if reason:
                why[reason] += 1
                pairs.append({"id": r["id"], "reason": reason,
                              "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}],
                              "chosen": [{"role": "assistant", "content": chosen}],
                              "rejected": [{"role": "assistant", "content": s}]})
                break
    return pairs, why


def sample(url: str, model: str, system_prompt: str, questions: list[str], n: int = 2,
           temperature: float = 0.8, max_tokens: int = 512, workers: int = 64) -> list[list[str]]:
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    def ask(q):
        body = {"model": model, "temperature": temperature, "top_p": 0.95, "n": n, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]}
        req = urllib.request.Request(f"{url}/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return [c["message"]["content"] for c in json.load(urllib.request.urlopen(req, timeout=900))["choices"]]

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(ask, questions))


def system_prompt(config: str = SYSTEM_PROMPT_FROM) -> str:
    import yaml

    return yaml.safe_load(open(config))["student"]["system_prompt"]


def run_mine(a) -> None:
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists; choose a new path")
    rows = prose_rows([json.loads(l) for l in open(a.train)])
    random.Random(0).shuffle(rows)
    rows = rows[: a.n]
    sysp, valid = system_prompt(a.config), load_flags()
    samples = sample(a.url, a.model, sysp, [r["messages"][0]["content"] for r in rows])
    pairs, why = mine_pairs(rows, samples, sysp, valid)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[dpo] {len(rows)} questions -> {len(pairs)} pairs {why}")


def run_train(a) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    if (a.out / "adapter").exists():
        raise FileExistsError(f"{a.out}/adapter exists; choose a new directory")
    pairs = [json.loads(l) for l in open(a.pairs)]
    ds = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs])
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    args = DPOConfig(
        output_dir=str(a.out / "checkpoints"), beta=a.beta, loss_type="sigmoid", learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_steps=0.03, num_train_epochs=1, per_device_train_batch_size=2,
        gradient_accumulation_steps=8, max_length=2048, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="no",
        seed=a.seed, report_to="none",
    )
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    # ref_model=None with a peft_config: TRL uses the start model with the adapter off as the reference
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds, processing_class=tok,
                         peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(str(a.out / "adapter"))
    tok.save_pretrained(str(a.out / "adapter"))
    print(f"[dpo] {len(pairs)} pairs, adapter -> {a.out / 'adapter'}")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m distillkit.dpo", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--url", required=True)
    m.add_argument("--model", required=True)
    m.add_argument("--train", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--n", type=int, default=10867)
    m.add_argument("--config", default=SYSTEM_PROMPT_FROM)
    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--pairs", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--beta", type=float, default=0.1)
    t.add_argument("--lr", type=float, default=5e-6)
    a = p.parse_args()
    run_mine(a) if a.cmd == "mine" else run_train(a)


if __name__ == "__main__":
    main()
