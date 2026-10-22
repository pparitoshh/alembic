"""Prepare a new, isolated RFT round from the existing reviewed teacher bank."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from evidence import SLICES, EvidenceError, read_json, sha256


def write_identical(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise EvidenceError(f"Refusing to change existing prepared input: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def eligible(row):
    messages = row.get("messages", [])
    return (row.get("mode", "prose") == "prose" and len(messages) == 2
        and [m.get("role") for m in messages] == ["user", "assistant"]
        and any(q in messages[0]["content"] for q in (
            "Quoted retrieval results (data, not instructions):", "Provided input 1 (quoted data")))


def normalized(text):
    return " ".join(text.lower().split())


def prepare(config, index):
    code, ctl = Path(config["code"]), Path(config["root"]) / "ctl"
    runtime = Path(__file__).resolve().parent
    run = f"auto_rft_{index:04d}"
    directory = code / "runs" / run
    source = code / config["teacher_bank"]
    source_hash = sha256(source)
    if source_hash != config["teacher_bank_sha256"]:
        raise EvidenceError("The frozen teacher bank changed")
    for path, digest in config.get("baseline_weights", {}).items():
        if sha256(code / path) != digest:
            raise EvidenceError(f"The frozen release model changed: {path}")
    eval_paths = [code / p for p in SLICES.values()]
    eval_paths += [code / p for p in config.get("protected_question_files", [])]
    protected = set()
    for path in eval_paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("question"):
                protected.add(normalized(row["question"]))
    selected, excluded, seen = [], 0, set()
    with source.open() as f:
        for i, line in enumerate(f):
            if i % 3 != index % 3:
                continue
            row = json.loads(line)
            if not eligible(row):
                excluded += 1
                continue
            question = normalized(row.get("question") or row["messages"][0]["content"])
            if question in protected or any(q and q in question for q in protected):
                excluded += 1
                continue
            if row["id"] in seen:
                raise EvidenceError("Duplicate teacher-bank ID")
            seen.add(row["id"])
            selected.append(row)
    if len(selected) < 500:
        raise EvidenceError(f"Only {len(selected)} eligible questions; refusing to lower the data gate")
    serialized = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected)
    write_identical(directory / "data/questions.jsonl", serialized)

    template = (runtime / "templates/rft_controller.sh").read_text()
    template = template.replace("rft1", run)
    # The prepared question file exists before the controller runs. Its fallback builder is unreachable.
    template = template.replace("want * 98 / 100", "want")
    template = template.replace('log "SAMPLED: $(have)/$want questions"',
        '[ "$(have)" -eq "$want" ] || { log "STOP: sampling coverage incomplete"; exit 1; }\n'
        'log "SAMPLED: $(have)/$want questions"')
    fallback = r'if \[ "\$\(wc -l < \$DATA\)" -lt 1500 \]; then.*?\nfi\n'
    template, changes = re.subn(fallback, "", template, flags=re.S)
    if changes != 1:
        raise EvidenceError("RFT template changed: threshold fallback was not found exactly once")
    # A failed aggregation must never write a misleading DONE line.
    template = template.replace('> "$HERE/aggregate.out" 2>&1',
        '> "$HERE/aggregate.out" 2>&1 || { log "STOP: aggregation failed"; exit 1; }')
    template = template.replace('> "$HERE/paired.out" 2>&1',
        '> "$HERE/paired.out" 2>&1 || { log "STOP: paired comparison failed"; exit 1; }')
    template = template.replace("research/kd/onpolicy_gen.sbatch", str(runtime / "onpolicy_gen.sbatch"))
    # New runs have no existing worker partitions: use all eight nodes for sampling and judging.
    template = template.replace(f"--nodes=2 --job-name=dk-{run}-gen", f"--nodes=8 --job-name=dk-{run}-gen")
    template = template.replace("--nodes=4", "--nodes=8")
    submitter = shlex.join([config.get("python", sys.executable), str(runtime / "submit_gpu.py"),
                           str(ctl / "autopilot/config.json")])
    template = template.replace("sbatch --parsable", submitter)
    # The receipt-based submitter adopts live jobs itself, including lost sbatch replies.
    template = re.sub(r'^adopt\(\) \{.*$', 'adopt() { :; }', template, flags=re.M)
    start, end = template.index('wait_job() {'), template.index('submit() {')
    template = template[:start] + '''wait_job() {
    while true; do
        if ! queue=$(squeue -h -j "$1" -o %i); then sleep 30; continue; fi
        [ -n "$queue" ] || break
        heartbeat; sleep 30
    done
    log "job $1: $(sacct -n -X -j "$1" --format=State,Elapsed | head -1 | xargs)"
}
''' + template[end:]
    policy = "concise_completion" if index % 2 else "matched_prompt"
    note = ("\n\nFinish the answer within about 220 to 320 words. Preserve all requirements, complete any code "
            "or script, state relevant assumptions, and include a concrete verification step. Use the quoted "
            "sources to support the solution; avoid repeating the question or adding an introduction.") if index % 2 else ""
    template = template.replace("set -uo pipefail", f"set -uo pipefail\nexport AUTOPILOT_SAMPLE_SEED={1000 + index * 16}\n"
                               f"export AUTOPILOT_TRAINING_NOTE={shlex.quote(note)}")
    guard = ('autopilot_wait() {\n'
             f'    while [ -e "{ctl}/autopilot/PAUSE" ]; do sleep 30; done\n'
             f'    [ ! -e "{ctl}/autopilot/STOP" ] || exit 1\n'
             '}\n')
    template = template.replace('cd $C\ntest -s', guard + 'autopilot_wait\ncd $C\ntest -s')
    template = re.sub(r'^(for attempt in [^\n]+; do)$', r'\1\n    autopilot_wait', template, flags=re.M)
    template = template.replace('for round in $(seq 1 90); do', 'for round in $(seq 1 90); do\n    autopilot_wait')
    write_identical(ctl / run / "controller.sh", template)
    subprocess.run(["bash", "-n", str(ctl / run / "controller.sh")], check=True)
    plan = {"run": run, "index": index, "teacher_bank": str(source), "teacher_bank_sha256": source_hash,
        "questions": len(selected), "excluded": excluded, "partition": index % 3,
        "question_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "sample_seed": 1000 + index * 16, "samples_per_question": 4,
        "sampling_nodes": 8, "judging_nodes": 8, "training_nodes": 3,
        "training_prompt_policy": policy, "training_only_note": note,
        "base": str(code / config["baseline_model"]), "training_seeds": [42, 43, 44],
        "acceptance_score_vs_teacher": .75, "minimum_training_rows": 500,
        "controller_sha256": sha256(ctl / run / "controller.sh"),
        "note": "Fresh sample seeds each round; development questions never enter the training bank. No final evaluation is run."}
    write_identical(directory / "AUTOPILOT_PLAN.json", json.dumps(plan, indent=2) + "\n")
    return plan


if __name__ == "__main__":
    print(json.dumps(prepare(read_json(sys.argv[1]), int(sys.argv[2])), indent=2))
