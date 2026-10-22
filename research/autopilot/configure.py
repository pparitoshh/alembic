"""Freeze the deployment inputs and render private controllers; does not submit jobs."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

from evidence import EvidenceError, SLICES, leaderboard_snapshot, read_json, sha256
from prepare import write_identical
from supervisor import atomic


ORIGINAL_ROOT = "/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017"


def render(root, python):
    runtime = Path(__file__).resolve().parent
    code = Path(root) / "code"
    config_path = Path(root) / "ctl/autopilot/config.json"
    if config_path.exists():
        raise EvidenceError("A deployment config already exists; inspect it instead of replacing frozen inputs")
    sources = read_json(runtime / "templates/SOURCES.json")
    for name, source in sources.items():
        if sha256(runtime / "templates" / name) != source["sha256"]:
            raise EvidenceError(f"Controller snapshot changed: {name}")

    def shell(name, text):
        text = text.replace(ORIGINAL_ROOT, str(root))
        write_identical(runtime / name, text)
        subprocess.run(["bash", "-n", str(runtime / name)], check=True)

    wrapper = "#!/bin/bash\nexec " + shlex.join([python, str(runtime / "gpu_submit.py"), str(config_path)]) + ' "$@"\n'
    shell("priority_submit.sh", wrapper)
    for name in ("vs_release.sh", "vs_teacher.sh"):
        text = (runtime / "templates" / name).read_text()
        text = text.replace("$R/ctl/priority_submit.sh", str(runtime / "priority_submit.sh"))
        text = text.replace("--nodes=4", "--nodes=8").replace("--nodes=3", "--nodes=8")
        text = re.sub(r'^echo "\$OUT \(setting up\)".*\n', '', text, flags=re.M)
        # Explicitly stop if any required aggregator fails, independent of legacy shell error handling.
        text = re.sub(r'(^\$VPY research/rag65/paired.py[^\n]+)', r'\1 || exit 1', text, flags=re.M)
        text = re.sub(r'(^PYTHONPATH=[^\n]+\$VPY research/rag65/aggregate.py[^\n]+)', r'\1 || exit 1', text, flags=re.M)
        shell(name, text)

    text = (runtime / "templates/stage_release.sh").read_text()
    start, end = text.index("decision=$(python3 - <<PY"), text.index("seed=$(echo $arm")
    approved = 'arm=$("$VPY" ' + shlex.join([str(runtime / "release_card.py"), "approved-arm", str(config_path)]) + ' "$RD") || exit 1\n'
    text = text[:start] + approved + text[end:]
    text = re.sub(r'^base=.*$', 'base=$("$VPY" -c \'import json,sys; print(json.load(open(sys.argv[1]))["student"]["model"])\' "configs/${RD}_seed${seed}.json") || exit 1', text, flags=re.M)
    text = text.replace("$R/ctl/priority_submit.sh", str(runtime / "priority_submit.sh"))
    # The export controller already owns four CPUs and 28 GiB. A nested serial allocation
    # would exceed the per-user CPU/memory cap and wait on its own parent indefinitely.
    text, changes = re.subn(r'srun -A EUHPC_D30_031 -p lrd_all_serial -c 4 --mem=30G -t 02:00:00 -J dk-gguf-\$RD bash -c',
                           'bash -c', text)
    if changes != 1:
        raise EvidenceError("Export allocation expression changed")
    text = text.replace('[ -s $G/merged/config.json ] || ${base:+BASE=$base} $VPY research/dpo/merge.py $adapter $G/merged',
        '[ -s $G/merged/MERGED_READY ] || ( ${base:+BASE=$base} $VPY research/dpo/merge.py $adapter $G/merged && echo complete > $G/merged/MERGED_READY )')
    text = text.replace('bash research/gguf/export.sh', 'bash ' + str(runtime / 'export_gguf.sh'))
    text = text.replace('ln -sf $C/$G/merged/model.safetensors $H/model.safetensors',
        'for f in $G/merged/model*.safetensors $G/merged/model.safetensors.index.json; do\n'
        '    [ -f "$f" ] && ln -sf "$C/$f" "$H/$(basename "$f")"\n'
        'done')
    text = re.sub(r'(^\$VPY research/rag65/(?:aggregate|paired).py[^\n]+)', r'\1 || exit 1', text, flags=re.M)
    card = '"$VPY" ' + shlex.join([str(runtime / "release_card.py"), "write", str(config_path)]) + ' "$RD" || exit 1\n'
    audit = '"$VPY" ' + shlex.join([str(runtime / "supervisor.py"), "audit-export", str(config_path)]) + ' "$RD" || exit 1\n'
    text = text.replace('echo "$RD $arm" > $H/READY', card + audit + 'echo "$RD $arm" > $H/READY')
    shell("stage_release.sh", text)

    export = (code / "research/gguf/export.sh").read_text()
    export = export.replace(
        '[ -s $O/model-bf16.gguf ] || PYTHONPATH=$R/llama.cpp/gguf-py $VPY $R/llama.cpp/convert_hf_to_gguf.py $M --outtype bf16 --outfile $O/model-bf16.gguf',
        'if [ ! -s $O/model-bf16.gguf ]; then\n'
        '    PYTHONPATH=$R/llama.cpp/gguf-py $VPY $R/llama.cpp/convert_hf_to_gguf.py $M --outtype bf16 --outfile $O/model-bf16.partial.gguf\n'
        '    mv $O/model-bf16.partial.gguf $O/model-bf16.gguf\nfi')
    export = export.replace('[ -s $O/model-$q.gguf ] || $QUANT $O/model-bf16.gguf $O/model-$q.gguf $q 4',
        'if [ ! -s $O/model-$q.gguf ]; then\n'
        '        $QUANT $O/model-bf16.gguf $O/model-$q.partial.gguf $q 4\n'
        '        mv $O/model-$q.partial.gguf $O/model-$q.gguf\n    fi')
    shell("export_gguf.sh", export)

    candidates = (code / "research/bestofn/candidates.py").read_text()
    if "'seed': 1000 + k" not in candidates:
        raise EvidenceError("Candidate seed expression changed")
    candidates = candidates.replace("'seed': 1000 + k", "'seed': int(os.environ['AUTOPILOT_SAMPLE_SEED']) + k")
    candidates = candidates.replace("ONLY_ORIGIN = os.environ.get('ONLY_ORIGIN')",
        "LENGTH_NOTE += os.environ.get('AUTOPILOT_TRAINING_NOTE', '')\n\nONLY_ORIGIN = os.environ.get('ONLY_ORIGIN')")
    candidates = candidates.replace('    for x in dest.read_text().splitlines():',
        '    good = []\n    for x in dest.read_text().splitlines():').replace(
        "            done.add(json.loads(x)['id'])", "            done.add(json.loads(x)['id'])\n            good.append(x)")
    candidates = candidates.replace('todo = [r for r in rows',
        "    dest.write_text(''.join(line + '\\n' for line in good))\ntodo = [r for r in rows")
    write_identical(runtime / "candidates.py", candidates)
    batch = (code / "research/kd/onpolicy_gen.sbatch").read_text()
    batch = batch.replace("research/bestofn/candidates.py", str(runtime / "sample_units.py"))
    shell("onpolicy_gen.sbatch", batch)
    tool = (code / "research/rag65/tool_eval.sbatch").read_text()
    tool = tool.replace("Qwen/Qwen3-4B-Instruct-2507", '"${BASE_MODEL:?}"')
    shell("tool_eval.sbatch", tool)

    protected = [str(p.relative_to(code)) for p in sorted((code / "data/eval").rglob("*final*.jsonl"))]
    inputs = set(SLICES.values()) | set(protected) | {
        "data/eval/dataset_v1/development_tools.jsonl", "configs/rag65_seed42.json",
        *[f"configs/rag65kd_seed{s}.json" for s in (42, 43, 44)]}
    for folder in ("research/rag65", "research/kd", "research/rft", "research/gguf", "research/dpo", "research/teacher_probe", "src/distillkit"):
        inputs.update(str(p.relative_to(code)) for p in (code / folder).rglob("*")
                      if p.is_file() and p.suffix in (".py", ".sbatch", ".sh"))
    bank = "runs/rag65v9/data/verified.jsonl"
    baseline_model = "runs/gguf_r12e/merged"
    baseline_files = sorted((code / baseline_model).glob("model*.safetensors"))
    if not baseline_files:
        raise EvidenceError("Frozen release 12e model weights are missing")
    # Freeze the published reference answers, too. Re-running a different baseline silently changes the goal.
    inputs.update(str(p.relative_to(code)) for p in (code / "runs/rag65kd9/answers").glob("*__s44_rag.jsonl"))
    inputs.update(str(p.relative_to(code)) for p in (code / "runs/rag65kd9/answers").glob("*__base*.jsonl"))
    config = {"version": 1, "root": str(root), "code": str(code), "python": python,
        "account": "EUHPC_D30_031", "cpu_partition": "lrd_all_serial", "alpha": .05,
        "gpu_node_limit": 8, "gpus_per_node": 4, "gpu_qos": "boost_qos_dbg",
        "poll_seconds": 10, "service_seconds": 13800,
        "pool": {"enabled": True, "foreground_elastic_nodes": 4, "lookahead_rounds": 6},
        "reports": {"enabled": True, "interval_seconds": 1800},
        "factory": {"batch": "ai_factory_all_v8", "controller": "factory_v8"},
        "existing_rounds": ["rft1", "rag65kd11"], "teacher_bank": bank,
        "teacher_bank_sha256": sha256(code / bank), "baseline_model": baseline_model,
        "baseline_weights": {str(p.relative_to(code)): sha256(p) for p in baseline_files},
        "pinned_files": {p: sha256(code / p) for p in sorted(inputs)},
        "protected_question_files": protected,
        "target": {"minimum_score": .90, "rank": 1, "scope": "frozen internal HPC leaderboard", "judges": ["gpt_oss", "gemma"]},
        "leaderboard_snapshot": leaderboard_snapshot(code),
        "stop_condition": "At least 90% on each of the three development benchmarks under both judges and first place on the frozen internal leaderboard; also beat 12e, preserve significance and tool/command checks, then export and verify.",
        "research_policy": "RFT on rotating thirds of the frozen teacher bank with new reproducible sample seeds; alternate concise-completion and matched-prompt training samples. Evaluation prompts, budgets and thresholds remain fixed. No threshold relaxation or final-holdout evaluation.",
        "publication": "Stage only; no credentials or automatic public upload."}
    atomic(config_path, config)
    return config


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--python", required=True)
    args = p.parse_args()
    config = render(Path(args.root), args.python)
    print(json.dumps({"configured": str(Path(args.root) / "ctl/autopilot/config.json"),
                      "pinned_files": len(config["pinned_files"]), "baseline_weight_files": len(config["baseline_weights"])}))
