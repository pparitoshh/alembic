"""Cluster workflow: config overrides, judge calibration, cached answers + tagged judges, seed
aggregation, and consistency of the Slurm scripts with the code (hermetic: only bash is run)."""

import json
import py_compile
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from distillkit import evaluate
from distillkit.aggregate import aggregate
from distillkit.calibrate import score
from distillkit.config import Config, apply_overrides, load_config
from distillkit.io import read_jsonl
from distillkit.schemas import JudgeVerdict

ROOT = Path(__file__).parent.parent
SLURM = ROOT / "slurm"
LEONARDO = ROOT / "configs/qwen3_4b_qdora.yaml"


# --- --set overrides -----------------------------------------------------------------------------


def test_apply_overrides():
    raw = {"run_dir": "runs/a", "train": {"seed": 42}, "judge": {"api_key_env": "X"}}
    apply_overrides(raw, ["run_dir=runs/b/seed_1", "train.seed=1", "train.bf16=true", "judge.api_key_env=null", "eval.tag=", "export.quants=[Q4_K_M]"])
    assert raw == {
        "run_dir": "runs/b/seed_1",
        "train": {"seed": 1, "bf16": True},
        "judge": {"api_key_env": None},
        "eval": {"tag": ""},
        "export": {"quants": ["Q4_K_M"]},
    }
    with pytest.raises(ValueError, match="section.key=value"):
        apply_overrides({}, ["train.seed"])
    with pytest.raises(ValueError, match="not a section"):
        apply_overrides({"run_dir": "x"}, ["run_dir.sub=1"])


def test_overrides_are_validated():
    cfg = load_config(LEONARDO, ["run_dir=runs/x/seed_1", "train.seed=1", "judge.base_url=http://127.0.0.1:20001/v1"])
    assert (str(cfg.run_dir), cfg.train.seed, cfg.judge.base_url) == ("runs/x/seed_1", 1, "http://127.0.0.1:20001/v1")
    with pytest.raises(ValidationError):
        load_config(LEONARDO, ["train.sede=1"])  # typo in a cluster job fails at submission, not hours later


# --- judge calibration ---------------------------------------------------------------------------


def test_calibration_score():
    rows = [{"label": "A", "defect": "invented_flag"}, {"label": "B", "defect": "wrong_fact"}, {"label": "T", "defect": "tie"}]
    perfect = score(rows, [("A", "B"), ("B", "A"), ("T", "T")])
    assert perfect["accuracy"] == 1 and perfect["position_consistency"] == 1
    biased = score(rows, [("A", "A"), ("A", "A"), ("A", "A")])  # always picks the first answer
    assert biased["accuracy"] == 2 / 6 and biased["position_consistency"] == 0
    assert biased["accuracy_by_defect"] == {"invented_flag": 0.5, "tie": 0.0, "wrong_fact": 0.5}
    assert score(rows[:1], [(None, "B")])["unparsed"] == 1


def test_calibration_set():
    rows = read_jsonl(ROOT / "data/eval/judge_calibration.jsonl")
    assert len({r["id"] for r in rows}) == len(rows) >= 20
    for r in rows:
        assert r["label"] in "ABT" and r["answer_a"] != r["answer_b"] and r["reference"]
        assert (r["label"] == "T") == (r["defect"] == "tie")
    labels = [r["label"] for r in rows]
    assert labels.count("A") == labels.count("B")  # no answer position is favoured by the data itself


# --- answers cached once, judged by several judges -----------------------------------------------


def _cfg(tmp_path, *overrides) -> Config:
    return load_config(
        ROOT / "configs/tools_pilot.yaml",
        [f"run_dir={tmp_path / 'run'}", f"eval.file={ROOT / 'data/eval/eval_all.jsonl'}", "eval.tool_file=null", *overrides],
    )


def test_answers_are_cached(tmp_path, monkeypatch):
    calls = []

    def fake_generate(cfg, questions, tools=None):
        calls.append(len(questions))
        return {"base": ["b"] * len(questions), "student": ["s"] * len(questions)}

    monkeypatch.setattr(evaluate, "generate_answers", fake_generate)
    cfg = _cfg(tmp_path)
    first = evaluate.run_answers(cfg)
    assert evaluate.run_answers(cfg) == first and len(calls) == 1  # second call reuses the cache
    (cfg.run_dir / "adapter").mkdir()
    (cfg.run_dir / "adapter" / "adapter_model.safetensors").write_text("new")
    evaluate.run_answers(cfg)
    assert len(calls) == 2  # a retrained adapter invalidates it


class FakeJudge:
    """Prefers the student's answer ("s"), wherever it is shown."""

    def __init__(self, cfg, name="judge"):
        self.cfg = cfg

    def chat_json(self, system, prompt, model: type[BaseModel], temperature=None):
        a = prompt.split("Answer A:\n", 1)[1].split("\n", 1)[0]
        v = JudgeVerdict(reasoning="r", verdict="A" if a == "s" else "B")
        return v, v.model_dump_json()

    def map(self, fn, items):
        return [fn(i) for i in items]


def test_tagged_judges_keep_separate_results(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "generate_answers", lambda cfg, q, tools=None: {"base": ["b"] * len(q), "student": ["s"] * len(q)})
    monkeypatch.setattr(evaluate, "Teacher", FakeJudge)
    main = evaluate.run(_cfg(tmp_path, "judge.model=openai/gpt-oss-20b"))
    cross = evaluate.run(_cfg(tmp_path, "judge.model=google/gemma-4-26B-A4B-it", "eval.tag=gemma4"))
    run = tmp_path / "run"
    assert main["student_vs_base_win_rate"] == cross["student_vs_base_win_rate"] == 1.0
    assert json.loads((run / "eval_summary.json").read_text())["judge"] == "openai/gpt-oss-20b"
    assert json.loads((run / "eval_summary_gemma4.json").read_text())["judge"] == "google/gemma-4-26B-A4B-it"
    assert (run / "eval_outputs_gemma4.jsonl").exists()


def test_aggregate_over_seeds(tmp_path):
    for seed, win in (("42", 0.6), ("1", 0.7), ("2", 0.8)):
        d = tmp_path / f"seed_{seed}"
        d.mkdir()
        (d / "eval_summary.json").write_text(json.dumps({"student_vs_base_win_rate": win, "student": {"tool_ast_acc": 0.5}, "judge": "x"}))
    res = aggregate(tmp_path)
    w = res["metrics"]["student_vs_base_win_rate"]
    assert w["n"] == 3 and abs(w["mean"] - 0.7) < 1e-9 and abs(w["std"] - 0.1) < 1e-9
    assert res["metrics"]["student.tool_ast_acc"]["std"] == 0.0 and "judge" not in res["metrics"]


# --- Slurm scripts --------------------------------------------------------------------------------

SCRIPTS = sorted(SLURM.glob("*.sh")) + sorted(SLURM.glob("*.sbatch"))
JOBS = sorted(SLURM.glob("*.sbatch"))


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_scripts_parse(path):
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("path", JOBS, ids=lambda p: p.name)
def test_job_headers(path):
    text = path.read_text()
    for opt in ("--job-name=dk-", "--time=", "--gres=gpu:", "--output=slurm/logs/"):
        assert f"#SBATCH {opt}" in text, opt
    assert "--account" not in text  # comes from SBATCH_ACCOUNT in env.sh
    assert 'source "${REPO:-$WORK/alembic}/slurm/env.sh"' in text and 'source "$REPO/slurm/common.sh"' in text
    if "--array=" in text:
        assert "seed_of_task" in text


def _fields(model: type[BaseModel], path: list[str]) -> bool:
    for part in path[:-1]:
        model = model.model_fields[part].annotation
    return path[-1] in model.model_fields


def test_script_overrides_are_valid_config_keys():
    keys = {k for p in SCRIPTS for k in re.findall(r"--set ([a-z_.]+)=", p.read_text())}
    assert {"run_dir", "train.seed", "judge.base_url", "teacher.base_url", "eval.tag"} <= keys
    for k in keys:
        assert _fields(Config, k.split(".")), k


def test_env_models_match_the_leonardo_config():
    env = dict(re.findall(r'^export (\w+)="([^"$]*)"', (SLURM / "env.sh").read_text(), re.MULTILINE))
    cfg = yaml.safe_load(LEONARDO.read_text())
    assert env["TEACHER_MODEL"] == cfg["teacher"]["model"]
    assert env["STUDENT_MODEL"] == cfg["student"]["model"]
    assert env["JUDGE_MODEL"] == cfg["judge"]["model"]


def test_seed_arrays_match_seeds():
    n = len(re.search(r'SEEDS="\$\{SEEDS:-([^}]*)\}"', (SLURM / "env.sh").read_text()).group(1).split())
    for job in ("train.sbatch", "evaluate.sbatch"):
        assert f"#SBATCH --array=0-{n - 1}" in (SLURM / job).read_text()


def test_smoke_checker_compiles(tmp_path):
    py_compile.compile(str(SLURM / "smoke_teacher.py"), cfile=str(tmp_path / "x.pyc"), doraise=True)
