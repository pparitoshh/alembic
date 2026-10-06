"""Exercise the production generation entry points with a fake teacher and in-memory output.

No model request, generated shell command or cluster-tool implementation is executed.
Config.model_validate tests require the real dependencies; load_config is never called (.env).
"""
import copy
from contextlib import ExitStack, contextmanager, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from distillkit import generate

ROOT = Path(__file__).resolve().parents[1]
MARKER = "HELDOUT_GOLD_SENTINEL_b80c521ca9244bdc"
HELDOUT = "fixture_heldout_origin"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MemoryAppender:
    storage = {}

    def __init__(self, path):
        self.path = path

    def existing(self):
        return copy.deepcopy(self.storage.get(str(self.path), []))

    def append(self, row):
        self.storage.setdefault(str(self.path), []).append(copy.deepcopy(row))


class FakeTeacher:
    instances = []

    def __init__(self, cfg):
        self.model = "synthetic-regression-fixture-not-a-model-run"
        self.requests = []
        self.call_once = False
        self.instances.append(self)

    def chat_json(self, system, user, schema):
        self.requests.append({"kind": "question", "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": user}]})
        return NS(question=f"Synthetic regression question {len(self.requests)}."), "synthetic"

    def complete(self, messages, **kwargs):
        self.requests.append({"kind": "answer", "messages": copy.deepcopy(messages), "kwargs": kwargs})
        if self.call_once and kwargs.get("tools") and messages[-1]["role"] != "tool":
            return NS(content="Synthetic lookup.", logprobs=None, tool_calls=[{
                "type": "function", "function": {"name": "job_status", "arguments": {"job_id": "24680001"}}}])
        return NS(content="Synthetic regression response only.", tool_calls=None, logprobs=None)

    def map(self, fn, items):
        return [fn(item) for item in items]


def fake_execute(name, args):
    if name != "job_status":
        raise AssertionError(f"Unexpected tool request in test: {name}")
    return {"job_id": args["job_id"], "state": "RUNNING", "partition": "fixture_partition"}


@contextmanager
def isolated_path(module=generate):
    FakeTeacher.instances = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(module, "Teacher", FakeTeacher))
        stack.enter_context(patch.object(module, "JsonlAppender", MemoryAppender))
        stack.enter_context(patch.object(module, "execute", side_effect=fake_execute))
        # No tool implementation or process is needed for this prompt-boundary test.
        stack.enter_context(patch.object(module, "set_valid_flags"))
        stack.enter_context(patch.object(module, "load_flags", return_value=set()))
        stack.enter_context(redirect_stdout(io.StringIO()))
        yield


class GoldProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gold-regression-", dir=ROOT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        shutil.copytree(ROOT / "data/seeds", self.base / "seeds")
        shutil.copytree(ROOT / "data/gold", self.base / "gold")
        self.cfg = NS(
            run_dir=self.base / "run",
            seeds=NS(dir=self.base / "seeds", eval_docs=["slurm_job_arrays", "slurm_requeue_signals"], chunk_chars=1500),
            generate=NS(gold_dir=self.base / "gold", questions_per_chunk=1, answers_per_question=1,
                        personas=["a synthetic test user"], task_types=["concept"], tool_fraction=0.0,
                        tool_mix={"call": 0.6, "ask": 0.2, "none": 0.2}, max_tool_rounds=1),
            teacher=NS(top_logprobs=0), verify=NS(flag_list=None))
        MemoryAppender.storage = {}
        FakeTeacher.instances = []

    def sidecar(self):
        return json.loads((self.cfg.generate.gold_dir / "provenance.json").read_text())

    def save_sidecar(self, data):
        (self.cfg.generate.gold_dir / "provenance.json").write_text(json.dumps(data))

    def inject_heldout(self, name):
        seed = self.cfg.seeds.dir / f"{HELDOUT}.md"
        seed.write_text(MARKER)
        if HELDOUT not in self.cfg.seeds.eval_docs:
            self.cfg.seeds.eval_docs.append(HELDOUT)
        path = self.cfg.generate.gold_dir / f"{name}.json"
        path.write_text(json.dumps({"messages": [
            {"role": "user", "content": "Synthetic gold question."},
            {"role": "assistant", "content": MARKER}]}))
        meta = self.sidecar()
        meta["examples"][path.name] = {"sha256": sha(path), "sources": [
            {"doc_id": HELDOUT, "split": "train", "sha256": sha(seed)}]}
        self.save_sidecar(meta)

    def assert_blocked(self, pattern):
        with isolated_path():
            with self.assertRaisesRegex(ValueError, pattern):
                generate.run(self.cfg)
        self.assertEqual(FakeTeacher.instances, [], "guard must precede teacher construction")

    def run_allowed(self, mode="prose"):
        self.cfg.generate.tool_fraction = 0.0 if mode == "prose" else 1.0
        self.cfg.generate.tool_mix = {mode: 1.0} if mode != "prose" else {"call": 1.0}
        MemoryAppender.storage = {}
        with isolated_path():
            generate.run(self.cfg)
        return FakeTeacher.instances[0].requests

    def test_heldout_prose_is_blocked_before_any_teacher_request(self):
        self.inject_heldout("prose")
        self.assert_blocked(r"prose\.json.*fixture_heldout_origin.*resolves to eval")
        self.assertEqual(MemoryAppender.storage, {})

    def test_heldout_tool_example_blocks_even_a_prose_only_plan(self):
        self.inject_heldout("tool_trace")
        self.assert_blocked(r"tool_trace\.json.*fixture_heldout_origin.*resolves to eval")

    def test_either_gold_is_guarded_in_every_mode(self):
        clean_gold = {p.name: p.read_bytes() for p in self.cfg.generate.gold_dir.iterdir()}
        for name in ("prose", "tool_trace"):
            for mode in ("prose", "call", "ask", "none"):
                with self.subTest(example=name, mode=mode):
                    for filename, content in clean_gold.items():
                        (self.cfg.generate.gold_dir / filename).write_bytes(content)
                    self.inject_heldout(name)
                    self.cfg.generate.tool_fraction = 0 if mode == "prose" else 1
                    self.cfg.generate.tool_mix = {"call" if mode == "prose" else mode: 1}
                    self.assert_blocked(r"resolves to eval")

    def test_unknown_source_id_fails_clearly(self):
        meta = self.sidecar()
        meta["examples"]["prose.json"]["sources"][0]["doc_id"] = "unresolved_fixture"
        self.save_sidecar(meta)
        self.assert_blocked(r"prose\.json.*unresolved_fixture.*unresolved split")

    def test_unknown_declared_split_fails_clearly(self):
        meta = self.sidecar()
        meta["examples"]["prose.json"]["sources"][0]["split"] = "unknown"
        self.save_sidecar(meta)
        self.assert_blocked(r"slurm_sbatch_basics.*unresolved declared split 'unknown'")

    def test_missing_example_provenance_fails_clearly(self):
        meta = self.sidecar()
        del meta["examples"]["prose.json"]
        self.save_sidecar(meta)
        self.assert_blocked(r"prose\.json.*unknown source split provenance")

    def test_missing_sidecar_fails_clearly(self):
        (self.cfg.generate.gold_dir / "provenance.json").unlink()  # this test's temporary copy only
        self.assert_blocked(r"prose\.json.*cannot read example/provenance")

    def test_missing_configured_example_does_not_silently_disable_gold(self):
        (self.cfg.generate.gold_dir / "tool_trace.json").unlink()
        self.assert_blocked(r"tool_trace\.json.*cannot read example/provenance")

    def test_changed_gold_requires_new_review_binding(self):
        path = self.cfg.generate.gold_dir / "prose.json"
        path.write_text(path.read_text() + "\n")
        self.assert_blocked(r"prose\.json.*example hash differs")

    def test_changed_seed_requires_new_review_binding(self):
        path = self.cfg.seeds.dir / "slurm_sbatch_basics.md"
        path.write_text(path.read_text() + "\nSynthetic revision.\n")
        self.assert_blocked(r"slurm_sbatch_basics.*differs from reviewed source hash")

    def test_effective_eval_split_overrides_sidecar_train_declaration(self):
        self.cfg.seeds.eval_docs.append("slurm_sbatch_basics")
        self.assert_blocked(r"slurm_sbatch_basics.*resolves to eval")

    def test_every_declared_source_is_checked(self):
        meta = self.sidecar()
        meta["examples"]["prose.json"]["sources"].append({
            "doc_id": "slurm_requeue_signals", "split": "train",
            "sha256": sha(self.cfg.seeds.dir / "slurm_requeue_signals.md")})
        self.save_sidecar(meta)
        self.assert_blocked(r"slurm_requeue_signals.*resolves to eval")

    def test_allowed_replacement_reaches_prose_system_prompt(self):
        text = json.loads((self.cfg.generate.gold_dir / "prose.json").read_text())["messages"][-1]["content"]
        requests = self.run_allowed()
        questions = [r for r in requests if r["kind"] == "question"]
        answers = [r for r in requests if r["kind"] == "answer"]
        self.assertTrue(questions and answers)
        for request in answers:
            self.assertEqual(request["messages"][0]["role"], "system")
            self.assertIn(text, request["messages"][0]["content"])
            self.assertNotIn("<example>", request["messages"][1]["content"])
        for request in questions:
            self.assertNotIn("<example>", json.dumps(request))

    def test_tool_anchor_reaches_call_ask_and_none_system_prompts(self):
        text = json.loads((self.cfg.generate.gold_dir / "tool_trace.json").read_text())["messages"][-1]["content"]
        for mode in ("call", "ask", "none"):
            with self.subTest(mode=mode):
                requests = self.run_allowed(mode)
                answers = [r for r in requests if r["kind"] == "answer"]
                self.assertTrue(answers)
                for request in answers:
                    self.assertIn(text, request["messages"][0]["content"])
                    self.assertIn("[assistant calls] job_accounting", request["messages"][0]["content"])
                    self.assertTrue(request["kwargs"]["tools"])
                self.assertTrue(all("<example>" not in json.dumps(r) for r in requests if r["kind"] == "question"))

    def test_direct_tool_trace_checks_split_before_request(self):
        self.inject_heldout("tool_trace")
        teacher = FakeTeacher(self.cfg.teacher)
        with self.assertRaisesRegex(ValueError, "resolves to eval"):
            generate.tool_trace(teacher, self.cfg, "Synthetic training chunk", "Synthetic question")
        self.assertEqual(teacher.requests, [])

    def test_direct_tool_trace_rejects_arbitrary_unvalidated_gold(self):
        teacher = FakeTeacher(self.cfg.teacher)
        for enabled in (True, False):
            with self.subTest(gold_enabled=enabled):
                if not enabled:
                    self.cfg.generate.gold_dir = None
                with self.assertRaisesRegex(ValueError, "provenance-checked"):
                    generate.tool_trace(teacher, self.cfg, "Synthetic chunk", "Synthetic question", MARKER)
                self.assertEqual(teacher.requests, [])

    def test_tool_round_trip_keeps_validated_anchor(self):
        teacher = FakeTeacher(self.cfg.teacher)
        teacher.call_once = True
        with isolated_path():
            convo = generate.tool_trace(teacher, self.cfg, "Synthetic chunk", "Synthetic question")
        self.assertEqual([m["role"] for m in convo], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(len(teacher.requests), 2)
        self.assertEqual(teacher.requests[0]["messages"][0], teacher.requests[1]["messages"][0])
        self.assertIn("<example>", teacher.requests[0]["messages"][0]["content"])

    def test_eval_docs_and_unique_marker_remain_excluded_from_question_jobs(self):
        (self.cfg.seeds.dir / f"{HELDOUT}.md").write_text(MARKER)
        self.cfg.seeds.eval_docs.append(HELDOUT)
        jobs = generate.question_jobs(self.cfg)
        self.assertTrue(jobs)
        self.assertFalse(set(self.cfg.seeds.eval_docs) & {j["doc_id"] for j in jobs})
        self.assertNotIn(MARKER, json.dumps(jobs))

    def test_resumed_questions_cannot_bypass_gold_preflight(self):
        jobs = generate.question_jobs(self.cfg)
        MemoryAppender.storage[str(self.cfg.run_dir / "questions.jsonl")] = [
            {**j, "question": f"Synthetic cached question {n}"} for n, j in enumerate(jobs)]
        before = copy.deepcopy(MemoryAppender.storage)
        self.inject_heldout("prose")
        self.assert_blocked("resolves to eval")
        self.assertEqual(MemoryAppender.storage, before)

    def test_explicitly_disabled_gold_remains_supported(self):
        self.cfg.generate.gold_dir = None
        requests = self.run_allowed()
        self.assertTrue(requests)
        self.assertTrue(all("<example>" not in json.dumps(r) for r in requests))


class ConfiguredGoldTests(unittest.TestCase):
    """Requires real PyYAML/Pydantic/SDK imports; no dotenv-loading entry point."""
    def config(self, name):
        import yaml
        from distillkit.config import Config
        raw = yaml.safe_load((ROOT / "configs" / name).read_text())
        raw["seeds"]["dir"] = str(ROOT / raw["seeds"]["dir"])
        if raw["generate"].get("gold_dir"):
            raw["generate"]["gold_dir"] = str(ROOT / raw["generate"]["gold_dir"])
        raw["run_dir"] = str(ROOT / "tests/never_written_memory_run")
        return Config.model_validate(raw)

    def test_actual_configured_paths_build_expected_prompts(self):
        for name in ("qwen3_4b_qdora.yaml", "tools_pilot.yaml", "toy_qdora.yaml"):
            with self.subTest(config=name):
                cfg = self.config(name)
                cfg.generate.questions_per_chunk = 1
                cfg.generate.answers_per_question = 1
                MemoryAppender.storage = {}
                with isolated_path():
                    generate.run(cfg)
                answers = [r for r in FakeTeacher.instances[0].requests if r["kind"] == "answer"]
                self.assertTrue(answers)
                self.assertTrue(all(("<example>" in r["messages"][0]["content"]) == bool(cfg.generate.gold_dir) for r in answers))

    def test_actual_configs_preserve_eval_docs_exclusion(self):
        for name in ("qwen3_4b_qdora.yaml", "tools_pilot.yaml", "toy_qdora.yaml"):
            with self.subTest(config=name):
                cfg = self.config(name)
                self.assertEqual(set(cfg.seeds.eval_docs), {"slurm_job_arrays", "slurm_requeue_signals"})
                jobs = generate.question_jobs(cfg)
                self.assertFalse(set(cfg.seeds.eval_docs) & {j["doc_id"] for j in jobs})


if __name__ == "__main__":
    unittest.main()
