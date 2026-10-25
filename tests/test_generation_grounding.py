"""Prompt-boundary regressions; fake completions do not establish semantic model quality."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from distillkit import generate
from test_gold_provenance import FakeTeacher, MemoryAppender, isolated_path


class GroundingPromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        seeds, gold = self.root / "seeds", self.root / "gold"
        seeds.mkdir()
        gold.mkdir()
        source = b"TRAIN_SOURCE_73e381: a synthetic prompt-boundary fixture."
        (seeds / "training_fixture.md").write_bytes(source)
        (seeds / "eval_fixture.md").write_text("FORBIDDEN_EVAL_MARKER_68ef03")
        examples = {}
        for name in ("prose", "tool_trace"):
            data = {"messages": [{"role": "user", "content": "GOLD_QUESTION_ONLY"},
                                 {"role": "assistant", "content": "GOLD_STYLE_ONLY_12bd8d"}]}
            raw = json.dumps(data).encode()
            (gold / f"{name}.json").write_bytes(raw)
            examples[f"{name}.json"] = {"sha256": hashlib.sha256(raw).hexdigest(), "sources": [
                {"doc_id": "training_fixture", "split": "train", "sha256": hashlib.sha256(source).hexdigest()}]}
        (gold / "provenance.json").write_text(json.dumps({"schema_version": 1, "examples": examples}))
        self.cfg = NS(run_dir=self.root / "run", seeds=NS(dir=seeds, eval_docs=["eval_fixture"], chunk_chars=1500),
                      generate=NS(gold_dir=gold, questions_per_chunk=1, answers_per_question=1,
                                  personas=["a synthetic test user"], task_types=["script"], tool_fraction=0.0,
                                  tool_mix={"call": 1}, max_tool_rounds=1),
                      teacher=NS(top_logprobs=0), verify=NS(flag_list=None))

    def requests(self, mode):
        MemoryAppender.storage = {}
        self.cfg.generate.tool_fraction = 0 if mode == "prose" else 1
        self.cfg.generate.tool_mix = {"call" if mode == "prose" else mode: 1}
        with isolated_path():
            generate.run(self.cfg)
        return FakeTeacher.instances[0].requests

    def assert_answer_rules(self, request):
        system = request["messages"][0]["content"]
        self.assertIn(generate.PROMPT_VERSION, system)
        self.assertIn("Use the supplied source for general technical facts, commands and syntax", system)
        self.assertIn("not proof of a diagnosis", system)
        self.assertIn("not evidence for the current task", system)
        self.assertNotIn("Only use options and commands you are sure exist", system)
        self.assertNotIn("say what you assume", system)
        self.assertNotIn("FORBIDDEN_EVAL_MARKER_68ef03", json.dumps(request))
        # The gold text is labeled inside the system prompt, never injected as a real tool result.
        self.assertIn("GOLD_STYLE_ONLY_12bd8d", system)
        for message in request["messages"][1:]:
            self.assertNotIn("GOLD_STYLE_ONLY_12bd8d", message.get("content", ""))

    def test_prose_answer_uses_assigned_source_and_grounding_rules(self):
        requests = self.requests("prose")
        answers = [r for r in requests if r["kind"] == "answer"]
        self.assertEqual(len(answers), 1)
        self.assert_answer_rules(answers[0])
        self.assertIn("TRAIN_SOURCE_73e381", answers[0]["messages"][1]["content"])

    def test_all_tool_modes_receive_rules_and_source(self):
        for mode in ("call", "ask", "none"):
            with self.subTest(mode=mode):
                answers = [r for r in self.requests(mode) if r["kind"] == "answer"]
                self.assertEqual(len(answers), 1)
                self.assert_answer_rules(answers[0])
                system = answers[0]["messages"][0]["content"]
                self.assertIn("TRAIN_SOURCE_73e381", system)
                self.assertIn("Recommend a remedy only when the evidence supports its cause", system)
                self.assertIn("safe discovery lookup", system)
                self.assertNotIn("give the fix if something failed", system)

    def test_question_prompt_supports_fragments_without_gold_or_eval(self):
        for mode in ("prose", "call", "ask", "none"):
            with self.subTest(mode=mode):
                questions = [r for r in self.requests(mode) if r["kind"] == "question"]
                self.assertEqual(len(questions), 1)
                request = questions[0]
                prompt = request["messages"][1]["content"]
                self.assertIn(generate.PROMPT_VERSION, request["messages"][0]["content"])
                self.assertIn("TRAIN_SOURCE_73e381", prompt)
                self.assertIn("explicitly scoped code fragment in a language supported by the source", prompt)
                self.assertIn("do not invent error messages or causal premises", prompt)
                # Regression for source-grounded answers to unanswerable stored
                # questions: the student never receives the hidden source table
                # or function that the question author saw. These checks exercise
                # all real question-building paths, not model compliance.
                self.assertIn("student's entire user context", prompt)
                self.assertIn("the source and scenario brief are not", prompt)
                self.assertIn("table values, code being changed, bounds", prompt)
                self.assertIn("an example that is absent from the question", prompt)
                self.assertIn("Do not copy the reference answer into it", prompt)
                self.assertIn("preserve the intentionally missing required user input", prompt)
                self.assertNotIn("GOLD_STYLE_ONLY_12bd8d", json.dumps(request))
                self.assertNotIn("FORBIDDEN_EVAL_MARKER_68ef03", json.dumps(request))

    def test_call_and_ask_questions_receive_actual_argument_contract(self):
        for mode in ("call", "ask"):
            with self.subTest(mode=mode):
                request = next(r for r in self.requests(mode) if r["kind"] == "question")
                prompt = request["messages"][1]["content"]
                for schema in generate.SCHEMAS:
                    self.assertIn(json.dumps(schema, ensure_ascii=False), prompt)
                if mode == "call":
                    self.assertIn("not a request to execute it", prompt)
                else:
                    self.assertIn("Omitting\nan optional filter is insufficient", prompt)

    def test_rules_survive_real_tool_trace_loop(self):
        teacher = FakeTeacher(self.cfg.teacher)
        teacher.call_once = True
        with isolated_path():
            transcript = generate.tool_trace(teacher, self.cfg, "TRAIN_SOURCE_73e381", "Synthetic lookup 24680001")
        self.assertEqual([m["role"] for m in transcript], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(len(teacher.requests), 2)
        for request in teacher.requests:
            self.assert_answer_rules(request)
        self.assertEqual(teacher.requests[0]["messages"][0], teacher.requests[1]["messages"][0])
        self.assertEqual(teacher.requests[1]["messages"][-1]["role"], "tool")
