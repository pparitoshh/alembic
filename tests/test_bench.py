from pathlib import Path

import pytest

from distillkit.bench import markdown, parse_llama_bench, physical_cores, summarize, system_load
from distillkit.config import ExportCfg

# two physical cores with hyperthreading, keys padded with tabs as in the real /proc/cpuinfo
CPUINFO = "\n\n".join(
    f"processor\t: {i}\nmodel name\t: Intel(R) Core(TM) i7-9750H\nphysical id\t: 0\ncore id\t\t: {i % 2}\nsiblings\t: 4"
    for i in range(4)
)


def test_physical_cores_ignores_hyperthreads():
    assert physical_cores(CPUINFO) == 2
    assert physical_cores("") >= 1  # no core ids (e.g. some VMs): falls back to logical CPUs


def test_parse_llama_bench():
    rows = [
        {"build_commit": "46847e615", "cpu_info": "i7", "model_type": "qwen3 4B Q4_K - Medium", "n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 61.234, "stddev_ts": 1.2},
        {"build_commit": "46847e615", "cpu_info": "i7", "model_type": "qwen3 4B Q4_K - Medium", "n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ts": 12.345, "stddev_ts": 0.11},
        {"build_commit": "46847e615", "cpu_info": "i7", "model_type": "qwen3 4B Q4_K - Medium", "n_prompt": 0, "n_gen": 128, "n_depth": 2048, "avg_ts": 9.87, "stddev_ts": 0.05},
    ]
    p = parse_llama_bench(rows)
    assert p["results"] == {
        "pp512@0": {"tok_s": 61.23, "stddev": 1.2},
        "tg128@0": {"tok_s": 12.35, "stddev": 0.11},
        "tg128@2048": {"tok_s": 9.87, "stddev": 0.05},
    }
    assert (p["build"], p["cpu"]) == ("46847e615", "i7")


def test_markdown_table():
    results = {"pp512@0": {"tok_s": 61.2}, "tg128@0": {"tok_s": 12.3}, "tg128@2048": {"tok_s": 9.9}}
    report = {
        "cpu": "i7", "threads": 6, "build": "b", "num_ctx": 8192, "depths": [0, 2048],
        "models": [{"name": "Q4_K_M", "file_gb": 2.5, "peak_rss_mb": 3584.0, "results": results}],
    }
    md = markdown(report)
    assert "| Q4_K_M | 2.50 | 3.50 | 61.2 ± 0.0 | 12.3 ± 0.0 | 9.9 ± 0.0 |" in md
    assert "6 threads" in md and "context 8192" in md


def test_bench_defaults():
    e = ExportCfg()
    assert e.merge and e.bench_threads is None and e.bench_depths == [0, 2048] and e.bench_runs >= 3


def test_bench_needs_exported_models(tmp_path):
    from distillkit import bench
    from distillkit.config import load_config

    cfg = load_config(Path(__file__).parent.parent / "configs/tools_pilot.yaml", [f"run_dir={tmp_path / 'run'}"])
    with pytest.raises(SystemExit, match="no model-"):
        bench.run(cfg)


def test_summarize_over_passes():
    def model(name, tg, rss):
        return {"name": name, "file_gb": 2.5, "peak_rss_mb": rss, "results": {"tg128@0": {"tok_s": tg}}}

    passes = [{"models": [model("Q4_K_M", 10.0, 3000.0), model("Q8_0", 6.0, 4500.0)]},
              {"models": [model("Q4_K_M", 12.0, 3100.0), model("Q8_0", 6.0, 4400.0)]}]
    s = {m["name"]: m for m in summarize(passes)}
    q4 = s["Q4_K_M"]
    assert q4["results"]["tg128@0"] == {"tok_s": 11.0, "std": 1.41, "runs": [10.0, 12.0]}
    assert q4["peak_rss_mb"] == 3100.0 and q4["peak_rss_mb_mean"] == 3050.0  # worst case is what must fit
    assert s["Q8_0"]["results"]["tg128@0"]["std"] == 0.0
    assert list(s) == ["Q4_K_M", "Q8_0"]


def test_system_load_is_recorded():
    load = system_load()
    assert load["loadavg_1m"] >= 0 and isinstance(load["busiest"], list)


def test_results_saved_after_each_model(tmp_path, monkeypatch):
    """An interrupted bench keeps every model measured so far, not just whole passes."""
    import json

    from distillkit import bench
    from distillkit.config import load_config

    cfg = load_config(Path(__file__).parent.parent / "configs/tools_pilot.yaml",
                      [f"run_dir={tmp_path / 'run'}", "export.quants=[Q4_K_M,Q8_0]", "export.bench_runs=1", "export.bench_ttft=false"])
    out = cfg.run_dir / "export"
    out.mkdir(parents=True)
    (out / "model-Q4_K_M.gguf").write_bytes(b"x")
    (out / "model-Q8_0.gguf").write_bytes(b"xx")
    calls = []

    def fake_bench(llama_cpp, model, threads, depths, reps):
        calls.append(model.name)
        if len(calls) > 1:
            raise KeyboardInterrupt
        return {"results": {"pp512@0": {"tok_s": 50.0, "stddev": 1.0}}, "build": "b", "cpu": "c"}

    monkeypatch.setattr(bench, "collect", lambda llama_cpp: {})
    monkeypatch.setattr(bench, "llama_bench", fake_bench)
    monkeypatch.setattr(bench, "peak_rss_mb", lambda *a: 3000.0)
    with pytest.raises(KeyboardInterrupt):
        bench.run(cfg)
    saved = json.loads((out / "bench.json").read_text())
    assert [m["name"] for m in saved["models"]] == ["Q4_K_M"]


class ChatTokenizer:
    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, **kw):
        text = f"<tools:{len(tools or [])}>" + "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


def test_follow_up_extends_the_first_turn():
    """The follow-up prompt must start with the cached first turn, or nothing is reused."""
    from distillkit.bench import chat_prompts

    first, _ = chat_prompts(ChatTokenizer(), "sys")
    _, follow = chat_prompts(ChatTokenizer(), "sys", "ANSWER")
    assert first.startswith("<tools:8><system>sys</system>")
    assert follow.startswith(first.removesuffix("<assistant>") + "<assistant>ANSWER")


def test_chat_latency_from_server_timings(monkeypatch, tmp_path):
    from contextlib import contextmanager

    from distillkit import bench, gguf_eval

    replies = iter([
        {"content": "A1", "timings": {"prompt_n": 1500, "prompt_ms": 30000.0, "predicted_n": 128, "predicted_ms": 12800.0, "predicted_per_second": 10.0}},
        {"content": "A2", "timings": {"prompt_n": 40, "prompt_ms": 1000.0, "predicted_n": 128, "predicted_ms": 16000.0, "predicted_per_second": 8.0}},
    ])
    sent = []

    @contextmanager
    def server(cmd_for_port, log, timeout=600):
        yield "http://fake"

    monkeypatch.setattr(gguf_eval, "llama_server", server)
    monkeypatch.setattr(gguf_eval, "post_completion", lambda url, prompt, n: sent.append(prompt) or next(replies))
    c = bench.chat_latency(tmp_path, tmp_path / "m.gguf", 6, 8192, ChatTokenizer(), "sys", tmp_path / "log")
    assert c == {"first_prompt_tokens": 1500, "ttft_first_s": 30.1, "followup_new_tokens": 40, "ttft_followup_s": 1.12, "stream_tok_s": 8.0}
    assert "<assistant>A1</assistant>" in sent[1]  # the follow-up carries the model's own first answer


def test_chat_summary_and_table():
    def model(ttft):
        chat = {"first_prompt_tokens": 1500, "ttft_first_s": 30.0, "followup_new_tokens": 40, "ttft_followup_s": ttft, "stream_tok_s": 8.0}
        return {"name": "Q4_K_M", "file_gb": 2.5, "peak_rss_mb": 3500.0, "chat": chat,
                "results": {"pp512@0": {"tok_s": 50.0}, "tg128@0": {"tok_s": 10.0}}}

    models = summarize([{"models": [model(1.0)]}, {"models": [model(2.0)]}])
    assert models[0]["chat"]["ttft_followup_s"] == 1.5 and models[0]["chat"]["ttft_followup_s_max"] == 2.0
    md = markdown({"cpu": "i7", "threads": 6, "build": "b", "num_ctx": 8192, "depths": [0], "models": models})
    assert "| Q4_K_M | 1500 | 30.0 | 40 | 1.5 (2.0) | 8.0 |" in md
