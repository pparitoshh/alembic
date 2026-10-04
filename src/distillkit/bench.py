"""Stage `bench`: laptop speed and memory of every exported GGUF (GOAL.md G3, G6).

For each `run_dir/export/model-*.gguf`, on the CPU only (the laptop target):
- llama-bench: prompt processing (pp512) and generation (tg128) tokens/s, with the context empty and
  with `export.bench_depths` tokens already in it (system prompt + tool schemas + RAG chunk);
- peak resident memory of a short llama-completion run at the deployed context (`export.num_ctx`),
  so the KV cache is included: this is the number for the "< 4 GB RAM" target;
- time to first token of a streaming chat on llama-server (`export.bench_ttft`): a first question
  with the system prompt and the tool schemas (cold), then a follow-up question that reuses the
  processed prompt, as Ollama does within a chat. GOAL.md: <= 3 s for the follow-up.

Writes run_dir/export/bench.json and bench.md. Machine-dependent by design: run it on the demo laptop.
"""

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from .config import Config
from .export import _bin
from .machine import collect

PROMPT = "Write an sbatch script that runs 100 array tasks, at most 10 at a time, each with 1 CPU and 2 GB."
FOLLOW_UP = "Now give each task 4 CPUs. How do I see which tasks failed?"

# runs one command in a fresh interpreter and prints the peak RSS of that command alone (KiB on Linux)
_PEAK_RSS = (
    "import resource, subprocess, sys\n"
    "r = subprocess.run(sys.argv[1:], capture_output=True, text=True)\n"
    "print(r.returncode, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)\n"
    "sys.stderr.write(r.stderr[-2000:] if r.returncode else '')\n"
)


def physical_cores(cpuinfo: str | None = None) -> int:
    """Physical cores (llama.cpp generates fastest with one thread per core, not per hyperthread);
    logical CPUs if /proc/cpuinfo has no core ids."""
    if cpuinfo is None:
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
        except OSError:
            cpuinfo = ""
    cores = set()
    for block in cpuinfo.split("\n\n"):
        # lines look like "core id\t\t: 3": keys are padded with tabs
        fields = {k.strip(): v.strip() for k, _, v in (line.partition(":") for line in block.splitlines())}
        if "core id" in fields:
            cores.add((fields.get("physical id"), fields["core id"]))
    return len(cores) or os.cpu_count() or 1


def parse_llama_bench(rows: list[dict]) -> dict:
    """llama-bench JSON -> {"pp512@0": tok/s, "tg128@2048": tok/s, ...} plus build/cpu info."""
    out = {"results": {}}
    for r in rows:
        test = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
        out["results"][f"{test}@{r.get('n_depth', 0)}"] = {"tok_s": round(r["avg_ts"], 2), "stddev": round(r["stddev_ts"], 2)}
        out |= {"build": r.get("build_commit"), "cpu": r.get("cpu_info"), "model_type": r.get("model_type")}
    return out


def llama_bench(llama_cpp: Path, model: Path, threads: int, depths: list[int], reps: int) -> dict:
    cmd = [
        _bin(llama_cpp, "llama-bench"), "-m", str(model), "-p", "512", "-n", "128",
        "-d", ",".join(map(str, depths)), "-t", str(threads), "-ngl", "0", "-r", str(reps), "-o", "json",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return parse_llama_bench(json.loads(r.stdout))


def peak_rss_mb(llama_cpp: Path, model: Path, threads: int, num_ctx: int) -> float:
    cmd = [
        _bin(llama_cpp, "llama-completion"), "-m", str(model), "-p", PROMPT, "-n", "64", "-c", str(num_ctx),
        "-t", str(threads), "-ngl", "0", "-no-cnv", "--no-display-prompt", "--no-warmup",
    ]
    r = subprocess.run([sys.executable, "-c", _PEAK_RSS, *cmd], capture_output=True, text=True, check=True)
    code, kib = r.stdout.split()
    if code != "0":
        raise RuntimeError(f"llama-completion failed on {model.name}: {r.stderr}")
    return round(int(kib) / 1024, 1)


def chat_prompts(tokenizer, system: str, answer: str = "") -> tuple[str, str]:
    """(first turn, follow-up turn) as rendered by the model's chat template, tool schemas included.
    The follow-up starts with the first turn plus `answer`, so llama-server reuses that prefix."""
    from .records import openai_tools
    from .tools import SCHEMAS

    first = [{"role": "system", "content": system}, {"role": "user", "content": PROMPT}]
    follow = first + [{"role": "assistant", "content": answer}, {"role": "user", "content": FOLLOW_UP}]
    render = lambda m: tokenizer.apply_chat_template(m, tools=openai_tools(SCHEMAS), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return render(first), render(follow)


def ttft_s(timings: dict) -> float:
    """Time to first token from llama-server timings: prompt processing + one generated token."""
    per_token = timings["predicted_ms"] / max(1, timings["predicted_n"])
    return round((timings["prompt_ms"] + per_token) / 1000, 2)


def chat_latency(llama_cpp: Path, model: Path, threads: int, num_ctx: int, tokenizer, system: str, log: Path) -> dict:
    from .gguf_eval import llama_server, post_completion, server_cmd

    with llama_server(lambda port: server_cmd(llama_cpp, model, port, num_ctx, threads, 0), log) as url:
        first, _ = chat_prompts(tokenizer, system)
        r1 = post_completion(url, first, 128)
        _, follow = chat_prompts(tokenizer, system, r1["content"])
        r2 = post_completion(url, follow, 128)
    t1, t2 = r1["timings"], r2["timings"]
    return {
        "first_prompt_tokens": t1["prompt_n"], "ttft_first_s": ttft_s(t1),
        "followup_new_tokens": t2["prompt_n"], "ttft_followup_s": ttft_s(t2),
        "stream_tok_s": round(t2["predicted_per_second"], 2),
    }


def system_load() -> dict:
    """What else the machine was doing: a real laptop is never idle, so results record it."""
    load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    try:
        ps = subprocess.run(["ps", "-eo", "pcpu,comm", "--sort=-pcpu"], capture_output=True, text=True, timeout=10).stdout
        busiest = [" ".join(l.split()) for l in ps.splitlines()[1:6]]
    except (OSError, subprocess.SubprocessError):
        busiest = []
    return {"loadavg_1m": round(load[0], 2), "busiest": busiest}


def summarize(passes: list[dict]) -> list[dict]:
    """Per model over all passes: mean ± std of each test's tok/s, mean and max peak RAM."""
    names = list(dict.fromkeys(m["name"] for p in passes for m in p["models"]))
    out = []
    for name in names:
        runs = [m for p in passes for m in p["models"] if m["name"] == name]
        tests = runs[0]["results"].keys()
        results = {}
        for t in tests:
            vals = [r["results"][t]["tok_s"] for r in runs]
            results[t] = {"tok_s": round(statistics.fmean(vals), 2), "std": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0, "runs": vals}
        rss = [r["peak_rss_mb"] for r in runs]
        m = {"name": name, "file_gb": runs[0]["file_gb"], "peak_rss_mb": max(rss), "peak_rss_mb_mean": round(statistics.fmean(rss), 1), "results": results}
        chats = [r["chat"] for r in runs if r.get("chat")]
        if chats:
            m["chat"] = {k: round(statistics.fmean(c[k] for c in chats), 2) for k in chats[0]}
            m["chat"]["ttft_followup_s_max"] = max(c["ttft_followup_s"] for c in chats)
        out.append(m)
    return out


def markdown(report: dict) -> str:
    depths = report["depths"]
    head = ["model", "file GB", "peak RAM GB (max)", "pp512 tok/s", *[f"tg128 tok/s @{d}" for d in depths]]
    loads = ", ".join(str(p["load"]["loadavg_1m"]) for p in report.get("passes", []))
    lines = [
        f"CPU: {report['cpu']} · {report['threads']} threads · llama.cpp {report['build']} · context {report['num_ctx']}",
        f"{len(report.get('passes', [])) or 1} pass(es), mean ± std; machine in normal use (1-min load average per pass: {loads or 'n/a'})",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    fmt = lambda r: f"{r['tok_s']:.1f} ± {r.get('std', r.get('stddev', 0.0)):.1f}"
    for m in report["models"]:
        r = m["results"]
        row = [m["name"], f"{m['file_gb']:.2f}", f"{m['peak_rss_mb'] / 1024:.2f}", fmt(r["pp512@0"])]
        row += [fmt(r[f"tg128@{d}"]) for d in depths]
        lines.append("| " + " | ".join(row) + " |")
    chats = [m for m in report["models"] if m.get("chat")]
    if chats:
        lines += [
            "",
            "Streaming chat on llama-server (system prompt + tool schemas; mean over passes):",
            "",
            "| model | first question: prompt tokens | TTFT s | follow-up: new tokens | TTFT s (max) | streaming tok/s |",
            "|---|---|---|---|---|---|",
        ]
        for m in chats:
            c = m["chat"]
            lines.append(f"| {m['name']} | {c['first_prompt_tokens']:.0f} | {c['ttft_first_s']:.1f} | {c['followup_new_tokens']:.0f} | "
                         f"{c['ttft_followup_s']:.1f} ({c['ttft_followup_s_max']:.1f}) | {c['stream_tok_s']:.1f} |")
    return "\n".join(lines) + "\n"


def run(cfg: Config) -> dict:
    ecfg = cfg.export
    llama_cpp = ecfg.llama_cpp.expanduser()
    out = cfg.run_dir / "export"
    # deployable quant levels only (bf16 is the accuracy reference, far over the RAM target)
    models = [out / f"model-{q}.gguf" for q in ecfg.quants if (out / f"model-{q}.gguf").exists()]
    if not models:
        raise SystemExit(f"[bench] no model-<quant>.gguf for {ecfg.quants} in {out}; run export first")
    models.sort(key=lambda p: p.stat().st_size)
    threads = ecfg.bench_threads or physical_cores()
    tokenizer = None
    if ecfg.bench_ttft:
        from .gguf_eval import _tokenizer

        tokenizer = _tokenizer(cfg.student.model)
    report = {"machine": collect(llama_cpp), "threads": threads, "num_ctx": ecfg.num_ctx, "depths": ecfg.bench_depths, "passes": [], "models": []}
    # interleaved passes: a burst of background load lands on every model, not on one
    for i in range(ecfg.bench_runs):
        p = {"pass": i + 1, "load": system_load(), "models": []}
        report["passes"].append(p)
        for m in models:
            print(f"[bench] pass {i + 1}/{ecfg.bench_runs} {m.name} ({threads} threads, depths {ecfg.bench_depths})", flush=True)
            speed = llama_bench(llama_cpp, m, threads, ecfg.bench_depths, ecfg.bench_reps)
            rss = peak_rss_mb(llama_cpp, m, threads, ecfg.num_ctx)
            report |= {"build": speed["build"], "cpu": speed["cpu"]}
            name = m.stem.removeprefix("model-")
            chat = chat_latency(llama_cpp, m, threads, ecfg.num_ctx, tokenizer, cfg.student.system_prompt, out / f"llama-server_{name}.log") if tokenizer else None
            p["models"].append({"name": name, "file_gb": m.stat().st_size / 1e9, "peak_rss_mb": rss, "chat": chat, **speed})
            report["models"] = summarize(report["passes"])
            (out / "bench.json").write_text(json.dumps(report, indent=2))  # partial results survive an interruption
    (out / "bench.md").write_text(markdown(report))
    print(markdown(report))
    return report
