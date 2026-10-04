"""Answers from the exported GGUF files: the eval measures the artifact users actually run.

With `eval.gguf: [Q4_K_M, ...]`, the `answer`/`evaluate` stages also answer every eval question with
`run_dir/export/model-<Q>.gguf` on a local llama-server, CPU only by default (`eval.gguf_ngl`). The
prompt is the same text the Hugging Face eval renders (same tokenizer chat template, system prompt and
tool schemas) and decoding is greedy, so quantization is the only difference.

Answers are cached per quant in run_dir/eval_answers_gguf_<Q>.json, keyed by the questions and the
GGUF file's mtime (a re-export invalidates them).
"""

import json
import socket
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from .bench import physical_cores
from .config import Config
from .export import _bin


def server_cmd(llama_cpp: Path, model: Path, port: int, num_ctx: int, threads: int, ngl: int) -> list[str]:
    # one slot: requests run one at a time with the full context, as on the laptop
    return [
        _bin(llama_cpp, "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "-c", str(num_ctx), "-np", "1", "-t", str(threads), "-ngl", str(ngl), "--no-webui",
    ]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def llama_server(cmd_for_port, log: Path, timeout: float = 600):
    """Start llama-server, wait for /health, yield its base URL, stop it."""
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd_for_port(port), stdout=f, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + timeout
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {proc.returncode}; see {log}")
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:  # refused while starting, 503 while loading the model
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"llama-server not ready after {timeout:.0f} s; see {log}")
            time.sleep(1)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()


def post_completion(url: str, prompt: str, max_tokens: int) -> dict:
    """Greedy completion of an already rendered chat prompt (raw /completion, no server-side template).
    The reply has "content" and llama-server's "timings"; the processed prompt stays cached in the slot."""
    body = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0.0, "top_k": 1, "cache_prompt": True}
    req = urllib.request.Request(f"{url}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def complete(url: str, prompt: str, max_tokens: int) -> str:
    return post_completion(url, prompt, max_tokens)["content"].strip()


def _tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def answers(cfg: Config, quant: str, questions: list[str], tools: list[list[dict] | None]) -> list[str]:
    from .prompting import render_prompt

    gguf = cfg.run_dir / "export" / f"model-{quant}.gguf"
    if not gguf.exists():
        raise SystemExit(f"[answer] {gguf} not found; run export with {quant} in export.quants")
    cache = cfg.run_dir / f"eval_answers_gguf_{quant}.json"
    stamp = gguf.stat().st_mtime
    if cache.exists():
        c = json.loads(cache.read_text())
        if c["questions"] == questions and c["gguf_mtime"] == stamp:
            print(f"[answer] reusing {cache}")
            return c["answers"]

    tok = _tokenizer(cfg.student.model)
    prompts = [render_prompt(tok, cfg.student.system_prompt, q, t) for q, t in zip(questions, tools)]
    ecfg = cfg.export
    llama_cpp = ecfg.llama_cpp.expanduser()
    threads = ecfg.bench_threads or physical_cores()
    cmd = lambda port: server_cmd(llama_cpp, gguf, port, ecfg.num_ctx, threads, cfg.eval.gguf_ngl)
    outs = []
    t0 = time.monotonic()
    with llama_server(cmd, cfg.run_dir / f"llama-server_{quant}.log") as url:
        for i, p in enumerate(prompts):
            outs.append(complete(url, p, cfg.eval.max_new_tokens))
            print(f"[answer] {quant} {i + 1}/{len(prompts)} ({time.monotonic() - t0:.0f} s)", flush=True)
    cache.write_text(json.dumps({"questions": questions, "gguf_mtime": stamp, "answers": outs}, ensure_ascii=False))
    return outs
