"""Stage 5: merge the adapter -> GGUF -> imatrix -> quantized GGUFs -> Ollama Modelfile (+ `ollama create`).

Outputs in `run_dir/export/`:
    merged/                      HF model, adapter merged into the full-precision base
    model-bf16.gguf              llama.cpp conversion of merged/
    calibration.txt, imatrix.gguf   importance matrix from the run's own training transcripts
    model-<QUANT>.gguf           one per `export.quants`
    Modelfile                    Ollama package for the first quant, with Qwen3's tool-calling template

With `export.ollama_name` set, the package is created in Ollama and smoke-tested: one tool-calling
request through Ollama's chat API must come back as a parsed `job_status` call.

Every step is skipped when its output exists, so a rerun after a failure resumes. Needs a llama.cpp
checkout with binaries in `<llama_cpp>/bin` (the release tarball) or `<llama_cpp>/build/bin`.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import Config

# Ollama's own qwen3 template (`ollama show qwen3:4b --modelfile`), minus the thinking parts: Ollama
# derives its tool-call parser from this template, so the <tool_call> section stays verbatim.
# @GEN_PREFIX@ is what the student saw after "<|im_start|>assistant\n" in training: "" for
# Qwen3-*-2507, an empty think block for hybrid Qwen3 trained with enable_thinking=False.
OLLAMA_TEMPLATE = """{{- if or .System .Tools }}<|im_start|>system
{{ if .System }}{{ .System }}

{{ end }}
{{- if .Tools }}# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end -}}
<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}{{ end }}
{{- if .ToolCalls }}
{{- range .ToolCalls }}
<tool_call>
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
</tool_call>
{{- end }}
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
@GEN_PREFIX@{{ end }}
{{- end }}"""

# quant types that don't use an importance matrix
_NO_IMATRIX = {"F32", "F16", "BF16", "Q8_0"}


def _bin(llama_cpp: Path, name: str) -> str:
    for d in ("bin", "build/bin"):
        if (p := llama_cpp / d / name).exists():
            return str(p)
    raise SystemExit(f"[export] {name} not found under {llama_cpp}/bin or build/bin; set export.llama_cpp")


def _run(cmd: list[str], **kwargs) -> None:
    print("[export] $", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, **kwargs)


def generation_prefix(tokenizer) -> str:
    """Text the chat template puts after '<|im_start|>assistant\\n' at inference (same call as render_prompt)."""
    from .prompting import render_prompt

    prompt = render_prompt(tokenizer, "S", "Q")
    marker = "<|im_start|>assistant\n"
    if marker not in prompt:
        raise SystemExit("[export] the Ollama template supports ChatML (Qwen) students only")
    return prompt.rsplit(marker, 1)[1]


def modelfile(gguf_name: str, system_prompt: str, gen_prefix: str, num_ctx: int, sampling: dict) -> str:
    template = OLLAMA_TEMPLATE.replace("@GEN_PREFIX@", gen_prefix)
    params = {"stop": ["<|im_start|>", "<|im_end|>"], "num_ctx": [num_ctx], **{k: [v] for k, v in sampling.items()}}
    lines = [f"FROM ./{gguf_name}", f'TEMPLATE """{template}"""', f'SYSTEM """{system_prompt}"""']
    lines += [f"PARAMETER {k} {v}" for k, vs in params.items() for v in vs]
    return "\n".join(lines) + "\n"


SMOKE_QUESTION = "Is my job 4718207 still running?"


def ollama_smoke_test(name: str, host: str = "http://localhost:11434") -> bool:
    """One tool request through Ollama's chat API: passes if Ollama parsed a valid, grounded job_status call."""
    import urllib.request

    from .records import openai_tools
    from .toolcheck import call_error, ungrounded_ids
    from .tools import SCHEMAS

    body = {"model": name, "messages": [{"role": "user", "content": SMOKE_QUESTION}], "tools": openai_tools(SCHEMAS), "stream": False, "options": {"temperature": 0}}
    req = urllib.request.Request(f"{host}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    msg = json.load(urllib.request.urlopen(req, timeout=600))["message"]
    calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in msg.get("tool_calls") or []]
    ok = bool(calls) and calls[0]["name"] == "job_status" and call_error(calls[0]) is None and not ungrounded_ids(calls[0], SMOKE_QUESTION)
    print(f"[export] ollama tool-call smoke test: {'PASS' if ok else 'FAIL'} -> {calls or msg.get('content', '')[:200]}")
    return ok


def merge(cfg: Config, out: Path) -> None:
    """Adapter -> full-precision base. (Q)DoRA adapters merge into the unquantized base, as in eval."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = cfg.run_dir / "adapter"
    if not adapter.exists():
        raise SystemExit(f"[export] no adapter at {adapter}; run train first")
    model = AutoModelForCausalLM.from_pretrained(cfg.student.model, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(cfg.student.model).save_pretrained(str(out))  # original chat template


def calibration_text(cfg: Config, tokenizer) -> str:
    """The run's verified transcripts in the student's chat format, tools included: domain text for the imatrix."""
    from .io import read_jsonl
    from .records import training_example

    parts = []
    for row in read_jsonl(cfg.run_dir / "verified.jsonl"):
        ex = training_example(row, cfg.student.system_prompt)
        tools = json.loads(ex["tools"]) if ex["tools"] else None
        parts.append(tokenizer.apply_chat_template(ex["messages"], tools=tools, tokenize=False, **ex["chat_template_kwargs"]))
    return "\n".join(parts)


def run(cfg: Config) -> Path:
    from transformers import AutoTokenizer

    ecfg = cfg.export
    llama_cpp = ecfg.llama_cpp.expanduser()
    out = cfg.run_dir / "export"
    out.mkdir(parents=True, exist_ok=True)
    merged, bf16 = out / "merged", out / "model-bf16.gguf"

    if not ecfg.merge:
        from huggingface_hub import snapshot_download

        merged = Path(snapshot_download(cfg.student.model))  # the untrained base, straight from the HF cache
    elif not (merged / "config.json").exists():
        merge(cfg, merged)
    if not bf16.exists():
        _run(
            [sys.executable, llama_cpp / "convert_hf_to_gguf.py", merged, "--outtype", "bf16", "--outfile", bf16],
            env={**os.environ, "PYTHONPATH": str(llama_cpp / "gguf-py")},  # the checkout's gguf-py matches its script
        )

    tok = AutoTokenizer.from_pretrained(str(merged))
    imatrix = out / "imatrix.gguf"
    needs_imatrix = ecfg.imatrix and any(q.upper() not in _NO_IMATRIX for q in ecfg.quants)
    if needs_imatrix and not imatrix.exists():
        calib = out / "calibration.txt"
        calib.write_text(calibration_text(cfg, tok))
        _run([_bin(llama_cpp, "llama-imatrix"), "-m", bf16, "-f", calib, "-o", imatrix, "-c", ecfg.imatrix_ctx, "--no-ppl"])

    for q in ecfg.quants:
        target = out / f"model-{q}.gguf"
        if target.exists():
            continue
        im = ["--imatrix", imatrix] if needs_imatrix and q.upper() not in _NO_IMATRIX else []
        _run([_bin(llama_cpp, "llama-quantize"), *im, bf16, target, q])

    first = f"model-{ecfg.quants[0]}.gguf"
    mf = out / "Modelfile"
    mf.write_text(modelfile(first, cfg.student.system_prompt, generation_prefix(tok), ecfg.num_ctx, ecfg.sampling))
    for f in sorted(out.glob("model-*.gguf")):
        print(f"[export] {f.name}: {f.stat().st_size / 1e9:.2f} GB")

    if ecfg.ollama_name:
        if shutil.which("ollama"):
            _run(["ollama", "create", ecfg.ollama_name, "-f", "Modelfile"], cwd=out)
            ollama_smoke_test(ecfg.ollama_name)
        else:
            print("[export] ollama not on PATH; skipping `ollama create`")
    print(f"[export] done -> {out}")
    return out
