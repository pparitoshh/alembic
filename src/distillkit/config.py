"""Typed run config: one YAML file drives every stage. Unknown keys are errors, so typos fail fast."""

from pathlib import Path
from typing import Literal

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SeedsCfg(_Section):
    dir: Path
    eval_docs: list[str]  # held out BEFORE any generation
    chunk_chars: int = 1500


class EndpointCfg(_Section):
    """Any OpenAI-compatible /chat/completions endpoint (OpenCode Zen, Ollama, vLLM, llama-server)."""

    base_url: str
    api_key_env: str | None = None  # name of the env var holding the key; None for local servers
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    concurrency: int = 4
    top_logprobs: int = 0  # >0: request and save the top-k logprobs of every answer token (for v2 logit KD)
    # server-specific request fields, e.g. vLLM: {return_tokens_as_token_ids: true,
    # chat_template_kwargs: {enable_thinking: false}}
    extra_body: dict = {}


class GenerateCfg(_Section):
    questions_per_chunk: int
    answers_per_question: int
    personas: list[str]
    task_types: list[str]
    # share of questions that become tool-calling traces, split by mode:
    # call = needs a tool, ask = needs a tool but a required detail is missing, none = tools offered but not needed
    tool_fraction: float = Field(0.0, ge=0, le=1)
    # None preserves the original planner; a list limits tool modes to compatible training sources.
    tool_doc_ids: list[str] | None = None
    tool_mix: dict[Literal["call", "ask", "none"], float] = {"call": 0.6, "ask": 0.2, "none": 0.2}
    max_tool_rounds: int = 3
    gold_dir: Path | None = None  # prose.json / tool_trace.json few-shot anchors (data/gold)


class VerifyCfg(_Section):
    dedup_threshold: float = 0.8  # Jaccard on question word 3-grams
    flag_list: Path | None = None  # None = the list shipped in the package (src/distillkit/data/slurm_flags.txt)
    max_answer_chars: int = 4000


class StudentCfg(_Section):
    model: str
    system_prompt: str


class TrainCfg(_Section):
    method: Literal["lora", "dora"] = "lora"
    load_in_4bit: bool = False  # QLoRA / QDoRA: bitsandbytes NF4 base weights
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05  # DoRA runs faster with 0
    learning_rate: float = 1e-4
    epochs: float = 3
    seed: int = 42
    batch_size: int = 4
    grad_accum: int = 4
    max_length: int = 1024
    fp16: bool = False  # GPUs without bf16 (e.g. RTX 20xx)
    bf16: bool | None = None  # default: the opposite of fp16

    @model_validator(mode="after")
    def _one_precision(self):
        if self.bf16 is None:
            self.bf16 = not self.fp16
        if self.fp16 and self.bf16:
            raise ValueError("set only one of train.fp16 / train.bf16")
        return self

    @property
    def dtype(self) -> str:
        return "float16" if self.fp16 else "bfloat16"


class EvalCfg(_Section):
    file: Path
    tool_file: Path | None = None  # tool-calling slice: scored by toolcheck, not the judge
    calibration_file: Path | None = None  # human-labelled answer pairs for the `calibrate` stage
    max_new_tokens: int = 512
    tag: str = ""  # suffix for judge outputs, e.g. "gemma4" for the cross-check judge on the same answers
    gguf: list[str] = []  # also answer with run_dir/export/model-<Q>.gguf on llama-server (gguf_eval.py)
    gguf_ngl: int = 0  # GPU layers for those runs: 0 = CPU only, as on the laptop


class ExportCfg(_Section):
    llama_cpp: Path = Path("~/tools/llama.cpp")  # checkout (convert_hf_to_gguf.py) + release binaries in bin/
    quants: list[str] = ["Q4_K_M", "Q8_0"]  # first one goes into the Modelfile
    imatrix: bool = True  # importance matrix from the run's own transcripts (domain text)
    imatrix_ctx: int = 512
    num_ctx: int = 8192  # Ollama context: keeps KV cache small for the < 4 GB laptop target
    sampling: dict = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}  # Qwen3-2507 recommendation
    ollama_name: str | None = None  # set to run `ollama create <name>` after export
    merge: bool = True  # False: export the untrained student.model itself (the baseline for the demo)
    bench_threads: int | None = None  # None = physical cores
    bench_depths: list[int] = [0, 2048]  # tokens already in context: empty, and system + tools + RAG chunk
    bench_reps: int = 2  # llama-bench repetitions inside one pass
    bench_runs: int = 3  # independent passes over all models: spread from background load on a real laptop
    bench_ttft: bool = True  # time to first token of a streaming chat on llama-server


class Config(_Section):
    run_dir: Path
    seeds: SeedsCfg
    teacher: EndpointCfg
    judge: EndpointCfg
    generate: GenerateCfg
    verify: VerifyCfg
    student: StudentCfg
    train: TrainCfg = TrainCfg()
    eval: EvalCfg
    export: ExportCfg = ExportCfg()


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """`section.key=value` overrides (value parsed as YAML: 42 -> int, true -> bool, null -> None),
    so one config file can drive many jobs: per-seed run_dir, per-job server port, another judge."""
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        *parents, leaf = key.split(".")
        node = raw
        for p in parents:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {item!r}: {p!r} is not a section")
        node[leaf] = yaml.safe_load(value) if value else ""  # `key=` sets an empty string, not null
    return raw


def load_config(path: str | Path, overrides: list[str] = ()) -> Config:
    # API keys from ./.env in the working directory only (not searched upward from this file), so a
    # run or test elsewhere never picks up the repo's secrets; real env vars take precedence
    load_dotenv(find_dotenv(usecwd=True))
    with open(path) as f:
        cfg = Config.model_validate(apply_overrides(yaml.safe_load(f), list(overrides)))
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    return cfg
