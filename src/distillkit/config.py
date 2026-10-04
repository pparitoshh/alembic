"""Typed run config: one YAML file drives every stage. Unknown keys are errors, so typos fail fast."""

from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, model_validator


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


class VerifyCfg(_Section):
    dedup_threshold: float = 0.8  # Jaccard on question word 3-grams
    flag_list: Path
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
    max_new_tokens: int = 512


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


def load_config(path: str | Path) -> Config:
    load_dotenv()  # picks up API keys from ./.env; real env vars take precedence
    with open(path) as f:
        cfg = Config.model_validate(yaml.safe_load(f))
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    return cfg
