"""Teacher = any OpenAI-compatible chat endpoint (OpenCode Zen, Ollama, or `vllm serve` on the cluster)."""

import os
import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from pydantic import BaseModel

from .schemas import parse_json

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class Teacher:
    def __init__(self, cfg: dict, section: str = "teacher"):
        t = cfg[section]
        # keys come from the environment, never from the config file
        api_key = os.environ.get(t["api_key_env"]) if "api_key_env" in t else "none"
        if not api_key:
            raise SystemExit(f"[{section}] set the {t['api_key_env']} environment variable")
        self.client = OpenAI(base_url=t["base_url"], api_key=api_key)
        self.model = t["model"]
        self.temperature = t.get("temperature", 0.7)
        self.max_tokens = t.get("max_tokens", 1024)
        self.concurrency = t.get("concurrency", 4)

    def chat(self, system: str, user: str, temperature: float | None = None, response_format: dict | None = None) -> str:
        # "/no_think" disables Qwen3 thinking; other models ignore it.
        extra = {"response_format": response_format} if response_format else {}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system + " /no_think"},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            **extra,
        )
        return _THINK.sub("", resp.choices[0].message.content or "").strip()

    def chat_json[M: BaseModel](
        self, system: str, user: str, model: type[M], temperature: float | None = None, retries: int = 1
    ) -> tuple[M | None, str]:
        """Ask for JSON matching `model` (schema sent as response_format, then validated here, since
        not every provider enforces it). Retries on invalid replies; returns (parsed or None, last raw reply)."""
        fmt = {"type": "json_schema", "json_schema": {"name": model.__name__, "schema": model.model_json_schema()}}
        raw = ""
        for _ in range(retries + 1):
            raw = self.chat(system, user, temperature, response_format=fmt)
            if (parsed := parse_json(model, raw)) is not None:
                return parsed, raw
        return None, raw

    def map(self, fn, items: list) -> list:
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(fn, items))
