"""Teacher = any OpenAI-compatible chat endpoint (OpenCode Zen, Ollama, `vllm serve` or llama-server)."""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import openai
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from .config import EndpointCfg
from .records import openai_tools
from .schemas import parse_json

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Provider-side transient failures (429 "endpoint unavailable", 5xx, timeouts, dropped connections).
_TRANSIENT = (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError)
_retry_transient = retry(
    retry=retry_if_exception_type(_TRANSIENT),
    wait=wait_random_exponential(multiplier=2, max=60),  # exponential backoff with jitter
    stop=stop_after_attempt(8),
    reraise=True,
)


@dataclass
class Completion:
    content: str
    tool_calls: list[dict] | None = None  # OpenAI format: [{"id", "type": "function", "function": {...}}]
    logprobs: dict | None = None  # {"tokens": [...], "logprobs": [...], "top": [[[token, logprob], ...], ...]}


def _compact_logprobs(lp) -> dict | None:
    """OpenAI logprobs object -> parallel lists (about 3x smaller than the raw JSON).

    Tokens are strings as the server returns them; vLLM gives `token_id:<n>` strings when the
    endpoint's `extra_body` sets `return_tokens_as_token_ids: true`.
    """
    if lp is None or not lp.content:
        return None
    return {
        "tokens": [t.token for t in lp.content],
        "logprobs": [t.logprob for t in lp.content],
        "top": [[[c.token, c.logprob] for c in (t.top_logprobs or [])] for t in lp.content],
    }


class Teacher:
    def __init__(self, cfg: EndpointCfg, name: str = "teacher"):
        # keys come from the environment, never from the config file
        api_key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else "none"
        if not api_key:
            raise SystemExit(f"[{name}] set the {cfg.api_key_env} environment variable")
        self.client = OpenAI(base_url=cfg.base_url, api_key=api_key, max_retries=0)  # retries: tenacity in _create
        self.cfg = cfg
        self.model = cfg.model

    @_retry_transient
    def _create(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        top_logprobs: int = 0,
    ) -> Completion:
        extra = {}
        if response_format:
            extra["response_format"] = response_format
        if tools:
            extra["tools"] = openai_tools(tools)
        if top_logprobs:
            extra |= {"logprobs": True, "top_logprobs": top_logprobs}
        if self.cfg.extra_body:
            extra["extra_body"] = self.cfg.extra_body
        if messages and messages[0]["role"] == "system":
            # "/no_think" disables Qwen3 hybrid thinking; other models ignore it
            messages = [{**messages[0], "content": messages[0]["content"] + " /no_think"}, *messages[1:]]
        resp = self._create(
            model=self.model,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
            max_tokens=self.cfg.max_tokens,
            **extra,
        )
        choice = resp.choices[0]
        calls = [c.model_dump() for c in choice.message.tool_calls] if choice.message.tool_calls else None
        return Completion(
            content=_THINK.sub("", choice.message.content or "").strip(),
            tool_calls=calls,
            logprobs=_compact_logprobs(choice.logprobs) if top_logprobs else None,
        )

    def chat(self, system: str, user: str, temperature: float | None = None, **kwargs) -> Completion:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.complete(messages, temperature, **kwargs)

    def chat_json[M: BaseModel](
        self, system: str, user: str, model: type[M], temperature: float | None = None, retries: int = 1
    ) -> tuple[M | None, str]:
        """Ask for JSON matching `model` (schema sent as response_format, then validated here, since
        not every provider enforces it). Retries on invalid replies; returns (parsed or None, last raw reply)."""
        fmt = {"type": "json_schema", "json_schema": {"name": model.__name__, "schema": model.model_json_schema()}}
        raw = ""
        for _ in range(retries + 1):
            raw = self.chat(system, user, temperature, response_format=fmt).content
            if (parsed := parse_json(model, raw)) is not None:
                return parsed, raw
        return None, raw

    def map(self, fn, items: list) -> list:
        with ThreadPoolExecutor(self.cfg.concurrency) as pool:
            return list(pool.map(fn, items))
