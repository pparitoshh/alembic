"""The one data format every stage reads and writes.

A row is a chat transcript plus provenance:

    {"doc_id", "chunk_id", "persona", "task", "question", "sample", "teacher",
     "messages": [{"role": "user", ...}, {"role": "assistant", ...}, ...],
     "tools": [<JSON schema>, ...]}            # optional; tool-calling traces only

Prose answers are one user + one assistant turn. Tool-calling traces add assistant `tool_calls`
and `tool` result turns in between (Hermes format, which the Qwen3 chat template renders).
The student's system prompt is *not* stored; `training_example` adds it, so it can change
without regenerating data.
"""

import json


def prose_row(meta: dict, question: str, answer: str) -> dict:
    return {
        **meta,
        "question": question,
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
    }


def messages(row: dict) -> list[dict]:
    """Transcript of a row; rows from before the messages format only have question/answer."""
    if "messages" in row:
        return row["messages"]
    return [{"role": "user", "content": row["question"]}, {"role": "assistant", "content": row["answer"]}]


def final_answer(row: dict) -> str:
    """Terminal assistant answer, never a pre-call plan from an unfinished trace."""
    transcript = messages(row)
    if isinstance(transcript, list) and transcript and isinstance(transcript[-1], dict):
        m = transcript[-1]
        if m.get("role") == "assistant" and m.get("tool_calls") in (None, []):
            content = m.get("content")
            return content if isinstance(content, str) else ""
    return ""


def openai_tools(schemas: list[dict] | None) -> list[dict] | None:
    """Wrap bare schemas as OpenAI `{"type": "function", "function": ...}` entries. vLLM, llama-server
    and Ollama hand tools to the Qwen chat template in this form, so training must render them the same way."""
    if not schemas:
        return None
    return [s if s.get("type") == "function" else {"type": "function", "function": s} for s in schemas]


def training_example(row: dict, system_prompt: str) -> dict:
    """Conversational SFT example for TRL (loss on assistant turns only)."""
    if any(m.get('origin') == 'runtime_guard' for m in messages(row)):
        raise ValueError('runtime guard responses are application output, not teacher training targets')
    if row.get('purpose') == 'diagnostic_only' or row.get('training_eligible') is False:
        raise ValueError('diagnostic or explicitly ineligible records are not training examples')
    tools = openai_tools(row.get("tools"))
    return {
        "messages": [{"role": "system", "content": system_prompt}, *messages(row)],
        # JSON string keeps the Arrow schema uniform across rows with different tools; TRL decodes it
        "tools": json.dumps(tools) if tools else None,
        "chat_template_kwargs": {"enable_thinking": False},  # Qwen3 hybrid; ignored by other templates
    }
