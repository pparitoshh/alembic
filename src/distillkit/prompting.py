"""One place that renders student prompts, so training and inference see identical text."""

from .records import openai_tools


def render_prompt(tokenizer, system: str, question: str, tools: list[dict] | None = None) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tools=openai_tools(tools),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3: adds an empty <think></think>; ignored by other templates
    )
