"""One place that renders student prompts, so training and inference see identical text."""


def render_prompt(tokenizer, system: str, question: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3: adds an empty <think></think>; ignored by other templates
    )
