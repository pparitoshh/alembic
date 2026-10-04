from distillkit.config import ExportCfg
from distillkit.export import OLLAMA_TEMPLATE, generation_prefix, modelfile


class FakeTokenizer:
    """apply_chat_template stand-in: a ChatML generation prompt with a model-specific suffix."""

    def __init__(self, suffix: str):
        self.suffix = suffix

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, enable_thinking=None):
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return body + "<|im_start|>assistant\n" + self.suffix


def test_generation_prefix_matches_training():
    assert generation_prefix(FakeTokenizer("")) == ""  # Qwen3-*-2507
    assert generation_prefix(FakeTokenizer("<think>\n\n</think>\n\n")) == "<think>\n\n</think>\n\n"  # hybrid, thinking off


def test_modelfile():
    mf = modelfile("model-Q4_K_M.gguf", "You are an HPC assistant.", "", 8192, ExportCfg().sampling)
    assert mf.startswith("FROM ./model-Q4_K_M.gguf\n")
    assert 'SYSTEM """You are an HPC assistant."""' in mf
    for line in ("PARAMETER stop <|im_start|>", "PARAMETER stop <|im_end|>", "PARAMETER num_ctx 8192", "PARAMETER temperature 0.7", "PARAMETER top_k 20"):
        assert line in mf
    # tool-calling section kept verbatim from Ollama's qwen3 template; no thinking left over
    assert "<tool_call>\n{\"name\": \"{{ .Function.Name }}\", \"arguments\": {{ .Function.Arguments }}}\n</tool_call>" in mf
    assert "<tool_response>" in mf and ".Thinking" not in mf and "@GEN_PREFIX@" not in mf
    assert "<|im_start|>assistant\n{{ end }}" in mf  # 2507: nothing after the assistant header


def test_template_placeholder_once():
    assert OLLAMA_TEMPLATE.count("@GEN_PREFIX@") == 1
