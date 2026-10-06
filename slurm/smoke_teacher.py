"""Day-1 teacher checks against a running vLLM server (called by smoke_teacher.sbatch).

1. top-20 logprobs come back with token ids, and the ids decode to the answer with the STUDENT's
   tokenizer (needed for v2 logit KD);
2. a tool call comes back parsed (hermes parser);
3. throughput at concurrency 1 / 16 / 64 (output tokens per second), to size the 10k generation.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from transformers import AutoTokenizer

from distillkit.records import openai_tools
from distillkit.tools import SCHEMAS

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--student", required=True)
    a = p.parse_args()
    client = OpenAI(base_url=a.base_url, api_key="none")
    tok = AutoTokenizer.from_pretrained(a.student)
    results = {}

    r = client.chat.completions.create(
        model=a.model, messages=[{"role": "user", "content": "What does sbatch --array=0-9%2 do? One sentence."}],
        max_tokens=128, temperature=0, logprobs=True, top_logprobs=20,
        extra_body={"return_tokens_as_token_ids": True, **NO_THINK},
    )
    toks = r.choices[0].logprobs.content
    ids = [int(t.token.removeprefix("token_id:")) for t in toks]
    raw_decoded = tok.decode(ids)
    decoded = tok.decode(ids, skip_special_tokens=True)
    content = r.choices[0].message.content or ""

    special_ids = set(tok.all_special_ids)
    special_positions = [i for i, token_id in enumerate(ids) if token_id in special_ids]
    terminal_specials_only = (
        not special_positions
        or special_positions == list(range(special_positions[0], len(ids)))
    )

    results["logprobs"] = {
        "tokens": len(toks),
        "top_k": min(len(t.top_logprobs) for t in toks),
        "ids_decode_to_content": (
            terminal_specials_only
            and decoded.strip() == content.strip()
        ),
        "terminal_special_tokens": [
            tok.convert_ids_to_tokens(ids[i]) for i in special_positions
        ],
    }
    if not results["logprobs"]["ids_decode_to_content"]:
        results["logprobs"]["raw_decoded"] = raw_decoded
        results["logprobs"]["decoded"] = decoded
        results["logprobs"]["content"] = content

    r = client.chat.completions.create(
        model=a.model, messages=[{"role": "user", "content": "Is my job 4718207 still running?"}],
        tools=openai_tools(SCHEMAS), max_tokens=256, temperature=0, extra_body=NO_THINK,
    )
    calls = r.choices[0].message.tool_calls or []
    results["tool_call"] = [{"name": c.function.name, "arguments": json.loads(c.function.arguments)} for c in calls]

    def one(_):
        out = client.chat.completions.create(
            model=a.model, messages=[{"role": "user", "content": "Explain Slurm job arrays with an example script."}],
            max_tokens=256, temperature=0.7, extra_body=NO_THINK,
        )
        return out.usage.completion_tokens

    results["throughput_tok_per_s"] = {}
    for c in (1, 16, 64):
        t0 = time.time()
        with ThreadPoolExecutor(c) as pool:
            n = sum(pool.map(one, range(c * 2)))
        results["throughput_tok_per_s"][c] = round(n / (time.time() - t0), 1)

    print(json.dumps(results, indent=2))
    ok = results["logprobs"]["ids_decode_to_content"] and results["logprobs"]["top_k"] >= 20 and results["tool_call"]
    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
