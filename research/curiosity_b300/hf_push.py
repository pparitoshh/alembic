"""Upload a release folder (release_build.sh) to VaibhavKKM/hpc-assistant-qwen3-4b `main`, after keeping the current
main as a branch (KEEP_BRANCH, default round27) once. The model card gets a dated "Latest update" section with the
new results above the existing text. The token comes from HF_TOKEN in this process only (read from ssh stdin).

usage: HF_TOKEN=... hf_push.py RELEASE_DIR
"""
import json
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = 'VaibhavKKM/hpc-assistant-qwen3-4b'
d = Path(sys.argv[1])
api = HfApi(token=os.environ['HF_TOKEN'])
keep = os.environ.get('KEEP_BRANCH', 'round27')
refs = [b.name for b in api.list_repo_refs(REPO).branches]
if keep not in refs:
    api.create_branch(REPO, branch=keep, revision=api.model_info(REPO).sha)
    print('kept current main as branch', keep)
res = json.loads((d / 'eval_curiosity_result.json').read_text())
r, name = res['result'], res['name']
maj = r['majority']
rows = '\n'.join(f"| {l} | {maj[l]['a']:.3f} | {maj[l]['b']:.3f} | {maj[l]['diff']:+.3f} [{maj[l]['paired_95'][0]:+.3f}, {maj[l]['paired_95'][1]:+.3f}] |"
                 for l in ('v2', 'v3', 'd1') if l in maj)
bj = '; '.join(f"{j.replace('judged_', '').replace('judged', 'gpt-oss-20b')}: " + ', '.join(f'{l} {v:+.3f}' for l, v in d2.items())
               for j, d2 in r['by_judge'].items() if j != 'judged_majority')
section = f"""## Latest update ({name}, Oct 26)

`main` now holds **{name}**: round 27 further distilled from Qwen3-235B-A22B-Instruct-2507 (FP8) with full-parameter
training on 8 NVIDIA B300 GPUs (teacher answers with top-20 log-probabilities for prompts with the app's documentation
excerpts). Same 4B size, same chat format and GGUF files. Previous `main` (round 27) stays on the
[`{keep}`](https://huggingface.co/{REPO}/tree/{keep}) branch.

Win rate against closed-book Qwen3-4B (same setup as the tables below: answers at 1,536 tokens, retriever e, reviewed
test sets, 3-judge majority), the new model vs round 27 judged in the same run:

| Test set | New model | Round 27 | Difference [paired 95% interval] |
|---|---:|---:|---:|
{rows}
| **Mean** | **{r['mean_cand']:.3f}** | **{r['mean_r27e']:.3f}** | **{r['mean_cand'] - r['mean_r27e']:+.3f}** |

Per judge (difference vs round 27): {bj}. Judged on our cluster with Qwen3-235B in FP8 instead of AWQ, so absolute
numbers can differ slightly from the tables below, which are for round 27.

"""
card = Path(hf_hub_download(REPO, 'README.md', token=os.environ['HF_TOKEN'])).read_text()
card = re.sub(r'## Latest update \(.*?\n(?=## )', '', card, flags=re.S)
card = card.replace('\n## Model family', '\n' + section + '## Model family', 1)
(d / 'README.md').write_text(card)
c = api.upload_folder(repo_id=REPO, folder_path=str(d), commit_message=f'{name}: B300 full-parameter distillation from Qwen3-235B',
                      allow_patterns=['*.safetensors', '*.json', '*.jinja', 'README.md', 'gguf/*'])
print('PUSHED', c)
