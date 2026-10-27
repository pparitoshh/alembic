"""Upload a release folder (release_build.sh) to VaibhavKKM/hpc-assistant-qwen3-4b `main` as the next ClusterPilot-4B
version: the current main (version 1.N, the highest v1.x branch + 1) is first kept as branch v1.N, the new model becomes
1.N+1. The model card gets a dated "Latest update" section, and the model-family table and the "current main" mentions
move to the new version. Version names only on the card, never internal run names. The token comes from HF_TOKEN in
this process only (read from ssh stdin). DRY=1 writes the card next to the release folder and uploads nothing.

usage: HF_TOKEN=... hf_push.py RELEASE_DIR
"""
import datetime
import json
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = 'VaibhavKKM/hpc-assistant-qwen3-4b'
d = Path(sys.argv[1])
dry = os.environ.get('DRY') == '1'
api = HfApi(token=os.environ['HF_TOKEN'])
refs = [b.name for b in api.list_repo_refs(REPO).branches]
minor = max(int(m.group(1)) for b in refs if (m := re.fullmatch(r'v1\.(\d+)', b)))
old, new = f'1.{minor + 1}', f'1.{minor + 2}'  # version now on main, version being published
keep = f'v{old}'
if keep not in refs and not dry:
    api.create_branch(REPO, branch=keep, revision=api.model_info(REPO).sha)
    print('kept current main as branch', keep)
res = json.loads((d / 'eval_curiosity_result.json').read_text())
r, name = res['result'], res['name']
maj = r['majority']
rows = '\n'.join(f"| {l} | {maj[l]['a']:.3f} | {maj[l]['b']:.3f} | {maj[l]['diff']:+.3f} [{maj[l]['paired_95'][0]:+.3f}, {maj[l]['paired_95'][1]:+.3f}] |"
                 for l in ('v2', 'v3', 'd1') if l in maj)
bj = '; '.join(f"{j.replace('judged_', '').replace('judged', 'gpt-oss-20b')}: " + ', '.join(f'{l} {v:+.3f}' for l, v in d2.items())
               for j, d2 in r['by_judge'].items() if j != 'judged_majority')
today = datetime.date.today().strftime('%b %-d')
section = f"""## Latest update (version {new}, {today})

`main` now holds **ClusterPilot-4B {new}**: version {old} further distilled from Qwen3-235B-A22B-Instruct-2507 (FP8) with
full-parameter training on 8 NVIDIA B300 GPUs (teacher answers with top-20 log-probabilities for prompts with the app's
documentation excerpts). Same 4B size, same chat format and GGUF files. Version {old} stays on the
[`{keep}`](https://huggingface.co/{REPO}/tree/{keep}) branch, version 1.0 on [`v1.0`](https://huggingface.co/{REPO}/tree/v1.0).

Win rate against closed-book Qwen3-4B (same setup as the tables below: answers at 1,536 tokens, retriever e, test sets
v2, v3, d1, 3-judge majority), version {new} vs version 1.0 judged in the same run:

| Test set | Version {new} | Version 1.0 | Difference [paired 95% interval] |
|---|---:|---:|---:|
{rows}
| **Mean** | **{r['mean_cand']:.3f}** | **{r['mean_r27e']:.3f}** | **{r['mean_cand'] - r['mean_r27e']:+.3f}** |

Per judge (difference vs version 1.0): {bj}. Judged on our B300 cluster with Qwen3-235B in FP8 instead of AWQ, so absolute
numbers can differ slightly from the tables below, which are for version 1.0. A second check on Leonardo follows.

"""
card = Path(hf_hub_download(REPO, 'README.md', token=os.environ['HF_TOKEN'])).read_text()
card = re.sub(r'## Latest update \(.*?\n(?=## )', '', card, flags=re.S)
card = card.replace('\n## Model family', '\n' + section + '## Model family', 1)
B = f'https://huggingface.co/{REPO}/tree/'
row = re.search(rf'^\| \*\*ClusterPilot-4B {re.escape(old)}\*\* \| `main` \(this branch\) \| (.*?) \| (.*?) \|$', card, re.M)
if row:
    gain = f"mean {r['mean_cand']:.3f} vs {r['mean_r27e']:.3f} for 1.0 (B300 test)"
    card = card.replace(row.group(0), f"| **ClusterPilot-4B {new}** | `main` (this branch) | {row.group(1)} | {gain} |\n"
                                      f"| ClusterPilot-4B {old} | [`{keep}`]({B}{keep}) | Previous release | {row.group(2)} |", 1)
card = card.replace(f'the current `main`, version {old},', f'the current `main`, version {new},')
card = card.replace(f'`main` is now {old}, see Latest update', f'`main` is now {new}, see Latest update')
card = card.replace(f'| **ClusterPilot-4B {old}** (`main`) |', f'| ClusterPilot-4B {old} ([`{keep}`]({B}{keep})) |')  # leaderboard row: the new version gets its own row once scored
assert not re.search(r'\bkd\d|kdall', card), 'internal run name on the card'
(d / 'README.md').write_text(card)
if dry:
    print('DRY card written', d / 'README.md', 'version', new, 'keep', keep)
    sys.exit(0)
c = api.upload_folder(repo_id=REPO, folder_path=str(d), commit_message=f'ClusterPilot-4B {new}: B300 full-parameter distillation from Qwen3-235B',
                      allow_patterns=['*.safetensors', '*.json', '*.jinja', 'README.md', 'gguf/*'])
print('PUSHED', c)
