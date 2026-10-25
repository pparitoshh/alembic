"""Fetch lesson sources (Markdown/reST) from GitHub repos whose GitHub Pages site is unreachable from Leonardo.

usage: fetch_github_lessons.py OUT_DIR ORG REPO[,REPO...] [--source-id ID] [--topic T] [--max-files N]

enccs.github.io resets TLS connections from the Leonardo login node, so the same CC-BY lesson text is taken
from the source repos: one GitHub API call per repo (tree listing + licence), then raw files at <= 1 request/s.
Rows go to OUT_DIR/manifest_github.jsonl with the same fields as crawl.py's manifest.jsonl (merged at finalize).
"""
import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

UA = "distillkit-hpc-docs-research/1.0 (+research use: offline documentation archive; <=1 req/s)"
KEEP = re.compile(r"^(content|episodes|docs|lesson)/(?!_).*\.(md|rst)$")


def get(url):
    time.sleep(1.0)
    return requests.get(url, headers={"User-Agent": UA}, timeout=60)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("org")
    p.add_argument("repos")
    p.add_argument("--source-id", default="enccs_lessons_src")
    p.add_argument("--topic", default="teaching")
    p.add_argument("--max-files", type=int, default=800)
    a = p.parse_args()
    mf = a.out / "manifest_github.jsonl"
    done = {json.loads(line)["url"] for line in open(mf)} if mf.exists() else set()
    n = 0
    for repo in a.repos.split(","):
        meta = get(f"https://api.github.com/repos/{a.org}/{repo}").json()
        if "default_branch" not in meta:  # e.g. unauthenticated API rate limit (60/h): rerun later, it resumes
            print(f"[github] {repo}: API error {meta.get('message', '')[:120]}", flush=True)
            continue
        branch = meta["default_branch"]
        lic = (meta.get("license") or {}).get("spdx_id") or "unstated (GitHub API: no licence detected)"
        tree = get(f"https://api.github.com/repos/{a.org}/{repo}/git/trees/{branch}?recursive=1").json().get("tree", [])
        for path in sorted(t["path"] for t in tree if t["type"] == "blob" and KEEP.match(t["path"])):
            url = f"https://raw.githubusercontent.com/{a.org}/{repo}/{branch}/{path}"
            if url in done or n >= a.max_files:
                continue
            row = {"url": url, "source_id": a.source_id, "topic": a.topic, "factory_ids": [], "source_kind": "training_materials",
                   "retrieved_at": datetime.now(timezone.utc).isoformat(), "discovered_from": f"github_tree:{a.org}/{repo}@{branch}",
                   "training_status": "not_approved_for_training", "reuse_status": "unreviewed", "license": lic,
                   "license_basis": "GitHub repository licence (API spdx_id); verify in licence review",
                   "rendered_url": f"https://{a.org.lower()}.github.io/{repo}/" + re.sub(r"^(content|episodes|docs|lesson)/|\.(md|rst)$", "", path) + "/"}
            r = get(url)
            row.update(http_status=r.status_code, resolved_url=r.url, content_type=r.headers.get("content-type", "").split(";")[0],
                       last_modified=r.headers.get("last-modified"), etag=r.headers.get("etag"))
            if r.status_code == 200:
                raw, key = r.content, hashlib.sha1(url.encode()).hexdigest()[:20]
                orig = a.out / "originals" / a.source_id / f"{key}.{path.rsplit('.', 1)[1]}"
                txt = a.out / "extracted" / a.source_id / f"{key}.md"
                orig.parent.mkdir(parents=True, exist_ok=True)
                txt.parent.mkdir(parents=True, exist_ok=True)
                orig.write_bytes(raw)
                md = raw.decode("utf-8", errors="replace")
                txt.write_text(md)
                title = re.search(r"^(?:#\s+(.+)|(.+)\n[=#*-]{3,})", md, re.M)
                row.update(status="downloaded", original_path=str(orig.relative_to(a.out)), sha256=hashlib.sha256(raw).hexdigest(),
                           bytes=len(raw), title=(title.group(1) or title.group(2)).strip() if title else path, language="en",
                           license_links=[], extraction_method="raw_text (lesson source)", extraction_error=None,
                           text_path=str(txt.relative_to(a.out)), text_sha256=hashlib.sha256(md.encode()).hexdigest(),
                           characters=len(md), words=len(md.split()))
                n += 1
            else:
                row["status"] = f"http_{r.status_code}"
            with open(mf, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[github] {repo}: licence {lic}, total files so far {n}", flush=True)


if __name__ == "__main__":
    main()
