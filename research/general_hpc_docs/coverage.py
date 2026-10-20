"""Measure HPC topic coverage of the documentation archives and Q&A sets against taxonomy.json. CPU only.

usage: coverage.py --archive NAME=DIR [--archive ...] --qa NAME=GLOB [--qa ...] --out OUT_DIR
                   [--taxonomy taxonomy.json] [--workers 16] [--doc-urls GLOB ...]

Rules (deterministic, see taxonomy.json "matching_rules"):
- document (manifest row with status "downloaded" and a text_path): maps to a leaf if any leaf regex matches its URL
  or title, or the leaf's regexes match >= MIN_BODY_HITS times in the first BODY_CHARS characters of its extracted
  text. Multi-label. Topical chunks ("est_chunks_1500") per mapped document: all of its characters/1500 if the
  URL/title matched (the page is about the topic), otherwise the number of distinct 1,500-char windows of the text
  that contain a match. Summed over documents (a window can count for several leaves).
- Q&A row: maps to a leaf if any leaf regex matches the question (first user turn, with any quoted
  "Provided input ..." excerpt removed) or the row's source URL. source_url comes from the row, or is resolved from
  doc_id via --doc-urls files (JSON {doc_id: {url: ...}} like ai_factory_all_v5/licences.json, or jsonl rows with
  doc_id + source_url). Multi-label. Q&A sets are counted separately; "qa_verified" = sum over sets whose name does
  not start with "planned".
- status (source material): GAP if docs < GAP_DOCS or topical chunks < GAP_CHUNKS; THIN if docs < THIN_DOCS or
  topical chunks < THIN_CHUNKS; else OK. qa_status: GAP if verified+planned Q&A < QA_GAP, THIN if < QA_THIN.

Per-archive document->leaf assignments are cached in OUT_DIR/doc_leaves_<NAME>.jsonl (keyed by taxonomy sha256), so
re-running with an extra archive only scans the new one; Q&A leaf counts are cached in OUT_DIR/qa_leaves_<NAME>.json
(keyed by taxonomy sha256 + file sizes/mtimes). Outputs: coverage.csv, coverage.json, coverage.md.
"""
import argparse
import csv
import glob
import hashlib
import json
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

MIN_BODY_HITS, BODY_CHARS = 3, 200_000
GAP_DOCS, GAP_CHUNKS, THIN_DOCS, THIN_CHUNKS = 10, 100, 40, 500
QA_GAP, QA_THIN = 20, 100
CACHE_VERSION = "topical-chunks-v2"
_RX = None


def compile_leaves(tax):
    return [(l["id"], re.compile("|".join(f"(?:{p})" for p in l["patterns"]), re.I))
            for a in tax["areas"] for l in a["leaves"]]


def _init(tax):
    global _RX
    _RX = compile_leaves(tax)


def doc_leaves(job):
    """(key, url, title, text_path, chars) -> (key, chars, {leaf id: topical 1,500-char chunks})"""
    key, url, title, path, chars = job
    try:
        body = open(path, errors="replace").read(BODY_CHARS)
    except OSError:
        body = ""
    head = f"{url}\n{title}"
    out = {}
    for lid, rx in _RX:
        if rx.search(head):
            out[lid] = max(1, round(chars / 1500))
            continue
        pos = [m.start() for m in rx.finditer(body)]
        if len(pos) >= MIN_BODY_HITS:
            out[lid] = len({p // 1500 for p in pos})
    return key, chars, out


def scan_archive(name, adir, tax, tax_sha, out, workers):
    cache = out / f"doc_leaves_{name}.jsonl"
    done = {}
    if cache.exists():
        for line in open(cache):
            r = json.loads(line)
            if r.get("taxonomy_sha256") == tax_sha:
                done[r["key"]] = r
    jobs, seen = [], set()
    for line in open(Path(adir) / "manifest.jsonl"):
        r = json.loads(line)
        if r.get("status") != "downloaded" or not r.get("text_path"):
            continue
        key = f'{r["url"]}|{r.get("text_sha256", "")}'
        if key in seen:
            continue
        seen.add(key)
        if key not in done:
            jobs.append((key, r["url"], r.get("title", ""), str(Path(adir) / r["text_path"]), r.get("characters", 0)))
    if jobs:
        with Pool(workers, initializer=_init, initargs=(tax,)) as pool, open(cache, "a") as f:
            for i, (key, chars, leaves) in enumerate(pool.imap_unordered(doc_leaves, jobs, chunksize=64)):
                row = {"key": key, "chars": chars, "leaves": leaves, "taxonomy_sha256": tax_sha}
                done[key] = row
                f.write(json.dumps(row) + "\n")
                if i % 5000 == 0:
                    print(f"[{name}] {i}/{len(jobs)}", flush=True)
    return [done[k] for k in seen]


def qa_leaves(job):
    """(question, url) -> [leaf ids]"""
    q, url = job
    return [lid for lid, rx in _RX if rx.search(q) or (url and rx.search(url))]


PROVIDED = re.compile(r"\n\s*Provided input\b.*", re.S)


def qa_question(r):
    q = r.get("question")
    if not q:
        q = next((m.get("content", "") for m in r.get("messages", []) if m.get("role") == "user"), "")
    return PROVIDED.sub("", q or "")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--taxonomy", default=str(Path(__file__).with_name("taxonomy.json")))
    p.add_argument("--archive", action="append", default=[], help="NAME=DIR (DIR has manifest.jsonl)")
    p.add_argument("--qa", action="append", default=[], help="NAME=GLOB of jsonl files (prefix NAME with 'planned' for not-yet-verified)")
    p.add_argument("--doc-urls", action="append", default=[], help="GLOB of doc_id->url maps (json dict or jsonl rows)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = open(a.taxonomy, "rb").read()
    tax, tax_sha = json.loads(raw), hashlib.sha256(raw + CACHE_VERSION.encode()).hexdigest()
    leaves = [(ar["id"], ar["name"], l["id"], l["name"]) for ar in tax["areas"] for l in ar["leaves"]]
    stats = {lid: defaultdict(float) for _, _, lid, _ in leaves}

    anames = []
    for spec in a.archive:
        name, adir = spec.split("=", 1)
        anames.append(name)
        rows = scan_archive(name, adir, tax, tax_sha, out, a.workers)
        print(f"[{name}] {len(rows)} documents", flush=True)
        for r in rows:
            for lid, ch in r["leaves"].items():
                s = stats[lid]
                s["docs"] += 1
                s[f"docs:{name}"] += 1
                s["chunks"] += ch
                s[f"chunks:{name}"] += ch
                s["doc_chars"] += r["chars"]

    doc_url = {}
    for g in a.doc_urls:
        for fn in sorted(glob.glob(g)):
            if fn.endswith(".json"):
                for k, v in json.load(open(fn)).items():
                    if isinstance(v, dict) and v.get("url"):
                        doc_url.setdefault(k, v["url"])
            else:
                for line in open(fn):
                    r = json.loads(line)
                    if r.get("doc_id") and r.get("source_url"):
                        doc_url.setdefault(r["doc_id"], r["source_url"])
    qnames, qtotals = [], {}
    for spec in a.qa:
        name, g = spec.split("=", 1)
        qnames.append(name)
        files = sorted(glob.glob(g))
        sig = hashlib.sha256(json.dumps([hashlib.sha256(raw).hexdigest()] + [(f, Path(f).stat().st_size, Path(f).stat().st_mtime) for f in files]).encode()).hexdigest()
        cache = out / f"qa_leaves_{name}.json"
        if cache.exists() and json.load(open(cache)).get("signature") == sig:
            c = json.load(open(cache))
        else:
            jobs = []
            for fn in files:
                for line in open(fn):
                    r = json.loads(line)
                    jobs.append((qa_question(r), r.get("source_url") or doc_url.get(r.get("doc_id", ""), "")))
            counts = defaultdict(int)
            with Pool(a.workers, initializer=_init, initargs=(tax,)) as pool:
                for leaves_hit in pool.imap(qa_leaves, jobs, chunksize=256):
                    for lid in leaves_hit:
                        counts[lid] += 1
            c = {"signature": sig, "rows": len(jobs), "counts": counts}
            cache.write_text(json.dumps(c))
        for lid, k in c["counts"].items():
            stats[lid][f"qa:{name}"] += k
        qtotals[name] = c["rows"]
        print(f"[qa {name}] {c['rows']} rows", flush=True)

    table = []
    for aid, aname, lid, lname in leaves:
        s = stats[lid]
        docs, chunks = int(s["docs"]), int(s["chunks"])
        status = "GAP" if docs < GAP_DOCS or chunks < GAP_CHUNKS else ("THIN" if docs < THIN_DOCS or chunks < THIN_CHUNKS else "OK")
        row = {"area": aid, "area_name": aname, "leaf": lid, "leaf_name": lname, "docs": docs, "est_chunks_1500": chunks,
               "mapped_doc_chars": int(s["doc_chars"]), "status": status}
        for n in anames:
            row[f"docs_{n}"] = int(s[f"docs:{n}"])
            row[f"chunks_{n}"] = int(s[f"chunks:{n}"])
        for n in qnames:
            row[f"qa_{n}"] = int(s[f"qa:{n}"])
        row["qa_verified"] = sum(row[f"qa_{n}"] for n in qnames if not n.startswith("planned"))
        row["qa_planned"] = sum(row[f"qa_{n}"] for n in qnames if n.startswith("planned"))
        qa = row["qa_verified"] + row["qa_planned"]
        row["qa_status"] = "GAP" if qa < QA_GAP else ("THIN" if qa < QA_THIN else "OK")
        table.append(row)
    with open(out / "coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    counts = {k: sum(r["status"] == k for r in table) for k in ("GAP", "THIN", "OK")}
    qa_counts = {k: sum(r["qa_status"] == k for r in table) for k in ("GAP", "THIN", "OK")}
    (out / "coverage.json").write_text(json.dumps({
        "taxonomy": a.taxonomy, "taxonomy_sha256": tax_sha, "n_leaves": len(table), "status_counts": counts, "qa_status_counts": qa_counts,
        "thresholds": {"min_body_hits": MIN_BODY_HITS, "body_chars": BODY_CHARS, "gap_docs": GAP_DOCS, "gap_chunks": GAP_CHUNKS,
                       "thin_docs": THIN_DOCS, "thin_chunks": THIN_CHUNKS, "qa_gap": QA_GAP, "qa_thin": QA_THIN},
        "archives": a.archive, "qa_sets": qtotals, "rows": table}, indent=1) + "\n")
    with open(out / "coverage.md", "w") as f:
        f.write(f"# HPC coverage ({len(table)} leaves)\n\nsource status: {counts}; Q&A status: {qa_counts}\n\n")
        f.write("| area | leaf | docs | topical chunks | " + " | ".join(f"docs {n}" for n in anames) +
                " | Q&A verified | Q&A planned | status | Q&A status |\n|" + "---|" * (7 + len(anames)) + "\n")
        for r in table:
            f.write(f"| {r['area']} | {r['leaf']} | {r['docs']} | {r['est_chunks_1500']} | " +
                    " | ".join(str(r[f'docs_{n}']) for n in anames) +
                    f" | {r['qa_verified']} | {r['qa_planned']} | {r['status']} | {r['qa_status']} |\n")
    print(json.dumps({"source": counts, "qa": qa_counts}))
    for r in table:
        if r["status"] != "OK" or r["qa_status"] != "OK":
            print(f"{r['status']:4} qa:{r['qa_status']:4} {r['leaf']:32} docs={r['docs']:5} chunks={r['est_chunks_1500']:6} qa={r['qa_verified']}+{r['qa_planned']}")


if __name__ == "__main__":
    main()
