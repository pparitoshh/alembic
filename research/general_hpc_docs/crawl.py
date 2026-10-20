"""Polite crawler for general HPC documentation (raw archive for later synthetic Q&A). CPU, login node.

usage: crawl.py SOURCES_JSON OUT_DIR [--old-manifest M [--old-manifest M2 ...]] [--hours H] [--max-total N] [--only id,id] [--check]

Per source (see sources.json): BFS from `seeds`, staying inside `prefixes`, honouring robots.txt and
<= 1 request/s per host (one thread per host), at most `max_pages` fetches. HTML is converted with
distillkit.fetch_sources.html_to_markdown, PDFs with pypdf (OUT_DIR/tools/pylib). URLs already in the
old archive manifest(s) (--old-manifest may be repeated) are skipped; identical raw/text sha256 are recorded as duplicate_content.
manifest.jsonl uses the old archive's field names (+ topic, license, license_basis) and is appended
row by row, so the run is resumable and partial results are usable. SOURCES.csv/TOPICS.csv/STATUS.json
are rewritten periodically. --check fetches only the seeds and prints their status.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import sys
import threading
import time
import urllib.robotparser
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from distillkit.fetch_sources import html_to_markdown  # noqa: E402  (reuse the repo converter)

UA = "distillkit-hpc-docs-research/1.0 (+research use: offline documentation archive; <=1 req/s)"
SKIP_EXT = re.compile(r"\.(png|jpe?g|gif|svg|ico|webp|css|js|map|zip|gz|tgz|bz2|xz|tar|whl|mp4|webm|mp3|pdf|ps|eps|"
                      r"txt|json|xml|ipynb|py|c|cc|f90|h|hpp|cpp|cu|sh|ya?ml|tex|bib|odp|pptx?|docx?|xlsx?|epub|"
                      r"woff2?|ttf|rss|atom|csv|dat|sif|def|toml|cfg|ini|patch|diff)$", re.I)
SKIP_PATH = re.compile(r"/_sources/|/_static/|/_images/|/_downloads/|/_modules/|genindex|/search(\.html)?/?$|"
                       r"py-modindex|Special:|/edit/|/blob/|/raw/|/tree/|/commits?/|/print/|/feed/|/tag/|/tags/|"
                       r"/instructor/|/discuss(ion)?/|/cdn-cgi/|/assets/|/login|/signin")
LICENSE_RX = re.compile(r'href="([^"]*(?:creativecommons\.org/licenses|creativecommons\.org/publicdomain|'
                        r'opensource\.org/licenses|apache\.org/licenses|gnu\.org/licenses|/LICEN[CS]E)[^"]*)"', re.I)
lock = threading.Lock()
state = {"downloaded": 0, "rows": 0, "stop": False}


def norm(url: str) -> str:
    """Comparable URL key: https, no query/fragment, no trailing index.html or slash."""
    p = urlparse(url.strip())
    path = re.sub(r"(index\.html?|index\.php)$", "", p.path).rstrip("/")
    return f"{p.netloc.lower()}{path}"


def clean(url: str) -> str:
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc.lower(), p.path or "/", "", "", ""))


class Host:
    """Per-host robots.txt and rate limit (>= 1 s, or the robots crawl-delay if larger)."""

    def __init__(self, session, base, verify):
        self.s, self.verify, self.last = session, verify, 0.0
        self.rp = urllib.robotparser.RobotFileParser()
        try:
            r = self.get(base + "/robots.txt")
            code = r.status_code
            # RFC 9309: 4xx = no restrictions, 5xx/unreachable = disallow all.
            self.rp.parse(r.text.splitlines() if code == 200 else ([] if code < 500 else ["User-agent: *", "Disallow: /"]))
        except requests.RequestException:
            self.rp.parse(["User-agent: *", "Disallow: /"])
        self.delay = max(1.0, float(self.rp.crawl_delay(UA) or 0))

    def get(self, url):
        wait = getattr(self, "delay", 1.0) - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        return self.s.get(url, headers={"User-Agent": UA}, timeout=60, verify=self.verify, allow_redirects=True)

    def allowed(self, url):
        return self.rp.can_fetch(UA, url)


def links(base: str, text: str) -> list[str]:
    out = [q or a or u for q, a, u in re.findall(r'href\s*=\s*(?:"([^"#\s]*)|\'([^\'#\s]*)|([^\s>"\'#]+))', text, re.I)]
    out += re.findall(r'http-equiv=["\']refresh["\'][^>]*url=([^"\'>\s]+)', text, re.I)
    res = []
    for h in out:
        if h.startswith(("mailto:", "javascript:", "tel:")):
            continue
        try:
            u = urljoin(base, h)
            urlparse(u).netloc  # malformed links (e.g. "Invalid IPv6 URL") are dropped, not fatal
        except ValueError:
            continue
        res.append(u)
    return res


def pdf_text(raw: bytes, pylib: Path) -> str:
    if str(pylib) not in sys.path:
        sys.path.insert(0, str(pylib))
    from pypdf import PdfReader
    return "\n\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(raw)).pages).strip() + "\n"


class Crawler:
    def __init__(self, a, sources):
        self.a, self.out, self.sources = a, Path(a.out), sources
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest = self.out / "manifest.jsonl"
        self.deadline = time.time() + a.hours * 3600
        self.old_urls, self.old_orig, self.seen_sha = set(), {}, {}
        for old_manifest in a.old_manifest or []:
            old_dir = Path(old_manifest).parent
            for line in open(old_manifest):
                r = json.loads(line)
                self.old_urls |= {norm(r["url"]), norm(r.get("resolved_url") or r["url"])}
                if r.get("original_path", "").endswith(".html"):  # old pages are not refetched, only re-read for links
                    self.old_orig[norm(r["url"])] = (old_dir / r["original_path"], r.get("resolved_url") or r["url"])
                for k in ("sha256", "text_sha256"):
                    if r.get(k):
                        self.seen_sha[r[k]] = "old:" + r["url"]
        self.done = {}
        if self.manifest.exists():
            for line in open(self.manifest):
                r = json.loads(line)
                self.done[norm(r["url"])] = r
                for k in ("sha256", "text_sha256"):
                    if r.get(k) and r["status"] == "downloaded":
                        self.seen_sha.setdefault(r[k], r["url"])
                state["downloaded"] += r["status"] == "downloaded"
        self.verify = str(self.out / "tools/ca_bundle.pem") if (self.out / "tools/ca_bundle.pem").exists() else True

    def write(self, row):
        with lock:
            with open(self.manifest, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.done[norm(row["url"])] = row
            state["rows"] += 1
            if row["status"] == "downloaded":
                state["downloaded"] += 1
            if state["rows"] % 250 == 0:
                summarize(self.out, self.sources)

    def fetch_one(self, src, host, url, depth):
        """Fetch, save and record one URL; returns (links to follow, fetched?)."""
        sid = src["id"]
        row = {"url": url, "source_id": sid, "topic": src["topic"], "factory_ids": [], "source_kind": src.get("kind", "technical_documentation"),
               "retrieved_at": datetime.now(timezone.utc).isoformat(), "discovered_from": "seed" if depth == 0 else "crawl_link",
               "training_status": "not_approved_for_training", "reuse_status": "unreviewed",
               "license": src["license"], "license_basis": src.get("license_basis", "")}
        if not host.allowed(url):
            self.write({**row, "status": "robots_disallowed"})
            return [], False
        try:
            r = host.get(url)
            if r.status_code in (429, 503):
                time.sleep(60)
                r = host.get(url)
        except requests.RequestException as e:
            self.write({**row, "status": "error", "error": str(e)[:300]})
            return [], True
        ct = r.headers.get("content-type", "").split(";")[0].strip()
        row.update(http_status=r.status_code, resolved_url=r.url, content_type=ct,
                   last_modified=r.headers.get("last-modified"), etag=r.headers.get("etag"))
        if r.status_code != 200:
            self.write({**row, "status": f"http_{r.status_code}"})
            return [], True
        if r.url != url and norm(r.url) in self.old_urls:
            self.write({**row, "status": "duplicate_url", "duplicate_of": "old:" + r.url})
            return [], True
        raw = r.content
        is_pdf = ct == "application/pdf" or url.lower().endswith(".pdf")
        if not is_pdf and "html" not in ct and not ct.startswith("text/"):
            self.write({**row, "status": "skipped_content_type"})
            return [], True
        sha = hashlib.sha256(raw).hexdigest()
        text = raw.decode(r.encoding or "utf-8", errors="replace") if not is_pdf else ""
        found = links(r.url, text) if "html" in ct else []
        try:
            md = pdf_text(raw, self.out / "tools/pylib") if is_pdf else (html_to_markdown(text) if "html" in ct else text)
            err = None
        except Exception as e:  # keep the original even if extraction fails
            md, err = "", f"{type(e).__name__}: {e}"[:300]
        tsha = hashlib.sha256(md.encode()).hexdigest()
        with lock:
            dup = self.seen_sha.get(sha) or (self.seen_sha.get(tsha) if len(md) > 200 else None)
            if not dup:
                self.seen_sha[sha] = url
                self.seen_sha[tsha] = url
        if dup:
            self.write({**row, "status": "duplicate_content", "sha256": sha, "duplicate_of": dup})
            return found, True
        key = hashlib.sha1(url.encode()).hexdigest()[:20]
        ext = "pdf" if is_pdf else ("html" if "html" in ct else "txt")
        orig, txt = self.out / "originals" / sid / f"{key}.{ext}", self.out / "extracted" / sid / f"{key}.md"
        orig.parent.mkdir(parents=True, exist_ok=True)
        txt.parent.mkdir(parents=True, exist_ok=True)
        orig.write_bytes(raw)
        txt.write_text(md)
        title = re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
        lang = re.search(r"<html[^>]*\blang=[\"']([\w-]+)", text, re.I)
        self.write({**row, "status": "downloaded", "original_path": str(orig.relative_to(self.out)), "sha256": sha,
                    "bytes": len(raw), "title": re.sub(r"\s+", " ", title.group(1)).strip()[:300] if title else url,
                    "language": lang.group(1) if lang else "", "license_links": sorted(set(LICENSE_RX.findall(text)))[:10],
                    "extraction_method": "pypdf" if is_pdf else ("distillkit.fetch_sources.html_to_markdown" if "html" in ct else "raw_text"),
                    "extraction_error": err, "text_path": str(txt.relative_to(self.out)), "text_sha256": tsha,
                    "characters": len(md), "words": len(md.split())})
        return found, True

    def in_scope(self, src, url):
        bare = lambda u: u.split("://", 1)[-1]  # http/https redirects stay in scope
        if not any(bare(url).startswith(bare(p)) for p in src["prefixes"]):
            return False
        path = urlparse(url).path
        if SKIP_PATH.search(url) or (SKIP_EXT.search(path) and not (src.get("pdf") and path.lower().endswith(".pdf"))):
            return False
        if src.get("exclude") and re.search(src["exclude"], url):
            return False
        return not src.get("include") or re.search(src["include"], url)

    def crawl(self, src, host):
        q, seen, fetched = deque((clean(s), 0) for s in src["seeds"]), set(), 0
        while q and fetched < src.get("max_pages", 800):
            if time.time() > self.deadline or state["downloaded"] >= self.a.max_total:
                state["stop"] = True
                return
            url, depth = q.popleft()
            k = norm(url)
            if k in seen:
                continue
            seen.add(k)
            prev, found = self.done.get(k), []
            if k in self.old_urls:  # in the old archive: follow its links from the old original, never refetch
                if k in self.old_orig and self.old_orig[k][0].exists():
                    found = links(self.old_orig[k][1], self.old_orig[k][0].read_text(errors="replace"))
            elif prev:  # resume: re-read links from the saved original instead of refetching
                if prev.get("original_path", "").endswith(".html"):
                    found = links(prev.get("resolved_url") or url, (self.out / prev["original_path"]).read_text(errors="replace"))
            else:
                found, did = self.fetch_one(src, host, url, depth)
                fetched += did
            if depth < src.get("max_depth", 99):
                for u in found:
                    u = clean(u)
                    if norm(u) not in seen and self.in_scope(src, u):
                        q.append((u, depth + 1))
        print(f"[{src['id']}] done: {fetched} fetches, queue left {len(q)}", flush=True)

    def run_host(self, netloc, srcs):
        session = requests.Session()
        host = Host(session, f"https://{netloc}", self.verify)
        for src in srcs:
            if state["stop"]:
                return
            try:
                self.crawl(src, host)
            except Exception as e:  # one broken source must not stop the host thread
                print(f"[{src['id']}] FAILED {type(e).__name__}: {e}", flush=True)
            summarize(self.out, self.sources)


def summarize(out: Path, sources):
    """Rewrite SOURCES.csv and STATUS.json from manifest.jsonl."""
    agg = defaultdict(lambda: defaultdict(int))
    status = defaultdict(lambda: defaultdict(int))
    lic_links = defaultdict(lambda: defaultdict(int))
    mf = out / "manifest.jsonl"
    if not mf.exists():
        return
    sources = list(sources)
    known = {s["id"] for s in sources}
    for line in open(mf):
        r = json.loads(line)
        if r["source_id"] not in known:  # e.g. rows merged from fetch_github_lessons.py
            known.add(r["source_id"])
            sources.append({"id": r["source_id"], "topic": r["topic"], "license": r["license"], "prefixes": [r["url"]]})
        a = agg[r["source_id"]]
        status[r["source_id"]][r["status"]] += 1
        if r["status"] == "downloaded":
            a["pages"] += 1
            a["bytes"] += r.get("bytes", 0)
            a["chars"] += r.get("characters", 0)
            for u in r.get("license_links", []):
                lic_links[r["source_id"]][u] += 1
    with open(out / "SOURCES.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "topic", "host", "pages", "bytes", "text_chars", "est_chunks_1500", "licence", "licence_links_seen", "other_statuses"])
        for s in sources:
            a = agg[s["id"]]
            other = {k: v for k, v in status[s["id"]].items() if k != "downloaded"}
            w.writerow([s["id"], s["topic"], urlparse(s["prefixes"][0]).netloc, a["pages"], a["bytes"], a["chars"],
                        round(a["chars"] / 1500), s["license"],
                        " ".join(sorted(lic_links[s["id"]], key=lic_links[s["id"]].get, reverse=True)[:3]), json.dumps(other)])
    topics = defaultdict(lambda: defaultdict(int))
    for s in sources:
        for k in ("pages", "bytes", "chars"):
            topics[s["topic"]][k] += agg[s["id"]][k]
        topics[s["topic"]]["sources"] += agg[s["id"]]["pages"] > 0
    with open(out / "TOPICS.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic", "sources", "pages", "bytes", "text_chars", "est_chunks_1500"])
        for t, a in sorted(topics.items(), key=lambda x: -x[1]["chars"]):
            w.writerow([t, a["sources"], a["pages"], a["bytes"], a["chars"], round(a["chars"] / 1500)])
    tot = {k: sum(a[k] for a in agg.values()) for k in ("pages", "bytes", "chars")}
    (out / "STATUS.json").write_text(json.dumps({"updated": datetime.now(timezone.utc).isoformat(), **tot,
                                                 "est_chunks_1500": round(tot["chars"] / 1500)}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sources")
    p.add_argument("out")
    p.add_argument("--old-manifest", action="append", help="repeatable: dedupe against several archives")
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--max-total", type=int, default=25000)
    p.add_argument("--only", default="")
    p.add_argument("--check", action="store_true")
    p.add_argument("--summarize-only", action="store_true", help="only rewrite SOURCES.csv/STATUS.json")
    a = p.parse_args()
    sources = json.load(open(a.sources))
    todo = [s for s in sources if not a.only or s["id"] in a.only.split(",")]
    by_host = defaultdict(list)  # sources keep file order (teaching first); one thread per host
    for s in todo:
        hosts = {urlparse(u).netloc for u in s["seeds"] + s["prefixes"]}
        assert len(hosts) == 1, f"{s['id']}: one host per source (rate limit is per host thread): {hosts}"
        by_host[hosts.pop()].append(s)
    if a.summarize_only:
        return summarize(Path(a.out), sources)
    c = Crawler(a, sources)
    if a.check:
        def chk(item):
            netloc, srcs = item
            h = Host(requests.Session(), f"https://{netloc}", c.verify)
            for s in srcs:
                for u in s["seeds"]:
                    try:
                        r = h.get(u)
                        print(f"{s['id']:24} {r.status_code} robots={h.allowed(u)} {len(r.content):>8} {u} -> {r.url if r.url != u else ''}", flush=True)
                    except requests.RequestException as e:
                        print(f"{s['id']:24} ERR {u} {str(e)[:120]}", flush=True)
        threads = [threading.Thread(target=chk, args=(it,)) for it in by_host.items()]
    else:
        threads = [threading.Thread(target=c.run_host, args=it) for it in by_host.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not a.check:
        summarize(c.out, sources)
        print(f"[crawl] finished: {state['downloaded']} downloaded pages in manifest", flush=True)


if __name__ == "__main__":
    main()
