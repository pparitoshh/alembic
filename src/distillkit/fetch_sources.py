"""Fetch pinned public HPC documentation into markdown seed documents for `generate`.

Runs on a login node (network). Each source in the list (e.g. data/sources/hpc10k_sources.json) is
downloaded, converted (roff man page or HTML), and every block matching the list's `heldout_pattern`
(topics the eval sets test, e.g. job arrays and requeue) is dropped. Writes <out>/<doc_id>.md plus
<out>/sources_manifest.json with URL, licence, fetched-bytes and seed SHA-256 and dropped-block counts.

    python -m distillkit.fetch_sources data/sources/hpc10k_sources.json runs/hpc10k/seeds [--limit N]
"""
import argparse
import hashlib
import html
import json
import re
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

FENCE = "```"


def blocks(text: str) -> list[str]:
    """Blank-line separated blocks; a fenced code block is never split."""
    out, buf, in_code = [], [], False
    for line in text.splitlines():
        if line.startswith(FENCE):
            in_code = not in_code
        if not line.strip() and not in_code:
            if buf:
                out.append("\n".join(buf).strip())
            buf = []
        else:
            buf.append(line)
    if buf:
        out.append("\n".join(buf).strip())
    return [b for b in out if b]


def drop_heldout(text: str, pattern: str) -> tuple[str, int]:
    """Remove every block matching `pattern`; returns (text, number of dropped blocks)."""
    rx = re.compile(pattern)
    kept = [b for b in blocks(text) if not rx.search(b)]
    return "\n\n".join(kept) + "\n", len(blocks(text)) - len(kept)


_ROFF_ESCAPES = [(r"\-", "-"), (r"\(em", "—"), (r"\(en", "–"), (r"\(aq", "'"), (r"\(dq", '"'),
                 (r"\(bu", "-"), (r"\&", ""), (r"\e", "\\"), (r"\ ", " ")]


def _roff_inline(line: str) -> str:
    line = re.sub(r"\\f(\[[A-Z]*\]|[BIRP])", "", line)  # font changes
    for a, b in _ROFF_ESCAPES:
        line = line.replace(a, b)
    return line


def roff_to_markdown(text: str) -> str:
    """Minimal man(7) conversion: sections become headings, each .TP option entry (term + all its
    paragraphs) one block, .nf/.fi and .EX/.EE fenced code."""
    out, in_code, in_entry, want_term = [], False, False, False
    for raw in text.splitlines():
        if raw.startswith(('.\\"', "'\\\"")):
            continue
        if raw.startswith((".nf", ".EX")):
            out += ["", FENCE]; in_code = True; continue
        if raw.startswith((".fi", ".EE")):
            out += [FENCE, ""]; in_code = False; continue
        if in_code:
            out.append(_roff_inline(raw)); continue
        m = re.match(r'^\.(S[HS])\s*"?(.*?)"?\s*$', raw)
        if m:
            out += ["", ("## " if m.group(1) == "SH" else "### ") + _roff_inline(m.group(2)).title(), ""]
            in_entry = False; continue
        if raw.startswith(".TP"):
            out.append(""); in_entry = True; want_term = True; continue
        if re.match(r"^\.(PP|LP|P)\b", raw):  # ends an option entry
            out.append(""); in_entry = False; continue
        if re.match(r"^\.(IP|sp|br)\b", raw) or raw.strip() in ("", "."):
            if not in_entry:  # inside an entry its paragraphs stay one block (dropped as a whole)
                out.append("")
            continue
        if raw.startswith((".TH", ".RS", ".RE", ".ad", ".na", ".hy", ".nh", ".in", ".ti", ".ft", ".ps", ".ne")):
            continue
        m = re.match(r"^\.(B|I|BR|IR|RB|RI|BI|IB)\s+(.*)$", raw)
        line = _roff_inline(" ".join(re.findall(r'"[^"]*"|\S+', m.group(2))).replace('"', "") if m else raw)
        if want_term:
            out.append(f"**{line.strip()}**"); want_term = False
        else:
            out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


class _HTML(HTMLParser):
    BLOCK = {"p", "div", "section", "article", "ul", "ol", "table", "tr", "blockquote", "dl", "br", "hr"}
    SKIP = {"script", "style", "nav", "header", "footer", "svg", "button", "form"}

    def __init__(self, main_only: bool):
        super().__init__(convert_charrefs=True)
        self.parts, self.depth_main, self.skip, self.pre, self.main_only = [], 0, 0, 0, main_only

    def _on(self) -> bool:
        return (self.depth_main > 0 or not self.main_only) and not self.skip

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self.depth_main:
            self.depth_main += 1
        elif self.main_only and (tag == "main" or a.get("role") == "main"):
            self.depth_main = 1
        if tag in self.SKIP:
            self.skip += 1
        if not self._on():
            return
        if tag == "pre":
            self.pre += 1; self.parts.append(f"\n\n{FENCE}\n")
        elif tag == "code" and not self.pre:
            self.parts.append("`")
        elif re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n" + "#" * min(int(tag[1]) + 1, 4) + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "dt":
            self.parts.append("\n\n**")
        elif tag == "dd":
            self.parts.append("\n")
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        on = self._on()
        if on:
            if tag == "pre" and self.pre:
                self.pre -= 1; self.parts.append(f"\n{FENCE}\n\n")
            elif tag == "code" and not self.pre:
                self.parts.append("`")
            elif tag == "dt":
                self.parts.append("**")
            elif re.fullmatch(r"h[1-6]", tag) or tag in self.BLOCK:
                self.parts.append("\n\n")
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if self.depth_main:
            self.depth_main -= 1

    def handle_startendtag(self, tag, attrs):
        if tag == "br" and self._on():
            self.parts.append("\n")

    def handle_data(self, data):
        if self._on():
            self.parts.append(data if self.pre else re.sub(r"\s+", " ", data))


def html_to_markdown(text: str) -> str:
    """Text of the page's main region (<main> or role="main"), or the whole page if it has none."""
    main_only = bool(re.search(r'<main\b|role="main"', text))
    p = _HTML(main_only)
    p.feed(text)
    md = html.unescape("".join(p.parts))
    lines, in_code = [], False
    for line in md.splitlines():
        if line.startswith(FENCE):
            in_code = not in_code
        lines.append(line if in_code else line.strip())
    md = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    md = re.sub(r"\*\*\s*\*\*", "", md)  # empty <dt>
    return md.strip() + "\n"


def convert(raw: bytes, fmt: str) -> str:
    text = raw.decode("utf-8", errors="replace")
    if fmt == "roff":
        return roff_to_markdown(text)
    if fmt == "html":
        return html_to_markdown(text)
    raise ValueError(f"unknown source format {fmt!r}")


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "distillkit-fetch-sources/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def run(sources_file: Path, out_dir: Path, limit: int | None = None, fetch=_fetch) -> list[dict]:
    spec = json.loads(sources_file.read_text())
    if spec.get("version") != "hpc-sources-v1":
        raise ValueError("unknown sources list version")
    pattern = spec["heldout_pattern"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for s in spec["sources"][:limit]:
        raw = fetch(s["url"])
        md, dropped = drop_heldout(convert(raw, s["format"]), pattern)
        path = out_dir / f"{s['doc_id']}.md"
        path.write_text(md)
        manifest.append({**s, "fetched_utc": datetime.now(timezone.utc).isoformat(),
                         "fetched_sha256": hashlib.sha256(raw).hexdigest(),
                         "seed_sha256": hashlib.sha256(md.encode()).hexdigest(),
                         "chars": len(md), "dropped_heldout_blocks": dropped})
        print(f"[fetch] {s['doc_id']}: {len(md)} chars, dropped {dropped} held-out blocks")
    (out_dir / "sources_manifest.json").write_text(json.dumps(
        {"sources_list": str(sources_file), "sources_list_sha256": hashlib.sha256(sources_file.read_bytes()).hexdigest(),
         "heldout_pattern": pattern, "documents": manifest}, indent=2) + "\n")
    print(f"[fetch] {len(manifest)} documents, {sum(m['chars'] for m in manifest)} chars -> {out_dir}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sources", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--limit", type=int, default=None, help="only the first N sources (pilot)")
    a = p.parse_args()
    run(a.sources, a.out, a.limit)


if __name__ == "__main__":
    main()
