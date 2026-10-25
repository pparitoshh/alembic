"""Quality-gated data chain pieces: source fetch/convert, review stage, decontam stage.

Hermetic: fetching uses an injected fake, the reviewer is a fake with synthetic verdicts, and the
held-out files are written to tmp_path; no network, model or GPU.
"""
import json
from pathlib import Path

from distillkit import fetch_sources, heldout, review, verify
from distillkit.io import read_jsonl, write_jsonl
from distillkit.teacher import Completion
from test_support_review import cfg, FakeReviewer  # noqa: F401  (cfg is a fixture)
from test_support_review_v3 import SYNTHETIC_SOURCE, payload, prose_record

ROOT = Path(__file__).resolve().parents[1]
PATTERN = json.loads((ROOT / "data/sources/hpc10k_sources.json").read_text())["heldout_pattern"]

ROFF = r'''.TH sbatch "1" "Slurm Commands"
.\" comment line
.SH "NAME"
sbatch \- Submit a batch script to Slurm.
.SH "OPTIONS"
.TP
\fB\-a\fR, \fB\-\-array\fR=<\fIindexes\fR>
Submit a job array.
.IP
A second paragraph of the array option.
.IP

.TP
\fB\-c\fR, \fB\-\-cpus\-per\-task\fR=<\fIncpus\fR>
Request ncpus per task.
.IP
.PP
.nf
sbatch \-\-cpus\-per\-task=4 job.sh
.fi
'''

HTML = '''<html><body><nav>Menu sbatch srun</nav>
<div class="content" role="main"><h1>Quick Start</h1>
<p>Slurm allocates <code>nodes</code> to jobs.</p>
<dl><dt>--requeue</dt><dd>Requeue the job.</dd><dt>--time</dt><dd>Set a limit.</dd></dl>
<pre>#SBATCH --time=01:00:00
srun hostname</pre></div><footer>Legal</footer></body></html>'''


def test_roff_options_become_whole_blocks_and_heldout_entries_drop_entirely():
    md = fetch_sources.roff_to_markdown(ROFF)
    assert "## Name" in md and "comment line" not in md and "\\f" not in md
    assert "**-c, --cpus-per-task=<ncpus>**\nRequest ncpus per task." in md
    assert "```\nsbatch --cpus-per-task=4 job.sh\n```" in md
    kept, dropped = fetch_sources.drop_heldout(md, PATTERN)
    assert dropped == 1 and "array" not in kept.lower()  # both paragraphs of --array went with it
    assert "--cpus-per-task" in kept


def test_html_keeps_main_region_code_and_definition_blocks():
    md = fetch_sources.html_to_markdown(HTML)
    assert "Menu" not in md and "Legal" not in md
    assert "`nodes`" in md and "```\n#SBATCH --time=01:00:00\nsrun hostname\n```" in md
    kept, dropped = fetch_sources.drop_heldout(md, PATTERN)
    assert dropped == 1 and "requeue" not in kept.lower() and "**--time**\nSet a limit." in kept


def test_fetch_run_writes_seeds_and_manifest(tmp_path):
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"version": "hpc-sources-v1", "heldout_pattern": PATTERN, "sources": [
        {"doc_id": "man_sbatch", "url": "https://example.invalid/sbatch.1", "format": "roff", "license": "GPL-2.0"},
        {"doc_id": "guide_quick", "url": "https://example.invalid/quick.html", "format": "html", "license": "GPL-2.0"},
        {"doc_id": "not_fetched", "url": "https://example.invalid/x.html", "format": "html", "license": "GPL-2.0"}]}))
    pages = {"https://example.invalid/sbatch.1": ROFF.encode(), "https://example.invalid/quick.html": HTML.encode()}
    manifest = fetch_sources.run(sources, tmp_path / "seeds", limit=2, fetch=pages.__getitem__)
    assert [m["doc_id"] for m in manifest] == ["man_sbatch", "guide_quick"]
    assert {p.name for p in (tmp_path / "seeds").iterdir()} == {"man_sbatch.md", "guide_quick.md", "sources_manifest.json"}
    saved = json.loads((tmp_path / "seeds/sources_manifest.json").read_text())
    assert saved["documents"][0]["dropped_heldout_blocks"] == 1 and saved["documents"][0]["license"] == "GPL-2.0"


def test_review_writes_reports_resumes_and_feeds_verify(cfg):  # noqa: F811
    cfg.verify.grounding_review_protocol = "source-support-v3"
    row = prose_record(cfg, "A count alone does not establish a failure cause.")
    write_jsonl(cfg.run_dir / "generated.jsonl", [row])
    reviewer = FakeReviewer(payload(row))
    out = review.run(cfg, reviewer=reviewer)
    assert out == cfg.verify.grounding_reviews and len(reviewer.requests) == 1
    assert [r["id"] for r in read_jsonl(out)] == [row["id"]]
    review.run(cfg, reviewer=reviewer)  # rerun: already reviewed, no new request
    assert len(reviewer.requests) == 1 and len(read_jsonl(out)) == 1
    verify.run(cfg)  # the stage's reports are what verify's support gate consumes
    kept = read_jsonl(cfg.run_dir / "verified.jsonl")
    assert [r["grounding_checks"]["status"] for r in kept] == ["supported"]
    assert read_jsonl(cfg.run_dir / "rejected.jsonl") == read_jsonl(cfg.run_dir / "pending_review.jsonl") == []


def test_review_redoes_malformed_reviews(cfg):  # noqa: F811
    cfg.verify.grounding_review_protocol = "source-support-v3"
    row = prose_record(cfg, "A count alone does not establish a failure cause.")
    write_jsonl(cfg.run_dir / "generated.jsonl", [row])
    truncated = FakeReviewer(None)
    truncated.complete = lambda messages, **kw: (truncated.requests.append(1), Completion('{"units": [{"unit_id": "u'))[1]
    review.run(cfg, reviewer=truncated)
    assert [r["review"] for r in read_jsonl(cfg.verify.grounding_reviews)] == [None]
    good = FakeReviewer(payload(row))
    review.run(cfg, reviewer=good)  # the malformed report is replaced, not duplicated
    reports = read_jsonl(cfg.verify.grounding_reviews)
    assert len(good.requests) == 1 and len(reports) == 1 and reports[0]["review"] is not None


def test_review_skips_unresolvable_sources_for_verify_to_hold(cfg, tmp_path):  # noqa: F811
    cfg.verify.grounding_review_protocol = "source-support-v3"
    row = prose_record(cfg, "A count alone does not establish a failure cause.")
    row = {**row, "source_sha256": "0" * 64}  # source changed since generation
    write_jsonl(cfg.run_dir / "generated.jsonl", [row])
    reviewer = FakeReviewer(payload(prose_record(cfg, "x")))
    review.run(cfg, reviewer=reviewer)
    assert reviewer.requests == [] and read_jsonl(cfg.verify.grounding_reviews) == []


def test_decontam_drops_overlap_with_any_heldout_file(cfg, tmp_path):  # noqa: F811
    root = tmp_path / "eval"
    (root / "dataset_v1").mkdir(parents=True)
    write_jsonl(root / "dataset_v1/final_normal.jsonl",
                [{"question": "How do I request two GPUs for one task with sbatch on the cluster?"}])
    write_jsonl(root / "tools.jsonl", [{"id": "t1"}])  # rows without a question are ignored
    write_jsonl(cfg.run_dir / "verified.jsonl", [
        {"id": "a", "question": "How do I request two GPUs for one task with sbatch on the cluster?"},
        {"id": "b", "question": "What does sinfo show about partitions?"}])
    heldout.run(cfg, root=root)
    assert [r["id"] for r in read_jsonl(cfg.run_dir / "clean.jsonl")] == ["b"]
    dirty = read_jsonl(cfg.run_dir / "contaminated.jsonl")
    assert dirty[0]["contaminated_with"]["file"] == "dataset_v1/final_normal.jsonl"


def test_repo_heldout_questions_cover_dev_and_final():
    files = {f for f, _, _ in heldout.heldout_questions()}
    assert {"eval_all_v2.jsonl", "dataset_v1/development_normal.jsonl", "dataset_v1/final_normal.jsonl"} <= files


def test_source_list_is_pinned_and_excludes_heldout_pages():
    spec = json.loads((ROOT / "data/sources/hpc10k_sources.json").read_text())
    urls = [s["url"] for s in spec["sources"]]
    assert len({s["doc_id"] for s in spec["sources"]}) == len(urls)
    assert all(s["license"] in {"GPL-2.0", "CC-BY-4.0"} for s in spec["sources"])
    assert all("/slurm-26-05-4-1/" in u for u in urls if "SchedMD" in u)
    assert not any(k in u for u in urls for k in ("job_array", "preempt", "21-array", "aalto"))
