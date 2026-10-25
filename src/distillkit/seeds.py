"""Load seed documents, chunk them, and split train/eval by document."""

from .config import Config
from .source_registry import admitted_sources


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Greedy paragraph packing so chunks never split a paragraph or code block."""
    blocks, buf, in_code = [], [], False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        buf.append(line)
        if not line.strip() and not in_code:
            blocks.append("\n".join(buf).strip())
            buf = []
    blocks.append("\n".join(buf).strip())

    chunks, cur = [], ""
    for b in filter(None, blocks):
        if cur and len(cur) + len(b) > max_chars:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n\n{b}" if cur else b
    if cur:
        chunks.append(cur)
    return chunks


def load_chunks(cfg: Config) -> tuple[list[dict], list[dict]]:
    """Return (train_chunks, eval_chunks). Split is by document, never by chunk."""
    scfg = cfg.seeds
    admitted = admitted_sources(cfg)
    eval_docs = set(scfg.eval_docs)
    train, held_out = [], []
    for path in sorted(scfg.dir.glob("*.md")):
        doc_id = path.stem
        for i, text in enumerate(chunk_text(path.read_text(), scfg.chunk_chars)):
            chunk = {"doc_id": doc_id, "chunk_id": f"{doc_id}#{i}", "text": text}
            if admitted is not None and doc_id not in eval_docs:
                chunk.update(admitted[doc_id])
            (held_out if doc_id in eval_docs else train).append(chunk)
    return train, held_out
