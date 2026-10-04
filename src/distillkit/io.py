import gzip
import json
import threading
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class JsonlAppender:
    """Thread-safe append-and-flush writer, so a killed job (wall time, preemption) keeps its finished rows.

    A `.gz` path appends gzip members, which `read_jsonl` reads back as one stream.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def existing(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return read_jsonl(self.path)
        except (EOFError, json.JSONDecodeError):
            # last line cut off by a kill: rewrite the file without it, or the next append would glue onto it
            rows = _read_valid_prefix(self.path)
            opener = gzip.open if self.path.suffix == ".gz" else open
            with opener(self.path, "wt") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            return rows

    def append(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            if self.path.suffix == ".gz":
                with gzip.open(self.path, "at") as f:
                    f.write(line)
            else:
                with open(self.path, "a") as f:
                    f.write(line)


def _read_valid_prefix(path: Path) -> list[dict]:
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt") as f:
            for line in f:
                rows.append(json.loads(line))
    except (EOFError, json.JSONDecodeError):
        pass
    return rows
