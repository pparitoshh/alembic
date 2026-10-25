"""Explicit batch-local source reuse, with byte checks before results are accepted.

No process-global cache or mtime trust. A caller must call assert_unchanged before
persisting/accepting batch results; source_for_records does so before returning.
"""
import copy
import hashlib
import io
import json
from pathlib import Path

from .source_registry import _admitted_snapshot, _record_binding
from .seeds import chunk_text


def _config_bytes(cfg):
    return json.dumps(cfg.model_dump(mode='json'), sort_keys=True,
                      ensure_ascii=False, separators=(',', ':')).encode()


def _text(raw):
    # Match Path.read_text()'s encoding and universal-newline behavior, including
    # CRLF sources whose raw registry hash differs from their normalized text hash.
    with io.TextIOWrapper(io.BytesIO(raw)) as stream:
        return stream.read()


class SourceResolver:
    """One immutable in-memory source view for one resolved Config and batch.

    Config has no backing-file field. Its effective serialized values are bound;
    callers that also own an original config file may bind config_path explicitly.
    """
    def __init__(self, cfg, *, config_path=None):
        self._cfg = cfg
        self._config = _config_bytes(cfg)
        self._config_path = Path(config_path) if config_path is not None else None
        self._config_file = self._config_path.read_bytes() if self._config_path else None
        self._registry_path = Path(cfg.seeds.registry) if cfg.seeds.registry is not None else None
        self._registry = self._registry_path.read_bytes() if self._registry_path else None
        self._paths = tuple(sorted(cfg.seeds.dir.glob('*.md')))
        self._resolved_paths = tuple(p.resolve() for p in self._paths)
        self._raw = {p.stem: p.read_bytes() for p in self._paths}
        self._hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in self._raw.items()}
        self._selected = _admitted_snapshot(cfg, self._registry, [(p, self._raw[p.stem]) for p in self._paths])
        self._texts = {}
        self._chunks = None
        self._train_chunks = None

    def check_config(self, cfg=None):
        if _config_bytes(self._cfg) != self._config or (cfg is not None and _config_bytes(cfg) != self._config):
            raise ValueError('source snapshot effective configuration changed or differs')

    @property
    def admitted(self):
        self.check_config()
        return copy.deepcopy(self._selected)

    def check_record(self, row):
        self.check_config()
        _record_binding(self._selected, row)

    def _document(self, doc_id):
        if doc_id not in self._raw:
            raise ValueError('training source document is unresolved')
        if doc_id not in self._texts:
            self._texts[doc_id] = _text(self._raw[doc_id])
        return self._texts[doc_id]

    def load_chunks(self):
        self.check_config()
        if self._chunks is None:
            train, held = [], []
            for path in self._paths:
                name = path.stem
                for i, text in enumerate(chunk_text(self._document(name), self._cfg.seeds.chunk_chars)):
                    chunk = {'doc_id': name, 'chunk_id': f'{name}#{i}', 'text': text}
                    if self._selected is not None and name not in self._cfg.seeds.eval_docs:
                        chunk.update(copy.deepcopy(self._selected[name]))
                    (held if name in self._cfg.seeds.eval_docs else train).append(chunk)
            self._chunks = (train, held)
            self._train_chunks = {(c['doc_id'], c['chunk_id']): c['text'] for c in train}
        return copy.deepcopy(self._chunks)

    def source_for_record(self, row):
        self.check_record(row)
        if row.get('doc_id') in self._cfg.seeds.eval_docs:
            raise ValueError('evaluation-only source cannot support a training record')
        if row.get('source_kind') == 'chunk':
            if self._chunks is None:
                self.load_chunks()
            key = (row.get('doc_id'), row.get('chunk_id'))
            if key not in self._train_chunks:
                raise ValueError('training source chunk is unresolved')
            source = self._train_chunks[key]
        elif row.get('source_kind', 'full') == 'full':
            source = self._document(row.get('doc_id'))
        else:
            raise ValueError('unknown source evidence kind')
        if hashlib.sha256(source.encode()).hexdigest() != row.get('source_sha256'):
            raise ValueError('source hash missing or different from generation evidence')
        return source

    def assert_unchanged(self):
        """Rehash actual inputs, including the selected file set, before publication."""
        self.check_config()
        paths = tuple(sorted(self._cfg.seeds.dir.glob('*.md')))
        if paths != self._paths or tuple(p.resolve() for p in paths) != self._resolved_paths:
            raise ValueError('source snapshot selected files or paths changed')
        for path in paths:
            if hashlib.sha256(path.read_bytes()).hexdigest() != self._hashes[path.stem]:
                raise ValueError(f'source snapshot document changed: {path.stem}')
        if self._registry_path and self._registry_path.read_bytes() != self._registry:
            raise ValueError('source snapshot registry changed')
        if self._config_path and self._config_path.read_bytes() != self._config_file:
            raise ValueError('source snapshot configuration file changed')
