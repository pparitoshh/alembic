"""Add the ClusterPilot notes (data/clusterpilot_notes/*.md: short, checked recipes for topics the model gets wrong)
to a documents.sqlite index as Slurm excerpts, so the assistant retrieves and cites them. Safe to run again: earlier
copies of the notes are replaced.

usage: python scripts/add_notes_to_index.py INDEX/documents.sqlite [NOTES_DIR]
"""
import sqlite3
import sys
import time
from pathlib import Path

URL = 'https://huggingface.co/VaibhavKKM/hpc-assistant-qwen3-4b/blob/main/index/notes/'


def add_notes(db_path, notes_dir):
    db = sqlite3.connect(db_path)
    db.execute('DELETE FROM chunks WHERE url LIKE ?', (URL + '%',))
    columns = {r[1] for r in db.execute('PRAGMA table_info(chunks)')}
    for f in sorted(Path(notes_dir).glob('*.md')):
        text = f.read_text()
        title = 'ClusterPilot notes: ' + text.splitlines()[0].lstrip('# ').strip()
        row = {'url': URL + f.name, 'site': 'Slurm', 'title': title, 'text': text, 'retrieved_at': time.strftime('%Y-%m-%d')}
        row = {k: v for k, v in row.items() if k in columns}  # older indexes have no retrieved_at
        db.execute(f'INSERT INTO chunks ({", ".join(row)}) VALUES ({", ".join("?" * len(row))})', list(row.values()))
    db.execute("INSERT INTO search(search) VALUES ('rebuild')")
    db.commit()
    n = db.execute('SELECT count(*) FROM chunks WHERE url LIKE ?', (URL + '%',)).fetchone()[0]
    db.close()
    return n


if __name__ == '__main__':
    notes = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / 'data/clusterpilot_notes'
    print(f'{add_notes(sys.argv[1], notes)} notes in {sys.argv[1]}')
