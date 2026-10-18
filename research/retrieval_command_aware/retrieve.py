"""Unsubmitted retrieval variant: reserve context for explicitly named Slurm commands."""
from pathlib import Path
import re
import sqlite3

COMMANDS = {'sbatch', 'srun', 'salloc', 'sacct', 'scontrol', 'squeue', 'scancel', 'sinfo'}
STOP = {'the','and','for','this','that','with','from','does','what','how','can','are','you','your','want','have','has','job','slurm','when','which','would','should'}


def command_names(question):
    return list(dict.fromkeys(t for t in re.findall(r'\b[a-z]+\b', question.lower()) if t in COMMANDS))


def retrieve(database, question, limit=3):
    if not 1 <= limit <= 3:
        raise ValueError('This comparison keeps the original at-most-three-source budget')
    low = question.lower()
    site = 'LUMI' if 'lumi' in low else 'CINECA' if any(s in low for s in ['leonardo','cineca','marconi']) else 'Slurm'
    terms = [t for t in re.findall(r'[a-z0-9_]+', low) if len(t)>2 and t not in STOP]
    query = ' OR '.join('"'+t+'"' for t in dict.fromkeys(terms))
    if not query:
        return []
    db = sqlite3.connect('file:'+str(Path(database).resolve())+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    sql = ('SELECT c.id,c.url,c.site,c.title,c.text,c.retrieved_at,bm25(search) AS rank '
           'FROM search JOIN chunks c ON c.id=search.rowid WHERE search MATCH ? AND c.site=? ')
    def search(url=None, n=limit):
        params = [query, site]
        statement = sql
        if url:
            statement += 'AND c.url=? '
            params.append(url)
        return [dict(r) for r in db.execute(statement+'ORDER BY rank,c.id LIMIT ?', [*params,n])]
    try:
        general = search(n=limit*3)
        selected = []
        # Do not mix generic defaults into explicitly site-specific queries in this first variant.
        commands = command_names(question) if site == 'Slurm' else []
        per_command = 2 if len(commands) == 1 else 1
        for cmd in commands[:limit]:
            for result in search('https://slurm.schedmd.com/'+cmd+'.html', per_command):
                if result['id'] not in {r['id'] for r in selected} and len(selected)<limit:
                    selected.append({**result, 'selection_reason': 'explicit_command:'+cmd})
        for result in general:
            if result['id'] not in {r['id'] for r in selected} and len(selected)<limit:
                selected.append({**result, 'selection_reason': 'same_site_bm25'})
        return selected
    finally:
        db.close()
