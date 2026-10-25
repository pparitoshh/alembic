"""Reserve original source chunks covering requested resource-option definitions."""
from pathlib import Path
import re
import sqlite3
import importlib.util

_spec = importlib.util.spec_from_file_location('command_context_v1', Path(__file__).resolve().parents[1]/'retrieval_command_aware/retrieve.py')
_command = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_command)
command_retrieve, command_names, STOP = _command.retrieve, _command.command_names, _command.STOP

FACETS = {
    'memory': (r'\b(memory|ram|[kmgt]i?b)\b', {'mem','mem-per-cpu','mem-per-gpu'}),
    'gpu': (r'\bgpus?\b', {'gpus','gpus-per-node','gpus-per-task','gres'}),
    'cpu': (r'\b(cpus?|cores?|threads?)\b', {'cpus-per-task','cpus-per-gpu','threads-per-core'}),
    'nodes': (r'\bnodes?\b', {'nodes'}),
    'tasks': (r'\b(tasks?|ranks?)\b', {'ntasks','ntasks-per-node','ntasks-per-gpu'}),
}


def definitions(text):
    # Actual option headings only; references to an option in prose are not its definition.
    return {flag for line in text.splitlines() if line.startswith('**-')
            for flag in re.findall(r'\*\*--([a-z][a-z0-9-]*)\*\*', line)}


def requested_facets(question):
    return {name for name, (pattern, _) in FACETS.items() if re.search(pattern, question, re.I)}


def resource_request(question, command):
    if command not in {'sbatch','srun','salloc'}:
        return False
    quantified=re.search(r'\b(?:\d+|one|two|three|four|eight)\s*(?:gpus?|cpus?|nodes?|tasks?|cores?|threads?|ranks?|[kmgt]i?b)\b',question,re.I)
    explicit=set(re.findall(r'--([a-z][a-z0-9-]*)',question.lower()))
    resource_options=set().union(*(flags for _,flags in FACETS.values()))
    return bool(quantified or explicit & resource_options or re.search(r'\bresource\s+(?:requests?|requirements?|directives?|allocation)\b',question,re.I))


def facet_options(question):
    options={name:set(flags) for name,(_,flags) in FACETS.items()}
    scope=re.search(r'\b(?:memory|ram|[kmgt]i?b)\b[^\n,.!?]{0,20}\bper\s+(node|cpu|gpu)\b',question,re.I)
    if scope:
        options['memory']={'node':{'mem'},'cpu':{'mem-per-cpu'},'gpu':{'mem-per-gpu'}}[scope.group(1).lower()]
    return options


def retrieve(database, question, limit=3):
    original = command_retrieve(database, question, limit)
    low = question.lower()
    commands = command_names(question)
    if len(commands) != 1 or any(site in low for site in ['lumi','leonardo','cineca','marconi']):
        return original
    # A persona mentioning GPUs is not a request for GPU allocation. Do not displace
    # dependency/mail/output documentation merely because the workload uses a GPU.
    wanted = requested_facets(question) if resource_request(question,commands[0]) else set()
    options = facet_options(question)
    explicit = set(re.findall(r'--([a-z][a-z0-9-]*)', low))
    if not wanted and not explicit:
        return original
    terms = [t for t in re.findall(r'[a-z0-9_]+', low) if len(t)>2 and t not in STOP]
    query = ' OR '.join('"'+t+'"' for t in dict.fromkeys(terms))
    url = 'https://slurm.schedmd.com/'+commands[0]+'.html'
    with sqlite3.connect('file:'+str(Path(database).resolve())+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        aliases=db.execute("SELECT 1 FROM sqlite_master WHERE name='source_aliases'").fetchone()
        if aliases:
            sql=('SELECT c.id,a.url,a.site,a.title,c.text,a.retrieved_at,bm25(search) AS rank '
                 'FROM search JOIN chunks c ON c.id=search.rowid JOIN source_aliases a ON a.chunk_id=c.id '
                 'WHERE search MATCH ? AND a.url=? AND a.site=? ORDER BY rank,c.id')
        else:
            sql=(
            'SELECT c.id,c.url,c.site,c.title,c.text,c.retrieved_at,bm25(search) AS rank '
            'FROM search JOIN chunks c ON c.id=search.rowid WHERE search MATCH ? AND c.url=? '
            'AND c.site=? ORDER BY rank,c.id')
        candidates = [dict(r) for r in db.execute(sql, (query,url,'Slurm'))]
    selected, covered, covered_explicit = [], set(), set()
    while candidates and len(selected)<limit:
        def evidence(row):
            defs = definitions(row['text'])
            facets = {name for name in wanted if defs & options[name]}
            exact = defs & explicit
            return facets, exact
        # Coverage first, then original lexical ranking. No reference answer is read.
        best = max(candidates, key=lambda r: (len(evidence(r)[1]-covered_explicit),
                                              len(evidence(r)[0]-covered), -r['rank'], -r['id']))
        facets, exact = evidence(best)
        if not (facets-covered or exact-covered_explicit):
            break
        selected.append({**best, 'selection_reason': 'requested_option_definitions',
                         'covered_facets': sorted(facets), 'explicit_option_definitions': sorted(exact)})
        covered |= facets
        covered_explicit |= exact
        candidates = [r for r in candidates if r['id'] != best['id']]
    for row in original:
        if len(selected)<limit and row['id'] not in {r['id'] for r in selected}:
            selected.append(row)
    return selected
