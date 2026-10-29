"""Blind key-point scoring of the 16 mentor cases (as in issue #115): for each case and system, Qwen3-235B gets the
question, the expected answer (key points) and the must_not list, and one answer without its system name.
Score 0-2 per case (0 = misses the key points, 1 = some, 2 = covers them), plus must_not violations.
usage: score_mentor.py CASES.jsonl OUT.json name=answers.jsonl [...]   (judge on :8103)"""
import concurrent.futures as cf, json, random, re, sys, urllib.request
cases = {json.loads(l)['id']: json.loads(l) for l in open(sys.argv[1])}
systems = {a.split('=')[0]: {json.loads(l)['id']: json.loads(l)['answer'] for l in open(a.split('=')[1])} for a in sys.argv[3:]}
PROMPT = """You score an answer of an HPC assistant against an expert's expected answer. Do not reward length or style.
Question:
{q}

Expected answer (key points):
{e}

Must not (each is a violation if the answer does it):
{m}

Answer to score:
{a}

Score key points: 2 = covers the essential key points correctly, 1 = some key points or partly wrong, 0 = misses them
or is wrong. Then list the must_not items the answer violates. Return JSON only: {{"score": 0|1|2, "violations": [..]}}"""
def judge(item):
    sysname, cid = item
    c = cases[cid]
    body = {'model': 'qwen3-235b', 'temperature': 0, 'max_tokens': 400, 'messages': [{'role': 'user', 'content': PROMPT.format(
        q=c['user_request'] + ('\nContext: ' + c['context'] if c.get('context') else ''), e=c['expected_answer'],
        m='\n'.join('- ' + x for x in c.get('must_not', [])) or '(none)', a=systems[sysname][cid][:9000])}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8103/v1/chat/completions',
        json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=900))
    text = r['choices'][0]['message']['content']
    m = re.search(r'\{.*\}', text, re.S)
    try:
        v = json.loads(m.group(0))
    except Exception:
        v = {'score': None, 'violations': [], 'raw': text[:200]}
    return sysname, cid, v
items = [(s, c) for s in systems for c in cases]
random.Random(42).shuffle(items)  # blind order
res = {}
with cf.ThreadPoolExecutor(8) as ex:
    for s, c, v in ex.map(judge, items):
        res.setdefault(s, {})[c] = v
json.dump(res, open(sys.argv[2], 'w'), indent=1)
for s, r in res.items():
    sc = [v['score'] for v in r.values() if v['score'] is not None]
    print(f"{s:14s} key points {sum(sc)}/{2*len(sc)}  zero-score cases {sum(1 for x in sc if x == 0)}  must_not violations {sum(len(v.get('violations') or []) for v in r.values())}")
