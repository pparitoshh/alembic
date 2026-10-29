"""Simon's #115 test for two models: each answers the Leonardo mentor cases without the app (system prompt + question,
no docs) and in the app (docs index + notes); answers go to ANSWERS_DIR/<label>_{bare,app}.jsonl for score_mentor.py.
usage: mentor_models.py CASES ANSWERS_DIR LABEL=URL [LABEL=URL ...]   (each URL an OpenAI endpoint serving 'cand')"""
import concurrent.futures as cf, io, json, os, sys, urllib.request
sys.path.insert(0, os.path.expanduser('~/cp/product/src'))
from distillkit.assistant import SYSTEM_PROMPT, Assistant  # noqa: E402
INDEX = os.path.expanduser('~/cp/product_index')
cases = [json.loads(l) for l in open(sys.argv[1])]
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
def question(c):
    return c['user_request'] + ('\n\nContext: ' + c['context'] if c.get('context') else '')
def bare(url, q):
    body = {'model': 'cand', 'temperature': 0, 'max_tokens': 3000,
            'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': q}]}
    r = urllib.request.Request(url + '/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=900))['choices'][0]['message']['content']
def app(url, q):
    ans, _ = Assistant(INDEX, model='cand', openai=url).ask(q, out=io.StringIO())
    return ans
for spec in sys.argv[3:]:
    label, url = spec.split('=', 1)
    for mode, fn in (('bare', bare), ('app', app)):
        if os.path.exists(f'{out}/{label}_{mode}.jsonl'):  # made earlier (e.g. by a side job): keep it
            print(label, mode, 'exists', flush=True)
            continue
        with cf.ThreadPoolExecutor(8) as ex:
            answers = list(ex.map(lambda c: fn(url, question(c)), cases))
        with open(f'{out}/{label}_{mode}.jsonl', 'w') as f:
            for c, a in zip(cases, answers):
                f.write(json.dumps({'id': c['id'], 'answer': a}) + '\n')
        print(label, mode, 'done', flush=True)
