"""Closed-book set (issues #115, #109, #111, #112): keep prompts that are not near a mentor use case (3-gram Jaccard
>= 0.5 with any user_request or follow_up, the same rule as the eval exclusion) and add the bare question as the
student's prompt. The teacher still answers with the app's excerpts (user), the student learns that answer from the
question alone (student_user).   usage: cb_filter.py RAW.jsonl MENTOR.jsonl OUT.jsonl"""
import json, re, sys
def grams(q):
    w = re.findall(r'[a-z0-9]+', q.lower())
    return {' '.join(w[i:i + 3]) for i in range(len(w) - 2)}
mentor = []
for line in open(sys.argv[2]):
    c = json.loads(line)
    mentor += [grams(c[k]) for k in ('user_request', 'follow_up') if c.get(k)]
kept = dropped = 0
with open(sys.argv[3], 'w') as out:
    for line in open(sys.argv[1]):
        r = json.loads(line)
        g = grams(r['question'])
        if any(len(g & m) / max(1, len(g | m)) >= 0.5 for m in mentor):
            dropped += 1
            continue
        out.write(json.dumps({**r, 'student_user': r['question']}, ensure_ascii=False) + '\n')
        kept += 1
print(f'CB FILTER: kept {kept}, dropped {dropped} near a mentor case')
