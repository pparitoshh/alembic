"""Keep prompts whose question is not already in another prompt file.  usage: drop_seen.py NEW.jsonl SEEN.jsonl OUT.jsonl"""
import json, sys
seen = {json.loads(l)['question'] for l in open(sys.argv[2])}
kept = [l for l in open(sys.argv[1]) if json.loads(l)['question'] not in seen]
open(sys.argv[3], 'w').writelines(kept)
print('DROP SEEN: kept', len(kept), 'new prompts')
