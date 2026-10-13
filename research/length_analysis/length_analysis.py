"""Length analysis (plan section "Length analysis"): per model and test set, pooled over seeds.

Score per question = mean over both judge orders (student = A first, then B): A/T/B -> 1/0.5/0, reversed.
Key terms of a reference: backtick spans, --options, ALL_CAPS_WITH_UNDERSCORE names, Slurm command names.
Coverage = fraction of the reference key terms that occur in the answer (exact substring).
"""
import glob, json, math, re, statistics as st, sys, collections

ROOT = sys.argv[1]  # wt-eval-v3/runs
CMDS = r"\b(sbatch|srun|salloc|squeue|scontrol|sacct|scancel|sinfo|sstat|sprio|sshare|sreport|sacctmgr)\b"


def terms(ref):
    ref = re.sub(r"\n\nSource:.*", "", ref, flags=re.S)
    t = set(re.findall(r"`([^`\n]{2,60})`", ref))
    t |= set(re.findall(r"(?<![\w-])--?[a-zA-Z][\w-]*(?:=[\w:.,%-]+)?", ref))
    t |= set(re.findall(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b", ref))
    t |= set(re.findall(CMDS, ref))
    return {x.strip() for x in t if len(x.strip()) > 1}


def cov(ans, ts):
    return sum(x in ans for x in ts) / len(ts) if ts else None


def score(v):
    a, b = v
    if a not in ("A", "T", "B") or b not in ("A", "T", "B"):
        return None
    return ({"A": 1, "T": .5, "B": 0}[a] + {"B": 1, "T": .5, "A": 0}[b]) / 2


def rank(x):
    o = sorted(range(len(x)), key=x.__getitem__); r = [0.0] * len(x); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def spearman(x, y): return pearson(rank(x), rank(y))


def partial(x, y, z):  # Spearman partial corr of x,y given z
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    return (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


# Extra args NAME=RUN_DIR: seed dirs of a training run (RUN_DIR/seed_*, scored on v3; RUN_DIR_v2eval on v2).
EXTRA = dict(a.split("=", 1) for a in sys.argv[2:])
groups = collections.defaultdict(list)
dirs = glob.glob(f"{ROOT}/judge_*") + glob.glob(f"{ROOT}/eval_v3_b7/seed_*")
for name, rd in EXTRA.items():
    dirs += [(name, "v3", d) for d in glob.glob(f"{rd}/seed_*")] + [(name, "v2", d) for d in glob.glob(f"{rd}_v2eval/seed_*")]
for d in sorted(dirs, key=str):
    if isinstance(d, tuple):
        model, ts, d = d
    elif "eval_v3_b7" in d:
        model, ts = "B7 (x1.0)", "v3"
    else:
        m = re.match(r"judge_b7s\d+(ck\d+|d5)?_t([\d.]+)_(v[23])$", d.rsplit("/", 1)[1])
        if not m: continue
        model = {"": f"x{m[2]}", "d5": "D5 (DPO)"}.get(m[1] or "", f"D4 step {(m[1] or '')[2:]}")
        ts = m[3]
    try:
        rows = [json.loads(l) for l in open(f"{d}/eval_outputs.jsonl")]
    except FileNotFoundError:
        continue
    for r in rows:
        s = score(r.get("judge", {}).get("student_vs_base") or ("", ""))
        if s is None: continue
        t = terms(r["reference"])
        groups[(model, ts)].append(dict(s=s, ls=len(r["answer_student"]), lb=len(r["answer_base"]),
                                        cs=cov(r["answer_student"], t), cb=cov(r["answer_base"], t), d=d))

order = lambda k: (k[1], {"B": 9, "D": 8}.get(k[0][0], 0), k[0])
print("model | set | n | seeds | win | med len S | med len B | ratio<0.5 | 0.5-1 | >1 | rho(ratio,score) "
      "| cov S | cov B | rho(covdiff,score) | rho(ratio,score | covdiff) | rho(covdiff,score | ratio)")
for k in sorted(groups, key=order):
    g = groups[k]
    ratio = [x["ls"] / max(x["lb"], 1) for x in g]; sc = [x["s"] for x in g]
    gc = [x for x in g if x["cs"] is not None]
    cd = [x["cs"] - x["cb"] for x in gc]; scc = [x["s"] for x in gc]; rc = [x["ls"] / max(x["lb"], 1) for x in gc]
    bins = []
    for lo, hi in [(0, .5), (.5, 1), (1, 99)]:
        b = [s for s, r in zip(sc, ratio) if lo <= r < hi]
        bins.append(f"{st.mean(b):.2f} ({len(b)})" if b else "–")
    print(f"{k[0]} | {k[1]} | {len(g)} | {len({x['d'] for x in g})} | {st.mean(sc):.3f} | {st.median(x['ls'] for x in g):.0f} "
          f"| {st.median(x['lb'] for x in g):.0f} | {' | '.join(bins)} | {spearman(ratio, sc):.2f} "
          f"| {st.mean(x['cs'] for x in gc):.2f} | {st.mean(x['cb'] for x in gc):.2f} | {spearman(cd, scc):.2f} "
          f"| {partial(rc, scc, cd):.2f} | {partial(cd, scc, rc):.2f}")

# Coverage by outcome (all scaled models pooled per set)
for ts in ("v2", "v3"):
    g = [x for k, v in groups.items() if k[1] == ts and k[0].startswith("x") for x in v if x["cs"] is not None]
    for name, f in [("loses both", lambda s: s == 0), ("wins both", lambda s: s == 1)]:
        h = [x for x in g if f(x["s"])]
        if h:
            print(f"scaled {ts} {name}: n={len(h)} cov S {st.mean(x['cs'] for x in h):.2f} cov B {st.mean(x['cb'] for x in h):.2f} "
                  f"len ratio median {st.median(x['ls']/max(x['lb'],1) for x in h):.2f}")
