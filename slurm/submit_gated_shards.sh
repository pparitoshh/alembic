#!/bin/bash
# Split the fetched seeds into $SHARDS shards of similar size and submit one slurm/generate_gated.sbatch
# job per shard ($RUN_PREFIX1 ... $RUN_PREFIXN), so the chain runs on N GPUs in parallel.
# Rerunning reuses existing shard seed dirs and resubmits every shard (generate/review resume), so it
# also restarts failed shards. Merge the shards' clean.jsonl afterwards (cross-shard duplicates).
#
# Login node, from $REPO:
#   source $WORK/alembic/me.env; cd $REPO
#   export SBATCH_ACCOUNT=EUHPC_D30_031 CONFIG=configs/gen_hpc10k.yaml; source slurm/env.sh
#   SHARDS=5 SEEDS_ALL=runs/hpc10k/seeds RUN_PREFIX=runs/hpc10k_s bash slurm/submit_gated_shards.sh [--dry-run]
set -euo pipefail
N="${SHARDS:-5}"
ALL="${SEEDS_ALL:-runs/hpc10k/seeds}"
PREFIX="${RUN_PREFIX:-runs/hpc10k_s}"
: "${CONFIG:?export CONFIG (e.g. configs/gen_hpc10k.yaml)}" "${VENV:?source slurm/env.sh first}"

"$VENV/bin/python" - "$ALL" "$PREFIX" "$N" <<'EOF'
import json, shutil, sys
from pathlib import Path

all_dir, prefix, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
manifest = json.loads((all_dir / "sources_manifest.json").read_text())
shards = [[] for _ in range(n)]
for doc in sorted(manifest["documents"], key=lambda d: -d["chars"]):  # largest first, into the smallest shard
    min(shards, key=lambda s: sum(d["chars"] for d in s)).append(doc)
for i, docs in enumerate(shards, 1):
    seeds = Path(f"{prefix}{i}/seeds")
    if (seeds / "sources_manifest.json").exists():
        print(f"shard {i}: reusing {seeds}")
        continue
    seeds.mkdir(parents=True)
    for doc in docs:
        shutil.copy(all_dir / f"{doc['doc_id']}.md", seeds)
    (seeds / "sources_manifest.json").write_text(json.dumps(
        {**manifest, "documents": docs, "shard": f"{i}/{n}", "shard_of": str(all_dir)}, indent=2) + "\n")
    print(f"shard {i}: {len(docs)} docs, {sum(d['chars'] for d in docs)} chars -> {seeds}")
EOF

[ "${1:-}" = "--dry-run" ] && { echo "dry run: nothing submitted"; exit 0; }
for i in $(seq 1 "$N"); do
    echo "shard $i: job $(RUN_DIR="$PREFIX$i" sbatch --parsable --time=24:00:00 slurm/generate_gated.sbatch)"
done
