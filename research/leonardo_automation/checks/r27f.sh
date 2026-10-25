#!/bin/bash
# Variant f test (Oct 25, d1 push): round 27 with d excerpts (general documentation index for non-Slurm questions) under the relevance header, vs round 27 + e at 1536.
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
ANSWER_MAX_TOKENS=1536 VARIANT=f bash $R/ctl/ragc.sh r27f /leonardo_work/EUHPC_D30_031/shared_split/models/r27_merged
REL_ANSWERS=runs/r27e/answers REL_ARM=base_rag bash $R/ctl/vs_newret.sh r27f
