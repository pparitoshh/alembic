#!/bin/bash
# HPC answer check of the agent model round 49b (9/10 on the agent smoke test): answers with variant e at 1536 tokens, judged against round 27 + e by the three judges.
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
ANSWER_MAX_TOKENS=1536 VARIANT=e bash \/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017/ctl/ragc.sh r49bq \/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017/code/runs/r49b/merged
REL_ANSWERS=runs/r27e/answers REL_ARM=base_rag bash \/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017/ctl/vs_newret.sh r49bq
