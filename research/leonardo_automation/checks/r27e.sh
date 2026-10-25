#!/bin/bash
# Variant e test (Oct 25): round 27 answered with retriever b excerpts under the relevance header, vs round 27 + b at 1536.
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
ANSWER_MAX_TOKENS=1536 VARIANT=e bash $R/ctl/ragc.sh r27e /leonardo_work/EUHPC_D30_031/shared_split/models/r27_merged
REL_ANSWERS=runs/tok1536/answers REL_ARM=base_rag bash $R/ctl/vs_newret.sh r27e
