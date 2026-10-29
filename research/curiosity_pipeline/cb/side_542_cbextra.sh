#!/bin/bash
# Side job, 3 GPUs (Oct 28 night, 7-GPU rule): (a) a second closed-book teacher set, 12k more verified questions (seed 11,
# none already in kdcb), Qwen3-235B FP8 TP2 on GPUs 0-1 -> ~/cp/exp/kdcbx/teacher; (b) GPU 2: 1.3's answers to the mentor
# cases without and with the app (~/cp/mentor/cb/v13_*.jsonl) so the later 541 task only answers with kdcb.
source $HOME/cp/bin/lib.sh; TASK=cbextra; cd $REPO
D=$CP/exp/kdcbx; mkdir -p $D/teacher $CP/mentor/cb
SITES='leonardo|cineca|marconi|galileo|lumi|mahti|puhti|roihu|csc|meluxina|luxembourg|mimer|alvis|berzelius|jupiter|juwels|jureca|marenostrum|deucalion|bsc|adastra|jean zay|idris|vega|discoverer|karolina|it4i|daedalus|aris|snellius|surf|hammerhai|lrz|gwdg|alps|cscs|daint|eiger|clariden|archer2|cirrus|plgrid|cyfronet|athena|helios|ares|devana|musica|vsc|komondor|cyclone|finisterrae|cesga|hortense|eurohpc|ai factory'
LAYOUT='ntasks-per-node|gpus-per-task|gpu-bind|cuda_visible_devices|slurm_localid|local rank|per gpu|mpirun|nccl|torchrun|multi-node|cpus-per-task|cpus per node|memory per node|node configuration|mem-per'
if [ ! -s $D/prompts.jsonl ]; then
    PROMPT_REGEX="\\b($SITES)\\b|$LAYOUT" python $CP/bin/build_prompts.py $D/prompts_raw.jsonl 20000 11 2>&1 | tail -1 | tee -a $CP/logs/tasks.log
    python $CP/cb/cb_filter.py $D/prompts_raw.jsonl $CP/cb/mentor_all.jsonl $D/prompts_f.jsonl | tee -a $CP/logs/tasks.log
    python $CP/cb/drop_seen.py $D/prompts_f.jsonl $CP/exp/kdcb/prompts.jsonl $D/prompts_n.jsonl | tee -a $CP/logs/tasks.log
    head -12000 $D/prompts_n.jsonl > $D/prompts.jsonl
fi
log "cbextra prompts: $(wc -l < $D/prompts.jsonl)"
CUDA_VISIBLE_DEVICES=0,1 TP=2 python $CP/cb/teacher_gen_cd.py $D/prompts.jsonl $D/teacher 0 1 $M/q235fp8 $CP/exp/kdall/model > $CP/logs/cbextra.teacher.log 2>&1 &
TP_PID=$!
serve 2 8100 $CP/exp/kdall/model cand --max-model-len 16384 --gpu-memory-utilization 0.85 && \
    python $CP/cb/mentor_models.py $CP/mentor/cases25.jsonl $CP/mentor/cb v13=http://127.0.0.1:8100/v1 2>&1 | tail -2 | tee -a $CP/logs/tasks.log
wait $TP_PID
log "cbextra teacher: $(cat $D/teacher/part0.jsonl 2>/dev/null | wc -l) answers"
stop_all
