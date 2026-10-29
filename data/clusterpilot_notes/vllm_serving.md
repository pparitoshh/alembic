# Serving LLMs with vLLM on a Slurm GPU cluster

How many GPUs: weights take parameters x 2 bytes in bf16 (x 1 in FP8 or INT8, about x 0.5 in 4-bit AWQ or GPTQ),
and vLLM needs room beyond that for the KV cache. So 8B bf16 is about 16 GB, 14B about 30 GB, 32B about 65 GB,
70B about 140 GB, 235B about 470 GB. Use the fewest GPUs that hold the weights plus KV cache:
on 64 GB A100s (Leonardo) 14B fits on 1 GPU, 32B needs 2 (or 1 with a 4-bit checkpoint), 70B needs 4.
`--gpu-memory-utilization` (default 0.9) is the share of each GPU vLLM may use; raising it does not make weights fit.

- Tensor parallelism (`--tensor-parallel-size`) must divide the model's number of attention heads, so use 1, 2, 4
  or 8, and keep it inside one node. Across nodes add pipeline parallelism (`--pipeline-parallel-size`, one stage per
  node) over a Ray cluster: `ray start --head --port=6379` on the first node, `ray start --address=<head>:6379` on the
  others (with `srun`), then `vllm serve ... --tensor-parallel-size 4 --pipeline-parallel-size <nodes>
  --distributed-executor-backend ray` on the first node.
- Start the server with `vllm serve <model or local path> --dtype bfloat16 --port <port>` (OpenAI-compatible API;
  the older `vllm.entrypoints.api_server` is not the OpenAI server).
- Downloads: compute nodes usually have no internet (Leonardo's do not). Download on a login node, into $WORK or
  $SCRATCH, not $HOME: `hf download Qwen/Qwen3-14B --local-dir $WORK/models/Qwen3-14B` (older tool:
  `huggingface-cli download`). In the job set `export HF_HUB_OFFLINE=1` and point vLLM at the local folder.
- Batch work in one job: start the server in the background on 127.0.0.1, wait until
  `curl -sf http://127.0.0.1:$PORT/health` succeeds, run the client, then stop the server:

```bash
vllm serve $WORK/models/Qwen3-14B --dtype bfloat16 --host 127.0.0.1 --port $PORT --max-model-len 8192 &
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do sleep 10; done
python client.py --base-url http://127.0.0.1:$PORT/v1
kill %1
```

- Many prompt files: a job array with one file per task (`--array=0-199%10`), each task writes its own output and
  skips finished files; offline inference (`vllm.LLM(...).generate(...)` or `vllm run-batch`) needs no server.
- From a laptop: find the job's node with `squeue -j <jobid> -o %N`, then
  `ssh -L 8000:<node>:8000 <user>@<login host>` (Leonardo: login.leonardo.cineca.it) and use
  http://localhost:8000/v1. Compute nodes are shared: bind a port only you use and set `--api-key`.
