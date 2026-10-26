# shared_split controllers (Leonardo, both accounts)

Copies of the controllers in `/leonardo_work/EUHPC_D30_031/shared_split/ctl` that run the factory batches on both
project accounts (the second account runs them through its own `queue_runner.sh` and `ctl/QUEUE`).

| Script | What it does |
|---|---|
| `queue_runner.sh` | Starts every task in `ctl/QUEUE` (round, review, factory) and restarts dead ones |
| `run_factory.sh` | One factory batch: generate and review jobs sized to the free debug-QoS nodes, then the merge |
| `feeder.sh` | Keeps 4 batches prepared; starts up to 3 batches on a08trc09 while it has idle nodes, queues the rest for a08trc08 |

Changes on Oct 25–26, each after nodes sat idle:

- Per-account submit lock: two controllers no longer size jobs from the same free-node count.
- Score work first only while a score job waits or is between pieces; a running score job leaves the second slot to
  a factory batch.
- A batch that is merging (CPU only) no longer counts as a running GPU batch in the feeder.
- Bundled tails: a 1-node tail that would take the user's last debug slot is offered to another batch's next job,
  which runs it as a second job step on extra nodes (`research/ai_factory_pilot/factory_dbg.sbatch` `SHARD_LIST2`).
  Unclaimed offers expire after 10 minutes.
- Only the debug QoS is usable: the project's fairshare puts normal-QoS jobs a week out.
