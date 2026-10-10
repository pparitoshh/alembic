"""Stage `track`: log dev-eval results per checkpoint, and the final adapter's eval, into the training run's MLflow run.

The trainer (train.report_to: [mlflow]) logs loss and validation loss under run name = run_dir.
Checkpoint adapters are kept in run_dir/adapters/step_<N>/; slurm/evaluate_checkpoints.sbatch
answers and judges each, writing step_<N>/eval_summary.json. This stage reads those summaries and
logs their numeric fields at step N, so loss and judged quality share one MLflow x-axis.
slurm/evaluate.sbatch judges the final adapter, writing run_dir/eval_summary[_<tag>].json; those are logged
as eval/[<tag>/]... at the last kept checkpoint step (the end of training).
The tracking URI and experiment come from MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT_NAME, as for the trainer.
"""

import json
from pathlib import Path

from .config import Config


def _numbers(summary: Path, prefix: str) -> dict[str, float]:
    """A summary's models' numeric fields and win rates, as {prefix/metric: value}."""
    metrics = {}
    for key, value in json.loads(summary.read_text()).items():
        if isinstance(value, dict):
            metrics |= {f"{prefix}/{key}/{k}": float(v) for k, v in value.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)}
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"{prefix}/{key}"] = float(value)
    return metrics


def checkpoint_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    """{step: {metric: value}} from adapters/step_<N>/eval_summary.json."""
    return {int(summary.parent.name.removeprefix("step_")): _numbers(summary, "dev")
            for summary in sorted(run_dir.glob("adapters/step_*/eval_summary.json"))}


def final_metrics(run_dir: Path) -> dict[str, float]:
    """{metric: value} from the final adapter's eval_summary.json (eval/...) and eval_summary_<tag>.json (eval/<tag>/...)."""
    metrics = {}
    for summary in sorted(run_dir.glob("eval_summary*.json")):
        tag = summary.stem.removeprefix("eval_summary").removeprefix("_")
        metrics |= _numbers(summary, f"eval/{tag}" if tag else "eval")
    return metrics


def final_step(run_dir: Path) -> int:
    """The last kept checkpoint step: the trainer saves one at the end of training (0 if none were kept)."""
    return max((int(d.name.removeprefix("step_")) for d in run_dir.glob("adapters/step_*")), default=0)


def run(cfg: Config) -> dict[int, dict[str, float]]:
    import mlflow

    metrics = checkpoint_metrics(cfg.run_dir)
    if final := final_metrics(cfg.run_dir):
        metrics.setdefault(final_step(cfg.run_dir), {}).update(final)
    found = mlflow.search_runs(filter_string=f"attributes.run_name = '{cfg.run_dir}'", output_format="list",
                               search_all_experiments=True)
    if not found:
        raise ValueError(f"no MLflow run named {cfg.run_dir}; train with train.report_to=[mlflow] first")
    run_id = max(found, key=lambda r: r.info.start_time).info.run_id
    client = mlflow.MlflowClient()
    for step, values in metrics.items():
        for key, value in values.items():
            client.log_metric(run_id, key, value, step=step)
    print(f"[track] logged {sum(map(len, metrics.values()))} metrics for steps {sorted(metrics)} -> run {run_id}")
    return metrics
