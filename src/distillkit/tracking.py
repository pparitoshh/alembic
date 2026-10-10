"""Stage `track`: log per-checkpoint dev-eval results into the training run's MLflow run.

The trainer (train.report_to: [mlflow]) logs loss and validation loss under run name = run_dir.
Checkpoint adapters are kept in run_dir/adapters/step_<N>/; slurm/evaluate_checkpoints.sbatch
answers and judges each, writing step_<N>/eval_summary.json. This stage reads those summaries and
logs their numeric fields at step N, so loss and judged quality share one MLflow x-axis.
The tracking URI and experiment come from MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT_NAME, as for the trainer.
"""

import json
from pathlib import Path

from .config import Config


def checkpoint_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    """{step: {metric: value}} from adapters/step_<N>/eval_summary.json; models' numeric fields and win rates."""
    out = {}
    for summary in sorted(run_dir.glob("adapters/step_*/eval_summary.json")):
        step = int(summary.parent.name.removeprefix("step_"))
        data = json.loads(summary.read_text())
        metrics = {}
        for key, value in data.items():
            if isinstance(value, dict):
                metrics |= {f"dev/{key}/{k}": float(v) for k, v in value.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)}
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"dev/{key}"] = float(value)
        out[step] = metrics
    return out


def run(cfg: Config) -> dict[int, dict[str, float]]:
    import mlflow

    metrics = checkpoint_metrics(cfg.run_dir)
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
