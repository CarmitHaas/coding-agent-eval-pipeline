"""Log a completed run to MLflow: python -m pipeline.log_run <run_dir>.

Runs inside the project venv (which has mlflow installed), so the Airflow
process itself never needs MLflow as a dependency.
"""

import json
import sys
from pathlib import Path

import mlflow

PARAM_KEYS = ["split", "subset", "workers", "model", "task_slice", "cost_limit", "max_output_tokens"]


def main(run_dir: str) -> None:
    run_path = Path(run_dir)
    config = json.loads((run_path / "config.json").read_text())
    metrics = json.loads((run_path / "metrics.json").read_text())
    manifest = json.loads((run_path / "manifest.json").read_text())

    mlflow.set_experiment("coding-agent-eval")
    with mlflow.start_run(run_name=config["run_id"]):
        mlflow.log_params({k: config[k] for k in PARAM_KEYS})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.set_tag("run_id", config["run_id"])
        mlflow.set_tag("artifact_uri", manifest["artifact_uri"])
        for name in ["config.json", "metrics.json", "manifest.json"]:
            mlflow.log_artifact(str(run_path / name))
    print(f"logged run {config['run_id']} to MLflow")


if __name__ == "__main__":
    main(sys.argv[1])
