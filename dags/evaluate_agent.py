"""Evaluation pipeline for coding-agent experiments.

prepare_run -> run_agent -> run_eval -> summarize_and_log

Every run writes a self-contained runs/<run-id>/ folder and logs params,
metrics, and the artifact reference to MLflow (when MLFLOW_TRACKING_URI is set).
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task
from airflow.sdk import Param, get_current_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import helpers  # noqa: E402


def _run(cmd: list[str], cwd: str | Path) -> None:
    print("running:", " ".join(cmd), "| cwd:", cwd)
    subprocess.run(cmd, cwd=cwd, env=helpers.subprocess_env(PROJECT_ROOT), check=True)


@dag(
    dag_id="evaluate_agent",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["swe-bench", "mini-swe-agent", "evaluation"],
    params={
        "split": Param("test", type="string", description="Dataset split (SWE-bench Verified uses 'test')"),
        "subset": Param("verified", type="string", enum=["verified", "lite", "full"], description="SWE-bench subset"),
        "workers": Param(1, type="integer", minimum=1, maximum=16, description="Parallel workers for agent and eval"),
        "model": Param("nebius/moonshotai/Kimi-K2.6", type="string", description="Model for the agent (litellm name)"),
        "task_slice": Param("0:1", type="string", description="Instance slice, e.g. '0:3'; empty runs the whole subset"),
        "run_id": Param("", type="string", description="Run ID; leave empty to auto-generate <timestamp>-<model>"),
        "cost_limit": Param(0, type="number", minimum=0, description="Per-instance cost limit in USD; 0 disables"),
        "max_output_tokens": Param(8192, type="integer", minimum=1024, description="Model output token budget per step"),
    },
)
def evaluate_agent():
    @task
    def prepare_run() -> dict:
        params = get_current_context()["params"]
        run_config = helpers.build_run_config(params, PROJECT_ROOT)
        helpers.prepare_run_dir(run_config)
        print(f"prepared {run_config['run_dir']}")
        return run_config

    @task
    def run_agent(run_config: dict) -> str:
        _run(helpers.agent_command(run_config), cwd=PROJECT_ROOT)
        preds = Path(run_config["run_dir"]) / "run-agent" / "preds.json"
        print(f"{helpers.preds_count(preds)} predictions at {preds}")
        return str(preds)

    @task
    def run_eval(run_config: dict, preds_path: str) -> str:
        eval_dir = Path(run_config["run_dir"]) / "run-eval"
        if helpers.preds_count(preds_path) == 0:
            note = "no predictions produced; evaluation skipped"
            (eval_dir / "SKIPPED.txt").write_text(note + "\n")
            print(note)
        else:
            _run(helpers.eval_command(run_config, preds_path), cwd=eval_dir)
        return str(eval_dir)

    @task
    def summarize_and_log(run_config: dict, eval_dir: str) -> dict:
        run_dir = Path(run_config["run_dir"])
        metrics = helpers.collect_metrics(eval_dir, run_config["run_id"])
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        manifest = helpers.build_manifest(run_config, metrics)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print("metrics:", json.dumps(metrics, indent=2))

        if os.environ.get("MLFLOW_TRACKING_URI"):
            _run(["python", "-m", "pipeline.log_run", str(run_dir)], cwd=PROJECT_ROOT)
        else:
            print("MLFLOW_TRACKING_URI not set; skipping MLflow logging")
        return metrics

    run_config = prepare_run()
    preds_path = run_agent(run_config)
    eval_dir = run_eval(run_config, preds_path)
    summarize_and_log(run_config, eval_dir)


evaluate_agent()
