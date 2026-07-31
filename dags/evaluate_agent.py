"""Evaluation pipeline for coding-agent experiments.

prepare_run -> run_agent -> run_eval -> summarize -> upload_artifacts -> log_mlflow

Every step after prepare_run executes `python -m pipeline.run_step <step> <run_dir>`.
EXECUTION_MODE selects the isolation level for those steps:
  local  (default) - subprocess in the project venv, for airflow standalone
  docker           - DockerOperator containers from the project image, for compose
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import Param, dag, get_current_context, task

PROJECT_ROOT = Path(os.environ.get("E2E_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import helpers  # noqa: E402

EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "local")

# (retries, execution timeout minutes) per step: the agent and eval calls are
# long and safely retryable (existing trajectories are skipped on rerun);
# upload and mlflow are short network calls that deserve more attempts.
STEP_POLICY = {
    "agent": (1, 120),
    "eval": (1, 60),
    "summarize": (1, 5),
    "upload": (2, 15),
    "mlflow": (2, 5),
}

RUN_ID_TEMPLATE = "{{ ti.xcom_pull(task_ids='prepare_run') }}"

PASSTHROUGH_ENV = [
    "NEBIUS_API_KEY", "MLFLOW_TRACKING_URI",
    "S3_BUCKET", "S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
]


def make_step(step: str):
    retries, timeout_min = STEP_POLICY[step]
    common = {
        "task_id": f"run_{step}" if step in ("agent", "eval") else step,
        "retries": retries,
        "retry_delay": timedelta(minutes=1),
        "execution_timeout": timedelta(minutes=timeout_min),
    }

    if EXECUTION_MODE == "docker":
        from airflow.providers.docker.operators.docker import DockerOperator
        from docker.types import Mount

        return DockerOperator(
            image=os.environ.get("PIPELINE_IMAGE", "e2emlops-pipeline:latest"),
            command=["python", "-m", "pipeline.run_step", step, f"/opt/project/runs/{RUN_ID_TEMPLATE}"],
            docker_url="unix://var/run/docker.sock",
            network_mode=os.environ.get("DOCKER_NETWORK", "bridge"),
            mounts=[
                Mount(source=os.environ["HOST_PROJECT_DIR"], target="/opt/project", type="bind"),
                Mount(source="/var/run/docker.sock", target="/var/run/docker.sock", type="bind"),
            ],
            working_dir="/opt/project",
            environment={k: os.environ.get(k, "") for k in PASSTHROUGH_ENV},
            mount_tmp_dir=False,
            auto_remove="success",
            **common,
        )

    @task(**common)
    def local_step(run_ref: str, _step: str = step) -> str:
        run_dir = PROJECT_ROOT / "runs" / run_ref
        cmd = ["python", "-m", "pipeline.run_step", _step, str(run_dir)]
        print("running:", " ".join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=helpers.subprocess_env(PROJECT_ROOT), check=True)
        return run_ref

    return local_step


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
    def prepare_run() -> str:
        params = get_current_context()["params"]
        run_config = helpers.build_run_config(params, PROJECT_ROOT)
        helpers.prepare_run_dir(run_config)
        print(f"prepared {run_config['run_dir']} (mode: {EXECUTION_MODE})")
        return run_config["run_id"]

    run_id = prepare_run()

    if EXECUTION_MODE == "docker":
        steps = [make_step(s) for s in ("agent", "eval", "summarize", "upload", "mlflow")]
        run_id >> steps[0]
        for left, right in zip(steps, steps[1:]):
            left >> right
    else:
        prev = run_id
        for s in ("agent", "eval", "summarize", "upload", "mlflow"):
            prev = make_step(s)(prev)


evaluate_agent()
