"""Single entry point for every pipeline step: python -m pipeline.run_step <step> <run_dir>.

The Airflow DAG calls this identically in both execution modes: as a local
subprocess in standalone mode, or as the container command under DockerOperator.
One code path, two isolation levels.

Paths inside config.json are rebased onto this process's view of the run
directory, so the same run works whether the project is mounted at
/opt/project (containers) or lives at its host path (standalone).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from pipeline import helpers


def load_config(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    config["run_dir"] = str(run_dir)
    config["project_root"] = str(run_dir.parents[1])
    return config


def step_agent(config: dict, run_dir: Path) -> None:
    root = config["project_root"]
    subprocess.run(
        helpers.agent_command(config), cwd=root,
        env=helpers.subprocess_env(root), check=True,
    )
    preds = run_dir / "run-agent" / "preds.json"
    print(f"{helpers.preds_count(preds)} predictions at {preds}")


def step_eval(config: dict, run_dir: Path) -> None:
    preds = run_dir / "run-agent" / "preds.json"
    eval_dir = run_dir / "run-eval"
    if helpers.preds_count(preds) == 0:
        note = "no predictions produced; evaluation skipped"
        (eval_dir / "SKIPPED.txt").write_text(note + "\n")
        print(note)
        return
    subprocess.run(
        helpers.eval_command(config, str(preds)), cwd=eval_dir,
        env=helpers.subprocess_env(config["project_root"]), check=True,
    )


def step_summarize(config: dict, run_dir: Path) -> None:
    metrics = helpers.collect_metrics(run_dir / "run-eval", config["run_id"])
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    uri = helpers.artifact_uri_for(config["run_id"], run_dir)
    manifest = helpers.build_manifest(config, metrics, artifact_uri=uri)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("metrics:", json.dumps(metrics, indent=2))
    print("artifact_uri:", uri)


def step_upload(config: dict, run_dir: Path) -> None:
    from pipeline import upload_run

    upload_run.main(str(run_dir))


def step_mlflow(config: dict, run_dir: Path) -> None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        print("MLFLOW_TRACKING_URI not set; skipping MLflow logging")
        return
    from pipeline import log_run

    log_run.main(str(run_dir))


STEPS = {
    "agent": step_agent,
    "eval": step_eval,
    "summarize": step_summarize,
    "upload": step_upload,
    "mlflow": step_mlflow,
}


def main() -> None:
    step, run_dir = sys.argv[1], Path(sys.argv[2]).resolve()
    STEPS[step](load_config(run_dir), run_dir)


if __name__ == "__main__":
    main()
