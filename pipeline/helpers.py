"""Pure helpers for the evaluate_agent pipeline.

No Airflow imports here: everything is testable with plain pytest and reusable
from the CLI, the DAG, or a container entrypoint.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}

METRIC_KEYS = [
    "total_instances",
    "submitted_instances",
    "completed_instances",
    "resolved_instances",
    "unresolved_instances",
    "empty_patch_instances",
    "error_instances",
]


def load_dotenv(project_root: str | Path) -> dict:
    """Read KEY=VALUE lines from <project_root>/.env into a dict (no export)."""
    env = {}
    path = Path(project_root) / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip("'\"")
    return env


def dataset_for_subset(subset: str) -> str:
    """Map a mini-swe-agent subset name to the SWE-bench dataset name."""
    return DATASETS.get(subset, subset)


def build_run_config(params: dict, project_root: str | Path) -> dict:
    """Resolve Airflow params into one immutable run configuration."""
    run_id = (params.get("run_id") or "").strip()
    if not run_id:
        model_tail = str(params["model"]).rsplit("/", 1)[-1]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-{model_tail}"
    run_dir = Path(project_root) / "runs" / run_id
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "project_root": str(project_root),
        "split": params["split"],
        "subset": params["subset"],
        "dataset": dataset_for_subset(params["subset"]),
        "workers": int(params["workers"]),
        "model": params["model"],
        "task_slice": str(params.get("task_slice") or ""),
        "cost_limit": float(params.get("cost_limit") or 0),
        "max_output_tokens": int(params.get("max_output_tokens") or 8192),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def prepare_run_dir(run_config: dict) -> str:
    """Create runs/<run-id>/ and persist config.json. Idempotent for reruns."""
    run_dir = Path(run_config["run_dir"])
    (run_dir / "run-agent").mkdir(parents=True, exist_ok=True)
    (run_dir / "run-eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    return str(run_dir)


def agent_command(run_config: dict) -> list[str]:
    """mini-swe-agent batch command. Config overrides are passed with -c so the
    exact experiment knobs live in config.json, not in a forked YAML."""
    cmd = [
        "mini-extra", "swebench",
        "--subset", run_config["subset"],
        "--split", run_config["split"],
        "--model", run_config["model"],
        "--workers", str(run_config["workers"]),
        "-o", str(Path(run_config["run_dir"]) / "run-agent"),
        "-c", "swebench.yaml",
        "-c", f"model.model_kwargs.max_tokens={run_config['max_output_tokens']}",
        "-c", f"agent.cost_limit={run_config['cost_limit']}",
    ]
    if run_config["task_slice"]:
        cmd += ["--slice", run_config["task_slice"]]
    return cmd


def eval_command(run_config: dict, preds_path: str) -> list[str]:
    """SWE-bench harness command; run it with cwd=runs/<run-id>/run-eval so all
    logs and the report land inside the run folder."""
    return [
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", run_config["dataset"],
        "--split", run_config["split"],
        "--predictions_path", preds_path,
        "--max_workers", str(run_config["workers"]),
        "--run_id", run_config["run_id"],
    ]


def preds_count(preds_path: str | Path) -> int:
    path = Path(preds_path)
    if not path.exists():
        return 0
    return len(json.loads(path.read_text() or "{}"))


def collect_metrics(eval_dir: str | Path, run_id: str) -> dict:
    """Parse the harness summary report (<model>.<run_id>.json) into flat metrics.
    A run with no evaluated predictions is a valid, recorded outcome: all zeros."""
    reports = sorted(Path(eval_dir).glob(f"*.{run_id}.json"))
    metrics = dict.fromkeys(METRIC_KEYS, 0)
    if reports:
        report = json.loads(reports[0].read_text())
        for key in METRIC_KEYS:
            metrics[key] = report.get(key, 0)
    submitted = metrics["submitted_instances"]
    metrics["resolve_rate"] = (
        metrics["resolved_instances"] / submitted if submitted else 0.0
    )
    metrics["eval_report_found"] = bool(reports)
    return metrics


def build_manifest(run_config: dict, metrics: dict, artifact_uri: str | None = None) -> dict:
    """Everything needed to reconstruct the run: config, key files, all files."""
    run_dir = Path(run_config["run_dir"])
    files = sorted(
        str(p.relative_to(run_dir))
        for p in run_dir.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    )
    sizes = {f: (run_dir / f).stat().st_size for f in files}
    return {
        "run_id": run_config["run_id"],
        "created_at": run_config["created_at"],
        "config": "config.json",
        "predictions": "run-agent/preds.json",
        "trajectories_dir": "run-agent/",
        "eval_dir": "run-eval/",
        "metrics": "metrics.json",
        "resolve_rate": metrics.get("resolve_rate"),
        "artifact_uri": artifact_uri or f"local://{run_dir}",
        "files": sizes,
    }


def artifact_uri_for(run_id: str, run_dir: str | Path) -> str:
    """Where the durable copy of this run lives: S3 when configured, else local."""
    bucket = os.environ.get("S3_BUCKET")
    if bucket:
        return f"s3://{bucket}/runs/{run_id}"
    return f"local://{run_dir}"


def subprocess_env(project_root: str | Path, extra: dict | None = None) -> dict:
    """Environment for agent/eval subprocesses: project venv first on PATH,
    .env values injected, cost tracking tolerant of unpriced models."""
    venv_bin = str(Path(project_root) / ".venv" / "bin")
    env = {
        **os.environ,
        **load_dotenv(project_root),
        "PATH": venv_bin + os.pathsep + os.environ.get("PATH", ""),
        "MSWEA_COST_TRACKING": "ignore_errors",
    }
    if extra:
        env.update(extra)
    return env
