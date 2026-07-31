import json
import shutil
from pathlib import Path

from pipeline import helpers

SAMPLE_REPORT = Path(__file__).resolve().parents[1] / "sample" / "nebius__moonshotai__Kimi-K2.6.test.json"


def test_build_run_config_generates_run_id():
    params = {"split": "test", "subset": "verified", "workers": 2,
              "model": "nebius/moonshotai/Kimi-K2.6", "task_slice": "0:3",
              "run_id": "", "cost_limit": 0, "max_output_tokens": 8192}
    cfg = helpers.build_run_config(params, "/tmp/proj")
    assert cfg["run_id"].endswith("-Kimi-K2.6")
    assert cfg["run_dir"] == f"/tmp/proj/runs/{cfg['run_id']}"
    assert cfg["dataset"] == "princeton-nlp/SWE-bench_Verified"


def test_explicit_run_id_wins():
    params = {"split": "test", "subset": "lite", "workers": 1,
              "model": "m", "task_slice": "", "run_id": "exp-42",
              "cost_limit": 1.5, "max_output_tokens": 4096}
    cfg = helpers.build_run_config(params, "/tmp/proj")
    assert cfg["run_id"] == "exp-42"
    assert cfg["cost_limit"] == 1.5


def test_agent_command_has_no_hardcoded_experiment_values():
    params = {"split": "dev", "subset": "lite", "workers": 3,
              "model": "some/model", "task_slice": "5:8", "run_id": "r1",
              "cost_limit": 2, "max_output_tokens": 4096}
    cmd = helpers.agent_command(helpers.build_run_config(params, "/tmp/proj"))
    joined = " ".join(cmd)
    assert "--subset lite" in joined and "--split dev" in joined
    assert "--model some/model" in joined and "--slice 5:8" in joined
    assert "model.model_kwargs.max_tokens=4096" in joined
    assert "agent.cost_limit=2" in joined


def test_collect_metrics_from_instructor_sample(tmp_path):
    shutil.copy(SAMPLE_REPORT, tmp_path / "nebius__moonshotai__Kimi-K2.6.test.json")
    metrics = helpers.collect_metrics(tmp_path, "test")
    assert metrics["submitted_instances"] == 3
    assert metrics["resolved_instances"] == 1
    assert round(metrics["resolve_rate"], 3) == 0.333
    assert metrics["eval_report_found"] is True


def test_collect_metrics_when_eval_skipped(tmp_path):
    metrics = helpers.collect_metrics(tmp_path, "nothing")
    assert metrics["submitted_instances"] == 0
    assert metrics["resolve_rate"] == 0.0
    assert metrics["eval_report_found"] is False


def test_artifact_uri_prefers_s3_when_configured(monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    assert helpers.artifact_uri_for("r1", "/x/runs/r1") == "local:///x/runs/r1"
    monkeypatch.setenv("S3_BUCKET", "my-bucket")
    assert helpers.artifact_uri_for("r1", "/x/runs/r1") == "s3://my-bucket/runs/r1"


def test_run_step_summarize_writes_metrics_and_manifest(tmp_path, monkeypatch):
    from pipeline import run_step

    monkeypatch.delenv("S3_BUCKET", raising=False)
    run_dir = tmp_path / "runs" / "r9"
    (run_dir / "run-eval").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"run_id": "r9", "run_dir": "/somewhere/else/runs/r9", "created_at": "now"}))
    config = run_step.load_config(run_dir)
    assert config["run_dir"] == str(run_dir)  # rebased onto local view
    run_step.step_summarize(config, run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["submitted_instances"] == 0 and metrics["eval_report_found"] is False
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["artifact_uri"] == f"local://{run_dir}"


def test_manifest_lists_files(tmp_path):
    cfg = {"run_id": "r", "run_dir": str(tmp_path), "created_at": "now"}
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "run-agent").mkdir()
    (tmp_path / "run-agent" / "preds.json").write_text(json.dumps({"a": 1}))
    manifest = helpers.build_manifest(cfg, {"resolve_rate": 0.5})
    assert manifest["artifact_uri"] == f"local://{tmp_path}"
    assert "config.json" in manifest["files"]
    assert manifest["files"]["run-agent/preds.json"] == len('{"a": 1}')
