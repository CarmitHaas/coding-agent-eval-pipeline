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


def test_manifest_lists_files(tmp_path):
    cfg = {"run_id": "r", "run_dir": str(tmp_path), "created_at": "now"}
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "run-agent").mkdir()
    (tmp_path / "run-agent" / "preds.json").write_text(json.dumps({"a": 1}))
    manifest = helpers.build_manifest(cfg, {"resolve_rate": 0.5})
    assert manifest["artifact_uri"] == f"local://{tmp_path}"
    assert "config.json" in manifest["files"]
    assert manifest["files"]["run-agent/preds.json"] == len('{"a": 1}')
