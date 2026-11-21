import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import manifest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("scoring_manifest", ROOT / "06_deployment_scoring.py")
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


@pytest.fixture
def artefacts(tmp_path):
    data, model = tmp_path / "data.csv", tmp_path / "model.bin"
    data.write_text("a,b\n1,2\n")
    model.write_bytes(b"model-v1")
    return tmp_path / "run_manifest.json", data, model


def test_recorded_stage_passes_the_consistency_check(artefacts):
    path, data, model = artefacts
    run_id = manifest.record_stage("train", [data], [model], {"seed": 42}, path=path)
    assert manifest.require_stages(["train"], path=path) == {"train": run_id}

    record = json.loads(path.read_text(encoding="utf-8"))["stages"]["train"]
    assert record["outputs"] == {model.resolve().as_posix(): manifest.sha256_file(model)}
    assert record["parameters"] == {"seed": 42}
    assert set(record["code"]) == {"git_commit", "git_dirty", "git_dirty_paths", "code_fingerprint_sha256"}


def test_changed_missing_or_unrecorded_artefacts_are_rejected(artefacts):
    path, data, model = artefacts
    manifest.record_stage("train", [data], [model], {}, path=path)

    model.write_bytes(b"model-v2")
    with pytest.raises(manifest.ProvenanceError, match="changed after that run"):
        manifest.require_stages(["train"], path=path)

    model.unlink()
    with pytest.raises(manifest.ProvenanceError, match="missing"):
        manifest.require_stages(["train"], path=path)

    with pytest.raises(manifest.ProvenanceError, match="not recorded"):
        manifest.require_stages(["evaluate"], path=path)


def test_changed_upstream_input_invalidates_downstream_results(artefacts):
    path, data, model = artefacts
    manifest.record_stage("train", [data], [model], {}, path=path)
    report = model.parent / "report.csv"
    report.write_text("x\n1\n")
    manifest.record_stage("evaluate", [model], [report], {}, path=path)

    model.write_bytes(b"retrained")
    with pytest.raises(manifest.ProvenanceError, match="evaluate"):
        manifest.require_stages(["evaluate"], path=path)


def test_identical_rerun_keeps_run_id_and_downstream_lineage(artefacts):
    path, data, model = artefacts
    train_run = manifest.record_stage("train", [data], [model], {}, path=path)
    report = model.parent / "report.csv"
    report.write_text("x\n1\n")
    manifest.record_stage("evaluate", [model], [report], {}, manifest.require_stages(["train"], path=path), path=path)

    assert manifest.record_stage("train", [data], [model], {}, path=path) == train_run
    assert manifest.require_stages(["evaluate"], path=path)["evaluate"]


def test_upstream_rerun_only_invalidates_stages_that_read_a_changed_file(artefacts, capsys):
    path, data, model = artefacts
    extra = model.parent / "extra.csv"
    extra.write_text("v1\n")
    manifest.record_stage("train", [data], [model, extra], {}, path=path)
    report = model.parent / "report.csv"
    report.write_text("x\n1\n")
    manifest.record_stage("evaluate", [model], [report], {}, manifest.require_stages(["train"], path=path), path=path)

    extra.write_text("v2\n")
    new_run = manifest.record_stage("train", [data], [model, extra], {}, path=path)
    assert manifest.require_stages(["train"], path=path) == {"train": new_run}
    assert "evaluate" in manifest.require_stages(["evaluate"], path=path)
    assert f"the recorded train run is now {new_run}" in capsys.readouterr().out

    model.write_bytes(b"retrained")
    manifest.record_stage("train", [data], [model, extra], {}, path=path)
    with pytest.raises(manifest.ProvenanceError, match="input of evaluate"):
        manifest.require_stages(["evaluate"], path=path)


def test_code_fingerprint_ignores_line_endings_and_detects_edits(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    crlf = manifest.code_fingerprint(tmp_path)
    (tmp_path / "a.py").write_bytes(b"x = 1\ny = 2\n")
    assert manifest.code_fingerprint(tmp_path) == crlf
    (tmp_path / "a.py").write_bytes(b"x = 1\ny = 3\n")
    assert manifest.code_fingerprint(tmp_path) != crlf


def test_code_version_without_git(tmp_path):
    version = manifest.code_version(tmp_path)
    assert version["git_commit"] is None and version["git_dirty"] is None
    assert len(version["code_fingerprint_sha256"]) == 64


def test_code_version_keeps_full_dirty_paths(tmp_path):
    git = ["git", "-C", str(tmp_path), "-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    (tmp_path / "05e_stage.py").write_text("x = 1\n")
    subprocess.run([*git, "add", "05e_stage.py"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], check=True)
    (tmp_path / "05e_stage.py").write_text("x = 2\n")
    version = manifest.code_version(tmp_path)
    assert version["git_dirty"] is True and version["git_dirty_paths"] == ["05e_stage.py"]


def scoring_frame(ids):
    return pd.DataFrame({"customerID": ids, "tenure": [1] * len(ids), "TotalCharges": ["10"] * len(ids), "SeniorCitizen": [0] * len(ids)})


def test_scoring_rejects_duplicated_or_missing_customer_ids():
    columns = ["tenure", "TotalCharges", "SeniorCitizen"]
    X, ids = scoring.prepare_scoring_features(scoring_frame(["a", "b"]), columns)
    assert ids.tolist() == ["a", "b"] and len(X) == 2

    with pytest.raises(ValueError, match="Duplicated customerID values: 2 rows share 1 ID"):
        scoring.prepare_scoring_features(scoring_frame(["a", "a", "b"]), columns)
    with pytest.raises(ValueError, match="Missing customerID"):
        scoring.prepare_scoring_features(scoring_frame(["a", None]), columns)
    with pytest.raises(ValueError, match="Missing customerID"):
        scoring.prepare_scoring_features(scoring_frame(["a", "  "]), columns)

    _, generated = scoring.prepare_scoring_features(scoring_frame(["a", "b"]).drop(columns="customerID"), columns)
    assert generated.tolist() == ["row_0", "row_1"]


def test_score_drift_row_uses_oof_reference():
    rng = np.random.default_rng(0)
    X_ref = pd.DataFrame({"tenure": rng.integers(0, 73, 3000)})
    reference_scores = rng.beta(2, 5, 3000)
    X_new = X_ref.sample(1000, random_state=1)

    same = scoring.monitoring_report(X_ref, X_new, reference_scores, rng.beta(2, 5, 1000)).set_index("feature")
    assert same.loc["churn_score", "type"] == "model_score" and same.loc["churn_score", "status"] == "ok"

    shifted = scoring.monitoring_report(X_ref, X_new, reference_scores, rng.beta(5, 2, 1000)).set_index("feature")
    assert shifted.loc["churn_score", "status"] == "alert"

    small = scoring.monitoring_report(X_ref, X_new.head(50), reference_scores, reference_scores[:50]).set_index("feature")
    assert small.loc["churn_score", "status"] == "skipped_small_batch" and np.isnan(small.loc["churn_score", "psi"])

    assert "churn_score" not in set(scoring.monitoring_report(X_ref, X_new)["feature"])
