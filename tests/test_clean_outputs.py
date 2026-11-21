import importlib.util
import os
import stat
import sys
from pathlib import Path

import pytest

import config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("clean_outputs", ROOT / "clean_outputs.py")
cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleaner)


def make_project(root: Path) -> Path:
    files = {
        "data/WA_Fn-UseC_-Telco-Customer-Churn.xlsx": "raw",
        "data/new_customers.xlsx": "demo",
        "data/processed/X_train.csv": "a,b",
        "models/final.joblib": "model",
        "results/run_manifest.json": "{}",
        "results/sub/table.csv": "x",
        "figures/fig.pdf": "pdf",
        "logs/20261016/04_modeling.log": "log",
        "archive/run_20261016_103420/results/old.csv": "old",
        "src/evaluation.py": "code",
        "config.py": "code"
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def snapshot(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_generated_dirs_match_the_pipeline_output_dirs():
    configured = {config.PROCESSED_DIR, config.MODELS_DIR, config.RESULTS_DIR, config.FIGURES_DIR, config.PROJECT_DIR / "logs"}
    assert {(config.PROJECT_DIR / d).resolve() for d in cleaner.GENERATED_DIRS} == {p.resolve() for p in configured}


def test_dry_run_changes_nothing(tmp_path):
    project = make_project(tmp_path)
    before = snapshot(project)
    files = cleaner.clean(project, execute=False, confirm=None, backup_name="b")
    assert snapshot(project) == before
    assert len(files) == 6 and not (project / "archive" / "b").exists()


@pytest.mark.parametrize("confirm", [None, "yes", "clean"])
def test_execute_without_exact_confirmation_refuses(tmp_path, confirm):
    project = make_project(tmp_path)
    before = snapshot(project)
    with pytest.raises(SystemExit):
        cleaner.clean(project, execute=True, confirm=confirm, backup_name="b")
    assert snapshot(project) == before


def test_execute_backs_up_then_empties_only_generated_dirs(tmp_path):
    project = make_project(tmp_path)
    before = snapshot(project)
    cleaner.clean(project, execute=True, confirm="CLEAN", backup_name="pre_clean_test")
    after = snapshot(project)

    for relative in ("data/WA_Fn-UseC_-Telco-Customer-Churn.xlsx", "data/new_customers.xlsx", "archive/run_20261016_103420/results/old.csv", "src/evaluation.py", "config.py"):
        assert after[relative] == before[relative]
    for directory in cleaner.GENERATED_DIRS:
        assert (project / directory).is_dir() and not any((project / directory).iterdir())
    generated = {k: v for k, v in before.items() if k.split("/")[0] in ("models", "results", "figures", "logs") or k.startswith("data/processed/")}
    assert {f"archive/pre_clean_test/{k}": v for k, v in generated.items()}.items() <= after.items()


@pytest.mark.skipif(sys.platform != "win32", reason="read-only folder attribute is Windows-specific")
def test_read_only_folders_are_emptied(tmp_path):
    project = make_project(tmp_path)
    locked = project / "logs" / "20261016"
    os.chmod(locked, stat.S_IREAD)
    cleaner.clean(project, execute=True, confirm="CLEAN", backup_name="b")
    assert not any((project / "logs").iterdir())
    assert (project / "archive" / "b" / "logs" / "20261016" / "04_modeling.log").exists()


def test_existing_backup_folder_is_never_overwritten(tmp_path):
    project = make_project(tmp_path)
    before = snapshot(project)
    with pytest.raises(FileExistsError):
        cleaner.clean(project, execute=True, confirm="CLEAN", backup_name="run_20261016_103420")
    assert snapshot(project) == before


def test_targets_overlapping_protected_paths_are_rejected(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    for bad in (("archive",), ("data",), ("..",)):
        monkeypatch.setattr(cleaner, "GENERATED_DIRS", bad)
        with pytest.raises(ValueError):
            cleaner.target_dirs(project)
