"""
Lightweight experiment manifest - which code, data and artefacts each pipeline stage used and produced
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from config import PROJECT_DIR, RESULTS_DIR

MANIFEST_PATH = RESULTS_DIR / "run_manifest.json"
TRACKED_LIBRARIES = ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "shap", "scipy", "joblib"]


class ProvenanceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_versions() -> dict[str, str | None]:
    versions = {}
    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def code_files(project_dir: Path = PROJECT_DIR) -> list[Path]:
    files = sorted(project_dir.glob("*.py")) + sorted((project_dir / "src").glob("*.py"))
    lock = project_dir / "requirements-lock.txt"
    return files + ([lock] if lock.exists() else [])


def code_fingerprint(project_dir: Path = PROJECT_DIR) -> str:
    digest = hashlib.sha256()
    for path in code_files(project_dir):
        digest.update(path.relative_to(project_dir).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _git(*args: str, project_dir: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(project_dir), *args], capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.rstrip()


def code_version(project_dir: Path = PROJECT_DIR) -> dict:
    commit = _git("rev-parse", "HEAD", project_dir=project_dir)
    dirty = []
    if commit:
        modified = _git("status", "--porcelain", "--untracked-files=no", project_dir=project_dir) or ""
        untracked = _git("ls-files", "--others", "--exclude-standard", "--", "*.py", project_dir=project_dir) or ""
        dirty = sorted({line[3:] for line in modified.splitlines() if line} | {line for line in untracked.splitlines() if line})

    return {
        "git_commit": commit,
        "git_dirty": bool(dirty) if commit else None,
        "git_dirty_paths": dirty[:100],
        "code_fingerprint_sha256": code_fingerprint(project_dir)
    }


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _absolute(relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_DIR / path


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if not path.exists():
        return {"manifest_version": 1, "stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def require_stages(stages: list[str], path: Path = MANIFEST_PATH) -> dict[str, str]:
    manifest = load_manifest(path)
    run_ids = {}

    for stage in stages:
        record = manifest["stages"].get(stage)
        if record is None:
            raise ProvenanceError(f"Stage {stage} is not recorded in {_relative(path)} - run it (and the following stages) first")

        for kind in ("inputs", "outputs"):
            for relative, expected in record[kind].items():
                file_path = _absolute(relative)
                if not file_path.exists():
                    raise ProvenanceError(f"{relative} ({kind[:-1]} of {stage}, run {record['run_id']}) is missing - re-run {stage} and the following stages")
                if sha256_file(file_path) != expected:
                    raise ProvenanceError(f"{relative} ({kind[:-1]} of {stage}, run {record['run_id']}) changed after that run - re-run {stage} and the following stages")

        for upstream, upstream_run in record["upstream_run_ids"].items():
            current = manifest["stages"].get(upstream, {}).get("run_id")
            if current != upstream_run:
                print(f"Note: {stage} (run {record['run_id']}) was built on {upstream} run {upstream_run}; the recorded {upstream} run is now {current}, and the files {stage} read are unchanged")

        run_ids[stage] = record["run_id"]

    return run_ids


def record_stage(stage: str, inputs: list[Path], outputs: list[Path], parameters: dict, upstream: dict[str, str] | None = None, path: Path = MANIFEST_PATH) -> str:
    manifest = load_manifest(path)
    hashes = {"inputs": {_relative(p): sha256_file(p) for p in inputs}, "outputs": {_relative(p): sha256_file(p) for p in outputs}}
    previous = manifest["stages"].get(stage)
    reproduced = previous is not None and all(previous[kind] == hashes[kind] for kind in hashes)
    run_id = previous["run_id"] if reproduced else uuid.uuid4().hex[:12]

    manifest["stages"][stage] = {
        "run_id": run_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "upstream_run_ids": upstream or {},
        "code": code_version(),
        "python_version": platform.python_version(),
        "library_versions": library_versions(),
        "parameters": parameters,
        **hashes
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return run_id
