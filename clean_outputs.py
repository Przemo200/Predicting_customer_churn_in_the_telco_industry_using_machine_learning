"""
Safe removal of generated artefacts before a clean run of the pipeline
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
GENERATED_DIRS = ("data/processed", "models", "results", "figures", "logs")
PROTECTED = ("archive", "data/WA_Fn-UseC_-Telco-Customer-Churn.xlsx", "data/new_customers.xlsx", ".git", ".venv", "src", "tests")
CONFIRMATION = "CLEAN"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_dirs(project_dir: Path) -> list[Path]:
    project_dir = project_dir.resolve()
    protected = [(project_dir / p).resolve() for p in PROTECTED]
    targets = []
    for relative in GENERATED_DIRS:
        path = (project_dir / relative).resolve()
        if project_dir not in path.parents:
            raise ValueError(f"{path} is not inside the project")
        if any(path == p or path in p.parents or p in path.parents for p in protected):
            raise ValueError(f"{path} overlaps a protected path")
        if path.is_symlink():
            raise ValueError(f"{path} is a symbolic link")
        targets.append(path)
    return targets


def generated_files(project_dir: Path) -> list[Path]:
    return sorted(f for d in target_dirs(project_dir) if d.exists() for f in d.rglob("*") if f.is_file())


def backup(project_dir: Path, files: list[Path], backup_dir: Path) -> None:
    if backup_dir.exists():
        raise FileExistsError(f"{backup_dir} already exists - choose another --backup-name")
    for source in files:
        target = backup_dir / source.relative_to(project_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    mismatched = [f for f in files if sha256_file(f) != sha256_file(backup_dir / f.relative_to(project_dir))]
    if mismatched:
        raise RuntimeError(f"Backup verification failed for {len(mismatched)} file(s), nothing was removed: {mismatched[:5]}")


def clear_readonly_and_retry(func, path, _) -> None:
    try:
        func(path)
    except PermissionError:
        os.chmod(path, stat.S_IWRITE)
        func(path)


def clean(project_dir: Path, execute: bool, confirm: str | None, backup_name: str) -> list[Path]:
    project_dir = project_dir.resolve()
    files = generated_files(project_dir)
    backup_dir = project_dir / "archive" / backup_name
    size = sum(f.stat().st_size for f in files)

    print(f"Generated directories: {', '.join(GENERATED_DIRS)}")
    for directory in target_dirs(project_dir):
        count = sum(1 for f in files if directory in f.parents)
        print(f"  {directory.relative_to(project_dir).as_posix():<16} {count:>5} file(s)")
    print(f"Total: {len(files)} file(s), {size / 1e6:.1f} MB; backup target: {backup_dir.relative_to(project_dir).as_posix()}")
    print(f"Never touched: {', '.join(PROTECTED)}")

    if not execute:
        print("\nDry run - nothing was copied or removed. To clean: python clean_outputs.py --execute --confirm CLEAN")
        return files
    if confirm != CONFIRMATION:
        raise SystemExit(f"Refusing to remove files: pass --confirm {CONFIRMATION} together with --execute")

    backup(project_dir, files, backup_dir)
    print(f"Backup verified (SHA-256 of {len(files)} file(s)) in {backup_dir}")
    if generated_files(project_dir) != files:
        raise RuntimeError("Generated files changed during the backup - nothing was removed, run again")

    for directory in target_dirs(project_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for child in directory.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, onerror=clear_readonly_and_retry)
            else:
                clear_readonly_and_retry(os.unlink, child, None)
    print("Generated directories emptied; run the pipeline to regenerate all artefacts")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Back up and remove generated artefacts (dry run by default)")
    parser.add_argument("--execute", action="store_true", help="really back up and remove; without it only a dry run is printed")
    parser.add_argument("--confirm", default=None, help=f"must be {CONFIRMATION} together with --execute")
    parser.add_argument("--backup-name", default=f"pre_clean_{datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}", help="folder name inside archive/ (must not exist)")
    args = parser.parse_args()
    clean(PROJECT_DIR, args.execute, args.confirm, args.backup_name)


if __name__ == "__main__":
    sys.exit(main())
