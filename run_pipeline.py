from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.manifest import code_version

PROJECT_DIR = Path(__file__).resolve().parent
STAGES = {
    "02": "02_data_understanding.py",
    "03": "03_data_preparation.py",
    "04": "04_modeling.py",
    "05": "05_evaluation.py",
    "05b": "05b_interpretability.py",
    "05c": "05c_counterfactuals.py",
    "05d": "05d_model_comparison.py",
    "05e": "05e_threshold_analysis.py",
    "05f": "05f_model_capacity.py",
    "06": "06_deployment_scoring.py"
}


def local_now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def run_stage(script: str, log_path: Path) -> int:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# {script} started {local_now().isoformat(timespec='seconds')} with {sys.executable}\n")
        process = subprocess.Popen([sys.executable, "-u", script], cwd=PROJECT_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
        log.write(f"# {script} finished {local_now().isoformat(timespec='seconds')} with exit code {code}\n")
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pipeline stages in order with logs")
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES), help="stages to run, in pipeline order")
    parser.add_argument("--log-dir", type=Path, default=None, help="default: logs/<timestamp>")
    args = parser.parse_args()

    log_dir = args.log_dir or PROJECT_DIR / "logs" / local_now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    version = code_version(PROJECT_DIR)
    print(f"Code: commit {version['git_commit']}, uncommitted code changes: {version['git_dirty']}, fingerprint {version['code_fingerprint_sha256'][:16]}")
    if version["git_dirty"]:
        print("Warning: the working tree has uncommitted code changes; commit them so that results can be traced to a commit")
    print(f"Logs: {log_dir}\n")

    for key in [k for k in STAGES if k in args.stages]:
        code = run_stage(STAGES[key], log_dir / f"{Path(STAGES[key]).stem}.log")
        if code != 0:
            print(f"\nStage {key} failed with exit code {code} - see {log_dir}")
            sys.exit(code)

    print(f"\nAll requested stages finished; run IDs and artefact hashes are in results/run_manifest.json, logs in {log_dir}")


if __name__ == "__main__":
    main()
