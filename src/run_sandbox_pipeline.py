from __future__ import annotations

"""Run the end-to-end Autotask recommendation pipeline in the supported order."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_SOURCE_PATH = PROJECT_ROOT / "Skillsdataset.csv"


def resolve_python_executable() -> Path:
    """Prefer the project virtualenv interpreter, but stay cross-platform."""
    candidate_paths = [
        PROJECT_ROOT / "venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / "venv" / "bin" / "python",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch tickets from the Autotask sandbox, run the recommendation pipeline, and load local PostgreSQL."
    )
    parser.add_argument("--days-back", type=int, default=365, help="Fetch tickets created in the last N days.")
    parser.add_argument(
        "--all-tickets",
        action="store_true",
        help="Fetch all sandbox tickets instead of applying the created-date filter.",
    )
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap on fetched records.")
    parser.add_argument(
        "--open-only",
        action="store_true",
        help="Fetch only open tickets. Use with care because the downstream model benefits from historical tickets.",
    )
    return parser.parse_args()


def run_step(command: Iterable[str], label: str) -> None:
    print(f"\n=== {label} ===")
    command_list = [str(part) for part in command]
    print(" ".join(command_list))
    subprocess.run(command_list, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    python_exe = resolve_python_executable()

    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found at {python_exe}")
    if not SKILLS_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Skill dataset not found at {SKILLS_SOURCE_PATH}. "
            "Place Skillsdataset.csv in the project root before running the pipeline."
        )

    fetch_command = [
        python_exe,
        "src/fetch_sandbox_tickets.py",
        "--replace-main-raw",
        "--load-postgres",
    ]

    if args.all_tickets:
        fetch_command.append("--all-tickets")
    else:
        fetch_command.extend(["--days-back", str(args.days_back)])

    if args.max_records is not None:
        fetch_command.extend(["--max-records", str(args.max_records)])
    if args.open_only:
        fetch_command.append("--open-only")

    steps = [
        (fetch_command, "Fetch sandbox tickets"),
        ([python_exe, "src/clean_ticket_data.py"], "Clean ticket data"),
        ([python_exe, "src/feature_engineering.py"], "Build engineered features"),
        ([python_exe, "src/clean_employee_skills.py"], "Normalize employee skills"),
        ([python_exe, "src/nlp_ticket_similarity.py"], "Compute NLP similarity"),
        ([python_exe, "src/complexity_scoring.py"], "Score complexity"),
        ([python_exe, "src/assignment_scorer.py"], "Generate workload-managed skill-aware recommendations"),
    ]

    steps.append(([python_exe, "src/load_outputs_to_postgres.py"], "Load outputs to local PostgreSQL"))

    for command, label in steps:
        run_step(command, label)

    print("\nSandbox pipeline completed successfully.")
    print("Open-ticket recommendations are loaded into local PostgreSQL table autotask_assignment_recommendations.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nPipeline stopped at command: {' '.join(exc.cmd)}", file=sys.stderr)
        raise
