from __future__ import annotations

"""Run the end-to-end Autotask recommendation pipeline in the supported order."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
SKILLS_SOURCE_PATH = PROJECT_ROOT / "Skillsdataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch tickets from the Autotask sandbox and run the full recommendation pipeline."
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
    parser.add_argument(
        "--skip-output-load",
        action="store_true",
        help="Skip the final load of generated outputs into PostgreSQL.",
    )
    return parser.parse_args()


def run_step(command: list[str], label: str) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()

    if not PYTHON_EXE.exists():
        raise FileNotFoundError(f"Python executable not found at {PYTHON_EXE}")
    if not SKILLS_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Skill dataset not found at {SKILLS_SOURCE_PATH}. "
            "Place Skillsdataset.csv in the project root before running the pipeline."
        )

    fetch_command = [
        str(PYTHON_EXE),
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
        ([str(PYTHON_EXE), "src/clean_ticket_data.py"], "Clean ticket data"),
        ([str(PYTHON_EXE), "src/feature_engineering.py"], "Build engineered features"),
        ([str(PYTHON_EXE), "src/clean_employee_skills.py"], "Normalize employee skills"),
        ([str(PYTHON_EXE), "src/nlp_ticket_similarity.py"], "Compute NLP similarity"),
        ([str(PYTHON_EXE), "src/complexity_scoring.py"], "Score complexity"),
        ([str(PYTHON_EXE), "src/assignment_scorer.py"], "Generate workload-managed skill-aware recommendations"),
    ]

    if not args.skip_output_load:
        steps.append(([str(PYTHON_EXE), "src/load_outputs_to_postgres.py"], "Load outputs to PostgreSQL"))

    for command, label in steps:
        run_step(command, label)

    print("\nSandbox pipeline completed successfully.")
    print("Open-ticket recommendations are available in data/Recommendations/assignment_recommendations.csv")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nPipeline stopped at command: {' '.join(exc.cmd)}", file=sys.stderr)
        raise
