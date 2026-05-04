from __future__ import annotations

"""Convenience launcher for the main project workflows."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"

PIPELINE_SCRIPT = PROJECT_ROOT / "src" / "run_sandbox_pipeline.py"
DASHBOARD_SCRIPT = PROJECT_ROOT / "src" / "interactive_dashboard.py"
LOAD_OUTPUTS_SCRIPT = PROJECT_ROOT / "src" / "load_outputs_to_postgres.py"

RAW_DATA_PATH = PROJECT_ROOT / "data" / "Raw_Data" / "autotask_raw_data.csv"
OPEN_DATASET_PATH = PROJECT_ROOT / "data" / "Feature_Engineered" / "autotask_open_tickets_dataset.csv"
RECOMMENDATIONS_PATH = PROJECT_ROOT / "data" / "Recommendations" / "assignment_recommendations.csv"
SKILLS_PROFILE_PATH = PROJECT_ROOT / "data" / "Feature_Engineered" / "employee_skills_profile.csv"
WORKLOAD_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "Recommendations" / "technician_workload_snapshot.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Main project launcher for the Autotask AI ticket assignment system."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Show a quick summary of project outputs.")
    status_parser.set_defaults(func=show_status)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full local PostgreSQL-backed pipeline.")
    pipeline_parser.add_argument(
        "--days-back",
        type=int,
        default=365,
        help="Fetch tickets created in the last N days when --all-tickets is not used.",
    )
    pipeline_parser.add_argument(
        "--all-tickets",
        action="store_true",
        help="Fetch all sandbox tickets instead of filtering by created date.",
    )
    pipeline_parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on the number of fetched records.",
    )
    pipeline_parser.add_argument(
        "--open-only",
        action="store_true",
        help="Fetch only open tickets.",
    )
    pipeline_parser.set_defaults(func=run_pipeline)

    dashboard_parser = subparsers.add_parser("dashboard", help="Launch the Streamlit dashboard.")
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for the Streamlit dashboard.",
    )
    dashboard_parser.set_defaults(func=run_dashboard)

    postgres_parser = subparsers.add_parser(
        "load-postgres",
        help="Load the generated CSV and JSON outputs into local PostgreSQL.",
    )
    postgres_parser.set_defaults(func=run_load_outputs)

    return parser


def ensure_python() -> Path:
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Virtual environment python was not found at {VENV_PYTHON}")
    return VENV_PYTHON


def run_command(command: list[str], label: str) -> int:
    print(f"\n=== {label} ===")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return completed.returncode


def show_status(_: argparse.Namespace) -> int:
    print("Autotask AI Project Status")
    print(f"Project root: {PROJECT_ROOT}")

    status_items = [
        ("Raw ticket dataset", RAW_DATA_PATH),
        ("Open ticket dataset", OPEN_DATASET_PATH),
        ("Employee skills profile", SKILLS_PROFILE_PATH),
        ("Technician workload snapshot", WORKLOAD_SNAPSHOT_PATH),
        ("Recommendation output", RECOMMENDATIONS_PATH),
    ]

    for label, path in status_items:
        if path.exists():
            row_count = "unavailable"
            try:
                row_count = f"{len(pd.read_csv(path)):,} rows"
            except Exception:
                pass
            print(f"- {label}: available at {path} ({row_count})")
        else:
            print(f"- {label}: not found")

    print("\nUseful commands:")
    print("  python main.py pipeline --all-tickets")
    print("  python main.py dashboard")
    print("  python main.py load-postgres")
    print("\nThe pipeline refreshes local PostgreSQL tables, employee skills, and workload-managed recommendations automatically.")
    return 0


def run_pipeline(args: argparse.Namespace) -> int:
    python_exe = ensure_python()
    command = [str(python_exe), str(PIPELINE_SCRIPT)]

    if args.all_tickets:
        command.append("--all-tickets")
    else:
        command.extend(["--days-back", str(args.days_back)])

    if args.max_records is not None:
        command.extend(["--max-records", str(args.max_records)])
    if args.open_only:
        command.append("--open-only")
    return run_command(command, "Run Full Sandbox Pipeline")


def run_dashboard(args: argparse.Namespace) -> int:
    python_exe = ensure_python()
    command = [
        str(python_exe),
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_SCRIPT),
        "--server.port",
        str(args.port),
    ]
    return run_command(command, "Launch Streamlit Dashboard")


def run_load_outputs(_: argparse.Namespace) -> int:
    python_exe = ensure_python()
    command = [str(python_exe), str(LOAD_OUTPUTS_SCRIPT)]
    return run_command(command, "Load Outputs To Local PostgreSQL")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
