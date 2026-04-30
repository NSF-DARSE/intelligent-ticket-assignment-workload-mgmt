# API Reference

This project is primarily script-driven rather than service-driven. The interfaces below are the main public entry points for users and teammates.

## Main Launcher

File: [main.py](../main.py)

### `python main.py status`
Shows a summary of whether the core generated outputs exist.

### `python main.py pipeline`
Runs the end-to-end backend workflow.

Important arguments:
- `--all-tickets`
- `--days-back`
- `--max-records`
- `--open-only`
- `--skip-output-load`

### `python main.py dashboard`
Launches the Streamlit dashboard.

Important arguments:
- `--port`

### `python main.py load-postgres`
Loads generated CSV/JSON outputs into PostgreSQL.

## Core Pipeline Scripts

### [src/fetch_sandbox_tickets.py](../src/fetch_sandbox_tickets.py)
Fetches ticket data from Autotask sandbox and saves raw data locally and optionally to PostgreSQL.

### [src/clean_ticket_data.py](../src/clean_ticket_data.py)
Normalizes raw ticket records and creates cleaned text/date/numeric fields.

Main callable:
- `clean_ticket_data(input_path: Path) -> tuple[pd.DataFrame, dict]`

### [src/feature_engineering.py](../src/feature_engineering.py)
Builds model features, training/open splits, technician profiles, and workflow indicators.

Main callable:
- `engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]`

### [src/clean_employee_skills.py](../src/clean_employee_skills.py)
Transforms the employee skills input into a profile table and normalized skill table.

Main callables:
- `build_profile_dataset(source_df: pd.DataFrame) -> pd.DataFrame`
- `build_normalized_dataset(profile_df: pd.DataFrame) -> pd.DataFrame`

### [src/nlp_ticket_similarity.py](../src/nlp_ticket_similarity.py)
Compares open tickets with historical tickets using TF-IDF, BM25, and MiniLM embeddings.

### [src/complexity_scoring.py](../src/complexity_scoring.py)
Builds ticket complexity scores and classes from historical and text-derived signals.

### [src/time_estimation_model.py](../src/time_estimation_model.py)
Generates workload-support time estimates from historical patterns.

### [src/assignment_scorer.py](../src/assignment_scorer.py)
Generates the top-3 technician recommendations.

Important helper functions:
- `build_workload_snapshot`
- `infer_required_skills`
- `compute_skill_alignment`
- `recommend_assignments`

### [src/load_outputs_to_postgres.py](../src/load_outputs_to_postgres.py)
Loads local generated outputs into PostgreSQL.

Main callable:
- `get_db_url() -> str`

## Dashboard

### [src/interactive_dashboard.py](../src/interactive_dashboard.py)
Provides the Streamlit interface for:
- overview reporting
- employee views
- recommendation views
- assigned/unassigned ticket board
- dispatch simulation stored in PostgreSQL
