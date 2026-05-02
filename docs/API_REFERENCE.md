# API Reference

This project is script-driven rather than service-driven. The interfaces below are the main entry points for teammates and reviewers.

## Main Launcher

File: [../main.py](../main.py)

### `python main.py status`
Shows whether the core generated outputs exist and reports basic row counts when possible.

### `python main.py pipeline`
Runs the supported end-to-end workflow.

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
Loads the generated CSV and JSON outputs into PostgreSQL.

## Core Pipeline Scripts

### [../src/fetch_sandbox_tickets.py](../src/fetch_sandbox_tickets.py)
Fetches Autotask sandbox tickets and stores the raw export locally and optionally in PostgreSQL.

### [../src/clean_ticket_data.py](../src/clean_ticket_data.py)
Normalizes raw ticket records and creates clean text, date, and numeric fields.

Main callable:
- `clean_ticket_data(input_path: Path) -> tuple[pd.DataFrame, dict]`

### [../src/feature_engineering.py](../src/feature_engineering.py)
Builds workflow-ready ticket features, training/open splits, and technician history profiles.

Main callable:
- `engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]`

### [../src/clean_employee_skills.py](../src/clean_employee_skills.py)
Transforms the skills input into a profile dataset and a normalized skill dataset.

Main callables:
- `build_profile_dataset(source_df: pd.DataFrame) -> pd.DataFrame`
- `build_normalized_dataset(profile_df: pd.DataFrame) -> pd.DataFrame`

### [../src/nlp_ticket_similarity.py](../src/nlp_ticket_similarity.py)
Builds historical similarity outputs for open tickets using a hybrid BM25 + MiniLM retrieval model.

Main callables:
- `prepare_ticket_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]`
- `build_similarity_outputs(completed: pd.DataFrame, open_tickets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]`

### [../src/complexity_scoring.py](../src/complexity_scoring.py)
Assigns ticket complexity scores and classes from effort, SLA, novelty, and keyword signals.

### [../src/assignment_scorer.py](../src/assignment_scorer.py)
Generates the top-3 technician recommendations for each active ticket.

Important helper functions:
- `build_workload_snapshot`
- `infer_required_skills`
- `compute_skill_alignment`
- `build_text_expertise_lookup`
- `recommend_assignments`

### [../src/load_outputs_to_postgres.py](../src/load_outputs_to_postgres.py)
Loads local generated CSV and JSON outputs into PostgreSQL reporting tables.

Main callable:
- `get_db_url() -> str`

## Dashboard

### [../src/interactive_dashboard.py](../src/interactive_dashboard.py)
Provides the Streamlit interface for:
- operational overview reporting
- employee and workload views
- recommendation views
- assigned/unassigned ticket board
- dispatch simulation stored in PostgreSQL
