# API Reference

This project is script-driven. The interfaces below are the stable commands and
callables used by teammates, reviewers, tests, and the dashboard.

Important current scope:
- the active runtime path supports **local PostgreSQL only**
- TF-IDF and the old time estimation model are no longer part of the active flow
- Azure deployment history may appear in branch history, but it is not the
  supported Python runtime target described in this repository guide

## Main Launcher

File: [../main.py](../main.py)

### `python main.py status`

Reports whether core generated outputs exist and prints row counts when the local files are available.

### `python main.py pipeline`

Runs the full supported workflow and loads results into local PostgreSQL.

Arguments:
- `--all-tickets`: fetch all sandbox tickets instead of applying a created-date filter
- `--days-back N`: fetch tickets created in the last `N` days when `--all-tickets` is not used
- `--max-records N`: cap the number of fetched records
- `--open-only`: fetch only open tickets

### `python main.py dashboard`

Launches the Streamlit dashboard.

Arguments:
- `--port N`: dashboard port, default `8501`

### `python main.py load-postgres`

Reloads generated CSV and JSON outputs into local PostgreSQL reporting tables.

## Core Pipeline Scripts

### [../src/fetch_sandbox_tickets.py](../src/fetch_sandbox_tickets.py)

Fetches Autotask sandbox tickets and writes local raw outputs. The `--load-postgres` flag also refreshes `autotask_raw` in local PostgreSQL.

Key arguments:
- `--all-tickets`
- `--days-back`
- `--max-records`
- `--open-only`
- `--replace-main-raw`
- `--load-postgres`

### [../src/clean_ticket_data.py](../src/clean_ticket_data.py)

Normalizes raw ticket records and creates reusable text, date, and numeric fields.

Main callable:
- `clean_ticket_data(input_path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, dict]`

### [../src/feature_engineering.py](../src/feature_engineering.py)

Builds workflow-ready ticket features, training/open splits, and technician history profiles.

Main callable:
- `engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]`

### [../src/clean_employee_skills.py](../src/clean_employee_skills.py)

Transforms `Skillsdataset.csv` into employee profile and normalized skill datasets.

Main callables:
- `build_profile_dataset(source_df: pd.DataFrame) -> pd.DataFrame`
- `build_normalized_dataset(profile_df: pd.DataFrame) -> pd.DataFrame`

### [../src/nlp_ticket_similarity.py](../src/nlp_ticket_similarity.py)

Builds historical similarity outputs for open tickets using BM25 lexical retrieval and MiniLM semantic embeddings.

Main callables:
- `prepare_ticket_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]`
- `build_similarity_outputs(completed: pd.DataFrame, open_tickets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]`

### [../src/complexity_scoring.py](../src/complexity_scoring.py)

Assigns ticket complexity scores and classes from effort, SLA, novelty, and keyword signals.

Key callables:
- `load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]`
- `add_complexity_score(df: pd.DataFrame) -> pd.DataFrame`
- `build_reason(row: pd.Series) -> str`

### [../src/assignment_scorer.py](../src/assignment_scorer.py)

Generates top-3 technician recommendations for active tickets.

Important helper functions:
- `build_workload_snapshot`
- `infer_required_skills`
- `infer_ticket_domains`
- `compute_skill_alignment`
- `build_text_expertise_lookup`
- `recommend_assignments`

### [../src/load_outputs_to_postgres.py](../src/load_outputs_to_postgres.py)

Loads generated CSV and JSON outputs into local PostgreSQL. This module intentionally rejects non-local hosts.

Main callables:
- `get_db_url() -> sqlalchemy.engine.URL`
- `load_csv_tables(engine) -> list[dict]`
- `load_json_tables(engine) -> list[dict]`

### [../src/benchmark_pipeline.py](../src/benchmark_pipeline.py)

Benchmarks pipeline stages and writes JSON/Markdown reports.

Arguments:
- `--include-db-load`: include local PostgreSQL load timing
- `--recompute-similarity`: force BM25 + MiniLM recomputation instead of using cached local similarity outputs

## Dashboard

### [../src/interactive_dashboard.py](../src/interactive_dashboard.py)

Provides the Streamlit interface for:
- operational overview reporting
- employee and workload views
- recommendation inspection
- assigned/unassigned ticket boards
- dispatch simulation stored in `autotask_dashboard_dispatch_actions`

The dashboard reads from local PostgreSQL reporting tables created by `main.py load-postgres` or `main.py pipeline`.
