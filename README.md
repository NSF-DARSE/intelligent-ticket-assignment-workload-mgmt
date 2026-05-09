# Intelligent Ticket Assignment & Workload Management

## Project Story

This project helps a dispatcher decide **who should handle an open IT support ticket**.

The system starts with raw Autotask ticket data, cleans it, studies historical ticket patterns, checks employee skills and workload, and then recommends the **top 3 technicians** for each unassigned ticket. The final results are stored in PostgreSQL and shown in a Streamlit dashboard for review and dispatch simulation.

In simple terms, the project answers this question:

> “Given a new support ticket, which technician should handle it next?”

## What The Project Does

The current working pipeline:

1. fetches ticket data from the Autotask sandbox
2. cleans and standardizes the ticket fields
3. engineers useful workflow and SLA features
4. normalizes the employee skill dataset
5. compares open tickets with historical completed tickets using **BM25 + MiniLM**
6. scores ticket complexity
7. ranks the **top 3 recommended technicians**
8. loads outputs into PostgreSQL
9. shows the results in a Streamlit dashboard

## Current Model Direction

The active text model is:
- **BM25 = 40%**
- **MiniLM = 60%**

The final recommendation does not rely on text alone. It also uses:
- skill alignment
- technician history
- workload balance
- SLA urgency
- complexity fit

## What Is In Scope Right Now

The current codebase supports:
- Autotask sandbox ingestion
- BM25 + MiniLM similarity
- workload-aware recommendation scoring
- skill-aware recommendation scoring
- local PostgreSQL reporting tables
- Streamlit dashboard and dispatch simulation

## Repository Walkthrough


### Step 1: Read These First

Start with these files in order:

1. `README.md`
2. `main.py`
3. `src/run_sandbox_pipeline.py`
4. `src/load_outputs_to_postgres.py`
5. `src/interactive_dashboard.py`

That order explains:
- what the project is
- how the pipeline runs
- how data moves into PostgreSQL
- how the dashboard reads and presents the results

## Folder Guide

### `src/`

This is the main code folder.

Important files:

- `autotask_api_client.py`
  - low-level Autotask API client
- `fetch_sandbox_tickets.py`
  - fetches raw ticket data
- `clean_ticket_data.py`
  - cleans and standardizes raw tickets
- `feature_engineering.py`
  - builds ticket features and train/open splits
- `clean_employee_skills.py`
  - prepares the employee skill dataset
- `nlp_ticket_similarity.py`
  - builds BM25 + MiniLM similarity outputs
- `complexity_scoring.py`
  - scores ticket complexity
- `assignment_scorer.py`
  - ranks technicians for each ticket
- `load_outputs_to_postgres.py`
  - loads generated outputs into PostgreSQL
- `db_connection.py`
  - quick DB connectivity check
- `interactive_dashboard.py`
  - Streamlit dashboard
- `run_sandbox_pipeline.py`
  - runs the full project workflow in order

Support and evaluation files:

- `benchmark_pipeline.py`
  - measures runtime and memory by stage
- `evaluate_similarity_model.py`
  - evaluates BM25 + MiniLM similarity quality
- `evaluate_full_recommendation_model.py`
  - evaluates the full recommendation model
- `export_raw_data.py`
  - exports raw DB data for inspection

### `data/`

This folder contains generated working outputs.

Important subfolders:

- `Raw_Data/`
- `Cleaned_Data/`
- `Feature_Engineered/`
- `NLP/`
- `Recommendations/`
- `Evaluation/`

These are outputs of the pipeline, not the main implementation.

### `tests/`

This folder contains automated tests.

The current suite checks:
- ticket cleaning behavior
- feature engineering outputs
- recommendation scoring helpers
- local database configuration rules

### `docs/`

This folder contains supporting project documentation.

Main files:

- `docs/API_REFERENCE.md`
- `docs/PERFORMANCE.md`

### Root Files

Important root-level files:

- `main.py`
  - simple launcher for the most common commands
- `requirements.txt`
  - Python dependency list
- `pyproject.toml`
  - project metadata and dependency configuration
- `.env.example`
  - environment variable template
- `CHANGELOG.md`
  - change history
- `RELEASE_NOTES.md`
  - release summary
- `KNOWN_ISSUES.md`
  - known limitations
- `MIGRATION_GUIDE.md`
  - handoff and migration support

## How The Pipeline Flows

Here is the full story of how data moves through the project:

```text
Autotask Sandbox API
        |
        v
Raw ticket export
        |
        v
Cleaning
        |
        v
Feature engineering
        |
        v
Employee skill normalization
        |
        v
BM25 + MiniLM similarity
        |
        v
Complexity scoring
        |
        v
Top-3 technician recommendation scoring
        |
        v
PostgreSQL reporting tables
        |
        v
Streamlit dashboard
```

## Setup 

### 1. Create A Virtual Environment

PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Bash:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

PowerShell:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Bash:

```bash
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
```

### 3. Configure The Environment

Copy:

```text
.env.example
```

to:

```text
.env
```

Then fill in:
- PostgreSQL credentials
- Autotask API credentials

### 4. Add The Employee Skills File

Place this file in the project root:

```text
Skillsdataset.csv
```

### 5. Create The Local Database

The current project expects a local PostgreSQL database such as:

```sql
CREATE DATABASE autotask_local;
```

### 6. Test The Connection

```powershell
.\venv\Scripts\python.exe .\src\db_connection.py
```

## Commands You Will Actually Use

### Run The Full Pipeline

```powershell
.\venv\Scripts\python.exe .\main.py pipeline --all-tickets
```

### Launch The Dashboard

```powershell
.\venv\Scripts\python.exe .\main.py dashboard
```

### Reload PostgreSQL From Existing Outputs

```powershell
.\venv\Scripts\python.exe .\main.py load-postgres
```

### Check Project Status

```powershell
.\venv\Scripts\python.exe .\main.py status
```

## Database Tables You Should Know

The main reporting tables include:

- `autotask_raw`
- `autotask_cleaned_data`
- `autotask_feature_engineered`
- `autotask_open_tickets_dataset`
- `autotask_ticket_similarity_matches`
- `autotask_ticket_similarity_summary`
- `autotask_complexity_scored`
- `autotask_assignment_recommendations`
- `autotask_dashboard_dispatch_actions`

## Dashboard Purpose

The dashboard is built in **Streamlit**.

It is used to:
- review open tickets
- inspect recommendation results
- view employee/workload summaries
- simulate dispatch actions

The largest active file in the repo is `src/interactive_dashboard.py`, so this is the first place to refactor in the future if deeper cleanup is needed.

## Testing

Run the tests with:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The project currently includes automated tests for:
- assignment scorer helpers
- ticket cleaning
- feature pipeline outputs
- database configuration behavior

## Performance Benchmarking

Run the benchmark with:

```powershell
.\venv\Scripts\python.exe .\src\benchmark_pipeline.py
```

Benchmark outputs are written to:

- `data/Evaluation/performance_benchmark.json`
- `data/Evaluation/performance_benchmark.md`

For more detail, see:

- `docs/PERFORMANCE.md`

## What Is Generated Vs What Is Source Code

### Source Code

These are the files teammates should edit and maintain:
- `src/*.py`
- `tests/*.py`
- `main.py`
- `README.md`
- `docs/*.md`
- `requirements.txt`
- `pyproject.toml`

### Generated / Local Files

These are runtime artifacts and should not be treated as hand-edited source:
- `data/Raw_Data/*`
- `data/Cleaned_Data/*`
- `data/Feature_Engineered/*`
- `data/NLP/*`
- `data/Recommendations/*`
- `data/Evaluation/*`
- `.env`

## Recommended Handoff Path
1. read `README.md`
2. check `.env`
3. make sure `Skillsdataset.csv` exists
4. run `main.py status`
5. verify PostgreSQL connection
6. run `main.py pipeline --all-tickets`
7. run `main.py dashboard`

If anything looks wrong, debug in this order:

1. raw data
2. cleaned data
3. feature-engineered data
4. recommendation outputs
5. PostgreSQL tables
6. dashboard

## Supporting Documents

For deeper project detail, use:

- `docs/API_REFERENCE.md`
- `docs/PERFORMANCE.md`
- `CHANGELOG.md`
- `RELEASE_NOTES.md`
- `KNOWN_ISSUES.md`
- `MIGRATION_GUIDE.md`

## Final Note

This repository is strongest when read as a **pipeline project first** and a **dashboard project second**.

The core logic lives in the data preparation, similarity, complexity, and recommendation files. The dashboard is the presentation layer on top of that pipeline.
