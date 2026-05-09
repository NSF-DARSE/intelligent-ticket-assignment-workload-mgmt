# Intelligent Ticket Assignment & Workload Management

## What This Repository Is

This repository contains the **current working codebase** for an Autotask ticket
recommendation workflow. It is designed to help a dispatcher review open tickets,
compare them with historical work, and see the **top three recommended
technicians** for each ticket.

The project is intended to be:
- understandable for a new teammate taking over the code
- reproducible on a normal development machine
- explainable enough for software engineering review and rubric-based assessment

## Current Supported Architecture

The active project flow is:

```text
Autotask Sandbox API
        |
        v
Raw ticket export
        |
        v
Cleaning -> Feature engineering -> Skill normalization
        |
        v
BM25 + MiniLM similarity -> Complexity scoring -> Recommendation scoring
        |
        v
Local PostgreSQL reporting tables
        |
        v
Streamlit dashboard
```

### Important Current Scope

The current Python codebase supports:
- **Autotask sandbox ticket ingestion**
- **BM25 + MiniLM similarity**
- **skill-aware and workload-aware recommendation scoring**
- **local PostgreSQL reporting tables**
- **Streamlit dashboard and dispatch simulation**

The current Python codebase does **not** support:
- TF-IDF in the active similarity model
- the removed time estimation model
- non-local PostgreSQL hosts in the active loader/runtime path

## Main Project Goal

Given an open Autotask ticket, the system:
1. prepares clean ticket text and operational metadata
2. compares the ticket to historical completed tickets
3. combines text similarity, skill fit, workload, SLA pressure, and complexity
4. ranks the **top 3 technicians**
5. shows those recommendations in the dashboard

## Core Model Design

### Text Similarity

The active text similarity model is:
- **BM25 = 40%**
- **MiniLM = 60%**

This logic lives in:
- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\nlp_ticket_similarity.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\nlp_ticket_similarity.py)

### Recommendation Scoring

The final technician ranking uses:
- text expertise from BM25 + MiniLM
- issue-type and skill alignment
- queue/domain fit
- account familiarity
- workload balance
- SLA urgency
- complexity fit

This logic lives in:
- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\assignment_scorer.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\assignment_scorer.py)

## Files New Teammates Should Read First

If someone is taking over the project, read these in order:

1. [D:\MS 2024\SEM4\RSE\Project_Autotask\README.md](D:\MS 2024\SEM4\RSE\Project_Autotask\README.md)
2. [D:\MS 2024\SEM4\RSE\Project_Autotask\main.py](D:\MS 2024\SEM4\RSE\Project_Autotask\main.py)
3. [D:\MS 2024\SEM4\RSE\Project_Autotask\src\run_sandbox_pipeline.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\run_sandbox_pipeline.py)
4. [D:\MS 2024\SEM4\RSE\Project_Autotask\src\load_outputs_to_postgres.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\load_outputs_to_postgres.py)
5. [D:\MS 2024\SEM4\RSE\Project_Autotask\src\interactive_dashboard.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\interactive_dashboard.py)

## File Guide

### Entry Points

- [D:\MS 2024\SEM4\RSE\Project_Autotask\main.py](D:\MS 2024\SEM4\RSE\Project_Autotask\main.py)
  - simple launcher for pipeline, dashboard, load, and status commands
- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\run_sandbox_pipeline.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\run_sandbox_pipeline.py)
  - runs the supported pipeline stages in the correct order

### Data Ingestion

- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\autotask_api_client.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\autotask_api_client.py)
  - low-level Autotask API client
- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\fetch_sandbox_tickets.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\fetch_sandbox_tickets.py)
  - fetches sandbox tickets and writes raw outputs

### Data Preparation

- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\clean_ticket_data.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\clean_ticket_data.py)
  - normalizes ticket fields
- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\feature_engineering.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\feature_engineering.py)
  - creates training/open splits and ticket features
- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\clean_employee_skills.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\clean_employee_skills.py)
  - converts `Skillsdataset.csv` into profile and normalized skill tables

### Modeling

- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\nlp_ticket_similarity.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\nlp_ticket_similarity.py)
  - BM25 + MiniLM similarity outputs
- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\complexity_scoring.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\complexity_scoring.py)
  - complexity score and reason generation
- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\assignment_scorer.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\assignment_scorer.py)
  - final recommendation ranking and explanation fields

### Database And Dashboard

- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\load_outputs_to_postgres.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\load_outputs_to_postgres.py)
  - loads CSV/JSON outputs into PostgreSQL
  - currently supports **local PostgreSQL only**
- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\db_connection.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\db_connection.py)
  - quick DB connectivity check
- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\interactive_dashboard.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\interactive_dashboard.py)
  - Streamlit dashboard

### Evaluation And Benchmarking

- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\evaluate_similarity_model.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\evaluate_similarity_model.py)
  - evaluates BM25 + MiniLM similarity ranking
- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\evaluate_full_recommendation_model.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\evaluate_full_recommendation_model.py)
  - evaluates the full recommendation scorer
- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\benchmark_pipeline.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\benchmark_pipeline.py)
  - measures runtime and peak memory by stage

### Utility Script

- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\export_raw_data.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\export_raw_data.py)
  - exports raw DB data for inspection
  - useful support script, not part of the main pipeline

## Setup For A New Developer

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

### 3. Add Local Configuration

Copy:
- [D:\MS 2024\SEM4\RSE\Project_Autotask\.env.example](D:\MS 2024\SEM4\RSE\Project_Autotask\.env.example)

to:
- `.env`

Fill in:
- local PostgreSQL credentials
- Autotask API credentials

### 4. Provide The Skill Dataset

Place this file in the project root:
- `Skillsdataset.csv`

### 5. Create The Local Database

The active code expects a **local PostgreSQL database**. A common setup is:

```sql
CREATE DATABASE autotask_local;
```

### 6. Check The Database Connection

```powershell
.\venv\Scripts\python.exe .\src\db_connection.py
```

## Commands People Will Actually Use

### Full Pipeline

```powershell
.\venv\Scripts\python.exe .\main.py pipeline --all-tickets
```

### Dashboard

```powershell
.\venv\Scripts\python.exe .\main.py dashboard
```

### Reload PostgreSQL From Existing Outputs

```powershell
.\venv\Scripts\python.exe .\main.py load-postgres
```

### Project Status

```powershell
.\venv\Scripts\python.exe .\main.py status
```

## What Gets Generated Locally

These are generated working artifacts and should not be treated as source code:
- `data/Raw_Data/*`
- `data/Cleaned_Data/*`
- `data/Feature_Engineered/*`
- `data/NLP/*`
- `data/Recommendations/*`
- `data/Evaluation/*`
- `.env`

These files help the pipeline and dashboard run, but they are outputs, not the core implementation.

## Testing

Run the test suite:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The current test suite covers:
- ticket cleaning behavior
- feature engineering outputs
- recommendation scoring helpers
- local database configuration restrictions

## Benchmarking

Run:

```powershell
.\venv\Scripts\python.exe .\src\benchmark_pipeline.py
```

Generated benchmark artifacts are written to:
- `data/Evaluation/performance_benchmark.json`
- `data/Evaluation/performance_benchmark.md`

See:
- [D:\MS 2024\SEM4\RSE\Project_Autotask\docs\PERFORMANCE.md](D:\MS 2024\SEM4\RSE\Project_Autotask\docs\PERFORMANCE.md)

## Known Readability Hotspot

The largest remaining file is:
- [D:\MS 2024\SEM4\RSE\Project_Autotask\src\interactive_dashboard.py](D:\MS 2024\SEM4\RSE\Project_Autotask\src\interactive_dashboard.py)

It is working and validated, but if someone continues development, this is the first file worth splitting into smaller helpers or modules.

## Handoff Notes

If another teammate takes over this repo, the safest order is:

1. verify `.env`
2. verify local PostgreSQL connection
3. verify `Skillsdataset.csv` is present
4. run `main.py status`
5. run `main.py pipeline --all-tickets`
6. run `main.py dashboard`

If something looks wrong, inspect:
- raw ticket data first
- engineered outputs second
- recommendation CSVs third
- dashboard last

## Supporting Docs

- [D:\MS 2024\SEM4\RSE\Project_Autotask\docs\API_REFERENCE.md](D:\MS 2024\SEM4\RSE\Project_Autotask\docs\API_REFERENCE.md)
- [D:\MS 2024\SEM4\RSE\Project_Autotask\docs\PERFORMANCE.md](D:\MS 2024\SEM4\RSE\Project_Autotask\docs\PERFORMANCE.md)
- [D:\MS 2024\SEM4\RSE\Project_Autotask\CHANGELOG.md](D:\MS 2024\SEM4\RSE\Project_Autotask\CHANGELOG.md)
- [D:\MS 2024\SEM4\RSE\Project_Autotask\RELEASE_NOTES.md](D:\MS 2024\SEM4\RSE\Project_Autotask\RELEASE_NOTES.md)
- [D:\MS 2024\SEM4\RSE\Project_Autotask\KNOWN_ISSUES.md](D:\MS 2024\SEM4\RSE\Project_Autotask\KNOWN_ISSUES.md)
- [D:\MS 2024\SEM4\RSE\Project_Autotask\MIGRATION_GUIDE.md](D:\MS 2024\SEM4\RSE\Project_Autotask\MIGRATION_GUIDE.md)
