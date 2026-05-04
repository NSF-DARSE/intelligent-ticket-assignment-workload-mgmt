# Intelligent Ticket Assignment & Workload Management

## Overview

This project builds a reproducible, local PostgreSQL-backed recommendation workflow for Autotask service desk tickets. It ingests sandbox tickets, prepares clean analytical datasets, ranks technicians for active tickets, and presents the results in a Streamlit dashboard.

The pipeline:
- fetches raw ticket data from the Autotask sandbox API
- cleans and standardizes ticket records
- engineers ticket, SLA, workload, and technician history features
- normalizes employee skills from `Skillsdataset.csv`
- compares open tickets with completed tickets using **BM25 + MiniLM**
- scores ticket complexity from effort, SLA pressure, novelty, and keyword signals
- ranks the top three technicians for each active ticket
- stores generated results in **local PostgreSQL**
- presents KPIs, technician views, recommendations, and dispatch simulation in Streamlit

Generated datasets, `.env` secrets, local PostgreSQL credentials, and private Autotask inputs are intentionally local-only and are not committed to Git.

## Architecture

```text
Autotask Sandbox API
        |
        v
data/Raw_Data/*.csv
        |
        v
cleaning -> feature engineering -> skills -> BM25 + MiniLM -> complexity -> recommendations
        |
        v
local PostgreSQL tables
        |
        v
Streamlit dashboard
```

Primary local PostgreSQL tables:
- `autotask_raw`
- `autotask_cleaned_data`
- `autotask_feature_engineered`
- `autotask_open_tickets_dataset`
- `autotask_ticket_similarity_matches`
- `autotask_ticket_similarity_summary`
- `autotask_complexity_scored`
- `autotask_assignment_recommendations`
- `autotask_dashboard_dispatch_actions`

## Recommendation Design

The current similarity stack uses:
- **BM25** for lexical retrieval
- **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`) for semantic similarity

Hybrid text score weights:
- **BM25 = 0.40**
- **MiniLM = 0.60**

The recommendation layer combines text similarity with operational constraints:
- required skill inference from title, description, issue type, and keyword signals
- employee skill matching from `Skillsdataset.csv`
- historical technician experience on similar tickets
- current workload and estimated open hours
- SLA urgency, ticket priority, and complexity class

The model is intentionally explainable. Recommendation rows include score components and rationale fields so reviewers can inspect why a technician was ranked.

## Project Structure

```text
Project_Autotask
|-- .github/workflows/       # CI and local validation workflows
|-- docs/
|   |-- API_REFERENCE.md
|   |-- PERFORMANCE.md
|   |-- PRESENTATION_OUTLINE.md
|   `-- RUBRIC_COVERAGE.md
|-- sql/
|   `-- create_autotask_table.sql
|-- src/
|   |-- assignment_scorer.py
|   |-- autotask_api_client.py
|   |-- benchmark_pipeline.py
|   |-- clean_employee_skills.py
|   |-- clean_ticket_data.py
|   |-- complexity_scoring.py
|   |-- db_connection.py
|   |-- feature_engineering.py
|   |-- fetch_sandbox_tickets.py
|   |-- interactive_dashboard.py
|   |-- load_outputs_to_postgres.py
|   |-- nlp_ticket_similarity.py
|   `-- run_sandbox_pipeline.py
|-- tests/
|   |-- test_assignment_scorer.py
|   |-- test_clean_ticket_data.py
|   |-- test_database_config.py
|   `-- test_feature_pipeline.py
|-- CHANGELOG.md
|-- KNOWN_ISSUES.md
|-- LICENSE
|-- MIGRATION_GUIDE.md
|-- README.md
|-- RELEASE_NOTES.md
|-- SCORING_FORMULA.txt
|-- main.py
|-- pyproject.toml
|-- requirements.txt
`-- .env.example
```

## Prerequisites

- Python `3.13`
- Local PostgreSQL running on `localhost:5432`
- Autotask API credentials
- `Skillsdataset.csv` in the project root
- Internet access the first time MiniLM is downloaded, unless the model is already cached

The project has been run locally on Windows and validated in GitHub Actions on Ubuntu for dependency installation, import checks, and automated tests.

## Setup

### 1. Create And Activate A Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dependency metadata is also tracked in [pyproject.toml](pyproject.toml). Runtime pins live in [requirements.txt](requirements.txt).

### 3. Create The Local PostgreSQL Database

Create a local database named `autotask_local` in pgAdmin or with `psql`:

```sql
CREATE DATABASE autotask_local;
```

The code intentionally rejects non-local database hosts. `DB_HOST` must be one of:
- `localhost`
- `127.0.0.1`
- `::1`

### 4. Configure `.env`

Copy [.env.example](.env.example) to `.env` and fill in values:

```env
DB_USER=postgres
DB_PASSWORD=your_local_postgres_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=autotask_local
DB_SSLMODE=

AUTOTASK_API_BASE_URL=https://webservices2.autotask.net/ATServicesRest
AUTOTASK_API_USERNAME=your_autotask_username
AUTOTASK_API_SECRET=your_autotask_secret
AUTOTASK_API_INTEGRATION_CODE=your_tracking_identifier
```

### 5. Verify The Database Connection

```powershell
.\venv\Scripts\python.exe src\db_connection.py
```

Expected result: the script prints the connected database, connected user, and a row count for `autotask_raw` after data has been loaded.

## Running The Project

### Recommended One-Command Workflow

```powershell
.\venv\Scripts\python.exe main.py pipeline --all-tickets
```

This command:
1. fetches Autotask tickets
2. cleans the raw data
3. builds engineered features
4. normalizes employee skills
5. builds BM25 + MiniLM similarity outputs
6. scores complexity
7. generates technician recommendations
8. loads CSV/JSON outputs into local PostgreSQL

### Launch The Dashboard

```powershell
.\venv\Scripts\python.exe main.py dashboard
```

Then open:

```text
http://localhost:8501
```

### Reload PostgreSQL From Existing Local Outputs

```powershell
.\venv\Scripts\python.exe main.py load-postgres
```

Use this when CSV/JSON outputs already exist and only the local PostgreSQL reporting tables need refreshing.

### Show Output Status

```powershell
.\venv\Scripts\python.exe main.py status
```

## Individual Pipeline Scripts

| Step | Script | Main Output |
|---|---|---|
| Fetch tickets | `src/fetch_sandbox_tickets.py` | `data/Raw_Data/autotask_raw_data.csv`, `autotask_raw` |
| Clean tickets | `src/clean_ticket_data.py` | `data/Cleaned_Data/autotask_cleaned_data.csv` |
| Engineer features | `src/feature_engineering.py` | `data/Feature_Engineered/*.csv` |
| Normalize skills | `src/clean_employee_skills.py` | `employee_skills_profile.csv`, `employee_skills_normalized.csv` |
| NLP similarity | `src/nlp_ticket_similarity.py` | `ticket_similarity_matches.csv`, `ticket_similarity_summary.csv` |
| Complexity scoring | `src/complexity_scoring.py` | `autotask_complexity_scored.csv` |
| Recommendations | `src/assignment_scorer.py` | `assignment_recommendations.csv` |
| PostgreSQL load | `src/load_outputs_to_postgres.py` | local PostgreSQL reporting tables |
| Dashboard | `src/interactive_dashboard.py` | Streamlit UI |

## Testing And CI

Run the full test suite:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Current coverage includes:
- unit tests for technician key cleanup, skill inference, domain inference, and scoring helpers
- unit tests for ticket cleaning and normalization
- integration-style tests for feature engineering and skill profile generation
- configuration tests that ensure database URLs are local-only
- GitHub Actions for dependency installation, import validation, and test execution

The latest local validation run passed with all tests.

## Performance And Resource Awareness

Performance notes are documented in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

Run the benchmark:

```powershell
.\venv\Scripts\python.exe src\benchmark_pipeline.py
```

By default, the benchmark reuses cached local similarity outputs to avoid failing when MiniLM cannot be downloaded. To force full BM25 + MiniLM recomputation:

```powershell
.\venv\Scripts\python.exe src\benchmark_pipeline.py --recompute-similarity
```

The most expensive stages are MiniLM embedding generation and recommendation scoring over active-ticket/technician combinations. PostgreSQL loading adds I/O cost but improves traceability and dashboard reproducibility.

## Packaging And Release Readiness

Release metadata:
- package metadata: [pyproject.toml](pyproject.toml)
- pinned dependencies: [requirements.txt](requirements.txt)
- current project version: `1.0.0`
- license: [LICENSE](LICENSE)
- release notes: [RELEASE_NOTES.md](RELEASE_NOTES.md)
- changelog: [CHANGELOG.md](CHANGELOG.md)
- migration guide: [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- known issues: [KNOWN_ISSUES.md](KNOWN_ISSUES.md)

Create a release tag after final validation:

```powershell
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

## Rubric Evidence

The project includes a rubric mapping in [docs/RUBRIC_COVERAGE.md](docs/RUBRIC_COVERAGE.md). Presentation structure is summarized in [docs/PRESENTATION_OUTLINE.md](docs/PRESENTATION_OUTLINE.md).

## Known Limitations

- The dashboard is functional but still concentrated in one large Streamlit file.
- Dispatch actions are simulated in local PostgreSQL and are not pushed back to Autotask.
- Generated datasets are not committed; a fresh user must run the pipeline.
- MiniLM must be downloadable or cached for a full recomputation run.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full license text.
