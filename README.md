# Intelligent Ticket Assignment & Workload Management

## Overview

This project builds an end-to-end recommendation workflow for Autotask service desk tickets.

The pipeline:
- fetches raw ticket data from the Autotask sandbox
- cleans and standardizes ticket records
- engineers ticket-level workflow and SLA features
- normalizes employee skills
- compares open tickets with completed tickets using a hybrid **BM25 + MiniLM** retrieval model
- scores ticket complexity
- ranks the best three technicians for each active ticket
- loads the processed outputs into PostgreSQL
- presents the final results in a Streamlit dashboard

The system is designed to stay explainable. Recommendation scores combine text similarity, skills, ticket history, SLA pressure, complexity, and workload rather than relying on a single opaque model.

Generated datasets, local secrets, Azure helpers, and private input files stay local and are not intended to be committed to Git.

## Current Recommendation Design

The current text similarity model uses:
- **BM25** for lexical retrieval
- **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`) for semantic retrieval

Hybrid text score weights:
- **BM25 = 0.40**
- **MiniLM = 0.60**

TF-IDF and the separate time-estimation model are no longer part of the active project pipeline.

## Workload Management and Skill-Aware Recommendation

The recommendation layer is not text-only. For every active ticket, the scorer combines:

- **Skillset matching** from `Skillsdataset.csv`
- **Historical technician experience** on similar issue types, queues, and accounts
- **BM25 + MiniLM ticket similarity**
- **Live workload management** using current open ticket counts and open estimated hours
- **SLA and complexity balancing**

The workflow uses the employee skill dataset to build:
- `data/Feature_Engineered/employee_skills_profile.csv`
- `data/Feature_Engineered/employee_skills_normalized.csv`

Those files are then used by the scorer to:
- infer required skills from the ticket title, description, and issue type
- match employees whose skillsets fit the ticket
- reduce recommendations for technicians who are already overloaded
- keep the final top-3 recommendation list practical for real dispatching

## Project Flow

The supported workflow is:

1. Configure environment variables and database access
2. Fetch raw Autotask tickets
3. Clean and standardize the dataset
4. Engineer ticket features and training/open splits
5. Clean and normalize employee skills from `Skillsdataset.csv`
6. Build BM25 + MiniLM ticket similarity outputs
7. Score ticket complexity
8. Generate workload-managed, skill-aware top-3 technician recommendations
9. Load outputs into PostgreSQL
10. Review the results in the Streamlit dashboard

## Project Structure

```text
Project_Autotask
|-- sql
|   `-- create_autotask_table.sql
|-- src
|   |-- assignment_scorer.py
|   |-- autotask_api_client.py
|   |-- azure_postgres_setup.py
|   |-- clean_employee_skills.py
|   |-- clean_ticket_data.py
|   |-- complexity_scoring.py
|   |-- db_connection.py
|   |-- export_raw_data.py
|   |-- feature_engineering.py
|   |-- fetch_sandbox_tickets.py
|   |-- interactive_dashboard.py
|   |-- load_outputs_to_postgres.py
|   |-- nlp_ticket_similarity.py
|   `-- run_sandbox_pipeline.py
|-- tests
|   |-- test_assignment_scorer.py
|   |-- test_clean_ticket_data.py
|   `-- test_feature_pipeline.py
|-- docs
|   |-- API_REFERENCE.md
|   `-- PERFORMANCE.md
|-- data/                   # local generated outputs (not committed)
|-- main.py
|-- pyproject.toml
|-- requirements.txt
|-- CHANGELOG.md
|-- RELEASE_NOTES.md
|-- MIGRATION_GUIDE.md
|-- KNOWN_ISSUES.md
|-- SCORING_FORMULA.txt
|-- .env.example
`-- README.md
```

## Setup

### 1. Create a virtual environment

```powershell
python -m venv venv
```

### 2. Activate the virtual environment

```powershell
.\venv\Scripts\activate
```

### 3. Install dependencies

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The repository also includes [pyproject.toml](pyproject.toml) for project metadata and dependency tracking.

### 4. Configure `.env`

Create a `.env` file from [.env.example](.env.example) and provide values for:

- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`
- `DB_SSLMODE` when required
- `AUTOTASK_API_BASE_URL`
- `AUTOTASK_API_USERNAME`
- `AUTOTASK_API_SECRET`
- `AUTOTASK_API_INTEGRATION_CODE`

### 5. Test the database connection

```powershell
.\venv\Scripts\python.exe src\db_connection.py
```

This confirms that the PostgreSQL settings in `.env` are valid.

### 6. Run the tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

This validates the main scoring, cleaning, and feature-engineering logic.

## Pipeline Scripts

### Step 1. Fetch raw Autotask tickets

Script:
- `src/fetch_sandbox_tickets.py`

What it does:
- connects to the Autotask sandbox API
- downloads ticket data
- saves the raw export locally
- optionally refreshes the raw PostgreSQL table

Run:

```powershell
.\venv\Scripts\python.exe src\fetch_sandbox_tickets.py --all-tickets --replace-main-raw --load-postgres
```

Main outputs:
- `data/Raw_Data/autotask_raw_data.csv`
- `data/Raw_Data/autotask_sandbox_raw_data.csv`
- `data/Raw_Data/autotask_sandbox_fetch_summary.json`

### Step 2. Clean ticket data

Script:
- `src/clean_ticket_data.py`

What it does:
- standardizes ticket fields
- normalizes title and description text
- converts dates and numeric values
- prepares reusable text for later modeling

Run:

```powershell
.\venv\Scripts\python.exe src\clean_ticket_data.py
```

Main outputs:
- `data/Cleaned_Data/autotask_cleaned_data.csv`
- `data/Cleaned_Data/autotask_cleaning_summary.json`

### Step 3. Build feature-engineered datasets

Script:
- `src/feature_engineering.py`

What it does:
- creates workflow and SLA features
- separates open tickets from completed tickets
- builds training and scoring datasets
- creates technician history profiles

Run:

```powershell
.\venv\Scripts\python.exe src\feature_engineering.py
```

Main outputs:
- `data/Feature_Engineered/autotask_feature_engineered.csv`
- `data/Feature_Engineered/autotask_open_tickets_dataset.csv`
- `data/Feature_Engineered/autotask_training_dataset.csv`
- `data/Feature_Engineered/technician_profiles.csv`
- `data/Feature_Engineered/feature_engineering_summary.json`

### Step 4. Normalize employee skills

Script:
- `src/clean_employee_skills.py`

Input:
- `Skillsdataset.csv`

What it does:
- cleans the employee skills source file
- builds a dashboard-friendly profile dataset
- builds a normalized skill-per-row dataset
- creates technician keys for joins and scoring

Run:

```powershell
.\venv\Scripts\python.exe src\clean_employee_skills.py
```

Main outputs:
- `data/Feature_Engineered/employee_skills_profile.csv`
- `data/Feature_Engineered/employee_skills_normalized.csv`
- `data/Feature_Engineered/employee_skills_summary.json`

### Step 5. Build BM25 + MiniLM similarity outputs

Script:
- `src/nlp_ticket_similarity.py`

What it does:
- compares open tickets with completed tickets
- builds lexical similarity using BM25
- builds semantic similarity using MiniLM embeddings
- blends both signals into a hybrid similarity score
- estimates a historical resolution-hours hint from the closest completed tickets

Run:

```powershell
.\venv\Scripts\python.exe src\nlp_ticket_similarity.py
```

Main outputs:
- `data/NLP/ticket_similarity_matches.csv`
- `data/NLP/ticket_similarity_summary.csv`
- `data/NLP/nlp_similarity_summary.json`

### Step 6. Score ticket complexity

Script:
- `src/complexity_scoring.py`

What it does:
- combines historical effort, SLA pressure, novelty, and keyword signals
- assigns a complexity score and complexity class
- creates a human-readable complexity reason

Run:

```powershell
.\venv\Scripts\python.exe src\complexity_scoring.py
```

Main outputs:
- `data/Complexity/autotask_complexity_scored.csv`
- `data/Complexity/complexity_scoring_summary.json`

### Step 7. Generate technician recommendations

Script:
- `src/assignment_scorer.py`

What it does:
- scores each technician against each active ticket
- combines issue-type history, skills, BM25/MiniLM text expertise, workload, SLA pressure, and complexity
- ranks and keeps the top 3 technician recommendations

Run:

```powershell
.\venv\Scripts\python.exe src\assignment_scorer.py
```

Main outputs:
- `data/Recommendations/assignment_recommendations.csv`
- `data/Recommendations/technician_workload_snapshot.csv`
- `data/Recommendations/recommendation_summary.json`

### Step 8. Load outputs into PostgreSQL

Script:
- `src/load_outputs_to_postgres.py`

What it does:
- loads the generated CSV outputs into PostgreSQL reporting tables
- loads summary JSON files into PostgreSQL summary tables
- prepares the data used by the dashboard and downstream reviews

Run:

```powershell
.\venv\Scripts\python.exe src\load_outputs_to_postgres.py
```

### Step 9. Open the dashboard

Script:
- `src/interactive_dashboard.py`

What it does:
- shows KPIs, employee views, recommendation views, and the ticket assignment board
- supports simulated dispatch actions stored in PostgreSQL

Run:

```powershell
.\venv\Scripts\python.exe -m streamlit run src\interactive_dashboard.py
```

## One-Command Pipeline

Run the full backend workflow from ticket fetch through PostgreSQL load:

```powershell
.\venv\Scripts\python.exe src\run_sandbox_pipeline.py --all-tickets
```

This runs:
1. ticket fetch
2. ticket cleaning
3. feature engineering
4. employee skill normalization
5. BM25 + MiniLM similarity
6. complexity scoring
7. recommendation scoring
8. PostgreSQL loading

## Main Project Launcher

Use [main.py](main.py) as the single entry point for the common workflows.

### Show project status

```powershell
.\venv\Scripts\python.exe main.py status
```

### Run the full pipeline

```powershell
.\venv\Scripts\python.exe main.py pipeline --all-tickets
```

### Load outputs into PostgreSQL

```powershell
.\venv\Scripts\python.exe main.py load-postgres
```

### Start the dashboard

```powershell
.\venv\Scripts\python.exe main.py dashboard
```

## Recommended Run Order for a New User

1. create and activate the virtual environment
2. install dependencies
3. configure `.env`
4. test the PostgreSQL connection
5. fetch Autotask tickets
6. clean tickets
7. engineer ticket features
8. normalize employee skills
9. run BM25 + MiniLM similarity
10. score complexity
11. generate recommendations
12. load outputs into PostgreSQL
13. open the dashboard

## Release and Project Documents

- [CHANGELOG.md](CHANGELOG.md)
- [RELEASE_NOTES.md](RELEASE_NOTES.md)
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- [SCORING_FORMULA.txt](SCORING_FORMULA.txt)

## License

This project is released under the MIT License.

See [LICENSE](LICENSE) for the full license text.
