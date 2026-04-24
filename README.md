# Autotask AI Project

## Project Overview

This project is building an AI-powered Intelligent Ticket Assignment and Workload Optimization System using historical Autotask ticket data.

The goal is to recommend the most suitable technician for incoming tickets by combining:
- ticket analysis
- resolution time estimation
- technician workload analysis
- skill matching
- priority balancing
- SLA-aware prioritization
- NLP-based ticket similarity with TF-IDF and BM25 hybrid ranking
- sandbox API ingestion for live Autotask ticket refreshes

The current implementation is focused on the data foundation needed for that system: raw data access, cleaning, and feature engineering.

## Current Status

Completed so far:
- Phase 1: data infrastructure setup
- Phase 2: ticket data cleaning
- Phase 2: feature engineering and technician profiling
- Phase 2: baseline SLA-aware assignment recommendation logic
- Phase 2: NLP ticket similarity pipeline and NLP-enhanced recommendations
- Phase 2: BM25-enhanced text expertise scoring for recommendations
- Phase 2: time estimation model with safe hybrid prediction
- Phase 2: explainable complexity scoring
- Sandbox API pipeline for fetching live Autotask tickets into PostgreSQL
- PostgreSQL loading of generated analytics outputs
- Dashboard reporting for employee and ticket analytics
- Interactive Streamlit dashboard for live filtering, drill-down, workload management, and exportable reports

Current outputs available:
- raw ticket export
- cleaned ticket dataset
- feature-engineered dataset
- training dataset for completed tickets
- open-ticket dataset for recommendation logic
- technician profile dataset
- technician workload snapshot
- SLA-aware assignment recommendations
- NLP similarity matches and summaries
- BM25/TF-IDF hybrid text match scores
- time estimation metrics and predictions
- complexity-scored ticket dataset
- PostgreSQL analytics tables for processed outputs
- interactive dashboard app with filters, KPI cards, recommendation views, and downloadable tables
- open-ticket recommendation board with current assignee and top-3 recommended technicians

## Project Structure

```text
Project_Autotask
|-- data
|   |-- Raw_Data
|   |-- Cleaned_Data
|   |-- Feature_Engineered
|   |-- NLP
|   |-- Time_Estimation
|   |-- Complexity
|   `-- Recommendations
|-- notebooks
|-- reports
|   `-- dashboard
|-- sql
|   `-- create_autotask_table.sql
|-- src
|   |-- clean_ticket_data.py
|   |-- assignment_scorer.py
|   |-- autotask_api_client.py
|   |-- db_connection.py
|   |-- export_raw_data.py
|   |-- fetch_sandbox_tickets.py
|   |-- feature_engineering.py
|   |-- interactive_dashboard.py
|   |-- complexity_scoring.py
|   |-- load_outputs_to_postgres.py
|   |-- nlp_ticket_similarity.py
|   |-- run_sandbox_pipeline.py
|   |-- time_estimation_model.py
|   `-- profiler_test.py
|-- requirements.txt
|-- README.md
`-- venv
```

## Phase 1: Data Infrastructure Setup

Phase 1 established the storage and access layer for historical Autotask ticket data.

Completed items:
- PostgreSQL setup for ticket storage
- raw ticket table creation
- Python database connection using `psycopg2`
- raw data export to CSV
- project dependency setup in a virtual environment

Key scripts:
- `src/db_connection.py`
- `src/export_raw_data.py`

## Sandbox API Integration Setup

The project can now pull live tickets from an Autotask sandbox API and normalize them into the same raw schema used by the existing Phase 2 pipeline.

New scripts:
- `src/autotask_api_client.py`
- `src/fetch_sandbox_tickets.py`

Configuration:
- copy `.env.example` values into `.env`
- set the sandbox API credentials:
  - `AUTOTASK_API_BASE_URL`
  - `AUTOTASK_API_USERNAME`
  - `AUTOTASK_API_SECRET`
  - `AUTOTASK_API_INTEGRATION_CODE`
- optional:
  - `AUTOTASK_ZONE_INFO_URL`
  - `AUTOTASK_PAGE_SIZE`

What the sandbox fetch does:
- reads ticket data from the sandbox API
- resolves company/resource/queue names when reference endpoints are available
- maps the API response into the project raw-ticket schema
- writes:
  - `data/Raw_Data/autotask_sandbox_raw_data.csv`
  - `data/Raw_Data/autotask_sandbox_fetch_summary.json`
- optionally:
  - replaces `data/Raw_Data/autotask_raw_data.csv`
  - refreshes PostgreSQL table `autotask_raw`

Run examples:

Fetch sandbox tickets without touching the current historical raw file:

```bash
venv\Scripts\python.exe src\fetch_sandbox_tickets.py --days-back 365
```

Fetch all tickets from the sandbox without a created-date filter:

```bash
venv\Scripts\python.exe src\fetch_sandbox_tickets.py --all-tickets --replace-main-raw --load-postgres
```

Fetch open tickets only:

```bash
venv\Scripts\python.exe src\fetch_sandbox_tickets.py --days-back 90 --open-only
```

Fetch sandbox tickets and feed them into the current pipeline source:

```bash
venv\Scripts\python.exe src\fetch_sandbox_tickets.py --days-back 365 --replace-main-raw --load-postgres
```

Run the full sandbox-to-recommendation pipeline:

```bash
venv\Scripts\python.exe src\run_sandbox_pipeline.py --all-tickets
```

Suggested live workflow:
1. pull sandbox tickets with `fetch_sandbox_tickets.py`
2. if needed, replace the canonical raw file and PostgreSQL raw table
3. rerun the existing Phase 2 pipeline:
   - `clean_ticket_data.py`
   - `feature_engineering.py`
   - `nlp_ticket_similarity.py`
   - `time_estimation_model.py`
   - `complexity_scoring.py`
   - `assignment_scorer.py`

## Phase 2: Data Cleaning

Phase 2 started by converting the raw ticket export into a cleaner, model-ready dataset.

Cleaning work completed:
- standardized column names
- normalized text fields
- converted placeholder values such as `Unknown`, `Not Available`, `Unassigned`, and `Uncategorized` to missing values
- parsed date columns into datetime values
- converted numeric and boolean columns to usable types
- created derived fields including:
  - `resolution_hours`
  - `resolution_days`
  - `first_response_minutes`
  - `hours_until_due`
  - `hours_past_due`
  - `ticket_text`

Cleaning script:
- `src/clean_ticket_data.py`

Cleaning outputs:
- `data/Cleaned_Data/autotask_cleaned_data.csv`
- `data/Cleaned_Data/autotask_cleaning_summary.json`

## Phase 2: Feature Engineering

After cleaning, feature engineering was added to prepare the dataset for recommendation logic and future machine learning models.

Feature groups created:
- time features
  - `created_year`
  - `created_month`
  - `created_day_of_week`
  - `created_hour`
  - `created_is_weekend`
  - `created_is_business_hours`
  - `created_part_of_day`
- priority features
  - `priority_weight`
  - `is_priority_low`
  - `is_priority_medium`
  - `is_priority_high`
  - `is_priority_critical`
- workflow features
  - `is_active_ticket`
  - `is_waiting_state`
  - `is_unassigned`
  - `is_overdue_open`
- text-derived features
  - `ticket_text_char_count`
  - keyword flags for `server`, `backup`, `vpn`, `email`, `printer`, access issues, and urgent language
- category grouping features
  - `issue_type_group`
  - `queue_group`
  - `role_group`
- target/helper features
  - `resolution_hours_log`
  - `met_estimate_flag`
  - `estimate_error_hours`
- SLA features
  - `sla_priority_class`
  - `sla_initial_response_hours`
  - `sla_status_update_hours`
  - `sla_weight`
  - `is_service_request`
  - `is_maintenance`
  - `sla_breach_risk`
  - `sla_breach_severity`
  - `ticket_age_hours`
  - `sla_age_ratio`

Feature engineering script:
- `src/feature_engineering.py`

Feature outputs:
- `data/Feature_Engineered/autotask_feature_engineered.csv`
- `data/Feature_Engineered/autotask_training_dataset.csv`
- `data/Feature_Engineered/autotask_open_tickets_dataset.csv`
- `data/Feature_Engineered/technician_profiles.csv`
- `data/Feature_Engineered/feature_engineering_summary.json`

## Phase 2: SLA-Aware Recommendation Logic

The DiamondEdge SLA rules were integrated into the dataset and recommendation layer so that ticket handling is aligned with business service expectations instead of relying on priority labels alone.

SLA-aware logic completed:
- mapped tickets into SLA classes:
  - `Low`
  - `Medium`
  - `High`
  - `Service Request`
  - `Maintenance`
- assigned SLA response and update targets based on the SLA document
- created breach-risk indicators and SLA severity fields
- built a technician workload snapshot for active tickets
- implemented a baseline rule-based assignment scorer
- generated top-3 technician recommendations for each active ticket
- included SLA class and workload rationale in recommendation outputs
- added BM25/TF-IDF text expertise as a technician-match signal
- added projected workload balancing so repeated top-1 recommendations increase a technician's effective load during the run
- added fair-distribution penalties to avoid one technician receiving most open-ticket recommendations

Recommendation scoring script:
- `src/assignment_scorer.py`

Recommendation outputs:
- `data/Recommendations/technician_workload_snapshot.csv`
- `data/Recommendations/assignment_recommendations.csv`
- `data/Recommendations/recommendation_summary.json`

## Phase 2: NLP Ticket Similarity and BM25 Ranking

An NLP similarity layer was added to improve ticket understanding beyond structured fields such as priority, queue, and issue type.

Why NLP was added:
- ticket titles and descriptions contain the real issue context
- structured labels can be broad, noisy, or inconsistent
- similar historical tickets can improve both technician recommendations and time estimation

NLP logic completed:
- used `ticket_text` as the text source
- normalized ticket text for analysis
- built TF-IDF vectors with `scikit-learn`
- computed cosine similarity between active tickets and completed historical tickets
- added BM25 scoring for stronger search-style keyword matching
- created a hybrid text score using TF-IDF and BM25 signals
- returned top-5 historical matches for each active ticket
- generated NLP-based estimated resolution hours
- generated technician suggestions from similar historical tickets
- integrated precomputed NLP/BM25 outputs into the assignment scorer efficiently

BM25 improvement:
- BM25 rewards rare, domain-specific ticket terms more effectively than plain frequency matching
- the recommendation scorer now includes `bm25_text_expertise_score`
- recommendation outputs also include `best_text_match_score`, `text_match_count`, and `best_text_match_rank`

NLP script:
- `src/nlp_ticket_similarity.py`

NLP outputs:
- `data/NLP/ticket_similarity_matches.csv`
- `data/NLP/ticket_similarity_summary.csv`
- `data/NLP/nlp_similarity_summary.json`

## Phase 2: Time Estimation Model

A dedicated time estimation pipeline was added to predict likely ticket resolution time using the completed historical ticket set.

Time estimation work completed:
- used completed tickets as the training dataset
- used structured engineered features such as:
  - priority
  - SLA class
  - issue type
  - queue
  - created time features
  - text length features
  - response-time and due-time features
- reused NLP similarity outputs as additional signals
- trained a baseline machine learning regressor
- benchmarked safe baselines against the model
- implemented a final safe hybrid estimator combining:
  - historical category baseline
  - model prediction
  - NLP estimate

Why the final estimator is hybrid:
- the dataset contains repeated ticket patterns and some noisy categories
- simple historical baselines were more stable than an aggressive standalone model
- the final approach keeps the model explainable and safer for project use

Time estimation script:
- `src/time_estimation_model.py`

Time estimation outputs:
- `data/Time_Estimation/time_estimation_test_predictions.csv`
- `data/Time_Estimation/time_estimation_open_ticket_predictions.csv`
- `data/Time_Estimation/time_estimation_metrics.json`

## Phase 2: Complexity Scoring

An explainable complexity scoring layer was added so tickets can be categorized by likely difficulty before assignment and workload balancing decisions are made.

Complexity scoring work completed:
- combined actual or predicted effort signals into a complexity indicator
- used SLA urgency as a complexity factor
- used historical issue-type difficulty as a complexity factor
- used NLP novelty as a complexity factor
- used technical keyword and text-length signals as supporting indicators
- generated:
  - `complexity_score`
  - `complexity_class`
  - `complexity_reason`
- integrated complexity into recommendation scoring
- surfaced complexity metrics in the dashboard
- loaded complexity outputs into PostgreSQL

Complexity scoring script:
- `src/complexity_scoring.py`

Complexity outputs:
- `data/Complexity/autotask_complexity_scored.csv`
- `data/Complexity/complexity_scoring_summary.json`

## PostgreSQL Output Loading

All major generated analytics outputs were loaded back into PostgreSQL so they can be queried directly from the database layer.

Database loading work completed:
- created a reusable PostgreSQL loader for generated CSV and JSON outputs
- loaded processed ticket datasets into PostgreSQL tables
- loaded NLP, recommendation, and time-estimation outputs into PostgreSQL tables
- loaded summary JSON files into PostgreSQL summary tables
- verified row counts for the main analytics tables after loading

Loader script:
- `src/load_outputs_to_postgres.py`

Main PostgreSQL tables loaded:
- `autotask_cleaned_data`
- `autotask_feature_engineered`
- `autotask_training_dataset`
- `autotask_open_tickets_dataset`
- `autotask_technician_profiles`
- `autotask_ticket_similarity_matches`
- `autotask_ticket_similarity_summary`
- `autotask_complexity_scored`
- `autotask_technician_workload_snapshot`
- `autotask_assignment_recommendations`
- `autotask_time_estimation_test_predictions`
- `autotask_time_estimation_open_ticket_predictions`

Summary tables loaded:
- `autotask_cleaning_summary`
- `autotask_feature_engineering_summary`
- `autotask_nlp_similarity_summary`
- `autotask_complexity_scoring_summary`
- `autotask_recommendation_summary`
- `autotask_time_estimation_metrics`

## Dashboard Reporting

The project now uses one dashboard layer:
- an interactive Streamlit dashboard for filtering, drill-down, workload management, and report exports

Interactive dashboard includes:
- sidebar filters for technician, priority, SLA class, complexity, issue type, and queue
- KPI cards for ticket volume, open tickets, assigned open tickets, unassigned open tickets, and recommendation coverage
- overview charts for SLA, complexity, workload, and priority mix
- attention table for tickets most in need of action
- employee drill-down views
- recommendation score and rationale reporting
- open-ticket recommendation board showing current assignee and top-3 recommended technicians
- workload-management plots for recommended technicians:
  - top-3 recommendation load by technician
  - rank-1 recommendation load by technician
- time-estimation comparison views
- downloadable CSV exports from the main report sections

Interactive dashboard script:
- `src/interactive_dashboard.py`

## Dataset Snapshot

Current processed counts:
- total tickets: `1137`
- completed tickets: `329`
- open-ticket dataset rows: `881`
- engineered feature columns: `81`
- technician profiles generated: `4`
- active tickets scored with recommendations: `335`
- recommendation rows generated: `1005`
- NLP/BM25 similarity matches generated: `1675`
- time estimation test rows evaluated: `52`
- time estimation open-ticket predictions generated: `335`
- complexity rows scored: `1137`
- PostgreSQL analytics tables loaded successfully: `18`
- interactive dashboard app available: `1`

Important observations:
- `sub_issue_type` and `work_type` are mostly missing, so they are currently weak modeling features
- `estimated_hours` has low variance, so it should not be treated as a strong predictor by itself
- open historical tickets appear overdue relative to the current date, so overdue logic should be interpreted carefully during live scoring
- SLA classification is currently rule-based using priority, queue, issue type, and ticket text patterns
- TF-IDF/BM25 hybrid similarity performs well for repeated ticket patterns, but vague ticket titles still produce weaker matches
- the safe hybrid time estimator currently performs better than the raw standalone model, but accuracy still has room for improvement
- complexity scoring is explainable and integrated, but thresholds can still be refined as more ticket history becomes available
- processed outputs are now available both as files and as PostgreSQL tables for querying and dashboard use
- Streamlit is now the single supported dashboard for live analysis, demos, and reporting

## Tools and Libraries

Main libraries currently used:
- pandas
- sqlalchemy
- psycopg2-binary
- python-dotenv
- jupyter
- scikit-learn
- numpy
- matplotlib
- seaborn
- streamlit
- plotly

## What Comes Next

The next development steps are:
- refine SLA classification with stronger text understanding
- improve the recommendation engine with dispatcher feedback and technician specialization scoring
- combine final estimated effort more directly into assignment scoring and dashboard views
- add assignment-log feedback so model weights can be tuned from dispatcher decisions
- continue improving API-backed live workflows

## Progress Log

- Initial project setup completed with Python environment, dependencies, and PostgreSQL connectivity
- Historical Autotask ticket data exported and organized for analysis
- Raw ticket dataset cleaned and standardized for downstream modeling
- Date, numeric, boolean, and missing-value handling implemented in the cleaning pipeline
- Derived fields such as resolution time, first response time, due-time metrics, and combined ticket text created
- Feature engineering pipeline implemented for time, priority, workflow, text, and category features
- SLA classification and SLA target features added to the feature engineering pipeline
- Training dataset for completed tickets generated
- Open-ticket dataset for recommendation testing generated
- Technician profile summary generated from historical completed tickets
- Technician workload snapshot generated for active tickets
- Baseline SLA-aware assignment recommendations generated for open tickets
- NLP similarity pipeline implemented using TF-IDF and cosine similarity
- BM25 ranking added and combined with TF-IDF as a hybrid text matching signal
- NLP/BM25 outputs integrated into assignment recommendations through precomputed lookup-based scoring
- Safe hybrid time estimation model implemented and evaluated on completed historical tickets
- Explainable complexity scoring implemented and integrated into downstream outputs
- Generated analytics outputs loaded into PostgreSQL tables for direct querying
- Interactive Streamlit dashboard upgraded with richer visuals, drill-down reporting, open-ticket assignment board, recommendation workload plots, filters, and CSV exports
- Sandbox API ingestion added with `--all-tickets` support and a full `run_sandbox_pipeline.py` runner
- README updated to reflect Phase 1 and Phase 2 progress

## Deliverables Completed So Far

- `src/db_connection.py`
- `src/export_raw_data.py`
- `src/clean_ticket_data.py`
- `src/assignment_scorer.py`
- `src/feature_engineering.py`
- `src/complexity_scoring.py`
- `src/interactive_dashboard.py`
- `src/load_outputs_to_postgres.py`
- `src/nlp_ticket_similarity.py`
- `src/autotask_api_client.py`
- `src/fetch_sandbox_tickets.py`
- `src/run_sandbox_pipeline.py`
- `src/time_estimation_model.py`
- `SCORING_FORMULA.txt`
- `data/Cleaned_Data/autotask_cleaned_data.csv`
- `data/Cleaned_Data/autotask_cleaning_summary.json`
- `data/Feature_Engineered/autotask_feature_engineered.csv`
- `data/Feature_Engineered/autotask_training_dataset.csv`
- `data/Feature_Engineered/autotask_open_tickets_dataset.csv`
- `data/Feature_Engineered/technician_profiles.csv`
- `data/Feature_Engineered/feature_engineering_summary.json`
- `data/NLP/ticket_similarity_matches.csv`
- `data/NLP/ticket_similarity_summary.csv`
- `data/NLP/nlp_similarity_summary.json`
- `data/Complexity/autotask_complexity_scored.csv`
- `data/Complexity/complexity_scoring_summary.json`
- `data/Time_Estimation/time_estimation_test_predictions.csv`
- `data/Time_Estimation/time_estimation_open_ticket_predictions.csv`
- `data/Time_Estimation/time_estimation_metrics.json`
- `data/Recommendations/technician_workload_snapshot.csv`
- `data/Recommendations/assignment_recommendations.csv`
- `data/Recommendations/recommendation_summary.json`
## How To Run

Run data cleaning:

```bash
python src/clean_ticket_data.py
```

Run feature engineering:

```bash
python src/feature_engineering.py
```

Run assignment scoring:

```bash
python src/assignment_scorer.py
```

Run NLP ticket similarity:

```bash
venv\Scripts\python.exe src/nlp_ticket_similarity.py
```

Run the full sandbox API pipeline using all tickets:

```bash
venv\Scripts\python.exe src\run_sandbox_pipeline.py --all-tickets
```

Run time estimation:

```bash
venv\Scripts\python.exe src/time_estimation_model.py
```

Run complexity scoring:

```bash
venv\Scripts\python.exe src/complexity_scoring.py
```

Run interactive Streamlit dashboard:

```bash
venv\Scripts\streamlit.exe run src/interactive_dashboard.py
```

Load generated outputs into PostgreSQL:

```bash
venv\Scripts\python.exe src/load_outputs_to_postgres.py
```
