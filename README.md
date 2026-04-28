# AUTO TASK AI TICKET RECOMMADATION SYSTEM

## Overview

This project builds an end-to-end ticket recommendation system for Autotask service desk data.

The system:
- fetches ticket data from the Autotask sandbox
- cleans and standardizes the raw records
- creates model-ready ticket features
- maps employee skills into a structured dataset
- compares open tickets with historical completed tickets
- scores ticket complexity and workload
- recommends the best 3 technicians for each open ticket
- loads the outputs into PostgreSQL
- displays the results in a Streamlit dashboard

The dashboard is the final user-facing portal. The Python pipeline is the backend that prepares and scores the data.

## Project Flow

The project runs in this order:

1. Configure environment variables and database access
2. Fetch raw Autotask ticket data
3. Clean and standardize the ticket dataset
4. Create feature-engineered ticket datasets
5. Clean and normalize employee skills
6. Compare open tickets with completed tickets using NLP
7. Estimate ticket effort from historical patterns
8. Score ticket complexity
9. Generate technician recommendations
10. Load processed outputs into PostgreSQL
11. Show the final outputs in the Streamlit dashboard

## Project Structure

```text
Project_Autotask
|-- data
|   |-- Raw_Data
|   |-- Cleaned_Data
|   |-- Feature_Engineered
|   |-- NLP
|   |-- Complexity
|   `-- Recommendations
|-- sql
|   `-- create_autotask_table.sql
|-- src
|   |-- assignment_scorer.py
|   |-- autotask_api_client.py
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
|   |-- run_sandbox_pipeline.py
|   `-- time_estimation_model.py
|-- main.py
|-- requirements.txt
|-- Skillsdataset.csv
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

### 4. Configure `.env`

Create a `.env` file from `.env.example` and provide values for:

- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`
- `AUTOTASK_API_BASE_URL`
- `AUTOTASK_API_USERNAME`
- `AUTOTASK_API_SECRET`
- `AUTOTASK_API_INTEGRATION_CODE`

### 5. Test the database connection

What this step does:
- checks whether PostgreSQL is reachable from the project
- confirms that the database settings in `.env` are correct

Run:

```powershell
.\venv\Scripts\python.exe src\db_connection.py
```

## Step-by-Step Pipeline

### Step 1. Fetch raw Autotask tickets

Script:
- `src/fetch_sandbox_tickets.py`

What this step does:
- connects to the Autotask sandbox API
- downloads ticket data
- saves the raw ticket records for local processing
- can also load the raw data into PostgreSQL

Why it matters:
- this is the source of truth for the entire project
- every later step depends on the raw ticket export from here

Main run command:

```powershell
.\venv\Scripts\python.exe src\fetch_sandbox_tickets.py --all-tickets --replace-main-raw --load-postgres
```

Alternative run command:

```powershell
.\venv\Scripts\python.exe src\fetch_sandbox_tickets.py --days-back 365 --replace-main-raw --load-postgres
```

Main outputs:
- `data/Raw_Data/autotask_raw_data.csv`
- `data/Raw_Data/autotask_sandbox_raw_data.csv`
- `data/Raw_Data/autotask_sandbox_fetch_summary.json`

### Step 2. Clean the ticket data

Script:
- `src/clean_ticket_data.py`

What this step does:
- standardizes column names
- cleans ticket title and description fields
- converts dates into usable datetime values
- converts numeric fields into proper numeric types
- handles blanks, placeholders, and missing values
- creates reusable ticket text used later by the NLP step

Why it matters:
- machine learning and scoring logic depend on consistent data
- bad dates, text noise, and invalid numeric values break later analysis

Run:

```powershell
.\venv\Scripts\python.exe src\clean_ticket_data.py
```

Main outputs:
- `data/Cleaned_Data/autotask_cleaned_data.csv`
- `data/Cleaned_Data/autotask_cleaning_summary.json`

### Step 3. Build ticket features

Script:
- `src/feature_engineering.py`

What this step does:
- creates engineered ticket features from the cleaned data
- separates open tickets from completed tickets
- creates training and scoring datasets
- builds technician history profiles from completed ticket data
- adds workflow, SLA, priority, issue-type, and state features

Why it matters:
- raw fields alone are not enough for recommendation
- this step turns ticket records into structured inputs for the models

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

### Step 4. Clean and normalize employee skills

Script:
- `src/clean_employee_skills.py`

Input:
- `Skillsdataset.csv`

What this step does:
- reads the employee skills dataset
- keeps the employee display name used in the dashboard
- creates a normalized technician key used for joins
- creates both a profile-style table and a one-skill-per-row table
- prepares the skills data for the recommendation model

Why it matters:
- the recommendation engine now considers employee skill alignment
- this step makes the skills dataset usable by both PostgreSQL and the model

Run:

```powershell
.\venv\Scripts\python.exe src\clean_employee_skills.py
```

Main outputs:
- `data/Feature_Engineered/employee_skills_profile.csv`
- `data/Feature_Engineered/employee_skills_normalized.csv`
- `data/Feature_Engineered/employee_skills_summary.json`

### Step 5. Compare tickets with NLP similarity

Script:
- `src/nlp_ticket_similarity.py`

What this step does:
- compares open tickets against historical completed tickets
- finds semantically and lexically similar past tickets
- produces similarity matches that feed recommendation scoring

Models used in this step:
- `TF-IDF`
- `BM25`
- `sentence-transformers/all-MiniLM-L6-v2`

Final text model weights:
- `TF-IDF = 0.20`
- `BM25 = 0.20`
- `MiniLM Embedding = 0.60`

Why it matters:
- this is how the system learns from previously solved tickets
- technicians who solved similar historical tickets get stronger recommendation scores

Run:

```powershell
.\venv\Scripts\python.exe src\nlp_ticket_similarity.py
```

Main outputs:
- `data/NLP/ticket_similarity_matches.csv`
- `data/NLP/ticket_similarity_summary.csv`
- `data/NLP/nlp_similarity_summary.json`

### Step 6. Estimate ticket effort

Script:
- `src/time_estimation_model.py`

What this step does:
- estimates likely resolution effort for tickets
- creates an additional signal for workload-aware assignment
- helps distinguish light tickets from heavier work

Why it matters:
- technician balancing should not depend only on ticket count
- a smaller number of hard tickets may still represent a high workload

Run:

```powershell
.\venv\Scripts\python.exe src\time_estimation_model.py
```

Main outputs:
- `data/Time_Estimation/time_estimation_test_predictions.csv`
- `data/Time_Estimation/time_estimation_open_ticket_predictions.csv`
- `data/Time_Estimation/time_estimation_metrics.json`

### Step 7. Score ticket complexity

Script:
- `src/complexity_scoring.py`

What this step does:
- assigns a complexity score and complexity class to each ticket
- supports safer technician matching for harder tickets

Why it matters:
- not every open ticket should be handled by the same type of technician
- complexity helps the model avoid poor or risky assignments

Run:

```powershell
.\venv\Scripts\python.exe src\complexity_scoring.py
```

Main outputs:
- `data/Complexity/autotask_complexity_scored.csv`
- `data/Complexity/complexity_scoring_summary.json`

### Step 8. Generate technician recommendations

Script:
- `src/assignment_scorer.py`

What this step does:
- scores each technician for each open ticket
- ranks technicians and keeps the top 3 recommendations
- combines historical experience, skills, NLP similarity, complexity, SLA, and workload

Recommendation logic used here includes:
- historical issue-type experience
- employee skill alignment
- matched skill count
- ticket similarity from TF-IDF, BM25, and MiniLM
- technician history on similar tickets
- workload balancing
- SLA urgency fit
- complexity fit

Why it matters:
- this is the main business logic of the project
- it turns all prepared signals into final assignment recommendations

Run:

```powershell
.\venv\Scripts\python.exe src\assignment_scorer.py
```

Main outputs:
- `data/Recommendations/assignment_recommendations.csv`
- `data/Recommendations/technician_workload_snapshot.csv`
- `data/Recommendations/recommendation_summary.json`

### Step 9. Load processed outputs into PostgreSQL

Script:
- `src/load_outputs_to_postgres.py`

What this step does:
- reads the processed CSV and JSON outputs from earlier steps
- loads them into PostgreSQL reporting tables
- prepares the data for the Streamlit dashboard

Why it matters:
- the dashboard reads from these reporting-ready tables
- this step creates one central place to query all final outputs

Run:

```powershell
.\venv\Scripts\python.exe src\load_outputs_to_postgres.py
```

Main loaded tables:
- `autotask_feature_engineered`
- `autotask_open_tickets_dataset`
- `autotask_technician_profiles`
- `autotask_employee_skills_profile`
- `autotask_employee_skills_normalized`
- `autotask_ticket_similarity_matches`
- `autotask_assignment_recommendations`
- `autotask_technician_workload_snapshot`

### Step 10. Open the dashboard

Script:
- `src/interactive_dashboard.py`

What this step does:
- shows the final project outputs in a Streamlit portal
- displays open-ticket trends, employee profiles, ticket recommendations, and assignment views
- gives a visual interface for managers and project reviewers

Why it matters:
- this is the final presentation layer of the project
- it is the easiest way to review recommendations and technician-level details

Run:

```powershell
.\venv\Scripts\python.exe -m streamlit run src\interactive_dashboard.py
```

Dashboard URL:
- [http://localhost:8501](http://localhost:8501)

## Run the Full Flow in One Command

If you want to run the whole backend flow from ticket fetch to PostgreSQL load:

```powershell
.\venv\Scripts\python.exe src\run_sandbox_pipeline.py --all-tickets
```

This runs:
1. ticket fetch
2. ticket cleaning
3. feature engineering
4. NLP similarity
5. time estimation
6. complexity scoring
7. recommendation scoring
8. PostgreSQL loading

## Use the Main Project Launcher

You can also use `main.py` as the single entry point.

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

## Utility Commands

### Export raw PostgreSQL ticket data

```powershell
.\venv\Scripts\python.exe src\export_raw_data.py
```

## Recommended Run Order for a New User

If someone is running this project for the first time, this is the clean order to follow:

1. create and activate the virtual environment
2. install dependencies
3. configure `.env`
4. test the PostgreSQL connection
5. fetch Autotask tickets
6. clean the tickets
7. build ticket features
8. clean employee skills
9. run NLP similarity
10. run time estimation
11. run complexity scoring
12. generate recommendations
13. load outputs into PostgreSQL
14. run the dashboard

## Notes

- The Streamlit dashboard is the main user-facing interface.
- The final recommendation model uses historical ticket experience, employee skill matching, workload balancing, SLA logic, complexity scoring, and NLP similarity together.
- Windows may lock CSV files if they are open in Excel or another tool. If a file does not refresh, close it and rerun the related step.
- Employee display names are kept for dashboard presentation, while technician keys are used internally for joins and recommendation logic.
