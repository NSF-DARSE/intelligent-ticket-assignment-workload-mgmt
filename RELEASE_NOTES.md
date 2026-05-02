# Release Notes

## Current Working Configuration

The current project state delivers a complete ticket recommendation workflow from Autotask ingestion through PostgreSQL-backed dashboard reporting.

### End-User Highlights

- Top-3 technician recommendation engine for active tickets
- Skill-aware matching between tickets and employees
- Hybrid text similarity using **BM25 + MiniLM**
- Workload-aware balancing so one technician is not overloaded unfairly
- SLA-aware and complexity-aware recommendation logic
- Streamlit dashboard for review and dispatch simulation

### Important Model Update

The active project configuration no longer uses:
- TF-IDF retrieval
- the standalone time-estimation model

Instead, the recommendation pipeline now uses:
- BM25 lexical retrieval
- MiniLM semantic similarity
- historical resolution-hours hints derived from the top similar completed tickets

### Installation Guide

1. Create a virtual environment.
2. Install dependencies from [requirements.txt](requirements.txt).
3. Copy [.env.example](.env.example) to `.env` and fill in the required values.
4. Run [main.py](main.py) commands or the individual scripts listed in [README.md](README.md).

### Current Command Surface

- `main.py status` reports the presence of core project outputs.
- `main.py pipeline` runs the supported end-to-end pipeline.
- `main.py dashboard` launches the Streamlit interface.
- `main.py load-postgres` refreshes PostgreSQL output tables from generated local files.

### Known Issues

- Azure PostgreSQL access still depends on firewall and network configuration outside the repo.
- The Streamlit dashboard is functional but still concentrated in one large file.
- Some generated outputs remain local-only by design and are not committed to Git.
