# Release Notes

## Version 1.0.0

### Summary

Version 1.0.0 delivers the first full working release of the Intelligent Ticket Assignment & Workload Management project. The release includes the production-style project flow from raw Autotask ticket ingestion through technician recommendation, PostgreSQL output loading, and interactive Streamlit reporting.

### End-User Highlights
- Top-3 technician recommendation engine for open tickets
- Skill-aware matching between tickets and employees
- Hybrid NLP similarity using TF-IDF, BM25, and MiniLM embeddings
- Workload-aware balancing so one technician is not overloaded unfairly
- SLA and complexity-aware recommendation logic
- Interactive Streamlit dashboard for operations review and dispatch simulation

### Installation Guide

1. Create a virtual environment.
2. Install dependencies from [requirements.txt](requirements.txt).
3. Copy [.env.example](.env.example) to `.env` and fill in the required values.
4. Run [main.py](main.py) commands or the individual pipeline scripts listed in [README.md](README.md).

### API / Behavior Notes
- `main.py status` reports the presence of core project outputs.
- `main.py pipeline` runs the sandbox-backed end-to-end pipeline.
- `main.py dashboard` launches the Streamlit interface.
- `main.py load-postgres` refreshes PostgreSQL output tables from local generated files.

### Known Issues
- Azure PostgreSQL setup still depends on the correct network/firewall configuration outside the repo.
- Some pipeline outputs are intentionally local-only and are not committed to Git.
- Streamlit dashboard logic is functional but still concentrated in a large single file, so future refactoring is recommended.

### Upgrade Impact
- This is the first formal release candidate for the repo. There are no prior tagged public releases to migrate from.
