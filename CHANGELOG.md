# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-04-30

### Added
- End-to-end Autotask ticket recommendation pipeline covering ingestion, cleaning, feature engineering, NLP similarity, complexity scoring, time estimation, recommendation scoring, PostgreSQL loading, and Streamlit reporting.
- Hybrid recommendation model using TF-IDF, BM25, and `all-MiniLM-L6-v2` sentence embeddings.
- Employee skill normalization pipeline and skill-aware recommendation scoring.
- Interactive Streamlit dashboard with recommendation views, employee profiles, assignment board, and simulated dispatch actions stored in PostgreSQL.
- Main project launcher in [main.py](main.py) for `status`, `pipeline`, `dashboard`, and `load-postgres` workflows.
- MIT license and deployment-safe repository cleanup.
- Unit and integration-style tests in the `tests/` folder.
- GitHub Actions workflow that installs dependencies and runs the automated tests.

### Changed
- README rewritten into a project-flow guide with step-by-step execution instructions.
- Sensitive defaults, hardcoded connection fallbacks, and tracked generated datasets removed from the Git repository.
- Dashboard terminology standardized from "recommandation" to "recommendation".
- `.env.example` updated to use blank placeholders for environment variables and optional SSL configuration.

### Fixed
- Assigned/unassigned ticket board rendering now handles simulated dispatch data safely.
- Recommendation board and employee views now map technician keys to display names consistently.
- Dashboard code cleaned to remove stale internal fields from visible tables.

### Security
- Removed hardcoded database fallback credentials from source code.
- Removed built-in endpoint defaults from tracked configuration and source files.
- Stopped tracking generated datasets and local skill input files in Git.
