# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Changed
- Simplified the text retrieval pipeline to use **BM25 + MiniLM** only.
- Removed the standalone TF-IDF similarity path from the active recommendation workflow.
- Removed the separate time-estimation model from the project pipeline and PostgreSQL loading flow.
- Updated project documentation, scoring notes, and pipeline descriptions to match the current architecture.

### Fixed
- Removed stale references to TF-IDF and time-estimation steps from the active source pipeline.
- Cleaned recommendation rationale text to describe the current BM25 + MiniLM retrieval model accurately.

## [1.0.0] - 2026-04-30

### Added
- End-to-end Autotask ticket recommendation pipeline covering ingestion, cleaning, feature engineering, NLP similarity, complexity scoring, recommendation scoring, PostgreSQL loading, and Streamlit reporting.
- Hybrid recommendation model using lexical and embedding-based similarity signals.
- Employee skill normalization pipeline and skill-aware recommendation scoring.
- Interactive Streamlit dashboard with recommendation views, employee profiles, assignment board, and simulated dispatch actions stored in PostgreSQL.
- Main project launcher in [main.py](main.py) for `status`, `pipeline`, `dashboard`, and `load-postgres` workflows.
- MIT license and deployment-safe repository cleanup.
- Unit and integration-style tests in the `tests/` folder.
- GitHub Actions workflow that installs dependencies and runs the automated tests.
