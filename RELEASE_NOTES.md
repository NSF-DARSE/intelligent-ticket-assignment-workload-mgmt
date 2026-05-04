# Release Notes

## Version 1.0.0

This release delivers a local PostgreSQL-backed Autotask ticket assignment recommendation system with a Streamlit dashboard.

## End-User Highlights

- Top-3 technician recommendations for active tickets
- Skill-aware matching between tickets and employees
- Hybrid text similarity using **BM25 + MiniLM**
- Workload balancing to reduce over-assignment
- SLA-aware and complexity-aware ranking
- Local PostgreSQL reporting tables for reproducible dashboard access
- Streamlit dashboard for operational review and dispatch simulation

## Model And Scoring Summary

The active recommendation pipeline uses:
- BM25 lexical retrieval
- MiniLM semantic embeddings
- historical ticket resolution hints from top similar completed tickets
- employee skill alignment
- technician workload and capacity penalties
- SLA urgency and complexity fit

Removed from the active workflow:
- standalone TF-IDF retrieval
- standalone time-estimation model

## Installation Guide

1. Install Python `3.13`.
2. Create and activate a virtual environment.
3. Install dependencies from [requirements.txt](requirements.txt).
4. Create a local PostgreSQL database named `autotask_local`.
5. Copy [.env.example](.env.example) to `.env`.
6. Fill in local PostgreSQL and Autotask credentials.
7. Run `python main.py pipeline --all-tickets`.
8. Run `python main.py dashboard`.

## Current Command Surface

- `main.py status`: reports core local output availability
- `main.py pipeline`: runs the end-to-end workflow and loads local PostgreSQL
- `main.py load-postgres`: reloads generated outputs into local PostgreSQL
- `main.py dashboard`: launches Streamlit
- `src/benchmark_pipeline.py`: records runtime and memory benchmark notes

## Release Readiness

Included:
- MIT license
- dependency pins in `requirements.txt`
- package metadata in `pyproject.toml`
- CI workflow for tests
- local validation workflow for imports
- changelog, migration guide, known issues, API reference, and performance notes

Before final submission, create a Git release tag:

```powershell
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```
