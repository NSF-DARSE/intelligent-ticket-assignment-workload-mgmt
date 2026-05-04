# Migration Guide

## Upgrading To Version 1.0.0

This version moves the project to a cleaner release-style workflow centered on local PostgreSQL.

## For Existing Team Members

1. Pull the latest `AI_Intelligent_Ticket_Assignment` branch.
2. Recreate or refresh the virtual environment if dependencies changed.
3. Copy `.env.example` to `.env` if needed.
4. Set local PostgreSQL credentials in `.env`.
5. Confirm `DB_HOST` is local: `localhost`, `127.0.0.1`, or `::1`.
6. Regenerate local outputs with:

```powershell
.\venv\Scripts\python.exe main.py pipeline --all-tickets
```

## Important Changes

- Generated datasets under `data/` remain local-only and ignored by Git.
- Database configuration must come from `.env`.
- The dashboard reads from local PostgreSQL reporting tables.
- Dispatch simulations are stored in local PostgreSQL.
- CI runs automated tests.
- The active similarity stack is BM25 + MiniLM.
- The standalone TF-IDF and time-estimation model paths are no longer part of the supported workflow.

## If You Used Azure PostgreSQL Previously

The active workflow no longer targets Azure PostgreSQL. Replace Azure values in `.env` with local values:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=autotask_local
DB_SSLMODE=
```

The database helper rejects non-local hosts to prevent accidental cloud usage.

## User-Facing Workflow

The main workflow remains:
- fetch data
- process data
- score recommendations
- load local PostgreSQL outputs
- view the Streamlit dashboard
