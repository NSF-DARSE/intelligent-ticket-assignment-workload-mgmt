# Migration Guide

## Upgrading to Version 1.0.0

This repository is moving from an evolving project workspace to a cleaner release-style structure.

### For Existing Team Members

1. Pull the latest `AI_Intelligent_Ticket_Assignment` branch.
2. Recreate or refresh the local virtual environment if package versions changed.
3. Copy your local secrets back into `.env` if you use a fresh checkout.
4. Regenerate local outputs by running the pipeline again, because generated data is no longer tracked in Git.

### Important Repository Changes
- Generated datasets under `data/` are now local-only and ignored by Git.
- Local skills input and Azure connection helper files are ignored by Git.
- Database configuration must come from `.env`; source code no longer includes fallback credentials.
- CI now runs an actual automated test suite.
- The active text similarity stack is now BM25 + MiniLM.
- The separate time-estimation model has been removed from the supported pipeline.

### If You Were Depending on Old Tracked Data Files

You now need to generate them locally:

```powershell
.\venv\Scripts\python.exe main.py pipeline --all-tickets
```

Or run the individual scripts in the order documented in [README.md](README.md).

### No Breaking User-Facing Interface Changes

The main workflow remains:
- fetch data
- process data
- score recommendations
- load PostgreSQL outputs
- view the Streamlit dashboard

The biggest change is that the repository is now cleaner and safer for sharing and deployment.
