# CISC 867-010 Rubric Coverage

This document maps the final-presentation rubric to repository evidence.

## Summary

| Rubric Area | Points | Coverage | Evidence |
|---|---:|---|---|
| Code organization and readability | 10 | Covered | `src/`, `tests/`, `main.py`, modular pipeline scripts |
| Performance and resource awareness | 10 | Covered | `docs/PERFORMANCE.md`, `src/benchmark_pipeline.py`, `data/Evaluation/` outputs when generated |
| Reproducible setup and execution | 15 | Covered | `README.md`, `.env.example`, `requirements.txt`, `pyproject.toml`, `main.py` |
| Testing strategy | 15 | Covered | `tests/`, `.github/workflows/ci.yml`, local PostgreSQL config tests |
| Create release notes | 10 | Covered | `CHANGELOG.md`, `RELEASE_NOTES.md`, `MIGRATION_GUIDE.md`, `KNOWN_ISSUES.md`, `docs/API_REFERENCE.md` |
| README and license | 15 | Covered | `README.md`, `LICENSE` |
| Packaging and release readiness | 10 | Mostly covered | `pyproject.toml`, `requirements.txt`, `main.py`, release tag instructions |
| Documentation for releases | 5 | Covered | release docs listed above |
| Presentation slides | 5 | Draft covered | `docs/PRESENTATION_OUTLINE.md` |
| Presentation delivery | 5 | Not code-verifiable | Must be demonstrated live |

## Detailed Mapping

### Code Organization And Readability

Evidence:
- `src/clean_ticket_data.py`: cleaning and normalization
- `src/feature_engineering.py`: feature generation
- `src/nlp_ticket_similarity.py`: BM25 + MiniLM similarity
- `src/complexity_scoring.py`: complexity scoring
- `src/assignment_scorer.py`: recommendation scoring
- `src/load_outputs_to_postgres.py`: local PostgreSQL loading
- `src/interactive_dashboard.py`: Streamlit dashboard
- `main.py`: clean runnable entry point

Notes:
- The pipeline is separated by responsibility.
- Functions use descriptive names and type hints where useful.
- Known maintainability limitation: the dashboard is still a large file.

### Performance And Resource Awareness

Evidence:
- `docs/PERFORMANCE.md` documents model choices, bottlenecks, tradeoffs, and benchmark usage.
- `src/benchmark_pipeline.py` measures runtime and peak Python-tracked memory per stage.
- Benchmark supports cached similarity outputs for stable local/CI-style runs and `--recompute-similarity` for full MiniLM recomputation.

Covered topics:
- ML models used: BM25 and MiniLM
- profiling method
- runtime and memory awareness
- bottlenecks and tradeoffs
- dataset scale assumptions

### Reproducible Setup And Execution

Evidence:
- `README.md` includes setup, `.env`, local PostgreSQL, run commands, testing, and dashboard instructions.
- `.env.example` lists required local PostgreSQL and Autotask variables.
- `requirements.txt` pins runtime dependencies.
- `pyproject.toml` records package metadata and Python version.

Expected quickstart:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py pipeline --all-tickets
.\venv\Scripts\python.exe main.py dashboard
```

### Testing Strategy

Evidence:
- `tests/test_assignment_scorer.py`: scoring helper unit tests
- `tests/test_clean_ticket_data.py`: cleaning and edge-case tests
- `tests/test_feature_pipeline.py`: integration-style feature workflow test
- `tests/test_database_config.py`: local-only database configuration tests
- `.github/workflows/ci.yml`: CI test runner

Command:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Release Notes And Release Documentation

Evidence:
- `CHANGELOG.md`: detailed changes
- `RELEASE_NOTES.md`: end-user release summary
- `docs/API_REFERENCE.md`: command and callable reference
- `README.md`: installation guide
- `MIGRATION_GUIDE.md`: upgrade steps
- `KNOWN_ISSUES.md`: limitations

### README And License

Evidence:
- `README.md`: overview, architecture, setup, execution, testing, benchmark, release readiness
- `LICENSE`: MIT license
- `pyproject.toml`: license metadata

### Packaging And Release Readiness

Evidence:
- `main.py`: clean runnable command entry point
- `pyproject.toml`: version and dependency metadata
- `requirements.txt`: pinned dependencies
- `RELEASE_NOTES.md`: release notes
- `README.md`: release tag instructions

Remaining manual step:
- create and push a final tag, for example `v1.0.0`, after final validation.

### Presentation Slides And Delivery

Evidence:
- `docs/PRESENTATION_OUTLINE.md` provides slide content and timing.

Remaining manual step:
- convert the outline into the final slide deck and practice delivery within the assigned time.
