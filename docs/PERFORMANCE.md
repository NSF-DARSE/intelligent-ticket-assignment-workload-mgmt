# Performance and Resource Notes

## Models Used

The project currently uses:
- **BM25** for lexical text retrieval
- **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`) for semantic similarity
- rule-based and feature-based scoring for workload, SLA pressure, skill alignment, and complexity

## Why These Choices Were Made

- BM25 gives an interpretable lexical baseline for matching open tickets to historical tickets.
- MiniLM improves semantic matching when wording differs across similar incidents.
- A weighted hybrid score keeps the model explainable while still capturing meaning-level similarity.
- The final recommendation layer combines text similarity with operational constraints rather than relying on text alone.

## Current Hybrid Weights

- BM25: `0.40`
- MiniLM: `0.60`

## Dataset Scale Assumption

The current implementation is intended for small-to-medium ticket datasets. During development, the pipeline has been exercised on low-thousands-scale ticket data rather than very large enterprise-scale archives.

## Resource Awareness

- BM25 is CPU-friendly for the current scale.
- MiniLM embedding generation is the most expensive modeling step in the pipeline.
- The dashboard optimizes for readability and reviewability rather than minimal render cost.
- PostgreSQL loading improves traceability and dashboard access, but adds I/O overhead compared with a CSV-only workflow.

## Main Bottlenecks

- sentence embedding generation for active and completed ticket text
- repeated CSV loading during local development runs
- dashboard logic concentrated in a single large Streamlit file

## Tradeoffs

- adding MiniLM improves semantic quality but increases runtime compared with BM25 alone
- storing intermediate outputs as files plus PostgreSQL tables makes debugging easier but increases storage and I/O work
- simulated dispatch recomputation adds realism to the dashboard but performs extra scoring work

## Suggested Profiling Method

For a quick end-to-end measurement:

```powershell
Measure-Command { .\venv\Scripts\python.exe main.py pipeline --all-tickets }
```

This measures the full pipeline on the current machine and dataset.

## What to Report in the Presentation

- why the project uses a hybrid lexical + semantic similarity model
- why the recommendation logic includes workload, SLA, and complexity beyond text matching
- that MiniLM is the costliest model component in the current workflow
- that the main future optimization targets are embedding generation, dashboard refactoring, and caching intermediate outputs
