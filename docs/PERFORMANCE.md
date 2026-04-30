# Performance and Resource Notes

## Models Used

The project currently uses:
- TF-IDF for sparse lexical similarity
- BM25 for ranked text relevance
- `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity
- rule-based and feature-based workload, SLA, and complexity scoring

## Why These Choices Were Made

- TF-IDF and BM25 are lightweight and easy to explain.
- MiniLM improves meaning-level similarity without requiring a very large embedding model.
- The final recommendation score blends text similarity with operational constraints such as workload, SLA, and complexity.

## Dataset Scale Assumption

The current implementation is intended for small-to-medium departmental ticket datasets. During development, the pipeline has been exercised on low-thousands-scale ticket data rather than millions of records.

## Resource Awareness

- TF-IDF and BM25 are CPU-friendly for the current scale.
- MiniLM embeddings are more expensive than lexical similarity, so they are used as a weighted component rather than the entire decision system.
- The Streamlit dashboard favors readability and interactivity over minimal render cost.

## Bottlenecks and Tradeoffs

### Main Bottlenecks
- sentence embedding generation for text similarity
- repeated CSV loading in local workflows
- dashboard complexity in a single large Streamlit file

### Tradeoffs
- higher semantic quality from MiniLM increases compute cost compared with TF-IDF only
- keeping generated outputs as CSV plus PostgreSQL improves traceability, but adds I/O overhead
- recommendation recomputation after simulated dispatch improves realism, but adds extra scoring work

## Suggested Profiling Method

The repository now includes automated tests, but deeper runtime profiling should be captured during demos or final benchmarking. A simple local timing command is:

```powershell
Measure-Command { .\venv\Scripts\python.exe main.py pipeline --all-tickets }
```

This gives an end-to-end runtime measurement for the full pipeline on the current machine.

## What to Report in the Presentation

- why the project uses a hybrid of lexical and embedding models
- that workload balancing is part of the scoring logic, not only text similarity
- that current performance is reasonable for course-project dataset scale
- that the main future optimization targets are embedding generation, dashboard refactoring, and cached intermediate results
