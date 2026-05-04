# Final Presentation Outline

Use this outline to build the final slide deck for CISC 867-010.

## Slide 1: Title

**Intelligent Ticket Assignment & Workload Management**

Include:
- team/project name
- course
- presentation date
- one-sentence project goal

## Slide 2: Problem And Motivation

Explain:
- service desks need fast and fair ticket routing
- manual assignment can overload technicians
- ticket text, SLA, skill fit, and workload all matter

## Slide 3: System Architecture

Show:
- Autotask API input
- local CSV/JSON pipeline artifacts
- local PostgreSQL reporting tables
- Streamlit dashboard

## Slide 4: Data Pipeline

Walk through:
- fetch raw tickets
- clean data
- engineer features
- normalize employee skills
- generate NLP similarity outputs
- score complexity
- rank technician recommendations

## Slide 5: ML And Scoring Approach

Cover:
- BM25 lexical matching
- MiniLM semantic similarity
- hybrid weights: BM25 `0.40`, MiniLM `0.60`
- skill alignment
- workload and SLA adjustments

## Slide 6: Recommendation Output

Show:
- top-3 technician recommendations
- score components
- rationale field
- dispatch board behavior

## Slide 7: Local PostgreSQL And Dashboard

Cover:
- local PostgreSQL as the reporting backend
- key tables such as `autotask_assignment_recommendations`
- dashboard reads/writes dispatch simulation locally

## Slide 8: Testing And Reproducibility

Include:
- `requirements.txt` and `pyproject.toml`
- `.env.example`
- `main.py pipeline --all-tickets`
- unit tests and integration-style tests
- GitHub Actions checks

## Slide 9: Performance And Resource Awareness

Include:
- benchmark runner: `src/benchmark_pipeline.py`
- MiniLM as the expensive stage
- cached similarity benchmark mode
- documented bottlenecks and tradeoffs

## Slide 10: Release Readiness

Show:
- README
- license
- changelog
- release notes
- API reference
- migration guide
- known issues
- release tag plan

## Slide 11: Limitations And Future Work

Mention:
- dashboard should be split into smaller modules
- full MiniLM recomputation needs model cache/download
- assignment actions are simulated and do not update Autotask yet
- possible future Autotask write-back

## Slide 12: Demo And Closing

Demo:
- run dashboard
- show recommendation board
- show technician/skill view
- show local PostgreSQL-backed dispatch simulation

Close with:
- what works
- how it is reproducible
- how it satisfies the rubric
