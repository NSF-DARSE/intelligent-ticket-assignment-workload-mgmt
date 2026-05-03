from __future__ import annotations

"""Benchmark the local recommendation pipeline stages.

The benchmark measures stage runtime and Python-tracked peak memory using the
current local datasets. It intentionally avoids overwriting the working output
CSV files so it can be run safely during development and presentations.
"""

import argparse
import json
import platform
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from sqlalchemy import create_engine

import assignment_scorer
import clean_employee_skills
import clean_ticket_data
import complexity_scoring
import feature_engineering
import load_outputs_to_postgres
import nlp_ticket_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "data" / "Evaluation"
BENCHMARK_JSON_PATH = EVALUATION_DIR / "performance_benchmark.json"
BENCHMARK_MD_PATH = EVALUATION_DIR / "performance_benchmark.md"
IN_MEMORY_OUTPUT_LABEL = "in-memory benchmark"


@dataclass(frozen=True)
class StageDefinition:
    name: str
    runner: Callable[[], dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the local recommendation pipeline stages.")
    parser.add_argument(
        "--include-db-load",
        action="store_true",
        help="Include the PostgreSQL output load stage in the benchmark.",
    )
    return parser.parse_args()


def ensure_output_dir() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def measure_stage(stage: StageDefinition) -> dict:
    tracemalloc.start()
    start = time.perf_counter()
    result = stage.runner()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result.update(
        {
            "stage": stage.name,
            "runtime_seconds": round(elapsed, 3),
            "peak_memory_mb": round(peak / (1024 * 1024), 3),
        }
    )
    return result


def benchmark_clean_data() -> dict:
    _, summary = clean_ticket_data.clean_ticket_data()
    return {
        "input_rows": int(summary["output_rows"]),
        "output_rows": int(summary["output_rows"]),
        "output_columns": int(summary["output_columns"]),
        "primary_output": IN_MEMORY_OUTPUT_LABEL,
    }


def benchmark_feature_engineering() -> dict:
    cleaned_df = feature_engineering.load_cleaned_data()
    _, _, _, _, summary = feature_engineering.engineer_features(cleaned_df)
    return {
        "input_rows": int(summary["input_rows"]),
        "output_rows": int(summary["feature_rows"]),
        "open_ticket_rows": int(summary["open_ticket_rows"]),
        "training_rows": int(summary["training_rows"]),
        "primary_output": IN_MEMORY_OUTPUT_LABEL,
    }


def benchmark_skill_normalization() -> dict:
    source_df = pd.read_csv(clean_employee_skills.SOURCE_PATH)
    profile_df = clean_employee_skills.build_profile_dataset(source_df)
    normalized_df = clean_employee_skills.build_normalized_dataset(profile_df)
    return {
        "input_rows": int(len(source_df)),
        "output_rows": int(len(normalized_df)),
        "employee_rows": int(len(profile_df)),
        "primary_output": IN_MEMORY_OUTPUT_LABEL,
    }


def benchmark_similarity() -> dict:
    feature_df = nlp_ticket_similarity.load_feature_data()
    completed_df, open_tickets_df = nlp_ticket_similarity.prepare_ticket_sets(feature_df)
    matches_df, summary_df, run_summary = nlp_ticket_similarity.build_similarity_outputs(completed_df, open_tickets_df)
    return {
        "input_rows": int(len(feature_df)),
        "completed_ticket_rows": int(run_summary["completed_ticket_count"]),
        "open_ticket_rows": int(run_summary["open_ticket_count"]),
        "output_rows": int(run_summary["match_rows"]),
        "summary_rows": int(run_summary["summary_rows"]),
        "primary_output": IN_MEMORY_OUTPUT_LABEL,
    }


def benchmark_complexity() -> dict:
    feature_df, nlp_summary_df = complexity_scoring.load_inputs()
    scored_df = complexity_scoring.add_effort_signal(feature_df)
    scored_df = complexity_scoring.add_nlp_signal(scored_df, nlp_summary_df)
    scored_df = complexity_scoring.add_complexity_score(scored_df)
    scored_df["complexity_reason"] = scored_df.apply(complexity_scoring.build_reason, axis=1)
    return {
        "input_rows": int(len(feature_df)),
        "output_rows": int(len(scored_df)),
        "primary_output": IN_MEMORY_OUTPUT_LABEL,
    }


def benchmark_recommendations() -> dict:
    feature_df = assignment_scorer.load_feature_data()
    _, _, summary = assignment_scorer.recommend_assignments(feature_df)
    return {
        "input_rows": int(len(feature_df)),
        "output_rows": int(summary["recommendation_rows"]),
        "open_ticket_rows": int(summary["open_ticket_count"]),
        "technician_pool_size": int(summary["technician_pool_size"]),
        "primary_output": IN_MEMORY_OUTPUT_LABEL,
    }


def benchmark_db_load() -> dict:
    engine = create_engine(load_outputs_to_postgres.get_db_url())
    csv_results = load_outputs_to_postgres.load_csv_tables(engine)
    json_results = load_outputs_to_postgres.load_json_tables(engine)
    return {
        "csv_tables_loaded": int(len(csv_results)),
        "json_tables_loaded": int(len(json_results)),
        "rows_loaded": int(sum(result["rows_loaded"] for result in csv_results + json_results)),
        "primary_output": "azure-postgresql",
    }


def build_stage_notes(stage: dict) -> str:
    notes: list[str] = []
    if "open_ticket_rows" in stage:
        notes.append(f"open={stage['open_ticket_rows']}")
    if "training_rows" in stage:
        notes.append(f"training={stage['training_rows']}")
    if "summary_rows" in stage:
        notes.append(f"summary={stage['summary_rows']}")
    if "technician_pool_size" in stage:
        notes.append(f"techs={stage['technician_pool_size']}")
    if "csv_tables_loaded" in stage:
        notes.append(f"csv={stage['csv_tables_loaded']}, json={stage['json_tables_loaded']}")
    return "; ".join(notes)


def build_markdown_report(payload: dict) -> str:
    lines = [
        "# Pipeline Performance Benchmark",
        "",
        f"- Benchmarked at: `{payload['benchmarked_at_utc']}`",
        f"- Python: `{payload['python_version']}`",
        f"- Platform: `{payload['platform']}`",
        f"- Included DB load: `{payload['included_db_load']}`",
        "",
        "| Stage | Runtime (s) | Peak Memory (MB) | Input Rows | Output Rows | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for stage in payload["stages"]:
        lines.append(
            f"| {stage['stage']} | {stage['runtime_seconds']} | {stage['peak_memory_mb']} | "
            f"{stage.get('input_rows', '')} | {stage.get('output_rows', '')} | {build_stage_notes(stage)} |"
        )

    lines.extend(
        [
            "",
            f"**Total runtime:** `{payload['total_runtime_seconds']} s`",
            f"**Slowest stage:** `{payload['slowest_stage']['stage']}` (`{payload['slowest_stage']['runtime_seconds']} s`)",
            "",
            "Interpretation:",
            "- BM25 + MiniLM similarity is expected to be one of the costlier stages because it builds lexical scores and semantic embeddings.",
            "- Recommendation scoring adds workload and skill-aware ranking across the active ticket pool.",
            "- Database loading adds I/O overhead and should be reported separately when included.",
            "- This benchmark measures stage computation on the current local datasets; it does not overwrite the working CSV outputs.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict) -> None:
    BENCHMARK_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    BENCHMARK_MD_PATH.write_text(build_markdown_report(payload), encoding="utf-8")


def get_stage_definitions(include_db_load: bool) -> list[StageDefinition]:
    stages = [
        StageDefinition("clean_ticket_data", benchmark_clean_data),
        StageDefinition("feature_engineering", benchmark_feature_engineering),
        StageDefinition("employee_skill_normalization", benchmark_skill_normalization),
        StageDefinition("bm25_minilm_similarity", benchmark_similarity),
        StageDefinition("complexity_scoring", benchmark_complexity),
        StageDefinition("recommendation_scoring", benchmark_recommendations),
    ]
    if include_db_load:
        stages.append(StageDefinition("postgres_output_load", benchmark_db_load))
    return stages


def main() -> None:
    args = parse_args()
    ensure_output_dir()

    results = [measure_stage(stage) for stage in get_stage_definitions(args.include_db_load)]
    total_runtime = round(sum(stage["runtime_seconds"] for stage in results), 3)
    slowest_stage = max(results, key=lambda stage: stage["runtime_seconds"])

    payload = {
        "benchmarked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "included_db_load": args.include_db_load,
        "total_runtime_seconds": total_runtime,
        "slowest_stage": {
            "stage": slowest_stage["stage"],
            "runtime_seconds": slowest_stage["runtime_seconds"],
        },
        "stages": results,
    }

    write_outputs(payload)

    print(f"Benchmark JSON saved to: {BENCHMARK_JSON_PATH}")
    print(f"Benchmark report saved to: {BENCHMARK_MD_PATH}")
    print(f"Total runtime: {total_runtime} seconds")
    print(f"Slowest stage: {slowest_stage['stage']} ({slowest_stage['runtime_seconds']} seconds)")


if __name__ == "__main__":
    main()
