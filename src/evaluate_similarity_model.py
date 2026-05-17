from __future__ import annotations

"""Evaluate BM25 + MiniLM technician prediction on historical completed tickets.

This script uses a leave-one-out setup:
1. take each completed ticket as the query ticket
2. compare it against the remaining completed tickets
3. rank technicians by the summed hybrid similarity of the top-K matches
4. measure whether the actual resolving technician appears in rank 1 or top 3

The goal is to report a realistic ranking metric for the active text-similarity
model rather than a generic classifier accuracy.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nlp_ticket_similarity import (
    EMBEDDING_MODEL_NAME,
    HYBRID_BM25_WEIGHT,
    HYBRID_EMBEDDING_WEIGHT,
    TOP_K,
    bm25_scores,
    build_bm25_index,
    load_embedding_model,
    normalize_scores,
    normalize_text,
)


TRAINING_DATA_PATH = Path("data/Feature_Engineered/autotask_training_dataset.csv")
RESULTS_DIR = Path("data/Evaluation")
RESULTS_PATH = RESULTS_DIR / "similarity_model_accuracy.json"


def load_completed_tickets() -> pd.DataFrame:
    df = pd.read_csv(TRAINING_DATA_PATH)
    df = df[df["completed_by"].notna() & df["ticket_text"].notna()].copy()
    df["completed_by"] = df["completed_by"].astype(str).str.strip().str.lower()
    df["nlp_text"] = normalize_text(df["ticket_text"])
    df = df[df["nlp_text"].ne("")].reset_index(drop=True)
    return df


def technician_ranking_from_matches(matches: pd.DataFrame) -> list[str]:
    if matches.empty:
        return []

    technician_scores = (
        matches.groupby("completed_by")["hybrid_score"]
        .sum()
        .sort_values(ascending=False)
    )
    return technician_scores.index.tolist()


def evaluate_similarity_model(df: pd.DataFrame) -> dict:
    bm25_index = build_bm25_index(df["nlp_text"])
    embedding_model = load_embedding_model()
    embeddings = embedding_model.encode(
        df["nlp_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embedding_similarity_matrix = np.matmul(embeddings, embeddings.T)

    top_1_hits = 0
    top_3_hits = 0
    evaluated_rows = []

    for idx, ticket in df.iterrows():
        raw_bm25 = bm25_scores(ticket["nlp_text"], bm25_index)
        raw_bm25[idx] = 0.0
        normalized_bm25 = normalize_scores(raw_bm25)

        embedding_scores = embedding_similarity_matrix[idx].copy()
        embedding_scores[idx] = -1.0

        hybrid_scores = (
            (HYBRID_BM25_WEIGHT * normalized_bm25)
            + (HYBRID_EMBEDDING_WEIGHT * embedding_scores)
        )
        hybrid_scores[idx] = -1.0

        top_indices = np.argsort(hybrid_scores)[::-1][:TOP_K]
        top_matches = df.iloc[top_indices][["ticket_id", "completed_by", "issue_type"]].copy()
        top_matches["hybrid_score"] = hybrid_scores[top_indices]
        top_matches = top_matches[top_matches["hybrid_score"] >= 0].copy()

        ranked_technicians = technician_ranking_from_matches(top_matches)
        actual_technician = ticket["completed_by"]

        if ranked_technicians:
            if ranked_technicians[0] == actual_technician:
                top_1_hits += 1
            if actual_technician in ranked_technicians[:3]:
                top_3_hits += 1

        evaluated_rows.append(
            {
                "ticket_id": ticket["ticket_id"],
                "actual_technician": actual_technician,
                "predicted_top_1": ranked_technicians[0] if ranked_technicians else None,
                "predicted_top_3": ranked_technicians[:3],
                "hit_top_1": bool(ranked_technicians and ranked_technicians[0] == actual_technician),
                "hit_top_3": bool(actual_technician in ranked_technicians[:3]),
            }
        )

    evaluated_count = len(evaluated_rows)
    results = {
        "evaluated_ticket_count": evaluated_count,
        "technician_count": int(df["completed_by"].nunique()),
        "top_k_neighbors": TOP_K,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "bm25_weight": HYBRID_BM25_WEIGHT,
        "embedding_weight": HYBRID_EMBEDDING_WEIGHT,
        "top_1_accuracy": round(top_1_hits / evaluated_count, 4) if evaluated_count else 0.0,
        "top_3_hit_rate": round(top_3_hits / evaluated_count, 4) if evaluated_count else 0.0,
        "per_technician_ticket_counts": df["completed_by"].value_counts().to_dict(),
    }
    return results


def write_results_with_fallback(results: dict) -> Path:
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return RESULTS_PATH
    except PermissionError:
        fallback_path = Path("similarity_model_accuracy_latest.json")
        fallback_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return fallback_path


def main() -> None:
    completed_tickets = load_completed_tickets()
    results = evaluate_similarity_model(completed_tickets)
    output_path = write_results_with_fallback(results)

    print(f"Evaluation results saved to: {output_path}")
    print(f"Completed tickets evaluated: {results['evaluated_ticket_count']}")
    print(f"Technicians evaluated: {results['technician_count']}")
    print(f"Top-1 accuracy: {results['top_1_accuracy']:.2%}")
    print(f"Top-3 hit rate: {results['top_3_hit_rate']:.2%}")


if __name__ == "__main__":
    main()
