from __future__ import annotations

"""Evaluate the full technician recommendation scorer on historical tickets.

This script backtests the current downstream recommendation logic on completed
tickets. Each completed ticket is treated as a query ticket, compared against
the remaining completed tickets for text expertise, and then ranked across the
technician pool using the current assignment-scoring heuristics.

Important note:
- This is a historical static backtest.
- Workload-sensitive features are evaluated with a neutral workload state,
  because true live workload at the moment of each historical ticket is not
  recoverable from the exported dataset.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from assignment_scorer import (
    TECHNICIAN_WEIGHTS,
    build_technician_history,
    canonicalize_technician_key,
    complexity_fit_score,
    compute_skill_alignment,
    compute_skill_score,
    default_workload_record,
    distribution_penalty,
    experienced_low_risk_penalty,
    exploration_capacity_bonus,
    get_technician_pool,
    high_risk_new_technician_block,
    is_eligible_technician_candidate,
    is_high_risk_ticket,
    is_new_technician,
    load_employee_skills,
    new_technician_bonus,
    new_technician_penalty,
    new_technician_top1_cap_block,
    priority_balance_score,
    safe_ratio,
    sla_pressure_score,
    sla_urgency_fit_score,
)
from evaluate_similarity_model import load_completed_tickets, technician_ranking_from_matches
from nlp_ticket_similarity import (
    EMBEDDING_MODEL_NAME,
    HYBRID_BM25_WEIGHT,
    HYBRID_EMBEDDING_WEIGHT,
    TOP_K,
    bm25_scores,
    build_bm25_index,
    load_embedding_model,
    normalize_scores,
)


COMPLEXITY_DATA_PATH = Path("data/Complexity/autotask_complexity_scored.csv")
RESULTS_DIR = Path("data/Evaluation")
RESULTS_PATH = RESULTS_DIR / "full_recommendation_model_accuracy.json"


def load_complexity_lookup() -> dict[str, dict]:
    if not COMPLEXITY_DATA_PATH.exists():
        return {}
    complexity_df = pd.read_csv(COMPLEXITY_DATA_PATH)
    complexity_df["ticket_id"] = complexity_df["ticket_id"].astype(str)
    keep_columns = ["ticket_id", "complexity_score", "complexity_class", "complexity_reason"]
    complexity_df = complexity_df[keep_columns].drop_duplicates(subset=["ticket_id"])
    return complexity_df.set_index("ticket_id").to_dict(orient="index")


def build_text_expertise_for_ticket(
    ticket_index: int,
    ticket_row: pd.Series,
    candidates: pd.DataFrame,
    bm25_index: dict,
    embedding_similarity_row: np.ndarray,
) -> dict[str, dict]:
    raw_bm25 = bm25_scores(ticket_row["nlp_text"], bm25_index)
    normalized_bm25 = normalize_scores(raw_bm25)
    embedding_scores = embedding_similarity_row
    hybrid_scores = (
        (HYBRID_BM25_WEIGHT * normalized_bm25)
        + (HYBRID_EMBEDDING_WEIGHT * embedding_scores)
    )

    top_indices = np.argsort(hybrid_scores)[::-1][:TOP_K]
    top_matches = candidates.iloc[top_indices][["ticket_id", "completed_by"]].copy()
    top_matches["hybrid_score"] = hybrid_scores[top_indices]
    top_matches["bm25_score"] = normalized_bm25[top_indices]
    top_matches["embedding_score"] = embedding_scores[top_indices]
    top_matches = top_matches[top_matches["hybrid_score"] >= 0].copy()

    if top_matches.empty:
        return {}

    grouped = (
        top_matches.groupby("completed_by")
        .agg(
            text_match_count=("ticket_id", "count"),
            bm25_text_expertise_score=("hybrid_score", "mean"),
            best_text_match_score=("hybrid_score", "max"),
            embedding_text_expertise_score=("embedding_score", "mean"),
            best_embedding_match_score=("embedding_score", "max"),
        )
        .reset_index()
    )

    lookup: dict[str, dict] = {}
    for _, row in grouped.iterrows():
        lookup[str(row["completed_by"])] = {
            "bm25_text_expertise_score": round(float(row["bm25_text_expertise_score"]), 4),
            "best_text_match_score": round(float(row["best_text_match_score"]), 4),
            "embedding_text_expertise_score": round(float(row["embedding_text_expertise_score"]), 4),
            "best_embedding_match_score": round(float(row["best_embedding_match_score"]), 4),
            "text_match_count": int(row["text_match_count"]),
            "best_text_match_rank": np.nan,
        }

    return lookup


def score_ticket_against_pool(
    ticket: pd.Series,
    completed_reference: pd.DataFrame,
    technician_pool: list[str],
    text_expertise_lookup: dict[str, dict],
    skill_lookup: dict[str, dict],
    profile_lookup: dict[str, dict],
    history_lookup: dict[str, dict],
) -> list[str]:
    neutral_workload = default_workload_record()
    ranked_rows = []

    for technician in technician_pool:
        tech_history = history_lookup.get(
            technician,
            {"resolution_efficiency_score": 0.5, "avg_resolution_hours": np.nan, "completed_ticket_count": 0},
        )
        tech_workload = neutral_workload
        skill = compute_skill_score(ticket, technician, completed_reference)
        skill_alignment = compute_skill_alignment(ticket, technician, skill_lookup)
        text_expertise = text_expertise_lookup.get(
            technician,
            {
                "bm25_text_expertise_score": 0.0,
                "best_text_match_score": 0.0,
                "embedding_text_expertise_score": 0.0,
                "best_embedding_match_score": 0.0,
                "text_match_count": 0,
                "best_text_match_rank": np.nan,
            },
        )
        if not is_eligible_technician_candidate(tech_history, text_expertise, skill_alignment):
            continue
        balance_score = priority_balance_score(ticket, pd.Series(tech_workload))
        sla_pressure = sla_pressure_score(ticket, pd.Series(tech_workload))
        sla_urgency_fit = sla_urgency_fit_score(ticket, skill, tech_workload)
        complexity_fit = complexity_fit_score(ticket, tech_workload)
        fairness_penalty = distribution_penalty(tech_workload, 0.0, 0.0)
        exploration_bonus = new_technician_bonus(tech_history, tech_workload, ticket)
        low_risk_senior_penalty = experienced_low_risk_penalty(tech_history, tech_workload, ticket)
        low_risk_new_tech_bonus = exploration_capacity_bonus(tech_history, tech_workload, ticket, 0)
        onboarding_penalty = new_technician_penalty(tech_history, tech_workload, ticket, 0)
        skill_experience_score = round(
            min(1.0, (skill_alignment["skill_alignment_score"] * 0.65) + (skill["issue_type_skill"] * 0.35)),
            4,
        )

        total_score = (
            TECHNICIAN_WEIGHTS["issue_type_skill"] * skill["issue_type_skill"]
            + TECHNICIAN_WEIGHTS["skill_experience"] * skill_experience_score
            + TECHNICIAN_WEIGHTS["bm25_text_expertise"] * text_expertise["bm25_text_expertise_score"]
            + TECHNICIAN_WEIGHTS["queue_group_skill"] * skill["queue_group_skill"]
            + TECHNICIAN_WEIGHTS["account_familiarity"] * skill["account_familiarity"]
            + TECHNICIAN_WEIGHTS["workload_hours"] * tech_workload["workload_hours_score"]
            + TECHNICIAN_WEIGHTS["workload_count"] * tech_workload["workload_count_score"]
            + TECHNICIAN_WEIGHTS["priority_balance"] * balance_score
            + TECHNICIAN_WEIGHTS["resolution_efficiency"] * tech_history["resolution_efficiency_score"]
            + TECHNICIAN_WEIGHTS["sla_pressure"] * sla_pressure
            + TECHNICIAN_WEIGHTS["sla_urgency_fit"] * sla_urgency_fit
            + TECHNICIAN_WEIGHTS["complexity_fit"] * complexity_fit
        )
        total_score = max(
            0.0,
            total_score
            - fairness_penalty
            - onboarding_penalty
            - low_risk_senior_penalty
            + exploration_bonus
            + low_risk_new_tech_bonus,
        )
        if high_risk_new_technician_block(tech_history, ticket):
            total_score = total_score * 0.05
        if new_technician_top1_cap_block(tech_history, 0):
            total_score = total_score * 0.10

        ranked_rows.append(
            {
                "technician": technician,
                "score": round(float(total_score), 4),
                "is_new_technician": is_new_technician(tech_history),
                "high_risk_ticket": is_high_risk_ticket(ticket),
                "employee_name": profile_lookup.get(technician, {}).get(
                    "employee_name",
                    skill_alignment["employee_name"],
                ),
            }
        )

    ranked_rows = sorted(ranked_rows, key=lambda row: row["score"], reverse=True)
    return [row["technician"] for row in ranked_rows]


def evaluate_full_recommendation_model(df: pd.DataFrame) -> dict:
    complexity_lookup = load_complexity_lookup()
    skill_lookup, profile_lookup = load_employee_skills()
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
        candidates = df.drop(index=idx).reset_index(drop=True)
        if candidates.empty:
            continue

        ticket_id = str(ticket["ticket_id"])
        ticket_eval = ticket.copy()
        for key, value in complexity_lookup.get(ticket_id, {}).items():
            ticket_eval[key] = value

        bm25_index = build_bm25_index(candidates["nlp_text"])
        embedding_row = np.delete(embedding_similarity_matrix[idx], idx)
        text_expertise_lookup = build_text_expertise_for_ticket(
            ticket_index=idx,
            ticket_row=ticket_eval,
            candidates=candidates,
            bm25_index=bm25_index,
            embedding_similarity_row=embedding_row,
        )
        history = build_technician_history(candidates)
        history_lookup = history.set_index("technician").to_dict(orient="index") if not history.empty else {}
        technician_pool = get_technician_pool(candidates)
        technician_pool = sorted(set(technician_pool) | set(skill_lookup.keys()))
        ranked_technicians = score_ticket_against_pool(
            ticket=ticket_eval,
            completed_reference=candidates,
            technician_pool=technician_pool,
            text_expertise_lookup=text_expertise_lookup,
            skill_lookup=skill_lookup,
            profile_lookup=profile_lookup,
            history_lookup=history_lookup,
        )

        actual_technician = canonicalize_technician_key(ticket["completed_by"])
        if ranked_technicians:
            if ranked_technicians[0] == actual_technician:
                top_1_hits += 1
            if actual_technician in ranked_technicians[:3]:
                top_3_hits += 1

        evaluated_rows.append(
            {
                "ticket_id": ticket_id,
                "actual_technician": actual_technician,
                "predicted_top_1": ranked_technicians[0] if ranked_technicians else None,
                "predicted_top_3": ranked_technicians[:3],
                "hit_top_1": bool(ranked_technicians and ranked_technicians[0] == actual_technician),
                "hit_top_3": bool(actual_technician in ranked_technicians[:3]),
            }
        )

    evaluated_count = len(evaluated_rows)
    results = {
        "evaluation_type": "historical_static_backtest_full_recommendation_scorer",
        "note": (
            "This backtest uses historical completed tickets and neutral workload "
            "state. It evaluates BM25 + MiniLM text matching plus downstream skill, "
            "history, SLA, and complexity heuristics."
        ),
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
        fallback_path = Path("full_recommendation_model_accuracy_latest.json")
        fallback_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return fallback_path


def main() -> None:
    completed_tickets = load_completed_tickets()
    results = evaluate_full_recommendation_model(completed_tickets)
    output_path = write_results_with_fallback(results)

    print(f"Evaluation results saved to: {output_path}")
    print(f"Completed tickets evaluated: {results['evaluated_ticket_count']}")
    print(f"Technicians evaluated: {results['technician_count']}")
    print(f"Top-1 accuracy: {results['top_1_accuracy']:.2%}")
    print(f"Top-3 hit rate: {results['top_3_hit_rate']:.2%}")


if __name__ == "__main__":
    main()
