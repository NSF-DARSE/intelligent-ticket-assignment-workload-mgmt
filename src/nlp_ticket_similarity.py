from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FEATURE_DATA_PATH = Path("data/Feature_Engineered/autotask_feature_engineered.csv")
OUTPUT_DIR = Path("data/NLP")
SIMILARITY_MATCHES_PATH = OUTPUT_DIR / "ticket_similarity_matches.csv"
SIMILARITY_SUMMARY_PATH = OUTPUT_DIR / "ticket_similarity_summary.csv"
RUN_SUMMARY_PATH = OUTPUT_DIR / "nlp_similarity_summary.json"

TOP_K = 5


def load_feature_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_DATA_PATH)
    return df


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def prepare_ticket_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = df[df["resolution_hours"].notna()].copy()
    open_tickets = df[df["is_active_ticket"]].copy()

    completed["nlp_text"] = normalize_text(completed["ticket_text"])
    open_tickets["nlp_text"] = normalize_text(open_tickets["ticket_text"])

    completed = completed[completed["nlp_text"].ne("")].copy()
    open_tickets = open_tickets[open_tickets["nlp_text"].ne("")].copy()

    return completed, open_tickets


def build_similarity_outputs(
    completed: pd.DataFrame, open_tickets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if completed.empty or open_tickets.empty:
        return pd.DataFrame(), pd.DataFrame(), {"open_ticket_count": int(len(open_tickets)), "match_rows": 0}

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
    )

    completed_matrix = vectorizer.fit_transform(completed["nlp_text"])
    open_matrix = vectorizer.transform(open_tickets["nlp_text"])
    similarity_matrix = cosine_similarity(open_matrix, completed_matrix)

    match_rows: list[dict] = []
    summary_rows: list[dict] = []

    completed_reset = completed.reset_index(drop=True)
    open_reset = open_tickets.reset_index(drop=True)

    for open_idx, ticket in open_reset.iterrows():
        scores = similarity_matrix[open_idx]
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        top_scores = scores[top_indices]
        matched_tickets = completed_reset.iloc[top_indices].copy()
        matched_tickets["similarity_score"] = top_scores

        weighted_resolution = np.average(
            matched_tickets["resolution_hours"], weights=np.clip(top_scores, 1e-6, None)
        )
        median_resolution = matched_tickets["resolution_hours"].median()
        top_technician = matched_tickets["completed_by"].mode().iloc[0] if not matched_tickets.empty else pd.NA
        top_issue_type = matched_tickets["issue_type"].mode().iloc[0] if not matched_tickets.empty else pd.NA

        for rank, (_, match) in enumerate(matched_tickets.iterrows(), start=1):
            match_rows.append(
                {
                    "open_ticket_id": ticket["ticket_id"],
                    "open_ticket_title": ticket["title"],
                    "open_ticket_priority": ticket["priority"],
                    "open_ticket_sla_class": ticket.get("sla_priority_class"),
                    "open_ticket_issue_type": ticket["issue_type"],
                    "match_rank": rank,
                    "matched_ticket_id": match["ticket_id"],
                    "matched_ticket_title": match["title"],
                    "matched_issue_type": match["issue_type"],
                    "matched_completed_by": match["completed_by"],
                    "matched_resolution_hours": round(float(match["resolution_hours"]), 2),
                    "matched_priority": match["priority"],
                    "similarity_score": round(float(match["similarity_score"]), 4),
                }
            )

        summary_rows.append(
            {
                "ticket_id": ticket["ticket_id"],
                "ticket_title": ticket["title"],
                "priority": ticket["priority"],
                "sla_priority_class": ticket.get("sla_priority_class"),
                "issue_type": ticket["issue_type"],
                "top_similarity_score": round(float(top_scores[0]), 4),
                "avg_similarity_score_top5": round(float(np.mean(top_scores)), 4),
                "estimated_resolution_hours_nlp": round(float(weighted_resolution), 2),
                "median_resolution_hours_top5": round(float(median_resolution), 2),
                "suggested_technician_from_similarity": top_technician,
                "suggested_issue_type_from_similarity": top_issue_type,
                "top_match_ticket_id": matched_tickets.iloc[0]["ticket_id"],
                "top_match_title": matched_tickets.iloc[0]["title"],
            }
        )

    matches_df = pd.DataFrame(match_rows)
    summary_df = pd.DataFrame(summary_rows)

    run_summary = {
        "open_ticket_count": int(len(open_reset)),
        "completed_ticket_count": int(len(completed_reset)),
        "match_rows": int(len(matches_df)),
        "summary_rows": int(len(summary_df)),
        "top_k": TOP_K,
        "average_top_similarity": round(float(summary_df["top_similarity_score"].mean()), 4),
        "average_estimated_resolution_hours_nlp": round(
            float(summary_df["estimated_resolution_hours_nlp"].mean()), 2
        ),
    }

    return matches_df, summary_df, run_summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_feature_data()
    completed, open_tickets = prepare_ticket_sets(df)
    matches_df, summary_df, run_summary = build_similarity_outputs(completed, open_tickets)

    matches_df.to_csv(SIMILARITY_MATCHES_PATH, index=False)
    summary_df.to_csv(SIMILARITY_SUMMARY_PATH, index=False)
    RUN_SUMMARY_PATH.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(f"Similarity matches saved to: {SIMILARITY_MATCHES_PATH}")
    print(f"Similarity summary saved to: {SIMILARITY_SUMMARY_PATH}")
    print(f"Run summary saved to: {RUN_SUMMARY_PATH}")
    print(f"Open tickets processed: {run_summary['open_ticket_count']}")
    print(f"Match rows: {run_summary['match_rows']}")


if __name__ == "__main__":
    main()
