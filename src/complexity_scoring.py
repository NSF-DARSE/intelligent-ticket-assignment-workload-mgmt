from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_DATA_PATH = Path("data/Feature_Engineered/autotask_feature_engineered.csv")
OPEN_TIME_ESTIMATION_PATH = Path("data/Time_Estimation/time_estimation_open_ticket_predictions.csv")
NLP_SUMMARY_PATH = Path("data/NLP/ticket_similarity_summary.csv")
OUTPUT_DIR = Path("data/Complexity")
COMPLEXITY_OUTPUT_PATH = OUTPUT_DIR / "autotask_complexity_scored.csv"
SUMMARY_PATH = OUTPUT_DIR / "complexity_scoring_summary.json"

SLA_COMPLEXITY_MAP = {
    "High": 1.0,
    "Medium": 0.55,
    "Low": 0.25,
    "Service Request": 0.35,
    "Maintenance": 0.45,
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_df = pd.read_csv(FEATURE_DATA_PATH)
    time_df = pd.read_csv(OPEN_TIME_ESTIMATION_PATH) if OPEN_TIME_ESTIMATION_PATH.exists() else pd.DataFrame()
    nlp_df = pd.read_csv(NLP_SUMMARY_PATH) if NLP_SUMMARY_PATH.exists() else pd.DataFrame()
    return feature_df, time_df, nlp_df


def normalize_series(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if pd.isna(minimum) or pd.isna(maximum) or minimum == maximum:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - minimum) / (maximum - minimum)


def build_issue_type_complexity_lookup(completed: pd.DataFrame) -> dict:
    issue_avg = completed.groupby("issue_type")["resolution_hours"].mean()
    normalized = normalize_series(issue_avg)
    return normalized.to_dict()


def build_keyword_score(df: pd.DataFrame) -> pd.Series:
    score = (
        df["ticket_text_has_server"].astype(int) * 0.30
        + df["ticket_text_has_backup"].astype(int) * 0.25
        + df["ticket_text_has_urgent_language"].astype(int) * 0.20
        + df["ticket_text_has_access_issue"].astype(int) * 0.10
        + df["ticket_text_has_vpn"].astype(int) * 0.05
        + df["ticket_text_has_email"].astype(int) * 0.05
        + df["ticket_text_has_printer"].astype(int) * 0.05
    )
    return score.clip(0, 1)


def add_effort_signal(df: pd.DataFrame, open_time_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if not open_time_df.empty:
        keep_cols = ["ticket_id", "predicted_resolution_hours_final"]
        df = df.merge(open_time_df[keep_cols], on="ticket_id", how="left")
    else:
        df["predicted_resolution_hours_final"] = np.nan

    df["effective_resolution_hours_for_complexity"] = np.where(
        df["resolution_hours"].notna(),
        df["resolution_hours"],
        df["predicted_resolution_hours_final"],
    )

    df["effort_complexity_score"] = normalize_series(df["effective_resolution_hours_for_complexity"].fillna(
        df["effective_resolution_hours_for_complexity"].median()
    ))

    return df


def add_nlp_signal(df: pd.DataFrame, nlp_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if not nlp_df.empty:
        keep_cols = ["ticket_id", "top_similarity_score", "avg_similarity_score_top5"]
        nlp_df = nlp_df.rename(
            columns={
                "top_similarity_score": "nlp_top_similarity_score",
                "avg_similarity_score_top5": "nlp_avg_similarity_score_top5",
            }
        )
        df = df.merge(nlp_df[["ticket_id", "nlp_top_similarity_score", "nlp_avg_similarity_score_top5"]], on="ticket_id", how="left")
    else:
        df["nlp_top_similarity_score"] = np.nan
        df["nlp_avg_similarity_score_top5"] = np.nan

    df["nlp_novelty_score"] = 1 - df["nlp_top_similarity_score"].fillna(0.5)
    return df


def add_complexity_score(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["resolution_hours"].notna()].copy()
    issue_type_lookup = build_issue_type_complexity_lookup(completed)

    df["issue_type_complexity_score"] = df["issue_type"].map(issue_type_lookup).fillna(0.5)
    df["sla_complexity_score"] = df["sla_priority_class"].map(SLA_COMPLEXITY_MAP).fillna(0.5)
    df["keyword_complexity_score"] = build_keyword_score(df)
    df["text_length_complexity_score"] = normalize_series(df["ticket_text_char_count"].fillna(df["ticket_text_char_count"].median()))

    combined = (
        0.40 * df["effort_complexity_score"]
        + 0.20 * df["issue_type_complexity_score"]
        + 0.15 * df["sla_complexity_score"]
        + 0.15 * df["nlp_novelty_score"]
        + 0.10 * ((df["keyword_complexity_score"] + df["text_length_complexity_score"]) / 2)
    )

    df["complexity_score"] = (1 + (combined * 4)).round(2)
    df["complexity_class"] = pd.cut(
        df["complexity_score"],
        bins=[0, 2.4, 3.6, 5.1],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype("string")

    return df


def build_reason(row: pd.Series) -> str:
    reasons = []

    if row["effort_complexity_score"] >= 0.7:
        reasons.append("high estimated effort")
    if row["sla_complexity_score"] >= 0.9:
        reasons.append("high SLA urgency")
    if row["issue_type_complexity_score"] >= 0.7:
        reasons.append("historically difficult issue type")
    if row["nlp_novelty_score"] >= 0.7:
        reasons.append("low similarity to routine historical tickets")
    if row["keyword_complexity_score"] >= 0.4:
        reasons.append("technical complexity keywords present")

    if not reasons:
        reasons.append("routine workload indicators")

    return "; ".join(reasons)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_df, open_time_df, nlp_df = load_inputs()
    scored = add_effort_signal(feature_df, open_time_df)
    scored = add_nlp_signal(scored, nlp_df)
    scored = add_complexity_score(scored)
    scored["complexity_reason"] = scored.apply(build_reason, axis=1)

    summary = {
        "rows_scored": int(len(scored)),
        "complexity_distribution": scored["complexity_class"].value_counts().to_dict(),
        "average_complexity_score": round(float(scored["complexity_score"].mean()), 2),
        "open_ticket_complexity_distribution": scored.loc[scored["is_active_ticket"], "complexity_class"].value_counts().to_dict(),
    }

    scored.to_csv(COMPLEXITY_OUTPUT_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Complexity dataset saved to: {COMPLEXITY_OUTPUT_PATH}")
    print(f"Complexity summary saved to: {SUMMARY_PATH}")
    print(f"Rows scored: {summary['rows_scored']}")


if __name__ == "__main__":
    main()
