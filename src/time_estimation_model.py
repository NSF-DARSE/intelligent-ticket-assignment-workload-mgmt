from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_DATA_PATH = Path("data/Feature_Engineered/autotask_feature_engineered.csv")
NLP_SUMMARY_PATH = Path("data/NLP/ticket_similarity_summary.csv")
OUTPUT_DIR = Path("data/Time_Estimation")
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "time_estimation_test_predictions.csv"
OPEN_PREDICTIONS_PATH = OUTPUT_DIR / "time_estimation_open_ticket_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "time_estimation_metrics.json"

TOP_K = 5
RANDOM_STATE = 42


def load_feature_data() -> pd.DataFrame:
    return pd.read_csv(FEATURE_DATA_PATH)


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def compute_completed_nlp_features(completed: pd.DataFrame) -> pd.DataFrame:
    completed = completed.copy().reset_index(drop=True)
    completed["nlp_text"] = normalize_text(completed["ticket_text"])

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=5000)
    matrix = vectorizer.fit_transform(completed["nlp_text"])
    similarity = (matrix * matrix.T).toarray()

    similarity_rows = []
    for idx, ticket in completed.iterrows():
        scores = similarity[idx].copy()
        scores[idx] = -1
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        top_scores = scores[top_indices]
        valid_mask = top_scores > 0
        top_indices = top_indices[valid_mask]
        top_scores = top_scores[valid_mask]

        if len(top_indices) == 0:
            similarity_rows.append(
                {
                    "ticket_id": ticket["ticket_id"],
                    "nlp_top_similarity_score": 0.0,
                    "nlp_avg_similarity_score_top5": 0.0,
                    "nlp_estimated_resolution_hours": np.nan,
                }
            )
            continue

        matched = completed.iloc[top_indices]
        weighted_resolution = np.average(
            matched["resolution_hours"], weights=np.clip(top_scores, 1e-6, None)
        )

        similarity_rows.append(
            {
                "ticket_id": ticket["ticket_id"],
                "nlp_top_similarity_score": round(float(top_scores[0]), 4),
                "nlp_avg_similarity_score_top5": round(float(np.mean(top_scores)), 4),
                "nlp_estimated_resolution_hours": round(float(weighted_resolution), 2),
            }
        )

    return completed.merge(pd.DataFrame(similarity_rows), on="ticket_id", how="left")


def merge_open_nlp_features(open_tickets: pd.DataFrame) -> pd.DataFrame:
    open_tickets = open_tickets.copy()
    if NLP_SUMMARY_PATH.exists():
        nlp_summary = pd.read_csv(NLP_SUMMARY_PATH)
        nlp_summary = nlp_summary.rename(
            columns={
                "ticket_id": "ticket_id",
                "top_similarity_score": "nlp_top_similarity_score",
                "avg_similarity_score_top5": "nlp_avg_similarity_score_top5",
                "estimated_resolution_hours_nlp": "nlp_estimated_resolution_hours",
            }
        )
        keep = [
            "ticket_id",
            "nlp_top_similarity_score",
            "nlp_avg_similarity_score_top5",
            "nlp_estimated_resolution_hours",
            "suggested_technician_from_similarity",
            "top_match_ticket_id",
            "top_match_title",
        ]
        open_tickets = open_tickets.merge(nlp_summary[keep], on="ticket_id", how="left")
    else:
        open_tickets["nlp_top_similarity_score"] = np.nan
        open_tickets["nlp_avg_similarity_score_top5"] = np.nan
        open_tickets["nlp_estimated_resolution_hours"] = np.nan

    return open_tickets


def prepare_datasets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = df[df["resolution_hours"].notna()].copy()
    open_tickets = df[df["is_active_ticket"]].copy()

    completed = compute_completed_nlp_features(completed)
    open_tickets = merge_open_nlp_features(open_tickets)

    return completed, open_tickets


def get_feature_columns() -> tuple[list[str], list[str], list[str]]:
    numeric_features = [
        "estimated_hours_clean",
        "first_response_minutes",
        "hours_until_due",
        "title_word_count",
        "description_word_count",
        "ticket_text_char_count",
        "created_hour",
        "created_day_of_week",
        "created_month",
        "priority_weight",
        "sla_initial_response_hours",
        "sla_status_update_hours",
        "sla_weight",
        "nlp_top_similarity_score",
        "nlp_avg_similarity_score_top5",
        "nlp_estimated_resolution_hours",
    ]

    categorical_features = [
        "priority",
        "sla_priority_class",
        "issue_type",
        "queue",
        "queue_group",
        "issue_type_group",
        "created_part_of_day",
        "sla_coverage_type",
    ]

    boolean_features = [
        "created_is_weekend",
        "created_is_business_hours",
        "ticket_text_has_server",
        "ticket_text_has_backup",
        "ticket_text_has_vpn",
        "ticket_text_has_email",
        "ticket_text_has_printer",
        "ticket_text_has_access_issue",
        "ticket_text_has_urgent_language",
        "is_priority_low",
        "is_priority_medium",
        "is_priority_high",
        "is_priority_critical",
        "is_sla_high_urgency",
        "is_service_request",
        "is_maintenance",
        "first_response_sla_met",
        "is_legacy",
    ]

    return numeric_features, categorical_features, boolean_features


def build_model_pipeline(numeric_features: list[str], categorical_features: list[str], boolean_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            ("bool", "passthrough", boolean_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_baseline_lookup(df: pd.DataFrame) -> dict:
    return {
        "issue_type_median": df.groupby("issue_type")["resolution_hours"].median().to_dict(),
        "priority_median": df.groupby("priority")["resolution_hours"].median().to_dict(),
        "sla_median": df.groupby("sla_priority_class")["resolution_hours"].median().to_dict(),
        "overall_median": float(df["resolution_hours"].median()),
    }


def apply_baseline_estimator(df: pd.DataFrame, lookup: dict) -> pd.Series:
    baseline = df["issue_type"].map(lookup["issue_type_median"])
    baseline = baseline.fillna(df["priority"].map(lookup["priority_median"]))
    baseline = baseline.fillna(df["sla_priority_class"].map(lookup["sla_median"]))
    baseline = baseline.fillna(lookup["overall_median"])
    return baseline.round(2)


def evaluate_and_predict(completed: pd.DataFrame, open_tickets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    numeric_features, categorical_features, boolean_features = get_feature_columns()
    feature_columns = numeric_features + categorical_features + boolean_features

    X = completed[feature_columns].copy()
    y = completed["resolution_hours"].copy()
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log, y_train_actual, y_test_actual = train_test_split(
        X, y_log, y, test_size=0.2, random_state=RANDOM_STATE
    )
    completed_train_rows = completed.loc[X_train.index].copy()
    completed_test_rows = completed.loc[X_test.index].copy()
    baseline_lookup_test = build_baseline_lookup(completed_train_rows)

    pipeline = build_model_pipeline(numeric_features, categorical_features, boolean_features)
    pipeline.fit(X_train, y_train_log)

    test_pred_log = pipeline.predict(X_test)
    test_pred = np.expm1(test_pred_log)
    test_pred = np.clip(test_pred, 0, None)
    test_nlp_estimate = completed_test_rows["nlp_estimated_resolution_hours"]
    test_baseline_pred = apply_baseline_estimator(completed_test_rows, baseline_lookup_test)
    test_final_pred = (
        0.8 * test_baseline_pred.to_numpy()
        + 0.1 * test_pred
        + 0.1 * test_nlp_estimate.fillna(test_baseline_pred).to_numpy()
    )
    test_hybrid_pred = np.where(
        test_nlp_estimate.notna(),
        0.6 * test_pred + 0.4 * test_nlp_estimate.to_numpy(),
        test_pred,
    )

    test_results = completed_test_rows[["ticket_id", "title", "priority", "sla_priority_class", "issue_type"]].copy()
    test_results["actual_resolution_hours"] = y_test_actual.round(2)
    test_results["predicted_resolution_hours_baseline"] = test_baseline_pred
    test_results["predicted_resolution_hours_model"] = np.round(test_pred, 2)
    test_results["nlp_estimated_resolution_hours"] = np.round(test_nlp_estimate, 2)
    test_results["predicted_resolution_hours_hybrid"] = np.round(test_hybrid_pred, 2)
    test_results["predicted_resolution_hours_final"] = np.round(test_final_pred, 2)
    test_results["absolute_error_hours"] = np.round(
        np.abs(test_results["actual_resolution_hours"] - test_results["predicted_resolution_hours_final"]), 2
    )

    pipeline.fit(X, y_log)
    baseline_lookup_full = build_baseline_lookup(completed)

    open_X = open_tickets[feature_columns].copy()
    open_pred_log = pipeline.predict(open_X)
    open_pred = np.expm1(open_pred_log)
    open_pred = np.clip(open_pred, 0, None)

    open_results = open_tickets[
        ["ticket_id", "title", "priority", "sla_priority_class", "issue_type", "queue", "nlp_estimated_resolution_hours"]
    ].copy()
    open_results["predicted_resolution_hours_baseline"] = apply_baseline_estimator(open_tickets, baseline_lookup_full)
    open_results["predicted_resolution_hours_model"] = np.round(open_pred, 2)
    open_results["predicted_resolution_hours_final"] = np.round(
        0.8 * open_results["predicted_resolution_hours_baseline"]
        + 0.1 * open_results["predicted_resolution_hours_model"]
        + 0.1 * open_results["nlp_estimated_resolution_hours"].fillna(open_results["predicted_resolution_hours_baseline"]),
        2,
    )

    metrics = {
        "training_rows": int(len(completed)),
        "test_rows": int(len(test_results)),
        "open_ticket_rows": int(len(open_results)),
        "baseline_mae_hours": round(float(mean_absolute_error(y_test_actual, test_baseline_pred)), 2),
        "baseline_rmse_hours": round(float(np.sqrt(mean_squared_error(y_test_actual, test_baseline_pred))), 2),
        "baseline_r2_score": round(float(r2_score(y_test_actual, test_baseline_pred)), 4),
        "mae_hours": round(float(mean_absolute_error(y_test_actual, test_pred)), 2),
        "rmse_hours": round(float(np.sqrt(mean_squared_error(y_test_actual, test_pred))), 2),
        "r2_score": round(float(r2_score(y_test_actual, test_pred)), 4),
        "hybrid_mae_hours": round(float(mean_absolute_error(y_test_actual, test_final_pred)), 2),
        "hybrid_rmse_hours": round(float(np.sqrt(mean_squared_error(y_test_actual, test_final_pred))), 2),
        "hybrid_r2_score": round(float(r2_score(y_test_actual, test_final_pred)), 4),
        "model_plus_nlp_mae_hours": round(float(mean_absolute_error(y_test_actual, test_hybrid_pred)), 2),
        "median_absolute_error_hours": round(
            float(np.median(np.abs(y_test_actual.to_numpy() - test_final_pred))), 2
        ),
        "average_predicted_open_resolution_hours": round(
            float(open_results["predicted_resolution_hours_final"].mean()), 2
        ),
        "model_type": "Safe hybrid estimator: issue-type baseline + RandomForest + NLP similarity estimate",
    }

    return test_results.sort_values("absolute_error_hours"), open_results, metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_feature_data()
    completed, open_tickets = prepare_datasets(df)
    test_results, open_results, metrics = evaluate_and_predict(completed, open_tickets)

    test_results.to_csv(TEST_PREDICTIONS_PATH, index=False)
    open_results.to_csv(OPEN_PREDICTIONS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Test predictions saved to: {TEST_PREDICTIONS_PATH}")
    print(f"Open ticket predictions saved to: {OPEN_PREDICTIONS_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Baseline MAE (hours): {metrics['baseline_mae_hours']}")
    print(f"Final Hybrid MAE (hours): {metrics['hybrid_mae_hours']}")
    print(f"Final Hybrid RMSE (hours): {metrics['hybrid_rmse_hours']}")
    print(f"Final Hybrid R2: {metrics['hybrid_r2_score']}")


if __name__ == "__main__":
    main()
