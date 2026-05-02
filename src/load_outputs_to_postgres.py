from __future__ import annotations

"""Load generated CSV and JSON outputs into PostgreSQL reporting tables."""

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_ENV_KEYS = ("DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME")

CSV_TABLE_MAP = {
    DATA_DIR / "Raw_Data" / "autotask_raw_data.csv": "autotask_raw",
    DATA_DIR / "Cleaned_Data" / "autotask_cleaned_data.csv": "autotask_cleaned_data",
    DATA_DIR / "Feature_Engineered" / "autotask_feature_engineered.csv": "autotask_feature_engineered",
    DATA_DIR / "Feature_Engineered" / "autotask_training_dataset.csv": "autotask_training_dataset",
    DATA_DIR / "Feature_Engineered" / "autotask_open_tickets_dataset.csv": "autotask_open_tickets_dataset",
    DATA_DIR / "Feature_Engineered" / "technician_profiles.csv": "autotask_technician_profiles",
    DATA_DIR / "Feature_Engineered" / "employee_skills_profile.csv": "autotask_employee_skills_profile",
    DATA_DIR / "Feature_Engineered" / "employee_skills_normalized.csv": "autotask_employee_skills_normalized",
    DATA_DIR / "NLP" / "ticket_similarity_matches.csv": "autotask_ticket_similarity_matches",
    DATA_DIR / "NLP" / "ticket_similarity_summary.csv": "autotask_ticket_similarity_summary",
    DATA_DIR / "Complexity" / "autotask_complexity_scored.csv": "autotask_complexity_scored",
    DATA_DIR / "Recommendations" / "technician_workload_snapshot.csv": "autotask_technician_workload_snapshot",
    DATA_DIR / "Recommendations" / "assignment_recommendations.csv": "autotask_assignment_recommendations",
}

JSON_TABLE_MAP = {
    DATA_DIR / "Cleaned_Data" / "autotask_cleaning_summary.json": "autotask_cleaning_summary",
    DATA_DIR / "Feature_Engineered" / "feature_engineering_summary.json": "autotask_feature_engineering_summary",
    DATA_DIR / "Feature_Engineered" / "employee_skills_summary.json": "autotask_employee_skills_summary",
    DATA_DIR / "NLP" / "nlp_similarity_summary.json": "autotask_nlp_similarity_summary",
    DATA_DIR / "Complexity" / "complexity_scoring_summary.json": "autotask_complexity_scoring_summary",
    DATA_DIR / "Recommendations" / "recommendation_summary.json": "autotask_recommendation_summary",
}

DATETIME_COLUMNS = {
    "created_at",
    "completed_at",
    "due_at",
    "first_response_at",
}


def get_db_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env")

    env_values = {key: os.getenv(key, "").strip() for key in DB_ENV_KEYS}
    missing = [name for name, value in env_values.items() if not value]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing database configuration in .env: {missing_text}")

    sslmode = os.getenv("DB_SSLMODE", "").strip()
    query = {"sslmode": sslmode} if sslmode else None

    return URL.create(
        "postgresql+psycopg2",
        username=env_values["DB_USER"],
        password=env_values["DB_PASSWORD"],
        host=env_values["DB_HOST"],
        port=int(env_values["DB_PORT"]),
        database=env_values["DB_NAME"],
        query=query,
    )


def coerce_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in df.columns:
        if column in DATETIME_COLUMNS or column.endswith("_at"):
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().sum() > 0:
                df[column] = parsed

    return df


def load_csv_tables(engine) -> list[dict]:
    results = []

    for csv_path, table_name in CSV_TABLE_MAP.items():
        df = pd.read_csv(csv_path)
        df = coerce_dataframe_types(df)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        results.append({"table_name": table_name, "rows_loaded": int(len(df)), "source_file": str(csv_path)})

    return results


def load_json_tables(engine) -> list[dict]:
    results = []

    for json_path, table_name in JSON_TABLE_MAP.items():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        df = pd.DataFrame([{"source_file": str(json_path), "payload": json.dumps(payload)}])
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        results.append({"table_name": table_name, "rows_loaded": 1, "source_file": str(json_path)})

        with engine.begin() as conn:
            conn.execute(
                text(
                    f"ALTER TABLE {table_name} "
                    "ALTER COLUMN payload TYPE jsonb USING payload::jsonb"
                )
            )

    return results


def main() -> None:
    engine = create_engine(get_db_url())

    csv_results = load_csv_tables(engine)
    json_results = load_json_tables(engine)

    print("PostgreSQL output load completed.")
    for result in csv_results + json_results:
        print(
            f"{result['table_name']}: {result['rows_loaded']} rows loaded "
            f"from {Path(result['source_file']).name}"
        )


if __name__ == "__main__":
    main()
