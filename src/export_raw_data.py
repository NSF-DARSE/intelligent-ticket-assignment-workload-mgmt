from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from load_outputs_to_postgres import get_db_url


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "Raw_Data" / "autotask_raw_data.csv"
RAW_TABLE_NAME = "autotask_raw"


def main() -> None:
    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        engine = create_engine(get_db_url())
        with engine.connect() as conn:
            df = pd.read_sql_query(text(f"SELECT * FROM {RAW_TABLE_NAME}"), conn)
    except Exception as error:
        print("Raw data export failed:", error)
        return

    df.to_csv(RAW_OUTPUT_PATH, index=False)
    print("Raw data exported successfully.")
    print(f"Rows exported: {len(df)}")
    print(f"File saved at: {RAW_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
