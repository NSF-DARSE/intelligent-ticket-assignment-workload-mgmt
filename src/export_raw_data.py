from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from load_outputs_to_postgres import get_db_url


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "Raw_Data" / "autotask_raw_data.csv"


def main() -> None:
    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(get_db_url())

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text("SELECT * FROM autotask_raw"), conn)
        df.to_csv(RAW_OUTPUT_PATH, index=False)
        print("Raw data exported successfully.")
        print(f"Rows exported: {len(df)}")
        print(f"File saved at: {RAW_OUTPUT_PATH}")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
