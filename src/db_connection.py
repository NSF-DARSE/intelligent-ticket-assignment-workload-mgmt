from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from load_outputs_to_postgres import get_db_url


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    engine = create_engine(get_db_url())

    try:
        with engine.connect() as conn:
            print("Database connected successfully.")

            table_name = "autotask_raw"
            row_count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()
            print(f"Total rows in {table_name}: {row_count}")

            sample_df = pd.read_sql_query(
                text(f"SELECT * FROM {table_name} LIMIT 5"),
                conn,
            )
            print("\nSample rows:")
            if sample_df.empty:
                print("No rows found.")
            else:
                print(sample_df.to_string(index=False))
    except Exception as error:
        print("Connection failed:", error)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
