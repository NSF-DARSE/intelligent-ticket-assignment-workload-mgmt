from __future__ import annotations

"""Lightweight database connectivity check for the configured PostgreSQL target."""

import pandas as pd
from sqlalchemy import create_engine, text

from load_outputs_to_postgres import get_db_url


RAW_TABLE_NAME = "autotask_raw"


def main() -> None:
    try:
        engine = create_engine(get_db_url())
        with engine.connect() as conn:
            print("Database connected successfully.")
            connection_info = conn.execute(text("SELECT current_database(), current_user")).fetchone()
            if connection_info:
                print(f"Connected database: {connection_info[0]}")
                print(f"Connected user: {connection_info[1]}")

            row_count = conn.execute(text(f"SELECT COUNT(*) FROM {RAW_TABLE_NAME}")).scalar_one()
            print(f"Total rows in {RAW_TABLE_NAME}: {row_count}")

            sample_df = pd.read_sql_query(
                text(f"SELECT * FROM {RAW_TABLE_NAME} LIMIT 5"),
                conn,
            )
            print("\nSample rows:")
            if sample_df.empty:
                print("No rows found.")
            else:
                print(sample_df.to_string(index=False))
    except Exception as error:
        print("Connection failed:", error)


if __name__ == "__main__":
    main()
