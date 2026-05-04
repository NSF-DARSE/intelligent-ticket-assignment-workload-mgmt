from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from load_outputs_to_postgres import get_db_url  # noqa: E402


class LocalDatabaseConfigTests(unittest.TestCase):
    def test_get_db_url_accepts_localhost(self) -> None:
        env = {
            "DB_USER": "postgres",
            "DB_PASSWORD": "password",
            "DB_HOST": "localhost",
            "DB_PORT": "5432",
            "DB_NAME": "autotask_local",
            "DB_SSLMODE": "",
        }

        with patch.dict(os.environ, env, clear=False):
            db_url = get_db_url()

        self.assertEqual(db_url.host, "localhost")
        self.assertEqual(db_url.database, "autotask_local")
        self.assertEqual(db_url.drivername, "postgresql+psycopg2")

    def test_get_db_url_rejects_non_local_host(self) -> None:
        env = {
            "DB_USER": "postgres",
            "DB_PASSWORD": "password",
            "DB_HOST": "example.postgres.database.azure.com",
            "DB_PORT": "5432",
            "DB_NAME": "autotask_local",
            "DB_SSLMODE": "require",
        }

        with patch.dict(os.environ, env, clear=False):
            with self.assertRaisesRegex(ValueError, "local PostgreSQL only"):
                get_db_url()


if __name__ == "__main__":
    unittest.main()
