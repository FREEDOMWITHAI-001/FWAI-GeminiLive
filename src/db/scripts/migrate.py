"""
One-time migration script: create PostgreSQL schema.
Usage: python -m src.db.scripts.migrate
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


def main():
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        print("ERROR: DATABASE_URL not set in environment or .env")
        sys.exit(1)

    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    schema_path = Path(__file__).parent / "init_schema.sql"
    schema_sql = schema_path.read_text()

    print(f"Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(schema_sql)
        cur.close()
        conn.close()
        print("Schema created successfully. All 5 tables + indexes ready.")
    except Exception as e:
        print(f"ERROR: Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
