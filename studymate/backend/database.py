"""
database.py  —  the connection to Supabase Postgres.

This file's only job is to hand back a live database connection that already
knows how to handle vectors. Everything else (rag_engine, main) asks this file
for a connection and then runs SQL on it.

Why a separate file? So there is ONE place that knows the connection details.
If the database ever changes, you edit here and nowhere else.
"""

import os
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# Load the variables written in the .env file (so SUPABASE_DB_URL becomes available)
load_dotenv()

# Read the connection string you pasted into .env
DB_URL = os.getenv("SUPABASE_DB_URL")

if not DB_URL:
    raise RuntimeError(
        "SUPABASE_DB_URL is not set. Open your .env file and make sure it has a "
        "line like: SUPABASE_DB_URL=postgresql://postgres.xxxx:password@...supabase.com:5432/postgres"
    )


def get_connection():
    """
    Open a fresh connection to Supabase Postgres and tell it about the `vector`
    type. The register_vector() call is the important bit: without it, psycopg2
    doesn't know how to send a Python list to a VECTOR(768) column and you'd get
    a 'can't adapt type list' error.
    """
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)   # teaches this connection how to handle vectors
    return conn


if __name__ == "__main__":
    # Quick self-test: run `python database.py` to check your connection works.
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM chunks;")
        count = cur.fetchone()[0]
        print(f"Connected to Supabase. The chunks table currently has {count} rows.")
        cur.close()
        conn.close()
    except Exception as e:
        print("Could not connect or query. Error was:")
        print(e)