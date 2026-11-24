import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


def get_connection() -> psycopg.Connection:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Check your .env file.")
    conn = psycopg.connect(url)
    # Teaches psycopg the `vector` type: send pgvector.Vector / numpy arrays, read back Vector.
    register_vector(conn)
    return conn
