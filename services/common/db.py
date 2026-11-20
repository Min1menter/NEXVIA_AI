"""Postgres helpers shared by the API and the inference worker."""
import contextlib
import os

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(1, int(os.environ.get("DB_POOL_MAX", "10")), os.environ["DATABASE_URL"])
    return _pool


@contextlib.contextmanager
def connection():
    """Yields a connection; commits on success, rolls back on error."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def cursor(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


Json = psycopg2.extras.Json
