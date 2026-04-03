"""
database.py
-----------
Persistence layer for AquaSense sensor logs.

Defaults to SQLite (zero-configuration, portable) but can transparently
switch to PostgreSQL by supplying a connection URL via the
``AQUASENSE_DB_URL`` environment variable:

    export AQUASENSE_DB_URL="postgresql://user:password@host:5432/aquasense"

When using PostgreSQL, ``psycopg2`` must be installed:

    pip install psycopg2-binary

All public functions accept a *connection-factory* callable so tests can
easily inject an in-memory SQLite connection without touching the file system.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import DB_PATH


# ── Connection helpers ─────────────────────────────────────────────────────

def _is_postgres() -> bool:
    """Return True if a PostgreSQL URL has been configured."""
    return bool(os.getenv("AQUASENSE_DB_URL", "").startswith("postgresql"))


def get_connection():
    """
    Return a live DB connection.

    * SQLite  → used when ``AQUASENSE_DB_URL`` is not set or doesn't start
                with ``postgresql``.
    * PostgreSQL → used when ``AQUASENSE_DB_URL`` is a valid psycopg2 DSN.

    Callers are responsible for closing the connection.
    """
    if _is_postgres():
        try:
            import psycopg2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "psycopg2 is required for PostgreSQL support.  "
                "Install it with:  pip install psycopg2-binary"
            ) from exc
        return psycopg2.connect(os.environ["AQUASENSE_DB_URL"])

    # SQLite fallback
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DB_PATH))


# ── Schema ─────────────────────────────────────────────────────────────────

_DDL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS sensor_logs (
    id                SERIAL PRIMARY KEY,   -- INTEGER AUTOINCREMENT in SQLite
    node_id           INTEGER      NOT NULL,
    timestep          INTEGER      NOT NULL,
    depth_m           REAL         NOT NULL,
    pressure_bar      REAL         NOT NULL,
    salinity_ppt      REAL         NOT NULL,
    temperature_c     REAL         NOT NULL,
    battery_voltage   REAL         NOT NULL,
    tx_freq_ppm       REAL         NOT NULL,
    packet_success_rt REAL         NOT NULL,
    depth_cluster     TEXT         NOT NULL,
    is_anomaly        INTEGER      NOT NULL DEFAULT 0,
    rul_hours         REAL         NOT NULL,
    inserted_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);
"""

# SQLite does not support SERIAL; use INTEGER PRIMARY KEY instead
_DDL_CREATE_TABLE_SQLITE = _DDL_CREATE_TABLE.replace(
    "SERIAL PRIMARY KEY",
    "INTEGER PRIMARY KEY AUTOINCREMENT",
)

_DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_sl_node    ON sensor_logs (node_id);",
    "CREATE INDEX IF NOT EXISTS idx_sl_ts      ON sensor_logs (timestep);",
    "CREATE INDEX IF NOT EXISTS idx_sl_cluster ON sensor_logs (depth_cluster);",
    "CREATE INDEX IF NOT EXISTS idx_sl_anomaly ON sensor_logs (is_anomaly);",
]


def init_schema(conn) -> None:
    """
    Create the ``sensor_logs`` table and supporting indexes if they do not
    already exist.  Safe to call on an already-initialised database.
    """
    cur = conn.cursor()
    ddl = _DDL_CREATE_TABLE if _is_postgres() else _DDL_CREATE_TABLE_SQLITE
    cur.execute(ddl)
    for idx_sql in _DDL_INDEXES:
        cur.execute(idx_sql)
    conn.commit()


# ── Write ──────────────────────────────────────────────────────────────────

def write_logs(df: pd.DataFrame, conn, *, replace: bool = True) -> int:
    """
    Persist a DataFrame of sensor readings to ``sensor_logs``.

    Parameters
    ----------
    df : pd.DataFrame
        Rows to insert (must match the table columns).
    conn :
        Open database connection.
    replace : bool
        If *True*, drop and recreate the table first (full refresh).
        If *False*, append to existing data.

    Returns
    -------
    int
        Number of rows written.
    """
    cols = [
        "node_id", "timestep", "depth_m", "pressure_bar",
        "salinity_ppt", "temperature_c", "battery_voltage",
        "tx_freq_ppm", "packet_success_rt", "depth_cluster",
        "is_anomaly", "rul_hours",
    ]
    subset = df[cols].copy()

    if replace:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS sensor_logs")
        conn.commit()
        init_schema(conn)

    # pandas to_sql works for both SQLite and PostgreSQL connections
    subset.to_sql(
        "sensor_logs",
        conn,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )
    conn.commit()
    return len(subset)


# ── Optimised read queries ─────────────────────────────────────────────────

def query_latest_per_node(conn) -> pd.DataFrame:
    """
    Return the most recent telemetry row for every node.

    Uses a correlated sub-query with an index on ``(node_id, timestep)``
    so the full table scan is avoided even at millions of rows.
    """
    sql = """
        SELECT s.*
        FROM   sensor_logs s
        INNER JOIN (
            SELECT   node_id, MAX(timestep) AS max_ts
            FROM     sensor_logs
            GROUP BY node_id
        ) latest
            ON  s.node_id  = latest.node_id
            AND s.timestep = latest.max_ts
        ORDER BY s.node_id;
    """
    return pd.read_sql_query(sql, conn)


def query_cluster_stats(conn) -> pd.DataFrame:
    """
    Aggregate energy-efficiency metrics per depth cluster.

    Returns
    -------
    pd.DataFrame with columns:
        depth_cluster, n_nodes, avg_battery, avg_psr,
        avg_rul, avg_tx_freq, total_anomalies
    """
    sql = """
        SELECT
            depth_cluster,
            COUNT(DISTINCT node_id)        AS n_nodes,
            ROUND(AVG(battery_voltage), 4) AS avg_battery,
            ROUND(AVG(packet_success_rt),4)AS avg_psr,
            ROUND(AVG(rul_hours),        2) AS avg_rul,
            ROUND(AVG(tx_freq_ppm),      3) AS avg_tx_freq,
            SUM(is_anomaly)                AS total_anomalies
        FROM   sensor_logs
        GROUP  BY depth_cluster
        ORDER  BY avg_rul DESC;
    """
    return pd.read_sql_query(sql, conn)


def query_critical_nodes(conn, rul_threshold: float = 50.0) -> pd.DataFrame:
    """
    Return the latest reading for nodes whose predicted RUL is below
    *rul_threshold* hours.  Useful for alerting dashboards.
    """
    latest_sql = """
        SELECT s.*
        FROM   sensor_logs s
        INNER JOIN (
            SELECT   node_id, MAX(timestep) AS max_ts
            FROM     sensor_logs
            GROUP BY node_id
        ) latest
            ON  s.node_id  = latest.node_id
            AND s.timestep = latest.max_ts
        WHERE  s.rul_hours < ?
        ORDER  BY s.rul_hours ASC;
    """
    # PostgreSQL uses %s placeholders; SQLite uses ?
    if _is_postgres():
        latest_sql = latest_sql.replace("?", "%s")

    return pd.read_sql_query(latest_sql, conn, params=(rul_threshold,))


def query_anomaly_timeline(conn) -> pd.DataFrame:
    """Hourly anomaly count across all nodes (for trend-line charts)."""
    sql = """
        SELECT
            timestep,
            SUM(is_anomaly)      AS true_anomalies,
            COUNT(*)             AS total_readings
        FROM   sensor_logs
        GROUP  BY timestep
        ORDER  BY timestep;
    """
    return pd.read_sql_query(sql, conn)
