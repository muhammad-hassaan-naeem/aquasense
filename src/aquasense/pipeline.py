"""
pipeline.py
-----------
End-to-end AquaSense pipeline with an argparse CLI.

Usage
-----
    # Run with defaults (80 nodes, 100 timesteps, SQLite)
    python -m aquasense.pipeline

    # Customise
    python -m aquasense.pipeline --nodes 200 --timesteps 150 --seed 7

    # Point at a PostgreSQL database
    export AQUASENSE_DB_URL="postgresql://user:pass@localhost/aquasense"
    python -m aquasense.pipeline

    # Save dashboard to a specific path
    python -m aquasense.pipeline --output my_dashboard.png
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import (
    OUTPUT_DIR,
    SIM_N_NODES,
    SIM_N_TIMESTEPS,
    SIM_RANDOM_SEED,
)
from .database import (
    get_connection,
    init_schema,
    query_cluster_stats,
    query_critical_nodes,
    query_latest_per_node,
    write_logs,
)
from .models import AnomalyDetector, DepthClusterer, RULRegressor
from .simulate import simulate_sensor_data
from .visualise import build_dashboard

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aquasense")


# ── CLI ────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aquasense",
        description=(
            "AquaSense — Underwater Sensor Node Monitoring System\n"
            "Simulates sensor data, trains ML models, stores logs in a\n"
            "database, and renders an 8-panel monitoring dashboard."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--nodes", type=int, default=SIM_N_NODES,
        metavar="N",
        help=f"Number of sensor nodes to simulate (default: {SIM_N_NODES})",
    )
    p.add_argument(
        "--timesteps", type=int, default=SIM_N_TIMESTEPS,
        metavar="T",
        help=f"Number of time-steps per node (default: {SIM_N_TIMESTEPS})",
    )
    p.add_argument(
        "--seed", type=int, default=SIM_RANDOM_SEED,
        metavar="S",
        help=f"Random seed for reproducibility (default: {SIM_RANDOM_SEED})",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        metavar="PATH",
        help="Output path for the dashboard PNG (default: outputs/aquasense_dashboard.png)",
    )
    p.add_argument(
        "--rul-alert", type=float, default=50.0,
        metavar="H",
        help="RUL threshold in hours below which nodes are flagged critical (default: 50)",
    )
    p.add_argument(
        "--no-dashboard", action="store_true",
        help="Skip dashboard rendering (useful for CI / headless pipelines)",
    )
    p.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress informational output",
    )
    return p


# ── Pipeline steps ─────────────────────────────────────────────────────────

def step_simulate(args: argparse.Namespace):
    log.info("Step 1/6  ·  Simulating sensor data …")
    t0 = time.perf_counter()
    df = simulate_sensor_data(
        n_nodes=args.nodes,
        n_timesteps=args.timesteps,
        random_seed=args.seed,
    )
    elapsed = time.perf_counter() - t0
    log.info(
        "           Dataset: %s rows × %s cols  (%.2fs)",
        f"{len(df):,}", len(df.columns), elapsed,
    )
    return df


def step_database(df, args: argparse.Namespace):
    log.info("Step 2/6  ·  Writing logs to database …")
    t0   = time.perf_counter()
    conn = get_connection()
    init_schema(conn)
    n    = write_logs(df, conn, replace=True)
    log.info("           %s rows persisted  (%.2fs)", f"{n:,}", time.perf_counter() - t0)

    latest_df     = query_latest_per_node(conn)
    cluster_stats = query_cluster_stats(conn)
    critical      = query_critical_nodes(conn, rul_threshold=args.rul_alert)
    conn.close()

    log.info("           Cluster summary:")
    for _, row in cluster_stats.iterrows():
        log.info(
            "             %-8s  nodes=%-3d  avg_RUL=%-7.1fh  "
            "avg_battery=%.2fV  anomalies=%d",
            row["depth_cluster"].upper(),
            int(row["n_nodes"]),
            float(row["avg_rul"]),
            float(row["avg_battery"]),
            int(row["total_anomalies"]),
        )

    if not critical.empty:
        log.warning(
            "           ⚠  %d node(s) below %.0f-hour RUL threshold!",
            len(critical), args.rul_alert,
        )
        for _, r in critical.iterrows():
            log.warning(
                "              Node %3d  depth=%.0fm  RUL=%.1fh  battery=%.2fV",
                int(r["node_id"]), float(r["depth_m"]),
                float(r["rul_hours"]), float(r["battery_voltage"]),
            )
    else:
        log.info("           All nodes above the %.0f-hour RUL threshold.", args.rul_alert)

    return latest_df, cluster_stats


def step_rul_model(df):
    log.info("Step 3/6  ·  Training RUL Regressor …")
    t0  = time.perf_counter()
    reg = RULRegressor()
    reg.fit(df)
    log.info(
        "           %s  (%.2fs)",
        reg, time.perf_counter() - t0,
    )
    return reg


def step_anomaly_model(df):
    log.info("Step 4/6  ·  Training Anomaly Detector …")
    t0  = time.perf_counter()
    det = AnomalyDetector()
    det.fit(df)
    log.info(
        "           %s  (%.2fs)",
        det, time.perf_counter() - t0,
    )
    anomaly_df = det.tag_dataframe(df)
    return det, anomaly_df


def step_clustering(df):
    log.info("Step 5/6  ·  Running K-Means depth clustering …")
    t0  = time.perf_counter()
    clr = DepthClusterer()
    clr.fit(df)
    km_df = clr.tag_dataframe(df)
    log.info("           K-Means cluster summary:")
    summary = clr.cluster_summary(df)
    for ci, row in summary.iterrows():
        log.info(
            "             Cluster %d  depth=%.0fm  battery=%.2fV  RUL=%.1fh",
            ci, float(row["depth_m"]),
            float(row["battery_voltage"]), float(row["rul_hours"]),
        )
    log.info("           (%.2fs)", time.perf_counter() - t0)
    return clr, km_df


def step_dashboard(
    df, latest_df, cluster_stats,
    reg, anomaly_df, km_df, args: argparse.Namespace,
) -> Path:
    log.info("Step 6/6  ·  Rendering dashboard …")
    t0 = time.perf_counter()
    path = build_dashboard(
        df                  = df,
        latest_df           = latest_df,
        cluster_stats       = cluster_stats,
        y_test              = reg._y_test,
        y_pred              = reg._y_pred,
        anomaly_df          = anomaly_df,
        km_df               = km_df,
        feature_importances = reg.feature_importances_,
        output_path         = args.output,
    )
    log.info("           Dashboard saved → %s  (%.2fs)", path, time.perf_counter() - t0)
    return path


# ── Main ───────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    if args.quiet:
        logging.disable(logging.WARNING)

    wall = time.perf_counter()
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║          AQUASENSE  ·  v1.0  ·  Starting …           ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

    df                        = step_simulate(args)
    latest_df, cluster_stats  = step_database(df, args)
    reg                       = step_rul_model(df)
    det, anomaly_df           = step_anomaly_model(df)
    clr, km_df                = step_clustering(df)

    if not args.no_dashboard:
        step_dashboard(df, latest_df, cluster_stats, reg, anomaly_df, km_df, args)

    print()
    log.info("✓  Pipeline complete in %.2fs", time.perf_counter() - wall)
    print()


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:                           # pragma: no cover
        log.exception("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
