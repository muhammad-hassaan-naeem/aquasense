"""
pipeline.py  –  End-to-end AquaSense pipeline + CLI.

Usage
-----
    python -m aquasense.pipeline                        # defaults
    python -m aquasense.pipeline --nodes 100 --bench    # with benchmarks
    python -m aquasense.pipeline --no-dashboard --quiet # CI mode
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import OUTPUT_DIR, SIM_N_NODES, SIM_N_TIMESTEPS, SIM_RANDOM_SEED
from .database import (
    get_connection, init_schema, query_cluster_stats,
    query_critical_nodes, query_latest_per_node, write_logs,
)
from .models import AnomalyDetector, DepthClusterer, RULRegressor
from .simulate import simulate_sensor_data
from .visualise import build_dashboard

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("aquasense")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aquasense",
        description="AquaSense — Underwater Sensor Node Monitoring & IoUT Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--nodes",        type=int,   default=SIM_N_NODES,      metavar="N")
    p.add_argument("--timesteps",    type=int,   default=SIM_N_TIMESTEPS,  metavar="T")
    p.add_argument("--seed",         type=int,   default=SIM_RANDOM_SEED,  metavar="S")
    p.add_argument("--rul-alert",    type=float, default=50.0,             metavar="H")
    p.add_argument("--output",       type=Path,  default=None,             metavar="PATH")
    p.add_argument("--bench",        action="store_true",
                   help="Run full benchmark suite (routing protocol comparison)")
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--quiet", "-q",  action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.quiet:
        logging.disable(logging.WARNING)

    wall = time.perf_counter()
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║     AQUASENSE  v2.0  ·  IoUT Research Framework     ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

    # 1. Simulate
    log.info("Step 1/6  · Simulating sensor data …")
    t0 = time.perf_counter()
    df = simulate_sensor_data(n_nodes=args.nodes,
                              n_timesteps=args.timesteps,
                              random_seed=args.seed)
    log.info("          Dataset: %s rows × %s cols  (%.2fs)",
             f"{len(df):,}", len(df.columns), time.perf_counter()-t0)

    # 2. Database
    log.info("Step 2/6  · Persisting logs …")
    t0   = time.perf_counter()
    conn = get_connection()
    init_schema(conn)
    n    = write_logs(df, conn, replace=True)
    log.info("          %s rows stored  (%.2fs)", f"{n:,}", time.perf_counter()-t0)
    latest_df     = query_latest_per_node(conn)
    cluster_stats = query_cluster_stats(conn)
    critical      = query_critical_nodes(conn, rul_threshold=args.rul_alert)
    conn.close()
    for _, row in cluster_stats.iterrows():
        log.info("          %-8s  nodes=%-3d  avg_RUL=%-7.1fh  battery=%.2fV",
                 row["depth_cluster"].upper(), int(row["n_nodes"]),
                 float(row["avg_rul"]), float(row["avg_battery"]))
    if not critical.empty:
        log.warning("⚠  %d node(s) below %.0f-hour RUL threshold!",
                    len(critical), args.rul_alert)

    # 3. RUL Regressor
    log.info("Step 3/6  · Training RUL Regressor …")
    t0  = time.perf_counter()
    reg = RULRegressor()
    reg.fit(df)
    log.info("          %s  (%.2fs)", reg, time.perf_counter()-t0)

    # 4. Anomaly Detector
    log.info("Step 4/6  · Training Anomaly Detector …")
    t0  = time.perf_counter()
    det = AnomalyDetector()
    det.fit(df)
    anomaly_df = det.tag_dataframe(df)
    log.info("          %s  (%.2fs)", det, time.perf_counter()-t0)

    # 5. Clustering
    log.info("Step 5/6  · Running K-Means depth clustering …")
    t0  = time.perf_counter()
    clr = DepthClusterer()
    clr.fit(df)
    km_df = clr.tag_dataframe(df)
    log.info("          (%.2fs)", time.perf_counter()-t0)

    # 6. Dashboard
    if not args.no_dashboard:
        log.info("Step 6/6  · Rendering dashboard …")
        t0   = time.perf_counter()
        path = build_dashboard(
            df=df, latest_df=latest_df, cluster_stats=cluster_stats,
            y_test=reg._y_test, y_pred=reg._y_pred,
            anomaly_df=anomaly_df, km_df=km_df,
            feature_importances=reg.feature_importances_,
            output_path=args.output,
        )
        log.info("          Saved → %s  (%.2fs)", path, time.perf_counter()-t0)

    # 7. Benchmarks (optional)
    if args.bench:
        log.info("Step 7/7  · Running routing protocol benchmarks …")
        t0 = time.perf_counter()
        from .research.benchmarks import run_full_benchmark_suite
        summary = run_full_benchmark_suite(df)
        log.info("Benchmark results:\n%s", summary.to_string())
        log.info("          (%.2fs)", time.perf_counter()-t0)

    log.info("✓  Pipeline complete in %.2fs", time.perf_counter()-wall)
    print()


if __name__ == "__main__":
    main()
