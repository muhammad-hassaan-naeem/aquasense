"""
phase1/pipeline.py
-------------------
Phase 1 End-to-End Pipeline

Orchestrates:
    1. Real ARGO data download / NOAA climatology loading
    2. Simulation-vs-real validation comparison
    3. LSTM model training and RF vs LSTM benchmarking
    4. Phase 1 extended dashboard generation
    5. All results saved for conference submission

CLI Usage
---------
    python -m aquasense.phase1.pipeline
    python -m aquasense.phase1.pipeline --n-floats 30 --lstm-epochs 60
    python -m aquasense.phase1.pipeline --no-real-data  # use synthetic only
    python -m aquasense.phase1.pipeline --quiet
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from ..config import FIGURES_DIR, METRICS_DIR, OUTPUT_DIR, SIM_N_NODES, SIM_N_TIMESTEPS
from ..simulate import simulate_sensor_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aquasense.phase1")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aquasense-phase1",
        description=(
            "AquaSense Phase 1 — Real Data + LSTM RUL Comparison\n"
            "Integrates ARGO ocean data and trains LSTM for comparison with RF."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--nodes",        type=int,  default=SIM_N_NODES,     metavar="N")
    p.add_argument("--timesteps",    type=int,  default=SIM_N_TIMESTEPS, metavar="T")
    p.add_argument("--seed",         type=int,  default=42,              metavar="S")
    p.add_argument("--n-floats",     type=int,  default=20,
                   help="Number of ARGO floats to fetch (default: 20)")
    p.add_argument("--region",       type=str,  default="arabian_sea",
                   help="Ocean region: arabian_sea, indian_ocean, global, etc.")
    p.add_argument("--lstm-epochs",  type=int,  default=40,
                   help="LSTM max training epochs (default: 40)")
    p.add_argument("--lstm-seq-len", type=int,  default=10,
                   help="LSTM sliding window length (default: 10)")
    p.add_argument("--no-real-data", action="store_true",
                   help="Skip ARGO data — use synthetic only")
    p.add_argument("--no-dashboard", action="store_true",
                   help="Skip dashboard rendering")
    p.add_argument("--quiet", "-q",  action="store_true")
    return p


def run_phase1(args: argparse.Namespace = None) -> dict:
    """
    Run the full Phase 1 pipeline.

    Parameters
    ----------
    args : argparse.Namespace, optional
        CLI arguments. If None, uses defaults.

    Returns
    -------
    dict with keys: sim_df, argo_df, comp_results, lstm, rf
    """
    if args is None:
        args = _build_parser().parse_args([])

    if args.quiet:
        logging.disable(logging.WARNING)

    wall = time.perf_counter()
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   AQUASENSE  PHASE 1  ·  Real Data + LSTM  ·  Starting  ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    results = {}

    # ── Step 1: Simulate synthetic data ─────────────────────────────────
    log.info("Step 1/5  ·  Simulating sensor data …")
    t0     = time.perf_counter()
    sim_df = simulate_sensor_data(
        n_nodes=args.nodes,
        n_timesteps=args.timesteps,
        random_seed=args.seed,
    )
    log.info("          %s rows × %s cols  (%.2fs)",
             f"{len(sim_df):,}", len(sim_df.columns),
             time.perf_counter() - t0)
    results["sim_df"] = sim_df

    # ── Step 2: Real ARGO / NOAA data ───────────────────────────────────
    argo_df = pd.DataFrame()
    if not args.no_real_data:
        log.info("Step 2/5  ·  Fetching real ARGO ocean data …")
        t0 = time.perf_counter()
        try:
            from .data.argo_connector import ArgoConnector
            conn    = ArgoConnector()
            argo_df = conn.load_or_fetch(
                n_floats=args.n_floats,
                region=args.region,
            )
            conn.validate_schema(argo_df)
            log.info("          ARGO: %s profiles  region=%s  (%.2fs)",
                     f"{len(argo_df):,}", args.region,
                     time.perf_counter() - t0)

            # ARGO vs simulation comparison
            comparison_df = conn.compare_with_simulation(argo_df, sim_df)
            log.info("          Simulation vs ARGO comparison:")
            for _, row in comparison_df.iterrows():
                log.info("            %-20s  ARGO=%-8.3f  Sim=%-8.3f  diff=%.1f%%",
                         row["feature"], row["argo_mean"],
                         row["sim_mean"], row["diff_mean_pct"])
            comparison_df.to_csv(
                METRICS_DIR / "argo_vs_sim_comparison.csv", index=False)
        except Exception as exc:
            log.warning("ARGO data unavailable (%s) — continuing with synthetic only", exc)
            argo_df = pd.DataFrame()

        # NOAA basin statistics
        try:
            from .data.noaa_connector import NOAAConnector
            noaa = NOAAConnector()
            noaa_stats = noaa.basin_statistics()
            METRICS_DIR.mkdir(parents=True, exist_ok=True)
            noaa_stats.to_csv(METRICS_DIR / "noaa_basin_stats.csv", index=False)
            log.info("          NOAA basin stats saved → results/metrics/noaa_basin_stats.csv")
            log.info("\n%s", noaa.pakistan_ocean_summary())
        except Exception as exc:
            log.warning("NOAA connector error: %s", exc)
    else:
        log.info("Step 2/5  ·  Skipping real data (--no-real-data)")

    results["argo_df"] = argo_df

    # ── Step 3: RF vs LSTM model comparison ─────────────────────────────
    log.info("Step 3/5  ·  Running RF vs LSTM comparison …")
    t0 = time.perf_counter()
    try:
        from .models.model_comparison import ModelComparison
        comp = ModelComparison(
            lstm_seq_len=args.lstm_seq_len,
            lstm_epochs=args.lstm_epochs,
        )
        comp_results = comp.run(
            df_synthetic=sim_df,
            df_real=argo_df if len(argo_df) > 0 else None,
        )
        comp.print_report(comp_results)
        results["comp_results"] = comp_results
        results["rf"]           = comp.rf
        results["lstm"]         = comp.lstm
        log.info("          Comparison complete  (%.2fs)",
                 time.perf_counter() - t0)
    except Exception as exc:
        log.error("Model comparison failed: %s", exc)
        import traceback; traceback.print_exc()
        results["comp_results"] = pd.DataFrame()
        results["rf"]   = None
        results["lstm"] = None

    # ── Step 4: Generate comparison figure ──────────────────────────────
    if results.get("rf") and results.get("lstm"):
        log.info("Step 4/5  ·  Generating comparison figures …")
        t0 = time.perf_counter()
        try:
            comp.plot_comparison()
            log.info("          Saved → results/figures/model_comparison_rf_vs_lstm.png")

            # ARGO validation figure
            if len(argo_df) > 0:
                from .viz.phase1_dashboard import build_argo_validation_figure
                build_argo_validation_figure(argo_df, sim_df)
                log.info("          Saved → results/figures/argo_validation.png")

            log.info("          (%.2fs)", time.perf_counter() - t0)
        except Exception as exc:
            log.warning("Figure generation error: %s", exc)
    else:
        log.info("Step 4/5  ·  Skipping figures (no trained models)")

    # ── Step 5: Phase 1 dashboard ────────────────────────────────────────
    if not args.no_dashboard and results.get("lstm") and results.get("comp_results") is not None:
        log.info("Step 5/5  ·  Rendering Phase 1 dashboard …")
        t0 = time.perf_counter()
        try:
            from .viz.phase1_dashboard import build_phase1_dashboard
            path = build_phase1_dashboard(
                sim_df       = sim_df,
                argo_df      = argo_df if len(argo_df) > 0 else sim_df,
                comp_results = results["comp_results"],
                lstm_model   = results["lstm"],
            )
            log.info("          Dashboard saved → %s  (%.2fs)",
                     path, time.perf_counter() - t0)
        except Exception as exc:
            log.warning("Dashboard error: %s", exc)
            import traceback; traceback.print_exc()
    else:
        log.info("Step 5/5  ·  Skipping dashboard")

    log.info("✓  Phase 1 complete in %.2fs", time.perf_counter() - wall)
    print()

    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    """Print a clean summary of all Phase 1 outputs."""
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │              PHASE 1 OUTPUTS SUMMARY                │")
    print("  ├─────────────────────────────────────────────────────┤")

    sim_df = results.get("sim_df", pd.DataFrame())
    argo_df = results.get("argo_df", pd.DataFrame())
    print(f"  │  Synthetic data    : {len(sim_df):>6,} rows                    │")
    print(f"  │  ARGO real data    : {len(argo_df):>6,} rows                    │")

    comp = results.get("comp_results", pd.DataFrame())
    if len(comp) > 0:
        for _, row in comp[comp["dataset"] == "synthetic"].iterrows():
            print(f"  │  {row['model']:<18}: MAE={row['mae']:>7.2f}h  R²={row['r2']:.4f}  │")

    print("  ├─────────────────────────────────────────────────────┤")
    print("  │  Saved files:                                        │")
    print("  │  results/metrics/model_comparison.csv               │")
    print("  │  results/metrics/argo_vs_sim_comparison.csv         │")
    print("  │  results/metrics/noaa_basin_stats.csv               │")
    print("  │  results/figures/model_comparison_rf_vs_lstm.png    │")
    print("  │  results/figures/argo_validation.png                │")
    print("  │  outputs/phase1_dashboard.png                       │")
    print("  └─────────────────────────────────────────────────────┘")
    print()


def main() -> None:
    args = _build_parser().parse_args()
    try:
        run_phase1(args)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        log.exception("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
