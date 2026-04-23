"""
phase1/comparison.py
---------------------
RF vs Temporal (LSTM-equivalent) Model Comparison

Generates publication-quality figures and a metrics table
comparing the Random Forest RUL Regressor against the
windowed Temporal (Gradient Boosting) model.

This is the Phase 1 deliverable for conference submission.

Figures generated
-----------------
1.  rf_vs_temporal_scatter.png      – Predicted vs Actual for both models
2.  rf_vs_temporal_bar.png          – MAE / RMSE / R² side-by-side bars
3.  rul_trend_node.png              – RUL trend over time for one node
4.  argo_vs_synthetic_dist.png      – Feature distributions comparison
5.  battery_degradation_clusters.png– Battery decay rate per depth cluster
6.  error_distribution.png          – Prediction error histogram for both
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from ..config import FIGURES_DIR, METRICS_DIR, PALETTE
from ..models import RULRegressor
from .lstm_model import TemporalRULModel, evaluate_model

log = logging.getLogger("aquasense.phase1.comparison")

# ── Plot style ─────────────────────────────────────────────────────────────
RF_COLOR   = "#00C8FF"    # cyan  — Random Forest
LSTM_COLOR = "#FF6B35"    # orange — Temporal / LSTM
ARGO_COLOR = "#00FFD5"    # teal  — real ARGO data
SYN_COLOR  = "#7B5EA7"    # purple — synthetic data

def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PALETTE["panel"])
    for sp in ax.spines.values():
        sp.set_color(PALETTE["border"]); sp.set_linewidth(1.2)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    if title:
        ax.set_title(title, color=PALETTE["accent2"], fontsize=10,
                     fontweight="bold", fontfamily="monospace", pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(color=PALETTE["border"], linestyle="--", linewidth=0.5, alpha=0.5)

def _legend(ax):
    ax.legend(fontsize=8, facecolor=PALETTE["panel"],
              labelcolor=PALETTE["text"], edgecolor=PALETTE["border"])

def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    log.info("  Saved → %s", out)
    return out


# ── Figure 1: Scatter comparison ───────────────────────────────────────────

def plot_scatter_comparison(
    rf_y_test:   np.ndarray,
    rf_y_pred:   np.ndarray,
    lstm_y_test: np.ndarray,
    lstm_y_pred: np.ndarray,
) -> Path:
    """Predicted vs Actual scatter for both models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("RUL PREDICTION  ·  Random Forest  vs  Temporal Model",
                 color=PALETTE["accent1"], fontsize=13,
                 fontweight="bold", fontfamily="monospace", y=1.01)

    for ax, y_te, y_pr, col, lbl in [
        (axes[0], rf_y_test,   rf_y_pred,   RF_COLOR,   "Random Forest"),
        (axes[1], lstm_y_test, lstm_y_pred, LSTM_COLOR, "Temporal (Window=10)"),
    ]:
        _style(ax, lbl, "Actual RUL (h)", "Predicted RUL (h)")
        ax.scatter(y_te, y_pr, alpha=0.25, s=8, color=col, edgecolors="none")
        lim = max(float(y_te.max()), float(y_pr.max())) * 1.05
        ax.plot([0, lim], [0, lim], color=PALETTE["accent3"],
                linewidth=1.5, linestyle="--", label="Perfect")
        mae = mean_absolute_error(y_te, y_pr)
        r2  = r2_score(y_te, y_pr)
        ax.text(0.97, 0.06,
                f"MAE = {mae:.1f} h\nR²  = {r2:.4f}",
                transform=ax.transAxes, ha="right",
                color=col, fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor=PALETTE["bg"], alpha=0.9,
                          edgecolor=PALETTE["border"]))
        _legend(ax)

    fig.tight_layout()
    return _save(fig, "rf_vs_temporal_scatter.png")


# ── Figure 2: Metrics bar chart ────────────────────────────────────────────

def plot_metrics_comparison(
    rf_metrics:   dict,
    lstm_metrics: dict,
) -> Path:
    """Side-by-side bar chart comparing MAE, RMSE, R²."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("MODEL PERFORMANCE COMPARISON  ·  RF vs Temporal",
                 color=PALETTE["accent1"], fontsize=12,
                 fontweight="bold", fontfamily="monospace")

    metrics_cfg = [
        ("mae",  "MAE (hours)",    "Lower is better ↓"),
        ("rmse", "RMSE (hours)",   "Lower is better ↓"),
        ("r2",   "R² Score",       "Higher is better ↑"),
    ]

    for ax, (key, ylabel, note) in zip(axes, metrics_cfg):
        vals   = [rf_metrics.get(key, 0), lstm_metrics.get(key, 0)]
        colors = [RF_COLOR, LSTM_COLOR]
        bars   = ax.bar(["Random\nForest", "Temporal\nModel"],
                        vals, color=colors, width=0.5,
                        edgecolor=PALETTE["bg"], linewidth=1.5)
        _style(ax, ylabel, "", ylabel)
        ax.set_facecolor(PALETTE["panel"])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val:.4f}" if key == "r2" else f"{val:.2f}",
                    ha="center", color=PALETTE["text"],
                    fontsize=9, fontweight="bold")
        ax.set_xticklabels(["Random\nForest", "Temporal\nModel"],
                           color=PALETTE["text"], fontsize=9)
        ax.text(0.5, -0.18, note, transform=ax.transAxes,
                ha="center", color=PALETTE["muted"], fontsize=8, style='italic')

    # Legend
    rf_patch   = mpatches.Patch(color=RF_COLOR,   label="Random Forest")
    lstm_patch = mpatches.Patch(color=LSTM_COLOR, label="Temporal Model")
    fig.legend(handles=[rf_patch, lstm_patch],
               fontsize=9, facecolor=PALETTE["panel"],
               labelcolor=PALETTE["text"], edgecolor=PALETTE["border"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return _save(fig, "rf_vs_temporal_bar.png")


# ── Figure 3: RUL trend over time ──────────────────────────────────────────

def plot_rul_trend(
    trend_df:  pd.DataFrame,
    node_id:   int,
    model_name: str = "Temporal Model",
) -> Path:
    """True vs predicted RUL over time for a single node."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    _style(ax,
           f"RUL TREND OVER TIME  ·  Node {node_id}",
           "Timestep", "RUL (hours)")

    ax.plot(trend_df["timestep"], trend_df["true_rul"],
            color=PALETTE["accent2"], linewidth=2.0, label="True RUL")
    ax.plot(trend_df["timestep"], trend_df["predicted_rul"],
            color=LSTM_COLOR, linewidth=1.5,
            linestyle="--", label=f"Predicted ({model_name})")
    ax.fill_between(trend_df["timestep"],
                    trend_df["true_rul"], trend_df["predicted_rul"],
                    alpha=0.15, color=LSTM_COLOR, label="Error band")
    _legend(ax)
    fig.tight_layout()
    return _save(fig, "rul_trend_node.png")


# ── Figure 4: ARGO vs Synthetic distributions ──────────────────────────────

def plot_argo_vs_synthetic(
    argo_df: pd.DataFrame,
    syn_df:  pd.DataFrame,
    features: list = ["depth_m", "temperature_c", "salinity_ppt", "battery_voltage"],
) -> Path:
    """Compare feature distributions between ARGO real and synthetic data."""
    fig, axes = plt.subplots(1, len(features), figsize=(16, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("ARGO Real Data  vs  Synthetic Data  ·  Feature Distributions",
                 color=PALETTE["accent1"], fontsize=12,
                 fontweight="bold", fontfamily="monospace")

    for ax, feat in zip(axes, features):
        _style(ax, feat.replace("_", " ").title(), feat, "Density")
        ax.hist(argo_df[feat].dropna(), bins=30, alpha=0.6,
                color=ARGO_COLOR, label="ARGO Real",
                density=True, edgecolor=PALETTE["bg"])
        ax.hist(syn_df[feat].dropna(), bins=30, alpha=0.6,
                color=SYN_COLOR, label="Synthetic",
                density=True, edgecolor=PALETTE["bg"])
        _legend(ax)

    fig.tight_layout()
    return _save(fig, "argo_vs_synthetic_dist.png")


# ── Figure 5: Battery degradation by cluster ───────────────────────────────

def plot_battery_degradation(df: pd.DataFrame) -> Path:
    """Battery voltage degradation curves coloured by depth cluster."""
    from ..config import CLUSTER_COLORS
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    _style(ax,
           "BATTERY DEGRADATION RATE  ·  By Depth Cluster",
           "Timestep", "Battery Voltage (V)")

    for cluster, grp in df.groupby("depth_cluster"):
        node_means = grp.groupby(["node_id", "timestep"])["battery_voltage"]\
                       .mean().reset_index()
        ts_mean = node_means.groupby("timestep")["battery_voltage"].mean()
        ts_std  = node_means.groupby("timestep")["battery_voltage"].std()
        col     = CLUSTER_COLORS.get(cluster, PALETTE["muted"])
        ax.plot(ts_mean.index, ts_mean.values, color=col,
                linewidth=2.0, label=cluster.upper())
        ax.fill_between(ts_mean.index,
                        (ts_mean - ts_std).values,
                        (ts_mean + ts_std).values,
                        color=col, alpha=0.12)

    # Mark the 2.5V cutoff line
    ax.axhline(2.5, color=PALETTE["warn"], linewidth=1.5,
               linestyle=":", label="Dead threshold (2.5V)")
    _legend(ax)
    fig.tight_layout()
    return _save(fig, "battery_degradation_clusters.png")


# ── Figure 6: Error distribution histogram ─────────────────────────────────

def plot_error_distribution(
    rf_y_test:   np.ndarray,
    rf_y_pred:   np.ndarray,
    lstm_y_test: np.ndarray,
    lstm_y_pred: np.ndarray,
) -> Path:
    """Histogram of absolute prediction errors for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("PREDICTION ERROR DISTRIBUTION  ·  RF vs Temporal",
                 color=PALETTE["accent1"], fontsize=12,
                 fontweight="bold", fontfamily="monospace")

    for ax, y_te, y_pr, col, lbl in [
        (axes[0], rf_y_test,   rf_y_pred,   RF_COLOR,   "Random Forest"),
        (axes[1], lstm_y_test, lstm_y_pred, LSTM_COLOR, "Temporal Model"),
    ]:
        errors = np.abs(y_te - y_pr)
        _style(ax, lbl, "Absolute Error (hours)", "Count")
        ax.hist(errors, bins=40, color=col, alpha=0.8,
                edgecolor=PALETTE["bg"], linewidth=0.5)
        ax.axvline(np.mean(errors), color=PALETTE["warn"],
                   linewidth=2.0, linestyle="--",
                   label=f"Mean={np.mean(errors):.1f}h")
        ax.axvline(np.median(errors), color=PALETTE["accent2"],
                   linewidth=2.0, linestyle="-.",
                   label=f"Median={np.median(errors):.1f}h")
        _legend(ax)

    fig.tight_layout()
    return _save(fig, "error_distribution.png")


# ══════════════════════════════════════════════════════════════════════════
# Master comparison runner
# ══════════════════════════════════════════════════════════════════════════

def run_phase1_comparison(
    sim_df:  pd.DataFrame,
    argo_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run the complete Phase 1 comparison pipeline.

    1. Train Random Forest RUL Regressor (existing model)
    2. Train Temporal RUL Model (new windowed model)
    3. Compare metrics
    4. Generate all 6 comparison figures
    5. Save metrics CSV

    Parameters
    ----------
    sim_df   : Simulation DataFrame (from simulate_sensor_data)
    argo_df  : Optional real ARGO data for distribution comparison

    Returns
    -------
    pd.DataFrame : Metrics comparison table
    """
    log.info("=" * 55)
    log.info(" PHASE 1: RF vs Temporal Model Comparison")
    log.info("=" * 55)

    # ── Train both models ──────────────────────────────────────────────
    log.info("Training Random Forest RUL Regressor …")
    rf = RULRegressor()
    rf.fit(sim_df)
    log.info("  %s", rf)

    log.info("Training Temporal RUL Model …")
    tm = TemporalRULModel()
    tm.fit(sim_df)
    log.info("  %s", tm)

    # ── Evaluate both ──────────────────────────────────────────────────
    rf_metrics   = evaluate_model(rf._y_test.values, rf._y_pred, "Random Forest")
    lstm_metrics = evaluate_model(tm._y_test,        tm._y_pred, "Temporal Model")

    metrics_df = pd.DataFrame([rf_metrics, lstm_metrics]).set_index("model")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = METRICS_DIR / "phase1_model_comparison.csv"
    metrics_df.to_csv(out_csv)
    log.info("Metrics saved → %s", out_csv)
    log.info("\n%s", metrics_df.to_string())

    # ── Generate all figures ───────────────────────────────────────────
    log.info("Generating Phase 1 comparison figures …")

    plot_scatter_comparison(
        rf._y_test.values, rf._y_pred,
        tm._y_test,        tm._y_pred,
    )

    plot_metrics_comparison(rf.metrics_, tm.metrics_)

    # RUL trend for a sample node with enough timesteps
    valid_nodes = sim_df.groupby("node_id")["timestep"]\
                        .count()[lambda x: x > tm.window + 5].index
    if len(valid_nodes) > 0:
        sample_node = int(valid_nodes[0])
        trend_df    = tm.predict_trend(sim_df, sample_node)
        if len(trend_df) > 0:
            plot_rul_trend(trend_df, sample_node)

    plot_battery_degradation(sim_df)

    plot_error_distribution(
        rf._y_test.values, rf._y_pred,
        tm._y_test,        tm._y_pred,
    )

    if argo_df is not None:
        plot_argo_vs_synthetic(argo_df, sim_df)
        log.info("  ARGO vs Synthetic distribution plot saved.")

    log.info("Phase 1 comparison complete.")
    return metrics_df
