"""
phase1/viz/phase1_dashboard.py
--------------------------------
Extended Phase 1 Dashboard

Adds 4 new panels to the standard AquaSense dashboard:
    Panel 9  : Real ARGO data vs Simulation comparison
    Panel 10 : ARGO depth profile (temperature / salinity)
    Panel 11 : Model comparison summary (RF vs LSTM)
    Panel 12 : LSTM training curves

Also generates standalone publication figures for each panel.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...config import FIGURES_DIR, OUTPUT_DIR, PALETTE

log = logging.getLogger("aquasense.phase1.viz")


def _sax(ax, title: str = "") -> None:
    ax.set_facecolor(PALETTE["panel"])
    for sp in ax.spines.values():
        sp.set_color(PALETTE["border"]); sp.set_linewidth(1.2)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    if title:
        ax.set_title(title, color=PALETTE["accent2"], fontsize=9,
                     fontweight="bold", pad=7, fontfamily="monospace")
    ax.grid(color=PALETTE["border"], linestyle="--", linewidth=0.5, alpha=0.5)


def _legend(ax) -> None:
    ax.legend(fontsize=7, facecolor=PALETTE["panel"],
              labelcolor=PALETTE["text"], edgecolor=PALETTE["border"])


# ── Individual panel builders ──────────────────────────────────────────────

def panel_argo_vs_sim(
    ax,
    argo_df: pd.DataFrame,
    sim_df:  pd.DataFrame,
    comparison_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Panel 9: Real ARGO data vs AquaSense simulation comparison.
    Side-by-side feature distributions (violin plots).
    """
    _sax(ax, "ARGO  REAL  DATA  vs  SIMULATION  ·  Feature Distributions")
    features = ["depth_m", "temperature_c", "salinity_ppt"]
    labels   = ["Depth (m)", "Temp (°C)", "Salinity (ppt)"]
    n        = len(features)
    x        = np.arange(n)
    w        = 0.3

    for i, (feat, label) in enumerate(zip(features, labels)):
        if feat not in argo_df.columns or feat not in sim_df.columns:
            continue
        argo_mean = argo_df[feat].mean()
        argo_std  = argo_df[feat].std()
        sim_mean  = sim_df[feat].mean()
        sim_std   = sim_df[feat].std()

        ax.bar(i - w/2, argo_mean, width=w,
               color=PALETTE["accent1"], alpha=0.8,
               edgecolor=PALETTE["bg"], label="ARGO Real" if i == 0 else "")
        ax.bar(i + w/2, sim_mean,  width=w,
               color=PALETTE["accent2"], alpha=0.8,
               edgecolor=PALETTE["bg"], label="Simulation" if i == 0 else "")
        ax.errorbar(i - w/2, argo_mean, yerr=argo_std,
                    fmt="none", color=PALETTE["warn"], capsize=4, linewidth=1.5)
        ax.errorbar(i + w/2, sim_mean,  yerr=sim_std,
                    fmt="none", color=PALETTE["warn"], capsize=4, linewidth=1.5)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, color=PALETTE["text"], fontsize=8)
    ax.set_ylabel("Mean ± Std", fontsize=8)
    _legend(ax)


def panel_argo_depth_profile(ax, argo_df: pd.DataFrame) -> None:
    """
    Panel 10: Real ARGO depth profiles (temperature and salinity vs depth).
    """
    _sax(ax, "ARGO  DEPTH  PROFILES  ·  Temp & Salinity vs Depth")
    if "temperature_c" not in argo_df.columns:
        ax.text(0.5, 0.5, "No ARGO data available",
                ha="center", color=PALETTE["muted"],
                transform=ax.transAxes)
        return

    # Bin by depth and compute mean
    argo_df = argo_df.copy()
    bins    = [0, 50, 100, 200, 300, 500, 750, 1000]
    argo_df["depth_bin"] = pd.cut(argo_df["depth_m"], bins=bins)
    profile = argo_df.groupby("depth_bin", observed=True).agg(
        temp_mean=("temperature_c", "mean"),
        sal_mean =("salinity_ppt",  "mean"),
        depth_mid=("depth_m",       "mean"),
    ).dropna()

    ax2 = ax.twiny()
    ax.plot(profile["temp_mean"], profile["depth_mid"],
            color=PALETTE["accent3"], linewidth=2.0,
            marker="o", markersize=5, label="Temperature (°C)")
    ax2.plot(profile["sal_mean"], profile["depth_mid"],
             color=PALETTE["accent2"], linewidth=2.0,
             linestyle="--", marker="s", markersize=5,
             label="Salinity (ppt)")

    ax.set_ylabel("Depth (m)", fontsize=8)
    ax.set_xlabel("Temperature (°C)", fontsize=8, color=PALETTE["accent3"])
    ax2.set_xlabel("Salinity (ppt)", fontsize=8, color=PALETTE["accent2"])
    ax.invert_yaxis()
    ax.tick_params(axis="x", colors=PALETTE["accent3"])
    ax2.tick_params(axis="x", colors=PALETTE["accent2"])
    ax2.set_facecolor(PALETTE["panel"])


def panel_model_comparison_bars(
    ax,
    comparison_results: pd.DataFrame,
) -> None:
    """
    Panel 11: RF vs LSTM metric comparison bars.
    """
    _sax(ax, "RF  vs  LSTM  ·  Metric Comparison")
    syn = comparison_results[
        comparison_results["dataset"] == "synthetic"].copy()

    if len(syn) < 2:
        ax.text(0.5, 0.5, "Run model comparison first",
                ha="center", color=PALETTE["muted"],
                transform=ax.transAxes)
        return

    metrics  = ["mae", "rmse", "mape_pct"]
    m_labels = ["MAE (h)", "RMSE (h)", "MAPE (%)"]
    x        = np.arange(len(metrics))
    w        = 0.35
    colors   = [PALETTE["accent1"], PALETTE["accent2"]]

    for mi, (met, lab) in enumerate(zip(metrics, m_labels)):
        for pi, (_, row) in enumerate(syn.iterrows()):
            val = row[met]
            ax.bar(mi + (pi - 0.5)*w, val, width=w,
                   color=colors[pi], alpha=0.85,
                   edgecolor=PALETTE["bg"],
                   label=row["model"] if mi == 0 else "")
            ax.text(mi + (pi - 0.5)*w, val + max(syn[met].max()*0.02, 0.5),
                    f"{val:.1f}",
                    ha="center", color=PALETTE["text"],
                    fontsize=7, fontweight="bold")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(m_labels, color=PALETTE["text"], fontsize=8)
    _legend(ax)


def panel_lstm_training(ax, lstm_model) -> None:
    """
    Panel 12: LSTM training and validation loss curves.
    """
    _sax(ax, "LSTM  TRAINING  HISTORY  ·  Loss Curves")
    hist = lstm_model.training_history_
    if not hist.get("train_loss"):
        ax.text(0.5, 0.5, "LSTM not trained yet",
                ha="center", color=PALETTE["muted"],
                transform=ax.transAxes)
        return

    epochs = range(1, len(hist["train_loss"]) + 1)
    ax.plot(epochs, hist["train_loss"],
            color=PALETTE["accent1"], linewidth=1.8, label="Train Loss")
    ax.plot(epochs, hist["val_loss"],
            color=PALETTE["accent3"], linewidth=1.8,
            linestyle="--", label="Val Loss")
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("MSE Loss", fontsize=8)

    # Mark best epoch
    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    best_loss  = min(hist["val_loss"])
    ax.axvline(best_epoch, color=PALETTE["warn"], linewidth=1,
               linestyle=":", alpha=0.8)
    ax.text(best_epoch + 0.3, best_loss,
            f"Best\nepoch {best_epoch}",
            color=PALETTE["warn"], fontsize=7)
    _legend(ax)


# ── Full Phase 1 dashboard ─────────────────────────────────────────────────

def build_phase1_dashboard(
    sim_df:       pd.DataFrame,
    argo_df:      pd.DataFrame,
    comp_results: pd.DataFrame,
    lstm_model,
    comparison_df: Optional[pd.DataFrame] = None,
    output_path:  Optional[Path] = None,
) -> Path:
    """
    Render the Phase 1 extended dashboard (4 new panels).

    Parameters
    ----------
    sim_df       : AquaSense simulated data
    argo_df      : Real ARGO / WOA data
    comp_results : Model comparison results DataFrame
    lstm_model   : Fitted LSTMRULPredictor
    comparison_df: ARGO vs sim comparison (from ArgoConnector)
    output_path  : Save path for PNG

    Returns
    -------
    Path to saved PNG
    """
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "phase1_dashboard.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 12), facecolor=PALETTE["bg"])
    fig.suptitle(
        "◈  AQUASENSE  PHASE 1  ·  REAL DATA + LSTM COMPARISON  ◈",
        fontsize=15, fontweight="bold", color=PALETTE["accent1"],
        y=0.97, fontfamily="monospace")

    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.05, right=0.97,
        top=0.91, bottom=0.06)

    # Row 1
    panel_argo_vs_sim(
        fig.add_subplot(gs[0, :2]), argo_df, sim_df, comparison_df)
    panel_argo_depth_profile(
        fig.add_subplot(gs[0, 2:]), argo_df)

    # Row 2
    panel_model_comparison_bars(
        fig.add_subplot(gs[1, :2]), comp_results)
    panel_lstm_training(
        fig.add_subplot(gs[1, 2:]), lstm_model)

    fig.text(
        0.5, 0.01,
        "AquaSense Phase 1  ·  ARGO Real Data Integration + LSTM RUL Predictor  "
        "·  Muhammad Hassaan Naeem",
        ha="center", color=PALETTE["muted"], fontsize=7,
        fontfamily="monospace")

    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    log.info("Phase 1 dashboard saved → %s", output_path)
    return output_path


def build_argo_validation_figure(
    argo_df:       pd.DataFrame,
    sim_df:        pd.DataFrame,
    comparison_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Generate a standalone ARGO validation figure for the thesis.
    Shows real ocean data aligns with simulation parameters.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             facecolor=PALETTE["bg"])

    panel_argo_vs_sim(axes[0], argo_df, sim_df, comparison_df)
    panel_argo_depth_profile(axes[1], argo_df)

    fig.suptitle(
        "ARGO Real Ocean Data — Validation of AquaSense Simulation Parameters",
        fontsize=12, fontweight="bold", color=PALETTE["accent1"],
        fontfamily="monospace")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "argo_validation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    log.info("ARGO validation figure saved → %s", out)
    return out
