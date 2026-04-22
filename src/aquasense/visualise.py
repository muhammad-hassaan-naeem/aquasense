"""
visualise.py  –  8-panel AquaSense monitoring dashboard.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .config import (
    CLUSTER_COLORS, DASHBOARD_DPI, DASHBOARD_SIZE,
    FEATURES, KM_COLORS, OUTPUT_DIR, PALETTE,
)


def _style_ax(ax, title: str = "") -> None:
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


def _legend(ax):
    ax.legend(fontsize=7, facecolor=PALETTE["panel"],
              labelcolor=PALETTE["text"], edgecolor=PALETTE["border"])


def _panel_kpi(ax, latest_df, n_anomalies):
    ax.set_facecolor(PALETTE["panel"])
    for sp in ax.spines.values(): sp.set_color(PALETTE["border"])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title("FLEET STATUS  ·  LIVE METRICS", color=PALETTE["accent2"],
                 fontsize=9, fontweight="bold", pad=7, fontfamily="monospace")
    kpis = [
        ("NODES",       f"{latest_df['node_id'].nunique()}",                 PALETTE["accent1"]),
        ("AVG RUL",     f"{latest_df['rul_hours'].mean():.1f}h",             PALETTE["accent2"]),
        ("ANOMALIES",   f"{n_anomalies}",                                    PALETTE["warn"]),
        ("AVG BATTERY", f"{latest_df['battery_voltage'].mean():.2f}V",      PALETTE["mid"]),
        ("AVG PSR",     f"{latest_df['packet_success_rt'].mean()*100:.1f}%", PALETTE["shallow"]),
    ]
    for i, (lbl, val, col) in enumerate(kpis):
        x = 0.1 + i * 0.19
        ax.text(x, 0.68, val, color=col, fontsize=17,
                fontweight="bold", ha="center", fontfamily="monospace")
        ax.text(x, 0.26, lbl, color=PALETTE["muted"], fontsize=7,
                ha="center", fontfamily="monospace")
        if i < len(kpis)-1:
            ax.axvline(x+0.095, color=PALETTE["border"], linewidth=1, alpha=0.6)


def _panel_cluster_rul(ax, cluster_stats):
    _style_ax(ax, "CLUSTER  ENERGY  EFFICIENCY  ·  AVG RUL")
    cdf  = cluster_stats.set_index("depth_cluster")
    xpos = np.arange(len(cdf))
    bars = ax.bar(xpos, cdf["avg_rul"],
                  color=[CLUSTER_COLORS.get(c, PALETTE["muted"]) for c in cdf.index],
                  width=0.5, edgecolor=PALETTE["bg"], linewidth=1.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels([c.upper() for c in cdf.index], color=PALETTE["text"], fontsize=8)
    ax.set_ylabel("Avg RUL (hours)", fontsize=8)
    for bar, val in zip(bars, cdf["avg_rul"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                f"{val:.1f}h", ha="center", color=PALETTE["text"],
                fontsize=8, fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(xpos, cdf["total_anomalies"], "o--", color=PALETTE["warn"],
             linewidth=1.5, markersize=6, label="Anomalies")
    ax2.set_ylabel("Total Anomalies", color=PALETTE["warn"], fontsize=8)
    ax2.tick_params(axis="y", colors=PALETTE["warn"], labelsize=8)
    ax2.set_facecolor(PALETTE["panel"])


def _panel_rul_scatter(ax, y_te, y_pred):
    _style_ax(ax, "RUL  PREDICTION  vs  ACTUAL  (test set)")
    ax.scatter(y_te, y_pred, alpha=0.3, s=10,
               color=PALETTE["accent1"], edgecolors="none")
    lim = max(float(y_te.max()), float(y_pred.max()))
    ax.plot([0,lim],[0,lim], color=PALETTE["accent3"],
            linewidth=1.5, linestyle="--", label="Perfect fit")
    ax.set_xlabel("Actual RUL (h)", fontsize=8)
    ax.set_ylabel("Predicted RUL (h)", fontsize=8)
    _legend(ax)
    ax.text(0.97, 0.08,
            f"MAE = {mean_absolute_error(y_te,y_pred):.1f} h\n"
            f"R²  = {r2_score(y_te,y_pred):.4f}",
            transform=ax.transAxes, ha="right",
            color=PALETTE["accent2"], fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["bg"],
                      alpha=0.85, edgecolor=PALETTE["border"]))


def _panel_battery_decay(ax, df):
    _style_ax(ax, "BATTERY  VOLTAGE  DECAY  BY  CLUSTER  (±1σ)")
    for cluster, grp in df.groupby("depth_cluster"):
        ts_mean = grp.groupby("timestep")["battery_voltage"].mean()
        ts_std  = grp.groupby("timestep")["battery_voltage"].std()
        col     = CLUSTER_COLORS[cluster]
        ax.plot(ts_mean.index, ts_mean.values, color=col,
                linewidth=1.8, label=cluster.upper())
        ax.fill_between(ts_mean.index, ts_mean-ts_std, ts_mean+ts_std,
                        color=col, alpha=0.12)
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_ylabel("Battery Voltage (V)", fontsize=8)
    _legend(ax)


def _panel_depth_vs_rul(ax, latest_df):
    _style_ax(ax, "DEPTH  vs  REMAINING  USEFUL  LIFE")
    for cluster, grp in latest_df.groupby("depth_cluster"):
        ax.scatter(grp["depth_m"], grp["rul_hours"],
                   color=CLUSTER_COLORS[cluster], alpha=0.70,
                   s=35, label=cluster.upper(),
                   edgecolors=PALETTE["bg"], linewidth=0.5)
    ax.set_xlabel("Depth (m)", fontsize=8)
    ax.set_ylabel("RUL (hours)", fontsize=8)
    _legend(ax)


def _panel_anomaly_distribution(ax, anomaly_df):
    _style_ax(ax, "ANOMALY  DISTRIBUTION  BY  CLUSTER")
    anom_grp = anomaly_df.groupby("depth_cluster")[["is_anomaly","anomaly_pred"]].sum()
    x = np.arange(len(anom_grp)); w = 0.35
    ax.bar(x-w/2, anom_grp["is_anomaly"], width=w, color=PALETTE["warn"],
           alpha=0.85, label="True", edgecolor=PALETTE["bg"])
    ax.bar(x+w/2, anom_grp["anomaly_pred"], width=w, color=PALETTE["accent3"],
           alpha=0.85, label="Detected", edgecolor=PALETTE["bg"])
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in anom_grp.index],
                       color=PALETTE["text"], fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    _legend(ax)


def _panel_kmeans(ax, km_df):
    _style_ax(ax, "K-MEANS  DEPTH  CLUSTERS  (depth vs battery)")
    for ci, col in enumerate(KM_COLORS):
        sub = km_df[km_df["km_cluster"] == ci]
        ax.scatter(sub["depth_m"], sub["battery_voltage"],
                   color=col, alpha=0.40, s=12,
                   label=f"Cluster {ci}", edgecolors="none")
    ax.set_xlabel("Depth (m)", fontsize=8)
    ax.set_ylabel("Battery Voltage (V)", fontsize=8)
    _legend(ax)


def _panel_feature_importance(ax, fi):
    _style_ax(ax, "FEATURE  IMPORTANCE  —  RUL  Regressor")
    idx    = np.argsort(fi)
    fnames = np.array(FEATURES)
    colors = [PALETTE["accent1"] if f == "battery_voltage"
              else PALETTE["mid"] for f in fnames[idx]]
    ax.barh(fnames[idx], fi[idx], color=colors,
            edgecolor=PALETTE["bg"], linewidth=0.8)
    ax.set_xlabel("Importance", fontsize=8)
    ax.tick_params(axis="y", labelsize=7, colors=PALETTE["text"])


def build_dashboard(
    df, latest_df, cluster_stats,
    y_test, y_pred, anomaly_df, km_df,
    feature_importances,
    output_path=None,
) -> Path:
    """Render and save the 8-panel monitoring dashboard."""
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "aquasense_dashboard.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_anomalies = int(anomaly_df["anomaly_pred"].sum())
    fig = plt.figure(figsize=DASHBOARD_SIZE, facecolor=PALETTE["bg"])
    fig.suptitle("◈  AQUASENSE  ·  UNDERWATER SENSOR NODE MONITORING SYSTEM  ◈",
                 fontsize=16, fontweight="bold", color=PALETTE["accent1"],
                 y=0.985, fontfamily="monospace")

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.52, wspace=0.38,
                           left=0.05, right=0.97, top=0.94, bottom=0.04)

    _panel_kpi(               fig.add_subplot(gs[0, :2]), latest_df, n_anomalies)
    _panel_cluster_rul(       fig.add_subplot(gs[0, 2:]), cluster_stats)
    _panel_rul_scatter(       fig.add_subplot(gs[1, :2]), y_test, y_pred)
    _panel_battery_decay(     fig.add_subplot(gs[1, 2:]), df)
    _panel_depth_vs_rul(      fig.add_subplot(gs[2, :2]), latest_df)
    _panel_anomaly_distribution(fig.add_subplot(gs[2, 2:]), anomaly_df)
    _panel_kmeans(            fig.add_subplot(gs[3, :2]), km_df)
    _panel_feature_importance(fig.add_subplot(gs[3, 2:]), feature_importances)

    fig.text(0.5, 0.003,
             "AquaSense v2.0  ·  RF RUL Regressor + Isolation Forest + "
             "Depth-Aware CH Selection  ·  IoUT Research Framework",
             ha="center", color=PALETTE["muted"], fontsize=7,
             fontfamily="monospace")

    fig.savefig(output_path, dpi=DASHBOARD_DPI,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    return output_path
