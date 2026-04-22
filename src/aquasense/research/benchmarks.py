"""
research/benchmarks.py
-----------------------
Benchmark comparison framework for the thesis Chapter 4 experiments.

Compares four routing protocols:
    1. Random          – equal weights (baseline)
    2. LEACH (Energy)  – energy-only CH selection
    3. DBR (Depth)     – depth-only CH selection
    4. Proposed        – depth-aware + energy-aware (AquaSense)

Generates:
    - Network lifetime comparison table  (CSV)
    - Alive-nodes-over-time plot         (PNG)
    - Energy consumption comparison      (PNG)
    - Packet delivery ratio comparison   (PNG)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import BENCHMARK_PROTOCOLS, FIGURES_DIR, METRICS_DIR, PALETTE
from .routing_protocol import protocol_summary, simulate_routing_rounds

log = logging.getLogger("aquasense.benchmarks")

PROTOCOL_COLORS = {
    "Random"         : "#aaaaaa",
    "LEACH (Energy)" : "#ffaa00",
    "DBR (Depth)"    : "#00aaff",
    "Proposed"       : "#00ffd5",
}


# ── Run all benchmarks ─────────────────────────────────────────────────────

def run_all_benchmarks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Run all four protocol simulations on *df*.

    Returns
    -------
    dict mapping protocol name → rounds DataFrame
    """
    results = {}
    for protocol in BENCHMARK_PROTOCOLS:
        log.info("  Running benchmark: %s", protocol)
        results[protocol] = simulate_routing_rounds(df, protocol=protocol)
    return results


# ── Summary table ──────────────────────────────────────────────────────────

def build_summary_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate per-protocol KPIs into a single comparison table.
    Saved as results/metrics/protocol_comparison.csv.
    """
    rows = [protocol_summary(rdf) for rdf in results.values()]
    summary = pd.DataFrame(rows).set_index("protocol")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "protocol_comparison.csv"
    summary.to_csv(out)
    log.info("  Saved benchmark table → %s", out)
    return summary


# ── Plots ──────────────────────────────────────────────────────────────────

def _base_fig(title: str, w: float = 10, h: float = 5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    for sp in ax.spines.values():
        sp.set_color(PALETTE["border"]); sp.set_linewidth(1.2)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    ax.set_title(title, color=PALETTE["accent2"], fontsize=11,
                 fontweight="bold", fontfamily="monospace")
    ax.grid(color=PALETTE["border"], linestyle="--", linewidth=0.5, alpha=0.5)
    return fig, ax


def plot_alive_nodes(results: dict[str, pd.DataFrame]) -> Path:
    """
    Line chart: number of alive nodes over simulation rounds.
    Corresponds to Figure 4.x in the thesis (Network Lifetime).
    """
    fig, ax = _base_fig("NETWORK  LIFETIME  —  Alive Nodes per Round")
    for protocol, rdf in results.items():
        col = PROTOCOL_COLORS.get(protocol, PALETTE["accent1"])
        lw  = 2.5 if protocol == "Proposed" else 1.5
        ax.plot(rdf["round_num"], rdf["alive_nodes"],
                color=col, linewidth=lw, label=protocol,
                linestyle="-" if protocol == "Proposed" else "--")

    ax.set_xlabel("Round (Timestep)", fontsize=9)
    ax.set_ylabel("Alive Nodes", fontsize=9)
    ax.legend(fontsize=8, facecolor=PALETTE["panel"],
              labelcolor=PALETTE["text"], edgecolor=PALETTE["border"])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "alive_nodes_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    log.info("  Saved → %s", out)
    return out


def plot_energy_consumption(results: dict[str, pd.DataFrame]) -> Path:
    """
    Bar chart: cumulative energy consumed per protocol.
    Corresponds to Figure 4.x in the thesis (Energy Efficiency).
    """
    protocols = list(results.keys())
    totals    = [results[p]["energy_consumed"].sum() for p in protocols]
    colors    = [PROTOCOL_COLORS.get(p, PALETTE["accent1"]) for p in protocols]

    fig, ax = _base_fig("CUMULATIVE  ENERGY  CONSUMPTION  BY  PROTOCOL")
    bars = ax.bar(protocols, totals, color=colors,
                  edgecolor=PALETTE["bg"], linewidth=1.5, width=0.5)
    for bar, val in zip(bars, totals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+max(totals)*0.01,
                f"{val:.1f}", ha="center", color=PALETTE["text"],
                fontsize=8, fontweight="bold")
    ax.set_ylabel("Total Energy (μJ)", fontsize=9)
    ax.set_xticklabels(protocols, color=PALETTE["text"], fontsize=8)

    out = FIGURES_DIR / "energy_consumption_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    log.info("  Saved → %s", out)
    return out


def plot_delivery_ratio(results: dict[str, pd.DataFrame]) -> Path:
    """
    Line chart: packet delivery ratio over rounds.
    Corresponds to Figure 4.x in the thesis (Reliability).
    """
    fig, ax = _base_fig("PACKET  DELIVERY  RATIO  OVER  ROUNDS")
    for protocol, rdf in results.items():
        col = PROTOCOL_COLORS.get(protocol, PALETTE["accent1"])
        lw  = 2.5 if protocol == "Proposed" else 1.5
        ax.plot(rdf["round_num"], rdf["delivery_ratio"] * 100,
                color=col, linewidth=lw, label=protocol,
                linestyle="-" if protocol == "Proposed" else "--")
    ax.set_xlabel("Round (Timestep)", fontsize=9)
    ax.set_ylabel("Delivery Ratio (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, facecolor=PALETTE["panel"],
              labelcolor=PALETTE["text"], edgecolor=PALETTE["border"])

    out = FIGURES_DIR / "delivery_ratio_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    log.info("  Saved → %s", out)
    return out


def plot_ch_fitness_distribution(df: pd.DataFrame) -> Path:
    """
    Box-plot of CH fitness scores across depth clusters for the Proposed protocol.
    """
    from .routing_protocol import assign_depth_clusters, compute_ch_fitness

    df2 = assign_depth_clusters(df).copy()
    df2["ch_fitness"] = df2.apply(
        lambda r: compute_ch_fitness(
            r["battery_voltage"], r["depth_m"], r["packet_success_rt"]), axis=1)

    fig, ax = _base_fig("CH  FITNESS  SCORE  DISTRIBUTION  BY  DEPTH  CLUSTER")
    cluster_order  = ["shallow", "mid", "deep"]
    cluster_colors = [PALETTE["shallow"], PALETTE["mid"], PALETTE["deep"]]
    data = [df2[df2["protocol_cluster"] == c]["ch_fitness"].values
            for c in cluster_order]

    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": PALETTE["bg"]})
    for patch, col in zip(bp["boxes"], cluster_colors):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    for element in ["whiskers","caps","fliers"]:
        for item in bp[element]:
            item.set_color(PALETTE["muted"])

    ax.set_xticks([1,2,3])
    ax.set_xticklabels([c.upper() for c in cluster_order],
                       color=PALETTE["text"], fontsize=9)
    ax.set_ylabel("CH Fitness Score", fontsize=9)

    out = FIGURES_DIR / "ch_fitness_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    log.info("  Saved → %s", out)
    return out


# ── Master benchmark runner ────────────────────────────────────────────────

def run_full_benchmark_suite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all benchmarks, generate all figures, and return summary table.
    Entry-point called from the pipeline.
    """
    log.info("Running full benchmark suite …")
    results = run_all_benchmarks(df)
    summary = build_summary_table(results)
    plot_alive_nodes(results)
    plot_energy_consumption(results)
    plot_delivery_ratio(results)
    plot_ch_fitness_distribution(df)
    log.info("Benchmark suite complete.")
    return summary
