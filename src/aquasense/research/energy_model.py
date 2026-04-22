"""
research/energy_model.py
------------------------
Acoustic underwater energy consumption model for IoUT.

Models energy costs for:
    - Intra-cluster transmission (node → CH)
    - Inter-cluster forwarding  (CH → CH → Surface Sink)
    - Data aggregation at CH
    - Idle listening

Based on standard underwater acoustic energy models used in IoUT research.
Provides the theoretical foundation for the energy analysis in thesis Chapter 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Physical constants ─────────────────────────────────────────────────────

# Acoustic absorption coefficient (dB/km) — depth-dependent
def absorption_coefficient(depth_m: float, frequency_khz: float = 25.0) -> float:
    """
    Compute acoustic absorption coefficient α (dB/km) using Thorp's formula.

    α = 0.11·f²/(1+f²) + 44·f²/(4100+f²) + 2.75×10⁻⁴·f² + 0.003

    Parameters
    ----------
    depth_m       : Node depth in metres (influences temperature correction)
    frequency_khz : Acoustic carrier frequency in kHz
    """
    f  = frequency_khz
    f2 = f * f
    alpha = (0.11 * f2 / (1 + f2)
             + 44  * f2 / (4100 + f2)
             + 2.75e-4 * f2
             + 0.003)
    # Depth correction: absorption increases slightly with depth
    depth_factor = 1.0 + 0.0005 * depth_m
    return alpha * depth_factor


def path_loss(distance_m: float, depth_m: float,
              frequency_khz: float = 25.0) -> float:
    """
    Compute path loss A(d) in dB for an underwater acoustic link.

    A(d) = d^k · a(f)^d     (in linear scale)
    In dB: A(d)_dB = k·10·log10(d) + α·d

    Parameters
    ----------
    distance_m    : Transmission distance in metres
    depth_m       : Sender depth (for absorption coefficient)
    frequency_khz : Carrier frequency in kHz

    Returns
    -------
    float : Path loss in dB
    """
    k     = 1.5    # spreading factor (cylindrical+spherical)
    alpha = absorption_coefficient(depth_m, frequency_khz)
    d_km  = distance_m / 1000.0
    return k * 10 * np.log10(distance_m + 1e-9) + alpha * d_km


# ── Energy cost models ─────────────────────────────────────────────────────

# Energy coefficients (μJ per bit)
E_TX_ELEC   = 50e-9    # Electronics energy (J/bit)
E_AMP_FREE  = 100e-12  # Amplifier energy   (J/bit/m²) — free-space
E_AMP_MP    = 0.0013e-12  # Multi-path fading
E_RX_ELEC   = 50e-9    # Receiver electronics
E_AGG       = 5e-9     # Data aggregation per bit

PACKET_SIZE_BITS = 4000   # Typical sensor packet (500 bytes)
THRESHOLD_DIST_M = 87.7   # Cross-over distance (m)


def tx_energy(distance_m: float, n_bits: int = PACKET_SIZE_BITS) -> float:
    """
    Energy (J) to transmit *n_bits* over *distance_m*.

    Uses free-space model for d < threshold, multi-path for d >= threshold.
    """
    if distance_m < THRESHOLD_DIST_M:
        return n_bits * (E_TX_ELEC + E_AMP_FREE  * distance_m ** 2)
    else:
        return n_bits * (E_TX_ELEC + E_AMP_MP    * distance_m ** 4)


def rx_energy(n_bits: int = PACKET_SIZE_BITS) -> float:
    """Energy (J) to receive *n_bits*."""
    return n_bits * E_RX_ELEC


def aggregation_energy(n_bits: int = PACKET_SIZE_BITS) -> float:
    """Energy (J) for data aggregation at CH."""
    return n_bits * E_AGG


# ── Per-round energy estimation ────────────────────────────────────────────

def estimate_round_energy(
    snapshot:   pd.DataFrame,
    cluster_col: str = "depth_cluster",
) -> dict:
    """
    Estimate total energy consumed in one routing round.

    Parameters
    ----------
    snapshot    : Latest readings per node (one row per node)
    cluster_col : Column identifying depth cluster membership

    Returns
    -------
    dict with keys:
        intra_cluster_tx  : Energy for node → CH transmissions (J)
        inter_cluster_tx  : Energy for CH → CH → sink forwarding (J)
        aggregation       : Energy for CH data aggregation (J)
        total             : Total round energy (J)
        total_uJ          : Total round energy (μJ)
    """
    alive = snapshot[snapshot["battery_voltage"] > 2.5]
    if alive.empty:
        return {"intra_cluster_tx": 0, "inter_cluster_tx": 0,
                "aggregation": 0, "total": 0, "total_uJ": 0}

    n_alive   = len(alive)
    clusters  = alive[cluster_col].value_counts()
    n_clusters = len(clusters)

    # Intra-cluster: each member node sends one packet to CH
    # Average intra-cluster distance estimated from depth variance
    avg_intra_dist = float(alive.groupby(cluster_col)["depth_m"]
                           .std().fillna(10).mean())
    e_intra = tx_energy(avg_intra_dist) * (n_alive - n_clusters)

    # Inter-cluster: each CH forwards to next CH or sink
    # Average inter-cluster distance = span of depth zones / n_clusters
    depth_span     = float(alive["depth_m"].max() - alive["depth_m"].min())
    avg_inter_dist = max(10.0, depth_span / max(n_clusters, 1))
    e_inter = tx_energy(avg_inter_dist) * n_clusters

    # Aggregation: each CH aggregates all member packets
    e_agg = aggregation_energy() * n_clusters

    total = e_intra + e_inter + e_agg
    return {
        "intra_cluster_tx": round(e_intra * 1e6, 4),   # μJ
        "inter_cluster_tx": round(e_inter * 1e6, 4),
        "aggregation"     : round(e_agg   * 1e6, 4),
        "total"           : round(total,           8),
        "total_uJ"        : round(total   * 1e6,  4),
    }


def energy_summary_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average per-round energy cost per depth cluster.
    Used in thesis Chapter 4 energy analysis tables.
    """
    records = []
    for cluster, grp in df.groupby("depth_cluster"):
        alive = grp[grp["battery_voltage"] > 2.5]
        if alive.empty:
            continue
        n_nodes   = len(alive)
        avg_depth = float(alive["depth_m"].mean())
        avg_dist  = float(alive["depth_m"].std() or 10)
        e_member  = tx_energy(avg_dist) * 1e6    # μJ
        e_ch      = (tx_energy(avg_depth) + aggregation_energy()) * 1e6
        records.append({
            "depth_cluster"       : cluster,
            "n_nodes"             : n_nodes,
            "avg_depth_m"         : round(avg_depth, 1),
            "avg_intra_dist_m"    : round(avg_dist, 1),
            "member_tx_energy_uJ" : round(e_member, 4),
            "ch_total_energy_uJ"  : round(e_ch,     4),
        })
    return pd.DataFrame(records)
