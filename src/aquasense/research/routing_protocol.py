"""
research/routing_protocol.py
-----------------------------
Proposed Energy-Efficient Depth-Aware Clustering-Based Routing Protocol
for Internet of Underwater Things (IoUT).

This module implements the CH selection algorithm described in Chapter 3
of the thesis: "Energy-Efficient Depth-Aware Clustering-Based Routing
Protocol for IoUT Using Depth Sensors"

Architecture (matches network_architecture.png):
    Surface Sink  ←  Cluster Head (CH)  ←  Underwater Sensor Nodes
                                ↑
                          Base Station

Protocol Phases
---------------
1. Cluster Formation  – nodes grouped by depth zone
2. CH Selection       – fitness-weighted score (energy + depth + link)
3. Data Aggregation   – member nodes report to CH
4. Multi-Hop Routing  – CHs forward toward surface sink
5. CH Rotation        – re-elect when CH battery drops below threshold
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import (
    BATTERY_CUTOFF_V,
    BENCHMARK_PROTOCOLS,
    CH_ROTATION_THRESHOLD,
    CH_WEIGHT_DEPTH,
    CH_WEIGHT_ENERGY,
    CH_WEIGHT_LINK,
    CLUSTER_BOUNDS,
)

log = logging.getLogger("aquasense.routing")


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class ClusterHead:
    """Represents a selected Cluster Head node."""
    node_id:         int
    depth_m:         float
    battery_voltage: float
    packet_success_rt: float
    fitness_score:   float
    cluster_label:   str


@dataclass
class RoutingRound:
    """Results of one routing protocol round."""
    round_num:        int
    cluster_heads:    List[ClusterHead]
    alive_nodes:      int
    total_nodes:      int
    avg_rul:          float
    avg_battery:      float
    energy_consumed:  float
    packets_delivered: int
    packets_lost:     int

    @property
    def delivery_ratio(self) -> float:
        total = self.packets_delivered + self.packets_lost
        return self.packets_delivered / total if total > 0 else 0.0

    @property
    def network_alive_ratio(self) -> float:
        return self.alive_nodes / self.total_nodes if self.total_nodes > 0 else 0.0


# ── CH Fitness Score ───────────────────────────────────────────────────────

def compute_ch_fitness(
    battery_voltage:   float,
    depth_m:           float,
    packet_success_rt: float,
    max_battery:       float = 4.2,
    max_depth:         float = 1000.0,
    energy_w:          float = CH_WEIGHT_ENERGY,
    depth_w:           float = CH_WEIGHT_DEPTH,
    link_w:            float = CH_WEIGHT_LINK,
) -> float:
    """
    Compute the CH fitness score for a candidate node.

    Fitness = w_e * E_norm + w_d * D_norm + w_l * L_norm

    Where:
        E_norm = battery_voltage / max_battery          (higher = better)
        D_norm = 1 - depth_m / max_depth               (shallower = better)
        L_norm = packet_success_rate                    (higher = better)

    Parameters
    ----------
    battery_voltage   : Current battery level (V)
    depth_m           : Node depth in metres
    packet_success_rt : Fraction of packets successfully received (0–1)
    max_battery       : Maximum possible battery voltage
    max_depth         : Maximum deployment depth
    energy_w          : Weight for energy component
    depth_w           : Weight for depth component
    link_w            : Weight for link quality component

    Returns
    -------
    float : Fitness score in [0, 1]

    Notes
    -----
    Weights must sum to 1.0 for the score to remain in [0, 1].
    Default weights follow the proposed protocol (0.5 / 0.3 / 0.2).
    """
    energy_score = battery_voltage / max_battery
    depth_score  = 1.0 - (depth_m / max_depth)
    link_score   = packet_success_rt

    return (energy_w * energy_score
            + depth_w  * depth_score
            + link_w   * link_score)


# ── Cluster Formation ──────────────────────────────────────────────────────

def assign_depth_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each node to a depth zone based on CLUSTER_BOUNDS.
    Overrides the simulation label with rule-based assignment
    so the protocol works on any input data.
    """
    df = df.copy()

    def _zone(depth):
        for label, (lo, hi) in CLUSTER_BOUNDS.items():
            if lo <= depth < hi:
                return label
        return "deep"

    df["protocol_cluster"] = df["depth_m"].apply(_zone)
    return df


# ── CH Selection ───────────────────────────────────────────────────────────

def select_cluster_heads(
    snapshot: pd.DataFrame,
    protocol: str = "Proposed",
) -> List[ClusterHead]:
    """
    Select one Cluster Head per depth zone for a given network snapshot.

    The snapshot should be the latest reading per node (one row per node).
    Dead nodes (battery <= BATTERY_CUTOFF_V) are excluded.

    Parameters
    ----------
    snapshot : pd.DataFrame
        Latest readings — one row per node.
    protocol : str
        Protocol variant to use (keys from config.BENCHMARK_PROTOCOLS).

    Returns
    -------
    List[ClusterHead]
        One CH per depth cluster, ordered shallow → mid → deep.
    """
    weights  = BENCHMARK_PROTOCOLS.get(protocol, BENCHMARK_PROTOCOLS["Proposed"])
    snapshot = assign_depth_clusters(snapshot)

    # Exclude dead nodes
    alive    = snapshot[snapshot["battery_voltage"] > BATTERY_CUTOFF_V].copy()

    if alive.empty:
        log.warning("All nodes are dead — no CHs can be elected.")
        return []

    alive["ch_fitness"] = alive.apply(
        lambda r: compute_ch_fitness(
            battery_voltage   = r["battery_voltage"],
            depth_m           = r["depth_m"],
            packet_success_rt = r["packet_success_rt"],
            energy_w          = weights["energy_w"],
            depth_w           = weights["depth_w"],
            link_w            = weights["link_w"],
        ), axis=1)

    cluster_heads = []
    for zone in ["shallow", "mid", "deep"]:
        zone_nodes = alive[alive["protocol_cluster"] == zone]
        if zone_nodes.empty:
            continue
        best = zone_nodes.loc[zone_nodes["ch_fitness"].idxmax()]
        cluster_heads.append(ClusterHead(
            node_id           = int(best["node_id"]),
            depth_m           = float(best["depth_m"]),
            battery_voltage   = float(best["battery_voltage"]),
            packet_success_rt = float(best["packet_success_rt"]),
            fitness_score     = float(best["ch_fitness"]),
            cluster_label     = zone,
        ))

    return cluster_heads


# ── Multi-Hop Routing ──────────────────────────────────────────────────────

def build_routing_path(
    cluster_heads: List[ClusterHead],
) -> List[ClusterHead]:
    """
    Build a multi-hop path from the deepest CH toward the surface sink.

    Strategy: route deep → mid → shallow → surface sink.
    Each hop selects the CH in the next shallower zone.

    Returns
    -------
    List[ClusterHead] ordered from deepest to shallowest (toward surface).
    """
    order = {"deep": 0, "mid": 1, "shallow": 2}
    return sorted(cluster_heads, key=lambda ch: order.get(ch.cluster_label, 0))


# ── CH Rotation ────────────────────────────────────────────────────────────

def needs_rotation(ch: ClusterHead, initial_battery: float = 4.0) -> bool:
    """
    Return True if the CH should be rotated (battery below threshold).

    Parameters
    ----------
    ch              : Current ClusterHead
    initial_battery : Nominal full battery voltage
    """
    return (ch.battery_voltage / initial_battery) < CH_ROTATION_THRESHOLD


# ── Network Lifetime Simulation ────────────────────────────────────────────

def simulate_routing_rounds(
    df:       pd.DataFrame,
    protocol: str = "Proposed",
) -> pd.DataFrame:
    """
    Simulate the full network lifetime round-by-round.

    For each timestep in *df*, selects CHs, estimates energy consumption,
    and records network statistics.  This produces the data needed for the
    network-lifetime comparison plots in Chapter 4 of the thesis.

    Parameters
    ----------
    df       : Full simulation DataFrame (all nodes, all timesteps)
    protocol : Protocol variant name (from config.BENCHMARK_PROTOCOLS)

    Returns
    -------
    pd.DataFrame with one row per timestep containing:
        round_num, alive_nodes, total_nodes, n_cluster_heads,
        avg_rul, avg_battery, energy_consumed,
        packets_delivered, packets_lost, delivery_ratio,
        network_alive_ratio, first_node_death
    """
    weights     = BENCHMARK_PROTOCOLS.get(protocol, BENCHMARK_PROTOCOLS["Proposed"])
    timesteps   = sorted(df["timestep"].unique())
    total_nodes = df["node_id"].nunique()
    records     = []
    first_death = None

    for t in timesteps:
        snapshot = df[df["timestep"] == t].copy()
        alive    = snapshot[snapshot["battery_voltage"] > BATTERY_CUTOFF_V]

        if first_death is None and len(alive) < total_nodes:
            first_death = t

        # Select CHs using the chosen protocol
        chs = select_cluster_heads(alive, protocol=protocol) if not alive.empty else []

        # Energy model:
        # - Each alive non-CH node pays intra-cluster tx cost
        # - Each CH pays aggregation + inter-cluster forwarding cost
        n_ch    = len(chs)
        n_member = max(0, len(alive) - n_ch)

        # Energy costs (simplified acoustic model, μJ per bit)
        tx_energy_member = n_member * 0.05   # short-range tx
        tx_energy_ch     = n_ch    * 0.20    # long-range forwarding
        energy_consumed  = tx_energy_member + tx_energy_ch

        # Packet delivery model
        if not alive.empty:
            avg_psr           = float(alive["packet_success_rt"].mean())
            packets_delivered = int(round(len(alive) * avg_psr))
            packets_lost      = len(alive) - packets_delivered
        else:
            avg_psr           = 0.0
            packets_delivered = 0
            packets_lost      = 0

        total = packets_delivered + packets_lost
        delivery_ratio = packets_delivered / total if total > 0 else 0.0

        records.append({
            "round_num"          : t,
            "protocol"           : protocol,
            "alive_nodes"        : len(alive),
            "total_nodes"        : total_nodes,
            "n_cluster_heads"    : n_ch,
            "avg_rul"            : float(alive["rul_hours"].mean()) if not alive.empty else 0.0,
            "avg_battery"        : float(alive["battery_voltage"].mean()) if not alive.empty else 0.0,
            "energy_consumed"    : round(energy_consumed, 4),
            "packets_delivered"  : packets_delivered,
            "packets_lost"       : packets_lost,
            "delivery_ratio"     : round(delivery_ratio, 4),
            "network_alive_ratio": round(len(alive) / total_nodes, 4),
            "first_node_death"   : first_death,
        })

    return pd.DataFrame(records)


# ── Protocol Summary ───────────────────────────────────────────────────────

def protocol_summary(rounds_df: pd.DataFrame) -> dict:
    """
    Compute high-level KPIs from a rounds DataFrame.
    Used for the benchmark comparison table (thesis Chapter 4).
    """
    last = rounds_df.iloc[-1]
    return {
        "protocol"            : rounds_df["protocol"].iloc[0],
        "network_lifetime"    : int(rounds_df[rounds_df["alive_nodes"] > 0]["round_num"].max()),
        "first_node_death"    : rounds_df["first_node_death"].dropna().iloc[0]
                                if rounds_df["first_node_death"].notna().any() else "N/A",
        "avg_delivery_ratio"  : round(float(rounds_df["delivery_ratio"].mean()), 4),
        "avg_energy_per_round": round(float(rounds_df["energy_consumed"].mean()), 4),
        "final_alive_nodes"   : int(last["alive_nodes"]),
        "avg_rul_final"       : round(float(last["avg_rul"]), 2),
    }
