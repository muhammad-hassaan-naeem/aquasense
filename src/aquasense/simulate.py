"""
simulate.py
-----------
Generates synthetic underwater sensor-node telemetry.

Each simulated node is placed in a depth cluster (shallow / mid / deep)
and produces a time-series of readings that mimic realistic battery drain,
pressure effects, salinity variation, and packet-loss patterns.

Approximately ``ANOMALY_RATE`` of rows have injected faults (sudden voltage
drop, degraded packet success rate) so the anomaly detector has signal to
learn from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    BATTERY_CUTOFF_V,
    BATTERY_RANGE,
    CLUSTER_BOUNDS,
    CLUSTER_PROBS,
    SIM_ANOMALY_RATE,
    SIM_N_NODES,
    SIM_N_TIMESTEPS,
    SIM_RANDOM_SEED,
    TX_FREQ_RANGE,
)


def _assign_cluster(rng: np.random.Generator) -> str:
    """Draw a depth cluster label using the configured probability weights."""
    clusters = list(CLUSTER_PROBS.keys())
    probs    = list(CLUSTER_PROBS.values())
    return rng.choice(clusters, p=probs)


def _node_params(cluster: str, rng: np.random.Generator) -> tuple[float, float, float]:
    """
    Return (base_depth_m, initial_battery_v, tx_freq_ppm) for one node.
    All values are drawn uniformly within the cluster's defined ranges.
    """
    depth_lo,   depth_hi   = CLUSTER_BOUNDS[cluster]
    batt_lo,    batt_hi    = BATTERY_RANGE[cluster]
    tx_lo,      tx_hi      = TX_FREQ_RANGE[cluster]

    base_depth   = rng.uniform(depth_lo,  depth_hi)
    base_battery = rng.uniform(batt_lo,   batt_hi)
    tx_freq      = rng.uniform(tx_lo,     tx_hi)
    return base_depth, base_battery, tx_freq


def simulate_sensor_data(
    n_nodes:     int   = SIM_N_NODES,
    n_timesteps: int   = SIM_N_TIMESTEPS,
    random_seed: int   = SIM_RANDOM_SEED,
    anomaly_rate: float = SIM_ANOMALY_RATE,
) -> pd.DataFrame:
    """
    Simulate a fleet of *n_nodes* underwater sensor nodes over *n_timesteps*
    measurement cycles.

    Parameters
    ----------
    n_nodes : int
        Number of distinct sensor nodes to simulate.
    n_timesteps : int
        Number of time-steps per node.
    random_seed : int
        NumPy random seed for reproducibility.
    anomaly_rate : float
        Fraction of readings that receive an injected fault (0–1).

    Returns
    -------
    pd.DataFrame
        One row per (node_id, timestep) observation with columns:

        ========================  ============================================
        node_id                   Unique integer node identifier
        timestep                  Time index (0-based)
        depth_m                   Water depth in metres
        pressure_bar              Hydrostatic pressure in bar
        salinity_ppt              Salinity in parts per thousand
        temperature_c             Water temperature in °C
        battery_voltage           Instantaneous battery voltage (V)
        tx_freq_ppm               Transmission frequency (packets / min)
        packet_success_rt         Fraction of packets successfully received
        depth_cluster             Categorical cluster label
        is_anomaly                Ground-truth anomaly flag (0/1)
        rul_hours                 Remaining Useful Life in hours
        ========================  ============================================
    """
    rng     = np.random.default_rng(random_seed)
    records = []

    for node_id in range(n_nodes):
        cluster                      = _assign_cluster(rng)
        base_depth, battery, tx_freq = _node_params(cluster, rng)

        for t in range(n_timesteps):
            # Environmental readings — depth stays near the node's fixed position
            depth      = base_depth + rng.normal(0, 0.5)
            pressure   = base_depth * 0.1 + rng.normal(0, 0.3)
            salinity   = 33 + base_depth * 0.002 + rng.normal(0, 0.5)
            temperature = 20 - base_depth * 0.01 + rng.normal(0, 0.4)

            # Battery drain model:
            #   • higher tx_freq → more radio activity
            #   • higher pressure → sealing stress increases current draw
            #   • salinity-induced corrosion slightly accelerates drain
            drain = (
                0.003 * tx_freq
                + 0.0005 * pressure
                + 0.0002 * salinity
                + rng.normal(0, 0.0005)
            )
            drain   = max(drain, 1e-6)                      # never negative
            battery = max(BATTERY_CUTOFF_V, battery - drain)

            # Packet success rate degrades with depth and low battery
            psr = float(np.clip(
                1.0
                - 0.0003 * base_depth
                - 0.2 * max(0.0, 3.5 - battery)
                + rng.normal(0, 0.05),
                0.0, 1.0,
            ))

            # Remaining Useful Life: hours of energy left above cut-off
            rul = max(0.0, (battery - BATTERY_CUTOFF_V) / drain * 60)

            # ── Fault injection ────────────────────────────────────────────
            is_anomaly = 0
            if rng.random() < anomaly_rate:
                battery    = battery    * rng.uniform(0.50, 0.85)
                psr        = psr        * rng.uniform(0.10, 0.50)
                rul        = rul        * rng.uniform(0.00, 0.40)
                is_anomaly = 1

            records.append({
                "node_id"          : node_id,
                "timestep"         : t,
                "depth_m"          : round(depth,       2),
                "pressure_bar"     : round(pressure,    3),
                "salinity_ppt"     : round(salinity,    3),
                "temperature_c"    : round(temperature, 3),
                "battery_voltage"  : round(battery,     4),
                "tx_freq_ppm"      : round(tx_freq,     3),
                "packet_success_rt": round(psr,         4),
                "depth_cluster"    : cluster,
                "is_anomaly"       : is_anomaly,
                "rul_hours"        : round(rul,         2),
            })

    return pd.DataFrame(records)
