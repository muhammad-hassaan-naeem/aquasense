"""
config.py
---------
Central configuration for AquaSense.
All tuneable constants live here so nothing is buried in business logic.
Override via environment variables or by editing this file before running.
"""

from __future__ import annotations
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parents[2]   # repo root
OUTPUT_DIR  = ROOT_DIR / "outputs"
DB_PATH     = Path(os.getenv("AQUASENSE_DB", str(ROOT_DIR / "outputs" / "sensor_logs.db")))

# ── Simulation ─────────────────────────────────────────────────────────────
SIM_N_NODES      = int(os.getenv("AQUASENSE_N_NODES",      "80"))
SIM_N_TIMESTEPS  = int(os.getenv("AQUASENSE_N_TIMESTEPS",  "100"))
SIM_RANDOM_SEED  = int(os.getenv("AQUASENSE_SEED",         "42"))
SIM_ANOMALY_RATE = float(os.getenv("AQUASENSE_ANOMALY_RATE", "0.08"))

# Depth-cluster boundaries (metres)
CLUSTER_BOUNDS = {
    "shallow": (5,   60),
    "mid":     (60,  300),
    "deep":    (300, 1000),
}
# Probability of each cluster being assigned
CLUSTER_PROBS = {"shallow": 0.35, "mid": 0.40, "deep": 0.25}

# Battery initial-voltage ranges per cluster (V)
BATTERY_RANGE = {
    "shallow": (3.8, 4.2),
    "mid":     (3.4, 4.0),
    "deep":    (3.0, 3.8),
}

# Transmission frequency ranges per cluster (packets / min)
TX_FREQ_RANGE = {
    "shallow": (0.5, 2.0),
    "mid":     (0.3, 1.5),
    "deep":    (0.1, 0.8),
}

# Battery cut-off voltage (V) — node considered dead below this
BATTERY_CUTOFF_V = 2.5

# ── ML ─────────────────────────────────────────────────────────────────────
FEATURES = [
    "depth_m",
    "pressure_bar",
    "salinity_ppt",
    "temperature_c",
    "battery_voltage",
    "tx_freq_ppm",
    "packet_success_rt",
]

# Random Forest hyperparameters
RF_N_ESTIMATORS = int(os.getenv("AQUASENSE_RF_TREES",  "150"))
RF_MAX_DEPTH    = int(os.getenv("AQUASENSE_RF_DEPTH",  "12"))
RF_TEST_SIZE    = 0.2

# Isolation Forest
IF_CONTAMINATION = SIM_ANOMALY_RATE

# K-Means
KM_N_CLUSTERS = 3

# ── Visualisation ──────────────────────────────────────────────────────────
DASHBOARD_DPI  = 160
DASHBOARD_SIZE = (22, 18)          # inches

PALETTE = {
    "bg"      : "#03090f",
    "panel"   : "#071828",
    "border"  : "#0a3050",
    "accent1" : "#00c8ff",
    "accent2" : "#00ffd5",
    "accent3" : "#ff6b35",
    "warn"    : "#ff3860",
    "text"    : "#c8e8f8",
    "muted"   : "#4a7a9b",
    "shallow" : "#00c8ff",
    "mid"     : "#00ffd5",
    "deep"    : "#7b5ea7",
}
CLUSTER_COLORS = {
    "shallow": PALETTE["shallow"],
    "mid"    : PALETTE["mid"],
    "deep"   : PALETTE["deep"],
}
KM_COLORS = ["#00c8ff", "#00ffd5", "#ff6b35"]
