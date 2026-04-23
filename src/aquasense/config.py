"""
config.py
---------
Central configuration for AquaSense.
All tuneable constants live here — override via environment variables.
"""

from __future__ import annotations
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parents[2]
OUTPUT_DIR  = ROOT_DIR / "outputs"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
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
CLUSTER_PROBS   = {"shallow": 0.35, "mid": 0.40, "deep": 0.25}
BATTERY_RANGE   = {"shallow": (3.8, 4.2), "mid": (3.4, 4.0), "deep": (3.0, 3.8)}
TX_FREQ_RANGE   = {"shallow": (0.5, 2.0), "mid": (0.3, 1.5), "deep": (0.1, 0.8)}
BATTERY_CUTOFF_V = 2.5

# ── ML ─────────────────────────────────────────────────────────────────────
FEATURES = [
    "depth_m", "pressure_bar", "salinity_ppt", "temperature_c",
    "battery_voltage", "tx_freq_ppm", "packet_success_rt",
]
RF_N_ESTIMATORS  = int(os.getenv("AQUASENSE_RF_TREES", "150"))
RF_MAX_DEPTH     = int(os.getenv("AQUASENSE_RF_DEPTH", "12"))
RF_TEST_SIZE     = 0.2
IF_CONTAMINATION = SIM_ANOMALY_RATE
KM_N_CLUSTERS    = 3

# ── Routing Protocol ───────────────────────────────────────────────────────
# Weight coefficients for CH fitness score (must sum to 1.0)
CH_WEIGHT_ENERGY = float(os.getenv("CH_W_ENERGY", "0.50"))
CH_WEIGHT_DEPTH  = float(os.getenv("CH_W_DEPTH",  "0.30"))
CH_WEIGHT_LINK   = float(os.getenv("CH_W_LINK",   "0.20"))

# Rotation threshold: re-elect CH when its battery drops below this fraction
CH_ROTATION_THRESHOLD = 0.30

# ── Benchmark Protocols ────────────────────────────────────────────────────
BENCHMARK_PROTOCOLS = {
    "Random":          {"energy_w": 0.33, "depth_w": 0.33, "link_w": 0.34},
    "LEACH (Energy)":  {"energy_w": 1.00, "depth_w": 0.00, "link_w": 0.00},
    "DBR (Depth)":     {"energy_w": 0.00, "depth_w": 1.00, "link_w": 0.00},
    "Proposed":        {"energy_w": CH_WEIGHT_ENERGY,
                        "depth_w":  CH_WEIGHT_DEPTH,
                        "link_w":   CH_WEIGHT_LINK},
}

# ── Visualisation ──────────────────────────────────────────────────────────
DASHBOARD_DPI  = 160
DASHBOARD_SIZE = (22, 18)

PALETTE = {
    "bg":      "#03090f", "panel":   "#071828", "border":  "#0a3050",
    "accent1": "#00c8ff", "accent2": "#00ffd5", "accent3": "#ff6b35",
    "warn":    "#ff3860", "text":    "#c8e8f8", "muted":   "#4a7a9b",
    "shallow": "#00c8ff", "mid":     "#00ffd5", "deep":    "#7b5ea7",
}
CLUSTER_COLORS = {
    "shallow": PALETTE["shallow"],
    "mid":     PALETTE["mid"],
    "deep":    PALETTE["deep"],
}
KM_COLORS = ["#00c8ff", "#00ffd5", "#ff6b35"]

# ── Phase 1 ────────────────────────────────────────────────────────────────
LSTM_WINDOW_SIZE   = int(os.getenv("AQUASENSE_LSTM_WINDOW", "10"))
LSTM_N_ESTIMATORS  = int(os.getenv("AQUASENSE_LSTM_TREES",  "200"))
ARGO_N_FLOATS      = int(os.getenv("AQUASENSE_ARGO_FLOATS", "30"))
ARGO_USE_CACHE     = os.getenv("AQUASENSE_ARGO_CACHE", "true").lower() == "true"
