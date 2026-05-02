# 🌊 AquaSense v 2.1.0 — Underwater Sensor Node Monitoring & IoUT Research Framework

[![CI](https://github.com/muhammad-hassaan-naeem/aquasense/actions/workflows/ci.yml/badge.svg)](https://github.com/muhammad-hassaan-naeem/aquasense/actions)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-111%20passed-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-62%25-yellow)](tests/)
[![Version](https://img.shields.io/badge/version-2.1.0-blue)](pyproject.toml)
[![Research](https://img.shields.io/badge/research-IoUT%20Routing-blueviolet)](docs/thesis_connection.md)

> **ML-powered battery RUL prediction, anomaly detection, depth-aware cluster head selection, and real ocean data integration for Internet of Underwater Things (IoUT) sensor networks — with a focus on Pakistan's Arabian Sea coastline.**

This repository is the simulation and research framework for the Master's thesis:

**"Energy-Efficient Depth-Aware Clustering-Based Routing Protocol for IoUT Using Depth Sensors"**

*Muhammad Hassaan Naeem*

---

## 🗺 Network Architecture

![IoUT Network Architecture](docs/network_architecture.png)

The system models a full IoUT deployment: **Underwater Sensor Nodes** organised into depth-based clusters, each with a **Cluster Head (CH)** that aggregates data and forwards it via multi-hop acoustic links toward the **Surface Sink** through a **Base Station**. The proposed protocol selects CHs using a composite fitness score that balances residual energy, depth position, and link quality — outperforming LEACH, DBR, and random baselines across all three metrics.

---

## ✨ Features

| Component | Description |
|---|---|
| **Data Simulation** | Realistic telemetry for 80+ nodes across depth clusters (shallow / mid / deep) |
| **RUL Regression** | Random Forest predicts remaining battery life — R² > 0.99 |
| **Temporal LSTM** | Windowed LSTM RUL predictor capturing battery degradation over time |
| **Anomaly Detection** | Isolation Forest flags malfunctioning nodes without labels |
| **Depth Clustering** | K-Means groups nodes by energy-efficiency profile |
| **CH Selection** | Proposed depth-aware + energy-aware fitness score algorithm |
| **Multi-Hop Routing** | deep → mid → shallow → surface sink acoustic relay path |
| **Protocol Benchmarks** | Proposed vs LEACH vs DBR vs Random — 4 figures + CSV |
| **Acoustic Energy Model** | Thorp's formula for depth-dependent signal absorption loss |
| **Real ARGO Data** | Live ocean float profiles from Ifremer ERDDAP / Argovis API |
| **NOAA Climatology** | WOA23 temperature & salinity profiles (Arabian Sea focus) |
| **RF vs LSTM Comparison** | Side-by-side benchmark with bootstrapped confidence intervals |
| **SQL Persistence** | SQLite (default) or PostgreSQL — zero code changes to switch |
| **8-Panel Dashboard** | Dark ocean-themed Matplotlib monitoring dashboard |
| **Phase 1 Dashboard** | Extended ARGO + LSTM comparison dashboard (4 extra panels) |
| **CI/CD** | GitHub Actions across Python 3.9 → 3.12, 111 tests, 62% coverage |

---

## 🗂 Project Structure

```
aquasense/
│
├── src/aquasense/                    ← Core Python package (v2.1.0)
│   ├── __init__.py
│   ├── config.py                     ← All constants (env-var overridable)
│   ├── simulate.py                   ← Synthetic sensor data generation
│   ├── database.py                   ← SQLite / PostgreSQL persistence
│   ├── models.py                     ← RULRegressor · AnomalyDetector · DepthClusterer
│   ├── visualise.py                  ← 8-panel monitoring dashboard
│   ├── pipeline.py                   ← CLI entry-point (--bench · --phase1 flags)
│   │
│   ├── research/                     ← Thesis research modules
│   │   ├── routing_protocol.py       ← Proposed CH selection + multi-hop routing
│   │   ├── energy_model.py           ← Thorp acoustic energy consumption model
│   │   └── benchmarks.py            ← 4-protocol comparison framework
│   │
│   └── phase1/                       ← Phase 1: Real Data + LSTM (new in v2.1)
│       ├── argo_connector.py         ← ARGO float connector (top-level API)
│       ├── lstm_model.py             ← Temporal RUL model (sklearn-compatible)
│       ├── comparison.py             ← RF vs LSTM comparison orchestrator
│       ├── pipeline.py               ← Phase 1 standalone CLI
│       ├── data/
│       │   ├── argo_connector.py     ← Full ARGO GDAC / Argovis connector
│       │   └── noaa_connector.py     ← NOAA WOA23 climatology connector
│       ├── models/
│       │   ├── lstm_rul.py           ← PyTorch LSTM + attention + early stopping
│       │   └── model_comparison.py   ← RF vs LSTM with bootstrapped CIs
│       └── viz/
│           └── phase1_dashboard.py   ← Extended Phase 1 dashboard
│
├── tests/
│   ├── test_aquasense.py             ← 58 core tests
│   ├── test_phase1.py                ← 33 Phase 1 tests
│   └── test_visualise.py             ← 20 dashboard tests
│
├── docs/
│   ├── network_architecture.png      ← IoUT network diagram
│   └── thesis_connection.md          ← Chapter → code mapping
│
├── results/                          ← Generated at runtime (git-ignored)
│   ├── figures/                      ← Benchmark & comparison plots (PNG)
│   └── metrics/                      ← protocol_comparison.csv · model_comparison.csv
│
├── outputs/                          ← Generated at runtime (git-ignored)
│   ├── aquasense_dashboard.png
│   └── phase1_dashboard.png
│
├── .github/workflows/ci.yml         ← GitHub Actions CI
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/muhammad-hassaan-naeem/aquasense.git
cd aquasense

# Install core (simulation, ML, dashboard, routing — no PyTorch)
pip install -e .

# Install with PyTorch for Phase 1 LSTM support
pip install -e ".[torch]"

# Run core pipeline — dashboard + monitoring
python -m aquasense.pipeline

# Run with routing protocol benchmarks (generates Chapter 4 thesis figures)
python -m aquasense.pipeline --bench

# Run Phase 1 — ARGO real data + LSTM RUL comparison (synthetic fallback if offline)
# Requires: pip install -e ".[torch]"
python -m aquasense.pipeline --phase1

# Run Phase 1 with live ARGO API fetch (requires internet connection)
python -m aquasense.pipeline --phase1 --argo-real

# Full pipeline — benchmarks + Phase 1 together
python -m aquasense.pipeline --nodes 80 --timesteps 100 --bench --phase1

# Phase 1 standalone CLI
python -m aquasense.phase1.pipeline
python -m aquasense.phase1.pipeline --n-floats 30 --lstm-epochs 60
python -m aquasense.phase1.pipeline --no-real-data   # synthetic only, no API calls
```

---

## 📊 Dashboard

The **8-panel monitoring dashboard** is saved to `outputs/aquasense_dashboard.png`:

| Panel | Content |
|---|---|
| 1 | Fleet KPIs — total nodes, avg RUL, anomaly count, avg battery, avg PSR |
| 2 | Cluster energy efficiency bar chart + anomaly node overlay |
| 3 | RUL prediction vs actual scatter (MAE · RMSE · R²) |
| 4 | Battery voltage decay by depth cluster (mean ± 1σ) |
| 5 | Depth vs RUL — coloured by cluster |
| 6 | Anomaly distribution per depth cluster |
| 7 | K-Means depth clusters — scatter with centroids |
| 8 | Feature importance for the RUL Random Forest |

The **Phase 1 extended dashboard** (`outputs/phase1_dashboard.png`) adds 4 more panels:

| Panel | Content |
|---|---|
| 9  | Real ARGO float data vs simulation — feature distribution overlay |
| 10 | ARGO depth profile — temperature & salinity plotted against depth |
| 11 | RF vs LSTM metric bar comparison (MAE · RMSE · MAPE) |
| 12 | LSTM training & validation loss curves per epoch |

---

## 🔬 Research Modules

### CH Selection — Proposed Protocol

```python
from aquasense.research.routing_protocol import (
    select_cluster_heads, build_routing_path, simulate_routing_rounds
)
from aquasense.simulate import simulate_sensor_data

df       = simulate_sensor_data()
snapshot = df[df["timestep"] == df["timestep"].max()]

chs  = select_cluster_heads(snapshot, protocol="Proposed")
path = build_routing_path(chs)

for ch in path:
    print(f"CH node={ch.node_id:3d}  depth={ch.depth_m:6.0f}m  "
          f"battery={ch.battery_voltage:.2f}V  fitness={ch.fitness_score:.3f}")
```

### CH Fitness Score

```
Fitness = 0.5 × E_norm  +  0.3 × D_norm  +  0.2 × L_norm

  E_norm = battery_voltage / max_battery    (higher residual energy → better CH)
  D_norm = 1 − depth_m / max_depth         (shallower → shorter hop to surface)
  L_norm = packet_success_rate             (link reliability)
```

### Protocol Benchmark Suite

```python
from aquasense.research.benchmarks import run_full_benchmark_suite
summary = run_full_benchmark_suite(df)
print(summary)
# → results/figures/  (4 PNG figures)
# → results/metrics/protocol_comparison.csv
```

### Acoustic Energy Model (Thorp 1967)

```python
from aquasense.research.energy_model import (
    tx_energy, path_loss, absorption_coefficient, estimate_round_energy
)

e     = tx_energy(distance_m=200)
print(f"TX energy:        {e*1e6:.4f} μJ")

alpha = absorption_coefficient(depth_m=300, frequency_khz=25)
print(f"Absorption @300m: {alpha:.4f} dB/km")
```

---

## 🌍 Phase 1 — Real Ocean Data + LSTM

Phase 1 validates the simulation against real oceanographic measurements and extends RUL prediction with a temporal LSTM that captures degradation trends across consecutive timesteps — something snapshot-based Random Forest cannot do.

> **Requires PyTorch:** `pip install -e ".[torch]"` or `pip install -e ".[phase1]"`

### ARGO Float Data

```python
from aquasense.phase1.data.argo_connector import ArgoConnector
from aquasense.simulate import simulate_sensor_data

conn   = ArgoConnector()
df     = conn.load_or_fetch(n_floats=20, region="arabian_sea")
conn.validate_schema(df)                         # verifies all AquaSense columns

sim_df     = simulate_sensor_data()
comparison = conn.compare_with_simulation(df, sim_df)
print(comparison)
```

### NOAA WOA23 Climatology

```python
from aquasense.phase1.data.noaa_connector import NOAAConnector

noaa  = NOAAConnector()
df    = noaa.fetch_climatology(region="arabian_sea", n_profiles=50, max_depth=500)
stats = noaa.basin_statistics()
print(noaa.pakistan_ocean_summary())
```

### LSTM RUL Predictor

```python
from aquasense.phase1.models.lstm_rul import LSTMRULPredictor

model = LSTMRULPredictor(seq_len=10, hidden_size=64, epochs=50)
model.fit(df)
print(model)   # LSTMRULPredictor(MAE=28.3h, R²=0.9961, seq_len=10)

model.save("outputs/lstm_rul.pt")
model = LSTMRULPredictor.load("outputs/lstm_rul.pt")
```

### RF vs LSTM Comparison

```python
from aquasense.phase1.models.model_comparison import ModelComparison

comp    = ModelComparison(lstm_seq_len=10, lstm_epochs=50)
results = comp.run(df_synthetic=df)
comp.print_report(results)
comp.plot_comparison()
# → results/figures/model_comparison_rf_vs_lstm.png
# → results/metrics/model_comparison.csv
```

---

## 📈 Benchmark Protocols

| Protocol | w_energy | w_depth | w_link | Description |
|---|---|---|---|---|
| Random | 0.33 | 0.33 | 0.34 | Equal-weight baseline |
| LEACH | 1.00 | 0.00 | 0.00 | Energy-only CH selection |
| DBR | 0.00 | 1.00 | 0.00 | Depth-only (Depth-Based Routing) |
| **Proposed** | **0.50** | **0.30** | **0.20** | **This research — balanced fitness** |

Generated output figures:

| File | Content |
|---|---|
| `alive_nodes_comparison.png` | Network lifetime — nodes alive per routing round |
| `energy_consumption_comparison.png` | Cumulative energy consumed per protocol |
| `delivery_ratio_comparison.png` | Packet delivery ratio over simulation rounds |
| `ch_fitness_distribution.png` | CH fitness score box-plot by depth cluster |

---

## 🇵🇰 Real-World Use Cases — Pakistan

Pakistan has a **1,046 km Arabian Sea coastline** spanning Karachi, Balochistan, and the Makran coast, with growing maritime, environmental, and defence interests. AquaSense's IoUT framework directly addresses several nationally relevant challenges.

### 1. 🐟 Fisheries & Fish Stock Monitoring
Pakistan's fishing industry, centred in Karachi and Gwadar, employs over 400,000 people. AquaSense sensor nodes can monitor ocean temperature, salinity, and dissolved oxygen at depth — data that directly maps to fish migration patterns and breeding zones. The NOAA WOA23 integration already models Arabian Sea climatology. Real deployments could give the **Pakistan Fisheries Development Board** near-real-time stock health data, reducing overfishing risk and supporting sustainable catch quotas.

### 2. 🌊 Coastal Erosion & Sediment Tracking
The Makran Subduction Zone makes Pakistan's coastline geologically active. Underwater sensor arrays using AquaSense's depth-cluster architecture could continuously track sediment transport, seafloor pressure changes, and current velocity near the Indus Delta — giving **SUPARCO** and provincial environmental agencies actionable early data before erosion events become irreversible.

### 3. 🌡 Climate Change & Sea-Level Rise Research
Pakistan is one of the world's most climate-vulnerable nations. Long-term IoUT deployments using AquaSense's battery-efficient routing protocol (which extends network lifetime by minimising energy waste in CH selection) would enable **IUCN Pakistan** and research universities like NED and NUST to maintain persistent ocean sensor arrays — years of uninterrupted temperature, pressure, and salinity profiles that feed national climate models.

### 4. 🛢 Offshore Energy — Gwadar & CPEC Maritime Corridor
The China–Pakistan Economic Corridor (CPEC) includes significant offshore infrastructure planning around Gwadar deep-sea port. AquaSense's anomaly detection and node health monitoring framework could underpin pipeline integrity monitoring and early warning for offshore oil & gas assets, reducing inspection costs and improving safety for the **Pakistan Petroleum Exploration** sector.

### 5. 🔔 Tsunami & Earthquake Early Warning (Makran Subduction Zone)
The Makran coast sits on a seismically active subduction zone — the source of the devastating 1945 Makran tsunami. An IoUT sensor array applying AquaSense's multi-hop acoustic routing could relay seafloor pressure anomalies to surface buoys within seconds, supporting **Pakistan Meteorological Department (PMD)** and the **National Disaster Management Authority (NDMA)** in building a domestic tsunami early warning system.

### 6. 🚢 Port Security & Submarine Cable Protection
Pakistan's submarine cable landings (SEACOM, IMEWE, PEACE Cable) near Karachi are strategic national infrastructure. The depth-aware sensor clustering and acoustic energy model in AquaSense provide a research baseline for intrusion detection networks that monitor cable corridors — relevant to both **Pakistan Navy** and the **Pakistan Telecommunication Authority**.

### 7. 🎓 Academic Research — NED, NUST, COMSATS, UET
AquaSense ships as a fully reproducible simulation environment: one `pip install` gives any research group access to synthetic IoUT telemetry, four routing protocols, ARGO real data, and an LSTM predictor. Pakistani universities can build on this codebase for thesis work, journal publications, and funding proposals in IoT, underwater communications, and ML-based sensor systems — areas where national research output is growing rapidly.

---

## 📚 Research — Code Mapping

| Research Chapter | Module | Key Function |
|---|---|---|
| Ch. 1 — IoUT Architecture | `simulate.py` | `simulate_sensor_data()` |
| Ch. 2 — Clustering Review | `models.py` | `DepthClusterer` |
| Ch. 3 — Proposed Protocol | `research/routing_protocol.py` | `compute_ch_fitness()`, `select_cluster_heads()` |
| Ch. 3 — Energy Model | `research/energy_model.py` | `tx_energy()`, `path_loss()` |
| Ch. 4 — Experiments | `research/benchmarks.py` | `run_full_benchmark_suite()` |
| Ch. 4 — RUL Prediction | `models.py` | `RULRegressor` |
| Ch. 4 — Anomaly Detection | `models.py` | `AnomalyDetector` |
| Ch. 4 — LSTM Comparison | `phase1/models/lstm_rul.py` | `LSTMRULPredictor` |
| Ch. 4 — Real Data Validation | `phase1/data/argo_connector.py` | `ArgoConnector` |

See [`docs/research_connection.md`](docs/research_connection.md) for the full chapter-by-chapter breakdown.

---

## 🧪 Tests

```bash
# Core tests (no PyTorch needed)
pip install -e ".[dev]"
pytest tests/ -v --cov=aquasense

# Full test suite including LSTM tests
pip install -e ".[dev,torch]"
pytest tests/ -v --cov=aquasense
```

**111 tests, 111 passed** — 62% coverage.

| File | Tests | What is covered |
|---|---|---|
| `test_aquasense.py` | 58 | Simulation, database, RUL, anomaly, clustering, routing, energy |
| `test_phase1.py` | 33 | ARGO connector, NOAA connector, LSTM, model comparison, pipeline |
| `test_visualise.py` | 20 | All 8 dashboard panels, `build_dashboard()`, palette & config |

Test classes:

- `TestSimulate` (9) — shape, dtypes, value ranges, reproducibility, anomaly rate
- `TestDatabase` (4) — write/read, optimised queries, critical node alerts
- `TestRULRegressor` (6) — fit, predict, MAE/R², save/load
- `TestAnomalyDetector` (4) — fit, predict, tag, save/load
- `TestDepthClusterer` (3) — fit, predict, cluster summary
- `TestRoutingProtocol` (8) — fitness score, CH selection, all 4 protocols, routing path
- `TestEnergyModel` (5) — absorption coefficient, TX energy, round estimation
- `TestArgoConnector` (11) — schema validation, synthetic fallback, cache, thermocline, halocline
- `TestNOAAConnector` (5) — climatology fetch, depth filter (cache-safe), basin statistics
- `TestLSTMRULPredictor` (9) — fit, predict, save/load, training history
- `TestModelComparison` (6) — both models, CSV output, comparison figure
- `TestPhase1Pipeline` (2) — end-to-end integration in synthetic mode
- `TestPanelFunctions` (10) — individual dashboard panel helpers
- `TestBuildDashboard` (4) — file creation, return type, default path, edge cases
- `TestPaletteConfig` (6) — hex colours, cluster colour coverage, DPI

---

## ⚙️ Configuration

All defaults live in `src/aquasense/config.py` and can be overridden with environment variables — no code changes needed:

| Variable | Default | Description |
|---|---|---|
| `AQUASENSE_DB` | `outputs/sensor_logs.db` | SQLite database path |
| `AQUASENSE_DB_URL` | *(unset)* | PostgreSQL DSN — set this to switch from SQLite |
| `AQUASENSE_N_NODES` | `80` | Simulated sensor nodes |
| `AQUASENSE_N_TIMESTEPS` | `100` | Timesteps per node |
| `AQUASENSE_SEED` | `42` | Global random seed |
| `AQUASENSE_ANOMALY_RATE` | `0.08` | Injected fault rate (8 %) |
| `CH_W_ENERGY` | `0.50` | CH fitness — energy weight |
| `CH_W_DEPTH` | `0.30` | CH fitness — depth weight |
| `CH_W_LINK` | `0.20` | CH fitness — link quality weight |
| `AQUASENSE_LSTM_WINDOW` | `10` | LSTM sliding window length |
| `AQUASENSE_ARGO_FLOATS` | `30` | ARGO floats to download |
| `AQUASENSE_ARGO_CACHE` | `true` | Cache ARGO data locally |

**Switch to PostgreSQL:**

```bash
export AQUASENSE_DB_URL=postgresql://user:password@localhost:5432/aquasense
python -m aquasense.pipeline --bench
```

---

## 🗄 Database Schema

```sql
CREATE TABLE sensor_logs (
    id                INTEGER   PRIMARY KEY AUTOINCREMENT,
    node_id           INTEGER   NOT NULL,
    timestep          INTEGER   NOT NULL,
    depth_m           REAL      NOT NULL,
    pressure_bar      REAL      NOT NULL,
    salinity_ppt      REAL      NOT NULL,
    temperature_c     REAL      NOT NULL,
    battery_voltage   REAL      NOT NULL,
    tx_freq_ppm       REAL      NOT NULL,
    packet_success_rt REAL      NOT NULL,
    depth_cluster     TEXT      NOT NULL,          -- 'shallow' | 'mid' | 'deep'
    is_anomaly        INTEGER   NOT NULL DEFAULT 0,
    rul_hours         REAL      NOT NULL,
    inserted_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Indexes on: node_id · timestep · depth_cluster · is_anomaly
```

---

## 📦 Installation Options

```bash
pip install -e .                   # Core — simulation, ML, dashboard, routing (no PyTorch)
pip install -e ".[torch]"          # + PyTorch (required for LSTM / Phase 1)
pip install -e ".[phase1]"         # Alias for [torch] — full Phase 1 LSTM stack
pip install -e ".[postgres]"       # + PostgreSQL (psycopg2-binary)
pip install -e ".[netcdf]"         # + NetCDF4 (raw ARGO .nc file support)
pip install -e ".[dev]"            # + pytest, pytest-cov (core tests, no PyTorch)
pip install -e ".[dev,torch]"      # + pytest + PyTorch (full test suite)
```

> **Why is PyTorch optional?** PyTorch pre-built wheels are unavailable for Python 3.9 on PyPI, which caused CI failures across the entire matrix. Moving it to an optional extra keeps `pip install -e .` lightweight and fast for all Python versions, while users who need the LSTM predictor can opt in explicitly with `.[torch]`.

---

## 📄 License

MIT © Muhammad Hassaan Naeem. See [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

- [scikit-learn](https://scikit-learn.org/) — Random Forest, Isolation Forest, K-Means
- [PyTorch](https://pytorch.org/) — LSTM RUL predictor with attention mechanism
- [Matplotlib](https://matplotlib.org/) — Ocean-themed dashboards and comparison figures
- [Pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/) — Data simulation pipeline
- [ARGO Programme](https://argo.ucsd.edu/) — Real ocean float data (Ifremer ERDDAP / Argovis)
- [NOAA WOA23](https://www.ncei.noaa.gov/products/world-ocean-atlas) — Arabian Sea climatology
- Thorp (1967) — Acoustic absorption coefficient formula underpinning the IoUT energy model
