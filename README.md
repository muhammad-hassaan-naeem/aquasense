# 🌊 AquaSense v2.1.0

[![CI](https://github.com/muhammad-hassaan-naeem/aquasense/actions/workflows/ci.yml/badge.svg)](https://github.com/muhammad-hassaan-naeem/aquasense/actions)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-91%20passed-brightgreen)](tests/)
[![Version](https://img.shields.io/badge/version-2.1.0-blueviolet)](pyproject.toml)

> **ML-powered battery life prediction, anomaly detection, and energy-efficient routing for Internet of Underwater Things (IoUT) sensor networks — with real ARGO ocean data integration and temporal deep learning models.**

---

## 🔍 What Is AquaSense?

AquaSense is a complete Python simulation and research framework for monitoring underwater IoT sensor networks. It addresses one of the core engineering challenges in ocean technology: **sensors deployed deep underwater cannot be retrieved easily, their batteries cannot be recharged, and when they fail silently you lose critical data permanently.**

AquaSense solves three problems using machine learning:

1. **When will this sensor battery die?** — Random Forest and Temporal (LSTM-equivalent) models predict Remaining Useful Life in hours with R² > 0.99
2. **Is this sensor malfunctioning?** — Isolation Forest detects anomalous nodes without any labelled fault data (fully unsupervised)
3. **How do we route data from the seabed to the surface efficiently?** — A proposed depth-aware Cluster Head selection algorithm minimises acoustic energy consumption across multi-hop routing paths

---

## 🌐 Research and Real-World Use Cases

### 🌊 Ocean and Climate Science
Long-duration ocean sensor deployments need to know when to schedule float replacements before data gaps occur. AquaSense's RUL predictor gives scientists precise estimates of remaining battery life. The anomaly detector catches sensors reporting physically impossible chemistry values. Depth-based clustering groups nodes by oceanographic zone for targeted analysis across ocean basins.

**Applicable to:** deep-sea climate monitoring, ocean acidification tracking, thermohaline circulation studies, ARGO float lifecycle management, sea surface temperature monitoring

### 🚨 Tsunami and Earthquake Early Warning
Pressure sensors on the seabed detect the rapid multi-node pressure wave signatures that precede tsunamis. AquaSense's anomaly detector can flag these patterns in real time. The routing protocol ensures critical alerts travel to the surface station via the most energy-efficient and reliable acoustic path, maximising the window available for coastal evacuation.

**Applicable to:** seabed pressure monitoring arrays, submarine earthquake detection, coastal early warning systems

### 🛢️ Industrial Subsea Infrastructure
Sensors along undersea pipelines detect pressure drops indicating leaks and electrochemical anomalies indicating corrosion — weeks before a failure event. The energy-efficient routing protocol extends deployment lifetimes from months to years, dramatically reducing the cost and risk of maintenance dives.

**Applicable to:** offshore oil and gas pipeline monitoring, subsea power cable health, offshore wind farm structural monitoring, port and harbour surveillance

### 🐠 Marine Conservation and Ecology
Large marine protected areas require continuous, long-duration sensor coverage. The proposed routing protocol directly extends the operational lifetime of monitoring networks. Temperature and salinity anomaly detection can predict coral bleaching events in advance, giving conservation teams time to respond.

**Applicable to:** coral reef health monitoring, marine protected area surveillance, fish stock habitat tracking, seagrass ecosystem monitoring

### 🔬 Academic and Engineering Research
AquaSense provides a complete, reproducible benchmark environment for IoUT protocol research. New routing algorithms can be compared against established baselines (LEACH, DBR, Random) on identical simulation data. The model comparison framework benchmarks Random Forest against Temporal models on identical degradation datasets. Real ARGO and NOAA data let researchers validate synthetic simulations against measured ocean conditions.

**Applicable to:** routing protocol comparison studies, ML degradation model benchmarking, synthetic-to-real data validation, acoustic energy modelling research

---

## 🇵🇰 Use Cases for Pakistan

Pakistan has a 1,046 km coastline along the Arabian Sea, significant river systems including the Indus, and growing maritime infrastructure under CPEC. AquaSense is directly applicable to several national priorities:

### Makran Coast Tsunami Warning
The Makran Subduction Zone off the Balochistan coast is a confirmed tsunami source — the 1945 Makran tsunami caused widespread destruction. A seabed pressure sensor network using AquaSense's routing and anomaly detection could provide 20–45 minutes of advance warning for coastal communities in Gwadar, Ormara, and Pasni.

### Karachi and Gwadar Port Pollution Monitoring
As one of South Asia's busiest ports and a growing CPEC oil terminal, Karachi and Gwadar face underwater pollution risk from shipping traffic and industrial effluent. An AquaSense-powered sensor network can detect oil spill signatures and chemical contamination in real time, alerting port authorities before pollution reaches fishing grounds or beaches.

### Indus River Flood and Water Quality Monitoring
The 2022 Pakistan floods caused over $30 billion in damage and affected 33 million people. Upstream IoUT-style sensor nodes using AquaSense's RUL prediction would remain operational through monsoon season, providing 6–24 hours of advance flood warning to downstream cities. The same sensor architecture also monitors industrial effluent from textile mills entering the Indus system.

### Tarbela and Mangla Dam Reservoir Health
Pakistan's largest dams accumulate sedimentation that reduces storage capacity and threatens structural integrity. AquaSense can monitor pressure, temperature, and turbidity sensors deployed in reservoir depths that are impractical to inspect manually, predicting sensor replacement schedules and flagging anomalous readings.

### Arabian Sea Fisheries Management
Over 300,000 fishing families depend on the Arabian Sea fishery. Ocean temperature and salinity monitoring using AquaSense-connected ARGO-style floats can predict seasonal fish stock migrations, helping fishing communities plan and supporting sustainable catch management.

---

## ✨ Features

| Component | Description |
|---|---|
| **Data Simulation** | Physics-based synthetic sensor data for 80+ nodes across depth zones |
| **ARGO Connector** | Fetches real profiles from 4,000 global ARGO profiling floats |
| **NOAA Connector** | Downloads World Ocean Atlas climatological profiles |
| **RUL Regression** | Random Forest predicts remaining battery life — R² > 0.99 |
| **Temporal Model** | Windowed Gradient Boosting captures battery drain trends over time |
| **Model Comparison** | Side-by-side RF vs Temporal benchmark with 6 publication-quality figures |
| **Anomaly Detection** | Isolation Forest flags malfunctioning nodes — no labels needed |
| **Depth Clustering** | K-Means discovers natural energy-efficiency groupings |
| **CH Selection** | Proposed depth-aware and energy-aware Cluster Head fitness score |
| **Multi-Hop Routing** | deep → mid → shallow → surface sink path |
| **Protocol Benchmarks** | Compare Proposed vs LEACH vs DBR vs Random |
| **Acoustic Energy Model** | Thorp's formula for depth-dependent signal absorption |
| **SQL Persistence** | SQLite by default, PostgreSQL via one environment variable |
| **8-Panel Dashboard** | Dark ocean-themed Matplotlib monitoring dashboard |
| **Phase 1 Dashboard** | Extended ARGO validation and model comparison dashboard |
| **CI/CD** | GitHub Actions across Python 3.9–3.12 with 91 automated tests |

---

## 🗂 Project Structure

```
aquasense/
│
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── docs/
│   └── network_architecture.png
│
├── src/
│   └── aquasense/
│       ├── __init__.py              # Package metadata v2.1.0
│       ├── config.py                # All constants, env-var overridable
│       ├── simulate.py              # Physics-based synthetic sensor data
│       ├── database.py              # SQLite / PostgreSQL persistence
│       ├── models.py                # RULRegressor · AnomalyDetector · DepthClusterer
│       ├── visualise.py             # 8-panel monitoring dashboard
│       ├── pipeline.py              # Main CLI entry-point
│       │
│       ├── research/
│       │   ├── __init__.py
│       │   ├── routing_protocol.py  # CH selection + multi-hop routing
│       │   ├── energy_model.py      # Thorp acoustic energy model
│       │   └── benchmarks.py        # 4-protocol comparison suite
│       │
│       └── phase1/
│           ├── __init__.py
│           ├── argo_connector.py    # Simple ARGO adapter
│           ├── lstm_model.py        # Temporal (windowed) RUL model
│           ├── comparison.py        # RF vs Temporal comparison
│           ├── pipeline.py          # Phase 1 CLI
│           │
│           ├── data/
│           │   ├── __init__.py
│           │   ├── argo_connector.py    # Full ARGO GDAC connector
│           │   └── noaa_connector.py    # NOAA WOA climatology connector
│           │
│           ├── models/
│           │   ├── __init__.py
│           │   ├── lstm_rul.py          # Full LSTM predictor (PyTorch)
│           │   └── model_comparison.py  # Complete RF vs LSTM benchmark
│           │
│           └── viz/
│               ├── __init__.py
│               └── phase1_dashboard.py  # Extended Phase 1 dashboard
│
├── tests/
│   ├── __init__.py
│   ├── test_aquasense.py            # 58 core tests
│   ├── test_phase1.py               # 33 Phase 1 tests
│   └── test_visualise.py            # Dashboard rendering tests
│
└── results/                         # Generated at runtime (git-ignored)
    ├── figures/                     # All comparison PNG figures
    └── metrics/                     # CSV metrics tables
```

---

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/muhammad-hassaan-naeem/aquasense.git
cd aquasense
pip install -e .
pip install seaborn
```

### Run the core pipeline

```bash
# Default run — 80 nodes, 100 timesteps, dashboard saved to outputs/
python -m aquasense.pipeline

# With routing protocol benchmarks
python -m aquasense.pipeline --bench

# With Phase 1 — ARGO data + RF vs Temporal model comparison
python -m aquasense.pipeline --phase1

# Full run — everything
python -m aquasense.pipeline --bench --phase1

# Live ARGO float data from Ifremer (requires internet)
python -m aquasense.pipeline --phase1 --argo-real

# Large fleet simulation
python -m aquasense.pipeline --nodes 200 --timesteps 200 --bench --phase1
```

### Run the Phase 1 pipeline standalone

```bash
python -m aquasense.phase1.pipeline
python -m aquasense.phase1.pipeline --no-real-data
python -m aquasense.phase1.pipeline --n-floats 50
```

---

## 📊 Generated Outputs

After `python -m aquasense.pipeline --bench --phase1`:

| File | Location | Description |
|---|---|---|
| `aquasense_dashboard.png` | `outputs/` | 8-panel fleet monitoring dashboard |
| `alive_nodes_comparison.png` | `results/figures/` | Network lifetime — all 4 protocols |
| `energy_consumption_comparison.png` | `results/figures/` | Cumulative energy per protocol |
| `delivery_ratio_comparison.png` | `results/figures/` | Packet delivery ratio over rounds |
| `ch_fitness_distribution.png` | `results/figures/` | CH fitness box-plots by depth cluster |
| `rf_vs_temporal_scatter.png` | `results/figures/` | Predicted vs actual RUL — both models |
| `rf_vs_temporal_bar.png` | `results/figures/` | MAE / RMSE / R² comparison bars |
| `rul_trend_node.png` | `results/figures/` | RUL prediction trend for a single node |
| `battery_degradation_clusters.png` | `results/figures/` | Battery decay curves by depth cluster |
| `error_distribution.png` | `results/figures/` | Prediction error histograms |
| `argo_vs_synthetic_dist.png` | `results/figures/` | Real vs synthetic feature distributions |
| `protocol_comparison.csv` | `results/metrics/` | Full protocol benchmark table |
| `phase1_model_comparison.csv` | `results/metrics/` | RF vs Temporal metrics table |

---

## 🔬 Using the Models

### RUL Regressor (Random Forest)

```python
from aquasense.models import RULRegressor
from aquasense.simulate import simulate_sensor_data

df  = simulate_sensor_data()
reg = RULRegressor()
reg.fit(df)
print(reg)  # RULRegressor(MAE=32.07h, R2=0.9947)
reg.save("rul_model.pkl")
```

### Temporal RUL Model (windowed sequences)

```python
from aquasense.phase1.lstm_model import TemporalRULModel

tm = TemporalRULModel(window=10)
tm.fit(df)
trend = tm.predict_trend(df, node_id=5)
tm.save("temporal_model.pkl")
```

### Anomaly Detector

```python
from aquasense.models import AnomalyDetector

det    = AnomalyDetector()
det.fit(df)
tagged = det.tag_dataframe(df)
# Adds: anomaly_pred (0/1), anomaly_score (float)
```

### Real Ocean Data

```python
from aquasense.phase1.data.argo_connector import ArgoConnector
from aquasense.phase1.data.noaa_connector import NOAAConnector

# ARGO float profiles (live API or synthetic fallback)
argo = ArgoConnector()
df   = argo.load_or_fetch(n_floats=20)

# NOAA World Ocean Atlas climatology
noaa = NOAAConnector()
df   = noaa.fetch_climatology(region="arabian_sea", n_profiles=50)
print(noaa.pakistan_ocean_summary())
```

---

## 📡 Routing Protocol

### CH Fitness Score

```
Fitness = 0.50 × Energy Score  +  0.30 × Depth Score  +  0.20 × Link Score

  Energy Score = battery_voltage / max_battery       (higher = better)
  Depth Score  = 1 − depth_m / max_depth             (shallower = better)
  Link Score   = packet_success_rate                 (higher = better)
```

### Protocol Benchmark Comparison

| Protocol | w_energy | w_depth | w_link | Description |
|---|---|---|---|---|
| Random | 0.33 | 0.33 | 0.34 | Equal-weight naive baseline |
| LEACH (Energy) | 1.00 | 0.00 | 0.00 | Classic WSN energy-only selection |
| DBR (Depth) | 0.00 | 1.00 | 0.00 | Depth-Based Routing |
| **Proposed** | **0.50** | **0.30** | **0.20** | **AquaSense proposed protocol** |

---

## ⚙️ Configuration

All defaults live in `src/aquasense/config.py`. Override via environment variables:

| Variable | Default | Description |
|---|---|---|
| `AQUASENSE_DB` | `outputs/sensor_logs.db` | SQLite database path |
| `AQUASENSE_DB_URL` | *(unset)* | PostgreSQL DSN — overrides SQLite |
| `AQUASENSE_N_NODES` | `80` | Number of simulated sensor nodes |
| `AQUASENSE_N_TIMESTEPS` | `100` | Time-steps per node |
| `AQUASENSE_SEED` | `42` | Global random seed for reproducibility |
| `AQUASENSE_ANOMALY_RATE` | `0.08` | Fault injection fraction (8%) |
| `AQUASENSE_RF_TREES` | `150` | Random Forest estimators |
| `CH_W_ENERGY` | `0.50` | CH fitness energy weight |
| `CH_W_DEPTH` | `0.30` | CH fitness depth weight |
| `CH_W_LINK` | `0.20` | CH fitness link quality weight |
| `AQUASENSE_LSTM_WINDOW` | `10` | Temporal model sliding window size |
| `AQUASENSE_ARGO_FLOATS` | `30` | Number of ARGO floats to fetch |

### PostgreSQL

```bash
pip install aquasense[postgres]
export AQUASENSE_DB_URL="postgresql://user:password@localhost:5432/aquasense"
python -m aquasense.pipeline
```

---

## 🧪 Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=aquasense
```

**91 tests** across 16 test classes, all passing on Python 3.9, 3.10, 3.11, and 3.12:

| Test File | Tests | What Is Covered |
|---|---|---|
| `test_aquasense.py` | 58 | Simulation, database, RUL regressor, anomaly detector, depth clusterer, routing protocol, energy model |
| `test_phase1.py` | 33 | ARGO connector, NOAA connector, LSTM model, model comparison, Phase 1 pipeline |
| `test_visualise.py` | new | Dashboard rendering, panel outputs, figure file creation |

---

## 🗄 Database Schema

```sql
CREATE TABLE sensor_logs (
    id                INTEGER  PRIMARY KEY AUTOINCREMENT,
    node_id           INTEGER  NOT NULL,
    timestep          INTEGER  NOT NULL,
    depth_m           REAL     NOT NULL,
    pressure_bar      REAL     NOT NULL,
    salinity_ppt      REAL     NOT NULL,
    temperature_c     REAL     NOT NULL,
    battery_voltage   REAL     NOT NULL,
    tx_freq_ppm       REAL     NOT NULL,
    packet_success_rt REAL     NOT NULL,
    depth_cluster     TEXT     NOT NULL,
    is_anomaly        INTEGER  NOT NULL DEFAULT 0,
    rul_hours         REAL     NOT NULL,
    inserted_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Indexes: node_id, timestep, depth_cluster, is_anomaly
```

---

## 📄 License

MIT © Muhammad Hassaan Naeem

---

## 🙏 Acknowledgements

- [ARGO Programme](https://argo.ucsd.edu/) — global ocean float network data
- [NOAA NCEI](https://www.ncei.noaa.gov/) — World Ocean Atlas climatology
- [scikit-learn](https://scikit-learn.org/) — Random Forest, Isolation Forest, K-Means
- [PyTorch](https://pytorch.org/) — LSTM model implementation
- [Matplotlib](https://matplotlib.org/) — Dashboard visualisation
- Thorp (1967) — Underwater acoustic absorption coefficient formula
