# 🌊 AquaSense — Underwater Sensor Node Monitoring System

[![CI](https://github.com/muhammad-hassaan-naeem/aquasense/actions/workflows/ci.yml/badge.svg)](https://github.com/muhammad-hassaan-naeem/aquasense/actions)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-orange)](https://github.com/astral-sh/ruff)

> **Predict when your underwater sensor nodes will fail — before they do.**

AquaSense is a fully self-contained Python toolkit that simulates, logs,
analyses, and visualises the health of underwater IoT sensor networks.
It uses **machine learning** to predict battery *Remaining Useful Life* (RUL)
and detect anomalous node behaviour caused by depth-induced interference,
corrosion, or hardware faults.

---

## ✨ Features

| Capability | Detail |
|---|---|
| **Data Simulation** | Realistic telemetry for *N* nodes across depth clusters (shallow / mid / deep) |
| **RUL Regression** | Random Forest predicts remaining battery life (hours) — R² > 0.99 on synthetic data |
| **Anomaly Detection** | Isolation Forest flags malfunctioning nodes (unsupervised, no labels needed) |
| **Depth Clustering** | K-Means discovers energy-efficiency groups independently of hand-crafted labels |
| **SQL Persistence** | SQLite by default; swap to PostgreSQL via one environment variable |
| **Dashboard** | 8-panel Matplotlib dashboard with dark ocean-themed design |
| **CLI** | Full `argparse` interface with logging, alerting, and `--help` |
| **CI/CD** | GitHub Actions workflow — lint, test, coverage, and dashboard smoke test |

---

## 🗂 Project Structure

```
aquasense/
├── src/
│   └── aquasense/
│       ├── __init__.py      # Package metadata
│       ├── config.py        # All tuneable constants (env-var overridable)
│       ├── simulate.py      # Synthetic data generation (NumPy / Pandas)
│       ├── database.py      # SQLite / PostgreSQL persistence layer
│       ├── models.py        # RULRegressor · AnomalyDetector · DepthClusterer
│       ├── visualise.py     # 8-panel Matplotlib dashboard
│       └── pipeline.py      # End-to-end orchestrator + CLI entry-point
├── tests/
│   └── test_aquasense.py    # Pytest unit tests (simulate, database, models)
├── outputs/                 # Generated dashboard PNG and database (git-ignored)
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions: lint → test → smoke
├── pyproject.toml           # Modern Python packaging (PEP 517/518)
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1 — Clone and install

```bash
git clone https://github.com/muhammad-hassaan-naeem/aquasense.git
cd aquasense
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### 2 — Run the pipeline

```bash
# Default: 80 nodes, 100 timesteps, SQLite, saves dashboard to outputs/
aquasense

# Custom run
aquasense --nodes 200 --timesteps 150 --seed 7 --rul-alert 30

# Skip the dashboard (headless / CI)
aquasense --no-dashboard --quiet
```

### 3 — View results

```
outputs/
├── aquasense_dashboard.png   ← 8-panel monitoring dashboard
└── sensor_logs.db            ← SQLite database with all telemetry
```

---

## 🔧 Configuration

All defaults live in `src/aquasense/config.py` and can be overridden via
environment variables without touching source code:

| Variable | Default | Description |
|---|---|---|
| `AQUASENSE_DB` | `outputs/sensor_logs.db` | SQLite database path |
| `AQUASENSE_DB_URL` | *(unset)* | PostgreSQL DSN — see below |
| `AQUASENSE_N_NODES` | `80` | Number of simulated nodes |
| `AQUASENSE_N_TIMESTEPS` | `100` | Time-steps per node |
| `AQUASENSE_SEED` | `42` | Global random seed |
| `AQUASENSE_ANOMALY_RATE` | `0.08` | Fraction of injected anomalies |
| `AQUASENSE_RF_TREES` | `150` | Random Forest estimators |
| `AQUASENSE_RF_DEPTH` | `12` | Random Forest max depth |

### PostgreSQL

```bash
pip install aquasense[postgres]
export AQUASENSE_DB_URL="postgresql://user:password@localhost:5432/aquasense"
aquasense
```

All queries use standard SQL and are compatible with PostgreSQL 13+.

---

## 📊 Dashboard Panels

| # | Panel | Description |
|---|---|---|
| 1 | Fleet KPIs | Live metrics: node count, avg RUL, anomaly count, battery, PSR |
| 2 | Cluster Efficiency | Avg RUL per depth cluster (bar) + anomaly overlay (line) |
| 3 | RUL Prediction | Predicted vs actual scatter with MAE / R² annotation |
| 4 | Battery Decay | Mean voltage over time per cluster with ±1σ bands |
| 5 | Depth vs RUL | Latest reading scatter coloured by cluster |
| 6 | Anomaly Distribution | True vs detected anomalies per depth cluster |
| 7 | K-Means Clusters | Data-driven depth/battery groupings |
| 8 | Feature Importance | Which sensor features drive RUL prediction most |

---

## 🧬 Simulation Model

### Depth Clusters

| Cluster | Depth (m) | Initial Battery (V) | Tx Freq (ppm) | Weight |
|---|---|---|---|---|
| Shallow | 5 – 60 | 3.8 – 4.2 | 0.5 – 2.0 | 35 % |
| Mid | 60 – 300 | 3.4 – 4.0 | 0.3 – 1.5 | 40 % |
| Deep | 300 – 1 000 | 3.0 – 3.8 | 0.1 – 0.8 | 25 % |

### Battery Drain Model

```
drain = 0.003 × tx_freq + 0.0005 × pressure_bar + 0.0002 × salinity + noise
RUL   = (voltage − 2.5 V) ÷ drain × 60     [hours]
```

### Anomaly Injection (~8 % of readings)

- Battery voltage multiplied by U(0.50, 0.85)
- Packet success rate multiplied by U(0.10, 0.50)
- RUL multiplied by U(0.00, 0.40)

---

## 🤖 ML Components

### `RULRegressor`

```python
from aquasense.models import RULRegressor
from aquasense.simulate import simulate_sensor_data

df  = simulate_sensor_data()
reg = RULRegressor(n_estimators=150, max_depth=12)
reg.fit(df)
print(reg)           # RULRegressor(MAE=37.50h, R²=0.9921)

predictions = reg.predict(df)
reg.save("models/rul_regressor.pkl")
```

### `AnomalyDetector`

```python
from aquasense.models import AnomalyDetector

det = AnomalyDetector(contamination=0.08)
det.fit(df)
print(det)           # AnomalyDetector(P=0.572, R=0.609, F1=0.589)

tagged = det.tag_dataframe(df)   # adds anomaly_pred + anomaly_score columns
```

### `DepthClusterer`

```python
from aquasense.models import DepthClusterer

clr = DepthClusterer(n_clusters=3)
clr.fit(df)
print(clr.cluster_summary(df))
```

---

## 🧪 Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=aquasense
```

Test coverage targets:

- `simulate.py` — shape, dtypes, ranges, reproducibility, anomaly rate
- `database.py` — write / read round-trips, optimised queries, alerting
- `models.py` — fit / predict / save / load for all three estimators

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

-- Indexes (applied automatically)
CREATE INDEX idx_sl_node    ON sensor_logs (node_id);
CREATE INDEX idx_sl_ts      ON sensor_logs (timestep);
CREATE INDEX idx_sl_cluster ON sensor_logs (depth_cluster);
CREATE INDEX idx_sl_anomaly ON sensor_logs (is_anomaly);
```

---

## 🔌 Using as a Library

```python
from aquasense.simulate  import simulate_sensor_data
from aquasense.database  import get_connection, init_schema, write_logs, query_critical_nodes
from aquasense.models    import RULRegressor, AnomalyDetector, DepthClusterer
from aquasense.visualise import build_dashboard

# 1. Generate data
df = simulate_sensor_data(n_nodes=100, n_timesteps=200)

# 2. Persist
conn = get_connection()
init_schema(conn)
write_logs(df, conn)

# 3. Train models
reg = RULRegressor().fit(df)
det = AnomalyDetector().fit(df)
clr = DepthClusterer().fit(df)

# 4. Alert on critical nodes
critical = query_critical_nodes(conn, rul_threshold=24.0)

# 5. Dashboard
build_dashboard(
    df=df,
    latest_df=...,
    cluster_stats=...,
    y_test=reg._y_test, y_pred=reg._y_pred,
    anomaly_df=det.tag_dataframe(df),
    km_df=clr.tag_dataframe(df),
    feature_importances=reg.feature_importances_,
)
```

---

## 📄 License

MIT © AquaSense Contributors.  See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [scikit-learn](https://scikit-learn.org/) — Random Forest, Isolation Forest, K-Means
- [Matplotlib](https://matplotlib.org/) — Dashboard visualisation
- [Pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/) — Data simulation
