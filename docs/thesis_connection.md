# Research — Code Mapping

**Research Title:** Energy-Efficient Depth-Aware Clustering-Based Routing Protocol  
for Internet of Underwater Things (IoUT) Using Depth Sensors

**Author:** Muhammad Hassaan Naeem

---

## Chapter → Module Mapping

| Research Chapter | Module | Key Functions |
|---|---|---|
| Ch. 1 — IoUT Architecture | `src/aquasense/simulate.py` | `simulate_sensor_data()` |
| Ch. 2 — Clustering Review | `src/aquasense/models.py` | `DepthClusterer` |
| Ch. 3 — Proposed Protocol | `src/aquasense/research/routing_protocol.py` | `compute_ch_fitness()`, `select_cluster_heads()` |
| Ch. 3 — Energy Model | `src/aquasense/research/energy_model.py` | `tx_energy()`, `path_loss()` |
| Ch. 4 — Experiments | `src/aquasense/research/benchmarks.py` | `run_full_benchmark_suite()` |
| Ch. 4 — RUL Prediction | `src/aquasense/models.py` | `RULRegressor` |
| Ch. 4 — Anomaly Detection | `src/aquasense/models.py` | `AnomalyDetector` |
| Ch. 4 — Results Figures | `results/figures/` | Generated automatically |

---

## Network Architecture

![Network Architecture](network_architecture.png)

The diagram shows the IoUT architecture implemented in this codebase:

| Diagram Component | Code Equivalent |
|---|---|
| Surface Sink | `query_latest_per_node()` collects top-layer data |
| Cluster Head (CH) | `select_cluster_heads()` in routing_protocol.py |
| Underwater Sensor Node | Each `node_id` in simulate.py |
| Base Station | SQL database storing all telemetry |
| Depth Layers | shallow / mid / deep clusters |
| Acoustic Links (dashed) | `packet_success_rt` feature |

---

## Proposed Protocol (Chapter 3)

### CH Fitness Score Formula

```
Fitness = w_e × E_norm + w_d × D_norm + w_l × L_norm

Where:
    E_norm = battery_voltage / max_battery     (energy score)
    D_norm = 1 - depth_m / max_depth           (depth score)
    L_norm = packet_success_rate               (link quality)

Default weights (proposed): w_e=0.5, w_d=0.3, w_l=0.2
```

### Benchmark Protocols

| Protocol | w_energy | w_depth | w_link | Notes |
|---|---|---|---|---|
| Random | 0.33 | 0.33 | 0.34 | Baseline |
| LEACH | 1.00 | 0.00 | 0.00 | Energy-only |
| DBR | 0.00 | 1.00 | 0.00 | Depth-only |
| **Proposed** | **0.50** | **0.30** | **0.20** | **This thesis** |

---

## Reproducing Research Results

```bash
# Install
pip install -e .
pip install seaborn

# Run full pipeline with benchmark comparison
python -m aquasense.pipeline --nodes 80 --timesteps 100 --bench

# Results saved to:
#   results/figures/alive_nodes_comparison.png
#   results/figures/energy_consumption_comparison.png
#   results/figures/delivery_ratio_comparison.png
#   results/figures/ch_fitness_distribution.png
#   results/metrics/protocol_comparison.csv
```
