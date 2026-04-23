"""
tests/test_aquasense.py  –  Full test suite (27+ tests).
Run with:  pytest tests/ -v --cov=aquasense
"""
from __future__ import annotations
import sqlite3
import numpy as np
import pandas as pd
import pytest


def _small_df(n_nodes=10, n_timesteps=20):
    from aquasense.simulate import simulate_sensor_data
    return simulate_sensor_data(n_nodes=n_nodes, n_timesteps=n_timesteps, random_seed=0)


# ══════════════════════════════════════════════════════════════════════════
# simulate.py
# ══════════════════════════════════════════════════════════════════════════

class TestSimulate:
    def test_shape(self):
        assert _small_df(5, 10).shape == (50, 12)

    def test_columns(self):
        expected = {"node_id","timestep","depth_m","pressure_bar","salinity_ppt",
                    "temperature_c","battery_voltage","tx_freq_ppm",
                    "packet_success_rt","depth_cluster","is_anomaly","rul_hours"}
        assert expected.issubset(_small_df().columns)

    def test_no_nulls(self):
        assert _small_df().isnull().sum().sum() == 0

    def test_battery_above_cutoff(self):
        from aquasense.config import BATTERY_CUTOFF_V
        df = _small_df()
        assert (df[df["is_anomaly"]==0]["battery_voltage"] >= BATTERY_CUTOFF_V).all()

    def test_psr_range(self):
        assert _small_df()["packet_success_rt"].between(0, 1).all()

    def test_rul_non_negative(self):
        assert (_small_df()["rul_hours"] >= 0).all()

    def test_cluster_labels(self):
        assert set(_small_df()["depth_cluster"].unique()).issubset({"shallow","mid","deep"})

    def test_reproducibility(self):
        pd.testing.assert_frame_equal(_small_df(), _small_df())

    def test_anomaly_rate(self):
        df   = _small_df(50, 100)
        rate = df["is_anomaly"].mean()
        assert 0.04 < rate < 0.16


# ══════════════════════════════════════════════════════════════════════════
# database.py
# ══════════════════════════════════════════════════════════════════════════

class TestDatabase:
    @pytest.fixture()
    def conn_and_df(self):
        from aquasense.database import init_schema, write_logs
        df   = _small_df()
        conn = sqlite3.connect(":memory:")
        init_schema(conn)
        write_logs(df, conn, replace=False)
        yield conn, df
        conn.close()

    def test_row_count(self, conn_and_df):
        conn, df = conn_and_df
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sensor_logs")
        assert cur.fetchone()[0] == len(df)

    def test_query_latest_per_node(self, conn_and_df):
        from aquasense.database import query_latest_per_node
        conn, df = conn_and_df
        latest   = query_latest_per_node(conn)
        assert len(latest) == df["node_id"].nunique()

    def test_query_cluster_stats(self, conn_and_df):
        from aquasense.database import query_cluster_stats
        conn, _ = conn_and_df
        stats   = query_cluster_stats(conn)
        assert set(stats.columns) >= {"depth_cluster","n_nodes","avg_battery",
                                      "avg_psr","avg_rul","avg_tx_freq","total_anomalies"}

    def test_query_critical_nodes(self, conn_and_df):
        from aquasense.database import query_critical_nodes
        conn, _ = conn_and_df
        critical = query_critical_nodes(conn, rul_threshold=1e9)
        assert len(critical) == 10


# ══════════════════════════════════════════════════════════════════════════
# models.py
# ══════════════════════════════════════════════════════════════════════════

class TestRULRegressor:
    @pytest.fixture(scope="class")
    def fitted(self):
        from aquasense.models import RULRegressor
        df  = _small_df(20, 50)
        reg = RULRegressor(n_estimators=20, max_depth=5)
        reg.fit(df)
        return reg

    def test_metrics_set(self, fitted):
        assert "mae" in fitted.metrics_ and "r2" in fitted.metrics_

    def test_r2_positive(self, fitted):
        assert fitted.metrics_["r2"] > 0

    def test_predict_shape(self, fitted):
        assert fitted.predict(_small_df(3,5)).shape == (15,)

    def test_predict_non_negative(self, fitted):
        assert (fitted.predict(_small_df(3,5)) >= 0).all()

    def test_feature_importances_shape(self, fitted):
        from aquasense.config import FEATURES
        assert fitted.feature_importances_.shape == (len(FEATURES),)

    def test_save_load(self, fitted, tmp_path):
        from aquasense.models import RULRegressor
        p = tmp_path / "reg.pkl"
        fitted.save(p)
        loaded = RULRegressor.load(p)
        df = _small_df(2, 5)
        np.testing.assert_allclose(fitted.predict(df), loaded.predict(df), rtol=1e-5)


class TestAnomalyDetector:
    @pytest.fixture(scope="class")
    def fitted(self):
        from aquasense.models import AnomalyDetector
        df  = _small_df(20, 50)
        det = AnomalyDetector()
        det.fit(df)
        return det, df

    def test_metrics_set(self, fitted):
        det, _ = fitted
        assert "precision" in det.metrics_

    def test_predict_binary(self, fitted):
        det, df = fitted
        assert set(det.predict(df)).issubset({0, 1})

    def test_tag_dataframe_columns(self, fitted):
        det, df = fitted
        tagged  = det.tag_dataframe(df)
        assert "anomaly_pred" in tagged.columns and "anomaly_score" in tagged.columns

    def test_save_load(self, fitted, tmp_path):
        from aquasense.models import AnomalyDetector
        det, df = fitted
        p       = tmp_path / "det.pkl"
        det.save(p)
        loaded  = AnomalyDetector.load(p)
        np.testing.assert_array_equal(det.predict(df), loaded.predict(df))


class TestDepthClusterer:
    @pytest.fixture(scope="class")
    def fitted(self):
        from aquasense.models import DepthClusterer
        df  = _small_df(20, 50)
        clr = DepthClusterer(n_clusters=3)
        clr.fit(df)
        return clr, df

    def test_predict_range(self, fitted):
        clr, df = fitted
        assert set(clr.predict(df)).issubset({0,1,2})

    def test_tag_dataframe(self, fitted):
        clr, df = fitted
        assert "km_cluster" in clr.tag_dataframe(df).columns

    def test_cluster_summary_shape(self, fitted):
        clr, df = fitted
        assert len(clr.cluster_summary(df)) == 3


# ══════════════════════════════════════════════════════════════════════════
# research/routing_protocol.py
# ══════════════════════════════════════════════════════════════════════════

class TestRoutingProtocol:
    @pytest.fixture(scope="class")
    def snapshot(self):
        from aquasense.simulate import simulate_sensor_data
        from aquasense.database import (get_connection, init_schema,
                                        query_latest_per_node, write_logs)
        df   = simulate_sensor_data(n_nodes=30, n_timesteps=10, random_seed=1)
        conn = sqlite3.connect(":memory:")
        init_schema(conn)
        write_logs(df, conn, replace=False)
        latest = query_latest_per_node(conn)
        conn.close()
        return latest, df

    def test_ch_fitness_range(self):
        from aquasense.research.routing_protocol import compute_ch_fitness
        score = compute_ch_fitness(3.8, 50.0, 0.9)
        assert 0.0 <= score <= 1.0

    def test_ch_fitness_higher_battery_wins(self):
        from aquasense.research.routing_protocol import compute_ch_fitness
        high = compute_ch_fitness(4.0, 50.0, 0.9)
        low  = compute_ch_fitness(2.6, 50.0, 0.9)
        assert high > low

    def test_ch_fitness_shallower_wins(self):
        from aquasense.research.routing_protocol import compute_ch_fitness
        shallow = compute_ch_fitness(3.8, 20.0,  0.9)
        deep    = compute_ch_fitness(3.8, 800.0, 0.9)
        assert shallow > deep

    def test_select_cluster_heads_returns_list(self, snapshot):
        from aquasense.research.routing_protocol import select_cluster_heads
        latest, _ = snapshot
        chs = select_cluster_heads(latest, protocol="Proposed")
        assert isinstance(chs, list)
        assert len(chs) >= 1

    def test_cluster_heads_have_valid_nodes(self, snapshot):
        from aquasense.research.routing_protocol import select_cluster_heads
        latest, _ = snapshot
        chs = select_cluster_heads(latest, protocol="Proposed")
        node_ids = set(latest["node_id"].unique())
        for ch in chs:
            assert ch.node_id in node_ids

    def test_routing_path_depth_order(self, snapshot):
        from aquasense.research.routing_protocol import (
            select_cluster_heads, build_routing_path)
        latest, _ = snapshot
        chs  = select_cluster_heads(latest)
        path = build_routing_path(chs)
        depths = [ch.depth_m for ch in path]
        assert depths == sorted(depths, reverse=True)

    def test_simulate_routing_rounds_shape(self, snapshot):
        from aquasense.research.routing_protocol import simulate_routing_rounds
        _, df = snapshot
        rounds = simulate_routing_rounds(df, protocol="Proposed")
        assert len(rounds) == df["timestep"].nunique()
        assert "alive_nodes" in rounds.columns

    def test_all_protocols_run(self, snapshot):
        from aquasense.research.routing_protocol import simulate_routing_rounds
        from aquasense.config import BENCHMARK_PROTOCOLS
        _, df = snapshot
        for proto in BENCHMARK_PROTOCOLS:
            rounds = simulate_routing_rounds(df, protocol=proto)
            assert len(rounds) > 0


# ══════════════════════════════════════════════════════════════════════════
# research/energy_model.py
# ══════════════════════════════════════════════════════════════════════════

class TestEnergyModel:
    def test_absorption_increases_with_depth(self):
        from aquasense.research.energy_model import absorption_coefficient
        shallow = absorption_coefficient(10)
        deep    = absorption_coefficient(500)
        assert deep > shallow

    def test_tx_energy_increases_with_distance(self):
        from aquasense.research.energy_model import tx_energy
        assert tx_energy(100) < tx_energy(500)

    def test_tx_energy_positive(self):
        from aquasense.research.energy_model import tx_energy
        assert tx_energy(200) > 0

    def test_estimate_round_energy_keys(self):
        from aquasense.research.energy_model import estimate_round_energy
        df  = _small_df(10, 5)
        snap = df[df["timestep"] == 0]
        result = estimate_round_energy(snap)
        assert set(result.keys()) >= {"total_uJ", "intra_cluster_tx",
                                      "inter_cluster_tx", "aggregation"}

    def test_estimate_round_energy_positive(self):
        from aquasense.research.energy_model import estimate_round_energy
        df   = _small_df(10, 5)
        snap = df[df["timestep"] == 0]
        assert estimate_round_energy(snap)["total_uJ"] >= 0


# ══════════════════════════════════════════════════════════════════════════
# phase1/argo_connector.py
# ══════════════════════════════════════════════════════════════════════════

class TestArgoConnector:
    @pytest.fixture(scope="class")
    def argo_df(self, tmp_path_factory):
        from aquasense.phase1.argo_connector import ArgoConnector
        tmp = tmp_path_factory.mktemp("argo")
        conn = ArgoConnector(cache_dir=tmp)
        return conn.get_data(n_floats=10, force_synthetic=True)

    def test_shape(self, argo_df):
        assert len(argo_df) > 0
        assert argo_df['node_id'].nunique() == 10

    def test_columns(self, argo_df):
        required = {"node_id","timestep","depth_m","battery_voltage",
                    "rul_hours","is_anomaly","data_source"}
        assert required.issubset(argo_df.columns)

    def test_data_source_label(self, argo_df):
        assert argo_df['data_source'].iloc[0] in ('argo_real','synthetic_argo')

    def test_no_nulls_key_cols(self, argo_df):
        for col in ["depth_m","battery_voltage","rul_hours"]:
            assert argo_df[col].notnull().all(), f"nulls in {col}"

    def test_rul_non_negative(self, argo_df):
        assert (argo_df['rul_hours'] >= 0).all()

    def test_battery_in_range(self, argo_df):
        assert argo_df['battery_voltage'].between(2.4, 4.3).all()

    def test_cluster_labels(self, argo_df):
        assert set(argo_df['depth_cluster'].unique()).issubset(
            {'shallow','mid','deep'})

    def test_summary_method(self, argo_df, tmp_path_factory):
        from aquasense.phase1.argo_connector import ArgoConnector
        tmp  = tmp_path_factory.mktemp("argo2")
        conn = ArgoConnector(cache_dir=tmp)
        s    = conn.summary(argo_df)
        assert len(s) == argo_df['node_id'].nunique()
        assert 'final_rul' in s.columns


# ══════════════════════════════════════════════════════════════════════════
# phase1/lstm_model.py
# ══════════════════════════════════════════════════════════════════════════

class TestTemporalRULModel:
    @pytest.fixture(scope="class")
    def fitted_tm(self):
        from aquasense.phase1.lstm_model import TemporalRULModel
        df  = _small_df(n_nodes=20, n_timesteps=30)
        tm  = TemporalRULModel(window=5, n_estimators=30)
        tm.fit(df)
        return tm, df

    def test_metrics_set(self, fitted_tm):
        tm, _ = fitted_tm
        assert "mae" in tm.metrics_ and "r2" in tm.metrics_

    def test_r2_positive(self, fitted_tm):
        tm, _ = fitted_tm
        assert tm.metrics_["r2"] > 0

    def test_predict_non_negative(self, fitted_tm):
        tm, df = fitted_tm
        preds = tm.predict(df)
        assert (preds >= 0).all()

    def test_predict_trend_shape(self, fitted_tm):
        tm, df = fitted_tm
        valid = df.groupby("node_id")["timestep"].count()
        node  = int(valid[valid > tm.window + 2].index[0])
        trend = tm.predict_trend(df, node)
        assert len(trend) > 0
        assert "true_rul" in trend.columns and "predicted_rul" in trend.columns

    def test_save_load(self, fitted_tm, tmp_path):
        from aquasense.phase1.lstm_model import TemporalRULModel
        tm, df = fitted_tm
        path   = tmp_path / "tm.pkl"
        tm.save(path)
        loaded = TemporalRULModel.load(path)
        assert loaded.window == tm.window
        np.testing.assert_allclose(
            tm.predict(df), loaded.predict(df), rtol=1e-4)

    def test_repr(self, fitted_tm):
        tm, _ = fitted_tm
        assert "TemporalRULModel" in repr(tm)
        assert "MAE" in repr(tm)


class TestBuildSequences:
    def test_sequence_count(self):
        from aquasense.phase1.lstm_model import build_sequences
        df = _small_df(n_nodes=5, n_timesteps=20)
        X, y = build_sequences(df, window=5)
        # Each node contributes (20-5) = 15 sequences
        assert len(X) == 5 * 15

    def test_sequence_width(self):
        from aquasense.phase1.lstm_model import build_sequences
        from aquasense.config import FEATURES
        df = _small_df(n_nodes=3, n_timesteps=15)
        X, y = build_sequences(df, window=5)
        assert X.shape[1] == 5 * len(FEATURES)

    def test_y_non_negative(self):
        from aquasense.phase1.lstm_model import build_sequences
        df = _small_df(n_nodes=3, n_timesteps=15)
        _, y = build_sequences(df, window=5)
        assert (y >= 0).all()


# ══════════════════════════════════════════════════════════════════════════
# phase1/comparison.py
# ══════════════════════════════════════════════════════════════════════════

class TestEvaluateModel:
    def test_returns_required_keys(self):
        from aquasense.phase1.lstm_model import evaluate_model
        y_true = np.array([100, 200, 300, 150, 250], dtype=float)
        y_pred = np.array([ 95, 210, 290, 160, 240], dtype=float)
        result = evaluate_model(y_true, y_pred, "TestModel")
        for k in ["model","mae","rmse","r2","mape_pct","within_10pct"]:
            assert k in result

    def test_perfect_prediction_r2(self):
        from aquasense.phase1.lstm_model import evaluate_model
        y = np.linspace(10, 500, 100)
        r = evaluate_model(y, y, "Perfect")
        assert r["r2"] == pytest.approx(1.0, abs=1e-5)
        assert r["mae"] == pytest.approx(0.0, abs=1e-5)
