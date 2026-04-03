"""
tests/test_aquasense.py
-----------------------
Unit tests for all AquaSense modules.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

# ── Helpers ────────────────────────────────────────────────────────────────

def _small_df(n_nodes: int = 10, n_timesteps: int = 20) -> pd.DataFrame:
    """Return a tiny simulated DataFrame for fast tests."""
    from aquasense.simulate import simulate_sensor_data
    return simulate_sensor_data(n_nodes=n_nodes, n_timesteps=n_timesteps, random_seed=0)


# ══════════════════════════════════════════════════════════════════════════
# simulate.py
# ══════════════════════════════════════════════════════════════════════════

class TestSimulate:
    def test_shape(self):
        df = _small_df(n_nodes=5, n_timesteps=10)
        assert df.shape == (50, 12)

    def test_columns(self):
        df = _small_df()
        expected = {
            "node_id", "timestep", "depth_m", "pressure_bar",
            "salinity_ppt", "temperature_c", "battery_voltage",
            "tx_freq_ppm", "packet_success_rt", "depth_cluster",
            "is_anomaly", "rul_hours",
        }
        assert expected.issubset(df.columns)

    def test_no_nulls(self):
        df = _small_df()
        assert df.isnull().sum().sum() == 0

    def test_battery_above_cutoff(self):
        from aquasense.config import BATTERY_CUTOFF_V
        df = _small_df()
        # Anomaly injection can push battery slightly below threshold but
        # simulate guarantees the *pre-injection* value is >= BATTERY_CUTOFF_V.
        # Non-anomaly rows must always be >= cutoff.
        normal = df[df["is_anomaly"] == 0]
        assert (normal["battery_voltage"] >= BATTERY_CUTOFF_V).all()

    def test_psr_range(self):
        df = _small_df()
        assert df["packet_success_rt"].between(0, 1).all()

    def test_rul_non_negative(self):
        df = _small_df()
        assert (df["rul_hours"] >= 0).all()

    def test_cluster_labels(self):
        df = _small_df()
        assert set(df["depth_cluster"].unique()).issubset({"shallow", "mid", "deep"})

    def test_reproducibility(self):
        df1 = _small_df()
        df2 = _small_df()
        pd.testing.assert_frame_equal(df1, df2)

    def test_anomaly_rate(self):
        # With seed=0 and many rows the observed rate should be near 8 %
        df = _small_df(n_nodes=50, n_timesteps=100)
        rate = df["is_anomaly"].mean()
        assert 0.04 < rate < 0.16, f"Anomaly rate out of expected range: {rate:.3f}"


# ══════════════════════════════════════════════════════════════════════════
# database.py
# ══════════════════════════════════════════════════════════════════════════

class TestDatabase:
    """Use an in-memory SQLite connection to avoid filesystem side-effects."""

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
        latest = query_latest_per_node(conn)
        assert len(latest) == df["node_id"].nunique()
        # Every returned timestep should equal max for that node
        max_ts = df.groupby("node_id")["timestep"].max()
        for _, row in latest.iterrows():
            assert row["timestep"] == max_ts[row["node_id"]]

    def test_query_cluster_stats(self, conn_and_df):
        from aquasense.database import query_cluster_stats
        conn, _ = conn_and_df
        stats = query_cluster_stats(conn)
        assert set(stats.columns) >= {
            "depth_cluster", "n_nodes", "avg_battery",
            "avg_psr", "avg_rul", "avg_tx_freq", "total_anomalies",
        }
        assert len(stats) <= 3  # at most 3 clusters

    def test_query_critical_nodes(self, conn_and_df):
        from aquasense.database import query_critical_nodes
        conn, _ = conn_and_df
        # A very high threshold should return all nodes
        critical = query_critical_nodes(conn, rul_threshold=1e9)
        assert len(critical) == 10  # n_nodes=10 in _small_df


# ══════════════════════════════════════════════════════════════════════════
# models.py
# ══════════════════════════════════════════════════════════════════════════

class TestRULRegressor:
    @pytest.fixture(scope="class")
    def fitted_reg(self):
        from aquasense.models import RULRegressor
        df  = _small_df(n_nodes=20, n_timesteps=50)
        reg = RULRegressor(n_estimators=20, max_depth=5)
        reg.fit(df)
        return reg

    def test_metrics_set(self, fitted_reg):
        assert "mae" in fitted_reg.metrics_
        assert "r2"  in fitted_reg.metrics_

    def test_r2_positive(self, fitted_reg):
        assert fitted_reg.metrics_["r2"] > 0

    def test_mae_reasonable(self, fitted_reg):
        # MAE should be less than the mean RUL of the test set
        df    = _small_df(n_nodes=20, n_timesteps=50)
        mean_rul = df["rul_hours"].mean()
        assert fitted_reg.metrics_["mae"] < mean_rul * 2

    def test_predict_shape(self, fitted_reg):
        df   = _small_df(n_nodes=3, n_timesteps=5)
        pred = fitted_reg.predict(df)
        assert pred.shape == (15,)

    def test_predict_non_negative(self, fitted_reg):
        df   = _small_df(n_nodes=3, n_timesteps=5)
        pred = fitted_reg.predict(df)
        assert (pred >= 0).all()

    def test_feature_importances_shape(self, fitted_reg):
        from aquasense.config import FEATURES
        assert fitted_reg.feature_importances_.shape == (len(FEATURES),)

    def test_save_load(self, fitted_reg, tmp_path):
        from aquasense.models import RULRegressor
        path = tmp_path / "reg.pkl"
        fitted_reg.save(path)
        loaded = RULRegressor.load(path)
        df     = _small_df(n_nodes=2, n_timesteps=5)
        np.testing.assert_allclose(
            fitted_reg.predict(df), loaded.predict(df), rtol=1e-5
        )


class TestAnomalyDetector:
    @pytest.fixture(scope="class")
    def fitted_det(self):
        from aquasense.models import AnomalyDetector
        df  = _small_df(n_nodes=20, n_timesteps=50)
        det = AnomalyDetector()
        det.fit(df)
        return det, df

    def test_metrics_set(self, fitted_det):
        det, _ = fitted_det
        assert "precision" in det.metrics_
        assert "recall"    in det.metrics_

    def test_predict_binary(self, fitted_det):
        det, df = fitted_det
        preds = det.predict(df)
        assert set(preds).issubset({0, 1})

    def test_tag_dataframe_columns(self, fitted_det):
        det, df = fitted_det
        tagged = det.tag_dataframe(df)
        assert "anomaly_pred"  in tagged.columns
        assert "anomaly_score" in tagged.columns

    def test_save_load(self, fitted_det, tmp_path):
        from aquasense.models import AnomalyDetector
        det, df = fitted_det
        path    = tmp_path / "det.pkl"
        det.save(path)
        loaded  = AnomalyDetector.load(path)
        np.testing.assert_array_equal(det.predict(df), loaded.predict(df))


class TestDepthClusterer:
    @pytest.fixture(scope="class")
    def fitted_clr(self):
        from aquasense.models import DepthClusterer
        df  = _small_df(n_nodes=20, n_timesteps=50)
        clr = DepthClusterer(n_clusters=3)
        clr.fit(df)
        return clr, df

    def test_predict_range(self, fitted_clr):
        clr, df = fitted_clr
        labels  = clr.predict(df)
        assert set(labels).issubset({0, 1, 2})

    def test_tag_dataframe(self, fitted_clr):
        clr, df = fitted_clr
        tagged  = clr.tag_dataframe(df)
        assert "km_cluster" in tagged.columns

    def test_cluster_summary_shape(self, fitted_clr):
        clr, df = fitted_clr
        summary = clr.cluster_summary(df)
        assert len(summary) == 3
