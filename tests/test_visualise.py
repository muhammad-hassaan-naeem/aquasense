"""
tests/test_visualise.py — Tests for visualization module.

Run with:
pytest tests/test_visualise.py -v --cov=aquasense.visualise
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")  # ✅ Headless backend for CI


def _small_df(n_nodes=10, n_timesteps=20):
    """Helper to create small test dataframe."""
    from aquasense.simulate import simulate_sensor_data
    return simulate_sensor_data(
        n_nodes=n_nodes,
        n_timesteps=n_timesteps,
        random_seed=0,
    )


class TestBuildDashboard:
    """Test the main build_dashboard function."""

    @pytest.fixture
    def dashboard_inputs(self):
        """Prepare all inputs needed for build_dashboard."""
        df = _small_df(n_nodes=15, n_timesteps=30)
        latest_df = df[df["timestep"] == df["timestep"].max()]

        # Cluster statistics
        cluster_stats = (
            df.groupby("depth_cluster")
            .agg(
                avg_rul=("rul_hours", "mean"),
                total_anomalies=("is_anomaly", "sum"),
            )
            .reset_index()
        )

        # Prediction arrays
        y_test = df["rul_hours"].values[:20]
        y_pred = y_test + np.random.normal(0, 5, size=20)
        y_pred = np.maximum(y_pred, 0)

        # Anomaly dataframe
        anomaly_df = df.copy()
        anomaly_df["anomaly_pred"] = (df["is_anomaly"] * 0.8).astype(int)

        # K-means dataframe
        km_df = df.copy()
        km_df["km_cluster"] = np.random.randint(0, 3, len(df))

        # Feature importances
        from aquasense.config import FEATURES
        feature_importances = np.random.rand(len(FEATURES))
        feature_importances /= feature_importances.sum()

        return {
            "df": df,
            "latest_df": latest_df,
            "cluster_stats": cluster_stats,
            "y_test": y_test,
            "y_pred": y_pred,
            "anomaly_df": anomaly_df,
            "km_df": km_df,
            "feature_importances": feature_importances,
        }

    def test_build_dashboard_returns_path(self, dashboard_inputs):
        from aquasense.visualise import build_dashboard

        with tempfile.TemporaryDirectory() as tmpdir:
            output = build_dashboard(
                **dashboard_inputs,
                output_path=Path(tmpdir) / "dashboard.png",
            )
            assert isinstance(output, Path)

    def test_build_dashboard_creates_file(self, dashboard_inputs):
        from aquasense.visualise import build_dashboard

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dashboard.png"
            build_dashboard(**dashboard_inputs, output_path=path)
            assert path.exists()
            assert path.suffix == ".png"

    def test_build_dashboard_file_size(self, dashboard_inputs):
        from aquasense.visualise import build_dashboard

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dashboard.png"
            build_dashboard(**dashboard_inputs, output_path=path)
            size = path.stat().st_size
            assert 10_000 < size < 10_000_000


class TestHelperFunctions:
    """Test individual helper plotting functions."""

    def test_style_ax(self):
        from aquasense.visualise import _style_ax
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _style_ax(ax, title="Test")
        assert ax.get_title() == "Test"
        plt.close(fig)

    def test_legend(self):
        from aquasense.visualise import _legend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], label="Test")
        _legend(ax)
        assert ax.get_legend() is not None
        plt.close(fig)
``
