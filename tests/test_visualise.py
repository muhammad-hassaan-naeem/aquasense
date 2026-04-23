from __future__ import annotations
name=tests/test_visualise.py
"""
tests/test_visualise.py  –  Tests for visualization module.
Run with:  pytest tests/test_visualise.py -v --cov=aquasense.visualise
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile


def _small_df(n_nodes=10, n_timesteps=20):
    """Helper to create small test dataframe."""
    from aquasense.simulate import simulate_sensor_data
    return simulate_sensor_data(n_nodes=n_nodes, n_timesteps=n_timesteps, random_seed=0)


class TestBuildDashboard:
    """Test the main build_dashboard function."""

    @pytest.fixture
    def dashboard_inputs(self):
        """Prepare all inputs needed for build_dashboard."""
        df = _small_df(n_nodes=15, n_timesteps=30)
        latest_df = df[df["timestep"] == df["timestep"].max()]
        
        # Create cluster stats
        cluster_stats = df.groupby("depth_cluster").agg({
            "rul_hours": "mean",
            "is_anomaly": "sum"
        }).reset_index()
        cluster_stats.columns = ["depth_cluster", "avg_rul", "total_anomalies"]
        
        # Test/pred arrays
        y_test = df["rul_hours"].values[:20]
        y_pred = df["rul_hours"].values[:20] + np.random.normal(0, 5, 20)
        y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
        
        # Anomaly dataframe
        anomaly_df = df.copy()
        anomaly_df["anomaly_pred"] = (df["is_anomaly"] * 0.8).astype(int)
        
        # K-means dataframe
        km_df = df.copy()
        km_df["km_cluster"] = np.random.randint(0, 3, len(df))
        
        # Feature importances
        from aquasense.config import FEATURES
        feature_importances = np.random.uniform(0, 1, len(FEATURES))
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
        """Test that build_dashboard returns a Path object."""
        from aquasense.visualise import build_dashboard
        with tempfile.TemporaryDirectory() as tmpdir:
            output = build_dashboard(
                df=dashboard_inputs["df"],
                latest_df=dashboard_inputs["latest_df"],
                cluster_stats=dashboard_inputs["cluster_stats"],
                y_test=dashboard_inputs["y_test"],
                y_pred=dashboard_inputs["y_pred"],
                anomaly_df=dashboard_inputs["anomaly_df"],
                km_df=dashboard_inputs["km_df"],
                feature_importances=dashboard_inputs["feature_importances"],
                output_path=Path(tmpdir) / "test_dashboard.png",
            )
            assert isinstance(output, Path)

    def test_build_dashboard_creates_file(self, dashboard_inputs):
        """Test that build_dashboard actually creates a PNG file."""
        from aquasense.visualise import build_dashboard
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_dashboard.png"
            output = build_dashboard(
                df=dashboard_inputs["df"],
                latest_df=dashboard_inputs["latest_df"],
                cluster_stats=dashboard_inputs["cluster_stats"],
                y_test=dashboard_inputs["y_test"],
                y_pred=dashboard_inputs["y_pred"],
                anomaly_df=dashboard_inputs["anomaly_df"],
                km_df=dashboard_inputs["km_df"],
                feature_importances=dashboard_inputs["feature_importances"],
                output_path=output_path,
            )
            assert output_path.exists()
            assert output_path.suffix == ".png"

    def test_build_dashboard_file_size(self, dashboard_inputs):
        """Test that generated PNG has reasonable file size."""
        from aquasense.visualise import build_dashboard
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_dashboard.png"
            build_dashboard(
                df=dashboard_inputs["df"],
                latest_df=dashboard_inputs["latest_df"],
                cluster_stats=dashboard_inputs["cluster_stats"],
                y_test=dashboard_inputs["y_test"],
                y_pred=dashboard_inputs["y_pred"],
                anomaly_df=dashboard_inputs["anomaly_df"],
                km_df=dashboard_inputs["km_df"],
                feature_importances=dashboard_inputs["feature_importances"],
                output_path=output_path,
            )
            file_size = output_path.stat().st_size
            assert file_size > 10000  # At least 10 KB
            assert file_size < 10000000  # Less than 10 MB

    def test_build_dashboard_default_output_path(self, dashboard_inputs):
        """Test that build_dashboard works with default output path."""
        from aquasense.visualise import build_dashboard
        output = build_dashboard(
            df=dashboard_inputs["df"],
            latest_df=dashboard_inputs["latest_df"],
            cluster_stats=dashboard_inputs["cluster_stats"],
            y_test=dashboard_inputs["y_test"],
            y_pred=dashboard_inputs["y_pred"],
            anomaly_df=dashboard_inputs["anomaly_df"],
            km_df=dashboard_inputs["km_df"],
            feature_importances=dashboard_inputs["feature_importances"],
        )
        assert isinstance(output, Path)
        assert output.name == "aquasense_dashboard.png"


class TestHelperFunctions:
    """Test individual helper panel functions."""

    def test_style_ax_applies_styling(self):
        """Test that _style_ax applies expected styling."""
        from aquasense.visualise import _style_ax
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        _style_ax(ax, title="Test Title")
        
        # Check that title was set
        assert ax.get_title() == "Test Title"
        # Check that grid is enabled
        assert ax.get_xgridlines()[0].get_visible()
        plt.close(fig)

    def test_legend_adds_legend(self):
        """Test that _legend function adds a legend."""
        from aquasense.visualise import _legend
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], label="Test")
        _legend(ax)
        
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_panel_kpi_no_error(self):
        """Test that _panel_kpi executes without error."""
        from aquasense.visualise import _panel_kpi
        import matplotlib.pyplot as plt
        
        df = _small_df(n_nodes=10, n_timesteps=5)
        latest_df = df[df["timestep"] == df["timestep"].max()]
        n_anomalies = int(df["is_anomaly"].sum())
        
        fig, ax = plt.subplots()
        _panel_kpi(ax, latest_df, n_anomalies)
        # If no error, test passes
        plt.close(fig)

    def test_panel_cluster_rul_no_error(self):
        """Test that _panel_cluster_rul executes without error."""
        from aquasense.visualise import _panel_cluster_rul
        import matplotlib.pyplot as plt
        
        df = _small_df(n_nodes=10, n_timesteps=5)
        cluster_stats = df.groupby("depth_cluster").agg({
            "rul_hours": "mean",
            "is_anomaly": "sum"
        }).reset_index()
        cluster_stats.columns = ["depth_cluster", "avg_rul", "total_anomalies"]
        
        fig, ax = plt.subplots()
        _panel_cluster_rul(ax, cluster_stats)
        plt.close(fig)

    def test_panel_rul_scatter_no_error(self):
        """Test that _panel_rul_scatter executes without error."""
        from aquasense.visualise import _panel_rul_scatter
        import matplotlib.pyplot as plt
        
        y_test = np.array([100, 200, 300, 150, 250], dtype=float)
        y_pred = np.array([95, 210, 290, 160, 240], dtype=float)
        
        fig, ax = plt.subplots()
        _panel_rul_scatter(ax, y_test, y_pred)
        plt.close(fig)

    def test_panel_battery_decay_no_error(self):
        """Test that _panel_battery_decay executes without error."""
        from aquasense.visualise import _panel_battery_decay
        import matplotlib.pyplot as plt
        
        df = _small_df(n_nodes=10, n_timesteps=20)
        fig, ax = plt.subplots()
        _panel_battery_decay(ax, df)
        plt.close(fig)

    def test_panel_depth_vs_rul_no_error(self):
        """Test that _panel_depth_vs_rul executes without error."""
        from aquasense.visualise import _panel_depth_vs_rul
        import matplotlib.pyplot as plt
        
        df = _small_df(n_nodes=10, n_timesteps=5)
        latest_df = df[df["timestep"] == df["timestep"].max()]
        
        fig, ax = plt.subplots()
        _panel_depth_vs_rul(ax, latest_df)
        plt.close(fig)

    def test_panel_anomaly_distribution_no_error(self):
        """Test that _panel_anomaly_distribution executes without error."""
        from aquasense.visualise import _panel_anomaly_distribution
        import matplotlib.pyplot as plt
        
        df = _small_df(n_nodes=10, n_timesteps=10)
        anomaly_df = df.copy()
        anomaly_df["anomaly_pred"] = (df["is_anomaly"] * 0.8).astype(int)
        
        fig, ax = plt.subplots()
        _panel_anomaly_distribution(ax, anomaly_df)
        plt.close(fig)

    def test_panel_kmeans_no_error(self):
        """Test that _panel_kmeans executes without error."""
        from aquasense.visualise import _panel_kmeans
        import matplotlib.pyplot as plt
        
        df = _small_df(n_nodes=10, n_timesteps=10)
        km_df = df.copy()
        km_df["km_cluster"] = np.random.randint(0, 3, len(df))
        
        fig, ax = plt.subplots()
        _panel_kmeans(ax, km_df)
        plt.close(fig)

    def test_panel_feature_importance_no_error(self):
        """Test that _panel_feature_importance executes without error."""
        from aquasense.visualise import _panel_feature_importance
        from aquasense.config import FEATURES
        import matplotlib.pyplot as plt
        
        fi = np.random.uniform(0, 1, len(FEATURES))
        fig, ax = plt.subplots()
        _panel_feature_importance(ax, fi)
        plt.close(fig)
