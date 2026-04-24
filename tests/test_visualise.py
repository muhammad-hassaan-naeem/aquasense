"""
tests/test_visualise.py  –  Dashboard / visualisation test suite.
Run with:  pytest tests/ -v --cov=aquasense

Tests every public function in src/aquasense/visualise.py without
writing to disk (tmp_path) and without spinning up a display server
(Matplotlib backend is forced to Agg via conftest / env var).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # must be set before any pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


# ── helpers ────────────────────────────────────────────────────────────────

def _small_df(n_nodes: int = 10, n_timesteps: int = 20) -> pd.DataFrame:
    from aquasense.simulate import simulate_sensor_data
    return simulate_sensor_data(
        n_nodes=n_nodes, n_timesteps=n_timesteps, random_seed=0)


def _cluster_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return cluster stats exactly as query_cluster_stats() does."""
    from aquasense.database import (
        get_connection, init_schema, query_cluster_stats, write_logs,
    )
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    write_logs(df, conn, replace=False)
    stats = query_cluster_stats(conn)
    conn.close()
    return stats


def _latest(df: pd.DataFrame) -> pd.DataFrame:
    from aquasense.database import (
        get_connection, init_schema, query_latest_per_node, write_logs,
    )
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    write_logs(df, conn, replace=False)
    latest = query_latest_per_node(conn)
    conn.close()
    return latest


def _trained_reg(df: pd.DataFrame):
    from aquasense.models import RULRegressor
    reg = RULRegressor(n_estimators=10, max_depth=4)
    reg.fit(df)
    return reg


def _anomaly_df(df: pd.DataFrame) -> pd.DataFrame:
    from aquasense.models import AnomalyDetector
    det = AnomalyDetector()
    det.fit(df)
    return det.tag_dataframe(df)


def _km_df(df: pd.DataFrame) -> pd.DataFrame:
    from aquasense.models import DepthClusterer
    clr = DepthClusterer(n_clusters=3)
    clr.fit(df)
    return clr.tag_dataframe(df)


# ── shared fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def base_df():
    return _small_df(n_nodes=15, n_timesteps=30)


@pytest.fixture(scope="module")
def dashboard_inputs(base_df):
    """Pre-compute all dashboard inputs once for the whole module."""
    df            = base_df
    latest_df     = _latest(df)
    cluster_stats = _cluster_stats(df)
    reg           = _trained_reg(df)
    anomaly_df    = _anomaly_df(df)
    km_df         = _km_df(df)
    return {
        "df":                  df,
        "latest_df":           latest_df,
        "cluster_stats":       cluster_stats,
        "y_test":              reg._y_test,
        "y_pred":              reg._y_pred,
        "anomaly_df":          anomaly_df,
        "km_df":               km_df,
        "feature_importances": reg.feature_importances_,
    }


# ══════════════════════════════════════════════════════════════════════════
# Individual panel functions
# ══════════════════════════════════════════════════════════════════════════

class TestPanelFunctions:
    """Each panel helper should run without error and return None."""

    def _ax(self):
        fig, ax = plt.subplots()
        return fig, ax

    def test_style_ax_no_title(self, dashboard_inputs):
        from aquasense.visualise import _style_ax
        fig, ax = self._ax()
        _style_ax(ax)          # no title — should not raise
        plt.close(fig)

    def test_style_ax_with_title(self, dashboard_inputs):
        from aquasense.visualise import _style_ax
        fig, ax = self._ax()
        _style_ax(ax, title="TEST TITLE")
        plt.close(fig)

    def test_panel_kpi(self, dashboard_inputs):
        from aquasense.visualise import _panel_kpi
        fig, ax = self._ax()
        n_anom = int(dashboard_inputs["anomaly_df"]["anomaly_pred"].sum())
        _panel_kpi(ax, dashboard_inputs["latest_df"], n_anom)
        plt.close(fig)

    def test_panel_cluster_rul(self, dashboard_inputs):
        from aquasense.visualise import _panel_cluster_rul
        fig, ax = self._ax()
        _panel_cluster_rul(ax, dashboard_inputs["cluster_stats"])
        plt.close(fig)

    def test_panel_rul_scatter(self, dashboard_inputs):
        from aquasense.visualise import _panel_rul_scatter
        fig, ax = self._ax()
        _panel_rul_scatter(
            ax,
            dashboard_inputs["y_test"],
            dashboard_inputs["y_pred"],
        )
        plt.close(fig)

    def test_panel_battery_decay(self, dashboard_inputs):
        from aquasense.visualise import _panel_battery_decay
        fig, ax = self._ax()
        _panel_battery_decay(ax, dashboard_inputs["df"])
        plt.close(fig)

    def test_panel_depth_vs_rul(self, dashboard_inputs):
        from aquasense.visualise import _panel_depth_vs_rul
        fig, ax = self._ax()
        _panel_depth_vs_rul(ax, dashboard_inputs["latest_df"])
        plt.close(fig)

    def test_panel_anomaly_distribution(self, dashboard_inputs):
        from aquasense.visualise import _panel_anomaly_distribution
        fig, ax = self._ax()
        _panel_anomaly_distribution(ax, dashboard_inputs["anomaly_df"])
        plt.close(fig)

    def test_panel_kmeans(self, dashboard_inputs):
        from aquasense.visualise import _panel_kmeans
        fig, ax = self._ax()
        _panel_kmeans(ax, dashboard_inputs["km_df"])
        plt.close(fig)

    def test_panel_feature_importance(self, dashboard_inputs):
        from aquasense.visualise import _panel_feature_importance
        fig, ax = self._ax()
        _panel_feature_importance(ax, dashboard_inputs["feature_importances"])
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# build_dashboard integration test
# ══════════════════════════════════════════════════════════════════════════

class TestBuildDashboard:

    def test_build_dashboard_creates_file(self, dashboard_inputs, tmp_path):
        """build_dashboard should save a PNG and return its Path."""
        from aquasense.visualise import build_dashboard
        out = tmp_path / "test_dashboard.png"
        result = build_dashboard(
            df=dashboard_inputs["df"],
            latest_df=dashboard_inputs["latest_df"],
            cluster_stats=dashboard_inputs["cluster_stats"],
            y_test=dashboard_inputs["y_test"],
            y_pred=dashboard_inputs["y_pred"],
            anomaly_df=dashboard_inputs["anomaly_df"],
            km_df=dashboard_inputs["km_df"],
            feature_importances=dashboard_inputs["feature_importances"],
            output_path=out,
        )
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 10_000   # non-trivial PNG

    def test_build_dashboard_returns_path(self, dashboard_inputs, tmp_path):
        """Return value should be a Path object."""
        from aquasense.visualise import build_dashboard
        out = tmp_path / "dash2.png"
        result = build_dashboard(
            df=dashboard_inputs["df"],
            latest_df=dashboard_inputs["latest_df"],
            cluster_stats=dashboard_inputs["cluster_stats"],
            y_test=dashboard_inputs["y_test"],
            y_pred=dashboard_inputs["y_pred"],
            anomaly_df=dashboard_inputs["anomaly_df"],
            km_df=dashboard_inputs["km_df"],
            feature_importances=dashboard_inputs["feature_importances"],
            output_path=out,
        )
        assert isinstance(result, Path)

    def test_build_dashboard_default_path(self, dashboard_inputs, tmp_path,
                                          monkeypatch):
        """
        When output_path is None, the file should be saved to OUTPUT_DIR.
        We monkeypatch OUTPUT_DIR to tmp_path so the test is self-contained.
        """
        import aquasense.visualise as vis_mod
        import aquasense.config    as cfg_mod

        monkeypatch.setattr(vis_mod, "OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(cfg_mod, "OUTPUT_DIR", tmp_path)

        from aquasense.visualise import build_dashboard
        result = build_dashboard(
            df=dashboard_inputs["df"],
            latest_df=dashboard_inputs["latest_df"],
            cluster_stats=dashboard_inputs["cluster_stats"],
            y_test=dashboard_inputs["y_test"],
            y_pred=dashboard_inputs["y_pred"],
            anomaly_df=dashboard_inputs["anomaly_df"],
            km_df=dashboard_inputs["km_df"],
            feature_importances=dashboard_inputs["feature_importances"],
            output_path=None,
        )
        assert result.exists()
        assert result.suffix == ".png"

    def test_dashboard_all_clusters_present(self, base_df):
        """
        Dashboard should work even when not all three depth clusters appear
        in a small simulated dataset — no KeyError should be raised.
        """
        from aquasense.visualise import build_dashboard
        import tempfile, os
        df        = _small_df(n_nodes=5, n_timesteps=10)
        latest_df = _latest(df)
        stats     = _cluster_stats(df)
        reg       = _trained_reg(df)
        anom_df   = _anomaly_df(df)
        km_df_    = _km_df(df)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "d.png"
            build_dashboard(
                df=df, latest_df=latest_df, cluster_stats=stats,
                y_test=reg._y_test, y_pred=reg._y_pred,
                anomaly_df=anom_df, km_df=km_df_,
                feature_importances=reg.feature_importances_,
                output_path=out,
            )
            assert out.exists()


# ══════════════════════════════════════════════════════════════════════════
# Palette / config sanity checks
# ══════════════════════════════════════════════════════════════════════════

class TestPaletteConfig:

    def test_palette_has_required_keys(self):
        from aquasense.config import PALETTE
        required = {"bg", "panel", "border", "accent1", "accent2",
                    "accent3", "warn", "text", "muted",
                    "shallow", "mid", "deep"}
        assert required.issubset(PALETTE.keys())

    def test_palette_values_are_hex(self):
        from aquasense.config import PALETTE
        for key, val in PALETTE.items():
            assert val.startswith("#"), \
                f"PALETTE['{key}'] = {val!r} is not a hex colour"

    def test_cluster_colors_covers_all_clusters(self):
        from aquasense.config import CLUSTER_COLORS
        assert set(CLUSTER_COLORS.keys()) == {"shallow", "mid", "deep"}

    def test_km_colors_length(self):
        from aquasense.config import KM_COLORS, KM_N_CLUSTERS
        assert len(KM_COLORS) >= KM_N_CLUSTERS

    def test_dashboard_size_is_tuple(self):
        from aquasense.config import DASHBOARD_SIZE
        assert isinstance(DASHBOARD_SIZE, tuple)
        assert len(DASHBOARD_SIZE) == 2

    def test_dashboard_dpi_positive(self):
        from aquasense.config import DASHBOARD_DPI
        assert DASHBOARD_DPI > 0
