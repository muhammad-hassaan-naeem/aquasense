"""
tests/test_phase1.py
---------------------
Phase 1 test suite — ARGO connector, LSTM model, model comparison.
Run with:  pytest tests/ -v
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


def _small_df(n_nodes=15, n_timesteps=30):
    from aquasense.simulate import simulate_sensor_data
    return simulate_sensor_data(n_nodes=n_nodes, n_timesteps=n_timesteps, random_seed=7)


# ══════════════════════════════════════════════════════════════════════════
# ARGO Connector
# ══════════════════════════════════════════════════════════════════════════

class TestArgoConnector:

    @pytest.fixture(scope="class")
    def connector(self, tmp_path_factory):
        from aquasense.phase1.data.argo_connector import ArgoConnector
        return ArgoConnector(
            cache_dir=tmp_path_factory.mktemp("argo_cache"),
            random_seed=0,
        )

    def test_generate_fallback_schema(self, connector):
        """ARGO fallback data must match AquaSense schema."""
        df = connector._generate_argo_realistic(n_floats=5, n_profiles=20)
        required = {"node_id","timestep","depth_m","pressure_bar",
                    "salinity_ppt","temperature_c","battery_voltage",
                    "tx_freq_ppm","packet_success_rt","depth_cluster",
                    "is_anomaly","rul_hours"}
        assert required.issubset(df.columns)

    def test_generate_fallback_shape(self, connector):
        df = connector._generate_argo_realistic(n_floats=5, n_profiles=20)
        assert len(df) == 5 * 20

    def test_fallback_no_nulls(self, connector):
        df = connector._generate_argo_realistic(n_floats=5, n_profiles=10)
        assert df[["depth_m","temperature_c","salinity_ppt",
                   "battery_voltage","rul_hours"]].isnull().sum().sum() == 0

    def test_fallback_depth_positive(self, connector):
        df = connector._generate_argo_realistic(n_floats=5, n_profiles=10)
        assert (df["depth_m"] >= 0).all()

    def test_fallback_psr_in_range(self, connector):
        df = connector._generate_argo_realistic(n_floats=5, n_profiles=10)
        assert df["packet_success_rt"].between(0, 1).all()

    def test_validate_schema_passes(self, connector):
        df = connector._generate_argo_realistic(n_floats=3, n_profiles=10)
        assert connector.validate_schema(df) is True

    def test_validate_schema_fails_on_missing_col(self, connector):
        df = connector._generate_argo_realistic(n_floats=3, n_profiles=10)
        df = df.drop(columns=["rul_hours"])
        with pytest.raises(ValueError, match="Missing"):
            connector.validate_schema(df)

    def test_load_or_fetch_uses_cache(self, connector):
        """Second call should use cached file."""
        df1 = connector.load_or_fetch(n_floats=3, region="global")
        df2 = connector.load_or_fetch(n_floats=3, region="global")
        pd.testing.assert_frame_equal(df1.reset_index(drop=True),
                                      df2.reset_index(drop=True))

    def test_thermocline(self):
        from aquasense.phase1.data.argo_connector import ArgoConnector
        surface = ArgoConnector._thermocline(0,   lat=15)
        deep    = ArgoConnector._thermocline(1000, lat=15)
        assert surface > deep, "Surface should be warmer than deep water"

    def test_halocline(self):
        from aquasense.phase1.data.argo_connector import ArgoConnector
        sal_surface = ArgoConnector._halocline(0)
        sal_deep    = ArgoConnector._halocline(1000)
        # Deep water converges toward ~35 ppt
        assert abs(sal_deep - 35.0) < 1.0

    def test_compare_with_simulation(self, connector):
        sim_df  = _small_df()
        argo_df = connector._generate_argo_realistic(n_floats=5, n_profiles=20)
        comp    = connector.compare_with_simulation(argo_df, sim_df)
        assert "feature" in comp.columns
        assert "diff_mean_pct" in comp.columns
        assert len(comp) >= 3


# ══════════════════════════════════════════════════════════════════════════
# NOAA Connector
# ══════════════════════════════════════════════════════════════════════════

class TestNOAAConnector:

    @pytest.fixture(scope="class")
    def noaa(self, tmp_path_factory):
        from aquasense.phase1.data.noaa_connector import NOAAConnector
        return NOAAConnector(
            cache_dir=tmp_path_factory.mktemp("noaa_cache"),
            random_seed=0,
        )

    def test_fetch_climatology_schema(self, noaa):
        df = noaa.fetch_climatology(region="arabian_sea", n_profiles=5)
        assert "depth_m" in df.columns
        assert "temperature_c" in df.columns
        assert "salinity_ppt" in df.columns

    def test_fetch_climatology_depth_range(self, noaa):
        df = noaa.fetch_climatology(
            region="arabian_sea", n_profiles=5, max_depth=500)
        assert df["depth_m"].max() <= 500

    def test_basin_statistics_shape(self, noaa):
        stats = noaa.basin_statistics()
        assert "basin" in stats.columns
        assert "depth_m" in stats.columns
        assert len(stats) > 0

    def test_temperature_decreases_with_depth(self, noaa):
        t_surf = noaa._woa_temperature(0,    basin="arabian_sea")
        t_deep = noaa._woa_temperature(1000, basin="arabian_sea")
        assert t_surf > t_deep

    def test_pakistan_summary_is_string(self, noaa):
        s = noaa.pakistan_ocean_summary()
        assert isinstance(s, str)
        assert "Arabian Sea" in s or "Pakistan" in s


# ══════════════════════════════════════════════════════════════════════════
# LSTM RUL Predictor
# ══════════════════════════════════════════════════════════════════════════

class TestLSTMRULPredictor:

    @pytest.fixture(scope="class")
    def fitted_lstm(self):
        from aquasense.phase1.models.lstm_rul import LSTMRULPredictor
        df    = _small_df(n_nodes=15, n_timesteps=30)
        model = LSTMRULPredictor(
            seq_len=5, hidden_size=16, num_layers=1,
            epochs=3, batch_size=32, patience=2)
        model.fit(df)
        return model

    def test_metrics_set_after_fit(self, fitted_lstm):
        assert "mae" in fitted_lstm.metrics_
        assert "r2"  in fitted_lstm.metrics_

    def test_mae_is_positive(self, fitted_lstm):
        assert fitted_lstm.metrics_["mae"] > 0

    def test_r2_is_valid(self, fitted_lstm):
        # R² can be negative for very bad models, but should be finite
        assert np.isfinite(fitted_lstm.metrics_["r2"])

    def test_predict_returns_array(self, fitted_lstm):
        df   = _small_df(n_nodes=3, n_timesteps=10)
        pred = fitted_lstm.predict(df)
        assert isinstance(pred, np.ndarray)
        assert len(pred) > 0

    def test_predict_non_negative(self, fitted_lstm):
        df   = _small_df(n_nodes=3, n_timesteps=10)
        pred = fitted_lstm.predict(df)
        assert (pred >= 0).all()

    def test_training_history_recorded(self, fitted_lstm):
        h = fitted_lstm.training_history_
        assert len(h["train_loss"]) > 0
        assert len(h["val_loss"])   > 0

    def test_save_and_load(self, fitted_lstm, tmp_path):
        from aquasense.phase1.models.lstm_rul import LSTMRULPredictor
        p      = tmp_path / "lstm.pt"
        fitted_lstm.save(p)
        loaded = LSTMRULPredictor.load(p)
        df     = _small_df(n_nodes=3, n_timesteps=10)
        p1     = fitted_lstm.predict(df)
        p2     = loaded.predict(df)
        np.testing.assert_allclose(p1, p2, rtol=1e-4)

    def test_repr_after_fit(self, fitted_lstm):
        s = repr(fitted_lstm)
        assert "LSTMRULPredictor" in s
        assert "MAE" in s

    def test_repr_before_fit(self):
        from aquasense.phase1.models.lstm_rul import LSTMRULPredictor
        m = LSTMRULPredictor()
        assert "unfitted" in repr(m)


# ══════════════════════════════════════════════════════════════════════════
# Model Comparison
# ══════════════════════════════════════════════════════════════════════════

class TestModelComparison:

    @pytest.fixture(scope="class")
    def comp_results(self):
        from aquasense.phase1.models.model_comparison import ModelComparison
        df   = _small_df(n_nodes=15, n_timesteps=30)
        comp = ModelComparison(lstm_seq_len=5, lstm_epochs=3, n_bootstrap=20)
        results = comp.run(df_synthetic=df, df_real=None)
        return comp, results

    def test_results_has_both_models(self, comp_results):
        _, results = comp_results
        models = results["model"].unique().tolist()
        assert "Random Forest" in models
        assert "LSTM" in models

    def test_results_columns(self, comp_results):
        _, results = comp_results
        for col in ["model","dataset","mae","rmse","r2","mape_pct","fit_time_s"]:
            assert col in results.columns

    def test_rf_r2_positive(self, comp_results):
        _, results = comp_results
        rf_r2 = results[results["model"] == "Random Forest"]["r2"].values[0]
        assert rf_r2 > 0

    def test_csv_saved(self, comp_results):
        from aquasense.config import METRICS_DIR
        assert (METRICS_DIR / "model_comparison.csv").exists()

    def test_plot_comparison_creates_file(self, comp_results):
        comp, _ = comp_results
        from aquasense.config import FIGURES_DIR
        comp.plot_comparison()
        assert (FIGURES_DIR / "model_comparison_rf_vs_lstm.png").exists()

    def test_print_report_runs(self, comp_results):
        comp, results = comp_results
        comp.print_report(results)  # should not raise


# ══════════════════════════════════════════════════════════════════════════
# Phase 1 Pipeline Integration
# ══════════════════════════════════════════════════════════════════════════

class TestPhase1Pipeline:

    def test_pipeline_runs_no_real_data(self, tmp_path):
        """Full pipeline should complete without errors in no-real-data mode."""
        import argparse
        from aquasense.phase1.pipeline import run_phase1
        args = argparse.Namespace(
            nodes=10, timesteps=20, seed=1,
            n_floats=5, region="global",
            lstm_epochs=2, lstm_seq_len=5,
            no_real_data=True, no_dashboard=True,
            quiet=True,
        )
        results = run_phase1(args)
        assert "sim_df" in results
        assert len(results["sim_df"]) == 10 * 20

    def test_pipeline_returns_models(self, tmp_path):
        import argparse
        from aquasense.phase1.pipeline import run_phase1
        args = argparse.Namespace(
            nodes=10, timesteps=20, seed=2,
            n_floats=3, region="global",
            lstm_epochs=2, lstm_seq_len=5,
            no_real_data=True, no_dashboard=True,
            quiet=True,
        )
        results = run_phase1(args)
        assert results.get("rf")   is not None
        assert results.get("lstm") is not None
