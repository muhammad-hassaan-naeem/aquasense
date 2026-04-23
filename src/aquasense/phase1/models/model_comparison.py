"""
phase1/models/model_comparison.py
-----------------------------------
Random Forest vs LSTM Comparison Framework

Runs a rigorous side-by-side comparison of:
    - Random Forest RUL Regressor  (baseline — AquaSense v2.0)
    - LSTM RUL Predictor           (new — Phase 1)

Generates publication-ready comparison figures and a metrics CSV
suitable for a conference paper or thesis Chapter 4 results section.

Evaluation protocol:
    - Same train/test split for both models
    - 5 metrics: MAE, RMSE, R², MAPE, Training Time
    - Evaluated on both synthetic and real ARGO data
    - Statistical significance: bootstrapped confidence intervals

Usage
-----
    from aquasense.phase1.models.model_comparison import ModelComparison

    comp = ModelComparison()
    results = comp.run(df_synthetic, df_real)
    comp.print_report(results)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ...config import FIGURES_DIR, METRICS_DIR, PALETTE
from ...models import RULRegressor
from .lstm_rul import LSTMRULPredictor

log = logging.getLogger("aquasense.phase1.comparison")


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring zero targets."""
    mask = y_true > 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(
        (y_true[mask] - y_pred[mask]) / y_true[mask]
    )) * 100)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 200,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrapped confidence interval for a metric."""
    rng    = np.random.default_rng(42)
    scores = []
    n      = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lo = float(np.percentile(scores, (1-ci)/2*100))
    hi = float(np.percentile(scores, (1-(1-ci)/2)*100))
    return lo, hi


class ModelComparison:
    """
    Comprehensive comparison between RF and LSTM RUL predictors.

    Parameters
    ----------
    lstm_seq_len : Sequence length for LSTM sliding window
    lstm_epochs  : Max training epochs for LSTM
    n_bootstrap  : Bootstrap samples for confidence intervals
    """

    def __init__(
        self,
        lstm_seq_len: int = 10,
        lstm_epochs:  int = 40,
        n_bootstrap:  int = 200,
    ) -> None:
        self.lstm_seq_len = lstm_seq_len
        self.lstm_epochs  = lstm_epochs
        self.n_bootstrap  = n_bootstrap
        self.rf:   Optional[RULRegressor]    = None
        self.lstm: Optional[LSTMRULPredictor] = None
        self._results: Optional[pd.DataFrame] = None

    # ── Main entry-point ───────────────────────────────────────────────────

    def run(
        self,
        df_synthetic: pd.DataFrame,
        df_real:      Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Run full comparison pipeline.

        Parameters
        ----------
        df_synthetic : AquaSense simulated data
        df_real      : Real ARGO/NOAA data (optional second evaluation set)

        Returns
        -------
        pd.DataFrame  with all metric results
        """
        log.info("=" * 55)
        log.info("MODEL COMPARISON: Random Forest vs LSTM")
        log.info("=" * 55)

        rows = []

        # ── Evaluate on synthetic data ──────────────────────────────────
        log.info("\n[Synthetic Data]")
        rf_metrics  = self._evaluate_rf(df_synthetic, dataset="synthetic")
        rows.append(rf_metrics)

        lstm_metrics = self._evaluate_lstm(df_synthetic, dataset="synthetic")
        rows.append(lstm_metrics)

        # ── Evaluate on real data (if provided) ─────────────────────────
        if df_real is not None and len(df_real) > 0:
            log.info("\n[Real ARGO Data]")
            rf_real = self._evaluate_rf_on(
                self.rf, df_real, dataset="real_argo")
            rows.append(rf_real)

            lstm_real = self._evaluate_lstm_on(
                self.lstm, df_real, dataset="real_argo")
            rows.append(lstm_real)

        self._results = pd.DataFrame(rows)

        # Save CSV
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        out = METRICS_DIR / "model_comparison.csv"
        self._results.to_csv(out, index=False)
        log.info("\nSaved comparison table → %s", out)

        return self._results

    # ── Individual evaluations ─────────────────────────────────────────────

    def _evaluate_rf(
        self,
        df: pd.DataFrame,
        dataset: str = "synthetic",
    ) -> dict:
        log.info("  Training Random Forest …")
        t0       = time.perf_counter()
        self.rf  = RULRegressor()
        self.rf.fit(df)
        fit_time = time.perf_counter() - t0
        return self._compute_metrics(
            model_name="Random Forest",
            dataset=dataset,
            y_true=self.rf._y_test.values,
            y_pred=self.rf._y_pred,
            fit_time=fit_time,
        )

    def _evaluate_lstm(
        self,
        df: pd.DataFrame,
        dataset: str = "synthetic",
    ) -> dict:
        log.info("  Training LSTM …")
        t0        = time.perf_counter()
        self.lstm = LSTMRULPredictor(
            seq_len=self.lstm_seq_len,
            epochs=self.lstm_epochs,
        )
        self.lstm.fit(df)
        fit_time = time.perf_counter() - t0
        return self._compute_metrics(
            model_name="LSTM",
            dataset=dataset,
            y_true=self.lstm._y_test,
            y_pred=self.lstm._y_pred,
            fit_time=fit_time,
        )

    def _evaluate_rf_on(
        self,
        rf: RULRegressor,
        df: pd.DataFrame,
        dataset: str = "real_argo",
    ) -> dict:
        """Evaluate pre-trained RF on a new dataset."""
        if rf is None:
            return {}
        y_pred = rf.predict(df)
        y_true = df["rul_hours"].values[:len(y_pred)]
        return self._compute_metrics("Random Forest", dataset, y_true, y_pred, 0)

    def _evaluate_lstm_on(
        self,
        lstm: LSTMRULPredictor,
        df: pd.DataFrame,
        dataset: str = "real_argo",
    ) -> dict:
        """Evaluate pre-trained LSTM on a new dataset."""
        if lstm is None:
            return {}
        try:
            y_pred = lstm.predict(df)
            y_true = df["rul_hours"].values[:len(y_pred)]
            return self._compute_metrics("LSTM", dataset, y_true, y_pred, 0)
        except Exception as exc:
            log.warning("LSTM evaluation on %s failed: %s", dataset, exc)
            return {}

    def _compute_metrics(
        self,
        model_name: str,
        dataset:    str,
        y_true:     np.ndarray,
        y_pred:     np.ndarray,
        fit_time:   float,
    ) -> dict:
        """Compute all metrics with bootstrapped CIs."""
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = _rmse(y_true, y_pred)
        r2   = float(r2_score(y_true, y_pred))
        mape = _mape(y_true, y_pred)

        mae_lo,  mae_hi  = _bootstrap_ci(
            y_true, y_pred,
            lambda a,b: mean_absolute_error(a,b),
            self.n_bootstrap)

        log.info("  %-16s  MAE=%7.2f h  RMSE=%7.2f h  "
                 "R²=%.4f  MAPE=%5.1f%%  time=%.1fs",
                 model_name, mae, rmse, r2, mape, fit_time)

        return {
            "model"      : model_name,
            "dataset"    : dataset,
            "mae"        : round(mae,  2),
            "rmse"       : round(rmse, 2),
            "r2"         : round(r2,   4),
            "mape_pct"   : round(mape, 2),
            "mae_ci_low" : round(mae_lo, 2),
            "mae_ci_high": round(mae_hi, 2),
            "fit_time_s" : round(fit_time, 2),
        }

    # ── Visualisation ──────────────────────────────────────────────────────

    def plot_comparison(self) -> Path:
        """
        Generate a 6-panel publication-quality comparison figure.

        Panels:
            1. MAE comparison bar chart with CIs
            2. R² comparison bar chart
            3. RF predicted vs actual scatter
            4. LSTM predicted vs actual scatter
            5. LSTM training/validation loss curves
            6. Metrics summary table
        """
        if self._results is None or self.rf is None or self.lstm is None:
            raise RuntimeError("Run .run() before plotting.")

        fig = plt.figure(figsize=(18, 12), facecolor=PALETTE["bg"])
        fig.suptitle(
            "MODEL COMPARISON  ·  Random Forest vs LSTM  ·  RUL Prediction",
            fontsize=14, fontweight="bold", color=PALETTE["accent1"],
            y=0.98, fontfamily="monospace")

        gs = gridspec.GridSpec(2, 3, figure=fig,
                               hspace=0.45, wspace=0.35,
                               left=0.06, right=0.97,
                               top=0.92, bottom=0.06)

        def sax(ax, title):
            ax.set_facecolor(PALETTE["panel"])
            for sp in ax.spines.values():
                sp.set_color(PALETTE["border"]); sp.set_linewidth(1.2)
            ax.tick_params(colors=PALETTE["muted"], labelsize=8)
            ax.xaxis.label.set_color(PALETTE["muted"])
            ax.yaxis.label.set_color(PALETTE["muted"])
            ax.set_title(title, color=PALETTE["accent2"], fontsize=9,
                         fontweight="bold", fontfamily="monospace")
            ax.grid(color=PALETTE["border"], linestyle="--",
                    linewidth=0.5, alpha=0.5)

        syn = self._results[self._results["dataset"] == "synthetic"]

        # ── Panel 1: MAE bar chart ──────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        sax(ax1, "MEAN ABSOLUTE ERROR  (lower = better)")
        models = syn["model"].tolist()
        maes   = syn["mae"].tolist()
        lo_err = [m - l for m, l in zip(maes, syn["mae_ci_low"].tolist())]
        hi_err = [h - m for m, h in zip(maes, syn["mae_ci_high"].tolist())]
        colors = [PALETTE["accent1"], PALETTE["accent2"]]
        bars   = ax1.bar(models, maes, color=colors[:len(models)],
                         edgecolor=PALETTE["bg"], linewidth=1.5, width=0.4)
        ax1.errorbar(models, maes,
                     yerr=[lo_err, hi_err],
                     fmt="none", color=PALETTE["warn"],
                     capsize=5, linewidth=2)
        for bar, val in zip(bars, maes):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(maes)*0.02,
                     f"{val:.1f}h",
                     ha="center", color=PALETTE["text"],
                     fontsize=9, fontweight="bold")
        ax1.set_ylabel("MAE (hours)", fontsize=8)
        ax1.set_xticklabels(models, color=PALETTE["text"], fontsize=9)

        # ── Panel 2: R² bar chart ───────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        sax(ax2, "R²  SCORE  (higher = better)")
        r2s  = syn["r2"].tolist()
        bars = ax2.bar(models, r2s, color=colors[:len(models)],
                       edgecolor=PALETTE["bg"], linewidth=1.5, width=0.4)
        for bar, val in zip(bars, r2s):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() - 0.005,
                     f"{val:.4f}",
                     ha="center", va="top",
                     color=PALETTE["bg"], fontsize=9, fontweight="bold")
        ax2.set_ylabel("R² Score", fontsize=8)
        ax2.set_ylim(min(r2s) - 0.005, 1.002)
        ax2.set_xticklabels(models, color=PALETTE["text"], fontsize=9)

        # ── Panel 3: RF scatter ─────────────────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        sax(ax3, "RANDOM FOREST  ·  Predicted vs Actual")
        yt = self.rf._y_test.values
        yp = self.rf._y_pred
        ax3.scatter(yt, yp, alpha=0.25, s=8,
                    color=PALETTE["accent1"], edgecolors="none")
        lim = max(yt.max(), yp.max())
        ax3.plot([0, lim], [0, lim], color=PALETTE["accent3"],
                 linewidth=1.5, linestyle="--", label="Perfect")
        ax3.set_xlabel("Actual RUL (h)", fontsize=8)
        ax3.set_ylabel("Predicted RUL (h)", fontsize=8)
        rf_row = syn[syn["model"] == "Random Forest"].iloc[0]
        ax3.text(0.97, 0.08,
                 f"MAE={rf_row['mae']:.1f}h\nR²={rf_row['r2']:.4f}",
                 transform=ax3.transAxes, ha="right",
                 color=PALETTE["accent2"], fontsize=8,
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor=PALETTE["bg"], alpha=0.8,
                           edgecolor=PALETTE["border"]))

        # ── Panel 4: LSTM scatter ───────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 0])
        sax(ax4, "LSTM  ·  Predicted vs Actual")
        yt = self.lstm._y_test
        yp = self.lstm._y_pred
        ax4.scatter(yt, yp, alpha=0.25, s=8,
                    color=PALETTE["accent2"], edgecolors="none")
        lim = max(yt.max(), yp.max())
        ax4.plot([0, lim], [0, lim], color=PALETTE["accent3"],
                 linewidth=1.5, linestyle="--", label="Perfect")
        ax4.set_xlabel("Actual RUL (h)", fontsize=8)
        ax4.set_ylabel("Predicted RUL (h)", fontsize=8)
        lstm_row = syn[syn["model"] == "LSTM"].iloc[0]
        ax4.text(0.97, 0.08,
                 f"MAE={lstm_row['mae']:.1f}h\nR²={lstm_row['r2']:.4f}",
                 transform=ax4.transAxes, ha="right",
                 color=PALETTE["accent2"], fontsize=8,
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor=PALETTE["bg"], alpha=0.8,
                           edgecolor=PALETTE["border"]))

        # ── Panel 5: LSTM loss curves ───────────────────────────────────
        ax5 = fig.add_subplot(gs[1, 1])
        sax(ax5, "LSTM  TRAINING  CURVES")
        hist = self.lstm.training_history_
        epochs_range = range(1, len(hist["train_loss"]) + 1)
        ax5.plot(epochs_range, hist["train_loss"],
                 color=PALETTE["accent1"], linewidth=1.5, label="Train Loss")
        ax5.plot(epochs_range, hist["val_loss"],
                 color=PALETTE["accent3"], linewidth=1.5,
                 linestyle="--", label="Val Loss")
        ax5.set_xlabel("Epoch", fontsize=8)
        ax5.set_ylabel("MSE Loss", fontsize=8)
        ax5.legend(fontsize=7, facecolor=PALETTE["panel"],
                   labelcolor=PALETTE["text"], edgecolor=PALETTE["border"])

        # ── Panel 6: Metrics summary table ──────────────────────────────
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor(PALETTE["panel"])
        for sp in ax6.spines.values():
            sp.set_color(PALETTE["border"])
        ax6.set_xticks([]); ax6.set_yticks([])
        ax6.set_title("METRICS  SUMMARY  TABLE",
                      color=PALETTE["accent2"], fontsize=9,
                      fontweight="bold", fontfamily="monospace")

        metrics_to_show = ["mae", "rmse", "r2", "mape_pct", "fit_time_s"]
        labels          = ["MAE (h)", "RMSE (h)", "R²", "MAPE (%)", "Train (s)"]
        x_positions = [0.25, 0.65]
        ax6.text(0.05, 0.92, "Metric",
                 color=PALETTE["accent2"], fontsize=8,
                 fontweight="bold", transform=ax6.transAxes)
        for xi, model in enumerate(syn["model"].tolist()):
            ax6.text(x_positions[xi], 0.92, model,
                     color=PALETTE["accent2"], fontsize=8,
                     fontweight="bold", transform=ax6.transAxes,
                     ha="center")

        row_colors = [PALETTE["panel"], "#0a1828"]
        for ri, (metric, label) in enumerate(zip(metrics_to_show, labels)):
            y = 0.78 - ri * 0.15
            ax6.text(0.05, y, label,
                     color=PALETTE["text"], fontsize=8,
                     transform=ax6.transAxes)
            for xi, (_, row) in enumerate(syn.iterrows()):
                val = row[metric]
                txt = (f"{val:.4f}" if metric == "r2"
                       else f"{val:.1f}")
                ax6.text(x_positions[xi], y, txt,
                         color=PALETTE["accent1"] if xi == 0
                               else PALETTE["accent2"],
                         fontsize=8, fontweight="bold",
                         transform=ax6.transAxes, ha="center")

        fig.text(0.5, 0.005,
                 "AquaSense Phase 1  ·  RF vs LSTM RUL Comparison  "
                 "·  95% Bootstrap CIs shown on MAE",
                 ha="center", color=PALETTE["muted"], fontsize=7,
                 fontfamily="monospace")

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURES_DIR / "model_comparison_rf_vs_lstm.png"
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        plt.close(fig)
        log.info("Saved comparison figure → %s", out)
        return out

    # ── Report ─────────────────────────────────────────────────────────────

    def print_report(self, results: Optional[pd.DataFrame] = None) -> None:
        """Print a formatted comparison report to stdout."""
        df = results if results is not None else self._results
        if df is None:
            print("No results yet. Call .run() first.")
            return

        print("\n" + "=" * 60)
        print("  MODEL COMPARISON REPORT — AquaSense Phase 1")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60)

        syn = df[df["dataset"] == "synthetic"]
        if len(syn) >= 2:
            rf_mae   = syn[syn["model"] == "Random Forest"]["mae"].values[0]
            lstm_mae = syn[syn["model"] == "LSTM"]["mae"].values[0]
            improvement = (rf_mae - lstm_mae) / rf_mae * 100
            winner = "LSTM" if lstm_mae < rf_mae else "Random Forest"
            print(f"\n  Winner (MAE): {winner}")
            print(f"  MAE improvement: {abs(improvement):.1f}%")
            print("=" * 60 + "\n")
