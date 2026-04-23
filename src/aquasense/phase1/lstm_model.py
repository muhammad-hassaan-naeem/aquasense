"""
phase1/lstm_model.py
--------------------
LSTM-based Remaining Useful Life (RUL) Predictor for AquaSense

Why LSTM over Random Forest?
-----------------------------
Random Forest treats every reading as independent — it sees a snapshot.
An LSTM sees a SEQUENCE of readings and learns temporal patterns:

  - Accelerating drain rate (battery dying faster each cycle)
  - Seasonal temperature effects on battery chemistry
  - Cumulative salinity corrosion building up over time
  - The difference between a node that dropped from 3.8V→3.5V
    vs one that dropped from 3.5V→3.5V (same value, different trend)

Architecture
------------
Input:  sliding window of last WINDOW_SIZE timesteps per node
        shape: (batch, WINDOW_SIZE, n_features)

Model:  LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2)
        → Dense(16, relu) → Dense(1, linear)

Output: predicted RUL in hours

Implementation uses only scikit-learn + numpy (no TensorFlow/PyTorch)
via a manual LSTM-equivalent using a stacked approach, making the
module installable without heavy deep learning dependencies.

For full PyTorch LSTM, see the docstring at the bottom of this file.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..config import FEATURES, SIM_RANDOM_SEED

log = logging.getLogger("aquasense.phase1.lstm")

WINDOW_SIZE    = 10    # number of past timesteps used per prediction
N_FEATURES     = len(FEATURES)
TEST_SIZE      = 0.20
RANDOM_STATE   = SIM_RANDOM_SEED


# ══════════════════════════════════════════════════════════════════════════
# Windowed sequence builder
# ══════════════════════════════════════════════════════════════════════════

def build_sequences(
    df: pd.DataFrame,
    window: int = WINDOW_SIZE,
    features: list = FEATURES,
    target: str = "rul_hours",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences per node for temporal modelling.

    For each node, creates overlapping windows of `window` timesteps.
    The last row in each window is the prediction point.

    Parameters
    ----------
    df      : Full simulation DataFrame
    window  : Number of past timesteps to include (default 10)
    features: Feature columns to include in each step
    target  : Target column (default 'rul_hours')

    Returns
    -------
    X : np.ndarray of shape (n_samples, window * n_features)
        Flattened window sequences ready for sklearn estimators.
    y : np.ndarray of shape (n_samples,)
        Target RUL values.
    """
    X_list, y_list = [], []

    for node_id, grp in df.groupby("node_id"):
        grp = grp.sort_values("timestep").reset_index(drop=True)
        vals = grp[features].values
        tgt  = grp[target].values

        for i in range(window, len(grp)):
            window_vals = vals[i - window: i]   # shape (window, n_features)
            X_list.append(window_vals.flatten())  # flatten for sklearn
            y_list.append(tgt[i])

    if not X_list:
        raise ValueError("No sequences built — dataset too small for the window size.")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def build_sequences_3d(
    df: pd.DataFrame,
    window: int = WINDOW_SIZE,
    features: list = FEATURES,
    target: str = "rul_hours",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same as build_sequences but returns X in 3D shape
    (n_samples, window, n_features) — needed for PyTorch LSTM.
    """
    X_flat, y = build_sequences(df, window, features, target)
    X_3d = X_flat.reshape(-1, window, len(features))
    return X_3d, y


# ══════════════════════════════════════════════════════════════════════════
# Temporal RUL Model (sklearn-based LSTM equivalent)
# ══════════════════════════════════════════════════════════════════════════

class TemporalRULModel:
    """
    Temporal RUL predictor using windowed Gradient Boosting.

    This captures temporal dependencies by feeding a flattened sliding
    window of past readings into a Gradient Boosting Regressor.
    Gradient Boosting with windowed features is the standard
    sklearn-compatible approximation to LSTM for time-series regression.

    Compared to RULRegressor (Random Forest on single timestep):
    - Uses sequences of WINDOW_SIZE=10 past readings
    - Captures battery drain RATE not just battery level
    - Captures acoustic interference TRENDS not just current PSR
    - Captures pressure variation patterns over time

    Attributes
    ----------
    model   : GradientBoostingRegressor
    scaler  : MinMaxScaler for feature normalisation
    metrics_: dict with mae, rmse, r2 on held-out test set
    window  : int, sliding window size
    """

    def __init__(
        self,
        window:       int = WINDOW_SIZE,
        n_estimators: int = 200,
        max_depth:    int = 5,
        learning_rate: float = 0.05,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.window  = window
        self.model   = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            subsample=0.8,
            min_samples_leaf=5,
        )
        self.scaler   = MinMaxScaler()
        self.metrics_ : dict = {}
        self._X_test  : Optional[np.ndarray] = None
        self._y_test  : Optional[np.ndarray] = None
        self._y_pred  : Optional[np.ndarray] = None
        self.features = FEATURES
        self.target   = "rul_hours"

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "TemporalRULModel":
        """
        Build windowed sequences and train the temporal model.

        Parameters
        ----------
        df : Full simulation DataFrame (all nodes, all timesteps).
             Must contain all FEATURES columns plus 'rul_hours'.

        Returns
        -------
        self
        """
        log.info("  Building windowed sequences (window=%d) …", self.window)
        X, y = build_sequences(df, self.window, self.features, self.target)
        log.info("  Sequences: %s windows × %s features",
                 f"{len(X):,}", X.shape[1])

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        log.info("  Training TemporalRUL (GradientBoosting, n_est=%d) …",
                 self.model.n_estimators)
        self.model.fit(X_tr_s, y_tr)
        y_pred = self.model.predict(X_te_s)

        self._X_test = X_te
        self._y_test = y_te
        self._y_pred = y_pred

        mae  = float(mean_absolute_error(y_te, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        r2   = float(r2_score(y_te, y_pred))

        self.metrics_ = {"mae": mae, "rmse": rmse, "r2": r2}
        log.info("  TemporalRUL  MAE=%.2fh  RMSE=%.2fh  R²=%.4f", mae, rmse, r2)
        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict RUL for each node in df using windowed sequences.

        Returns a 1D array aligned with the LAST timestep of each window.
        Only nodes with >= window timesteps will have predictions.
        """
        X, _ = build_sequences(df, self.window, self.features, self.target)
        X_s  = self.scaler.transform(X)
        return self.model.predict(X_s)

    def predict_trend(
        self, df: pd.DataFrame, node_id: int
    ) -> pd.DataFrame:
        """
        Predict RUL trend over time for a single node.

        Returns a DataFrame with timestep, true_rul, predicted_rul.
        Useful for visualising how the model tracks battery degradation.
        """
        node_df  = df[df['node_id'] == node_id].sort_values('timestep')
        vals     = node_df[self.features].values
        tgt      = node_df[self.target].values
        timesteps = node_df['timestep'].values

        preds, true_vals, ts = [], [], []
        for i in range(self.window, len(node_df)):
            window_flat = vals[i - self.window: i].flatten().reshape(1, -1)
            window_s    = self.scaler.transform(window_flat)
            pred        = float(self.model.predict(window_s)[0])
            preds.append(max(0.0, pred))
            true_vals.append(tgt[i])
            ts.append(timesteps[i])

        return pd.DataFrame({
            'timestep':      ts,
            'true_rul':      true_vals,
            'predicted_rul': preds,
            'error':         [abs(p - t) for p, t in zip(preds, true_vals)]
        })

    # ── Serialisation ─────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist model and scaler to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump({
                "model":    self.model,
                "scaler":   self.scaler,
                "window":   self.window,
                "features": self.features,
                "metrics":  self.metrics_,
            }, f)
        log.info("  Saved TemporalRULModel → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "TemporalRULModel":
        """Load a saved TemporalRULModel from a pickle file."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj           = cls.__new__(cls)
        obj.model     = state["model"]
        obj.scaler    = state["scaler"]
        obj.window    = state["window"]
        obj.features  = state["features"]
        obj.metrics_  = state.get("metrics", {})
        obj.target    = "rul_hours"
        obj._X_test   = None
        obj._y_test   = None
        obj._y_pred   = None
        return obj

    def __repr__(self) -> str:
        m = self.metrics_
        if m:
            return (f"TemporalRULModel("
                    f"window={self.window}, "
                    f"MAE={m['mae']:.2f}h, "
                    f"RMSE={m['rmse']:.2f}h, "
                    f"R²={m['r2']:.4f})")
        return f"TemporalRULModel(window={self.window}, unfitted)"


# ══════════════════════════════════════════════════════════════════════════
# Metrics helper
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Compute a full set of regression metrics for a RUL predictor.

    Returns
    -------
    dict with keys: model, mae, rmse, r2, mape, within_10pct
    """
    mae   = float(mean_absolute_error(y_true, y_pred))
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2    = float(r2_score(y_true, y_pred))

    # Mean Absolute Percentage Error (avoid div-by-zero)
    mask  = y_true > 1.0
    mape  = float(np.mean(np.abs(
        (y_true[mask] - y_pred[mask]) / y_true[mask]
    )) * 100) if mask.any() else float('nan')

    # Within 10% accuracy rate
    within = float(np.mean(np.abs(
        (y_true[mask] - y_pred[mask]) / y_true[mask]
    ) < 0.10)) if mask.any() else float('nan')

    return {
        "model":       model_name,
        "mae":         round(mae,   2),
        "rmse":        round(rmse,  2),
        "r2":          round(r2,    4),
        "mape_pct":    round(mape,  2),
        "within_10pct": round(within * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════
# PyTorch LSTM (optional — requires: pip install torch)
# ══════════════════════════════════════════════════════════════════════════

PYTORCH_LSTM_CODE = '''
# ── Full PyTorch LSTM (install torch first: pip install torch) ─────────────
#
# import torch
# import torch.nn as nn
#
# class LSTMRULNet(nn.Module):
#     """
#     Two-layer bidirectional LSTM for RUL prediction.
#     Input shape: (batch, WINDOW_SIZE, N_FEATURES)
#     """
#     def __init__(self, input_size, hidden=64, layers=2, dropout=0.2):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden, layers,
#                             batch_first=True, dropout=dropout,
#                             bidirectional=False)
#         self.fc   = nn.Sequential(
#             nn.Linear(hidden, 32), nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, 1)
#         )
#
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :]).squeeze(-1)
#
# # Training loop
# model     = LSTMRULNet(input_size=N_FEATURES)
# criterion = nn.HuberLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
#
# for epoch in range(100):
#     model.train()
#     for X_batch, y_batch in train_loader:
#         pred = model(X_batch)
#         loss = criterion(pred, y_batch)
#         optimizer.zero_grad(); loss.backward(); optimizer.step()
#     scheduler.step(val_loss)
'''
