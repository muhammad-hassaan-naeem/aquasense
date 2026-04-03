"""
models.py
---------
Machine-learning components for AquaSense.

Three independent estimators are implemented as self-contained classes so
they can be trained, evaluated, serialised, and swapped independently.

Classes
-------
RULRegressor
    Random Forest regressor that predicts *Remaining Useful Life* (hours)
    from sensor telemetry features.

AnomalyDetector
    Isolation Forest that flags readings whose multivariate feature
    distribution deviates significantly from the majority.

DepthClusterer
    K-Means model that groups nodes by depth / battery / transmission
    behaviour without relying on hand-crafted cluster labels.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    FEATURES,
    IF_CONTAMINATION,
    KM_N_CLUSTERS,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RF_TEST_SIZE,
    SIM_RANDOM_SEED,
)


# ── RUL Regressor ──────────────────────────────────────────────────────────

class RULRegressor:
    """
    Predicts the *Remaining Useful Life* (hours) of a sensor node's battery.

    The underlying estimator is a Random Forest trained on the seven
    environmental / electrical features defined in ``config.FEATURES``.

    Attributes
    ----------
    model : RandomForestRegressor
    scaler : StandardScaler
    feature_importances_ : np.ndarray  (set after ``fit``)
    metrics_ : dict  (``mae``, ``r2``  — set after ``fit``)
    """

    def __init__(
        self,
        n_estimators: int = RF_N_ESTIMATORS,
        max_depth:    int = RF_MAX_DEPTH,
        random_state: int = SIM_RANDOM_SEED,
    ) -> None:
        self.model  = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler  = StandardScaler()
        self.metrics_: dict = {}
        self.feature_importances_: Optional[np.ndarray] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_test: Optional[pd.Series]    = None
        self._y_pred: Optional[np.ndarray]   = None

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "RULRegressor":
        """
        Train the regressor on *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all columns in ``config.FEATURES`` plus ``rul_hours``.

        Returns
        -------
        self
        """
        X = df[FEATURES]
        y = df["rul_hours"]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=RF_TEST_SIZE, random_state=SIM_RANDOM_SEED
        )
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        self.model.fit(X_tr_s, y_tr)
        y_pred = self.model.predict(X_te_s)

        self.feature_importances_ = self.model.feature_importances_
        self._X_test = X_te
        self._y_test = y_te
        self._y_pred = y_pred
        self.metrics_ = {
            "mae": float(mean_absolute_error(y_te, y_pred)),
            "r2" : float(r2_score(y_te, y_pred)),
        }
        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict RUL (hours) for each row in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all columns in ``config.FEATURES``.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        X_s = self.scaler.transform(df[FEATURES])
        return self.model.predict(X_s)

    # ── Serialisation ─────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Pickle the trained regressor + scaler to *path*."""
        with open(path, "wb") as fh:
            pickle.dump({"model": self.model, "scaler": self.scaler}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "RULRegressor":
        """Load a previously saved ``RULRegressor`` from *path*."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.model   = state["model"]
        obj.scaler  = state["scaler"]
        obj.metrics_ = {}
        obj.feature_importances_ = obj.model.feature_importances_
        return obj

    def __repr__(self) -> str:
        m = self.metrics_
        if m:
            return f"RULRegressor(MAE={m['mae']:.2f}h, R²={m['r2']:.4f})"
        return "RULRegressor(unfitted)"


# ── Anomaly Detector ───────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Flags sensor readings that deviate from the expected operational envelope.

    Uses an Isolation Forest — an ensemble of random trees that isolates
    anomalies in fewer splits than normal points.  No labelled anomaly data
    is needed for training (unsupervised).

    Attributes
    ----------
    model : IsolationForest
    scaler : StandardScaler
    metrics_ : dict  (``precision``, ``recall``, ``f1`` — set after ``fit``)
    """

    def __init__(
        self,
        contamination: float = IF_CONTAMINATION,
        random_state:  int   = SIM_RANDOM_SEED,
    ) -> None:
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler   = StandardScaler()
        self.metrics_: dict = {}

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """
        Fit the Isolation Forest on *df* and evaluate against ground-truth
        ``is_anomaly`` labels (if present).
        """
        X_s = self.scaler.fit_transform(df[FEATURES])
        self.model.fit(X_s)

        if "is_anomaly" in df.columns:
            preds    = (self.model.predict(X_s) == -1).astype(int)
            true     = df["is_anomaly"].values
            tp       = int((preds & true).sum())
            prec_den = int(preds.sum())
            rec_den  = int(true.sum())
            precision = tp / prec_den if prec_den else 0.0
            recall    = tp / rec_den  if rec_den  else 0.0
            f1        = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) else 0.0
            )
            self.metrics_ = {
                "precision": precision,
                "recall"   : recall,
                "f1"       : f1,
            }
        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return a binary anomaly flag (0 = normal, 1 = anomalous) for each row.
        """
        X_s = self.scaler.transform(df[FEATURES])
        return (self.model.predict(X_s) == -1).astype(int)

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return the raw anomaly score (lower = more anomalous).
        Useful for ranking nodes by risk.
        """
        X_s = self.scaler.transform(df[FEATURES])
        return self.model.score_samples(X_s)

    def tag_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of *df* with two extra columns:
        ``anomaly_pred`` (0/1) and ``anomaly_score`` (float).
        """
        out = df.copy()
        out["anomaly_pred"]  = self.predict(df)
        out["anomaly_score"] = self.score_samples(df)
        return out

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"model": self.model, "scaler": self.scaler}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.model  = state["model"]
        obj.scaler = state["scaler"]
        obj.metrics_ = {}
        return obj

    def __repr__(self) -> str:
        m = self.metrics_
        if m:
            return (
                f"AnomalyDetector("
                f"P={m['precision']:.3f}, R={m['recall']:.3f}, "
                f"F1={m['f1']:.3f})"
            )
        return "AnomalyDetector(unfitted)"


# ── Depth Clusterer ────────────────────────────────────────────────────────

class DepthClusterer:
    """
    Data-driven node grouping via K-Means.

    Unlike the hand-crafted shallow / mid / deep labels used during
    simulation, this model discovers clusters purely from the sensor readings,
    providing an independent validation of the depth-based taxonomy.

    Clustering features: depth_m, battery_voltage, tx_freq_ppm,
                         packet_success_rt.
    """

    CLUSTER_FEATURES = [
        "depth_m", "battery_voltage", "tx_freq_ppm", "packet_success_rt"
    ]

    def __init__(
        self,
        n_clusters:   int = KM_N_CLUSTERS,
        random_state: int = SIM_RANDOM_SEED,
    ) -> None:
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        self.scaler = StandardScaler()
        self.n_clusters = n_clusters

    def fit(self, df: pd.DataFrame) -> "DepthClusterer":
        X_s = self.scaler.fit_transform(df[self.CLUSTER_FEATURES])
        self.model.fit(X_s)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X_s = self.scaler.transform(df[self.CLUSTER_FEATURES])
        return self.model.predict(X_s)

    def tag_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with a ``km_cluster`` column added."""
        out = df.copy()
        out["km_cluster"] = self.predict(df)
        return out

    def cluster_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return per-cluster mean statistics for the clustering features.
        """
        tagged = self.tag_dataframe(df)
        return (
            tagged.groupby("km_cluster")[self.CLUSTER_FEATURES + ["rul_hours"]]
            .mean()
            .round(3)
            .sort_values("depth_m")
        )

    def __repr__(self) -> str:
        return f"DepthClusterer(n_clusters={self.n_clusters})"
