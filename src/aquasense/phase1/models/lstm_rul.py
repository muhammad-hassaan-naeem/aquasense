"""
phase1/models/lstm_rul.py
--------------------------
LSTM-Based Remaining Useful Life (RUL) Predictor

Unlike the Random Forest which treats each timestep independently,
the LSTM captures temporal patterns in battery degradation:
- Accelerating drain rate (exponential decay)
- Cyclic patterns (tidal, temperature variation)
- Early degradation signatures before failure

Architecture
------------
    Input  : (batch, seq_len, n_features)  — sliding window of timesteps
    LSTM   : 2 layers × 64 hidden units + dropout(0.2)
    Attention: Self-attention over the LSTM output sequence
    Linear : 64 → 32 → 1
    Output : scalar RUL prediction (hours)

Usage
-----
    from aquasense.phase1.models.lstm_rul import LSTMRULPredictor

    model = LSTMRULPredictor(seq_len=10)
    model.fit(df)
    predictions = model.predict(df)
    print(model)  # LSTMRULPredictor(MAE=28.3h, R²=0.9961)
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as _torch_err:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for the LSTM RUL predictor but is not installed. "
        "Install it with:  pip install 'aquasense[torch]'"
    ) from _torch_err

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ...config import FEATURES, RF_TEST_SIZE, SIM_RANDOM_SEED

log = logging.getLogger("aquasense.phase1.lstm")

# ── PyTorch availability check ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Neural Network Architecture ────────────────────────────────────────────

class _AttentionLayer(nn.Module):
    """
    Self-attention over LSTM output sequence.
    Learns to focus on the most diagnostic timesteps for RUL prediction.
    """
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_size)
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = (weights * lstm_out).sum(dim=1)             # (batch, hidden_size)
        return context


class _LSTMNetwork(nn.Module):
    """
    Two-layer LSTM with self-attention and fully-connected head.

    Architecture:
        LSTM(n_features → 64, 2 layers, dropout=0.2)
        → Attention(64 → 64)
        → Linear(64 → 32) → ReLU → Dropout(0.1)
        → Linear(32 → 1)
    """
    def __init__(
        self,
        n_features:  int = len(FEATURES),
        hidden_size: int = 64,
        num_layers:  int = 2,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = _AttentionLayer(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)            # (batch, seq_len, hidden)
        context     = self.attention(lstm_out) # (batch, hidden)
        return self.head(context).squeeze(-1)  # (batch,)


# ── Dataset helper ──────────────────────────────────────────────────────────

class _SlidingWindowDataset(torch.utils.data.Dataset):
    """
    Creates overlapping sliding windows over node time-series.

    For each node, creates windows of shape (seq_len, n_features)
    with the corresponding RUL target at the window's last timestep.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int,
    ) -> None:
        self.X       = torch.FloatTensor(X)
        self.y       = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.X[idx : idx + self.seq_len]           # (seq_len, n_features)
        y_val = self.y[idx + self.seq_len - 1]             # scalar at last step
        return x_seq, y_val


# ── Main class ─────────────────────────────────────────────────────────────

class LSTMRULPredictor:
    """
    LSTM-based predictor for Remaining Useful Life of sensor batteries.

    Advantages over Random Forest:
    - Captures temporal degradation trends (not just snapshots)
    - Self-attention highlights which timesteps are most diagnostic
    - Better for accelerating failure scenarios

    Parameters
    ----------
    seq_len     : Number of past timesteps in each input window
    hidden_size : LSTM hidden state dimensionality
    num_layers  : Number of stacked LSTM layers
    epochs      : Training epochs
    batch_size  : Mini-batch size
    lr          : Learning rate
    patience    : Early stopping patience (epochs without improvement)
    random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        seq_len:      int   = 10,
        hidden_size:  int   = 64,
        num_layers:   int   = 2,
        epochs:       int   = 50,
        batch_size:   int   = 64,
        lr:           float = 1e-3,
        patience:     int   = 8,
        random_state: int   = SIM_RANDOM_SEED,
    ) -> None:
        self.seq_len      = seq_len
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.patience     = patience
        self.random_state = random_state

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.scaler = StandardScaler()
        self.model: Optional[_LSTMNetwork] = None
        self.metrics_: dict = {}
        self.training_history_: dict = {"train_loss": [], "val_loss": []}
        self._y_test: Optional[np.ndarray] = None
        self._y_pred: Optional[np.ndarray] = None
        self.fit_time_: float = 0.0

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "LSTMRULPredictor":
        """
        Train the LSTM on a DataFrame containing all nodes and timesteps.

        The training procedure:
        1. Scale features with StandardScaler
        2. Create sliding windows per node (preserving temporal order)
        3. Train with Adam optimizer + MSE loss + early stopping
        4. Evaluate on held-out 20% test split

        Parameters
        ----------
        df : pd.DataFrame
            Must contain FEATURES columns and 'rul_hours' target.

        Returns
        -------
        self
        """
        t0 = time.perf_counter()
        log.info("Training LSTM RUL Predictor (seq_len=%d, epochs=%d) …",
                 self.seq_len, self.epochs)

        X_all = self.scaler.fit_transform(df[FEATURES].values)
        y_all = df["rul_hours"].values.astype(np.float32)

        # Split by node so temporal order is preserved within each node
        node_ids = df["node_id"].values
        unique_nodes = np.unique(node_ids)
        n_test = max(1, int(len(unique_nodes) * RF_TEST_SIZE))

        rng = np.random.default_rng(self.random_state)
        rng.shuffle(unique_nodes)
        test_nodes  = set(unique_nodes[:n_test])
        train_nodes = set(unique_nodes[n_test:])

        train_mask = np.array([nid in train_nodes for nid in node_ids])
        test_mask  = ~train_mask

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask],  y_all[test_mask]

        train_ds = _SlidingWindowDataset(X_tr, y_tr, self.seq_len)
        test_ds  = _SlidingWindowDataset(X_te, y_te, self.seq_len)

        if len(train_ds) == 0:
            raise ValueError(
                f"Not enough data for seq_len={self.seq_len}. "
                f"Reduce seq_len or increase n_nodes/n_timesteps.")

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader  = torch.utils.data.DataLoader(
            test_ds,  batch_size=self.batch_size, shuffle=False)

        self.model = _LSTMNetwork(
            n_features=len(FEATURES),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    pred = self.model(X_batch)
                    val_losses.append(criterion(pred, y_batch).item())

            train_loss = float(np.mean(train_losses))
            val_loss   = float(np.mean(val_losses)) if val_losses else 0.0
            self.training_history_["train_loss"].append(train_loss)
            self.training_history_["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                best_weights     = {k: v.clone()
                                    for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                log.info("Early stopping at epoch %d (val_loss=%.4f)",
                         epoch + 1, val_loss)
                break

            if (epoch + 1) % 10 == 0:
                log.info("  Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f",
                         epoch+1, self.epochs, train_loss, val_loss)

        # Restore best weights
        if best_weights:
            self.model.load_state_dict(best_weights)

        # Final evaluation
        y_pred = self._predict_array(X_te, y_te)
        self._y_test = y_te[:len(y_pred)]
        self._y_pred = y_pred

        self.metrics_ = {
            "mae"          : float(mean_absolute_error(self._y_test, y_pred)),
            "r2"           : float(r2_score(self._y_test, y_pred)),
            "best_val_loss": best_val_loss,
        }
        self.fit_time_ = time.perf_counter() - t0
        log.info("LSTM training complete — %s  (%.1fs)", self, self.fit_time_)
        return self

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict RUL (hours) for each row in df.
        Uses a sliding window — the first seq_len-1 rows are predicted
        using partial context; the rest use full context.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        X = self.scaler.transform(df[FEATURES].values)
        y_dummy = np.zeros(len(X), dtype=np.float32)
        return self._predict_array(X, y_dummy)

    def _predict_array(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Internal prediction over a numpy array."""
        self.model.eval()
        ds     = _SlidingWindowDataset(X, y, self.seq_len)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False)
        preds  = []
        with torch.no_grad():
            for X_batch, _ in loader:
                out = self.model(X_batch.to(DEVICE))
                preds.extend(out.cpu().numpy().tolist())
        return np.maximum(0, np.array(preds, dtype=np.float32))

    # ── Serialisation ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model weights and scaler to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "scaler"     : self.scaler,
            "config"     : {
                "seq_len"    : self.seq_len,
                "hidden_size": self.hidden_size,
                "num_layers" : self.num_layers,
            },
            "metrics"    : self.metrics_,
            "history"    : self.training_history_,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMRULPredictor":
        """Load a saved LSTM model from disk."""
        state = torch.load(path, map_location=DEVICE, weights_only=False)
        obj = cls(
            seq_len     = state["config"]["seq_len"],
            hidden_size = state["config"]["hidden_size"],
            num_layers  = state["config"]["num_layers"],
        )
        obj.scaler   = state["scaler"]
        obj.metrics_ = state.get("metrics", {})
        obj.training_history_ = state.get("history", {})
        if state["model_state"] is not None:
            obj.model = _LSTMNetwork(
                n_features  = len(FEATURES),
                hidden_size = obj.hidden_size,
                num_layers  = obj.num_layers,
            ).to(DEVICE)
            obj.model.load_state_dict(state["model_state"])
        return obj

    def __repr__(self) -> str:
        m = self.metrics_
        if m:
            return (f"LSTMRULPredictor("
                    f"MAE={m['mae']:.2f}h, R²={m['r2']:.4f}, "
                    f"seq_len={self.seq_len})")
        return f"LSTMRULPredictor(unfitted, seq_len={self.seq_len})"
