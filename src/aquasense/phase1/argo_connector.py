"""
phase1/argo_connector.py
------------------------
ARGO Float Real Ocean Data Connector for AquaSense

The Argo programme deploys ~4,000 autonomous profiling floats worldwide.
Each float dives to 2,000m, collects temperature/salinity/pressure profiles,
then surfaces to transmit data via satellite — exactly matching the IoUT
architecture modelled in AquaSense.

Data Source: Ifremer ERDDAP public API (no authentication required)
URL:         https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.json

This module:
  1. Fetches real float profiles from the ARGO API
  2. Adapts ARGO columns to AquaSense feature schema
  3. Estimates battery_voltage and RUL from float age + dive count
  4. Saves adapted data locally as CSV for offline use
  5. Falls back to high-quality synthetic data if API is unreachable

Usage
-----
    from aquasense.phase1.argo_connector import ArgoConnector

    conn = ArgoConnector()
    df   = conn.get_data(n_floats=20, use_cache=True)
    print(df.head())
    print(df.describe())
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

log = logging.getLogger("aquasense.phase1.argo")

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
CACHE_DIR  = _HERE / "data"
CACHE_FILE = CACHE_DIR / "argo_cache.csv"

# ── ARGO API parameters ────────────────────────────────────────────────────
ARGO_BASE_URL = (
    "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.json"
    "?platform_number,cycle_number,direction,time,latitude,longitude,"
    "pres,temp,psal"
    "&pres>=0&pres<=2000"
    "&orderBy(\"platform_number,cycle_number\")"
    "&limit={limit}"
)

# Typical ARGO float battery specs
ARGO_BATTERY_FULL_V  = 4.2   # fresh lithium pack
ARGO_BATTERY_DEAD_V  = 2.5   # cut-off voltage
ARGO_TYPICAL_CYCLES  = 150   # average float lifespan in dive cycles
ARGO_DEPTH_MAX       = 2000  # metres


class ArgoConnector:
    """
    Fetch and adapt ARGO float data to the AquaSense feature schema.

    Parameters
    ----------
    cache_dir : Path
        Directory for caching downloaded data.
    random_seed : int
        Seed for reproducible synthetic fallback data.
    """

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        random_seed: int = 42,
    ) -> None:
        self.cache_dir   = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file  = self.cache_dir / "argo_cache.csv"
        self.rng         = np.random.default_rng(random_seed)

    # ── Public API ─────────────────────────────────────────────────────────

    def get_data(
        self,
        n_floats:  int  = 30,
        use_cache: bool = True,
        force_synthetic: bool = False,
    ) -> pd.DataFrame:
        """
        Return ARGO data adapted to AquaSense schema.

        Tries in order:
          1. Local cache (if use_cache=True and cache exists)
          2. ARGO ERDDAP API (live download)
          3. High-quality synthetic fallback

        Parameters
        ----------
        n_floats      : Number of unique floats to include
        use_cache     : Load from local cache if available
        force_synthetic : Skip API, use synthetic data directly

        Returns
        -------
        pd.DataFrame with AquaSense-compatible columns:
            node_id, timestep, depth_m, pressure_bar, salinity_ppt,
            temperature_c, battery_voltage, tx_freq_ppm,
            packet_success_rt, depth_cluster, is_anomaly, rul_hours,
            data_source  (new column: 'argo_real' or 'synthetic')
        """
        if force_synthetic:
            log.info("Using synthetic fallback data (force_synthetic=True)")
            return self._synthetic_fallback(n_floats)

        # Try cache first
        if use_cache and self.cache_file.exists():
            log.info("Loading ARGO data from cache: %s", self.cache_file)
            df = pd.read_csv(self.cache_file)
            log.info("  Loaded %s rows, %s unique floats",
                     f"{len(df):,}", df['node_id'].nunique())
            return df

        # Try live API
        log.info("Fetching live ARGO data from Ifremer ERDDAP …")
        raw = self._fetch_argo_api(n_rows=n_floats * 50)
        if raw is not None and len(raw) > 0:
            df = self._adapt_argo_to_aquasense(raw)
            # Keep only requested number of floats
            floats = df['node_id'].unique()[:n_floats]
            df     = df[df['node_id'].isin(floats)].copy()
            df.to_csv(self.cache_file, index=False)
            log.info("  Saved %s rows to cache", f"{len(df):,}")
            return df

        # Fallback
        log.warning("ARGO API unreachable — using high-quality synthetic fallback")
        df = self._synthetic_fallback(n_floats)
        df.to_csv(self.cache_file, index=False)
        return df

    def clear_cache(self) -> None:
        """Delete the local ARGO cache to force a fresh download."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            log.info("Cache cleared: %s", self.cache_file)

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-float summary statistics."""
        return (
            df.groupby('node_id')
            .agg(
                n_readings    = ('timestep',        'count'),
                depth_min     = ('depth_m',         'min'),
                depth_max     = ('depth_m',         'max'),
                avg_temp      = ('temperature_c',   'mean'),
                avg_salinity  = ('salinity_ppt',    'mean'),
                avg_battery   = ('battery_voltage', 'mean'),
                final_rul     = ('rul_hours',       'last'),
                data_source   = ('data_source',     'first'),
            )
            .round(3)
            .sort_values('final_rul')
        )

    # ── Private: API fetch ─────────────────────────────────────────────────

    def _fetch_argo_api(self, n_rows: int = 1500) -> Optional[pd.DataFrame]:
        """
        Fetch raw ARGO profiles from the Ifremer ERDDAP API.
        Returns None if the API is unreachable.
        """
        try:
            import urllib.request
            url = ARGO_BASE_URL.format(limit=min(n_rows, 2000))
            log.debug("  GET %s", url[:80])

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AquaSense-Research/2.0 (academic use)"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw_json = json.loads(resp.read().decode("utf-8"))

            cols = [c["name"] for c in raw_json["table"]["columnNames"]
                    if isinstance(c, dict)] if isinstance(
                        raw_json["table"]["columnNames"][0], dict) else \
                   raw_json["table"]["columnNames"]
            rows = raw_json["table"]["rows"]

            df = pd.DataFrame(rows, columns=cols)
            log.info("  ARGO API returned %s rows", f"{len(df):,}")
            return df

        except Exception as exc:
            log.warning("  ARGO API error: %s", exc)
            return None

    # ── Private: Schema adaptation ─────────────────────────────────────────

    def _adapt_argo_to_aquasense(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Map ARGO columns to AquaSense feature schema.

        ARGO columns → AquaSense columns:
            platform_number → node_id
            cycle_number    → timestep
            pres            → pressure_bar  (1 dbar ≈ 0.1 bar)
            pres * 10       → depth_m       (1 dbar ≈ 1 metre)
            temp            → temperature_c
            psal            → salinity_ppt
            (computed)      → battery_voltage
            (computed)      → tx_freq_ppm
            (computed)      → packet_success_rt
            (computed)      → rul_hours
        """
        df = raw.copy()

        # Clean numeric columns
        for col in ['pres', 'temp', 'psal']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['pres', 'temp', 'psal'])
        df = df[df['pres'] > 0].copy()

        # Encode float IDs as integers
        float_ids = {v: i for i, v in
                     enumerate(df['platform_number'].unique())}
        df['node_id'] = df['platform_number'].map(float_ids)

        # Timestep from cycle number
        df['cycle_number'] = pd.to_numeric(
            df['cycle_number'], errors='coerce').fillna(0).astype(int)
        df['timestep'] = df.groupby('node_id')['cycle_number']\
                           .transform(lambda x: x - x.min())

        # Depth and pressure
        df['depth_m']      = (df['pres'] * 1.0).round(2)   # 1 dbar ≈ 1 m
        df['pressure_bar'] = (df['pres'] * 0.1).round(3)   # 1 dbar = 0.1 bar
        df['temperature_c'] = df['temp'].round(3)
        df['salinity_ppt']  = df['psal'].round(3)

        # Depth cluster assignment
        def _cluster(d):
            if d < 60:   return 'shallow'
            if d < 300:  return 'mid'
            return 'deep'
        df['depth_cluster'] = df['depth_m'].apply(_cluster)

        # Battery voltage model:
        # ARGO floats drain battery with each dive cycle
        # Estimate: linear drain from 4.2V to 2.5V over typical lifespan
        df['battery_voltage'] = df.apply(
            lambda r: max(
                ARGO_BATTERY_DEAD_V,
                ARGO_BATTERY_FULL_V
                - (ARGO_BATTERY_FULL_V - ARGO_BATTERY_DEAD_V)
                * (r['timestep'] / ARGO_TYPICAL_CYCLES)
                + self.rng.normal(0, 0.05)
            ), axis=1
        ).round(4)

        # TX frequency: ARGO floats transmit once per surface visit
        # Deeper floats take longer → lower effective tx rate
        df['tx_freq_ppm'] = (
            0.5 / (1 + df['depth_m'] / 500) + self.rng.normal(
                0, 0.05, len(df))
        ).clip(0.05, 2.0).round(3)

        # Packet success rate: degrades with depth and low battery
        df['packet_success_rt'] = (
            0.95
            - 0.0002 * df['depth_m']
            - 0.1    * (4.2 - df['battery_voltage']).clip(0)
            + self.rng.normal(0, 0.03, len(df))
        ).clip(0.0, 1.0).round(4)

        # RUL: hours remaining based on estimated drain rate
        drain_per_step = (ARGO_BATTERY_FULL_V - ARGO_BATTERY_DEAD_V) \
                         / max(ARGO_TYPICAL_CYCLES, 1)
        df['rul_hours'] = (
            (df['battery_voltage'] - ARGO_BATTERY_DEAD_V)
            / max(drain_per_step, 1e-6) * 24  # 1 cycle ≈ 1 day
        ).clip(0).round(2)

        # Anomaly flag: very low battery or very poor PSR
        df['is_anomaly'] = (
            (df['battery_voltage'] < 2.8) |
            (df['packet_success_rt'] < 0.4)
        ).astype(int)

        df['data_source'] = 'argo_real'

        # Select and order final columns
        cols = [
            'node_id', 'timestep', 'depth_m', 'pressure_bar',
            'salinity_ppt', 'temperature_c', 'battery_voltage',
            'tx_freq_ppm', 'packet_success_rt', 'depth_cluster',
            'is_anomaly', 'rul_hours', 'data_source'
        ]
        return df[cols].reset_index(drop=True)

    # ── Private: Synthetic fallback ────────────────────────────────────────

    def _synthetic_fallback(self, n_floats: int = 30) -> pd.DataFrame:
        """
        High-quality synthetic fallback that mimics real ARGO statistics.
        Used when the API is unreachable.

        Based on published ARGO float statistics:
        - Mean profile depth: 1,000m
        - Mean temperature range: 2–28°C
        - Mean salinity range: 33–37 ppt
        - Typical lifespan: 100–200 cycles
        """
        from aquasense.simulate import simulate_sensor_data
        log.info("Generating synthetic ARGO-mimicking data (%d floats)", n_floats)

        df = simulate_sensor_data(
            n_nodes=n_floats,
            n_timesteps=60,        # typical ARGO ~60 profiles before battery low
            random_seed=int(self.rng.integers(0, 9999)),
        )

        # Scale to ARGO realistic ranges
        # ARGO goes much deeper (0–2000m) vs default sim (5–1000m)
        df['depth_m']      = (df['depth_m'] * 1.8).clip(5, 2000).round(2)
        df['pressure_bar'] = (df['depth_m'] * 0.1).round(3)
        # ARGO salinity range: 33–37 ppt (slightly higher than default)
        df['salinity_ppt'] = (df['salinity_ppt'] + self.rng.normal(
            1.5, 0.3, len(df))).clip(30, 40).round(3)
        # ARGO temperature range: 2–28°C (broader range)
        df['temperature_c'] = (
            25 - df['depth_m'] * 0.012 + self.rng.normal(0, 1.0, len(df))
        ).clip(-2, 30).round(3)
        # Battery: ARGO floats drain over ~150 cycles
        df['battery_voltage'] = (
            ARGO_BATTERY_FULL_V
            - (df['timestep'] / 60)
              * (ARGO_BATTERY_FULL_V - ARGO_BATTERY_DEAD_V)
            + self.rng.normal(0, 0.05, len(df))
        ).clip(ARGO_BATTERY_DEAD_V, ARGO_BATTERY_FULL_V).round(4)

        df['data_source'] = 'synthetic_argo'
        return df


# ── Convenience function ───────────────────────────────────────────────────

def load_argo_data(
    n_floats: int = 30,
    use_cache: bool = True,
    force_synthetic: bool = False,
) -> pd.DataFrame:
    """
    One-line convenience wrapper for ArgoConnector.get_data().

    Examples
    --------
    >>> from aquasense.phase1.argo_connector import load_argo_data
    >>> df = load_argo_data(n_floats=20)
    >>> print(f"Loaded {df['node_id'].nunique()} floats, {len(df):,} readings")
    """
    return ArgoConnector().get_data(
        n_floats=n_floats,
        use_cache=use_cache,
        force_synthetic=force_synthetic,
    )
