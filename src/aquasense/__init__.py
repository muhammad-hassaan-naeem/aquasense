"""
AquaSense — Underwater Sensor Node Monitoring System
=====================================================
Predicts battery Remaining Useful Life (RUL) and detects anomalous
behaviour in underwater IoT sensor networks using machine learning.

Modules
-------
simulate   – Synthetic data generation (NumPy / Pandas)
database   – SQLite / PostgreSQL persistence layer
models     – Random Forest RUL regressor + Isolation Forest detector
visualise  – 8-panel Matplotlib dashboard
pipeline   – End-to-end orchestrator (CLI entry-point)
"""

__version__ = "1.0.0"
__author__  = "Muhammad Hassaan Naeem"
__license__ = "MIT"
