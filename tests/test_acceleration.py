"""
Unit tests validating biological age acceleration calculation and summary metrics.
"""

import numpy as np
from src.acceleration import calculate_age_acceleration, analyze_acceleration_distribution

def test_acceleration_directional_sign():
    pred = np.array([55.0, 40.0, 70.0])
    chrono = np.array([50.0, 45.0, 65.0])
    accel = calculate_age_acceleration(pred, chrono)
    
    # 55 - 50 = +5 (accelerated)
    assert accel[0] > 0
    # 40 - 45 = -5 (decelerated)
    assert accel[1] < 0
    # 70 - 65 = +5 (accelerated)
    assert accel[2] > 0

def test_acceleration_distribution_structure(tmp_path):
    rng = np.random.RandomState(42)
    accel = rng.normal(0, 3.5, size=200)
    summary_df = analyze_acceleration_distribution(accel, output_dir=str(tmp_path))
    
    metrics = summary_df["Metric"].tolist()
    assert "Mean" in metrics
    assert "Standard_Deviation" in metrics
    assert "IQR" in metrics
