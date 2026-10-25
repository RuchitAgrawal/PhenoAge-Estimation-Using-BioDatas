"""
Pytest test fixtures providing reproducible synthetic cohort partitions.
"""

import pytest
import numpy as np
import pandas as pd
from src.config import FEATURES_A, FEATURES_B, TARGET, SEED

@pytest.fixture
def synthetic_cohort():
    """Generates synthetic physiological observations mimicking NHANES distributions."""
    rng = np.random.RandomState(SEED)
    n = 250
    data = {
        "age": rng.uniform(20.0, 85.0, size=n),
        "albumin_gL": rng.normal(42.0, 4.0, size=n),
        "alp": rng.exponential(65.0, size=n),
        "lncrp": rng.normal(0.5, 0.8, size=n),
        "totchol": rng.normal(190.0, 35.0, size=n),
        "lncreat_umol": rng.normal(4.3, 0.25, size=n),
        "hba1c": rng.normal(5.6, 0.9, size=n),
        "sbp": rng.normal(125.0, 18.0, size=n),
        "bun": rng.normal(14.0, 5.0, size=n),
        "uap": rng.normal(5.3, 1.3, size=n),
        "lymph": rng.normal(30.0, 7.5, size=n),
        "mcv": rng.normal(89.0, 5.5, size=n),
        "wbc": rng.normal(7.1, 2.0, size=n),
    }
    # Synthetic target correlated with age and metabolic markers
    data[TARGET] = (
        0.85 * data["age"] 
        + 1.8 * data["hba1c"] 
        + 2.2 * data["lncreat_umol"] 
        - 0.4 * data["albumin_gL"] 
        + rng.normal(0, 3.0, size=n)
    )
    return pd.DataFrame(data)
