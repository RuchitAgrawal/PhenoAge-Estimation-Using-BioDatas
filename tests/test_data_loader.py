"""
Unit tests verifying data splitting integrity and absence of train/test leakage.
"""

from src.data_loader import prepare_splits
from src.config import FEATURES_A, FEATURES_B

def test_split_shapes(synthetic_cohort, tmp_path):
    splits = prepare_splits(synthetic_cohort, test_size=0.2, seed=42, out_dir=tmp_path)
    n = len(synthetic_cohort)
    
    assert splits["X_train_A"].shape[1] == len(FEATURES_A)
    assert splits["X_test_A"].shape[1] == len(FEATURES_A)
    assert splits["X_train_B"].shape[1] == len(FEATURES_B)
    assert splits["X_test_B"].shape[1] == len(FEATURES_B)
    assert len(splits["y_train"]) + len(splits["y_test"]) == n

def test_no_data_leakage(synthetic_cohort, tmp_path):
    splits = prepare_splits(synthetic_cohort, test_size=0.2, seed=42, out_dir=tmp_path)
    train_idx = set(splits["train_df"].index)
    test_idx = set(splits["test_df"].index)
    
    assert len(train_idx.intersection(test_idx)) == 0, "Train and test partition indices overlap!"

def test_scaler_fitting_isolation(synthetic_cohort, tmp_path):
    splits = prepare_splits(synthetic_cohort, test_size=0.2, seed=42, out_dir=tmp_path)
    scaler_mean = splits["scaler_A"].mean_[0]
    train_mean = splits["train_df"]["age"].mean()
    
    # Scaler mean must match the training split mean exactly
    assert abs(scaler_mean - train_mean) < 1e-6
