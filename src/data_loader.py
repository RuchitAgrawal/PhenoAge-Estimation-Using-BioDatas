import os
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    SEED,
    TEST_SIZE,
    FEATURES_A,
    FEATURES_B,
    TARGET,
    MODELS_DIR,
    NHANES_CORE_CSV
)

logger = logging.getLogger(__name__)

def load_data(filepath=None):
    """Load core processed NHANES dataset."""
    target_path = Path(filepath) if filepath else NHANES_CORE_CSV
    if not target_path.exists():
        raise FileNotFoundError(f"Missing processed data file: {target_path}")
    df = pd.read_csv(target_path)
    logger.info(f"Loaded cohort data with shape {df.shape} from {target_path.name}")
    return df

def prepare_splits(df, test_size=TEST_SIZE, seed=SEED, out_dir=None):
    """Partition dataset into leakage-free train and holdout test splits."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    
    target_out = Path(out_dir) if out_dir else MODELS_DIR
    target_out.mkdir(parents=True, exist_ok=True)
    
    # Save partition row indices for full participant mapping
    np.save(target_out / "train_indices.npy", train.index.values)
    np.save(target_out / "test_indices.npy", test.index.values)
    
    y_train = train[TARGET].values
    y_test = test[TARGET].values
    
    # --- Setting A: Chronological Age + 12 Biomarkers ---
    scaler_a = StandardScaler()
    x_train_a = scaler_a.fit_transform(train[FEATURES_A].values)
    x_test_a = scaler_a.transform(test[FEATURES_A].values)
    joblib.dump(scaler_a, target_out / "scaler_A.joblib")
    
    # --- Setting B: 12 Biomarkers only ---
    scaler_b = StandardScaler()
    x_train_b = scaler_b.fit_transform(train[FEATURES_B].values)
    x_test_b = scaler_b.transform(test[FEATURES_B].values)
    joblib.dump(scaler_b, target_out / "scaler_B.joblib")
    
    logger.info(f"Prepared splits: Train N={len(train):,} | Test N={len(test):,}")
    return {
        "train_df": train,
        "test_df": test,
        "X_train_A": x_train_a,
        "X_test_A": x_test_a,
        "X_train_B": x_train_b,
        "X_test_B": x_test_b,
        "y_train": y_train,
        "y_test": y_test,
        "scaler_A": scaler_a,
        "scaler_B": scaler_b,
        "feature_cols_A": FEATURES_A,
        "feature_cols_B": FEATURES_B,
        "target_col": TARGET
    }

# alias for compatibility
get_train_test_splits = prepare_splits
