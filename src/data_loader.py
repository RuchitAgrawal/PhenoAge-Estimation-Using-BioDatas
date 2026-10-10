import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

# 13 features for Experiment A (includes chronological age)
FEATURES_A = [
    "age", "albumin_gL", "alp", "lncrp", "totchol", 
    "lncreat_umol", "hba1c", "sbp", "bun", "uap", 
    "lymph", "mcv", "wbc"
]

# 12 biomarkers for Experiment B (age removed to test pure biological signal)
FEATURES_B = [f for f in FEATURES_A if f != "age"]

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGET = "phenoage0"

def load_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(PACKAGE_DIR, "data", "processed", "nhanes4_core_35723.csv")
    elif not os.path.isabs(filepath) and not os.path.exists(filepath):
        cand = os.path.join(PACKAGE_DIR, filepath)
        if os.path.exists(cand):
            filepath = cand
            
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing processed data file: {filepath}")
    return pd.read_csv(filepath)

def prepare_splits(df, test_size=0.2, seed=SEED, out_dir=None):
    # standard 80/20 train-test split
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    
    if out_dir is None:
        out_dir = os.path.join(PACKAGE_DIR, "results", "models")
    os.makedirs(out_dir, exist_ok=True)
    
    # save row indices so we can always map predictions back to participants
    np.save(os.path.join(out_dir, "train_indices.npy"), train.index.values)
    np.save(os.path.join(out_dir, "test_indices.npy"), test.index.values)
    
    y_train = train[TARGET].values
    y_test = test[TARGET].values
    
    # --- Setting A: Chrono Age + Biomarkers ---
    scaler_a = StandardScaler()
    x_train_a = scaler_a.fit_transform(train[FEATURES_A].values)
    x_test_a = scaler_a.transform(test[FEATURES_A].values)
    joblib.dump(scaler_a, os.path.join(out_dir, "scaler_A.joblib"))
    
    # --- Setting B: Biomarkers only ---
    scaler_b = StandardScaler()
    x_train_b = scaler_b.fit_transform(train[FEATURES_B].values)
    x_test_b = scaler_b.transform(test[FEATURES_B].values)
    joblib.dump(scaler_b, os.path.join(out_dir, "scaler_B.joblib"))
    
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
