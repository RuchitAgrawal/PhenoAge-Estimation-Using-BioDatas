import os
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
from xgboost import XGBRegressor

SEED = 42

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_TABLES_DIR = PACKAGE_DIR / "results" / "tables"
DEFAULT_MODELS_DIR = PACKAGE_DIR / "results" / "models"

def build_model_dict():
    # baseline linear + tree ensembles + svm + simple neural net
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=SEED,
            n_jobs=-1
        ),
        # note: linear svr avoids the O(n^2) scaling bottleneck of rbf kernel on 28k rows
        "SVR": LinearSVR(
            C=1.0,
            max_iter=5000,
            random_state=SEED,
            dual="auto"
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=SEED
        ),
        # histogram-based binning speeds up tree building on large tabular data
        "XGBoost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1
        ),
        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=150,
            early_stopping=True,
            random_state=SEED
        )
    }

# alias for compatibility
get_base_models = build_model_dict

def run_cross_validation(models, x_train, y_train, cv_folds=5):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    metrics = {
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error"
    }
    
    rows = []
    for name, clf in models.items():
        print(f"  [CV] evaluating {name} ({cv_folds} folds)...")
        scores = cross_validate(clf, x_train, y_train, cv=kf, scoring=metrics, n_jobs=-1)
        mae = -scores["test_neg_mae"]
        rmse = -scores["test_neg_rmse"]
        rows.append({
            "Model": name,
            "CV_MAE_Mean": float(np.mean(mae)),
            "CV_MAE_Std": float(np.std(mae)),
            "CV_RMSE_Mean": float(np.mean(rmse)),
            "CV_RMSE_Std": float(np.std(rmse))
        })
    return pd.DataFrame(rows)

def tune_hyperparameters(x_train, y_train):
    print("  [tuning] running randomized search on subsample...")
    # subsample 12k rows to keep search runtime under 1-2 mins while preserving distribution
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(x_train), size=min(12000, len(x_train)), replace=False)
    x_sub, y_sub = x_train[idx], y_train[idx]
    
    best_params = {}
    
    # 1. Random forest
    rf_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, 20],
        "min_samples_split": [2, 5, 10]
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        rf_grid, n_iter=6, cv=3, scoring="neg_root_mean_squared_error",
        random_state=SEED, n_jobs=-1
    )
    rf_search.fit(x_sub, y_sub)
    best_params["Random Forest"] = rf_search.best_params_
    
    # 2. XGBoost
    xgb_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.08, 0.15],
        "subsample": [0.8, 1.0]
    }
    xgb_search = RandomizedSearchCV(
        XGBRegressor(tree_method="hist", random_state=SEED, n_jobs=-1),
        xgb_grid, n_iter=6, cv=3, scoring="neg_root_mean_squared_error",
        random_state=SEED, n_jobs=-1
    )
    xgb_search.fit(x_sub, y_sub)
    best_params["XGBoost"] = xgb_search.best_params_
    
    # 3. SVR penalty
    svr_grid = {"C": [0.1, 0.5, 1.0, 2.0, 5.0]}
    svr_search = RandomizedSearchCV(
        LinearSVR(max_iter=5000, random_state=SEED, dual="auto"),
        svr_grid, n_iter=5, cv=3, scoring="neg_root_mean_squared_error",
        random_state=SEED, n_jobs=-1
    )
    svr_search.fit(x_sub, y_sub)
    best_params["SVR"] = svr_search.best_params_
    
    target_out_dir = pathlib.Path(out_dir) if out_dir else DEFAULT_TABLES_DIR
    target_out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(best_params.items()), columns=["Model", "Tuned_Parameters"]).to_csv(
        target_out_dir / "hyperparameter_tuning_results.csv", index=False
    )
    return best_params

def train_and_save_all(models, x_train, y_train, exp_name="A", base_models_dir=None):
    target_base = pathlib.Path(base_models_dir) if base_models_dir else DEFAULT_MODELS_DIR
    out_dir = target_base / f"exp_{exp_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    trained = {}
    for name, clf in models.items():
        print(f"  [fit] {name} (setting {exp_name})...")
        clf.fit(x_train, y_train)
        trained[name] = clf
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(clf, out_dir / fname)
    return trained
