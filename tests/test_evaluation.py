"""
Unit tests validating regression metric computations and bootstrap intervals.
"""

from sklearn.linear_model import LinearRegression
from src.data_loader import prepare_splits
from src.evaluation import evaluate_models, bootstrap_confidence_intervals

def test_evaluation_metrics_generation(synthetic_cohort, tmp_path):
    splits = prepare_splits(synthetic_cohort, test_size=0.2, seed=42, out_dir=tmp_path)
    model = LinearRegression().fit(splits["X_train_A"], splits["y_train"])
    
    res_df, preds = evaluate_models(
        {"Linear Regression": model}, 
        splits["X_test_A"], 
        splits["y_test"], 
        exp_name="test",
        out_dir=tmp_path
    )
    
    assert "MAE" in res_df.columns
    assert "RMSE" in res_df.columns
    assert "R2" in res_df.columns
    assert res_df["R2"].iloc[0] > 0.0

def test_bootstrap_intervals(synthetic_cohort, tmp_path):
    splits = prepare_splits(synthetic_cohort, test_size=0.2, seed=42, out_dir=tmp_path)
    model = LinearRegression().fit(splits["X_train_A"], splits["y_train"])
    
    ci = bootstrap_confidence_intervals(model, splits["X_test_A"], splits["y_test"], n_iterations=100)
    
    assert "MAE" in ci
    assert ci["MAE"]["ci_lower"] <= ci["MAE"]["mean"] <= ci["MAE"]["ci_upper"]
