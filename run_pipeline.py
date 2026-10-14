import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# add local src directory
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(curr_dir, "src"))

from data_loader import load_data, prepare_splits
from eda import run_eda
from models import build_model_dict, run_cross_validation, tune_hyperparameters, train_and_save_all
from evaluation import evaluate_models, identify_best_model, plot_evaluation_diagnostics, plot_experiment_comparison
from acceleration import calculate_age_acceleration, analyze_acceleration_distribution, plot_acceleration_analysis
from explainability import run_explainability
from survival_analysis import run_survival_analysis

def main():
    print(">> Biological Age Estimation & Survival Risk Pipeline")
    print(">> Cohort: NHANES IV (Dayoon Kwon BioAge dataset)")
    
    # 1. Load data
    data_file = os.path.join(curr_dir, "data/processed/nhanes4_core_35723.csv")
    miss_file = os.path.join(curr_dir, "data/processed/missingness_before_filter.csv")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing data file: {data_file}. Run R/extract_and_validate_data.R first.")
        
    df = load_data(data_file)
    df_missing = pd.read_csv(miss_file) if os.path.exists(miss_file) else None
    print(f"[+] Loaded {len(df):,} participants (13 biomarkers + phenoage0 target)")
    
    # 2. EDA visual generation
    print("[+] Generating EDA diagnostic figures...")
    run_eda(df, missing_df=df_missing, output_dir=os.path.join(curr_dir, "results/figures"))
    
    # 3. Splits (leakage free, scalers fitted on train only)
    print("[+] Splitting cohort 80/20 (seed=42)...")
    splits = prepare_splits(df, test_size=0.2, seed=42)
    print(f"    Train size: {len(splits['train_df']):,} | Test size: {len(splits['test_df']):,}")
    
    # 4. Check for cached models unless --force-retrain is passed
    force_retrain = "--force-retrain" in sys.argv
    trained_a = {}
    trained_b = {}
    dir_a = os.path.join(curr_dir, "results/models/exp_A")
    dir_b = os.path.join(curr_dir, "results/models/exp_B")
    
    if os.path.exists(dir_a) and len(os.listdir(dir_a)) >= 6 and not force_retrain:
        print("[+] Loading pre-trained model weights from disk...")
        m_files = {
            "Linear Regression": "linear_regression.joblib",
            "Random Forest": "random_forest.joblib",
            "SVR": "svr.joblib",
            "Gradient Boosting": "gradient_boosting.joblib",
            "XGBoost": "xgboost.joblib",
            "Neural Network": "neural_network.joblib"
        }
        for name, fn in m_files.items():
            trained_a[name] = joblib.load(os.path.join(dir_a, fn))
            trained_b[name] = joblib.load(os.path.join(dir_b, fn))
    else:
        print("[+] Running 5-fold cross-validation on train split...")
        base_a = build_model_dict()
        base_b = build_model_dict()
        
        cv_a = run_cross_validation(base_a, splits["X_train_A"], splits["y_train"], cv_folds=5)
        cv_a.to_csv(os.path.join(curr_dir, "results/tables/cv_results_exp_A.csv"), index=False)
        
        cv_b = run_cross_validation(base_b, splits["X_train_B"], splits["y_train"], cv_folds=5)
        cv_b.to_csv(os.path.join(curr_dir, "results/tables/cv_results_exp_B.csv"), index=False)
        
        print("[+] Tuning hyperparameters for RF, XGBoost, and SVR...")
        tuned_params = tune_hyperparameters(splits["X_train_A"], splits["y_train"])
        
        # update models with tuned params
        m_a = build_model_dict()
        m_b = build_model_dict()
        for k, p in tuned_params.items():
            if k in m_a: m_a[k].set_params(**p)
            if k in m_b: m_b[k].set_params(**p)
            
        trained_a = train_and_save_all(m_a, splits["X_train_A"], splits["y_train"], exp_name="A")
        trained_b = train_and_save_all(m_b, splits["X_train_B"], splits["y_train"], exp_name="B")
        
    # 5. Out-of-sample evaluation on unseen test partition
    print("[+] Evaluating on holdout test set...")
    eval_a, preds_a = evaluate_models(trained_a, splits["X_test_A"], splits["y_test"], exp_name="A")
    eval_b, preds_b = evaluate_models(trained_b, splits["X_test_B"], splits["y_test"], exp_name="B")
    
    best_name, best_stats, reason = identify_best_model(eval_a)
    print(f"\n[+] Best Model Selected: {best_name}")
    print(f"    {reason}\n")
    
    with open(os.path.join(curr_dir, "results/tables/best_model_info.json"), "w") as f:
        json.dump({"best_model_name": best_name, "metrics": best_stats, "reasoning": reason}, f, indent=4)
        
    best_y_hat = preds_a[best_name]
    plot_evaluation_diagnostics(splits["y_test"], best_y_hat, best_name, output_dir=os.path.join(curr_dir, "results/figures"))
    plot_experiment_comparison(eval_a, eval_b, output_dir=os.path.join(curr_dir, "results/figures"))
    
    # 6. Biological age acceleration
    print("[+] Computing biological age acceleration delta...")
    ca_test = splits["test_df"]["age"].values
    accel = calculate_age_acceleration(best_y_hat, ca_test)
    analyze_acceleration_distribution(accel, output_dir=os.path.join(curr_dir, "results/tables"))
    plot_acceleration_analysis(ca_test, best_y_hat, accel, best_name, output_dir=os.path.join(curr_dir, "results/figures"))
    
    # 7. Explainability
    print("[+] Computing permutation importance & TreeSHAP...")
    tree_clf = trained_a.get("XGBoost", trained_a.get("Random Forest"))
    run_explainability(
        trained_a[best_name], tree_clf, splits["X_train_A"], splits["X_test_A"], splits["y_test"],
        splits["feature_cols_A"], output_dir=os.path.join(curr_dir, "results/figures")
    )
    
    # 8. Longitudinal survival & mortality analysis
    print("[+] Running Kaplan-Meier & Cox Proportional Hazards regression...")
    cox_res, _ = run_survival_analysis(splits["test_df"], best_y_hat, output_dir=os.path.join(curr_dir, "results/figures"))
    
    print("\n[+] Pipeline completed successfully. All artifacts and figures saved.")

if __name__ == "__main__":
    main()
