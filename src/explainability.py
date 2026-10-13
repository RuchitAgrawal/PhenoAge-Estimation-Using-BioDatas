import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

LABEL_MAP = {
    "age": "Chronological Age",
    "albumin_gL": "Albumin (g/L)",
    "alp": "Alkaline Phosphatase (ALP)",
    "lncrp": "log(C-Reactive Protein)",
    "totchol": "Total Cholesterol",
    "lncreat_umol": "log(Creatinine)",
    "hba1c": "Glycated Hemoglobin (HbA1c)",
    "sbp": "Systolic Blood Pressure",
    "bun": "Blood Urea Nitrogen (BUN)",
    "uap": "Uric Acid",
    "lymph": "Lymphocyte Percent",
    "mcv": "Mean Corpuscular Volume (MCV)",
    "wbc": "White Blood Cell Count (WBC)"
}

def run_explainability(best_model, tree_model, x_train, x_test, y_test, feature_names, output_dir="results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    out_tables = "results/tables"
    os.makedirs(out_tables, exist_ok=True)
    
    clean_labels = [LABEL_MAP.get(f, f) for f in feature_names]
    
    # 1. XGBoost native gain/weight
    if hasattr(tree_model, "feature_importances_"):
        df_tree = pd.DataFrame({
            "Feature": feature_names,
            "DisplayName": clean_labels,
            "Native_Importance": tree_model.feature_importances_
        }).sort_values(by="Native_Importance", ascending=False)
        df_tree.to_csv(os.path.join(out_tables, "native_tree_feature_importance.csv"), index=False)
        
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
        sns.barplot(data=df_tree, y="DisplayName", x="Native_Importance", color="#1f77b4", ax=ax, edgecolor="black")
        ax.set_title("XGBoost Feature Importance (Split Gain)", fontweight="bold")
        ax.set_xlabel("Relative Importance Weight")
        ax.set_ylabel("")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "11_native_feature_importance.png"))
        plt.close()
        
    # 2. Permutation importance on holdout test set (model agnostic, evaluates on best model)
    print("  [explain] computing permutation importance on test set...")
    rng = np.random.RandomState(42)
    sub = rng.choice(len(x_test), size=min(3000, len(x_test)), replace=False)
    
    perm = permutation_importance(
        best_model, x_test[sub], y_test[sub],
        n_repeats=5, random_state=42, n_jobs=-1, scoring="neg_root_mean_squared_error"
    )
    df_perm = pd.DataFrame({
        "Feature": feature_names,
        "DisplayName": clean_labels,
        "Permutation_Importance_Mean": perm.importances_mean,
        "Permutation_Importance_Std": perm.importances_std
    }).sort_values(by="Permutation_Importance_Mean", ascending=False)
    df_perm.to_csv(os.path.join(out_tables, "permutation_importance.csv"), index=False)
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    sns.barplot(data=df_perm, y="DisplayName", x="Permutation_Importance_Mean", color="#2ca02c", ax=ax, edgecolor="black")
    ax.errorbar(
        x=df_perm["Permutation_Importance_Mean"], y=np.arange(len(df_perm)),
        xerr=df_perm["Permutation_Importance_Std"], fmt='none', c='black', capsize=3
    )
    ax.set_title("Permutation Importance on Holdout Set (RMSE Delta)", fontweight="bold")
    ax.set_xlabel("Mean RMSE Increase (Years)")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "12_permutation_importance.png"))
    plt.close()
    
    # 3. TreeSHAP via TreeExplainer on XGBoost
    print("  [explain] extracting TreeSHAP values...")
    try:
        sub_shap = rng.choice(len(x_test), size=min(1000, len(x_test)), replace=False)
        x_shap = x_test[sub_shap]
        
        explainer = shap.TreeExplainer(tree_model)
        shap_vals = explainer.shap_values(x_shap)
        
        # summary beeswarm
        plt.figure(figsize=(9.5, 6.5), dpi=300)
        shap.summary_plot(shap_vals, x_shap, feature_names=clean_labels, show=False)
        plt.title("SHAP Beeswarm Plot: Feature Impact on Biological Age", fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "13_shap_beeswarm_plot.png"), bbox_inches="tight")
        plt.close()
        
        # summary bar
        plt.figure(figsize=(8.5, 5.5), dpi=300)
        shap.summary_plot(shap_vals, x_shap, feature_names=clean_labels, plot_type="bar", show=False)
        plt.title("Mean Absolute SHAP Value per Feature", fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "14_shap_bar_plot.png"), bbox_inches="tight")
        plt.close()
        
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        df_shap = pd.DataFrame({
            "Feature": feature_names,
            "DisplayName": clean_labels,
            "Mean_Abs_SHAP": mean_abs
        }).sort_values(by="Mean_Abs_SHAP", ascending=False)
        df_shap.to_csv(os.path.join(out_tables, "shap_importance_ranking.csv"), index=False)
    except Exception as e:
        print(f"  [warn] shap generation exception: {e}")
