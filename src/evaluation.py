import os
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_TABLES_DIR = PACKAGE_DIR / "results" / "tables"
DEFAULT_FIGURES_DIR = PACKAGE_DIR / "results" / "figures"

def evaluate_models(models_dict, x_test, y_test, exp_name="A", out_dir=None):
    rows = []
    preds = {}
    
    for name, clf in models_dict.items():
        y_hat = clf.predict(x_test)
        preds[name] = y_hat
        
        # calculate core regression metrics on holdout test set
        mae = mean_absolute_error(y_test, y_hat)
        rmse = root_mean_squared_error(y_test, y_hat)
        r2 = r2_score(y_test, y_hat)
        pr, _ = pearsonr(y_test, y_hat)
        sr, _ = spearmanr(y_test, y_hat)
        
        rows.append({
            "Model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "Pearson_r": round(pr, 4),
            "Spearman_r": round(sr, 4)
        })
        
    df_res = pd.DataFrame(rows)
    target_out_dir = pathlib.Path(out_dir) if out_dir else DEFAULT_TABLES_DIR
    target_out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(target_out_dir / f"model_comparison_exp_{exp_name}.csv", index=False)
    return df_res, preds

def identify_best_model(res_df):
    # rank primarily by lowest test rmse, break ties with mae
    ranked = res_df.sort_values(by=["RMSE", "MAE"], ascending=True)
    top = ranked.iloc[0]
    best_name = top["Model"]
    
    summary = (
        f"Selected '{best_name}' as optimal model based on lowest test RMSE ({top['RMSE']:.4f} yrs) "
        f"and MAE ({top['MAE']:.4f} yrs), capturing {top['R2']*100:.1f}% variance (Pearson r={top['Pearson_r']:.4f})."
    )
    return best_name, top.to_dict(), summary

def plot_evaluation_diagnostics(y_true, y_pred, model_name, output_dir="results/figures", prefix="06"):
    os.makedirs(output_dir, exist_ok=True)
    resid = y_true - y_pred
    
    # subsample points for scatter plots to prevent overplotting
    rng = np.random.RandomState(42)
    sub = rng.choice(len(y_true), size=min(4000, len(y_true)), replace=False)
    
    # 1. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    ax.scatter(y_true[sub], y_pred[sub], alpha=0.25, color="#1f77b4", s=20, edgecolors="none")
    
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    ax.plot([vmin, vmax], [vmin, vmax], color="#d62728", linestyle="--", lw=1.8, label="Ideal (1:1 line)")
    
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    ax.set_title(f"Actual vs Predicted PhenoAge ({model_name})", fontweight="bold")
    ax.set_xlabel("True BioAge (Years)")
    ax.set_ylabel("Predicted BioAge (Years)")
    ax.text(0.05, 0.88, f"R² = {r2:.4f}\nRMSE = {rmse:.3f} yrs\nMAE = {mae:.3f} yrs",
            transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.85, edgecolor="#ccc"))
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_actual_vs_predicted.png"))
    plt.close()
    
    # 2. Residuals vs Predicted
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    ax.scatter(y_pred[sub], resid[sub], alpha=0.25, color="#2ca02c", s=20, edgecolors="none")
    ax.axhline(0, color="#d62728", linestyle="--", lw=1.5)
    ax.set_title(f"Residual Plot ({model_name})", fontweight="bold")
    ax.set_xlabel("Predicted PhenoAge (Years)")
    ax.set_ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_residuals_plot.png"))
    plt.close()
    
    # 3. Residual histogram
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)
    sns.histplot(resid, kde=True, color="#9467bd", bins=40, ax=ax, edgecolor="black", alpha=0.55)
    ax.axvline(0, color="#d62728", linestyle="--", lw=1.5)
    ax.set_title(f"Prediction Error Distribution ({model_name})", fontweight="bold")
    ax.set_xlabel("Residual (Years)")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_prediction_error_distribution.png"))
    plt.close()

def plot_experiment_comparison(res_a, res_b, output_dir="results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    merged = pd.merge(res_a[["Model", "RMSE", "R2"]], res_b[["Model", "RMSE", "R2"]], 
                      on="Model", suffixes=("_SettingA", "_SettingB"))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    x = np.arange(len(merged))
    w = 0.35
    
    # RMSE plot
    ax1.bar(x - w/2, merged["RMSE_SettingA"], w, label="Setting A (with Age)", color="#1f77b4", edgecolor="black")
    ax1.bar(x + w/2, merged["RMSE_SettingB"], w, label="Setting B (Biomarkers only)", color="#ff7f0e", edgecolor="black")
    ax1.set_ylabel("Holdout RMSE (Years)")
    ax1.set_title("Test RMSE by Setting", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(merged["Model"], rotation=30, ha="right", fontsize=9)
    ax1.legend()
    
    # R2 plot
    ax2.bar(x - w/2, merged["R2_SettingA"], w, label="Setting A (with Age)", color="#1f77b4", edgecolor="black")
    ax2.bar(x + w/2, merged["R2_SettingB"], w, label="Setting B (Biomarkers only)", color="#ff7f0e", edgecolor="black")
    ax2.set_ylabel("Explained Variance (R²)")
    ax2.set_title("Test R² by Setting", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(merged["Model"], rotation=30, ha="right", fontsize=9)
    ax2.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "07_experiment_A_vs_B_comparison.png"))
    plt.close()
