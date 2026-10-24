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
    fig.savefig(fig_dir / f"{prefix}_prediction_error_distribution.png")
    plt.close()

def plot_bland_altman(y_true, y_pred, model_name, output_dir=None, prefix="06"):
    """Generate clinical method-agreement Bland-Altman diagnostic plot."""
    fig_dir = pathlib.Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    mean_vals = (y_true + y_pred) / 2.0
    diff_vals = y_true - y_pred
    mean_diff = float(np.mean(diff_vals))
    std_diff = float(np.std(diff_vals))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    
    # Subsample for clear visualization
    rng = np.random.RandomState(42)
    sub = rng.choice(len(y_true), size=min(4000, len(y_true)), replace=False)
    
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    ax.scatter(mean_vals[sub], diff_vals[sub], alpha=0.22, color="#1f77b4", s=18, edgecolors="none")
    ax.axhline(mean_diff, color="#d62728", lw=1.8, label=f"Mean Bias: {mean_diff:+.2f} yrs")
    ax.axhline(loa_upper, color="#ff7f0e", linestyle="--", lw=1.4, label=f"+1.96 SD: {loa_upper:+.2f} yrs")
    ax.axhline(loa_lower, color="#ff7f0e", linestyle="--", lw=1.4, label=f"-1.96 SD: {loa_lower:+.2f} yrs")
    ax.axhline(0, color="gray", linestyle=":", lw=1.0)
    
    ax.set_title(f"Bland-Altman Agreement ({model_name})", fontweight="bold")
    ax.set_xlabel("Mean of True and Predicted PhenoAge (Years)")
    ax.set_ylabel("Difference (True - Predicted) [Years]")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_bland_altman_agreement.png")
    plt.close()

def bootstrap_confidence_intervals(model, x_test, y_test, n_iterations=1000, ci=0.95, seed=42):
    """Compute empirical bootstrap 95% confidence intervals on holdout test partition."""
    rng = np.random.RandomState(seed)
    mae_list, rmse_list, r2_list = [], [], []
    n_samples = len(x_test)
    
    for _ in range(n_iterations):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        x_b, y_b = x_test[idx], y_test[idx]
        y_hat = model.predict(x_b)
        mae_list.append(mean_absolute_error(y_b, y_hat))
        rmse_list.append(root_mean_squared_error(y_b, y_hat))
        r2_list.append(r2_score(y_b, y_hat))
        
    alpha = (1.0 - ci) / 2.0
    return {
        "MAE": {
            "mean": round(float(np.mean(mae_list)), 4),
            "ci_lower": round(float(np.percentile(mae_list, alpha * 100)), 4),
            "ci_upper": round(float(np.percentile(mae_list, (1 - alpha) * 100)), 4)
        },
        "RMSE": {
            "mean": round(float(np.mean(rmse_list)), 4),
            "ci_lower": round(float(np.percentile(rmse_list, alpha * 100)), 4),
            "ci_upper": round(float(np.percentile(rmse_list, (1 - alpha) * 100)), 4)
        },
        "R2": {
            "mean": round(float(np.mean(r2_list)), 4),
            "ci_lower": round(float(np.percentile(r2_list, alpha * 100)), 4),
            "ci_upper": round(float(np.percentile(r2_list, (1 - alpha) * 100)), 4)
        }
    }

def plot_experiment_comparison(res_a, res_b, output_dir=None):
    fig_dir = pathlib.Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
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
