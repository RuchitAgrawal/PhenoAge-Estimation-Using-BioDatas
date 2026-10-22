import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_TABLES_DIR = PACKAGE_DIR / "results" / "tables"
DEFAULT_FIGURES_DIR = PACKAGE_DIR / "results" / "figures"

def run_survival_analysis(test_df, y_pred, output_dir=None, tables_dir=None):
    fig_dir = pathlib.Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    tab_dir = pathlib.Path(tables_dir) if tables_dir else DEFAULT_TABLES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    
    surv_data = test_df.copy()
    surv_data["pred_bioage"] = y_pred
    surv_data["bio_age_acceleration"] = y_pred - surv_data["age"]
    
    # drop rows missing vital event or follow-up time
    cohort = surv_data.dropna(subset=["status", "time"]).copy()
    cohort["status"] = cohort["status"].astype(int)
    cohort["time"] = cohort["time"].astype(float)
    
    n_events = int(cohort["status"].sum())
    print(f"  [survival] evaluating holdout subset: {len(cohort)} individuals ({n_events} deaths)")
    
    # 1. Kaplan-Meier estimator
    # split into accelerated vs decelerated based on zero delta threshold
    is_accelerated = cohort["bio_age_acceleration"] > 0
    
    km_accel = KaplanMeierFitter()
    km_decel = KaplanMeierFitter()
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    
    km_decel.fit(cohort.loc[~is_accelerated, "time"], cohort.loc[~is_accelerated, "status"], 
                 label=f"Decelerated Aging (Delta <= 0, n={(~is_accelerated).sum()})")
    km_decel.plot_survival_function(ax=ax, color="#2ca02c", lw=2)
    
    km_accel.fit(cohort.loc[is_accelerated, "time"], cohort.loc[is_accelerated, "status"], 
                 label=f"Accelerated Aging (Delta > 0, n={is_accelerated.sum()})")
    km_accel.plot_survival_function(ax=ax, color="#d62728", lw=2)
    
    lr = logrank_test(
        cohort.loc[is_accelerated, "time"], cohort.loc[~is_accelerated, "time"],
        cohort.loc[is_accelerated, "status"], cohort.loc[~is_accelerated, "status"]
    )
    p_str = "p < 0.0001" if lr.p_value < 1e-4 else f"p = {lr.p_value:.4f}"
    ax.text(0.04, 0.18, f"Log-Rank Chi2 = {lr.test_statistic:.2f}\n{p_str}",
            transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.9, edgecolor="#ccc"))
    
    ax.set_title("Kaplan-Meier Survival Curves by Age Acceleration", fontweight="bold")
    ax.set_xlabel("Observation Window (Person-Months)")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(fig_dir / "15_kaplan_meier_survival_curves.png")
    plt.close()
    
    # 2. Cox Proportional Hazards regression
    # model 1: unadjusted acceleration
    cph1 = CoxPHFitter()
    cph1.fit(cohort[["time", "status", "bio_age_acceleration"]], duration_col="time", event_col="status")
    
    hr1 = float(np.exp(cph1.params_["bio_age_acceleration"]))
    ci1_low = float(np.exp(cph1.confidence_intervals_.loc["bio_age_acceleration", "95% lower-bound"]))
    ci1_up = float(np.exp(cph1.confidence_intervals_.loc["bio_age_acceleration", "95% upper-bound"]))
    p1 = cph1.summary.loc["bio_age_acceleration", "p"]
    
    # model 2: adjusted for baseline chronological age
    cph2 = CoxPHFitter()
    cph2.fit(cohort[["time", "status", "bio_age_acceleration", "age"]], duration_col="time", event_col="status")
    
    hr2_acc = float(np.exp(cph2.params_["bio_age_acceleration"]))
    ci2_low = float(np.exp(cph2.confidence_intervals_.loc["bio_age_acceleration", "95% lower-bound"]))
    ci2_up = float(np.exp(cph2.confidence_intervals_.loc["bio_age_acceleration", "95% upper-bound"]))
    p2_acc = cph2.summary.loc["bio_age_acceleration", "p"]
    
    hr2_age = float(np.exp(cph2.params_["age"]))
    ci2_age_low = float(np.exp(cph2.confidence_intervals_.loc["age", "95% lower-bound"]))
    ci2_age_up = float(np.exp(cph2.confidence_intervals_.loc["age", "95% upper-bound"]))
    p2_age = cph2.summary.loc["age", "p"]
    
    cox_records = [
        {
            "Model": "Univariate Bio-Age Acceleration",
            "Variable": "bio_age_acceleration (per 1 yr)",
            "Hazard_Ratio": round(hr1, 4),
            "CI_95_Lower": round(ci1_low, 4),
            "CI_95_Upper": round(ci1_up, 4),
            "p_value": "< 0.0001" if p1 < 1e-4 else round(p1, 4)
        },
        {
            "Model": "Multivariable (Adjusted for Chrono Age)",
            "Variable": "bio_age_acceleration (per 1 yr)",
            "Hazard_Ratio": round(hr2_acc, 4),
            "CI_95_Lower": round(ci2_low, 4),
            "CI_95_Upper": round(ci2_up, 4),
            "p_value": "< 0.0001" if p2_acc < 1e-4 else round(p2_acc, 4)
        },
        {
            "Model": "Multivariable (Adjusted for Chrono Age)",
            "Variable": "Chronological Age (per 1 yr)",
            "Hazard_Ratio": round(hr2_age, 4),
            "CI_95_Lower": round(ci2_age_low, 4),
            "CI_95_Upper": round(ci2_age_up, 4),
            "p_value": "< 0.0001" if p2_age < 1e-4 else round(p2_age, 4)
        }
    ]
    
    # baseline BioAge PhenoAge advance reference (guarded if column exists)
    if "phenoage_advance0" in cohort.columns:
        try:
            cph_ref = CoxPHFitter()
            cph_ref.fit(cohort[["time", "status", "phenoage_advance0", "age"]], duration_col="time", event_col="status")
            hr_ref = float(np.exp(cph_ref.params_["phenoage_advance0"]))
            ci_ref_low = float(np.exp(cph_ref.confidence_intervals_.loc["phenoage_advance0", "95% lower-bound"]))
            ci_ref_up = float(np.exp(cph_ref.confidence_intervals_.loc["phenoage_advance0", "95% upper-bound"]))
            p_ref = cph_ref.summary.loc["phenoage_advance0", "p"]
            cox_records.append({
                "Model": "Baseline BioAge phenoage_advance0 (Adjusted)",
                "Variable": "phenoage_advance0 (per 1 yr)",
                "Hazard_Ratio": round(hr_ref, 4),
                "CI_95_Lower": round(ci_ref_low, 4),
                "CI_95_Upper": round(ci_ref_up, 4),
                "p_value": "< 0.0001" if p_ref < 1e-4 else round(p_ref, 4)
            })
        except Exception as e:
            print(f"  [survival] Notice: baseline reference model skipped ({e})")
            
    cox_df = pd.DataFrame(cox_records)
    cox_df.to_csv(tab_dir / "survival_cox_models.csv", index=False)
    
    # 3. Forest plot
    fig, ax = plt.subplots(figsize=(8.5, 4), dpi=300)
    sub_items = cox_df[cox_df["Variable"].str.contains("acceleration|phenoage")].copy()
    y_pos = np.arange(len(sub_items))
    
    ax.errorbar(
        sub_items["Hazard_Ratio"], y_pos,
        xerr=[sub_items["Hazard_Ratio"] - sub_items["CI_95_Lower"], sub_items["CI_95_Upper"] - sub_items["Hazard_Ratio"]],
        fmt='o', color="#1f77b4", ecolor="#d62728", lw=1.8, capsize=3.5, markersize=7
    )
    ax.axvline(1.0, color="gray", linestyle="--", lw=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['Model']}\n({r['Variable']})" for _, r in sub_items.iterrows()], fontsize=9)
    ax.set_xlabel("Mortality Hazard Ratio (HR with 95% CI)")
    ax.set_title("Cox Proportional Hazards: Biological Acceleration Effect", fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / "16_cox_hazard_ratios_forest_plot.png")
    plt.close()
    
    return cox_df, lr
    
    return cox_df, lr
