import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def run_eda(df, missing_df=None, output_dir="results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    out_tables = "results/tables"
    os.makedirs(out_tables, exist_ok=True)
    
    # 1. PhenoAge target histogram
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    sns.histplot(df["phenoage0"], kde=True, color="#1f77b4", bins=40, ax=ax, edgecolor="black", alpha=0.6)
    ax.axvline(df["phenoage0"].mean(), color="#d62728", linestyle="--", lw=1.8, label=f"Mean: {df['phenoage0'].mean():.1f}")
    ax.axvline(df["phenoage0"].median(), color="#2ca02c", linestyle=":", lw=1.8, label=f"Median: {df['phenoage0'].median():.1f}")
    ax.set_title("PhenoAge Distribution (NHANES IV Cohort)", fontweight="bold")
    ax.set_xlabel("PhenoAge (Years)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_target_distribution.png"))
    plt.close()
    
    # 2. Chrono Age vs PhenoAge scatter
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    sample_df = df.sample(min(5000, len(df)), random_state=42)
    ax.scatter(sample_df["age"], sample_df["phenoage0"], alpha=0.2, color="#2b5c8f", edgecolors="none", s=18)
    
    lims = [min(df["age"].min(), df["phenoage0"].min()), max(df["age"].max(), df["phenoage0"].max())]
    ax.plot(lims, lims, color="#d62728", linestyle="--", lw=1.8, label="Identity (1:1 line)")
    
    sns.regplot(data=sample_df, x="age", y="phenoage0", scatter=False, ax=ax, color="#ff7f0e", line_kws={"lw": 1.8, "label": "Linear fit"})
    r = df["age"].corr(df["phenoage0"])
    ax.text(0.05, 0.9, f"Pearson r = {r:.3f}\nN = {len(df):,}", transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#ccc"))
    ax.set_title("Chronological Age vs PhenoAge Target", fontweight="bold")
    ax.set_xlabel("Chronological Age (Years)")
    ax.set_ylabel("Phenotypic Biological Age (Years)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_chronological_vs_biological_age.png"))
    plt.close()
    
    # 3. 12-analyte distribution grid
    biomarkers = [
        "albumin_gL", "alp", "lncrp", "totchol", 
        "lncreat_umol", "hba1c", "sbp", "bun", "uap", 
        "lymph", "mcv", "wbc"
    ]
    labels = [
        "Albumin (g/L)", "ALP (U/L)", "ln(CRP) (mg/dL)", "Total Chol (mg/dL)",
        "ln(Creatinine) (umol/L)", "HbA1c (%)", "SBP (mmHg)", "BUN (mg/dL)", "Uric Acid (mg/dL)",
        "Lymphocyte (%)", "MCV (fL)", "WBC (10^3/uL)"
    ]
    
    fig, axes = plt.subplots(4, 3, figsize=(13, 11), dpi=300)
    for i, (bm, lab) in enumerate(zip(biomarkers, labels)):
        ax_i = axes.flatten()[i]
        sns.histplot(df[bm], kde=True, ax=ax_i, color="#3470a3", bins=30, edgecolor="black", alpha=0.5)
        ax_i.set_title(lab, fontsize=10, fontweight="bold")
        ax_i.set_xlabel("")
        ax_i.set_ylabel("")
    plt.suptitle("Clinical Biomarker Distributions across NHANES Cohort", fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_biomarker_distributions.png"))
    plt.close()
    
    # 4. Correlation heatmap
    cols = ["age"] + biomarkers + ["phenoage0"]
    c_mat = df[cols].corr()
    c_mat.to_csv(os.path.join(out_tables, "correlation_matrix.csv"))
    
    fig, ax = plt.subplots(figsize=(10.5, 8.5), dpi=300)
    sns.heatmap(c_mat, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax, annot_kws={"size": 7.5})
    ax.set_title("Biomarker Correlation Matrix (Pearson)", fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_correlation_heatmap.png"))
    plt.close()
    
    # 5. Missing values breakdown
    if missing_df is not None:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
        s_df = missing_df.sort_values(by="Missing_Percentage", ascending=True)
        bars = ax.barh(s_df["Variable"], s_df["Missing_Percentage"], color="#2b5c8f", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Missingness Rate (%)")
        ax.set_title("Variable Missingness in Raw NHANES IV Cohort (N = 55,081)", fontweight="bold")
        for b in bars:
            val = b.get_width()
            ax.text(val + 0.4, b.get_y() + b.get_height()/2, f"{val:.1f}%", va="center", fontsize=8.5)
        ax.set_xlim(0, max(s_df["Missing_Percentage"]) + 5)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "05_missing_values_analysis.png"))
        plt.close()
