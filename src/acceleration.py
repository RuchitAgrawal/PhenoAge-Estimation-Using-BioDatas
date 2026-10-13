import os
import numpy as np
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def calculate_age_acceleration(predicted_age, chrono_age):
    # delta: positive = accelerated aging, negative = decelerated aging
    return predicted_age - chrono_age

def analyze_acceleration_distribution(accel, output_dir="results/tables"):
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "Metric": ["Mean", "Median", "Standard_Deviation", "Minimum", "Maximum", "IQR", "Skewness"],
        "Value": [
            round(float(np.mean(accel)), 4),
            round(float(np.median(accel)), 4),
            round(float(np.std(accel)), 4),
            round(float(np.min(accel)), 4),
            round(float(np.max(accel)), 4),
            round(float(np.percentile(accel, 75) - np.percentile(accel, 25)), 4),
            round(float(skew(accel)), 4)
        ]
    }
    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(output_dir, "biological_age_acceleration_summary.csv"), index=False)
    return df

def plot_acceleration_analysis(chrono_age, predicted_age, accel, model_name, output_dir="results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Distribution histogram
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)
    sns.histplot(accel, kde=True, color="#2b5c8f", bins=40, ax=ax, edgecolor="black", alpha=0.6)
    ax.axvline(0, color="#d62728", linestyle="--", lw=1.8, label="Zero line (BioAge = ChronoAge)")
    ax.axvline(np.mean(accel), color="#2ca02c", linestyle=":", lw=1.8, label=f"Mean: {np.mean(accel):.2f}")
    ax.axvline(np.median(accel), color="#ff7f0e", linestyle="-.", lw=1.8, label=f"Median: {np.median(accel):.2f}")
    ax.set_title(f"Biological Age Acceleration Profile ({model_name})", fontweight="bold")
    ax.set_xlabel("Age Acceleration Delta (Years)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "08_bio_age_acceleration_distribution.png"))
    plt.close()
    
    # 2. Chrono Age vs Predicted BioAge
    rng = np.random.RandomState(42)
    sub = rng.choice(len(chrono_age), size=min(4000, len(chrono_age)), replace=False)
    
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    ax.scatter(chrono_age[sub], predicted_age[sub], alpha=0.25, color="#17becf", edgecolors="none", s=20)
    
    vmin = min(chrono_age.min(), predicted_age.min())
    vmax = max(chrono_age.max(), predicted_age.max())
    ax.plot([vmin, vmax], [vmin, vmax], color="#d62728", linestyle="--", lw=1.8, label="Identity (BioAge = ChronoAge)")
    
    ax.set_title(f"Chronological vs Predicted Biological Age ({model_name})", fontweight="bold")
    ax.set_xlabel("Chronological Age (Years)")
    ax.set_ylabel("Predicted BioAge (Years)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "09_chrono_vs_predicted_bio_age.png"))
    plt.close()
    
    # 3. Chrono Age vs Acceleration delta
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    ax.scatter(chrono_age[sub], accel[sub], alpha=0.25, color="#bcbd22", edgecolors="none", s=20)
    ax.axhline(0, color="#d62728", linestyle="--", lw=1.8)
    ax.set_title(f"Age Acceleration Trend across Lifespan ({model_name})", fontweight="bold")
    ax.set_xlabel("Chronological Age (Years)")
    ax.set_ylabel("Acceleration Delta (Years)")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "10_chrono_vs_age_acceleration.png"))
    plt.close()
