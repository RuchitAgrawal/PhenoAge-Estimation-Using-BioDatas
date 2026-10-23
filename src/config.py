"""
PhenoAge Estimation & Mortality Risk Pipeline Configuration.
Provides central paths, hyperparameters, feature definitions, and reproducible seeds.
"""

import os
from pathlib import Path

# --- Reproducibility ---
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --- Package Directory Structure ---
PACKAGE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_DIR / "data" / "processed"
RESULTS_DIR = PACKAGE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
AUDIT_FIGURES_DIR = RESULTS_DIR / "figures_audit"
AUDIT_TABLES_DIR = RESULTS_DIR / "tables_audit"

# Ensure essential output directories exist
for d in [MODELS_DIR, FIGURES_DIR, TABLES_DIR, AUDIT_FIGURES_DIR, AUDIT_TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Feature Space Definitions ---
# 13 clinical analytes and demographics for Experiment A (Primary setting)
FEATURES_A = [
    "age", "albumin_gL", "alp", "lncrp", "totchol", 
    "lncreat_umol", "hba1c", "sbp", "bun", "uap", 
    "lymph", "mcv", "wbc"
]

# 12 analytes for Experiment B (Pure biological signal, chronological age excluded)
FEATURES_B = [f for f in FEATURES_A if f != "age"]

# Target variable: PhenoAge biological age in years
TARGET = "phenoage0"

# Feature display names for clinical figures and tables
FEATURE_LABELS = {
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

# --- Core Data Filepaths ---
NHANES_CORE_CSV = DATA_DIR / "nhanes4_core_35723.csv"
NHANES_SURVIVAL_CSV = DATA_DIR / "nhanes4_survival_subset.csv"
MISSINGNESS_CSV = DATA_DIR / "missingness_before_filter.csv"
SUMMARY_STATS_CSV = DATA_DIR / "data_summary_stats.csv"
