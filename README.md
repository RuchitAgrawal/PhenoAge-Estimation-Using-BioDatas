# Biological Age & Longevity Estimation Using Biomarkers

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![R 4.x](https://img.shields.io/badge/R-4.x-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B.svg)](app.py)

An end-to-end, reproducible Machine Learning academic research project investigating biological age estimation from routine clinical and blood chemistry biomarkers, and evaluating whether biological-age acceleration provides independent prognostic utility for all-cause mortality risk.

Designed for a B.Tech / M.Tech Machine Learning research project and suitable as the empirical basis for a peer-reviewed research paper.

---

## 1. Project Overview & Scientific Context

Chronological age records calendar time since birth, failing to capture heterogeneous individual rates of physiological senescence. Established biological-aging paradigms, specifically the **PhenoAge** algorithm (Levine et al., 2018) formalized in the R package **BioAge** by Dayoon Kwon et al. ([https://github.com/dayoonkwon/BioAge](https://github.com/dayoonkwon/BioAge)), provide objective metrics of physiological decline.

This project treats the BioAge repository as an established scientific baseline and builds an independent, rigorous Machine Learning and Survival Analysis pipeline around it.

### Core Research Questions
1. **Biological Age Estimation (Component A):** Can machine learning algorithms accurately estimate PhenoAge biological age from 12 routine blood and clinical biomarkers, and how much variance is explained by chronological age versus physiological markers?
2. **Longevity & Mortality Risk Stratification (Component B):** Does biological-age acceleration ($\Delta = \text{Predicted BioAge} - \text{Chronological Age}$) independently predict all-cause mortality hazard in longitudinal follow-up?

---

## 2. Repository Structure

```text
biological-age-ml/
├── README.md                           # Main project documentation and reproduction guide
├── app.py                              # Streamlit interactive clinical research web app
├── run_pipeline.py                     # Master end-to-end Python ML reproduction pipeline
├── data/
│   ├── README.md                       # Dataset documentation and provenance
│   └── processed/                      # Derived complete-case datasets (unaltered originals)
│       ├── nhanes4_core_35723.csv      # Complete cohort (35,723 participants x 18 cols)
│       ├── nhanes4_survival_subset.csv # Survival cohort (26,425 participants)
│       ├── missingness_before_filter.csv
│       └── data_summary_stats.csv
├── notebooks/                          # Step-by-step modular R research scripts
│   ├── 01_data_exploration.R           # Script 1: Target, biomarker, and correlation EDA
│   ├── 02_preprocessing.R              # Script 2: Data partitioning and leakage-free scaling
│   ├── 03_baseline_models.R            # Script 3: Baseline linear models (Exp A and B)
│   ├── 04_model_training.R             # Script 4: 5-fold cross-validation on training set
│   ├── 05_model_comparison.R           # Script 5: Holdout test set metrics and best model
│   ├── 06_explainability.R             # Script 6: Feature importances and biomarker analysis
│   └── 07_survival_analysis.R          # Script 7: Kaplan-Meier curves and Cox PH regression
├── R/                                  # Modular R library functions
│   ├── extract_and_validate_data.R     # Extraction from BioAge NHANES4.rda
│   ├── preprocessing.R                 # Leakage-free train/test scaling
│   ├── models.R                        # R model training routines
│   ├── evaluation.R                    # Metrics computation (MAE, RMSE, R2, r, rho)
│   ├── explainability.R                # Coefficient interpretability
│   └── survival_analysis.R             # KM estimation and Cox models
├── src/                                # Modular Python ML package
│   ├── data_loader.py                  # Train/test split (80/20, seed 42) & scaling
│   ├── eda.py                          # Publication-quality EDA figure generation
│   ├── models.py                       # ML regressors, CV, and hyperparameter tuning
│   ├── evaluation.py                   # Multi-metric evaluation and diagnostics
│   ├── acceleration.py                 # Biological age acceleration analysis
│   ├── explainability.py               # Tree importance, permutation, and SHAP
│   └── survival_analysis.py            # Kaplan-Meier, log-rank, Cox PH regression
├── results/
│   ├── figures/                        # 18 Publication-ready high-resolution figures (.png)
│   ├── tables/                         # 12 Empirical evaluation and benchmark tables (.csv, .json)
│   └── models/                         # Serialized models (.joblib) and split indices (.npy)
├── paper/
│   └── research_paper.md               # Complete academic research paper
└── requirements/
    ├── requirements.txt                # Python package dependencies
    └── R_packages.txt                  # R package dependencies
```

---

## 3. Dataset Summary

- **Source:** NHANES IV (1999–2018) via the R `BioAge` package.
- **Original Cohort:** 55,081 participants × 89 variables.
- **Analyzed Cohort:** 35,723 adult participants (complete cases across 14 core variables).
- **Embedded Longitudinal Survival Cohort:** 26,425 participants (2,181 mortality events; median follow-up 116.0 person-months).
- **Input Features ($d=13$):**
  - Demographics: `age`
  - Liver / Protein: `albumin_gL`, `alp`
  - Inflammation: `lncrp` (log-transformed), `wbc`
  - Lipids / Cardiovascular: `totchol`, `sbp`
  - Renal Filtration: `lncreat_umol` (log-transformed), `bun`, `uap`
  - Glycemic Control: `hba1c`
  - Hematology / Immune: `lymph`, `mcv`
- **Target Variable:** `phenoage0` (Phenotypic Biological Age in years).

---

## 4. Key Experimental Findings

### 4.1 Model Benchmark on Holdout Test Set ($N = 7,145$)

#### Experiment A: Biomarkers + Chronological Age ($d=13$)
| Model | MAE (Years) | RMSE (Years) | $R^2$ | Pearson $r$ | Spearman $\rho$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Neural Network (MLP)** | **2.8401** | **3.8643** | **0.9640** | **0.9819** | **0.9843** |
| XGBoost Regressor | 2.8674 | 3.9122 | 0.9631 | 0.9814 | 0.9838 |
| Gradient Boosting | 2.9173 | 3.9740 | 0.9619 | 0.9808 | 0.9833 |
| Linear Regression (OLS) | 2.9995 | 4.0992 | 0.9595 | 0.9795 | 0.9822 |
| Support Vector Regression | 2.9578 | 4.1445 | 0.9586 | 0.9794 | 0.9821 |
| Random Forest Regressor | 3.0659 | 4.1565 | 0.9583 | 0.9790 | 0.9820 |

#### Experiment B: Biomarkers Only ($d=12$, Excluding Chronological Age)
| Model | MAE (Years) | RMSE (Years) | $R^2$ | Pearson $r$ | Spearman $\rho$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Neural Network (MLP)** | **8.9199** | **11.3648** | **0.6886** | **0.8299** | **0.8249** |
| XGBoost Regressor | 9.0631 | 11.4855 | 0.6819 | 0.8258 | 0.8213 |
| Gradient Boosting | 9.1038 | 11.5553 | 0.6780 | 0.8235 | 0.8183 |
| Random Forest Regressor | 9.2997 | 11.8453 | 0.6617 | 0.8140 | 0.8079 |
| Linear Regression (OLS) | 10.4157 | 12.9101 | 0.5981 | 0.7734 | 0.7815 |
| Support Vector Regression | 10.3445 | 12.9754 | 0.5940 | 0.7729 | 0.7845 |

### 4.2 Biomarker Explainability (SHAP & Permutation Hierarchy)
1. **Chronological Age:** Dominant macroeconomic anchor (Mean |SHAP| = 15.10 years).
2. **Glycated Hemoglobin (HbA1c):** Primary metabolic driver (Mean |SHAP| = 1.51 years; Permutation RMSE increase = 1.48 years).
3. **log(Creatinine):** Primary renal/filtration driver (Mean |SHAP| = 1.68 years; Permutation RMSE increase = 1.27 years).
4. **Serum Albumin:** Protective longevity marker (Mean |SHAP| = 1.48 years; inverse association).
5. **Lymphocyte % & WBC:** Hallmarks of immunosenescence and chronic low-grade sterile inflammation ("inflammaging").

### 4.3 Longitudinal Mortality Risk (Cox Proportional Hazards)
- **Kaplan-Meier Survival Divergence:** Log-Rank $\chi^2 = 625.76, p < 0.0001$.
- **Univariate Model:** Each additional year of biological age acceleration increases all-cause mortality hazard by **9.78%** ($\text{HR} = 1.0978$, 95% CI: 1.0818–1.1141, $p < 0.0001$).
- **Multivariable Model (Adjusted for Chronological Age):** Biological age acceleration confers an independent **6.78% increase in mortality hazard per year** ($\text{HR} = 1.0678$, 95% CI: 1.0516–1.0842, $p < 0.0001$), slightly outperforming the baseline BioAge `phenoage_advance0` ($\text{HR} = 1.0623$).

---

## 5. How to Reproduce

### 5.1 Environment Setup

#### Python Environment
```bash
# Using modern Python environment manager (e.g. uv)
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install -r biological-age-ml/requirements/requirements.txt
```

#### R Environment
Ensure R (version $\ge 4.0$) is installed with the required packages:
```r
install.packages(c("caret", "dplyr", "e1071", "ggplot2", "survival", "data.table"))
# BioAge package installed from GitHub:
# remotes::install_github("dayoonkwon/BioAge")
```

### 5.2 End-to-End Pipeline Reproduction

1. **Extract and Validate Data from BioAge:**
   ```bash
   Rscript biological-age-ml/R/extract_and_validate_data.R
   ```
2. **Execute Full Machine Learning Pipeline:**
   ```bash
   python biological-age-ml/run_pipeline.py
   ```
3. **Run R Validation Notebooks (Optional):**
   ```bash
   Rscript biological-age-ml/notebooks/01_data_exploration.R
   Rscript biological-age-ml/notebooks/07_survival_analysis.R
   ```

### 5.3 Launch the Interactive Streamlit Web Application

To launch the web interface:
```bash
streamlit run biological-age-ml/app.py
```
The application will launch locally at `http://localhost:8501`, featuring:
- Clinical biomarker input form with realistic clinical defaults.
- Real-time estimation of Predicted Biological Age and Biological Age Acceleration.
- 95% prediction intervals calibrated against test RMSE.
- Empirical mortality risk stratification derived from our multivariable Cox Proportional Hazards model.
- Dynamic population SHAP biomarker contributions.
- Full academic research disclaimers.

---

## 6. Ethical and Scientific Disclaimers

> **RESEARCH AND EDUCATIONAL USE ONLY:**  
> This project is designed strictly for academic, educational, and computational research purposes. Biological-age estimates and survival associations represent observational statistical predictions derived from population-level epidemiological survey data (NHANES IV). They **MUST NOT** be interpreted as clinical diagnoses, definitive lifespan predictions, or individual medical advice. Consult qualified healthcare providers for medical decision-making.

---

## 7. Citation

If you build upon this project for academic coursework or publications, please cite:

```bibtex
@article{bioage_ml_2026,
  title={Biological Age & Longevity Estimation Using Biomarkers: A Machine Learning Approach},
  author={Antigravity ML Research Group},
  year={2026},
  journal={Academic Research Repository},
  note={Based on the Dayoon Kwon BioAge NHANES benchmark}
}

@article{levine2018epigenetic,
  title={An epigenetic biomarker of aging for lifespan and healthspan},
  author={Levine, Morgan E and Lu, Ake T and Quach, Austin and Chen, Brian H and others},
  journal={Aging (Albany NY)},
  volume={10},
  number={4},
  pages={573--591},
  year={2018}
}

@software{kwon2021bioage,
  title={BioAge: An R package for calculating biological age},
  author={Kwon, Dayoon and Belsky, Daniel W and BioAge Consortium},
  url={https://github.com/dayoonkwon/BioAge},
  year={2021}
}
```
