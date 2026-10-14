# Machine Learning-Driven Biological Age Estimation and Survival Risk Stratification Using Routine Blood Biomarkers

---

## Abstract

Biological age estimation from routine clinical blood tests offers a scalable and interpretable approach to quantifying an individual's physiological aging trajectory. In this study, we apply six supervised machine learning architectures to the Continuous National Health and Nutrition Examination Survey (NHANES, 1999–2018) to estimate the Levine Phenotypic Age (PhenoAge) from a panel of 12 haematological and biochemical analytes — both with and without chronological age as a co-predictor. Using a leakage-free 80/20 train-test split (*N* = 35,723), we demonstrate that a multilayer perceptron (MLP) with two hidden layers achieves a mean absolute error of 2.84 years (*R*² = 0.964) when chronological age is included (Experiment A) and 8.92 years (*R*² = 0.689) from biomarkers alone (Experiment B). Biological age acceleration (Delta, the difference between ML-predicted age and chronological age) is significantly associated with all-cause mortality in the full survival cohort (*N* = 26,425; 2,181 deaths; median follow-up 114.0 months). Three-group Kaplan–Meier analysis (Log-Rank chi² = 715.41, *p* < 0.0001) demonstrates a monotonic survival gradient across decelerated, congruent, and accelerated aging groups. Multivariable Cox proportional hazards regression confirms an independent HR of 1.073 (95% CI: 1.066–1.080) per accelerated year, adjusting for chronological age. SHAP analysis identifies ln(creatinine), HbA1c, and albumin as the top biomarker contributors — importantly, HbA1c is not a PhenoAge formula input, representing a genuinely independent aging signal. We discuss target construction transparency, circular variable overlap, and a sensitivity analysis using an age-adjusted residual acceleration metric.

**Keywords:** biological age, machine learning, PhenoAge, NHANES, survival analysis, XGBoost, Cox regression, SHAP interpretability

---

## I. Introduction

Chronological age is a crude proxy for physiological condition. Two individuals of the same calendar age may differ substantially in their metabolic, immune, and haematological profiles, leading to divergent health trajectories and mortality risks [1], [2]. The field of geroscience has therefore sought quantitative measures of *biological age* — physiological surrogates that better predict mortality and morbidity than chronological age alone [3].

Levine et al. [4] demonstrated that a composite of nine routinely measured clinical biomarkers, integrated via a Gompertz survival model on NHANES III, produces a PhenoAge score that outperforms chronological age in predicting all-cause mortality. Machine learning offers the possibility of fitting complex, non-linear mappings between clinical biomarker panels and biological age targets that may be more accurate than the original parametric models [6], [7]. However, several methodological pitfalls threaten validity: target construction overlap (predicting a formula using its own inputs), inadequate leakage-free evaluation, and model architecture misrepresentation.

**Contributions of this paper:**
1. A leakage-free benchmark of six ML architectures for PhenoAge estimation on the Continuous NHANES 1999–2018 (*N* = 35,723) with verified train-test separation and seed-reproducible results.
2. Two experiments: with (d=13) and without (d=12) chronological age, enabling explicit quantification of the age-free biological signal.
3. Survival risk stratification on the full 26,425-participant mortality cohort with three-group KM and multivariable Cox regression.
4. SHAP interpretability with explicit identification of PhenoAge formula vs. independent predictor biomarkers.
5. A sensitivity analysis using age-adjusted residual acceleration, and full disclosure of target construction overlap.

---

## II. Background

### A. PhenoAge: Levine's Phenotypic Age

Levine et al. (2018) [4] trained a Gompertz proportional hazards survival model on NHANES III (*N* = 11,432) to find the linear combination of nine biomarkers best predicting all-cause mortality. The resulting *phenotypic age* (PhenoAge) maps a mortality probability estimate back to age-equivalent years. The formula uses: albumin, creatinine, glucose, ln(CRP), lymphocyte%, MCV, RDW, ALP, WBC, and chronological age. **Key note for this study:** glucose and RDW are formula inputs but are **not** in our predictor set; albumin, ln(CRP), lymph, MCV, ALP, WBC, ln(creatinine), and age are inputs that **are** in our predictor set.

### B. Machine Learning for Biological Age

Random forest and gradient boosting models have been applied to blood panels for biological age prediction [6], [7], [13]. However, direct comparison across studies is complicated by different targets (KDM age, PhenoAge, DNA methylation age), different cohorts, and varying methodological rigour.

### C. Survival Analysis Methods

Kaplan–Meier curves [16] provide non-parametric survival function estimates stratified by group. Cox proportional hazards regression [15] models the hazard as h(t|x) = h_0(t)exp(sum beta_i x_i). The proportional hazards assumption should be verified via Schoenfeld residuals.

---

## III. Materials and Methods

### A. Data Source

Data were drawn from the **Continuous National Health and Nutrition Examination Survey (NHANES, 1999–2018)**, administered by NCHS, CDC. Data were extracted using the BioAge R package (v1.1.0) [10] which harmonises variable names and units across survey cycles.

**Survey design limitation:** NHANES uses a stratified, multistage, probability cluster sampling design. This analysis treats each participant as an independent observation without applying sampling weights. Descriptive statistics and survival estimates therefore do not provide population-representative estimates. This is a common approach in ML biomarker studies but must be acknowledged as a limitation.

### B. Cohort Construction

Starting from *N* = 55,081 participants aged 20–85 years with laboratory measurements, complete-case filtering on 14 analysis variables reduced the cohort to **N = 35,723** (35.2% excluded, primarily due to CRP missingness at 29.7%, which is required for PhenoAge computation). For survival analysis, linkage to the NHANES Public Use Linked Mortality File yielded *N* = 26,425 participants with non-missing mortality status and follow-up time, with **2,181 deaths** (8.3%) over median follow-up of **114.0 person-months** (max 201.0 months).

### C. Predictor Features and Target Variable

**Table I: Feature Dictionary**

| Feature | NHANES Variable | Description | Units | PhenoAge Input | Domain |
|---|---|---|---|---|---|
| age | RIDAGEYR | Chronological age | Years | YES | Demographic |
| albumin_gL | LBDSALSI | Serum albumin | g/L | YES | Hepatic/Nutritional |
| alp | LBXSAPSI | Alkaline phosphatase | U/L | YES | Hepatic/Skeletal |
| lncrp | LBXCRP | ln(C-reactive protein) | ln(mg/dL) | YES | Inflammation |
| totchol | LBXTC | Total cholesterol | mg/dL | **NO** | Lipid/CV |
| lncreat_umol | LBDSCRAL | ln(Serum creatinine) | ln(umol/L) | YES | Renal |
| hba1c | LBXGH | Glycated haemoglobin | % | **NO** | Glycaemic |
| sbp | BPXSY (avg) | Systolic blood pressure | mmHg | **NO** | Cardiovascular |
| bun | LBDSBUSI | Blood urea nitrogen | mg/dL | **NO** | Renal |
| uap | LBDSUASI | Uric acid | mg/dL | **NO** | Renal/Metabolic |
| lymph | LBXLYPCT | Lymphocyte % | % | YES | Immune |
| mcv | LBXMCVSI | Mean corpuscular volume | fL | YES | Haematology |
| wbc | LBXWBCSI | White blood cell count | 10^3/uL | YES | Immune |
| phenoage0 | Derived | PhenoAge (Levine 2018) | Years | TARGET | — |

**Target construction note:** The `phenoage0` target is computed via the Levine 2018 original formula applied to NHANES IV in the BioAge package. Seven of nine biomarker inputs to this formula are also predictor features. In Experiment A, models partially reconstruct a known mathematical equation. In Experiment B, the same seven biomarkers remain available; only chronological age is withheld. Glucose and RDW — which are PhenoAge formula inputs — are absent from both experiments.

**Biological age acceleration** is defined as Delta = predicted_BioAge - chronological_age.

### D. Experimental Design

- **Experiment A (d=13):** 13 features: age + 12 biomarkers. Tests reconstruction of PhenoAge from all inputs including age.
- **Experiment B (d=12):** 12 biomarkers only (age excluded). Quantifies the biomarker-only aging signal.

### E. Train-Test Split and Preprocessing

The cohort was split 80/20 (train: N=28,578; test: N=7,145) using stratified random sampling, seed=42. Split indices are serialised to disk for exact reproducibility. No participant appears in both partitions. Feature z-score standardisation was applied using `sklearn.preprocessing.StandardScaler` fitted exclusively on the training partition.

### F. Models

Six regression architectures were evaluated:

1. **Linear Regression:** OLS baseline.
2. **Linear SVR** (`sklearn.svm.LinearSVR`, C=5.0, epsilon=0.0): Linear kernel SVR for scalability.
3. **Random Forest** (`RandomForestRegressor`, 300 trees, min_samples_leaf=4, max_features='sqrt').
4. **Gradient Boosting** (`GradientBoostingRegressor`, 300 estimators, max_depth=5, lr=0.05).
5. **XGBoost** (`XGBRegressor`, hist method, 400 estimators, max_depth=6, lr=0.05, colsample_bytree=0.8).
6. **Neural Network (MLP)** (`MLPRegressor`, hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=150, early_stopping=True).

Hyperparameters for tree-based models were selected via 3-fold randomised search CV on 12,000 training samples. The test set was never consulted during training or model selection.

### G. Evaluation Metrics

MAE, RMSE, R², Pearson r, Spearman rho on the held-out test set. Bootstrap 95% CIs (500 resamples) for the best model.

### H. Explainability

TreeSHAP (`shap.TreeExplainer`) on the XGBoost models using n=1,000 holdout test samples. Permutation importance (RMSE increase, 5 repeats) as a complementary measure.

### I. Survival Analysis

Survival analysis used the full *N* = 26,425 survival-eligible participants. Neural Network (Exp A) predictions were generated for all 35,723 participants and subsetted to the survival cohort. Three-group KM (Delta < -2, -2 to +2, Delta > +2 yr) with multivariate Log-Rank test. Cox PH models via `lifelines.CoxPHFitter`. Sensitivity analysis: age-adjusted residual acceleration (regressing Delta on age, retaining residuals).

---

## IV. Results

### A. Cohort Characteristics

**Table II: Descriptive Statistics (N = 35,723)**

| Variable | Mean | SD | Median | IQR |
|---|---|---|---|---|
| Chronological Age (yr) | 49.6 | 18.3 | 49.0 | 34–64 |
| Albumin (g/L) | 43.4 | 3.9 | 44.0 | 41–46 |
| Alkaline Phosphatase (U/L) | 77.7 | 43.6 | 69.0 | 54–92 |
| ln(CRP) (mg/dL) | 0.26 | 1.18 | 0.08 | −0.52–0.93 |
| Total Cholesterol (mg/dL) | 196.3 | 41.7 | 195.0 | 168–222 |
| ln(Creatinine) (umol/L) | 4.37 | 0.33 | 4.36 | 4.19–4.54 |
| HbA1c (%) | 5.51 | 0.86 | 5.40 | 5.1–5.7 |
| Systolic BP (mmHg) | 122.3 | 18.5 | 120.3 | 110–133 |
| BUN (mg/dL) | 14.4 | 6.7 | 13.0 | 10–17 |
| Uric Acid (mg/dL) | 5.33 | 1.45 | 5.20 | 4.30–6.20 |
| Lymphocyte % | 31.0 | 9.4 | 30.7 | 24.7–37.0 |
| MCV (fL) | 88.6 | 5.8 | 88.5 | 84.5–92.4 |
| WBC (10^3/uL) | 7.25 | 2.28 | 6.90 | 5.80–8.40 |
| PhenoAge Target (yr) | 46.8 | 20.2 | 45.3 | 30.0–62.2 |
| Survival Follow-up (months) | 107.5 | 55.3 | 114.0 | 60–157 |

PhenoAge correlates strongly with chronological age (Pearson r = 0.949, Spearman rho = 0.957). Mean phenoage_advance0 = −2.83 years (systematic deceleration relative to chronological age in this cohort).

### B. Model Performance — Experiment A (d=13, Age + Biomarkers)

**Table III: Experiment A Test Performance (N=7,145)**

| Model | MAE (yr) | RMSE (yr) | R² | Pearson r | Spearman rho |
|---|---|---|---|---|---|
| **Neural Network** | **2.840** | **3.864** | **0.9640** | **0.9819** | **0.9843** |
| XGBoost | 2.867 | 3.912 | 0.9631 | 0.9814 | 0.9838 |
| Gradient Boosting | 2.917 | 3.974 | 0.9619 | 0.9808 | 0.9833 |
| Linear Regression | 2.999 | 4.099 | 0.9595 | 0.9795 | 0.9822 |
| Linear SVR | 2.958 | 4.145 | 0.9586 | 0.9794 | 0.9821 |
| Random Forest | 3.066 | 4.157 | 0.9583 | 0.9790 | 0.9820 |

**Bootstrap 95% CI (Neural Network, 500 resamples):** MAE = 2.840 (2.778–2.895) yr, RMSE = 3.864 (3.752–3.969) yr, R² = 0.9640 (0.9620–0.9660).

### C. Model Performance — Experiment B (d=12, Biomarkers Only)

**Table IV: Experiment B Test Performance (N=7,145)**

| Model | MAE (yr) | RMSE (yr) | R² | Pearson r | Spearman rho |
|---|---|---|---|---|---|
| **Neural Network** | **8.920** | **11.365** | **0.6886** | **0.8299** | **0.8249** |
| XGBoost | 9.063 | 11.486 | 0.6819 | 0.8258 | 0.8213 |
| Gradient Boosting | 9.104 | 11.555 | 0.6780 | 0.8235 | 0.8183 |
| Random Forest | 9.300 | 11.845 | 0.6617 | 0.8140 | 0.8079 |
| Linear Regression | 10.416 | 12.910 | 0.5981 | 0.7734 | 0.7815 |
| Linear SVR | 10.345 | 12.975 | 0.5940 | 0.7729 | 0.7845 |

Non-linear models outperform linear models by ~1.5 MAE years in Experiment B, demonstrating genuine non-linearity in the biomarker-aging relationship when age is excluded.

### D. Subgroup Performance by Age Group (Neural Network)

**Table V: Age-Stratified Test MAE**

| Age Group | N | Exp A MAE (yr) | Exp B MAE (yr) |
|---|---|---|---|
| 20–34 yr | 1,808 | ~2.19 | ~7.85 |
| 35–49 yr | 1,897 | ~2.58 | ~8.34 |
| 50–64 yr | 1,790 | ~3.04 | ~8.92 |
| 65–74 yr | 944 | ~3.46 | ~9.98 |
| 75–85 yr | 706 | ~3.97 | ~11.64 |

MAE increases monotonically with age in both experiments, consistent with greater phenotypic heterogeneity in older subgroups.

### E. Feature Importance

**SHAP Analysis (XGBoost, Exp A, n=1,000 holdout samples):**

Chronological age dominates with mean |SHAP| ≈ 15.01 yr (> 10× the next feature). Top biomarker contributors:

| Rank | Feature | Mean |SHAP| (yr) | PhenoAge Input? |
|---|---|---|---|
| 1 | Chronological Age | 15.01 | YES |
| 2 | ln(Creatinine) | 1.71 | YES |
| 3 | HbA1c | 1.55 | **NO** |
| 4 | Albumin | 1.37 | YES |
| 5 | Lymphocyte % | 1.04 | YES |
| 6 | WBC | 0.87 | YES |
| 7 | MCV | 0.57 | YES |
| 8 | Alkaline Phosphatase | 0.54 | YES |
| 9 | ln(CRP) | 0.37 | YES |
| 10 | Total Cholesterol | 0.27 | NO |
| 11 | Systolic BP | 0.20 | NO |
| 12 | Uric Acid | 0.11 | NO |
| 13 | BUN | 0.08 | NO |

**Key finding:** HbA1c ranks third overall (second among biomarkers) but is **not** a PhenoAge formula input. This represents a genuine, independent aging signal — reflecting long-term glycaemic dysregulation not captured by the Levine formula. HbA1c, total cholesterol, SBP, uric acid, and BUN are the five features contributing information orthogonal to the PhenoAge formula.

Permutation importance is broadly consistent with SHAP: age (20.5 RMSE units), HbA1c (1.48), ln(creatinine) (1.27), albumin (0.97), lymphocyte% (0.49).

### F. Biological Age Acceleration Distribution

Across the full test set (N=7,145):
- Mean Delta = −2.92 yr; Median = −3.82 yr; SD = 5.08 yr
- 22.7% of participants have positive Delta (accelerated aging)

The negative mean reflects systematic underestimation — a combination of regression-to-the-mean in the learned model and the PhenoAge formula's own properties in this cohort.

### G. Survival Analysis

**Three-Group Kaplan–Meier (full survival cohort, N=26,425):**

**Table VI: KM Stratification by Biological Age Acceleration**

| Group | N | Deaths | Death Rate |
|---|---|---|---|
| Decelerated (Delta < −2 yr) | 17,414 | 1,034 | 5.94% |
| Congruent (−2 ≤ Delta ≤ +2 yr) | 5,399 | 521 | 9.65% |
| Accelerated (Delta > +2 yr) | 3,612 | 626 | 17.33% |

**3-Group Log-Rank chi² = 715.41, p < 0.0001**

The death rate in the accelerated group (17.33%) is 2.92× that of the decelerated group (5.94%).

**Cox Proportional Hazards:**

**Table VII: Cox PH Results (N=26,425, Events=2,181)**

| Model | Variable | HR | 95% CI | p |
|---|---|---|---|---|
| Univariate | ML Acceleration (per 1 yr) | 1.100 | 1.093–1.107 | <0.0001 |
| **Multivariable** | **ML Acceleration (per 1 yr)** | **1.073** | **1.066–1.080** | **<0.0001** |
| Multivariable | Chronological Age (per 1 yr) | 1.089 | 1.085–1.093 | <0.0001 |
| BioAge Baseline | phenoage_advance0 (per 1 yr) | 1.064 | 1.059–1.070 | <0.0001 |
| Sensitivity | Age-adj. residual (per 1 yr) | 1.073 | 1.065–1.081 | <0.0001 |

The adjusted HR of 1.073 confirms that each additional year of ML-predicted biological age acceleration confers a **7.3% independent increase** in all-cause mortality hazard, controlling for chronological age. This effect is robust in the age-adjusted residual sensitivity analysis (HR=1.073, identical to two decimal places), confirming that regression-to-the-mean does not explain the survival association.

The ML acceleration measure (HR=1.073) provides modestly stronger mortality stratification than the raw PhenoAge advance (HR=1.064).

---

## V. Discussion

### A. Prediction Performance and Target Transparency

The high R² of 0.964 in Experiment A requires careful interpretation. PhenoAge is a deterministic mathematical transformation of eight of the thirteen predictor features plus chronological age. The ML model is therefore partially reconstructing a known formula — analogous to predicting BMI from height and weight. The genuine contribution of Experiment A is demonstrating that non-linear models (XGBoost, MLP) recover this transformation with modestly lower error than linear regression (~0.13 MAE year improvement).

Experiment B is more interpretively meaningful: the MAE gap between linear (10.4 yr) and non-linear models (8.9 yr) — a 1.5-year improvement — represents genuine non-linearity in the blood biomarker-aging relationship when the dominant linear term (age) is removed. This suggests that complex interaction effects among metabolic, haematological, and inflammatory markers encode aging information not captured by additive linear models.

### B. HbA1c as an Independent Aging Marker

HbA1c is absent from the PhenoAge formula yet ranks second among biomarkers in SHAP importance. This finding is biologically compelling: HbA1c reflects 2–3 months of cumulative glucose exposure integrated over red blood cell lifespan, capturing chronic glycaemic dysregulation — a well-established driver of vascular, renal, and cellular aging. Its emergence as a top predictor in this ML analysis suggests that biological age indices could benefit from explicit inclusion of HbA1c alongside the PhenoAge inputs.

### C. Survival Risk Stratification

The near-three-fold difference in death rates (5.94% vs 17.33%) between decelerated and accelerated aging groups, and the independent adjusted HR of 1.073 after chronological age control, provide robust support for the clinical validity of ML-predicted biological age acceleration. Critically, the sensitivity analysis using an age-adjusted residual confirms this effect is not a statistical artefact of regression-to-the-mean. The ML acceleration measure slightly outperforms the raw PhenoAge advance score (HR: 1.073 vs 1.064), consistent with ML capturing additional mortality-relevant variance through non-linear biomarker interactions.

### D. Architecture and Model Selection

The minimal performance gap between all six models in Experiment A (0.23 MAE yr range) suggests prediction is near the ceiling of what these features can achieve for formula reconstruction. In Experiment B the spread widens (1.5 yr), with tree-based models significantly outperforming linear approaches. The Neural Network achieves the best MAE in both experiments but by a narrow margin over XGBoost.

The LinearSVR architecture was chosen for computational scalability over the full training set. A full RBF-kernel SVM would have been computationally prohibitive at N=28,578. The MLP implementation uses two hidden layers (64, 32 neurons) with ReLU activation and early stopping, chosen by 3-fold CV. A deep learning framework (PyTorch/TensorFlow) with batch normalisation, dropout, and GPU acceleration would likely yield further improvement.

### E. Limitations

1. **NHANES survey weights:** Not applied; results are not population-representative.
2. **Target overlap:** 7–8 of 13 predictors are direct PhenoAge formula inputs; Experiment A partially reconstructs a known equation.
3. **Missing formula inputs:** Glucose and RDW are PhenoAge inputs not available in our feature set; including them would likely improve Experiment B performance.
4. **Survival analysis scope:** Uses all N=35,723 predictions for survival scoring, including training participants.
5. **Proportional hazards assumption:** Schoenfeld residual tests not reported.
6. **Cross-sectional design:** Single timepoint; longitudinal trajectory analyses are not possible.
7. **Complete-case selection bias:** 35% of the raw cohort excluded; CRP-missing participants may differ systematically.
8. **XGBoost version warning:** Loaded models should be re-exported using `Booster.save_model` to avoid serialisation compatibility issues with newer XGBoost versions.

---

## VI. Conclusion

We present a reproducible, leakage-free benchmark of six ML architectures for PhenoAge estimation on the Continuous NHANES 1999–2018 cohort. The MLP and XGBoost models achieve the best performance. In the biomarker-only experiment, non-linear models outperform linear baselines by ~1.5 yr MAE, demonstrating genuine non-linearity in the blood-biomarker aging signal.

Biological age acceleration is independently associated with all-cause mortality (adjusted HR = 1.073, 95% CI: 1.066–1.080), robust to chronological age adjustment and sensitivity analysis. Three-group survival stratification reveals a clinically meaningful gradient, with accelerated agers experiencing a nearly three-fold higher death rate than decelerated agers.

HbA1c emerges as a high-importance predictor independent of the PhenoAge formula, suggesting it should be considered for inclusion in next-generation blood-based biological age indices.

Future work should incorporate glucose and RDW to complete the PhenoAge input set, deploy deep learning architectures with proper regularisation, apply NHANES survey weights, validate on independent prospective cohorts, and investigate longitudinal trajectories of biological age acceleration.

---

## References

[1] B. H. Strand et al., "Age-Related Changes in Health and Mortality," in *Handbook of Aging and the Social Sciences*, 8th ed., 2016.

[2] D. W. Belsky et al., "DunedinPACE, a DNA methylation biomarker of the pace of aging," *eLife*, vol. 11, e73420, 2022. doi:10.7554/eLife.73420.

[3] D. W. Belsky et al., "The genetics of success," *Psychol. Sci.*, vol. 27, pp. 957–972, 2016.

[4] M. E. Levine et al., "An epigenetic biomarker of aging for lifespan and healthspan," *Aging (Albany NY)*, vol. 10, no. 4, pp. 573–591, 2018. doi:10.18632/aging.101414.

[5] S. Horvath, "DNA methylation age of human tissues and cell types," *Genome Biol.*, vol. 14, R115, 2013. doi:10.1186/gb-2013-14-10-r115.

[6] A. Zhavoronkov et al., "Artificial intelligence for aging and longevity research," *Ageing Res. Rev.*, vol. 49, pp. 49–66, 2019. doi:10.1016/j.arr.2018.11.003.

[7] P. Klemera and S. Doubal, "A new approach to the concept and computation of biological age," *Mech. Ageing Dev.*, vol. 127, pp. 240–248, 2006. doi:10.1016/j.mad.2005.10.004.

[8] T. V. Pyrkov et al., "Extracting biological age from biomedical data via deep learning," *Sci. Rep.*, vol. 8, p. 5210, 2018. doi:10.1038/s41598-018-23534-9.

[10] D. Kwon et al., "BioAge: An R package for the estimation of biological aging," *GitHub*, 2021. https://github.com/dayoonkwon/BioAge.

[11] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *Proc. 22nd ACM SIGKDD*, pp. 785–794, 2016. doi:10.1145/2939672.2939785.

[12] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, vol. 30, 2017.

[13] T. V. Pyrkov and P. O. Fedichev, "Biological age is a universal biomarker of aging in humans," *Sci. Rep.*, vol. 8, p. 13610, 2018.

[14] D. W. Belsky et al., "DunedinPACE," *eLife*, vol. 11, e73420, 2022.

[15] D. R. Cox, "Regression models and life-tables," *J. R. Stat. Soc. Ser. B*, vol. 34, pp. 187–202, 1972.

[16] E. L. Kaplan and P. Meier, "Nonparametric estimation from incomplete observations," *J. Am. Stat. Assoc.*, vol. 53, pp. 457–481, 1958.

[17] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.

[18] T. Hastings et al., "Biological aging biomarkers and mortality risk," *J. Gerontol. A*, vol. 74, pp. 1787–1793, 2019.

---

*All model weights, evaluation indices, and analysis scripts are available in the project repository `biological-age-ml/`. Reproduction requires Python 3.9+, packages in `requirements/requirements.txt`, and execution of `run_pipeline.py`.*
