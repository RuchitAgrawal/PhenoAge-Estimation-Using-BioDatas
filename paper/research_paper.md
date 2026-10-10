# Biological Age & Longevity Estimation Using Biomarkers: A Machine Learning Approach

**Author:** Department of Computer Science & Engineering, Academic Research Group  
**Curriculum Track:** Machine Learning Course Research Project  
**Date:** Academic Term 2026  

---

## Abstract

- **Contextual Premise:** While chronological duration measures elapsed solar years, it systematically masks the divergent trajectories of organ-specific physiological decay among humans. Parametric proxies of multi-system vulnerability, notably the Phenotypic Age formulation pioneered by Levine et al. and implemented within the *BioAge* repository by Kwon et al., synthesize physiological strain into an age-equivalent metric. Nonetheless, conventional parametric survivor regressions cannot readily resolve subtle multi-analyte non-linearities, compensatory biochemical interactions, or complex inter-organ cross-talk.
- **Primary Objective:** This investigation systematically assesses whether diverse supervised learning algorithms can estimate biological age from a panel of routine hematological and clinical analytes, dissects the relative predictive potency of calendar time versus somatic biomarkers, and evaluates whether algorithmic biological age acceleration independently informs longitudinal survivorship.
- **Empirical Cohort:** We leveraged the NHANES IV multi-cycle epidemiological database accessed via the *BioAge* benchmarking library ($N = 55,081$). Following rigorous complete-case verification over 13 candidate clinical inputs and the phenotypic target, we established an analytic cohort of 35,723 adult records, including an embedded longitudinal surveillance subset of 26,425 individuals observed over a median follow-up of 116.0 person-months ($2,181$ recorded all-cause fatalities).
- **Methodological Pipeline:** To guarantee total insulation against data contamination, an 80/20 train-to-test partition was executed under fixed pseudorandom initialization (seed 42), restricting all affine transformations and scaling parameters strictly to the training fold ($N = 28,578$). Two distinct experimental configurations were interrogated: Setting A (incorporating calendar age alongside 12 circulating biomarkers) and Setting B (confining inputs strictly to the 12 somatic analytes). Five-fold cross-validation guided internal calibration across six learning paradigms: Ordinary Least Squares, Random Forest, Support Vector Regression, Gradient Boosted Machines, XGBoost, and a deep Multilayer Perceptron. Interpretability was audited via test-set permutation degradation and tree-based Shapley additive explanations (TreeSHAP). Downstream longevity risk was appraised via stratified Kaplan-Meier estimators and multivariable Cox proportional hazards modeling.
- **Key Empirical Findings:** Within Setting A, the Multilayer Perceptron achieved optimal generalization on the sequestered holdout set ($N = 7,145$), recording a Mean Absolute Error of 2.8401 years, Root Mean Squared Error of 3.8643 years, $R^2$ of 0.9640, Pearson $r = 0.9819$, and Spearman $\rho = 0.9843$, closely rivaled by XGBoost (RMSE: 3.9122 years, $R^2 = 0.9631$). Conversely, withholding chronological age in Setting B triggered a pronounced performance drop (best holdout RMSE: 11.3648 years, $R^2 = 0.6886$), substantiating that calendar age functions as a primary temporal anchor while laboratory analytes govern individual physiological deviation. Explainability metrics highlighted glycated hemoglobin, serum creatinine, albumin, lymphocyte fraction, and leukocyte abundance as the foremost somatic drivers of biological aging. Furthermore, algorithmic biological age acceleration ($\delta = \widehat{\text{BA}} - \text{CA}$) exhibited profound association with prospective mortality (log-rank $\chi^2 = 625.76, p < 0.0001$). Multivariable Cox modeling indicated that each additional year of biological age acceleration confers a 6.78% elevation in all-cause mortality hazard ($\text{HR} = 1.0678$, 95% CI: 1.0516–1.0842, $p < 0.0001$), persisting robustly after adjusting for baseline chronological age ($\text{HR} = 1.0894$, 95% CI: 1.0811–1.0977, $p < 0.0001$).
- **Concluding Synthesis:** Algorithmic estimators successfully reconstruct composite phenotypic biological age from standard clinical chemistry and yield an acceleration metric possessing demonstrable prognostic relevance for population-level mortality risk.

---

## 1. Introduction

### 1.1 Divergence Between Solar Time and Physiological Senescence
Human aging represents an intricate, progressive compromise in homeostatic equilibrium, cellular maintenance, and functional capacity spanning virtually all physiological compartments. Chronological age, documented merely as elapsed calendar days since birth, functions ubiquitously across society as the primary index of individual maturity and disease vulnerability. Unquestionably, advancing calendar time correlates with surging incidences of atherosclerosis, neurodegenerative syndromes, oncogenesis, and metabolic compromise.

Nonetheless, clinical practice constantly confronts the fundamental inadequacy of chronological time: two distinct persons sharing the exact same birth year frequently exhibit profoundly incongruous somatic viability. One subject may retain supple arterial vasculature, sharp cognitive acuity, and high metabolic efficiency, while another presents extensive microvascular rarefaction, systemic low-grade inflammation, and accelerated sarcopenia. Chronological age is blind to this individual-specific heterogeneity.

Consequently, modern biogerontology distinguishes chronological duration from *biological age*—a dynamic construct reflecting an individual's accumulated somatic damage, functional reserve, and remaining homeostatic bandwidth relative to population baselines. When an individual's composite physiological status reflects an older demographic distribution, they display *biological age acceleration*; conversely, individuals with preserved cellular integrity exhibit *biological age deceleration*.

### 1.2 The Role of Blood Analytes and Machine Learning Opportunities
Although high-throughput multi-omics (such as single-cell transcriptomics and mass-spectrometry proteomics) have expanded molecular insights, their substantial expense and specialized laboratory requirements severely impede widespread clinical deployment. In sharp contrast, routine clinical chemistry panels—encompassing kidney clearance markers, hepatic functional proteins, glycemic gauges, hematological counts, and cardiovascular indices—are ubiquitously procured across primary care environments.

Historically, composite biological aging scores derived from routine laboratory panels have relied on linear regressions or Gompertz mortality calibrations, epitomized by the Klemera-Doubal Method (KDM) and the Phenotypic Age construct developed by Morgan Levine and colleagues. While these traditional formulations represented major epidemiological leaps, they presuppose largely linear relationships among circulating analytes and cannot natively untangle subtle threshold nonlinearities, multi-organ compensatory interactions, or complex feedback networks. Modern machine learning paradigms offer a natural inductive approach to address these gaps: they can learn complex, multi-analyte mappings, quantify feature interactions without parametric straightjackets, and isolate which specific physiological domains govern biological acceleration.

### 1.3 Hypotheses and Scope
This academic research project investigates two distinct experimental components:
1. **Component A (Estimation and Calibration):** Evaluating how accurately modern learning architectures—spanning regularized linear baselines, ensemble tree machines, kernel machines, and feedforward neural networks—can estimate Phenotypic Biological Age from routine blood chemistry, while explicitly isolating the exact predictive value supplied by chronological age versus somatic analytes.
2. **Component B (Survivorship Prognostication):** Assessing whether an individual's algorithmically inferred biological age acceleration ($\delta$) conveys statistically genuine, independent prognostic utility regarding prospective all-cause mortality across extended longitudinal surveillance.

---

## 2. Theoretical Grounding & Literature Context

### 2.1 Foundational Biological Aging Paradigms
Over the past three decades, quantitative biogerontologists have advanced diverse conceptual frameworks to translate physiological measurements into scalar indices of organismal aging:

1. **Inverse Functional Regression (The Klemera-Doubal Method):** Formalized by Klemera and Doubal in 2006, KDM frames biological age by minimizing the orthogonal distances between an individual's measured biomarkers and their expected age-dependent regressions across a reference cohort.
2. **Multisystem Variance Metrics (Homeostatic Dysregulation):** Developed by Cohen et al. (2013), this statistical formulation models age-associated decline not as directional shifts along predefined biomarker vectors, but rather as statistical deviation from an idealized multi-dimensional centroid of young, healthy physiological homeostasis quantified via Mahalanobis distance.
3. **Mortality-Calibrated Phenotypic Age (PhenoAge):** Introduced by Levine et al. in 2018, PhenoAge restructured biological aging by calibrating an elastic-net Gompertz proportional hazards survival model on longitudinal mortality records within NHANES III. Ten selected physiological parameters (chronological age plus 9 blood markers) were mathematically mapped into an equivalent chronological risk scale.

Dayoon Kwon and colleagues subsequently unified these distinct algorithmic paradigms into the standardized R software framework *BioAge* (Kwon et al., 2021). Within the present study, the BioAge PhenoAge metric (`phenoage0`) serves strictly as our established scientific ground-truth target. Our work does not mirror or re-implement BioAge; rather, it designs an independent, contamination-free machine learning and survival discovery pipeline operating around this target.

### 2.2 Deep and Ensemble Learning in Geroscience
Parallel computational efforts have explored machine learning across molecular epigenetic clocks (Horvath DNAm, Hannum methylation, and GrimAge) and neuroimaging structural matrices (brain age estimation via volumetric MRI). While epigenetic clocks capture cellular mitotic history, blood-based clinical ML architectures provide direct windows into active organ-level pathophysiology, hemodynamic stress, and metabolic strain at a fraction of the cost.

---

## 3. Cohort Architecture & Data Auditing

### 3.1 The NHANES IV Analytical Corpus
Empirical data were drawn from the continuous National Health and Nutrition Examination Survey (NHANES IV, encompassing survey cohorts from 1999 to 2018), provided through the BioAge distribution. The initial uncurated database contained **55,081 participant records** across **89 recorded attributes**.

### 3.2 Candidate Features and Physiological Domains
Thirteen primary physiological indicators were selected, capturing vital organ systems, systemic inflammation, and metabolic regulation:

| Variable Identifier | Clinical Description | Target Physiological Domain | Operational Measurement Units | Cohort Missingness ($N=55,081$) |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Chronological Age | Macro-Temporal Baseline | Solar Years | 0.00% |
| `albumin_gL` | Serum Albumin | Hepatic Protein Output / Nutrition | Grams per Liter (g/L) | 11.05% |
| `alp` | Alkaline Phosphatase | Hepatobiliary & Skeletal Remodeling | Units per Liter (U/L) | 11.27% |
| `lncrp` | Log C-Reactive Protein | Systemic Sterile Inflammation | Natural Log $\text{mg/dL}$ | 29.69% |
| `totchol` | Total Cholesterol | Circulating Lipid Homeostasis | Milligrams per Deciliter (mg/dL) | 10.86% |
| `lncreat_umol`| Log Serum Creatinine | Glomerular Filtration & Muscle Turnover | Natural Log $\mu\text{mol/L}$ | 11.37% |
| `hba1c` | Glycated Hemoglobin | Long-Term Glycemic Maintenance | Percentage of Total Hemoglobin (%) | 10.45% |
| `sbp` | Systolic Blood Pressure | Arterial Hemodynamic Burden | Millimeters of Mercury (mmHg) | 9.33% |
| `bun` | Blood Urea Nitrogen | Renal Nitrogen Excretion & Hydration | Milligrams per Deciliter (mg/dL) | 11.44% |
| `uap` | Serum Uric Acid | Purine Catabolic Flux | Milligrams per Deciliter (mg/dL) | 11.07% |
| `lymph` | Lymphocyte Percentage | Adaptive Immune Cell Quota | Percentage of Leukocytes (%) | 9.97% |
| `mcv` | Mean Corpuscular Volume | Erythrocyte Morphology & Stress | Femtoliters (fL) | 9.73% |
| `wbc` | Leukocyte Concentration | Innate Immune Priming (*Inflammaging*) | Thousands per Microliter ($10^3/\mu\text{L}$) | 9.74% |
| `phenoage0` | **Phenotypic Biological Age** | **Composite Biological Age Target** | **Equivalent Solar Years** | **31.77%** |
| `status` | Mortality Indicator | Vital Longitudinal Outcome | 1 = Deceased, 0 = Censored Alive | 20.63% |
| `time` | Observation Window | Follow-up Surveillance Span | Elapsed Person-Months | 24.48% |

### 3.3 Cohort Curation and Baseline Distribution
Prior to statistical learning, data auditing revealed that missingness clustered heavily within C-reactive protein (29.69%) and prospective mortality linkage (20.63%), attributable to cycle-specific laboratory rotation protocols rather than non-random attrition. 

Applying strict complete-case filtering across the 14 core attributes (`age`, the 12 clinical analytes, and `phenoage0`) isolated an analytic dataset of **35,723 adult participants**. Within this group, a longitudinal surveillance cohort of **26,425 individuals** possessed verified survival tracking (2,181 documented fatalities; median follow-up 116.0 person-months, span: 0.0 to 201.0 months).

Parametric distribution auditing of the complete-case cohort ($N = 35,723$) yielded the following baseline metrics:
- **Calendar Age:** Mean of 49.61 years (Standard Deviation: 18.27, Median: 49.00, Extrema: [20.0, 85.0])
- **Phenotypic Target (`phenoage0`):** Mean of 46.78 years (Standard Deviation: 20.19, Median: 45.30, Extrema: [4.24, 115.29])
- **Circulating Albumin:** Mean of 42.30 g/L (SD: 3.60, Median: 42.00)
- **Systolic Hemodynamics:** Mean of 124.97 mmHg (SD: 19.57, Median: 122.00)
- **Glycated Hemoglobin Fraction:** Mean of 5.62% (SD: 0.83, Median: 5.40%)
- **Log Creatinine Concentration:** Mean of 4.33 $\mu\text{mol/L}$ (SD: 0.26, Median: 4.34)

---

## 4. Methodological Framework

### 4.1 Anti-Contamination Protocols & Preprocessing Isolation
To eliminate data snooping and subtle leakage:
1. The 35,723 curated observations were partitioned into an **80% training development corpus ($N = 28,578$)** and an unbreached **20% holdout test partition ($N = 7,145$)** using a fixed pseudorandom seed (42).
2. Standardization centering and scaling parameters were computed **strictly and solely from the training partition**:
   $$\mu_j = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i \in \mathcal{D}_{\text{train}}} x_{ij}, \quad \sigma_j = \sqrt{\frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i \in \mathcal{D}_{\text{train}}} (x_{ij} - \mu_j)^2}$$
3. The derived empirical tuples $(\mu_j, \sigma_j)$ were serialized as dedicated artifacts (`scaler_A.joblib` and `scaler_B.joblib`) and applied downstream to standardize test records without retraining.
4. The holdout test corpus remained completely isolated from hyperparameter tuning and model ranking.

### 4.2 Comparative Experimental Settings
To rigorously parse how much predictive fidelity stems from calendar progression versus physiological dysregulation, two distinct experimental configurations were created:
- **Setting A (Holistic Biomarker Matrix + Chronological Time):** Models receive all 13 features ($d = 13$), combining calendar time with circulating analytes.
- **Setting B (Acellular Biomarker Matrix Exclusively):** Chronological age is completely removed ($d = 12$). Models must deduce biological age purely from circulating somatic markers.

### 4.3 Evaluated Predictive Algorithms
Six distinct supervised learning paradigms were benchmarked:
1. **Ordinary Least Squares (OLS Regression):** Establishes an unregularized linear baseline.
2. **Random Forest Regression:** An ensemble of de-correlated randomized decision trees capturing non-linear feature splits.
3. **Support Vector Regression (SVR):** Solves a dual quadratic program enforcing $\epsilon$-insensitive loss to establish robust bounding hyperplanes.
4. **Gradient Boosted Decision Trees (GBM):** Sequentially constructs decision stumps to greedily minimize pseudo-residuals.
5. **XGBoost Architecture:** Employs second-order Taylor approximations of the loss function coupled with histogram-based feature quantization and explicit tree-complexity regularization.
6. **Multilayer Perceptron (Feedforward Neural Network):** A deep non-linear network consisting of two fully connected hidden layers (64 and 32 units) utilizing rectified linear unit (ReLU) activations, adaptive moment estimation (Adam), and early-stopping regularization on an internal validation split.

### 4.4 Model Optimization & Cross-Validation Strategy
Internal tuning relied upon **5-fold cross-validation** executed strictly across the 28,578 training records. Focused hyperparameter optimization for Random Forest, XGBoost, and SVR explored pre-specified parameter grids using 3-fold cross-validation:
- *Random Forest:* Parameter tuning settled upon 200 estimators, maximum tree depth of 20, and a minimum split requirement of 2 samples.
- *XGBoost:* Optimization converged upon 200 trees, maximum depth of 3, learning rate $\eta = 0.15$, and a subsampling ratio of 0.8.
- *SVR:* Regularization parameter converged at $C = 5.0$.

### 4.5 Generalization Metrics
Model performance on the unseen test set was evaluated across five statistical metrics:
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$
- **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$
- **Explained Variance Ratio ($R^2$):** $R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$
- **Pearson Linear Correlation ($r$):** Bivariate linear agreement between predicted and ground-truth values.
- **Spearman Monotonic Rank Correlation ($\rho$):** Rank-order concordance.

### 4.6 Quantification of Biological Age Acceleration
For every participant in the holdout test set, biological age acceleration ($\delta$) was defined as:
$$\delta_i = \widehat{\text{BA}}_i - \text{CA}_i$$
where $\widehat{\text{BA}}_i$ represents the model's inferred biological age and $\text{CA}_i$ denotes true chronological age. Values of $\delta_i > 0$ reflect biologically accelerated senescence, whereas $\delta_i < 0$ denotes biological deceleration.

### 4.7 Survivorship Modeling & Hazard Estimation
To verify whether inferred acceleration ($\delta$) provides genuine longevity insights rather than mathematical noise:
1. **Stratified Kaplan-Meier Estimation:** Holdout survival records were partitioned into accelerated individuals ($\delta > 0$) versus decelerated peers ($\delta \le 0$). Empirical survival curves $S(t)$ were generated and compared using the two-tailed log-rank test.
2. **Cox Proportional Hazards Regression:** Semi-parametric hazards were estimated via:
   $$h(t | \mathbf{z}) = h_0(t) \exp\left( \beta_{\text{accel}} \delta + \beta_{\text{age}} \text{CA} \right)$$
   Univariate models evaluated unadjusted risk, while multivariable models controlled for chronological age to assess whether biological age acceleration provides statistically independent prognostic value. Proportional hazards assumptions were confirmed via Schoenfeld residual analyses.

---

## 5. Experimental Discovery & Analysis

### 5.1 Exploratory Analysis of Somatic Analytes
Initial screening of the 35,723 complete records confirmed several epidemiological patterns:
- Phenotypic age spans a near-Gaussian distribution with mild positive tailing (Mean: 46.78 years, Median: 45.30 years, Skewness: 0.28).
- Chronological age correlates tightly with PhenoAge ($r = 0.9488$), reflecting the broad temporal progression of senescence.
- Univariate correlation screening revealed that systolic hemodynamic pressure ($r = 0.4856$), urea nitrogen ($r = 0.4786$), glycated hemoglobin ($r = 0.4541$), and log-creatinine ($r = 0.3645$) increase alongside biological age, whereas serum albumin ($r = -0.2390$) and lymphocyte percentage ($r = -0.1916$) decline, reflecting diminished hepatic reserve and adaptive immunosenescence.

### 5.2 Training Fold 5-Fold Cross-Validation Performance
Cross-validation on the 28,578 training records demonstrated low fold-to-fold variance and stable convergence across all algorithms:

| Analytical Regime | Evaluated Architecture | Cross-Val Mean MAE (Yrs) | Cross-Val SD MAE | Cross-Val Mean RMSE (Yrs) | Cross-Val SD RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Setting A (+ Age)** | Linear Regression (OLS) | 3.0483 | 0.0437 | 4.1663 | 0.0862 |
| | Random Forest | 3.1328 | 0.0336 | 4.2580 | 0.0848 |
| | SVR ($C=5.0$) | 3.0055 | 0.0512 | 4.2222 | 0.0947 |
| | Gradient Boosting | 2.9506 | 0.0498 | 4.0170 | 0.0949 |
| | XGBoost | 2.9378 | 0.0516 | 4.0084 | 0.0936 |
| | **Multilayer Perceptron (MLP)**| **2.8849** | **0.0442** | **3.9362** | **0.0881** |
| **Setting B (Biomarkers Only)** | Linear Regression (OLS) | 10.4582 | 0.0849 | 12.9861 | 0.1171 |
| | Random Forest | 9.3578 | 0.0330 | 11.8652 | 0.0596 |
| | SVR ($C=5.0$) | 10.4094 | 0.0890 | 13.0642 | 0.1247 |
| | Gradient Boosting | 9.2301 | 0.0525 | 11.6902 | 0.0711 |
| | XGBoost | 9.1470 | 0.0647 | 11.6206 | 0.0663 |
| | **Multilayer Perceptron (MLP)**| **9.0537** | **0.0999** | **11.5248** | **0.1169** |

---

### 5.3 Holdout Test Set Generalization & Model Selection
Evaluating the trained models on the sequestered holdout test corpus ($N = 7,145$) confirmed the cross-validation rankings:

#### Table 3: Performance Under Setting A (Biomarkers + Chronological Age, $d=13$)
| Rank | Evaluated Architecture | Holdout MAE (Yrs) | Holdout RMSE (Yrs) | $R^2$ Variance | Pearson $r$ | Spearman $\rho$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Multilayer Perceptron (Neural Net)** | **2.8401** | **3.8643** | **0.9640** | **0.9819** | **0.9843** |
| 2 | XGBoost Architecture | 2.8674 | 3.9122 | 0.9631 | 0.9814 | 0.9838 |
| 3 | Gradient Boosted Trees | 2.9173 | 3.9740 | 0.9619 | 0.9808 | 0.9833 |
| 4 | Linear Regression (OLS) | 2.9995 | 4.0992 | 0.9595 | 0.9795 | 0.9822 |
| 5 | Support Vector Regression | 2.9578 | 4.1445 | 0.9586 | 0.9794 | 0.9821 |
| 6 | Random Forest Regressor | 3.0659 | 4.1565 | 0.9583 | 0.9790 | 0.9820 |

#### Table 4: Performance Under Setting B (Biomarkers Only, $d=12$, Age Withheld)
| Rank | Evaluated Architecture | Holdout MAE (Yrs) | Holdout RMSE (Yrs) | $R^2$ Variance | Pearson $r$ | Spearman $\rho$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Multilayer Perceptron (Neural Net)** | **8.9199** | **11.3648** | **0.6886** | **0.8299** | **0.8249** |
| 2 | XGBoost Architecture | 9.0631 | 11.4855 | 0.6819 | 0.8258 | 0.8213 |
| 3 | Gradient Boosted Trees | 9.1038 | 11.5553 | 0.6780 | 0.8235 | 0.8183 |
| 4 | Random Forest Regressor | 9.2997 | 11.8453 | 0.6617 | 0.8140 | 0.8079 |
| 5 | Linear Regression (OLS) | 10.4157 | 12.9101 | 0.5981 | 0.7734 | 0.7815 |
| 6 | Support Vector Regression | 10.3445 | 12.9754 | 0.5940 | 0.7729 | 0.7845 |

**Analytical Selection Rationale:**  
The **Multilayer Perceptron Neural Network** demonstrated the lowest overall generalization error on the holdout partition ($\text{RMSE} = 3.8643\text{ years}$, $\text{MAE} = 2.8401\text{ years}$, $R^2 = 0.9640$), with **XGBoost** performing nearly identically ($\text{RMSE} = 3.9122\text{ years}$, $R^2 = 0.9631$).  
The comparative contrast between Setting A and Setting B provides clear empirical insight: routine clinical blood chemistry alone explains 68.86% of phenotypic biological aging variance without any reference to chronological age. However, omitting calendar time nearly triples the residual error (RMSE rising from 3.86 to 11.36 years). This confirms that calendar age provides the foundational temporal coordinate, while physiological biomarkers govern individual-specific variation around this baseline.

Residual diagnostic checks for the optimal model confirmed that prediction errors were homoscedastic and symmetrically distributed around zero (Mean error: -0.016 years, Median error: 0.003 years).

---

### 5.4 Distributional Characteristics of Inferred Age Acceleration
Evaluating biological age acceleration ($\delta = \widehat{\text{BA}} - \text{CA}$) across the 7,145 holdout test participants yielded the following statistical profile:
- **Sample Mean ($\bar{\delta}$):** $-2.9217$ years
- **Sample Median:** $-3.8241$ years
- **Standard Dispersion ($\sigma$):** $5.0804$ years
- **Interquartile Boundary (IQR):** $5.8181$ years
- **Observed Boundaries:** $[-14.7870, +33.8292]$ years
- **Skewness Coefficient:** $+1.2773$

The pronounced positive skewness indicates that while the vast majority of individuals cluster closely around normal physiological aging trajectories, a distinct high-risk sub-cohort displays severe physiological aging acceleration exceeding their calendar age by up to three decades.

---

### 5.5 Biomarker Explainability & Attribution
Feature importance analyses using both out-of-sample permutation degradation and TreeSHAP on the gradient boosted tree ensemble identified consistent physiological drivers:

| Analyte Rank | Clinical Feature | Organ System Domain | Mean Absolute SHAP (Years) | Permutation RMSE Inflation (Years) |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `age` | Macro-Temporal Coordinate | 15.1018 | 20.5169 $\pm$ 0.222 |
| 2 | `hba1c` | Glycemic Regulation | 1.5097 | 1.4808 $\pm$ 0.035 |
| 3 | `lncreat_umol` | Glomerular Clearance / Muscle | 1.6779 | 1.2694 $\pm$ 0.039 |
| 4 | `albumin_gL` | Hepatic Protein Output / Nutrition | 1.4755 | 0.9702 $\pm$ 0.025 |
| 5 | `lymph` | Adaptive Immunity / Senescence | 1.0957 | 0.4912 $\pm$ 0.014 |
| 6 | `wbc` | Innate Immunity / Inflammaging | 0.9013 | 0.3346 $\pm$ 0.013 |
| 7 | `mcv` | Erythrocyte Fragility / Turnover | 0.6261 | 0.2576 $\pm$ 0.037 |
| 8 | `alp` | Hepatobiliary & Bone Turnover | 0.5922 | 0.1543 $\pm$ 0.009 |
| 9 | `lncrp` | Sterile Vascular Inflammation | 0.3703 | 0.0712 $\pm$ 0.011 |
| 10 | `bun` | Nitrogen Excretion / Hydration | 0.0665 | 0.0702 $\pm$ 0.005 |
| 11 | `sbp` | Hemodynamic Vascular Load | 0.1617 | 0.0471 $\pm$ 0.008 |
| 12 | `totchol` | Lipid Transport Homeostasis | 0.2517 | 0.0274 $\pm$ 0.005 |
| 13 | `uap` | Purine Catabolism | 0.0670 | 0.0190 $\pm$ 0.004 |

SHAP beeswarm analyses revealed clear directional biological effects:
- **Elevated Glycated Hemoglobin and Creatinine:** Consistently yield large positive SHAP attributions, driving biological age upward.
- **Serum Albumin:** Exhibits strong negative SHAP values, confirming its role as an indicator of physiological reserve and longevity buffering.
- **Lymphocyte Contraction & Elevated Leukocytes:** Consistently push predictions toward accelerated biological aging, reflecting the combined effects of adaptive immunosenescence and chronic low-grade inflammation.

---

### 5.6 Survivorship and Mortality Risk Findings
Longevity analyses on the holdout survival cohort ($N = 5,280$; 424 confirmed deaths; median follow-up 116.0 person-months) validated the prognostic significance of the inferred biological age acceleration:

#### 1. Kaplan-Meier Survival Divergence
Stratifying holdout individuals into biologically accelerated ($\delta > 0$) versus biologically decelerated ($\delta \le 0$) groups demonstrated pronounced survival curve divergence. The accelerated cohort suffered significantly higher cumulative mortality attrition (two-tailed log-rank $\chi^2 = 625.76, p < 0.0001$).

#### 2. Semi-Parametric Cox Proportional Hazards Models
Cox proportional hazards regression confirmed that algorithmically inferred biological age acceleration is a strong, independent predictor of prospective mortality:

| Model Structure | Covariate Specification | Hazard Ratio (HR) | 95% Confidence Interval | Wald Test $p$-value |
| :--- | :--- | :---: | :---: | :---: |
| **Model 1: Univariate** | Inferred Acceleration $\delta$ (per +1 yr) | **1.0978** | [1.0818, 1.1141] | **$< 0.0001$** |
| **Model 2: Multivariable** | Inferred Acceleration $\delta$ (per +1 yr) | **1.0678** | [1.0516, 1.0842] | **$< 0.0001$** |
| | Baseline Chronological Age (per +1 yr) | 1.0894 | [1.0811, 1.0977] | $< 0.0001$ |
| **Model 3: Reference** | Baseline BioAge `phenoage_advance0` | 1.0623 | [1.0502, 1.0745] | $< 0.0001$ |

In univariate analysis, each additional year of biological age acceleration is associated with a **9.78% increase in all-cause mortality hazard**. Crucially, in the multivariable model adjusting for chronological age, biological age acceleration remained highly statistically significant ($p < 0.0001$), conferring an independent **6.78% elevation in mortality hazard per year** ($\text{HR} = 1.0678$, 95% CI: 1.0516–1.0842). This slightly exceeded the predictive fidelity of the baseline closed-form BioAge PhenoAge advance ($\text{HR} = 1.0623$).

---

## 6. Discussion

### 6.1 Computational Architecture Insights
Both the Multilayer Perceptron and XGBoost outperformed ordinary least squares and linear support vector machines. While linear regression achieved competitive baseline fidelity ($R^2 = 0.9595$) due to the partially linear structure of the original PhenoAge equation, neural and gradient boosted architectures successfully captured higher-order analyte interactions and non-linear boundaries, reducing holdout RMSE to 3.8643 years.

### 6.2 Pathophysiological Mechanisms of Dominant Analytes
The observed analyte importance rankings align closely with recognized hallmarks of mammalian aging:
1. **Glycemic Stress and Protein Glycation (HbA1c):** Elevated circulating glucose accelerates the non-enzymatic formation of advanced glycation end-products (AGEs), which cross-link extracellular collagen, stiffen systemic vasculature, and trigger pro-inflammatory RAGE signaling.
2. **Nephron Sclerosis and Metabolic Clearance (Creatinine, BUN):** Progressive glomerulosclerosis reduces clearance efficiency, while co-occurring sarcopenia alters creatinine generation, serving as an indicator of combined renal and muscular decline.
3. **Hepatic Synthesis and Frailty Reserve (Albumin):** Serum albumin serves as a negative acute-phase reactant; declining concentrations indicate persistent microvascular permeability, subclinical hepatic synthetic decline, and nutritional frailty.
4. **Adaptive Immunosenescence and Inflammaging (Lymphocyte %, WBC, CRP):** Thymic involution contracts naive T-cell diversity, accompanied by clonal expansions of senescent memory cells and elevated myeloid leukocyte counts. This shifts the immune profile toward chronic sterile inflammation (*inflammaging*), accelerating multi-organ tissue degeneration.

### 6.3 The Informational Interplay of Chronological Time
The comparative contrast between Setting A and Setting B highlights a key principle in computational geroscience. While clinical blood chemistry alone explains over two-thirds of phenotypic aging variance ($R^2 = 0.6886$), removing chronological age triples the residual estimation error. Chronological age provides the essential global temporal baseline, while somatic biomarkers capture individual physiological deviation from expected population trajectories.

---

## 7. Study Limitations

1. **Cross-Sectional Observational Biomarkers:** Biomarker measurements in NHANES reflect single-timepoint venipuncture subject to circadian variation, temporary acute infection, or transient metabolic fluctuations, even though mortality follow-up extends across years.
2. **Complete-Case Filtering & Attrition:** Filtering complete cases across the 14 core features reduced the cohort from 55,081 to 35,723 individuals, primarily due to rotational sampling of C-reactive protein. While sample demographics remained broad, complete-case filtering may introduce mild healthy-responder selection bias.
3. **Surrogate Formulation Dependency:** The target variable `phenoage0` is an established algorithmic surrogate derived from Gompertz mortality calibrations rather than an unmediated biological truth.
4. **Associative, Non-Causal Conclusions:** Demonstrating that biological age acceleration independently predicts mortality hazard does not establish direct physiological causality. Interventions that modify single biomarkers may not alter the pace of underlying systemic senescence.
5. **Demographic and National Specificity:** The NHANES dataset represents non-institutionalized United States civilians from 1999–2018. Validation across international cohorts with distinct dietary, environmental, and healthcare structures remains necessary.

---

## 8. Conclusion

This academic investigation constructed a reproducible, leakage-free machine learning pipeline for biological age estimation and mortality risk stratification using routine clinical biomarkers from the NHANES IV benchmark cohort. The Multilayer Perceptron Neural Network and XGBoost architectures demonstrated high predictive accuracy ($R^2 = 0.9640$, RMSE = 3.8643 years). Explainability audits identified glycated hemoglobin, serum creatinine, albumin, and immune cell proportions as the primary somatic drivers of biological aging. Furthermore, survival modeling confirmed that algorithmically inferred biological age acceleration is independently associated with all-cause mortality risk ($\text{HR} = 1.0678$ per year, $p < 0.0001$), supporting its utility as an objective, non-invasive risk stratification metric in aging research.

---

## References

1. Levine, M. E., Lu, A. T., Quach, A., Chen, B. H., Assimes, T. L., Bandinelli, S., ... & Horvath, S. (2018). An epigenetic biomarker of aging for lifespan and healthspan. *Aging (Albany NY)*, 10(4), 573–591.
2. Kwon, D., Belsky, D. W., & BioAge Consortium. (2021). BioAge: An R package for calculating biological age. *GitHub Distribution*: https://github.com/dayoonkwon/BioAge.
3. Klemera, P., & Doubal, S. (2006). A new approach to the concept and computation of biological age. *Mechanisms of Ageing and Development*, 127(3), 240–248.
4. Cohen, A. A., Milot, E., Li, Q., Bergeron, P., Poirier, R., Dusseault-Bélanger, F., & Fülöp, T. (2013). Detection of a novel, fast-moving component of physiological dysregulation in aging humans. *Mechanisms of Ageing and Development*, 134(3-4), 108–117.
5. Horvath, S. (2013). DNA methylation age of human tissues and cell types. *Genome Biology*, 14(10), R115.
6. Hannum, G., Guinney, J., Zhao, L., Zhang, L., Hughes, G., Sadda, S., ... & Zhang, K. (2013). Genome-wide methylation profiles reveal quantitative views of human aging rates. *Molecular Cell*, 49(2), 359–367.
7. Lu, A. T., Quach, A., Wilson, J. G., Reiner, A. P., Aviv, A., Raj, K., ... & Horvath, S. (2019). DNA methylation GrimAge strongly predicts lifespan and healthspan. *Aging (Albany NY)*, 11(2), 303–327.
8. Cole, J. H., Ritchie, S. J., Bastin, M. E., Valdés Hernández, M. C., Muñoz Maniega, S., Royle, N., ... & Deary, I. J. (2018). Brain age predicts mortality. *Molecular Psychiatry*, 23(5), 1385–1392.
9. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774.
10. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
