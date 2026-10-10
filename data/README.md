# Data Documentation: NHANES IV Biological Age Cohort

## Source & Provenance
This project utilizes the built-in `NHANES4` benchmark dataset from the publicly available R package **BioAge** by Dayoon Kwon et al. ([https://github.com/dayoonkwon/BioAge](https://github.com/dayoonkwon/BioAge)).

- **Primary Source:** National Health and Nutrition Examination Survey (NHANES) continuous waves (1999–2018), National Center for Health Statistics (NCHS) / CDC.
- **Repository Package:** `BioAge` (Version 0.1.0)
- **Original Dataset File:** `BioAge/data/NHANES4.rda` (55,081 participants × 89 variables)

## Processed Analytical Datasets
The original `NHANES4.rda` file was preserved without modification. Clean derived datasets were generated and stored in `data/processed/`:

1. `nhanes4_core_35723.csv`:
   - Complete cases across the 14 core variables (`age`, 12 clinical biomarkers, and `phenoage0`).
   - Dimensions: **35,723 participants × 18 columns** (includes `sampleID`, core features, and linked survival outcome variables).
2. `nhanes4_survival_subset.csv`:
   - Complete cases for core features with complete linked mortality data (`status` and `time`).
   - Dimensions: **26,425 participants** (2,181 confirmed deaths, 24,244 censored alive).
3. `missingness_before_filter.csv`:
   - Audit of missing values across all 55,081 participants prior to filtering.
4. `data_summary_stats.csv`:
   - Mean, standard deviation, median, IQR, range, and skewness for all core variables.

## Feature Dictionary

| Variable | Full Name | Clinical Domain | Standard Units |
| :--- | :--- | :--- | :--- |
| `sampleID` | Participant Identifier | Administrative | Alphanumeric |
| `age` | Chronological Age | Demographics | Years |
| `albumin_gL` | Serum Albumin | Hepatic / Nutrition | g/L |
| `alp` | Alkaline Phosphatase | Hepatobiliary / Bone | U/L |
| `lncrp` | log(C-Reactive Protein) | Systemic Inflammation | mg/dL (log) |
| `totchol` | Total Cholesterol | Cardiovascular / Lipid | mg/dL |
| `lncreat_umol` | log(Serum Creatinine) | Renal Filtration / Muscle | $\mu\text{mol/L}$ (log) |
| `hba1c` | Glycated Hemoglobin | Glycemic Homeostasis | % |
| `sbp` | Systolic Blood Pressure | Hemodynamic Load | mmHg |
| `bun` | Blood Urea Nitrogen | Renal Filtration | mg/dL |
| `uap` | Uric Acid | Purine Metabolism | mg/dL |
| `lymph` | Lymphocyte Percentage | Adaptive Immunity | % |
| `mcv` | Mean Corpuscular Volume | Hematopoiesis | fL |
| `wbc` | White Blood Cell Count | Innate Immunity | $10^3/\mu\text{L}$ |
| `phenoage0` | Phenotypic Biological Age | Target Biological Age | Years |
| `phenoage_advance0` | PhenoAge Advance | Baseline Age Acceleration | Years |
| `status` | Mortality Status | Vital Outcome | 1 = Deceased, 0 = Censored |
| `time` | Follow-up Duration | Longitudinal Survival | Person-Months |
