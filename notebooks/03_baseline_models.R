# Baseline ordinary least squares regressions
# Compares setting A (with chronological age) vs setting B (biomarkers only)

source("R/preprocessing.R")
source("R/models.R")
source("R/evaluation.R")

raw_df <- load_clean_cohort("data/processed/nhanes4_core_35723.csv")
splits <- make_train_test_split(raw_df, split_ratio = 0.8, seed = 42)

form_a <- as.formula(paste("phenoage0 ~", paste(FEATS_A, collapse = " + ")))
form_b <- as.formula(paste("phenoage0 ~", paste(FEATS_B, collapse = " + ")))

# 1. Fit setting A baseline
fit_a <- fit_ols(splits$train_scaled_A, form_a)
preds_a <- predict(fit_a, newdata = splits$test_scaled_A)
res_a <- calc_regression_metrics(splits$test_scaled_A$phenoage0, preds_a, "OLS Baseline (Setting A)")
print(res_a)

# 2. Fit setting B baseline
fit_b <- fit_ols(splits$train_scaled_B, form_b)
preds_b <- predict(fit_b, newdata = splits$test_scaled_B)
res_b <- calc_regression_metrics(splits$test_scaled_B$phenoage0, preds_b, "OLS Baseline (Setting B)")
print(res_b)
