# Cross-validation stability check for linear baseline on training data

source("R/preprocessing.R")
source("R/models.R")

raw_df <- load_clean_cohort("data/processed/nhanes4_core_35723.csv")
splits <- make_train_test_split(raw_df, split_ratio = 0.8, seed = 42)

form_a <- as.formula(paste("phenoage0 ~", paste(FEATS_A, collapse = " + ")))

cat("Running 5-fold cross-validation across training rows (n = 28,578)...\n")
cv_res <- cv_ols_5fold(splits$train_scaled_A, form_a, seed = 42)

cat(sprintf("Linear model 5-fold CV results:\n"))
cat(sprintf("  Mean MAE:  %.4f (sd = %.4f)\n", cv_res$mae_mean, cv_res$mae_sd))
cat(sprintf("  Mean RMSE: %.4f (sd = %.4f)\n", cv_res$rmse_mean, cv_res$rmse_sd))
