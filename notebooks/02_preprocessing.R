# Partitioning and standardization workflow
# Demonstrates 80/20 train/test split with zero leakage

source("R/preprocessing.R")

raw_df <- load_clean_cohort("data/processed/nhanes4_core_35723.csv")
splits <- make_train_test_split(raw_df, split_ratio = 0.8, seed = 42)

cat(sprintf("Training split: %d rows\n", nrow(splits$train_raw)))
cat(sprintf("Holdout test split: %d rows\n", nrow(splits$test_raw)))

cat("\nSample training centering/scaling parameters (derived strictly on train split):\n")
for (var_name in names(splits$scale_params_A)[1:5]) {
  vals <- splits$scale_params_A[[var_name]]
  cat(sprintf("  %s -> mean: %.3f, sd: %.3f\n", var_name, vals["m"], vals["s"]))
}
