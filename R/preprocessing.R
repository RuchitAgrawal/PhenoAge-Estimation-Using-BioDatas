suppressPackageStartupMessages({
  library(dplyr)
})

# full predictor set (setting a)
FEATS_A <- c("age", "albumin_gL", "alp", "lncrp", "totchol", 
             "lncreat_umol", "hba1c", "sbp", "bun", "uap", 
             "lymph", "mcv", "wbc")

# biomarkers only (setting b)
FEATS_B <- FEATS_A[FEATS_A != "age"]

TARGET_COL <- "phenoage0"

load_clean_cohort <- function(path = "data/processed/nhanes4_core_35723.csv") {
  read.csv(path, stringsAsFactors = FALSE)
}

# 80/20 train-test partition with leakage-free scaling
make_train_test_split <- function(dat, split_ratio = 0.8, seed = 42) {
  set.seed(seed)
  n <- nrow(dat)
  train_idx <- sample(seq_len(n), size = floor(split_ratio * n))
  test_idx <- setdiff(seq_len(n), train_idx)
  
  tr <- dat[train_idx, ]
  te <- dat[test_idx, ]
  
  # compute means/sds strictly on training rows
  params_a <- lapply(tr[, FEATS_A], function(x) c(m = mean(x), s = sd(x)))
  params_b <- lapply(tr[, FEATS_B], function(x) c(m = mean(x), s = sd(x)))
  
  apply_scaler <- function(df, params, cols) {
    out <- df
    for (col in cols) {
      out[[col]] <- (df[[col]] - params[[col]]["m"]) / params[[col]]["s"]
    }
    out
  }
  
  list(
    train_raw = tr,
    test_raw = te,
    train_scaled_A = apply_scaler(tr, params_a, FEATS_A),
    test_scaled_A = apply_scaler(te, params_a, FEATS_A),
    train_scaled_B = apply_scaler(tr, params_b, FEATS_B),
    test_scaled_B = apply_scaler(te, params_b, FEATS_B),
    scale_params_A = params_a,
    scale_params_B = params_b
  )
}

# aliases for notebook compatibility
load_and_clean_data <- load_clean_cohort
split_data <- make_train_test_split
CORE_FEATURES_A <- FEATS_A
CORE_FEATURES_B <- FEATS_B
TARGET <- TARGET_COL
