suppressPackageStartupMessages({
  library(caret)
  library(e1071)
})

fit_ols <- function(df, form) {
  lm(form, data = df)
}

# train linear SVR using e1071 package
fit_svr <- function(df, form, cost = 1.0) {
  e1071::svm(form, data = df, type = "eps-regression", kernel = "linear", cost = cost)
}

# 5-fold CV helper for linear regression benchmark
cv_ols_5fold <- function(df, form, seed = 42) {
  set.seed(seed)
  folds <- caret::createFolds(df$phenoage0, k = 5, list = TRUE)
  maes <- numeric(5)
  rmses <- numeric(5)
  
  for (i in seq_along(folds)) {
    val_rows <- folds[[i]]
    fit <- lm(form, data = df[-val_rows, ])
    preds <- predict(fit, newdata = df[val_rows, ])
    diffs <- df[val_rows, "phenoage0"] - preds
    
    maes[i] <- mean(abs(diffs))
    rmses[i] <- sqrt(mean(diffs^2))
  }
  
  list(
    mae_mean = mean(maes),
    mae_sd = sd(maes),
    rmse_mean = mean(rmses),
    rmse_sd = sd(rmses)
  )
}

# aliases for notebook compatibility
train_linear_regression <- fit_ols
train_svr <- fit_svr
run_5fold_cv_lm <- cv_ols_5fold
