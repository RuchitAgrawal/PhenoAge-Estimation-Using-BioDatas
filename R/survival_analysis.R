suppressPackageStartupMessages({
  library(survival)
  library(ggplot2)
})

# fit Kaplan-Meier curves and calculate log-rank test statistic
fit_km_stratified <- function(df, time_var = "time", event_var = "status", strata_var = "accel_group") {
  f <- as.formula(paste0("Surv(", time_var, ", ", event_var, ") ~ ", strata_var))
  km <- survfit(f, data = df)
  diff_test <- survdiff(f, data = df)
  
  # chi-squared p-value with k-1 degrees of freedom
  df_deg <- length(diff_test$n) - 1
  pval <- 1 - pchisq(diff_test$chisq, df = df_deg)
  
  list(
    km_fit = km,
    logrank = diff_test,
    chisq = diff_test$chisq,
    p_value = pval
  )
}

# fit univariate and age-adjusted Cox PH models
fit_cox_hazard_models <- function(df, time_var = "time", event_var = "status") {
  # 1. unadjusted acceleration
  f_uni <- as.formula(paste0("Surv(", time_var, ", ", event_var, ") ~ bio_age_acceleration"))
  fit_uni <- coxph(f_uni, data = df)
  
  # 2. adjusted for chronological age
  f_multi <- as.formula(paste0("Surv(", time_var, ", ", event_var, ") ~ bio_age_acceleration + age"))
  fit_multi <- coxph(f_multi, data = df)
  
  # assess proportional hazards assumption via Schoenfeld residuals
  ph_check <- cox.zph(fit_multi)
  
  format_cox_tbl <- function(fit) {
    s <- summary(fit)
    data.frame(
      Variable = rownames(s$coefficients),
      Hazard_Ratio = exp(s$coefficients[, "coef"]),
      CI_Lower = s$conf.int[, "lower .95"],
      CI_Upper = s$conf.int[, "upper .95"],
      p_value = s$coefficients[, "Pr(>|z|)"]
    )
  }
  
  list(
    cox_uni = fit_uni,
    cox_multi = fit_multi,
    zph_test = ph_check,
    summary_uni = format_cox_tbl(fit_uni),
    summary_multi = format_cox_tbl(fit_multi)
  )
}

# aliases for notebook compatibility
run_km_analysis <- fit_km_stratified
run_coxph_models <- fit_cox_hazard_models
