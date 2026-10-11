# Longitudinal survivorship modeling using Kaplan-Meier and Cox Proportional Hazards

source("R/survival_analysis.R")

surv_file <- "data/processed/nhanes4_survival_subset.csv"
if (!file.exists(surv_file)) {
  stop("Survival dataset missing. Run R/extract_and_validate_data.R first.")
}

surv_df <- read.csv(surv_file)
cat(sprintf("[info] loaded survival cohort: %d records (%d events, %d censored)\n",
            nrow(surv_df), sum(surv_df$status == 1), sum(surv_df$status == 0)))

# classify participants by age acceleration delta sign
surv_df$bio_age_acceleration <- surv_df$phenoage_advance0
surv_df$accel_group <- ifelse(surv_df$bio_age_acceleration > 0, "Accelerated", "Decelerated")

# 1. Log-rank test on Kaplan-Meier curves
km_res <- fit_km_stratified(surv_df, time_var = "time", event_var = "status", strata_var = "accel_group")
cat(sprintf("Log-Rank Test: Chi2 = %.2f, p = %.3e\n", km_res$chisq, km_res$p_value))

# 2. Cox PH regression
cox_res <- fit_cox_hazard_models(surv_df, time_var = "time", event_var = "status")

cat("\nUnadjusted Model (Biological Age Acceleration alone):\n")
print(cox_res$summary_uni)

cat("\nMultivariable Model (Adjusted for Chronological Age):\n")
print(cox_res$summary_multi)
