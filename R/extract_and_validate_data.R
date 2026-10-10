# load the built-in rda file from Dayoon Kwon's BioAge package
cat("[+] loading BioAge NHANES4.rda dataset...\n")
load("c:/Users/krm06/Downloads/ml/BioAge/data/NHANES4.rda")

n_total <- nrow(NHANES4)
n_cols <- ncol(NHANES4)
cat(sprintf("    raw cohort dimensions: %d rows x %d columns\n", n_total, n_cols))

# 13 candidate clinical markers + PhenoAge target + survival tracking
features <- c("age", "albumin_gL", "alp", "lncrp", "totchol", 
              "lncreat_umol", "hba1c", "sbp", "bun", "uap", 
              "lymph", "mcv", "wbc")
target <- "phenoage0"
surv_cols <- c("status", "time", "phenoage_advance0")
all_cols <- c(features, target, surv_cols)

# audit missingness proportions across full population
na_counts <- sapply(NHANES4[, all_cols], function(col) sum(is.na(col)))
na_pcts <- sapply(NHANES4[, all_cols], function(col) mean(is.na(col)) * 100)

df_miss <- data.frame(
  Variable = all_cols,
  Missing_Count = as.numeric(na_counts),
  Missing_Percentage = round(as.numeric(na_pcts), 2),
  Available_Count = n_total - as.numeric(na_counts),
  Total_N = n_total
)
write.csv(df_miss, "data/processed/missingness_before_filter.csv", row.names = FALSE)

# filter complete cases on the 14 core modeling features
core_vars <- c(features, target)
valid_idx <- complete.cases(NHANES4[, core_vars])
df_clean <- NHANES4[valid_idx, c("sampleID", core_vars, surv_cols)]
cat(sprintf("[+] filtered core complete cases: %d rows (retained %d input variables)\n", 
            nrow(df_clean), length(core_vars)))

write.csv(df_clean, "data/processed/nhanes4_core_35723.csv", row.names = FALSE)

# calculate baseline distribution statistics for research paper
df_stats <- do.call(rbind, lapply(core_vars, function(v) {
  vals <- df_clean[[v]]
  data.frame(
    Variable = v,
    Mean = mean(vals, na.rm = TRUE),
    SD = sd(vals, na.rm = TRUE),
    Median = median(vals, na.rm = TRUE),
    IQR = IQR(vals, na.rm = TRUE),
    Min = min(vals, na.rm = TRUE),
    Max = max(vals, na.rm = TRUE),
    Skewness = e1071::skewness(vals, na.rm = TRUE)
  )
}))
write.csv(df_stats, "data/processed/data_summary_stats.csv", row.names = FALSE)

# extract linked NDI longitudinal survival cohort
has_surv <- valid_idx & !is.na(NHANES4$status) & !is.na(NHANES4$time)
df_surv <- NHANES4[has_surv, c("sampleID", core_vars, surv_cols)]
n_dead <- sum(df_surv$status == 1, na.rm = TRUE)
cat(sprintf("[+] survival subset: %d participants (%d confirmed deaths, %d censored)\n", 
            nrow(df_surv), n_dead, nrow(df_surv) - n_dead))

write.csv(df_surv, "data/processed/nhanes4_survival_subset.csv", row.names = FALSE)
cat("[+] data processing complete.\n")
