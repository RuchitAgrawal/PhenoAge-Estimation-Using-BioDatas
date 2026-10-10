# Exploratory screening of NHANES IV cohort for biological age project
# Checks distribution of PhenoAge target and correlations across the 12 biomarkers

data_file <- "data/processed/nhanes4_core_35723.csv"
if (!file.exists(data_file)) {
  stop("Input dataset not found. Run R/extract_and_validate_data.R first.")
}

cohort <- read.csv(data_file)
cat(sprintf("[info] loaded %d observations (%d attributes)\n", nrow(cohort), ncol(cohort)))

# check target distribution
cat(sprintf("PhenoAge mean = %.2f (sd = %.2f), median = %.2f, range = [%.2f, %.2f]\n",
            mean(cohort$phenoage0), sd(cohort$phenoage0), median(cohort$phenoage0), 
            min(cohort$phenoage0), max(cohort$phenoage0)))

# check raw correlation between calendar age and PhenoAge
r_age <- cor(cohort$age, cohort$phenoage0)
cat(sprintf("Chronological age vs PhenoAge correlation: r = %.4f\n", r_age))

# rank biomarker correlations with target
biomarker_cols <- c("age", "albumin_gL", "alp", "lncrp", "totchol", 
                    "lncreat_umol", "hba1c", "sbp", "bun", "uap", 
                    "lymph", "mcv", "wbc", "phenoage0")
c_mat <- cor(cohort[, biomarker_cols])
cat("\nBiomarker correlations ranked against PhenoAge target:\n")
ranked <- sort(c_mat["phenoage0", colnames(c_mat) != "phenoage0"], decreasing = TRUE)
print(round(ranked, 4))
