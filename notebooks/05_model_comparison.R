# Inspect holdout test set metrics across all trained architectures

path_a <- "results/tables/model_comparison_exp_A.csv"
path_b <- "results/tables/model_comparison_exp_B.csv"

if (file.exists(path_a) && file.exists(path_b)) {
  tbl_a <- read.csv(path_a)
  tbl_b <- read.csv(path_b)
  
  cat("Setting A: Chronological Age + Biomarkers (Holdout N = 7,145)\n")
  print(tbl_a)
  
  cat("\nSetting B: Biomarkers Only (Holdout N = 7,145)\n")
  print(tbl_b)
  
  top_row <- which.min(tbl_a$RMSE)
  cat(sprintf("\nTop Performing Model: %s (RMSE = %.4f yrs, R2 = %.4f)\n",
              tbl_a$Model[top_row], tbl_a$RMSE[top_row], tbl_a$R2[top_row]))
} else {
  cat("[notice] Execute python run_pipeline.py to compute and save benchmark tables.\n")
}
