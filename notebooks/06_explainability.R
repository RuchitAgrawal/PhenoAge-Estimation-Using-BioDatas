# Inspect biomarker importance tables from permutation and SHAP runs

path_perm <- "results/tables/permutation_importance.csv"
path_shap <- "results/tables/shap_importance_ranking.csv"

if (file.exists(path_perm)) {
  tbl_perm <- read.csv(path_perm)
  cat("Top Analytes by Permutation Error Inflation (Test Set):\n")
  print(head(tbl_perm, 7))
}

if (file.exists(path_shap)) {
  tbl_shap <- read.csv(path_shap)
  cat("\nTop Analytes by Mean Absolute TreeSHAP Attribution:\n")
  print(head(tbl_shap, 7))
}
