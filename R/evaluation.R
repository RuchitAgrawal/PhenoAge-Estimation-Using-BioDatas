suppressPackageStartupMessages({
  library(ggplot2)
})

# compute standard regression validation metrics
calc_regression_metrics <- function(y_true, y_pred, m_label = "Model") {
  diffs <- y_true - y_pred
  mae_val <- mean(abs(diffs))
  rmse_val <- sqrt(mean(diffs^2))
  
  tot_ss <- sum((y_true - mean(y_true))^2)
  res_ss <- sum(diffs^2)
  r2_val <- 1 - (res_ss / tot_ss)
  
  r_pearson <- cor(y_true, y_pred, method = "pearson")
  r_spearman <- cor(y_true, y_pred, method = "spearman")
  
  data.frame(
    Model = m_label,
    MAE = round(mae_val, 4),
    RMSE = round(rmse_val, 4),
    R2 = round(r2_val, 4),
    Pearson_r = round(r_pearson, 4),
    Spearman_r = round(r_spearman, 4)
  )
}

# scatter plot with 45-degree identity reference line
plot_pred_vs_truth <- function(y_true, y_pred, m_label = "Model", save_path = NULL) {
  plt_df <- data.frame(True_Age = y_true, Pred_Age = y_pred)
  g <- ggplot(plt_df, aes(x = True_Age, y = Pred_Age)) +
    geom_point(alpha = 0.25, color = "steelblue", size = 1) +
    geom_abline(intercept = 0, slope = 1, color = "darkred", linetype = "dashed", linewidth = 0.8) +
    geom_smooth(method = "lm", color = "darkorange", se = FALSE, linewidth = 0.8) +
    theme_bw() +
    labs(
      title = paste("Predicted vs True PhenoAge -", m_label),
      x = "Ground Truth BioAge (Years)",
      y = "Model Prediction (Years)"
    )
  
  if (!is.null(save_path)) {
    ggsave(save_path, plot = g, width = 7, height = 6, dpi = 300)
  }
  g
}

# aliases for notebook compatibility
compute_metrics <- calc_regression_metrics
plot_actual_vs_predicted_r <- plot_pred_vs_truth
