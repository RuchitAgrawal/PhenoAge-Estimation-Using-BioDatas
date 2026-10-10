suppressPackageStartupMessages({
  library(ggplot2)
})

# extract beta weights from linear model fitted on standardized predictors
get_standardized_betas <- function(fit) {
  smry <- summary(fit)
  betas <- smry$coefficients[-1, ] # drop intercept
  
  tbl <- data.frame(
    Feature = rownames(betas),
    Beta = betas[, "Estimate"],
    SE = betas[, "Std. Error"],
    t_stat = betas[, "t value"],
    p_val = betas[, "Pr(>|t|)"]
  )
  tbl[order(abs(tbl$Beta), decreasing = TRUE), ]
}

# plot horizontal bar chart of beta coefficients
plot_beta_rankings <- function(tbl, save_path = NULL) {
  g <- ggplot(tbl, aes(x = reorder(Feature, abs(Beta)), y = Beta)) +
    geom_bar(stat = "identity", fill = "steelblue", color = "black", alpha = 0.85) +
    coord_flip() +
    theme_bw() +
    labs(
      title = "Standardized Regression Betas (Feature Contribution)",
      x = "Analyte",
      y = "Standardized Beta Weight"
    )
  
  if (!is.null(save_path)) {
    ggsave(save_path, plot = g, width = 8, height = 5, dpi = 300)
  }
  g
}

# aliases for notebook compatibility
extract_standardized_coefs <- function(lm_model) {
  df <- get_standardized_betas(lm_model)
  names(df) <- c("Feature", "Estimate", "Std_Error", "t_value", "p_value")
  df
}
plot_coef_importance <- plot_beta_rankings
