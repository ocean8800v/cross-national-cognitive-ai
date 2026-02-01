packages <- c("rmda", "gridExtra", "grid")
lapply(packages, function(p) {
  if (!require(p, character.only = TRUE)) {
    install.packages(p)
    library(p, character.only = TRUE)
  }
})

scale_map <- list("1" = 14, "2" = 3, "3" = 2, "4" = 1, "5" = 2)

country_titles <- c(
  "1" = "HRS (US, Questionnaire #14)",
  "2" = "ELSA (England, Questionnaire #3)",
  "3" = "LASI (India, Questionnaire #2)",
  "4" = "MHAS (Mexico, Questionnaire #1)",
  "5" = "External CHARLS (China, Questionnaire #2)"
)

resource_thresholds <- list(
  "1" = c(0.076, 0.095, 0.112, 0.127, 0.136),
  "2" = c(0.076, 0.093, 0.105, 0.124, 0.140),
  "3" = c(0.385, 0.485, 0.558, 0.610, 0.647),
  "4" = c(0.207, 0.269, 0.317, 0.348, 0.369),
  "5" = c(0.259, 0.287, 0.329, 0.384, 0.451)
)

resource_labels <- c(
  "5% under-identification",
  "10% under-identification",
  "15% under-identification",
  "20% under-identification",
  "25% under-identification"
)
resource_colors <- c("#7B1FA2", "#1976D2", "#2E7D32", "#F57C00", "#00BCD4")

dir_data <- "DCA"
df_group3 <- read.csv(file.path(dir_data, "Group3_calibrated_4country_test_predictions.csv"))
df_charls <- read.csv(file.path(dir_data, "Group3_calibrated_CHARLS_test_predictions.csv"))

df <- rbind(df_group3, df_charls)
df$prob_cog_impairment <- df$prob_CI_calibrated
df$true_cog_impairment <- ifelse(df$health_vs_mci_vs_dementia == 0, 0, 1)

pdf("Combined_5Countries_DCA.pdf", width = 7.2, height = 4.5)

layout(matrix(c(1,2,6,4,5,3), nrow = 2, byrow = TRUE),
       widths = c(1.2, 1, 1), heights = c(1, 1.15))
plot_order <- c(1, 2, 5, 3, 4)

for (i in 1:length(plot_order)) {
  code <- plot_order[i]
  code_str <- as.character(code)
  subset_df <- subset(df, country == code & scale == scale_map[[code_str]])
  
  show_ylab <- (i == 1 || i == 4)
  show_xlab <- (i >= 3)
  
  if (show_ylab && show_xlab) {
    par(mar = c(3.5, 3.5, 2.8, 0.3))
  } else if (show_ylab && !show_xlab) {
    par(mar = c(1, 3.5, 2.5, 0.3))
  } else if (!show_ylab && show_xlab) {
    par(mar = c(3.5, 0.5, 2.8, 0.3))
  } else {
    par(mar = c(1, 0.5, 2.5, 0.3))
  }
  
  dca_country <- decision_curve(
    true_cog_impairment ~ prob_cog_impairment,
    data = subset_df,
    thresholds = seq(0.01, 0.70, 0.01),
    confidence.intervals = NA
  )
  
  plot_decision_curve(dca_country,
                      curve.names = country_titles[code_str],
                      cost.benefit.axis = FALSE,
                      confidence.intervals = FALSE,
                      standardize = TRUE,
                      legend.position = "none",
                      lwd = 1.2,  
                      col = "#E15A56",
                      ylab = "",
                      xlab = "",
                      yaxt = ifelse(show_ylab, "s", "n"),
                      cex.axis = 0.7)
  
  title(main = country_titles[code_str], cex.main = 0.85, line = 0.8)
  
  if (show_ylab) {
    mtext("Standardized Net Benefit", side = 2, line = 2.2, cex = 0.6)
  }
  if (show_xlab) {
    mtext("Cognitive Impairment Risk Thresholds", side = 1, line = 2.2, cex = 0.6)
  }
  
  thresholds <- resource_thresholds[[code_str]]
  for (j in 1:length(thresholds)) {
    abline(v = thresholds[j], col = resource_colors[j], lty = 2, lwd = 1)
    idx <- which.min(abs(dca_country$dca$threshold - thresholds[j]))
    points(thresholds[j], dca_country$dca$standardized_net_benefit[idx], 
           pch = 19, col = resource_colors[j], cex = 0.8)
  }
}

plot.new()
par(mar = c(0, 0, 0, 0), xpd = NA)   

legend(x = 0.15, y =1,       
       legend = c("Proposed Cross-national Model", "Identify  All", "Identify None"),
       col = c("#E15A56", "gray", "black"),
       lty = c(1, 1, 1),
       lwd = 1.5,
       title = "Decision Strategies",
       title.adj = 0.05,
       cex = 0.7,
       title.cex = 0.8,
       y.intersp = 2,             
       bty = "n",
       adj = 0)

legend(x = 0.17, y = 0.42,          
       legend = resource_labels,
       col = resource_colors,
       lty = rep(2, 5),
       pch = rep(19, 5),
       lwd = 1.3,
       pt.cex = 0.8,
       title = "Under-identification tolerance",
       title.adj = 0,
       cex = 0.7,
       title.cex = 0.8,
       y.intersp = 2,            
       bty = "n",
       adj = 0)

dev.off()