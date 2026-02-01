library(lme4)
library(tidyr)
library(dplyr)
library(performance)

df <- read.csv("stacked_five_country.csv")

# Convert to long format (individual represents id)
df_long <- df %>%
  pivot_longer(cols = c(asinmiss, extreme, gnorm, logresvar, lz, mdistance, u3),
               names_to = "LQR_index",
               values_to = "LQR_value")

# Fit three-level mixed-effects model (country-individual-indicator)
model <- lmer(LQR_value ~ 1 + (1 | country) + (1 | country:id),
              data = df_long, REML = TRUE)

summary(model)

var_comp <- as.data.frame(VarCorr(model))
total_var <- sum(var_comp$vcov)
var_comp$proportion <- var_comp$vcov / total_var

print("=== Variance Component Analysis ===")
print(var_comp)

print("\n=== ICC calculation results ===")
# Extract variance at each level
country_var <- var_comp$vcov[var_comp$grp == "country"]
individual_var <- var_comp$vcov[var_comp$grp == "country:id"]
residual_var <- var_comp$vcov[var_comp$grp == "Residual"]

# Calculate ICC at different levels
ICC_country <- country_var / total_var
ICC_individual <- individual_var / total_var
ICC_residual <- residual_var / total_var

cat("ICC_country (country-level):", round(ICC_country, 4), "\n")
cat("ICC_individual (within-country individual-level):", round(ICC_individual, 4), "\n")
cat("ICC_residual (within-individual indicator-level):", round(ICC_residual, 4), "\n")

# Cross-country stacking applicability assessment
print("\n=== Cross-country Stacking Applicability Assessment ===")
if(ICC_country < 0.05) {
  cat("ICC_country =", round(ICC_country, 4), "< 0.05: Low country-level variance\n")
} else if(ICC_country < 0.10) {
  cat("ICC_country =", round(ICC_country, 4), "0.05-0.10: Moderate country-level variance\n")
} else {
  cat("ICC_country =", round(ICC_country, 4), "> 0.10: Higher country-level variance\n")
}

# Validate ICC calculation using performance package
print("\n=== Performance Package ICC Validation ===")
icc_result <- icc(model)
print(icc_result)

# Package information
cat("\nPackages used:\n")
cat("lme4 version:", as.character(packageVersion("lme4")), "\n")
cat("tidyr version:", as.character(packageVersion("tidyr")), "\n")
cat("dplyr version:", as.character(packageVersion("dplyr")), "\n")
cat("performance version:", as.character(packageVersion("performance")), "\n")