library(ggplot2)
library(dplyr)
library(broom)
library(ggpubr)
library(corrplot)
library(MASS)
library(DescTools)
library(caret)
library(pROC)
library(ResourceSelection)
library(glmnet)

dataset <- read.csv("diabetes_prediction_dataset.csv")

## Data Cleaning
dataset$age <- round(dataset$age)
dataset <- subset(dataset, smoking_history != "No Info")

# Create a factor with custom levels and sort
ordinal_scale <- c("current", "ever", "former", "never", "not current")
numeric_values <- c(1, 2, 3, 4, 5)
dataset$smoking_history <- as.integer(factor(dataset$smoking_history, levels = ordinal_scale))
dataset$smoking_history <- numeric_values[dataset$smoking_history]

## Logistic Regression
pdf("logistic_reg.pdf")

v1 <- c("diabetes", "blood_glucose_level", "hypertension", "heart_disease", 
        "smoking_history", "bmi", "HbA1c_level")
## Check data distribution (dependent variable)
for (variable in v1) {
  print(hist(dataset[[variable]], main = variable, xlab = variable))
}
v2 <- c("blood_glucose_level", "hypertension", "heart_disease", 
        "smoking_history", "bmi", "HbA1c_level")
## Check distribution and data points
for (variable in v2) {
  print(plot(diabetes ~ dataset[[variable]], main = variable, xlab = variable, data = dataset))
}

# Perform Logistic Regression
Diabetes_LogisticR <- glm(diabetes ~ blood_glucose_level + hypertension + heart_disease + smoking_history + bmi + HbA1c_level,
                          data = dataset, family = binomial)

# Summary statistics and diagnostics
summary_table <- tidy(Diabetes_LogisticR)
print(summary_table)
# Analysis of deviance
anova(Diabetes_LogisticR, test = "Chisq")


# Hosmer-Lemeshow test
hoslem.test(dataset$diabetes, fitted(Diabetes_LogisticR))

# Calculate correlation matrix
cor_matrix <- cor(dataset[v1])

# Visualize correlation matrix
corrplot(cor_matrix, type = "upper", method = "circle")

# Split dataset into training and testing sets
set.seed(123)
train_index <- createDataPartition(dataset$diabetes, p = 0.7, list = FALSE)
train_data <- dataset[train_index, ]
test_data <- dataset[-train_index, ]

# Fit logistic regression model on training data
Diabetes_LogisticR_Train <- glm(diabetes ~ blood_glucose_level 
                                + hypertension + heart_disease 
                                + smoking_history + bmi + HbA1c_level,
                          data = train_data, family = binomial)
summary(Diabetes_LogisticR_Train)

# Predict on testing data
predictions <- predict(Diabetes_LogisticR_Train, newdata = test_data, type = "response")

# Calculate accuracy, sensitivity, specificity, and AUC-ROC
roc_obj <- roc(test_data$diabetes, predictions)
accuracy <- mean((predictions > 0.5) == test_data$diabetes)
sensitivity <- roc_obj$sensitivities[which.min(abs(roc_obj$specificities - 0.5))]
specificity <- roc_obj$specificities[which.min(abs(roc_obj$sensitivities - 0.5))]
auc_roc <- auc(roc_obj)

# Print performance metrics
cat("Accuracy:", accuracy, "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("Specificity:", specificity, "\n")
cat("AUC-ROC:", auc_roc, "\n")

plot(roc_obj, print.auc = TRUE,
     main = "AUROC Curve",
     xlab = "False Positive Rate",
     ylab = "True Positive Rate")

# Odds ratios and confidence intervals
odds_ratios <- exp(coef(Diabetes_LogisticR_Train))
conf_intervals <- confint(Diabetes_LogisticR_Train)

# Print odds ratios and confidence intervals
results <- data.frame(Odds_Ratio = odds_ratios, Conf_Interval = conf_intervals)
print(results)

results$Variable <- rownames(results)
plot_data <- data.frame(
  Variable = results$Variable,
  Odds_Ratio = results$Odds_Ratio,
  Lower_CI = conf_intervals[, 1],
  Upper_CI = conf_intervals[, 2]
)

# Sort the data frame by Odds Ratio
plot_data <- plot_data[order(plot_data$Odds_Ratio), ]

# Create the forest plot
ggplot(plot_data, aes(x = Odds_Ratio, y = Variable)) +
  geom_point(size = 3, color = "blue") +
  geom_errorbarh(aes(xmin = Lower_CI, xmax = Upper_CI), height = 0.2,
                 color = "blue") +
  xlim(c(min(conf_intervals), max(conf_intervals))) +
  labs(x = "Odds Ratio", y = "Variable") +
  ggtitle("Odds Ratios and Confidence Intervals") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


# Fit alternative models
Alt_Model <- glm(diabetes ~ blood_glucose_level + bmi, data = dataset, family = binomial)
Alt_Model2 <- glm(diabetes ~ blood_glucose_level + smoking_history + bmi, data = dataset, family = binomial)

# Compare models using AIC
models <- list(Diabetes_LogisticR, Alt_Model, Alt_Model2)
AIC_values <- sapply(models, AIC)
print(AIC_values)
dev.off()


