library(caret)  
library(estimatr) 
library(Metrics)  
library(skimr)
library(glmnet)
library(dplyr)
rm(list=ls())

setwd("C:/Users/user/Desktop/MA2y/Data_Analysis_3/DA3_Machine_Learning_2026/Assignment_1/data")

data2025q2 <- read.csv("Budapest2025q2_final.csv")
data2025q1 <- read.csv("Budapest2025q1_final.csv")

#_____________________________________Budapest__________________________________
#_____________________________________2025q1______________________________________
#_____________________________________OLS_______________________________________

skim(data2025q1$ln_price)

set.seed(11)
folds <- createFolds(data2025q1$ln_price, k = 5, list = TRUE) 
folds

rmse_matrix <- matrix(NA, nrow = 5, ncol = 3) 
colnames(rmse_matrix) <- c("Model1", "Model2", "Model3")
rownames(rmse_matrix) <- paste("Fold", 1:5)

as.factor(data2025q1$property_type)
str(data2025q1$accommodates)
str(data2025q1$number_of_reviews_n)
str(data2025q1$f_bathroom)
str(data2025q1$f_minimum_nights)
as.factor(data2025q1$refrigerator)
as.factor(data2025q1$wifi)
as.factor(data2025q1$microwave)
as.factor(data2025q1$f_minimum_nights)
as.factor(data2025q1$f_bathroom)

for (i in 1:5) { 
  test_indices <- folds[[i]]
  train_data2025q1 <- data2025q1[-test_indices, ]  # Train set: all except current fold
  test_data2025q1 <- data2025q1[test_indices, ]    # Test set: current fold
  
  formulas <- list(
    reg1 = ln_price ~ property_type,
    reg2 = ln_price ~ property_type + accommodates + number_of_reviews_n +
      f_bathroom + f_minimum_nights,
    reg3 = ln_price ~ property_type + accommodates + number_of_reviews_n +
      f_bathroom + f_minimum_nights + refrigerator + microwave + wifi
    )
  predictions_list <- list()
  
  for (j in 1:3) {
    # Train model on training data
    model <- lm_robust(formulas[[j]], data = train_data2025q1)
    
    # Make predictions on test data
    preds <- predict(model, newdata = test_data2025q1)
    
    # Compute RMSE for this fold and model
    rmse_matrix[i, j] <- RMSE(preds, test_data2025q1$ln_price)
    
    predictions_list[[paste0("Fold_", i, "_Model_", j)]] <- data.frame(
      Fold = i,
      Model = paste0("Model_", j),
      Actual = test_data2025q1$ln_price,
      Predicted = preds
    )
  }
}
rmse_matrix <- as.data.frame(rmse_matrix)
rmse_matrix$`Model 1`
mean_rmse <- data.frame(Model1 = mean(rmse_matrix[, 1]),
                        Model2 = mean(rmse_matrix[, 2]),
                        Model3 = mean(rmse_matrix[, 3]))

rmse_matrix <- rbind(rmse_matrix, mean_rmse)

rownames(rmse_matrix)[6] <- 'mean value'
rmse_matrix


# Convert RMSE matrix to data frame
rmse_df <- as.data.frame(rmse_matrix)
rmse_df$Fold <- rownames(rmse_matrix)

library(data.table)

rmse_long2025q1 <- melt(
  as.data.table(rmse_df),
  id.vars = "Fold",
  variable.name = "Model",
  value.name = "RMSE"
)

ggplot(rmse_long2025q1, aes(x = Fold, y = RMSE, group = Model, color = Model)) +
  geom_line(size = 1) + 
  geom_point(size = 3) +
  labs(title = "RMSE Across 5 Folds for Each Model", 
       x = "Fold", 
       y = "RMSE") +
  theme_minimal()


predictions_df <- do.call(rbind, predictions_list)
predictions_df$Residuals <- predictions_df$Actual - predictions_df$Predicted

ggplot(predictions_df, aes(x = Residuals, fill = Model)) +
  geom_density(alpha = 0.5) +
  labs(title = "Residual Distribution", 
       x = "Residuals (Actual - Predicted)", 
       y = "Density") +
  theme_minimal()


ggplot(rmse_long2025q1, aes(x = Model, y = RMSE, fill = Model)) +
  geom_boxplot() +
  labs(title = "mean RMSE Distribution Across Models after 5 fold cross-validation", 
       x = "Model", 
       y = "RMSE") +
  theme_minimal()

#_____________________________________Budapest__________________________________
#_____________________________________2025q1____________________________________
#_____________________________________LASSO_____________________________________

lasso_formula <- as.formula(
  ln_price ~ accommodates + number_of_reviews_n +
    f_bathroom + f_minimum_nights + property_type + f_room_type +
    refrigerator + microwave + wifi +
    accommodates*property_type +
    f_room_type*property_type +
    property_type*refrigerator +
    property_type*microwave +
    property_type*wifi
)

set.seed(111)
library(tidyr)

vars_needed <- all.vars(lasso_formula)

data_lasso <- na.omit(data2025q1[, c("ln_price", vars_needed)])

folds_lasso <- createFolds(data_lasso$ln_price, k = 5, list = TRUE)
lasso_rmse <- numeric(5)

for (i in 1:5) {
  test_idx <- folds_lasso[[i]]
  train_idx <- setdiff(seq_len(nrow(data_lasso)), test_idx)
  
  train <- data_lasso[train_idx, ]
  test <- data_lasso[test_idx, ]
  
  # Create model matrix with interactions
  X_train <- model.matrix(lasso_formula, data = train)[, -1]
  y_train <- train$ln_price
  X_test <- model.matrix(lasso_formula, data = test)[, -1]
  y_test <- test$ln_price
  
  # CV LASSO on train
  lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 5)
  best_lambda <- lasso_cv$lambda.min
  
  # Predict & store RMSE
  preds <- predict(lasso_cv, newx = X_test, s = best_lambda)
  lasso_rmse[i] <- RMSE(preds, y_test)
}

lasso_df <- data.frame(
  Fold = paste("Fold", 1:5),
  RMSE = round(lasso_rmse, 4)
)
lasso_df <- rbind(lasso_df, c("mean value", round(mean(lasso_rmse), 4)))
lasso_df


ggplot(lasso_df[1:5, ], aes(x = Fold, y = as.numeric(RMSE))) +
  geom_col(fill = "steelblue") +
  geom_hline(yintercept = mean(lasso_rmse), linetype = "dashed", color = "red") +
  labs(title = "LASSO RMSE Across 5 Folds", y = "RMSE", x = "Fold") +
  theme_minimal()

# Fit full LASSO model on all data to get final coefficients
X_full <- model.matrix(lasso_formula, data = data_lasso)[, -1]
y_full <- data_lasso$ln_price

lasso_final <- glmnet(X_full, y_full, alpha = 1, lambda = best_lambda)
non_zero_coefs <- coef(lasso_final)
non_zero_coefs <- non_zero_coefs[non_zero_coefs[, 1] != 0, , drop = FALSE]
print(non_zero_coefs)

#------------------------------

lasso_df <- data.frame(
  Fold = paste("Fold", 1:5),
  Model = "LASSO (with interactions)",
  RMSE = round(lasso_rmse, 4)
)

rmse_all <- rbind(rmse_long2025q1, lasso_df)

ggplot(rmse_all, aes(x = Fold, y = as.numeric(RMSE), group = Model, color = Model)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  labs(title = "RMSE Across 5 Folds (Including Mean)", y = "RMSE", x = "Fold") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


