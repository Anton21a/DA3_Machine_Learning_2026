library(haven)
library(glmnet)
library(purrr)
library(margins)
library(skimr)
library(kableExtra)
library(Hmisc)
library(cowplot)
library(gmodels) 
library(lspline)
library(sandwich)
library(modelsummary)
library(rattle)
library(caret)
library(pROC)
library(ranger)
library(rpart)
library(partykit)
library(rpart.plot)
library(readr)
library(dplyr)
library(tidyr)

rm(list=ls())

setwd("C:/Users/user/Desktop/MA2y/Data_Analysis_3/DA3_Machine_Learning_2026/Assignment_2/data")

data <- read_csv("data_cleaned.csv")
str(data)

data$fast_growth <- ifelse(data$growth_ln_sales > 1, 1, 0)
data$fast_growth_f <- factor(ifelse(data$growth_ln_sales > 1, "yes", "no"))

data$m_region_loc <- as.factor(data$m_region_loc)
data$gender_m <- as.factor(data$gender_m)
data$foreign_management <- as.factor(data$foreign_management)
data$industry_type <- as.factor(data$industry_type)


# Summarize them
#datasummary_skim(data %>% select(all_of(first_50_cat_vars)), type = "categorical")

set.seed(765)
train_indices <- as.integer(createDataPartition(data$fast_growth_f, p = 0.8, list = FALSE))
data_train <- data[train_indices, ]
data_holdout <- data[-train_indices, ]

dim(data_train)
dim(data_holdout)

#Probability Prediction
str(data)

data_list <- data %>%
  select(-sales, -sales_2012,
         -dummy_drop50_growth_ln_sales,
         -dummy_gt2x_growth_ln_sales,
         -flag_pos_growth_ln_sales,
         -growth_ln_sales,
         -fast_growth,
         -fast_growth_f)

growth_vars <- grep("^growth_ln_", names(data_list), value = TRUE)
flag_vars <- grep("^flag_pos_", names(data_list), value = TRUE)
dummy_gtx_vars <- grep("^dummy_gt2x_", names(data_list), value = TRUE)
dummy_drop50_vars <- grep("^dummy_drop50_", names(data_list), value = TRUE)

interaction <- c("industry_type*growth_ln_curr_assets", "industry_type*growth_ln_curr_liab",
                 "industry_type*growth_ln_extra_inc", "industry_type*growth_ln_inventories",
                 "industry_type*growth_ln_material_exp", "industry_type*growth_ln_personnel_exp",
                 "industry_type*growth_ln_amort", "growth_ln_extra_exp",
                 "female*growth_ln_curr_assets", "female*growth_ln_curr_liab",
                 "female*growth_ln_extra_inc", "female*growth_ln_inventories",
                 "female*growth_ln_material_exp", "female*growth_ln_personnel_exp",
                 "female*growth_ln_amort", "growth_ln_extra_exp")

X1 <- c("age", "age2", "foreign_management", "female", "m_region_loc", "industry_type")
X2 <- c("age", "age2", "foreign_management", "female", "m_region_loc", "industry_type",
        growth_vars)
X3 <- c("age", "age2", "foreign_management", "female", "m_region_loc", "industry_type",
        growth_vars, dummy_gtx_vars)
X4 <- c("age", "age2", "foreign_management", "female", "m_region_loc", "industry_type",
        growth_vars, dummy_gtx_vars, dummy_drop50_vars)
X5 <- c("age", "age2", "foreign_management", "female", "m_region_loc", "industry_type",
        growth_vars, dummy_gtx_vars, dummy_drop50_vars, interaction)


twoClassSummaryExtended <- function (data, lev = NULL, model = NULL) {
  requireNamespace("pROC")
  out <- twoClassSummary(data, lev = lev, model = model)
  out["Accuracy"] <- sum(data$pred == data$obs) / nrow(data)
  out["RMSE"] <- sqrt(mean((as.numeric(data$pred) - as.numeric(data$obs))^2))
  return(out)
}

# 5 fold cross-validation
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummaryExtended,
  savePredictions = TRUE
)

logit_model_vars <- list("X1" = X1, "X2" = X2, "X3" = X3, "X4" = X4, "X5" = X5)

CV_RMSE_folds <- list()
logit_models <- list()

for (model_name in names(logit_model_vars)) {
  
  features <- logit_model_vars[[model_name]]
  
set.seed(13505)
glm_model <- train(
  formula(paste0("fast_growth_f ~", paste0(features, collapse = " + "))),
  method = "glm",
  data = data_train,
  family = binomial,
  trControl = train_control
)
  
  logit_models[[model_name]] <- glm_model
  # Calculate RMSE on test for each fold
  CV_RMSE_folds[[model_name]] <- glm_model$resample[,c("Resample", "RMSE")]
  # Add mean RMSE output
  mean_rmse <- mean(CV_RMSE_folds[[model_name]]$RMSE)
  cat(paste0("Model ", model_name, ": Mean RMSE = ", round(mean_rmse, 4), "\n"))
  
}
glm_model$pred
logit_models[[model_name]] <- glm_model
# Calculate RMSE on test for each fold
CV_RMSE_folds[[model_name]] <- glm_model$resample[,c("Resample", "RMSE")]


# Logit lasso -----------------------------------------------------------
logitvars <- X4
lambda <- 10^seq(-1, -4, length = 10)
grid <- expand.grid("alpha" = 1, lambda = lambda)

set.seed(721)
system.time({
  logit_lasso_model <- train(
    formula(paste0("fast_growth_f ~", paste0(logitvars, collapse = " + "))),
    data = data_train,
    method = "glmnet",
    preProcess = c("center", "scale"),
    family = "binomial",
    trControl = train_control,
    tuneGrid = grid,
    na.action=na.exclude
  )
})

tuned_logit_lasso_model <- logit_lasso_model$finalModel
best_lambda <- logit_lasso_model$bestTune$lambda
logit_models[["LASSO"]] <- logit_lasso_model
lasso_coeffs <- as.matrix(coef(tuned_logit_lasso_model, best_lambda))

CV_RMSE_folds[["LASSO"]] <- logit_lasso_model$resample[,c("Resample", "RMSE")]

rmse_plot_data <- bind_rows(
  lapply(names(CV_RMSE_folds), function(model) {
    df <- CV_RMSE_folds[[model]]
    df$model <- model
    return(df)
  }),
  .id = "model_id"
)

# Plot
ggplot(rmse_plot_data, aes(x = Resample, y = RMSE, color = model, group = model)) +
  geom_point(size = 2) +
  geom_line(aes(group = model), alpha = 1) +
  labs(title = "Cross-Validated RMSE by Fold for Each Logit Model",
       x = "Fold",
       y = "RMSE",
       color = "Model") +
  theme_minimal()

#----------------------------------Random Forest--------------------------------
rf_vars <- c("age", "age2", "foreign_management", "female", "m_region_loc", "industry_type",
             growth_vars, dummy_gtx_vars, dummy_drop50_vars)
train_control_rf <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummaryExtended,
  savePredictions = "final",  
  verboseIter = TRUE
)
tune_grid_rf <- expand.grid(
  .mtry = c(5, 10, 15),
  .splitrule = "gini",
  .min.node.size = c(5, 10)
)
set.seed(5433)
rf_model <- train(
  formula(paste0("fast_growth_f ~ ", paste0(rf_vars, collapse = " + "))),
  method = "ranger",
  data = data_train,
  tuneGrid = tune_grid_rf,
  trControl = train_control_rf
)

rf_probs <- predict(rf_model, newdata = data_holdout, type = "prob")
data_holdout$rf_prob <- rf_probs[,"yes"]

logit_models[["RF"]] <- rf_model
names(logit_models[["RF"]]$pred)

# Compute AUC and RMSE
roc_obj_rf <- roc(data_holdout$fast_growth_f, data_holdout$rf_prob)
auc_rf <- auc(roc_obj_rf)

rmse_rf <- RMSE(data_holdout$rf_prob, as.numeric(data_holdout$fast_growth_f) - 1)

cat("Holdout AUC (RF):", round(auc_rf, 4), "\n")
cat("Holdout RMSE (RF):", round(rmse_rf, 4), "\n")

CV_RMSE_folds[["RF"]] <- rf_model$resample[,c("Resample", "RMSE")]
rmse_plot_data <- bind_rows(
  lapply(names(CV_RMSE_folds), function(model) {
    df <- CV_RMSE_folds[[model]]
    df$model <- model
    return(df)
  }),
  .id = "model_id"
)
ggplot(rmse_plot_data %>% filter(model %in% c("LASSO", "X4", "X5", "RF")), 
       aes(x = Resample, y = RMSE, color = model, group = model)) +
  geom_point(size = 2) +
  geom_line(aes(group = model), alpha = 1) +
  labs(title = "Cross-Validated RMSE by Fold for Selected Models",
       x = "Fold",
       y = "RMSE",
       color = "Model") +
  theme_minimal()
#--------------------------------------AUC--------------------------------------

models_to_plot <- c("X4", "X5", "LASSO", "RF")

# Initialize list for fold-wise AUCs
auc_folds_list <- list()

# Loop through each model
for (model_name in models_to_plot) {
  model <- logit_models[[model_name]]
  
  if (!is.null(model$pred)) {
    folds <- unique(model$pred$Resample)
    
    for (fold in folds) {
      fold_data <- model$pred %>% filter(Resample == fold)
      
      if ("yes" %in% colnames(fold_data)) {
        roc_obj <- tryCatch({
          roc(fold_data$obs, fold_data$yes)
        }, error = function(e) return(NULL))
        
        if (!is.null(roc_obj)) {
          auc_value <- auc(roc_obj)
          auc_folds_list[[length(auc_folds_list) + 1]] <- data.frame(
            Model = model_name,
            Fold = fold,
            AUC = as.numeric(auc_value)
          )
        }
      }
    }
  }
}

# Combine all AUCs into one data frame
auc_df <- bind_rows(auc_folds_list)

# Plot AUC distribution
ggplot(auc_df, aes(x = Model, y = AUC, fill = Model)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.6, size = 2) +
  labs(title = "Distribution of AUC Across Folds",
       x = "Model",
       y = "AUC") +
  theme_minimal() +
  theme(legend.position = "none")

#----------------------------------LOSS FUNCTION--------------------------------

# Define cost parameters
FP <- 1
FN <- 10
cost_weights <- c(FP, FN)

# Calculate prevalence
prevalence <- mean(data_train$fast_growth == 1)
normalized_weights <- cost_weights / sum(cost_weights)

# Initialize storage
best_thresholds <- list()
expected_losses <- list()
model_rocs <- list()
threshold_fold5 <- list()
loss_fold5 <- list()

models_to_check <- c("X4", "X5", "LASSO", "RF")

for (model_name in models_to_check) {
  
  model <- logit_models[[model_name]]
  folds <- unique(model$pred$Resample)
  threshold_per_fold <- c()
  loss_per_fold <- c()
  
  for (fold in folds) {
    cv_fold <- model$pred %>% filter(Resample == fold)
    
    # Use ROC to find optimal threshold based on weighted Youden index
    roc_obj <- roc(cv_fold$obs, cv_fold$yes)
    
    best_coords <- coords(
      roc_obj, 
      x = "best", 
      best.method = "youden", 
      best.weights = normalized_weights,
      ret = "all", 
      transpose = FALSE
    )
    
    # Expected loss
    fold_loss <- (best_coords$fp * FP + best_coords$fn * FN) / nrow(cv_fold)
    threshold_per_fold <- c(threshold_per_fold, best_coords$threshold)
    loss_per_fold <- c(loss_per_fold, fold_loss)
  }
  
  best_thresholds[[model_name]] <- mean(threshold_per_fold)
  expected_losses[[model_name]] <- mean(loss_per_fold)
  
  # Save ROC & threshold from Fold5 for plotting
  model_rocs[[model_name]] <- roc_obj
  threshold_fold5[[model_name]] <- best_coords
  loss_fold5[[model_name]] <- fold_loss
  
  cat(paste0("Model ", model_name, 
             ": Avg Expected Loss = ", round(mean(loss_per_fold), 3), 
             ", Avg Threshold = ", round(mean(threshold_per_fold), 4), "\n"))
}

loss_summary <- data.frame(
  Model = names(expected_losses),
  Avg_Expected_Loss = unlist(expected_losses),
  Avg_Optimal_Threshold = unlist(best_thresholds)
)
loss_summary


ggplot(loss_summary, aes(x = reorder(Model, Avg_Expected_Loss), y = Avg_Expected_Loss, fill = Model)) +
  geom_col(width = 0.6, color = "black") +
  geom_text(aes(label = round(Avg_Expected_Loss, 3)), vjust = -0.5) +
  labs(title = "Average Expected Loss by Model (FP = 1, FN = 10)",
       x = "Model", y = "Expected Loss") +
  theme_minimal() +
  theme(legend.position = "none")


ggplot(loss_summary, aes(x = Model, y = Avg_Optimal_Threshold, fill = Model)) +
  geom_col(width = 0.6, color = "black") +
  geom_text(aes(label = round(Avg_Optimal_Threshold, 3)), vjust = -0.5) +
  labs(title = "Average Optimal Threshold by Model",
       x = "Model", y = "Threshold") +
  theme_minimal() +
  theme(legend.position = "none")



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------





















