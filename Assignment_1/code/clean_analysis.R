library(caret)
library(estimatr)
library(Metrics)
library(skimr)
library(glmnet)
library(dplyr)
library(tidyr)
library(data.table)
library(ggplot2)

rm(list = ls())

setwd("C:/Users/user/Desktop/MA2y/Data_Analysis_3/DA3_Machine_Learning_2026/Assignment_1/data")

data2025q2 <- read.csv("Budapest2025q2_final.csv")
data2025q1 <- read.csv("Budapest2025q1_final.csv")

skim(data2025q1$ln_price)
skim(data2025q2$ln_price)

data2025q1$f_room_type
data2025q2$f_room_type

#--------------------------------------Budapest 2025q1--------------------------
#-----------------------------------OLS FUNCTION--------------------------------

ols_cv_rmse <- function(data, seed_val = 111) {
  set.seed(seed_val)
  folds <- createFolds(data$ln_price, k = 5)
  
  rmse_matrix <- matrix(NA, nrow = 5, ncol = 3)
  colnames(rmse_matrix) <- c("Model1", "Model2", "Model3")
  rownames(rmse_matrix) <- paste("Fold", 1:5)
  
  formulas <- list(
    ln_price ~ property_type,
    ln_price ~ property_type + f_room_type + ln_accommodates + ln_accommodates2 +
      ln_number_of_reviews + f_bathroom + f_minimum_nights,
    ln_price ~ property_type + f_room_type + ln_accommodates + ln_accommodates2 + ln_number_of_reviews +
      f_bathroom + beds_n + f_minimum_nights + refrigerator + microwave + wifi + smoke.alarm +
      hot.water + essentials + dining.table + bed.linens 
  )
  
  predictions_list <- list()
  
  for (i in 1:5) {
    train <- data[-folds[[i]], ]
    test <- data[folds[[i]], ]
    
    for (j in 1:3) {
      model <- lm_robust(formulas[[j]], data = train)
      preds <- predict(model, newdata = test)
      rmse_matrix[i, j] <- RMSE(preds, test$ln_price)
      
      predictions_list[[paste0("Fold_", i, "_Model_", j)]] <- data.frame(
        Fold = i,
        Model = paste0("Model_", j),
        Actual = test$ln_price,
        Predicted = preds
      )
    }
  }
  
  rmse_df <- as.data.frame(rmse_matrix)
  mean_row <- data.frame(t(colMeans(rmse_df)))
  rownames(mean_row) <- "mean value"
  rmse_df <- rbind(rmse_df, mean_row)
  rmse_df$Fold <- rownames(rmse_df)
  
  long_df <- data.table::melt(
    data.table::as.data.table(rmse_df),
    id.vars = "Fold",
    variable.name = "Model",
    value.name = "RMSE"
  )
  predictions_df <- do.call(rbind, predictions_list)
  predictions_df$Residuals <- predictions_df$Actual - predictions_df$Predicted
  
  list(rmse = rmse_df, long = long_df, residuals = predictions_df)
}

eval2025q1 <- ols_cv_rmse(data2025q1, seed_val = 113)

eval2025q1$rmse
eval2025q1$long
eval2025q1$residuals

ggplot(eval2025q1$long, aes(x = Fold, y = RMSE, group = Model, color = Model)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(title = "RMSE Across 5 Folds for Each Model (2024)", x = "Fold", y = "RMSE") +
  theme_minimal()

ggplot(eval2025q1$long, aes(x = Model, y = RMSE, fill = Model)) +
  geom_boxplot() +
  labs(title = "RMSE Distribution Across Models (2024)", x = "Model", y = "RMSE") +
  theme_minimal()

#------------------------------- LASSO FUNCTION --------------------------------
lasso_cv_rmse <- function(data, formula, seed_val = 5554) {
  set.seed(seed_val)
  vars_needed <- all.vars(formula)
  data_lasso <- na.omit(data[, c("ln_price", vars_needed)])
  folds <- createFolds(data_lasso$ln_price, k = 5)
  
  rmse_list <- numeric(5)
  
  for (i in 1:5) {
    train <- data_lasso[-folds[[i]], ]
    test <- data_lasso[folds[[i]], ]
    
    X_train <- model.matrix(formula, data = train)[, -1]
    y_train <- train$ln_price
    X_test <- model.matrix(formula, data = test)[, -1]
    y_test <- test$ln_price
    
    lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 5)
    best_lambda <- lasso_cv$lambda.min
    
    preds <- predict(lasso_cv, newx = X_test, s = best_lambda)
    rmse_list[i] <- RMSE(preds, y_test)
  }
  
  mean_rmse <- mean(rmse_list)
  
  data.frame(
    Fold = c(paste("Fold", 1:5), "mean value"),
    Model = "LASSO (with interactions)",
    RMSE = round(c(rmse_list, mean_rmse), 4)
  )
}

lasso_formula <- ln_price ~ property_type + f_room_type + ln_accommodates + ln_accommodates2 + ln_number_of_reviews +
  f_bathroom + beds_n + f_minimum_nights + refrigerator + microwave + wifi + smoke.alarm +
  hot.water + essentials + dining.table + bed.linens +
  property_type*ln_accommodates + property_type*f_room_type + property_type*f_minimum_nights +
  property_type*refrigerator + property_type*microwave + property_type*wifi +
  property_type*smoke.alarm + property_type*hot.water + 
  property_type*dining.table + property_type*beds_n + property_type*f_bathroom
  

lasso2025q1_df <- lasso_cv_rmse(data2025q1, lasso_formula, seed_val = 1117)

# Combine and plot
rmse_all_2025q1 <- rbind(eval2025q1$long, lasso2025q1_df)

# Plot combined RMSEs
ggplot(rmse_all_2025q1, aes(x = Fold, y = as.numeric(RMSE), group = Model, color = Model)) +
  geom_line(size = 1) + geom_point(size = 2) +
  labs(title = "RMSE Across 5 Folds (2024)", y = "RMSE", x = "Fold") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(rmse_all_2025q1, aes(x = Model, y = RMSE, fill = Model)) +
  geom_boxplot() +
  labs(title = "RMSE Distribution Across Models (2024)", x = "Model", y = "RMSE") +
  theme_minimal()

#------------------------------- RANDOM FOREST ---------------------------------

rf_formula <- ln_price ~ property_type + f_room_type + ln_accommodates + ln_accommodates2 + ln_number_of_reviews +
  f_bathroom + beds_n + f_minimum_nights + refrigerator + microwave + wifi + smoke.alarm +
   hot.water + essentials + dining.table + bed.linens

run_rf_cv <- function(data, year_label) {
  set.seed(2223)
  vars_rf <- all.vars(rf_formula)
  data_rf <- na.omit(data[, c("ln_price", vars_rf)])
  
  train_control <- trainControl(method = "cv", number = 5, savePredictions = "all",
                                returnResamp = "all")
  tune_grid <- expand.grid(
    mtry = c(3, 5, 7, 9),
    splitrule = "variance",
    min.node.size = 5
  )
  
  rf_model_cv <- train(
    rf_formula,
    data = data_rf,
    method = "ranger",
    trControl = train_control,
    tuneGrid = tune_grid,
    num.trees = 500,
    importance = "impurity"
  )
  
  print(rf_model_cv)
  cat("\nBest mtry (", year_label, "):", rf_model_cv$bestTune$mtry,
      "| RMSE:", min(rf_model_cv$results$RMSE), "\n")
  
  vip <- varImp(rf_model_cv)$importance %>%
    as.data.frame() %>%
    tibble::rownames_to_column(var = "Variable") %>%
    arrange(desc(Overall)) %>%
    head(10)
  
  return(list(model = rf_model_cv, vip = vip))
}

rf_2025q1 <- run_rf_cv(data2025q1, "2025q1")


ggplot(rf_2025q1$vip, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_col(fill = "forestgreen") +
  coord_flip() +
  labs(title = paste("Top 10 Variable Importance - 2025q1",
                     "(mtry =", rf_2025q1$model$bestTune$mtry, ")"),
       x = "Variable", y = "Importance") +
  theme_minimal()


rf_rmse_folds <- rf_2025q1$model$resample %>%
  filter(mtry == rf_2025q1$model$bestTune$mtry) %>%
  mutate(
    Fold = paste("Fold", row_number()),
    Model = "Random Forest",
    RMSE = RMSE
  ) %>%
  select(Fold, Model, RMSE)

#---------------------------------- XGB BOOSTING -------------------------------
library(xgboost)

boost_formula <- ln_price ~ property_type + f_room_type + ln_accommodates + ln_accommodates2 + ln_number_of_reviews +
  f_bathroom + beds_n + f_minimum_nights + refrigerator + microwave + wifi + smoke.alarm +
  hot.water + essentials + dining.table + bed.linens

vars <- all.vars(boost_formula)
data_boost <- na.omit(data2025q1[, c("ln_price", vars)])

X <- model.matrix(boost_formula, data = data_boost)[, -1]
y <- data_boost$ln_price

dtrain <- xgb.DMatrix(data = X, label = y)

params <- list(
  objective = "reg:squarederror",
  max_depth = 6,
  eta = 0.05,
  subsample = 0.6,
  colsample_bytree = 0.6,
  min_child_weight = 5,
  gamma = 0
)

set.seed(223)
xgb_cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 300,
  nfold = 5,
  metrics = "rmse",
  early_stopping_rounds = 20,
  verbose = 0
)

best_rmse <- min(xgb_cv$evaluation_log$test_rmse_mean)
best_rmse


best_nrounds <- which.min(xgb_cv$evaluation_log$test_rmse_mean)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 0
)


xgb_imp <- xgb.importance(
  feature_names = colnames(X),
  model = xgb_model
)

head(xgb_imp, 10)

xgb.plot.importance(
  xgb_imp[1:10, ],
  rel_to_first = TRUE,
  xlab = "Relative Importance",
  main = "Top 10 Features – XGBoost 2025Q1",
  col = "forestgreen"
)


#--------------------------------- GBM BOOSTING --------------------------------
library(gbm)

run_gbm_cv <- function(data, year_label) {
  set.seed(2025)
  
  gbm_formula <- ln_price ~ property_type + f_room_type + ln_accommodates + ln_accommodates2 +
    ln_number_of_reviews + f_bathroom + beds_n + f_minimum_nights +
    refrigerator + microwave + wifi + smoke.alarm + hot.water +
    essentials + dining.table + bed.linens
  
  vars <- all.vars(gbm_formula)
  data_gbm <- na.omit(data[, c("ln_price", vars)])
  
  train_control <- trainControl(
    method = "cv",
    number = 5,
    savePredictions = "all",
    returnResamp = "all"
  )
  
  gbm_grid <- expand.grid(
    interaction.depth = c(3, 5),
    n.trees = c(100, 200),
    shrinkage = c(0.05, 0.1),
    n.minobsinnode = 10
  )
  
  gbm_model_cv <- train(
    gbm_formula,
    data = data_gbm,
    method = "gbm",
    trControl = train_control,
    tuneGrid = gbm_grid,
    verbose = FALSE
  )
  
  return(gbm_model_cv)
}

gbm_2025q1 <- run_gbm_cv(data2025q1, "2025q1")
#gbm_2025 <- run_gbm_cv(data2025, "2025")

gbm_importance <- varImp(gbm_2025q1)$importance %>%
  as.data.frame() %>%
  tibble::rownames_to_column(var = "Variable") %>%
  arrange(desc(Overall))

ggplot(gbm_importance[1:10, ], aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top 10 Variable Importance - GBM (2025q1)",
    x = "Variable",
    y = "Importance"
  ) +
  theme_minimal()

#-----------------HORSE RACE: Comparing OLS, LASSO, RF, and XGBoost 2025q1--------

# Extract RMSEs for Random Forest
best_params <- rf_2025q1$model$bestTune
rf_rmse_folds <- rf_2025q1$model$resample %>%
  filter(
    mtry == best_params$mtry,
    splitrule == best_params$splitrule,
    min.node.size == best_params$min.node.size
  ) %>%
  mutate(
    Fold = paste("Fold", row_number()),
    Model = "Random Forest",
    RMSE = RMSE
  ) %>%
  select(Fold, Model, RMSE)


# Extract RMSEs for XGBoost
set.seed(223)
folds <- createFolds(y, k = 5)

xgb_rmse_folds <- data.frame(
  Fold = character(),
  Model = character(),
  RMSE = numeric()
)

for (i in seq_along(folds)) {
  
  test_idx  <- folds[[i]]
  train_idx <- setdiff(seq_along(y), test_idx)
  
  dtrain_i <- xgb.DMatrix(X[train_idx, ], label = y[train_idx])
  dtest_i  <- xgb.DMatrix(X[test_idx, ],  label = y[test_idx])
  
  model_i <- xgb.train(
    params = params,
    data = dtrain_i,
    nrounds = best_nrounds,
    verbose = 0
  )
  
  preds_i <- predict(model_i, dtest_i)
  
  rmse_i <- RMSE(preds_i, y[test_idx])
  
  xgb_rmse_folds <- rbind(
    xgb_rmse_folds,
    data.frame(
      Fold = paste0("Fold", i),
      Model = "XGBoost",
      RMSE = rmse_i
    )
  )
}
xgb_rmse_folds

# Extract RMSEs for GBMboost
best_gbm <- gbm_2025q1$bestTune

gbm_preds <- gbm_2025q1$pred %>%
  filter(
    interaction.depth == best_gbm$interaction.depth,
    n.trees == best_gbm$n.trees,
    shrinkage == best_gbm$shrinkage,
    n.minobsinnode == best_gbm$n.minobsinnode
  )

gbm_rmse_folds <- gbm_preds %>%
  group_by(Resample) %>%
  summarise(RMSE = RMSE(obs, pred), .groups = "drop") %>%
  mutate(
    Fold = gsub("Fold", "Fold ", Resample),
    Model = "GBM"
  ) %>%
  select(Fold, Model, RMSE)

# Extract RMSEs for OLS (Model 3 only)
ols_rmse_folds <- eval2025q1$long %>%
  filter(Model == "Model3" & Fold != "mean value") %>%
  mutate(Model = "OLS")

# Extract RMSEs for LASSO
lasso_rmse_folds <- lasso2025q1_df %>%
  filter(Fold != "mean value") %>%
  rename(RMSE = RMSE) %>%
  select(Fold, Model, RMSE)

# Combine all models
xgb_rmse_folds <- xgb_rmse_folds %>%
  mutate(Fold = gsub("Fold", "Fold ", Fold))

horserace_rmse <- bind_rows(ols_rmse_folds, lasso_rmse_folds, rf_rmse_folds,
                            xgb_rmse_folds, gbm_rmse_folds)


# Plot: RMSE per fold
ggplot(horserace_rmse, aes(x = Fold, y = as.numeric(RMSE), group = Model, color = Model)) +
  geom_line(size = 1) + geom_point(size = 2) +
  labs(title = "Horserace: Model Comparison (Budapest 2025q1)", y = "RMSE", x = "Fold") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot: RMSE distribution across models
ggplot(horserace_rmse, aes(x = Model, y = as.numeric(RMSE), fill = Model)) +
  geom_boxplot() +
  labs(title = "RMSE Distribution by Model (Budapest 2025q1)", y = "RMSE", x = "Model") +
  theme_minimal()

#------------------------test data = Budapest 2025q2----------------------------
#-----------------------------------OLS FUNCTION--------------------------------

ols_time_validation <- function(train_data, test_data) {
  
  formulas <- list(
    ln_price ~ property_type,
    ln_price ~ property_type + f_room_type + ln_accommodates + ln_accommodates2 +
      ln_number_of_reviews + f_bathroom + f_minimum_nights,
    ln_price ~ property_type + f_room_type + ln_accommodates + ln_accommodates2 + ln_number_of_reviews +
      f_bathroom + beds_n + f_minimum_nights + refrigerator + microwave + wifi + smoke.alarm +
      hot.water + essentials + bed.linens
  )
  
  rmse_vec <- numeric(3)
  predictions_list <- list()
  
  # --- align factor levels (CRITICAL for live data)
  factor_vars <- c(
    "property_type","f_room_type","f_bathroom","f_minimum_nights",
    "refrigerator","microwave","wifi","smoke.alarm","hot.water",
    "essentials","dining.table","bed.linens"
  )
  
  for (v in factor_vars) {
    if (v %in% names(train_data) &&
        v %in% names(test_data) &&
        is.factor(train_data[[v]])) {
      
      test_data[[v]] <- factor(
        test_data[[v]],
        levels = levels(train_data[[v]])
      )
    }
  }
  
  for (j in 1:3) {
    
    model <- lm_robust(formulas[[j]], data = train_data)
    preds <- predict(model, newdata = test_data)
    
    rmse_vec[j] <- RMSE(preds, test_data$ln_price)
    
    predictions_list[[paste0("Model_", j)]] <- data.frame(
      Fold = "Live (2025Q2)",
      Model = paste0("Model_", j),
      Actual = test_data$ln_price,
      Predicted = preds,
      Residuals = test_data$ln_price - preds
    )
  }
  
  rmse_df <- data.frame(
    Fold = "Live (2025Q2)",
    Model1 = rmse_vec[1],
    Model2 = rmse_vec[2],
    Model3 = rmse_vec[3]
  )
  
  long_df <- data.table::melt(
    data.table::as.data.table(rmse_df),
    id.vars = "Fold",
    variable.name = "Model",
    value.name = "RMSE"
  )
  
  predictions_df <- do.call(rbind, predictions_list)
  
  list(
    rmse = rmse_df,
    long = long_df,
    residuals = predictions_df
  )
}

eval_time_2025 <- ols_time_validation(
  train_data = data2025q1,
  test_data  = data2025q2
)

ggplot(eval_time_2025$long, aes(x = Model, y = RMSE, fill = Model)) +
  geom_col() +
  labs(
    title = "Live RMSE by Model (Train: 2025Q1 → Test: 2025Q2)",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal()

ggplot(eval_time_2025$residuals, aes(x = Residuals, fill = Model)) +
  geom_density(alpha = 0.5) +
  theme_minimal() +
  labs(
    title = "Residual Distribution (Train: 2025Q1 → Test: 2025Q2)",
    x = "Residuals",
    y = "Density"
  )


#------------------------test data = Budapest 2025q2----------------------------
#------------------------------- LASSO FUNCTION --------------------------------

lasso_formula <- ln_price ~ property_type + f_room_type + ln_accommodates + ln_accommodates2 + ln_number_of_reviews +
  f_bathroom + beds_n + f_minimum_nights + refrigerator + microwave + wifi + smoke.alarm +
  hot.water + essentials + bed.linens +
  property_type*ln_accommodates + property_type*f_room_type + property_type*f_minimum_nights +
  property_type*refrigerator + property_type*microwave + property_type*wifi +
  property_type*smoke.alarm + property_type*hot.water + 
  property_type*beds_n + property_type*f_bathroom

lasso_time_validation <- function(train_data, test_data, formula, seed_val = 5554) {
  
  set.seed(seed_val)
  
  vars_needed <- all.vars(formula)
  train_lasso <- na.omit(train_data[, c("ln_price", vars_needed)])
  test_lasso  <- na.omit(test_data[,  c("ln_price", vars_needed)])
  
  # --- build model matrices (SAME formula!)
  X_train <- model.matrix(formula, data = train_lasso)[, -1]
  y_train <- train_lasso$ln_price
  
  X_test <- model.matrix(formula, data = test_lasso)[, -1]
  y_test <- test_lasso$ln_price
  
  # --- align columns (CRITICAL for live data)
  common_cols <- intersect(colnames(X_train), colnames(X_test))
  
  X_train <- X_train[, common_cols]
  X_test  <- X_test[,  common_cols]
  
  # --- LASSO with internal CV for lambda
  lasso_cv <- cv.glmnet(
    X_train,
    y_train,
    alpha = 1,
    nfolds = 5
  )
  
  best_lambda <- lasso_cv$lambda.min
  
  preds <- as.numeric(predict(
    lasso_cv,
    newx = X_test,
    s = best_lambda
  ))
  
  rmse_val <- RMSE(preds, y_test)
  
  rmse_df <- data.frame(
    Fold = "Live (2025Q2)",
    Model = "LASSO (with interactions)",
    RMSE = round(rmse_val, 4)
  )
  
  predictions_df <- data.frame(
    Fold = "Live (2025Q2)",
    Model = "LASSO (with interactions)",
    Actual = y_test,
    Predicted = preds,
    Residuals = y_test - preds
  )
  
  list(
    rmse = rmse_df,
    residuals = predictions_df,
    lambda = best_lambda
  )
}


lasso_time_2025 <- lasso_time_validation(
  train_data = data2025q1,
  test_data  = data2025q2,
  formula    = lasso_formula,
  seed_val   = 1117
)

rmse_all_time_2025 <- rbind(
  eval_time_2025$long,
  lasso_time_2025$rmse
)

ggplot(rmse_all_time_2025, aes(x = Model, y = RMSE, fill = Model)) +
  geom_col() +
  labs(
    title = "Live RMSE Comparison (Train: 2025Q1 → Test: 2025Q2)",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal()

#------------------------test data = Budapest 2025q2----------------------------
#------------------------------- RANDOM FOREST ---------------------------------

rf_formula <- ln_price ~ property_type + f_room_type + ln_accommodates + ln_accommodates2 + ln_number_of_reviews +
  f_bathroom + beds_n + f_minimum_nights + refrigerator + microwave + wifi + smoke.alarm +
  hot.water + essentials + bed.linens

rf_time_validation <- function(train_data, test_data) {
  
  set.seed(2223)
  
  vars_rf <- all.vars(rf_formula)
  train_rf <- na.omit(train_data[, c("ln_price", vars_rf)])
  test_rf  <- na.omit(test_data[,  c("ln_price", vars_rf)])
  
  # --- align factor levels (CRITICAL)
  factor_vars <- c(
    "property_type","f_room_type","f_bathroom","f_minimum_nights",
    "refrigerator","microwave","wifi","smoke.alarm","hot.water",
    "essentials","bed.linens"
  )
  
  for (v in factor_vars) {
    if (v %in% names(train_rf) &&
        v %in% names(test_rf) &&
        is.factor(train_rf[[v]])) {
      
      test_rf[[v]] <- factor(
        test_rf[[v]],
        levels = levels(train_rf[[v]])
      )
    }
  }
  
  # ---------------------------
  # 1. Tune RF on TRAIN ONLY
  # ---------------------------
  train_control <- trainControl(
    method = "cv",
    number = 5
  )
  
  tune_grid <- expand.grid(
    mtry = c(3, 5, 7, 9),
    splitrule = "variance",
    min.node.size = 5
  )
  
  rf_model <- train(
    rf_formula,
    data = train_rf,
    method = "ranger",
    trControl = train_control,
    tuneGrid = tune_grid,
    num.trees = 500,
    importance = "impurity"
  )
  
  best_mtry <- rf_model$bestTune$mtry
  
  # ---------------------------
  # 2. LIVE prediction on Q2
  # ---------------------------
  preds <- predict(rf_model, newdata = test_rf)
  
  live_rmse <- RMSE(preds, test_rf$ln_price)
  
  # ---------------------------
  # 3. Feature importance
  # ---------------------------
  vip <- varImp(rf_model)$importance %>%
    as.data.frame() %>%
    tibble::rownames_to_column(var = "Variable") %>%
    arrange(desc(Overall)) %>%
    head(10)
  
  # ---------------------------
  # 4. Output objects
  # ---------------------------
  rmse_df <- data.frame(
    Fold  = "Live (2025Q2)",
    Model = "Random Forest",
    RMSE  = round(live_rmse, 4)
  )
  
  predictions_df <- data.frame(
    Fold = "Live (2025Q2)",
    Model = "Random Forest",
    Actual = test_rf$ln_price,
    Predicted = preds,
    Residuals = test_rf$ln_price - preds
  )
  
  list(
    model = rf_model,
    best_mtry = best_mtry,
    rmse = rmse_df,
    vip = vip,
    residuals = predictions_df
  )
}

rf_time_2025 <- rf_time_validation(
  train_data = data2025q1,
  test_data  = data2025q2
)

ggplot(rf_time_2025$vip, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_col(fill = "forestgreen") +
  coord_flip() +
  labs(
    title = paste(
      "Top 10 Variable Importance – Random Forest (Live)",
      "(mtry =", rf_time_2025$best_mtry, ")"
    ),
    x = "Variable",
    y = "Importance"
  ) +
  theme_minimal()

rmse_all_time_2025 <- rbind(
  eval_time_2025$long,
  lasso_time_2025$rmse,
  rf_time_2025$rmse
)

ggplot(rmse_all_time_2025, aes(x = Model, y = RMSE, fill = Model)) +
  geom_col() +
  labs(
    title = "Live RMSE Comparison (Train: 2025Q1 → Test: 2025Q2)",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal()

#------------------------test data = Budapest 2025q2----------------------------
#---------------------------------- XGB BOOSTING -------------------------------

xgb_time_validation <- function(train_data, test_data) {
  
  library(xgboost)
  
  boost_formula <- ln_price ~ property_type + f_room_type +
    ln_accommodates + ln_accommodates2 + ln_number_of_reviews +
    f_bathroom + beds_n + f_minimum_nights +
    refrigerator + microwave + wifi + smoke.alarm +
    hot.water + essentials + bed.linens
  
  vars <- all.vars(boost_formula)
  
  train_boost <- na.omit(train_data[, c("ln_price", vars)])
  test_boost  <- na.omit(test_data[,  c("ln_price", vars)])
  
  # ---------------------------
  # 1. Design matrices
  # ---------------------------
  X_train <- model.matrix(boost_formula, data = train_boost)[, -1]
  y_train <- train_boost$ln_price
  
  X_test <- model.matrix(boost_formula, data = test_boost)[, -1]
  y_test <- test_boost$ln_price
  
  # --- align columns (CRITICAL)
  common_cols <- intersect(colnames(X_train), colnames(X_test))
  X_train <- X_train[, common_cols]
  X_test  <- X_test[,  common_cols]
  
  dtrain <- xgb.DMatrix(data = X_train, label = y_train)
  dtest  <- xgb.DMatrix(data = X_test,  label = y_test)
  
  # ---------------------------
  # 2. Parameters
  # ---------------------------
  params <- list(
    objective = "reg:squarederror",
    max_depth = 6,
    eta = 0.05,
    subsample = 0.6,
    colsample_bytree = 0.6,
    min_child_weight = 5,
    gamma = 0
  )
  
  # ---------------------------
  # 3. CV on TRAIN ONLY
  # ---------------------------
  set.seed(223)
  xgb_cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 300,
    nfold = 5,
    metrics = "rmse",
    early_stopping_rounds = 20,
    verbose = 0
  )
  
  best_nrounds <- if (!is.null(xgb_cv$best_iteration)) {
    xgb_cv$best_iteration
  } else {
    which.min(xgb_cv$evaluation_log$test_rmse_mean)
  }
  
  cv_rmse <- min(xgb_cv$evaluation_log$test_rmse_mean)
  
  # ---------------------------
  # 4. Train final model
  # ---------------------------
  xgb_model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = best_nrounds,
    verbose = 0
  )
  
  # ---------------------------
  # 5. LIVE prediction on Q2
  # ---------------------------
  preds <- predict(xgb_model, dtest)
  live_rmse <- RMSE(preds, y_test)
  
  # ---------------------------
  # 6. Feature importance
  # ---------------------------
  xgb_imp <- xgb.importance(
    feature_names = colnames(X_train),
    model = xgb_model
  )
  
  top10_imp <- xgb_imp[1:10, ]
  
  # ---------------------------
  # 7. Outputs
  # ---------------------------
  rmse_df <- data.frame(
    Fold  = "Live (2025Q2)",
    Model = "XGBoost",
    RMSE  = round(live_rmse, 4)
  )
  
  predictions_df <- data.frame(
    Fold = "Live (2025Q2)",
    Model = "XGBoost",
    Actual = y_test,
    Predicted = preds,
    Residuals = y_test - preds
  )
  
  list(
    model = xgb_model,
    cv_rmse = cv_rmse,
    live_rmse = live_rmse,
    rmse = rmse_df,
    importance = top10_imp,
    residuals = predictions_df
  )
}

xgb_time_2025 <- xgb_time_validation(
  train_data = data2025q1,
  test_data  = data2025q2
)

xgb.plot.importance(
  xgb_time_2025$importance,
  rel_to_first = TRUE,
  xlab = "Relative Importance",
  main = "Top 10 Features – XGBoost (Live Validation)",
  col = "forestgreen"
)

rmse_all_time_2025 <- rbind(
  eval_time_2025$long,
  lasso_time_2025$rmse,
  rf_time_2025$rmse,
  xgb_time_2025$rmse
)

ggplot(rmse_all_time_2025, aes(x = Model, y = RMSE, fill = Model)) +
  geom_col() +
  labs(
    title = "Live RMSE Comparison (Train: 2025Q1 → Test: 2025Q2)",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal()


#------------------------test data = Budapest 2025q2----------------------------
#--------------------------------- GBM BOOSTING --------------------------------

gbm_time_validation <- function(train_data, test_data) {
  
  library(gbm)
  library(caret)
  
  set.seed(2025)
  
  gbm_formula <- ln_price ~ property_type + f_room_type +
    ln_accommodates + ln_accommodates2 + ln_number_of_reviews +
    f_bathroom + beds_n + f_minimum_nights +
    refrigerator + microwave + wifi + smoke.alarm +
    hot.water + essentials + bed.linens
  
  vars <- all.vars(gbm_formula)
  
  train_gbm <- na.omit(train_data[, c("ln_price", vars)])
  test_gbm  <- na.omit(test_data[,  c("ln_price", vars)])
  
  # ---------------------------
  # 1. Align factor levels (CRITICAL)
  # ---------------------------
  factor_vars <- c(
    "property_type","f_room_type","f_bathroom","f_minimum_nights",
    "refrigerator","microwave","wifi","smoke.alarm","hot.water",
    "essentials","dining.table","bed.linens"
  )
  
  for (v in factor_vars) {
    if (v %in% names(train_gbm) &&
        v %in% names(test_gbm) &&
        is.factor(train_gbm[[v]])) {
      
      test_gbm[[v]] <- factor(
        test_gbm[[v]],
        levels = levels(train_gbm[[v]])
      )
    }
  }
  
  # ---------------------------
  # 2. Tune GBM on TRAIN ONLY
  # ---------------------------
  train_control <- trainControl(
    method = "cv",
    number = 5
  )
  
  gbm_grid <- expand.grid(
    interaction.depth = c(3, 5),
    n.trees = c(100, 200),
    shrinkage = c(0.05, 0.1),
    n.minobsinnode = 10
  )
  
  gbm_model <- train(
    gbm_formula,
    data = train_gbm,
    method = "gbm",
    trControl = train_control,
    tuneGrid = gbm_grid,
    verbose = FALSE
  )
  
  # ---------------------------
  # 3. LIVE prediction on Q2
  # ---------------------------
  preds <- predict(gbm_model, newdata = test_gbm)
  live_rmse <- RMSE(preds, test_gbm$ln_price)
  
  # ---------------------------
  # 4. Variable importance
  # ---------------------------
  gbm_importance <- varImp(gbm_model)$importance %>%
    as.data.frame() %>%
    tibble::rownames_to_column(var = "Variable") %>%
    arrange(desc(Overall)) %>%
    head(10)
  
  # ---------------------------
  # 5. Outputs
  # ---------------------------
  rmse_df <- data.frame(
    Fold  = "Live (2025Q2)",
    Model = "GBM",
    RMSE  = round(live_rmse, 4)
  )
  
  predictions_df <- data.frame(
    Fold = "Live (2025Q2)",
    Model = "GBM",
    Actual = test_gbm$ln_price,
    Predicted = preds,
    Residuals = test_gbm$ln_price - preds
  )
  
  list(
    model = gbm_model,
    best_tune = gbm_model$bestTune,
    rmse = rmse_df,
    importance = gbm_importance,
    residuals = predictions_df
  )
}

gbm_time_2025 <- gbm_time_validation(
  train_data = data2025q1,
  test_data  = data2025q2
)

ggplot(gbm_time_2025$importance,
       aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top 10 Variable Importance – GBM (Live Validation)",
    x = "Variable",
    y = "Importance"
  ) +
  theme_minimal()

rmse_all_time_2025 <- rbind(
  eval_time_2025$long,
  lasso_time_2025$rmse,
  rf_time_2025$rmse,
  xgb_time_2025$rmse,
  gbm_time_2025$rmse
)


ggplot(rmse_all_time_2025, aes(x = Model, y = RMSE, fill = Model)) +
  geom_col() +
  labs(
    title = "Live RMSE Comparison (Train: 2025Q1 → Test: 2025Q2)",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal()

#-------------------------Prague------------------------------------------------
data2025q2 <- read.csv("Prague2025q1_final.csv")

unique(data2025q1$property_type)
unique(data2025q2$property_type)

data2025q2 <- data2025q2 %>%
  filter(property_type != "Serviced Apt")


#------------------------test data = Prague 2025q2------------------------------
#-----------------------------------OLS FUNCTION--------------------------------

ols_time_validation <- function(train_data, test_data) {
  
  formulas <- list(
    ln_price ~ property_type,
    ln_price ~ property_type + f_room_type + ln_accommodates + ln_accommodates2 +
      ln_number_of_reviews + f_bathroom + f_minimum_nights,
    ln_price ~ property_type + f_room_type + ln_accommodates + ln_accommodates2 + ln_number_of_reviews +
      f_bathroom + beds_n + f_minimum_nights + refrigerator + microwave + wifi + smoke.alarm +
      hot.water + essentials + bed.linens
  )
  
  rmse_vec <- numeric(3)
  predictions_list <- list()
  
  # --- align factor levels (CRITICAL for live data)
  factor_vars <- c(
    "property_type","f_room_type","f_bathroom","f_minimum_nights",
    "refrigerator","microwave","wifi","smoke.alarm","hot.water",
    "essentials","dining.table","bed.linens"
  )
  
  for (v in factor_vars) {
    if (v %in% names(train_data) &&
        v %in% names(test_data) &&
        is.factor(train_data[[v]])) {
      
      test_data[[v]] <- factor(
        as.character(test_data[[v]]),
        levels = levels(train_data[[v]])
      )
    }
  }
  
  
  for (j in 1:3) {
    
    model <- lm_robust(formulas[[j]], data = train_data)
    preds <- predict(model, newdata = test_data)
    
    rmse_vec[j] <- RMSE(preds, test_data$ln_price)
    
    predictions_list[[paste0("Model_", j)]] <- data.frame(
      Fold = "Live (2025Q2)",
      Model = paste0("Model_", j),
      Actual = test_data$ln_price,
      Predicted = preds,
      Residuals = test_data$ln_price - preds
    )
  }
  
  rmse_df <- data.frame(
    Fold = "Live (2025Q2)",
    Model1 = rmse_vec[1],
    Model2 = rmse_vec[2],
    Model3 = rmse_vec[3]
  )
  
  long_df <- data.table::melt(
    data.table::as.data.table(rmse_df),
    id.vars = "Fold",
    variable.name = "Model",
    value.name = "RMSE"
  )
  
  predictions_df <- do.call(rbind, predictions_list)
  
  list(
    rmse = rmse_df,
    long = long_df,
    residuals = predictions_df
  )
}

eval_time_2025 <- ols_time_validation(
  train_data = data2025q1,
  test_data  = data2025q2
)

ggplot(eval_time_2025$long, aes(x = Model, y = RMSE, fill = Model)) +
  geom_col() +
  labs(
    title = "Live RMSE by Model (Train: Budapest 2025Q1 → Test: Prague 2025Q2)",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal()

ggplot(eval_time_2025$residuals, aes(x = Residuals, fill = Model)) +
  geom_density(alpha = 0.5) +
  theme_minimal() +
  labs(
    title = "Residual Distribution (Train: Budapest 2025Q1 → Test: Prague 2025Q2)",
    x = "Residuals",
    y = "Density"
  )


#------------------------test data = Prague 2025q2------------------------------
#------------------------------- LASSO FUNCTION --------------------------------

lasso_formula <- ln_price ~ property_type + f_room_type + ln_accommodates + ln_accommodates2 + ln_number_of_reviews +
  f_bathroom + beds_n + f_minimum_nights + refrigerator + microwave + wifi + smoke.alarm +
  hot.water + essentials + bed.linens +
  property_type*ln_accommodates + property_type*f_room_type + property_type*f_minimum_nights +
  property_type*refrigerator + property_type*microwave + property_type*wifi +
  property_type*smoke.alarm + property_type*hot.water + 
  property_type*beds_n + property_type*f_bathroom

lasso_time_validation <- function(train_data, test_data, formula, seed_val = 5554) {
  
  set.seed(seed_val)
  
  vars_needed <- all.vars(formula)
  train_lasso <- na.omit(train_data[, c("ln_price", vars_needed)])
  test_lasso  <- na.omit(test_data[,  c("ln_price", vars_needed)])
  
  # --- build model matrices (SAME formula!)
  X_train <- model.matrix(formula, data = train_lasso)[, -1]
  y_train <- train_lasso$ln_price
  
  X_test <- model.matrix(formula, data = test_lasso)[, -1]
  y_test <- test_lasso$ln_price
  
  # --- align columns (CRITICAL for live data)
  common_cols <- intersect(colnames(X_train), colnames(X_test))
  
  X_train <- X_train[, common_cols]
  X_test  <- X_test[,  common_cols]
  
  # --- LASSO with internal CV for lambda
  lasso_cv <- cv.glmnet(
    X_train,
    y_train,
    alpha = 1,
    nfolds = 5
  )
  
  best_lambda <- lasso_cv$lambda.min
  
  preds <- as.numeric(predict(
    lasso_cv,
    newx = X_test,
    s = best_lambda
  ))
  
  rmse_val <- RMSE(preds, y_test)
  
  rmse_df <- data.frame(
    Fold = "Live (2025Q2)",
    Model = "LASSO (with interactions)",
    RMSE = round(rmse_val, 4)
  )
  
  predictions_df <- data.frame(
    Fold = "Live (2025Q2)",
    Model = "LASSO (with interactions)",
    Actual = y_test,
    Predicted = preds,
    Residuals = y_test - preds
  )
  
  list(
    rmse = rmse_df,
    residuals = predictions_df,
    lambda = best_lambda
  )
}


lasso_time_2025 <- lasso_time_validation(
  train_data = data2025q1,
  test_data  = data2025q2,
  formula    = lasso_formula,
  seed_val   = 1117
)

rmse_all_time_2025 <- rbind(
  eval_time_2025$long,
  lasso_time_2025$rmse
)

ggplot(rmse_all_time_2025, aes(x = Model, y = RMSE, fill = Model)) +
  geom_col() +
  labs(
    title = "Live RMSE Comparison (Train: Budapest 2025Q1 → Test: Prague 2025Q2)",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal()

#------------------------test data = Prague 2025q2------------------------------
#------------------------------- RANDOM FOREST ---------------------------------

rf_formula <- ln_price ~ property_type + f_room_type + ln_accommodates + ln_accommodates2 + ln_number_of_reviews +
  f_bathroom + beds_n + f_minimum_nights + refrigerator + microwave + wifi + smoke.alarm +
  hot.water + essentials + bed.linens

rf_time_validation <- function(train_data, test_data) {
  
  set.seed(2223)
  
  vars_rf <- all.vars(rf_formula)
  train_rf <- na.omit(train_data[, c("ln_price", vars_rf)])
  test_rf  <- na.omit(test_data[,  c("ln_price", vars_rf)])
  
  # --- align factor levels (CRITICAL)
  factor_vars <- c(
    "property_type","f_room_type","f_bathroom","f_minimum_nights",
    "refrigerator","microwave","wifi","smoke.alarm","hot.water",
    "essentials","bed.linens"
  )
  
  for (v in factor_vars) {
    if (v %in% names(train_rf) &&
        v %in% names(test_rf) &&
        is.factor(train_rf[[v]])) {
      
      test_rf[[v]] <- factor(
        test_rf[[v]],
        levels = levels(train_rf[[v]])
      )
    }
  }
  
  # ---------------------------
  # 1. Tune RF on TRAIN ONLY
  # ---------------------------
  train_control <- trainControl(
    method = "cv",
    number = 5
  )
  
  tune_grid <- expand.grid(
    mtry = c(3, 5, 7, 9),
    splitrule = "variance",
    min.node.size = 5
  )
  
  rf_model <- train(
    rf_formula,
    data = train_rf,
    method = "ranger",
    trControl = train_control,
    tuneGrid = tune_grid,
    num.trees = 500,
    importance = "impurity"
  )
  
  best_mtry <- rf_model$bestTune$mtry
  
  # ---------------------------
  # 2. LIVE prediction on Q2
  # ---------------------------
  preds <- predict(rf_model, newdata = test_rf)
  
  live_rmse <- RMSE(preds, test_rf$ln_price)
  
  # ---------------------------
  # 3. Feature importance
  # ---------------------------
  vip <- varImp(rf_model)$importance %>%
    as.data.frame() %>%
    tibble::rownames_to_column(var = "Variable") %>%
    arrange(desc(Overall)) %>%
    head(10)
  
  # ---------------------------
  # 4. Output objects
  # ---------------------------
  rmse_df <- data.frame(
    Fold  = "Live (2025Q2)",
    Model = "Random Forest",
    RMSE  = round(live_rmse, 4)
  )
  
  predictions_df <- data.frame(
    Fold = "Live (2025Q2)",
    Model = "Random Forest",
    Actual = test_rf$ln_price,
    Predicted = preds,
    Residuals = test_rf$ln_price - preds
  )
  
  list(
    model = rf_model,
    best_mtry = best_mtry,
    rmse = rmse_df,
    vip = vip,
    residuals = predictions_df
  )
}

rf_time_2025 <- rf_time_validation(
  train_data = data2025q1,
  test_data  = data2025q2
)

ggplot(rf_time_2025$vip, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_col(fill = "forestgreen") +
  coord_flip() +
  labs(
    title = paste(
      "Top 10 Variable Importance – RF. Train: Budapest 2025Q1 → Test: Prague 2025Q2",
      "(mtry =", rf_time_2025$best_mtry, ")"
    ),
    x = "Variable",
    y = "Importance"
  ) +
  theme_minimal()

rmse_all_time_2025 <- rbind(
  eval_time_2025$long,
  lasso_time_2025$rmse,
  rf_time_2025$rmse
)

ggplot(rmse_all_time_2025, aes(x = Model, y = RMSE, fill = Model)) +
  geom_col() +
  labs(
    title = "Live RMSE Comparison (Train: Budapest 2025Q1 → Test: Prague 2025Q2)",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal()

#------------------------test data = Prague 2025q2------------------------------
#---------------------------------- XGB BOOSTING -------------------------------

xgb_time_validation <- function(train_data, test_data) {
  
  library(xgboost)
  
  boost_formula <- ln_price ~ property_type + f_room_type +
    ln_accommodates + ln_accommodates2 + ln_number_of_reviews +
    f_bathroom + beds_n + f_minimum_nights +
    refrigerator + microwave + wifi + smoke.alarm +
    hot.water + essentials + bed.linens
  
  vars <- all.vars(boost_formula)
  
  train_boost <- na.omit(train_data[, c("ln_price", vars)])
  test_boost  <- na.omit(test_data[,  c("ln_price", vars)])
  
  # ---------------------------
  # 1. Design matrices
  # ---------------------------
  X_train <- model.matrix(boost_formula, data = train_boost)[, -1]
  y_train <- train_boost$ln_price
  
  X_test <- model.matrix(boost_formula, data = test_boost)[, -1]
  y_test <- test_boost$ln_price
  
  # --- align columns (CRITICAL)
  common_cols <- intersect(colnames(X_train), colnames(X_test))
  X_train <- X_train[, common_cols]
  X_test  <- X_test[,  common_cols]
  
  dtrain <- xgb.DMatrix(data = X_train, label = y_train)
  dtest  <- xgb.DMatrix(data = X_test,  label = y_test)
  
  # ---------------------------
  # 2. Parameters
  # ---------------------------
  params <- list(
    objective = "reg:squarederror",
    max_depth = 6,
    eta = 0.05,
    subsample = 0.6,
    colsample_bytree = 0.6,
    min_child_weight = 5,
    gamma = 0
  )
  
  # ---------------------------
  # 3. CV on TRAIN ONLY
  # ---------------------------
  set.seed(223)
  xgb_cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 300,
    nfold = 5,
    metrics = "rmse",
    early_stopping_rounds = 20,
    verbose = 0
  )
  
  best_nrounds <- if (!is.null(xgb_cv$best_iteration)) {
    xgb_cv$best_iteration
  } else {
    which.min(xgb_cv$evaluation_log$test_rmse_mean)
  }
  
  cv_rmse <- min(xgb_cv$evaluation_log$test_rmse_mean)
  
  # ---------------------------
  # 4. Train final model
  # ---------------------------
  xgb_model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = best_nrounds,
    verbose = 0
  )
  
  # ---------------------------
  # 5. LIVE prediction on Q2
  # ---------------------------
  preds <- predict(xgb_model, dtest)
  live_rmse <- RMSE(preds, y_test)
  
  # ---------------------------
  # 6. Feature importance
  # ---------------------------
  xgb_imp <- xgb.importance(
    feature_names = colnames(X_train),
    model = xgb_model
  )
  
  top10_imp <- xgb_imp[1:10, ]
  
  # ---------------------------
  # 7. Outputs
  # ---------------------------
  rmse_df <- data.frame(
    Fold  = "Live (2025Q2)",
    Model = "XGBoost",
    RMSE  = round(live_rmse, 4)
  )
  
  predictions_df <- data.frame(
    Fold = "Live (2025Q2)",
    Model = "XGBoost",
    Actual = y_test,
    Predicted = preds,
    Residuals = y_test - preds
  )
  
  list(
    model = xgb_model,
    cv_rmse = cv_rmse,
    live_rmse = live_rmse,
    rmse = rmse_df,
    importance = top10_imp,
    residuals = predictions_df
  )
}

xgb_time_2025 <- xgb_time_validation(
  train_data = data2025q1,
  test_data  = data2025q2
)

xgb.plot.importance(
  xgb_time_2025$importance,
  rel_to_first = TRUE,
  xlab = "Relative Importance",
  main = "Top 10 Features – XGBoost. Train: Budapest 2025Q1 → Test: Prague 2025Q2",
  col = "forestgreen"
)

rmse_all_time_2025 <- rbind(
  eval_time_2025$long,
  lasso_time_2025$rmse,
  rf_time_2025$rmse,
  xgb_time_2025$rmse
)

ggplot(rmse_all_time_2025, aes(x = Model, y = RMSE, fill = Model)) +
  geom_col() +
  labs(
    title = "Live RMSE Comparison (Train: Budapest 2025Q1 → Test: Prague 2025Q2)",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal()


#------------------------test data = Prague 2025q2------------------------------
#--------------------------------- GBM BOOSTING --------------------------------

gbm_time_validation <- function(train_data, test_data) {
  
  library(gbm)
  library(caret)
  
  set.seed(2025)
  
  gbm_formula <- ln_price ~ property_type + f_room_type +
    ln_accommodates + ln_accommodates2 + ln_number_of_reviews +
    f_bathroom + beds_n + f_minimum_nights +
    refrigerator + microwave + wifi + smoke.alarm +
    hot.water + essentials + bed.linens
  
  vars <- all.vars(gbm_formula)
  
  train_gbm <- na.omit(train_data[, c("ln_price", vars)])
  test_gbm  <- na.omit(test_data[,  c("ln_price", vars)])
  
  # ---------------------------
  # 1. Align factor levels (CRITICAL)
  # ---------------------------
  factor_vars <- c(
    "property_type","f_room_type","f_bathroom","f_minimum_nights",
    "refrigerator","microwave","wifi","smoke.alarm","hot.water",
    "essentials","dining.table","bed.linens"
  )
  
  for (v in factor_vars) {
    if (v %in% names(train_gbm) &&
        v %in% names(test_gbm) &&
        is.factor(train_gbm[[v]])) {
      
      test_gbm[[v]] <- factor(
        test_gbm[[v]],
        levels = levels(train_gbm[[v]])
      )
    }
  }
  
  # ---------------------------
  # 2. Tune GBM on TRAIN ONLY
  # ---------------------------
  train_control <- trainControl(
    method = "cv",
    number = 5
  )
  
  gbm_grid <- expand.grid(
    interaction.depth = c(3, 5),
    n.trees = c(100, 200),
    shrinkage = c(0.05, 0.1),
    n.minobsinnode = 10
  )
  
  gbm_model <- train(
    gbm_formula,
    data = train_gbm,
    method = "gbm",
    trControl = train_control,
    tuneGrid = gbm_grid,
    verbose = FALSE
  )
  
  # ---------------------------
  # 3. LIVE prediction on Q2
  # ---------------------------
  preds <- predict(gbm_model, newdata = test_gbm)
  live_rmse <- RMSE(preds, test_gbm$ln_price)
  
  # ---------------------------
  # 4. Variable importance
  # ---------------------------
  gbm_importance <- varImp(gbm_model)$importance %>%
    as.data.frame() %>%
    tibble::rownames_to_column(var = "Variable") %>%
    arrange(desc(Overall)) %>%
    head(10)
  
  # ---------------------------
  # 5. Outputs
  # ---------------------------
  rmse_df <- data.frame(
    Fold  = "Live (2025Q2)",
    Model = "GBM",
    RMSE  = round(live_rmse, 4)
  )
  
  predictions_df <- data.frame(
    Fold = "Live (2025Q2)",
    Model = "GBM",
    Actual = test_gbm$ln_price,
    Predicted = preds,
    Residuals = test_gbm$ln_price - preds
  )
  
  list(
    model = gbm_model,
    best_tune = gbm_model$bestTune,
    rmse = rmse_df,
    importance = gbm_importance,
    residuals = predictions_df
  )
}

gbm_time_2025 <- gbm_time_validation(
  train_data = data2025q1,
  test_data  = data2025q2
)

ggplot(gbm_time_2025$importance,
       aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top 10 Variable Importance – GBM. Train: Budapest 2025Q1 → Test: Prague 2025Q2",
    x = "Variable",
    y = "Importance"
  ) +
  theme_minimal()

rmse_all_time_2025 <- rbind(
  eval_time_2025$long,
  lasso_time_2025$rmse,
  rf_time_2025$rmse,
  xgb_time_2025$rmse,
  gbm_time_2025$rmse
)


ggplot(rmse_all_time_2025, aes(x = Model, y = RMSE, fill = Model)) +
  geom_col() +
  labs(
    title = "Live RMSE Comparison (Train: Budapest 2025Q1 → Test: Prague 2025Q2)",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal()
