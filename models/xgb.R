# first two imports are there to avoid masking issues
library(plyr)
library(dplyr)

library(Metrics)
library(caret)
source('XGBoost/XGBoost_function.R')
source('XGBoost/XGBoost_predict.R')


# =========================== #
# Reading and processing data #
# =========================== #

df <- read.csv('data/df_pca_with_extra_data.csv')

df_preprocessed <- XGBoost_data_preprocessing(df, 'BuyPrice')
train <- df_preprocessed$train
test <- df_preprocessed$test

y_train <- train$BuyPrice
y_test <- test$BuyPrice


# =============================== #
# A model with default parameters #
# =============================== #

model_def <- XGBoost_function(df, 'BuyPrice')


# ======================================= #
# A model with manually chosen parameters #
# ======================================= #

model_man <- XGBoost_function(df, 'BuyPrice',
                              nrounds = 1000,
                              eta = 0.05, 
                              max_depth = 7)


# ========================================= #
# Tree models with random search parameters #
# ========================================= #

train_control <- trainControl(method = 'cv', 
                              number = 5, 
                              search = 'random')

params_grid_tree <- expand.grid(
  max_depth = 3:8,
  eta = seq(0.05, 0.3, by = 0.01),
  gamma = 0:10,
  min_child_weight = 1:5,
  nrounds = seq(50, 200, by = 10),
  colsample_bytree = 1,
  subsample = 1
)

set.seed(3)
params_sample_tree_1 <- params_grid_tree[
  sample(1:nrow(params_grid_tree), 40), ]
model_rs_tree_1 <- train(BuyPrice~.,
                         data = train,
                         method = 'xgbTree',
                         trControl = train_control,
                         verbosity = 0,
                         tuneGrid = params_sample_tree_1)$finalModel

set.seed(4)
params_sample_tree_2 <- params_grid_tree[
  sample(1:nrow(params_grid_tree), 40), ]
model_rs_tree_2 <- train(BuyPrice~.,
                         data = train,
                         method = 'xgbTree',
                         trControl = train_control,
                         verbosity = 0,
                         tuneGrid = params_sample_tree_2)$finalModel

set.seed(8)
params_sample_tree_3 <- params_grid_tree[
  sample(1:nrow(params_grid_tree), 40), ]
model_rs_tree_3 <- train(BuyPrice~.,
                         data = train,
                         method = 'xgbTree',
                         trControl = train_control,
                         verbosity = 0,
                         tuneGrid = params_sample_tree_3)$finalModel


# ========================================= #
# Dart models with random search parameters #
# ========================================= #

params_grid_dart <- expand.grid(
  max_depth = 3:8,
  eta = seq(0.05, 0.3, by = 0.01),
  gamma = 0:10,
  min_child_weight = 1:5,
  nrounds = seq(50, 200, by = 10),
  colsample_bytree = 1,
  subsample = 1,
  rate_drop = seq(0.05, 0.2, by = 0.025),
  skip_drop = seq(0, 0.5, by = 0.05)
)

set.seed(34)
params_sample_dart_1 <- params_grid_dart[
  sample(1:nrow(params_grid_tree), 40), ]
model_rs_dart_1 <- train(BuyPrice~.,
                         data = train,
                         method = 'xgbDART',
                         trControl = train_control,
                         verbosity = 0,
                         tuneGrid = params_sample_dart_1)$finalModel

set.seed(1)
params_sample_dart_2 <- params_grid_dart[
  sample(1:nrow(params_grid_tree), 40), ]
model_rs_dart_2 <- train(BuyPrice~.,
                         data = train,
                         method = 'xgbDART',
                         trControl = train_control,
                         verbosity = 0,
                         tuneGrid = params_sample_dart_2)$finalModel

set.seed(10)
params_sample_dart_3 <- params_grid_dart[
  sample(1:nrow(params_grid_tree), 40), ]
model_rs_dart_3 <- train(BuyPrice~.,
                         data = train,
                         method = 'xgbDART',
                         trControl = train_control,
                         verbosity = 0,
                         tuneGrid = params_sample_dart_3)$finalModel


# ======= #
# Summary #
# ======= #

all_models <- list(
  "Default params" = model_def,
  "Manual params" = model_man,
  "Regular tree 1" = model_rs_tree_1,
  "Regular tree 2" = model_rs_tree_2,
  "Regular tree 3" = model_rs_tree_3,
  "Dart 1" = model_rs_dart_1,
  "Dart 2" = model_rs_dart_2,
  "Dart 3" = model_rs_dart_3
)

print_metrics <- function(xgb_model) {
  y_train_pred <- XGBoost_predict(train, 'BuyPrice', xgb_model)
  y_test_pred <- XGBoost_predict(test, 'BuyPrice', xgb_model)
  
  metrics <- list(
    'RMSE' = rmse,
    'MAE' = mae
  )
  
  cat('-> Metrics values (train | test):\n')
  for (metric_name in names(metrics)) {
    metric <- metrics[[metric_name]]
    cat(metric_name, ': ', metric(y_train, y_train_pred), 
        ' | ', metric(y_test, y_test_pred), '\n',
        sep = '')
  }
}

print_params <- function(xgb_model) {
  params_list <- xgb_model$params
  for (param_name in names(params_list)) {
    cat('-> ', param_name, ': ', params_list[[param_name]], '\n',
        sep = '')
  }
}

# print metrics for all models
for (model_name in names(all_models)) {
  cat('# ==== ', model_name, ' ==== #\n', sep = '')
  print_metrics(all_models[[model_name]])
  cat('\n')
}

# print params for all models
for (model_name in names(all_models)) {
  cat('# ==== ', model_name, ' ==== #\n', sep = '')
  print_params(all_models[[model_name]])
  cat('\n')
}
