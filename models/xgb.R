# first two imports are there to avoid masking issues
library(plyr)
library(dplyr)

library(Metrics)
library(caret)
library(ggplot2)
library(DALEX)
source('XGBoost/XGBoost_function.R')
source('XGBoost/XGBoost_predict.R')


# =========================== #
# Reading and processing data #
# =========================== #

df <- read.csv('data/df_pca_with_extra_data.csv')
df_no_pca <- read.csv('data/final_df_with_extra_data.csv')

df_preprocessed <- XGBoost_data_preprocessing(df, 'BuyPrice')
df_preproc_no_pca <- XGBoost_data_preprocessing(df_no_pca, 'BuyPrice')

train <- df_preprocessed$train
test <- df_preprocessed$test

# x_train <- train %>% select(-BuyPrice)
# x_test <- test %>% select(-BuyPrice)

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

# no pca

model_man_no_pca <- XGBoost_function(df_no_pca, 'BuyPrice',
                                     nrounds = 1000,
                                     eta = 0.05, 
                                     max_depth = 7)

y_pred_no_pca <- XGBoost_predict(df_preproc_no_pca$test, 'BuyPrice', model_man_no_pca)
rmse(y_test, y_pred_no_pca)


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

final_models <- list(
    "Manual params" = model_man,
    "Regular tree 1" = model_rs_tree_1,
    "Regular tree 2" = model_rs_tree_2
)

print_metrics <- function(xgb_model) {
  y_train_pred <- XGBoost_predict(train, 'BuyPrice', xgb_model)
  y_test_pred <- XGBoost_predict(test, 'BuyPrice', xgb_model)
  
  metrics <- list(
    'RMSE' = rmse,
    'MAE' = mae,
    'R2' = function(actual, predicted) {
      sum((predicted - mean(actual))^2) / sum((actual - mean(actual))^2)
    }
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


# ============ #
# Final models #
# ============ #

final_models <- list(
  'Manual params' = model_man,
  'Regular tree 1' = model_rs_tree_1,
  'Regular tree 2' = model_rs_tree_2
)

y_pred_man_train <- XGBoost_predict(test, 'BuyPrice', model_man)
y_pred_man_test <- XGBoost_predict(test, 'BuyPrice', model_man)
y_pred_rstree1_train <- XGBoost_predict(test, 'BuyPrice', model_rs_tree_1)
y_pred_rstree1_test <- XGBoost_predict(test, 'BuyPrice', model_rs_tree_1)
y_pred_rstree2_train <- XGBoost_predict(test, 'BuyPrice', model_rs_tree_2)
y_pred_rstree2_test <- XGBoost_predict(test, 'BuyPrice', model_rs_tree_2)

# print metrics for final models
for (model_name in names(final_models)) {
  cat('# ==== ', model_name, ' ==== #\n', sep = '')
  print_metrics(all_models[[model_name]])
  cat('\n')
}

plot_residuals <- function(model_name, xgb_model) {
  y_pred <- XGBoost_predict(test, 'BuyPrice', xgb_model)
  rplot <- ggplot() +
    geom_point(aes(x = 1:length(y_test), 
                   y = y_pred - y_test), color= 'white') +
    geom_hline(yintercept = 0, color = 'blue') +
    xlab("Predykcje") +
    ylab("Rezydua") +
    ggtitle(paste("Wykres rezyduów dla", model_name)) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", color = "white", size = 14),
      #plot.subtitle = element_text(hjust = 0.5, face = "italic", color = "white"),
      axis.title = element_text(color = "white", size = 13),
      axis.text = element_text(color = "white", size = 12),
      legend.position = "none", ## transparent
      legend.background = element_rect(fill = "transparent"),
      legend.box.background = element_rect(fill = "transparent"),
      panel.background = element_rect(fill = "transparent"),
      panel.grid = element_line(color = "white"),
      panel.border = element_rect(color = "transparent"),
      plot.background = element_rect(fill = "transparent", color = NA))
  ggsave(rplot,
         filename = paste(model_name, ".png", sep = ""),
         bg = 'transparent')
}

plot_residuals('XGBoost 1', model_man)
plot_residuals('XGBoost 2', model_rs_tree_1)
plot_residuals('XGBoost 3', model_rs_tree_2)



# =================== #
# Other teams' models #
# =================== #

other_models <- list(
  'Jodek 1' = XGBoost_function(df, 'BuyPrice', 
                               booster = 'gbtree', 
                               nrounds = 464,
                               eta = 0.04462844,
                               max_depth = 3,
                               gamma = 3.580246,
                               min_child_weight = 1,
                               subsample = 0.5339151,
                               colsample_bytree = 0.6035825),
  'Jodek 2' = XGBoost_function(df, 'BuyPrice', 
                               booster = 'gbtree', 
                               nrounds = 305,
                               eta = 0.05525418,
                               max_depth = 3,
                               gamma = 1.062662,
                               min_child_weight = 10,
                               subsample = 0.9131714,
                               colsample_bytree = 0.6054747),
  'Jodek 3' = XGBoost_function(df, 'BuyPrice', 
                               booster = 'gbtree', 
                               nrounds = 81,
                               eta = 0.19592307,
                               max_depth = 2,
                               gamma = 8.526370,
                               min_child_weight = 16,
                               subsample = 0.5217781,
                               colsample_bytree = 0.6626087),
  'Jodek 4' = XGBoost_function(df, 'BuyPrice', 
                               booster = 'gbtree',
                               nrounds = 405,
                               eta = 0.20411763,
                               max_depth = 2,
                               gamma = 3.240579,
                               min_child_weight = 1,
                               subsample = 0.6497563,
                               colsample_bytree = 0.6905933),
  'Jodek 5' = XGBoost_function(df, 'BuyPrice', 
                               booster = 'gbtree', 
                               nrounds = 163,
                               eta = 0.08603091,
                               max_depth = 2,
                               gamma = 3.549911,
                               min_child_weight = 0,
                               subsample = 0.6369859,
                               colsample_bytree = 0.4167207),
  'BezNazwy 1' = XGBoost_function(df, 'BuyPrice', 
                                  booster = 'gbtree', 
                                  nrounds = 100,
                                  eta = 0.07,
                                  max_depth = 7,
                                  gamma = 0,
                                  min_child_weight = 1,
                                  subsample = 1,
                                  colsample_bytree = 1),
  'BezNazwy 2' = XGBoost_function(df, 'BuyPrice', 
                                  booster = 'gbtree', 
                                  nrounds = 79,
                                  eta = 0.1,
                                  max_depth = 7,
                                  gamma = 0,
                                  min_child_weight = 1,
                                  subsample = 1,
                                  colsample_bytree = 1),
  'BezNazwy 3' = XGBoost_function(df, 'BuyPrice', 
                                  booster = 'gbtree', 
                                  nrounds = 100,
                                  eta = 0.1,
                                  max_depth = 7,
                                  gamma = 0,
                                  min_child_weight = 1,
                                  subsample = 1,
                                  colsample_bytree = 1),
  'BezNazwy 4' = XGBoost_function(df, 'BuyPrice', 
                                  booster = 'gbtree', 
                                  nrounds = 91,
                                  eta = 0.05,
                                  max_depth = 7,
                                  gamma = 0,
                                  min_child_weight = 2,
                                  subsample = 1,
                                  colsample_bytree = 1),
  'BezNazwy 5' = XGBoost_function(df, 'BuyPrice', 
                                  booster = 'gbtree', 
                                  nrounds = 115,
                                  eta = 0.06,
                                  max_depth = 7,
                                  gamma = 0,
                                  min_child_weight = 1,
                                  subsample = 1,
                                  colsample_bytree = 1),
  'DeepRession 1' = XGBoost_function(df, 'BuyPrice', 
                                     booster = 'gbtree', 
                                     nrounds = 29,
                                     eta = 0.1300062,
                                     max_depth = 24,
                                     gamma = 0,
                                     min_child_weight = 0,
                                     subsample = 0.5,
                                     colsample_bytree = 1,
                                     lambda = 1,
                                     alpha = 100),
  'DeepRession 2' = XGBoost_function(df, 'BuyPrice', 
                                     booster = 'gbtree', 
                                     nrounds = 59,
                                     eta = 0.229115,
                                     max_depth = 24,
                                     gamma = 0,
                                     min_child_weight = 5.694278,
                                     subsample = 0.5,
                                     colsample_bytree = 1,
                                     lambda = 100,
                                     alpha = 100),
  'DeepRession 3' = XGBoost_function(df, 'BuyPrice', 
                                     booster = 'gbtree', 
                                     nrounds = 29,
                                     eta = 0.6057803,
                                     max_depth = 11,
                                     gamma = 0,
                                     min_child_weight = 0,
                                     subsample = 0.7833569,
                                     colsample_bytree = 1,
                                     lambda = 100,
                                     alpha = 65.16275),
  'DeepRession 4' = XGBoost_function(df, 'BuyPrice', 
                                     booster = 'gbtree', 
                                     nrounds = 41,
                                     eta = 0.4608851,
                                     max_depth = 4,
                                     gamma = 0,
                                     min_child_weight = 0,
                                     subsample = 1,
                                     colsample_bytree = 1,
                                     lambda = 1,
                                     alpha = 100),
  'DeepRession 5' = XGBoost_function(df, 'BuyPrice', 
                                     booster = 'gbtree', 
                                     nrounds = 32,
                                     eta = 0.2775322,
                                     max_depth = 24,
                                     gamma = 0,
                                     min_child_weight = 0,
                                     subsample = 0.5,
                                     colsample_bytree = 1,
                                     lambda = 100,
                                     alpha = 100)
)


# print metrics for their models
for (model_name in names(other_models)) {
  cat('# ==== ', model_name, ' ==== #\n', sep = '')
  print_metrics(other_models[[model_name]])
  cat('\n')
}


# Explaining the best model #


final_model <- XGBoost_function(df_no_pca, 'BuyPrice',
                                nrounds = 1000,
                                eta = 0.05, 
                                max_depth = 7)

df <- df_preproc_no_pca$train

explainer <- explain(final_model,
                     model.matrix(BuyPrice ~ . - 1, df),
                     y = df$BuyPrice,
                     label = 'xgboost', colorize = FALSE)

mp <- model_performance(explainer)
mp
