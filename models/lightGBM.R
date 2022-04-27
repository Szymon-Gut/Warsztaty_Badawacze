source("functions.R")
library(purrr)
library(Metrics)
library(caret)
library(lightgbm)
set.seed(101)

df <- read.csv('df_pca_with_extra_data.csv')
lgbm_test_regression_rmse <- function(model) {
  set.seed(101)
  
  df <- read.csv('df_pca_with_extra_data.csv')
  
  split1 <- sample(c(rep(0, 0.7 * nrow(df)), rep(1, 0.3 * nrow(df))))
  train <- df[split1 == 0, ]  
  
  
  test <- df[split1== 1, ]
  test <- test[,-which(names(df) == "BuyPrice")]
  
  predicted <- lightGBM_predict(model, test,autofactor = TRUE) 
  actual <- df[split1 == 1, which(names(df) == "BuyPrice")]
  return(rmse(predicted,actual))
}
lgbm_test_regression_mse <- function(model) {
  set.seed(101)
  
  df <- read.csv('df_pca_with_extra_data.csv')
  
  split1 <- sample(c(rep(0, 0.7 * nrow(df)), rep(1, 0.3 * nrow(df))))
  train <- df[split1 == 0, ]  
  
  
  test <- df[split1== 1, ]
  test <- test[,-which(names(df) == "BuyPrice")]
  
  predicted <- lightGBM_predict(model, test,autofactor = TRUE) 
  actual <- df[split1 == 1, which(names(df) == "BuyPrice")]
  return(mse(predicted,actual))
}
lgbm_test_regression_mae <- function(model) {
  set.seed(101)
  
  df <- read.csv('df_pca_with_extra_data.csv')
  
  split1 <- sample(c(rep(0, 0.7 * nrow(df)), rep(1, 0.3 * nrow(df))))
  train <- df[split1 == 0, ]  
  
  
  test <- df[split1== 1, ]
  test <- test[,-which(names(df) == "BuyPrice")]
  
  predicted <- lightGBM_predict(model, test,autofactor = TRUE) 
  actual <- df[split1 == 1, which(names(df) == "BuyPrice")]
  return(mae(predicted,actual))
}

split1 <- sample(c(rep(0, 0.7 * nrow(df)), rep(1, 0.3 * nrow(df))))
train <- df[split1 == 0, ]
X_train <- train[,-which(names(df) == "BuyPrice")]
y_train <- df[split1 == 0, which(names(df) == "BuyPrice")]
test <- df[split1 == 1, ]
fifth_model <- lightGBM_function(
train,
"BuyPrice",
"regression",
max_depth=4,
num_leaves=7,
num_iterations=100,
learning_rate=0.1,
autofactor = TRUE
 )
lgbm_test_regression_rmse(model)
predicted <- lightGBM_predict(model, X_train,autofactor = TRUE) 
rmse(predicted,y_train)


X_test <- test[,-which(names(df) == "BuyPrice")]
y_test <- df[split1 == 1, which(names(df) == "BuyPrice")]
predicted <- lightGBM_predict(model, X_test,autofactor = TRUE) 

#checking metrics MSE

# Customsing the tuning grid
gbmGrid <-  expand.grid(max_depth = c(7,9,12), 
                        num_leaves = c(10,12,16),
                        num_iterations=c(50,100,150),
                        learning_rate=c(0.005,0.05,0.1)
                        )


mymodels <- pmap(gbmGrid, .f=lightGBM_function, data=train,target="BuyPrice",objective="regression",autofactor=TRUE)
rmse_models <- lapply(mymodels, lgbm_test_regression_rmse)
mse_models <- lapply(mymodels, lgbm_test_regression_mse)
mae_models <- lapply(mymodels, lgbm_test_regression_mae)



idx_mse <- which.min(mse_models)
idx_rmse <- which.min(rmse_models)
idx_mae <- which.min(mae_models)

best_model_mse <- mymodels[[idx_mse]]
best_model_rmse <- mymodels[[idx_rmse]]
best_model_mae <- mymodels[[idx_mae]]
print(best_model)

print("mae")
print(idx_mae)
print(best_model_mae$params)
print("mse")
print(idx_mse)
print(best_model_mse$params)
print("rmse")
print(idx_rmse)
print(best_model_rmse$params)

first_model <- mymodels[[idx_mae]]
second_model <- mymodels[[idx_mse]]




predicted <- lightGBM_predict(best_model, X_train,autofactor = TRUE) 
rmse(predicted,y_train)



params_grid <- expand.grid(
  max_depth = 3:10,
  num_leaves = 4:18,
  num_iterations=seq(50,200, by=50),
  learning_rate=seq(0.001,0.5,by=0.01)
)
set.seed(26)
params_sample <- params_grid[
  sample(1:nrow(params_grid),50),
]

mymodels_rs <- pmap(params_sample, .f=lightGBM_function, data=train,target="BuyPrice",objective="regression",autofactor=TRUE)
rmse_models_rs <- lapply(mymodels_rs, lgbm_test_regression_rmse)
mse_models_rs <- lapply(mymodels_rs, lgbm_test_regression_mse)
mae_models_rs <- lapply(mymodels_rs, lgbm_test_regression_mae)


idx_mse_rs <- which.min(mse_models_rs)
idx_rmse_rs <- which.min(rmse_models_rs)
idx_mae_rs <- which.min(mae_models_rs)

best_model_mse_rs <- mymodels_rs[[idx_mse_rs]]
best_model_rmse_rs <- mymodels_rs[[idx_rmse_rs]]
best_model_mae_rs <- mymodels_rs[[idx_mae_rs]]
print("mae")
print(idx_mae_rs)
print(best_model_mae_rs$params)
print("mse")
print(idx_mse_rs)
print(best_model_mse_rs$params)
print("rmse")
print(idx_rmse_rs)
print(best_model_rmse_rs$params)

third_model <- best_model_mae_rs

fourth_model <- best_model_mse_rs



#metrics


#1
predicted <- lightGBM_predict(first_model, X_train,autofactor = TRUE) 
rmse(predicted,y_train)
#train
#75682.42
predicted <- lightGBM_predict(first_model, X_test,autofactor = TRUE) 
rmse(predicted,y_test)
#test
#74744.98


#2
predicted <- lightGBM_predict(second_model, X_train,autofactor = TRUE) 
rmse(predicted,y_train)
#train
#75621.59
predicted <- lightGBM_predict(second_model, X_test,autofactor = TRUE) 
rmse(predicted,y_test)
#test
#74396.56

#3
predicted <- lightGBM_predict(third_model, X_train,autofactor = TRUE) 
rmse(predicted,y_train)
#train
#72069.84
predicted <- lightGBM_predict(third_model, X_test,autofactor = TRUE) 
rmse(predicted,y_test)
#test
#77780.66

#4
predicted <- lightGBM_predict(fourth_model, X_train,autofactor = TRUE) 
rmse(predicted,y_train)
#train
#75637.04
predicted <- lightGBM_predict(fourth_model, X_test,autofactor = TRUE) 
rmse(predicted,y_test)
#test
#74744.98

#5
predicted <- lightGBM_predict(fifth_model, X_train,autofactor = TRUE) 
rmse(predicted,y_train)
#train
#70978.47
predicted <- lightGBM_predict(fifth_model, X_test,autofactor = TRUE) 
rmse(predicted,y_test)
#test
#80082.44
