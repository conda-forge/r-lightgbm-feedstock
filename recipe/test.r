# derived from https://github.com/Microsoft/LightGBM/blob/master/R-package/README.md
library(lightgbm)
data(agaricus.train, package='lightgbm')
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label = train$label)
model <- lgb.cv(
    params = list(
        objective = "regression"
        , metric = "l2"
    )
    , data = dtrain
)
