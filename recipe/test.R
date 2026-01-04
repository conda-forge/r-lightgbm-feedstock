print("--- beginning test.R --")

library(lightgbm)

data(agaricus.train, package = "lightgbm")

num_iterations <- 7L

params <- list(
    objective = "binary"
    , metric = "binary_logloss"
    # train a small model
    , num_iterations = num_iterations
    , num_leaves = 5L
    # get DEBUG-level logs
    , verbose = 1L
    # test OpenMP codepaths
    , num_threads = 2L
    # make results deterministic
    , deterministic = TRUE
    , force_row_wise = TRUE
    , seed = 708L
)

# test training
dtrain <- lgb.Dataset(
    agaricus.train$data
    , label = agaricus.train$label
    , params = params
)
bst <- lgb.train(params, dtrain)

stopifnot(bst$current_iter() == num_iterations)
stopifnot(bst$num_iter() == num_iterations)
stopifnot(bst$num_trees() == num_iterations)

# test model serialization works
bst_from_string <- lightgbm::lgb.load(
    model_str = bst$save_model_to_string()
)

# test prediction
stopifnot(
    predict(bst, agaricus.train$data) == predict(bst_from_string, agaricus.train$data)
)

print("--- done running test.R --")
