library(xgboost)

#------------------------------------------
# Reading the training data and the labels.
#------------------------------------------

train <- read.csv("train.csv", stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)

#------------------------------------
# Reading the aggreagte data tables.
#------------------------------------

train_agg <- read.csv("train_aggregates.csv", stringsAsFactors = FALSE)
test_agg <- read.csv("test_aggregates.csv", stringsAsFactors = FALSE)

#----------------------------------------------------
# Joining the aggregate tables and the original ones.
#----------------------------------------------------

train <- cbind(train, train_agg)
test <- cbind(test,test_agg)

#-------------------------------
# Removing the aggregate tables.
#-------------------------------

rm(train_agg, test_agg)

#------------------------------
# Reading the target vectors.
#------------------------------

labels <- read.csv("target.csv", stringsAsFactors = FALSE)

#----------------------------------------------------------------------------------------------
# Missing values are encoded as negative values -- this only works with tree based algorithms.
#----------------------------------------------------------------------------------------------

train[is.na(train)] <- -300
test[is.na(test)] <- -300

#----------------------------
# Selecting the id variable.
#----------------------------

test_id <- test$id
train_id <- train$id

#-------------------------------------------
# Dropping the ID variable before training.
#-------------------------------------------

train <- train[, 2:ncol(train)]
test <- test[, 2:ncol(test)]

#-----------------------
# Selecting the target.
#-----------------------

target <- labels$x

#------------------------
# Transforming the sets.
#------------------------

train <- data.matrix(train)
test <- data.matrix(test)

#-------------------------------------
# A vector to contain the predictions.
#-------------------------------------

zeros <- rep(0, 39420)

#--------------------------------------------
# I will fit 50 models.
# The predictions are averaged out.
# So this is simply an ensemble of boosters.
#--------------------------------------------

control <- 50

for (i in 1:control){
  
  bst <- xgboost(data = train,
                 label = target,
                 eta = 0.1,
                 max_depth = 6,
                 subsample = 0.5,
                 colsample_bytree = 1,
                 nrounds = 400,
                 objective = "reg:linear",
                 eval_metric = "rmse",
                 maximize = FALSE)
 
  yhat <- predict(bst,test)
  zeros <- zeros + yhat
}

zeros <- zeros/control

#---------------------------------------------
# Creating a submission frame.
# Renaming columns and dealing with the bias.
#---------------------------------------------

submission <- data.frame(test_id, zeros, stringsAsFactors = FALSE)

colnames(submission) <- c("ID", "Footfall")

submission$Footfall <- submission$Footfall*(0.987)

#------------------------------
# Dumping the submission file.
#------------------------------
write.csv(submission, file = "final_submission_eval.csv", row.names = FALSE)
