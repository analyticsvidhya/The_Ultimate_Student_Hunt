library(dplyr)

#-------------------------------------
# Reading the training and test sets.
#-------------------------------------

train <- read.csv('Train.csv', stringsAsFactors = FALSE)
test <- read.csv("Test.csv", stringsAsFactors = FALSE)

#----------------------------------------------------------
# Dropping the target and transforming the variable names.
#----------------------------------------------------------

train <- train[, 1:17]
colnames(train) <- tolower(colnames(test))
colnames(test) <- tolower(colnames(test))

#-------------------------------------
# Transforming the id to be a time ID.
#-------------------------------------

train$time_id <- as.numeric(substr(as.character(train$id), 1, 5))
test$time_id <- as.numeric(substr(test$id, 1, 5))

#-------------------------------------
# Selecting the exogeneous varaibles.
#-------------------------------------

keepers <-  c("direction_of_wind", 
                "average_breeze_speed",
                "max_breeze_speed",            
                "min_breeze_speed",
                "var1",           
                "average_atmospheric_pressure",
                "max_atmospheric_pressure",    
                "min_atmospheric_pressure",
                "min_ambient_pollution",       
                "max_ambient_pollution",
                "average_moisture_in_park",    
                "max_moisture_in_park",
                "min_moisture_in_park")

#------------------------------------------------------------
# Generating aggregates of the exogeneous variables.
# Means, minimae, maximae and standard deviation aggregates.
#------------------------------------------------------------

train_means <- aggregate(train[keepers], list(train$time_id), mean, na.rm = TRUE)
test_means <- aggregate(test[keepers], list(test$time_id), mean, na.rm = TRUE)

train_mins <- aggregate(train[keepers], list(train$time_id), min, na.rm = TRUE)
test_mins <- aggregate(test[keepers], list(test$time_id), min, na.rm = TRUE)

train_maxs <-aggregate(train[keepers], list(train$time_id), max, na.rm = TRUE)
test_maxs <- aggregate(test[keepers], list(test$time_id), max, na.rm = TRUE)

train_sds <-aggregate(train[keepers], list(train$time_id), sd, na.rm = TRUE)
test_sds <- aggregate(test[keepers], list(test$time_id), sd, na.rm = TRUE)

#--------------------------------------------
# The aggregate tables are joined together.
#--------------------------------------------

new_train <- cbind(train_means, train_mins[, 2:14], train_maxs[, 2:14], train_sds[, 2:14])
new_test <- cbind(test_means, test_mins[, 2:14], test_maxs[, 2:14], test_sds[, 2:14])

#----------------------------------------------
# A convenient variable naming scheme is made.
#----------------------------------------------

colnames(new_train) <- c("time_id", paste0("X_", 1:52))
colnames(new_test) <- c("time_id", paste0("X_", 1:52))

#--------------------------------------------------------------
# These tables are joined to the original train and test ID.
#--------------------------------------------------------------

train_key <- data.frame(train$time_id)
colnames(train_key) <- "time_id"
new_train <- left_join(train_key, new_train)
new_train <- new_train[, 2:53]

test_key <- data.frame(test$time_id)
colnames(test_key) <- "time_id"
new_test <- left_join(test_key, new_test)
new_test <- new_test[, 2:53]

#------------------------------------------------------
# Dumping the aggregate tables for the whole datasets.
#------------------------------------------------------

write.csv(new_train, "train_aggregates.csv", row.names = FALSE)
write.csv(new_test, "test_aggregates.csv", row.names = FALSE)
