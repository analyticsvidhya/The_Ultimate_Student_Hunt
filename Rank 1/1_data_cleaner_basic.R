library(caTools)
library(lubridate)

#---------------------------------
# Reading the train and test sets.
#---------------------------------


train <- read.csv('Train.csv', stringsAsFactors = FALSE)
test <- read.csv("Test.csv", stringsAsFactors = FALSE)

#--------------------------------------------------------------
# Selecting the target and dropping it from the training frame.
#--------------------------------------------------------------

target <- train$Footfall

train <- train[, 1:17]

proper_feature_names <- function(input_table){
  
  #--------------------------------------------
  # This function normalizes the column names.
  # INPUT -- Table with messed up column names.
  # OUTPUT -- Table with proper column names.
  #--------------------------------------------
  
  colnames(input_table) <- tolower(colnames(input_table))
  
  colnames(input_table) <- gsub('([[:punct:]])|\\s+','_',colnames(input_table))
  
  while (any(grepl("__",colnames(input_table),fixed = TRUE)) == TRUE){
    colnames(input_table) <- gsub("__","_",colnames(input_table),fixed = TRUE) 
  }
  
  colnames(input_table) <- gsub("\\*$", "",colnames(input_table))
  
  return(input_table)
}


dummygen <- function(new_table, original_table, dummified_column, column_values, new_name){ 
  
  #---------------------------------------------------------------------------------------
  # This function generates dummies from a categorical variable and adds them to a table.
  # INPUT 1. -- The new cleaned table -- I will attach the dummies.
  # INPUT 2. -- The original table that is being cleaned.
  # INPUT 3. -- The column that has the strings.
  # INPUT 4. -- The unique values in the column encoded.
  # INPUT 5. -- The new name of the columns.
  # OUTPUT -- The new table with the dummy variables.
  #---------------------------------------------------------------------------------------
  
  i <- 0
  
  for (val in column_values){
    i <- i + 1
    new_variable <- data.frame(matrix(0, nrow(new_table), 1))
    new_variable[original_table[,dummified_column] == val, 1] <- 1
    colnames(new_variable) <- paste0(new_name, i)
    new_table <- cbind(new_table,new_variable)
  }
  return(new_table)
}

#------------------------------------------------------------
# Normalizing the feature names in the train and test tables.
#------------------------------------------------------------

train <- proper_feature_names(train)
test <- proper_feature_names(test)

data_munger <- function(input_table){
  
  #------------------------------------
  # This function cleans the data.
  # INPUT -- The table to be cleaned.
  # OUTPUT -- The cleaned numeric table.
  #------------------------------------
  
  #----------------------------------------------
  # Defining a target table for the cleaned data.
  #----------------------------------------------
  
  new_table <- data.frame(matrix(0, nrow(input_table), 1))
  new_table[, 1] <- input_table$id
  
  #-----------------------------------------------------
  # The first variable is an artifical ID.
  #-----------------------------------------------------
  
  colnames(new_table) <- c("id")
  
  #----------------------------
  # Park ID dummy generation.
  #----------------------------
  
  park_id <- c(12:39)
  
  new_table <- dummygen(new_table, input_table, "park_id", park_id, "park_id_")
  
  #------------------------------------------------------
  # Generating a proper day variable in the input table.
  #------------------------------------------------------
  
  input_table$monkey_day <- paste0(substr(input_table$date, 7, 10),"-",substr(input_table$date, 4, 5),"-",substr(input_table$date, 1, 2)) 
  
  #-------------------------------------
  # Generating  a day of week indicator.
  #-------------------------------------
  
  input_table$days <- lubridate::wday(input_table$monkey_day)
  
  #---------------------------------------
  # Generating  a day of year categorical.
  #---------------------------------------
  
  new_table$super_monkey_day <- yday(input_table$monkey_day)
  
  #--------------------------------------
  # Generating  a day of month categorical.
  #--------------------------------------
   
  new_table$hyper_monkey_day <- mday(input_table$monkey_day)
  
  #---------------------------------------------------
  # Creating dummies from the day of week categorical.
  #---------------------------------------------------
  
  days <- c(1:7)

  new_table <- dummygen(new_table, input_table, "days", days, "week_days_")

  #-----------------------------------------------------------------------
  # Days simple solution -- this is biased, but works as a biweekly proxy.
  #-----------------------------------------------------------------------
  
  new_table$date <- yday(input_table$date)
  
  #------------------------
  # Month dummy variables.
  #------------------------
   
  input_table$first_two <- substr(input_table$date, 6, 7)
  
  first_two <- c("01", "02", "03", "04", "05", "06",
                 "07", "08", "09", "10", "11", "12")
  
  
  new_table <- dummygen(new_table, input_table, "first_two", first_two, "first_two_")
  
  #----------------------------------------------------
  # Extracting the numeric variables the way they are.
  #----------------------------------------------------
  
  columns_to_extract_exactly <- c("direction_of_wind", 
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
  
  sub_table <- input_table[, columns_to_extract_exactly]
  
  new_table <- cbind(new_table, sub_table)
  
  #---------------------------------------------------------------------
  # Creating moving window standard deviation variables for the different parks.
  #------------------------------------------------------------------------------
  
  names_to_use <- colnames(sub_table)
  
  keys <- unique(input_table$park_id)
  
  for (i in 1:ncol(sub_table)){
    for (k in keys){
      sub_table[input_table$park_id == k, i] <- runsd(sub_table[input_table$park_id == k, i], 4, endrule = "constant")
    }
  }
  
  colnames(sub_table) <- paste0("sd_", names_to_use)
  
  new_table <- cbind(new_table, sub_table)
  
  #----------------------------------------------------------------
  # Creating moving window mean variables for the different parks.
  #----------------------------------------------------------------
  
  keys <- unique(input_table$park_id)
  
  for (i in 1:ncol(sub_table)){
    for (k in keys){
      sub_table[input_table$park_id == k, i] <- runmean(sub_table[input_table$park_id == k, i], 4, endrule = "constant")
    }
  }
  
  colnames(sub_table) <- paste0("mean_", names_to_use)
  
  new_table <- cbind(new_table, sub_table)
  
  #-----------------------------------------------------------------
  # Creating moving window maxima variables for the different parks.
  #-----------------------------------------------------------------
  
  keys <- unique(input_table$park_id)
  
  for (i in 1:ncol(sub_table)){
    for (k in keys){
      sub_table[input_table$park_id == k, i] <- runmax(sub_table[input_table$park_id == k, i], 7, endrule = "constant")
    }
  }
  
  colnames(sub_table) <- paste0("max_", names_to_use)
  
  new_table <- cbind(new_table, sub_table)
  
  #-----------------------------------------------------------------
  # Creating moving window minima variables for the different parks.
  #-----------------------------------------------------------------
  
  keys <- unique(input_table$park_id)
  
  for (i in 1:ncol(sub_table)){
    for (k in keys){
      sub_table[input_table$park_id == k, i] <- runmin(sub_table[input_table$park_id == k, i], 7, endrule="constant")
    }
  }
  
  colnames(sub_table) <- paste0("min_", names_to_use)
  
  new_table <- cbind(new_table, sub_table)
  
  #----------------------------
  # Creating location dummies.
  #----------------------------
  
  location_type <- c(1:4)
  
  new_table <- dummygen(new_table, input_table, "location_type", location_type, "location_type_")
  
  return(new_table)
}

#--------------------------------------------
# Creating the cleaned test and train tables.
#--------------------------------------------

new_train <- data_munger(train)
new_test <- data_munger(test)

#-----------------------------------------
# Dumping the tables and arget variables.
#-----------------------------------------

write.csv(new_train, file = "train.csv", row.names = FALSE)
write.csv(new_test, file = "test.csv", row.names = FALSE)
write.csv(target, file = "target.csv", row.names = FALSE)
