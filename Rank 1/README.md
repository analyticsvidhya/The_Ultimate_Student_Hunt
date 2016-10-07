# AV Ultimate Student Hunt

This set of scripts is my solution for the Ultimate Student Hunt Challenge from Analyics Vidhya. With this analytic solution I was able to take the first place on the private leaderboard. The machine learning problem itself was regression based on panel data with daily frequency. The good public and private leaderboard results of the predictor were mainly based on the fact that the across cross-sectional unit aggregates helped with the quite prevalent missing feature problem.

## Data cleaning
### Functions in the data cleaner
There are three customly defined functions in the data cleaning script these are:

1. The feature name normalizer -- this removes dots and capital letters.

2. The hot-one-encoder -- this attaches dummies to an existing dataframe based on vector of possible values that a certain column might take.

3. The data munger -- this function is the actual data cleaner.

### The baseline data cleaning process itself
As a first step I normalized the feature names in the table -- capital letters can only lead to problems. During the baseline data cleaning process I included the following features:

* The park identifier is hot one encoded -- it is interesting that the train and test samples are different in this regard there is a park that is only present in one of them. This step used the hot-one encoder.
* The day variable cannot be used for extracting variables with *lubridate*, so I generated a new day variable. This new variable has the name monkey day.
* Based on this day of week (1-7), day of month(1-31) and day of year can be generated -- it has to be emhasized that the day of year variable in itself leads to low test set performance if the missing values are not treated properly. I will elaborate on this statement later.
* Monthly dummiy features are also extracted. Here the hot one encoder is used again.
* The exogenous weather features are extracted as they are -- I did not differentiate them I assumed that on yearly basis stationarity is satisfied. 
* Based on the weather features I calculated moving window means and standard deviation values. Also I used minimum and maximum values for these variables. These variables capture local tendencies in the data and they also deal with the high frequncy noise in the data.
* The location type variable is encoded with  binary features.

With these variables (if day of year was excluded) and  with the application of model stacking (extreme gradient boosting) one could have a solid 106 root mean squared error on the public leaderboard. Moreover, it became evident that the predictor is biased -- namely that it has consequently overestimated the footfall. Discounting every prediction by a factor below 1, imporeved the error. This is not so surprising if one considers that time series trends might have changed also a spatial separation of the test and training sets might affect the estimator in a similar way. Namely, one can say that somekind of residual autocorrelation probably results in a biased estimator.

## The introduction of aggregates

Based on the fact that there are data points from the same time period (panel nature of the data) one can calculate time-period specific  aggregated weather meaures. To put it simply, one migh calculate the average minimal air pressure on a given day or the standard deviation of average pollution in the different parks. These tables of aggregates later are left joined to the clean data tables -- because of the neat functionalities of R I used column binding.  The joined subtables are the following:

* Means of the weather variables.
* Minima of the weather variables
* Maxima of the weather variables.
* Standard deviation of the weather variables.

The inclusion of the above mentioned aggregates resulted in a root mean squared error of 100 on public the leaderboard. The day of year variable without theses aggregated led to overfitting, however if these variables were included it started to help with obtaining a better fit. If the aggregates and the day of year were included together the resulting root mean squared error on the public leaderboard was about 95.

## Model fitting and generating a submission file

Before model fitting I have imputed negative values in place of the missing feature values -- this is meaningful because the features in this specific case cannot take negative values. For model fitting I have used extreme gradient boosting. The model parameters were the following:
* Number of trees at 400.
* Learning rate of 0.1.
* Depth of 6.
* Subsampling rate at 0.5 -- it helped with the quite strong noise. The model did not try to generalize to the noise.

Initally I have tried cross-validation and grid search to find an optimal parameter setting, but it was inconclusive due to the fact that the cross-validation and actual error were quite far off from each other. 

Because there wass a random element (subsampling of rows) in the model fitting an estimation averaging process is meaningful -- this is why I estimated 50 models. The average of these predictions was used for creating the submission file. In the end I have discounted the resulting estimates with a factor close to 0.99.
