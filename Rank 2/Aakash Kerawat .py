
# coding: utf-8

# In[164]:

import pandas as pd
import os
from dateutil import parser
from xgboost import XGBRegressor
import numpy as np


# In[165]:

#set train and test file locations
df_train = pd.read_csv('Train.csv')
df_test = pd.read_csv('Test.csv')


# In[167]:

def rolling_diff(series):
    return series[1]-series[0]

def divide(x, y):
    return x/y

# In[168]:

def add_all_features(df_train):
    #Adding the features from date
    df_train['Date'] = pd.to_datetime(df_train.Date, dayfirst=True)
    df_train['day'] = df_train.Date.apply(lambda x: x.day)
    df_train['month'] = df_train.Date.apply(lambda x: x.month)
    df_train['year'] = df_train.Date.apply(lambda x: x.year)
    df_train['weekday'] = df_train.Date.apply(lambda x : x.weekday())
    df_train['day_of_year'] = df_train['month']*30+df_train['day']
    print('date features added\n')
    
    print('changing Var1 to some integer values')
    #By carrying out this operation 99% of Var1 values got converted into some integral values.
    #Integers gave us a better intuition but we were unable to figure out a different feature from it.
    df_train['Var1'] = df_train.Var1*(3/2.49)
    df_train.loc[df_train.Var1==0, 'Var1'] = -1
    
    #These are some of the most important features of our model. The percentage changes of the weather conditions.
    print('\nAdding percentage change features...')
    df_train['per_chng_direction'] = df_train['Direction_Of_Wind'].pct_change()
    df_train['per_chng_direction_p7'] = df_train['Direction_Of_Wind'].pct_change(periods=7)#%age change every week
    df_train['per_chng_direction_p30'] = df_train['Direction_Of_Wind'].pct_change(periods=30)#%age change every month
    df_train['per_chng_avg_speed'] = df_train['Average_Breeze_Speed'].pct_change()
    df_train['per_chng_min_moisture'] = df_train['Min_Moisture_In_Park'].pct_change()
    df_train['per_chng_avg_pressure'] = df_train['Average_Atmospheric_Pressure'].pct_change()
    df_train['Avg_Ambient_Pollution'] = df_train['Max_Ambient_Pollution']+df_train['Min_Ambient_Pollution']
    #A little workaround since my kernel died when I directly divided any series with something, idk why. :D
    df_train['Avg_Ambient_Pollution'] = df_train['Avg_Ambient_Pollution'].apply(lambda x: x/2)
    df_train['per_chng_avg_pollution'] = df_train['Avg_Ambient_Pollution'].pct_change(periods=3)
    
    print('\nAdding rolling means...')    
    #Moving averages / Rolling means proved to be another set of very important features.
    df_train['rolling_mean_direction'] = df_train.Direction_Of_Wind.rolling(window=3).mean()
    df_train['rolling_mean_direction_w7'] = df_train.Direction_Of_Wind.rolling(window=7).mean()
    df_train['rolling_mean_direction_w30'] = df_train.Direction_Of_Wind.rolling(window=30).mean()
    df_train['rolling_mean_direction_w75'] = df_train.Direction_Of_Wind.rolling(window=75).mean()
    df_train['rolling_mean_per_chng_direction'] = df_train.per_chng_direction.rolling(window=3).mean()
    df_train['rolling_mean_avg_speed'] = df_train.Average_Breeze_Speed.rolling(window=3).mean()
    df_train['rolling_mean_avg_speed_w7'] = df_train.Average_Breeze_Speed.rolling(window=7).mean()
    df_train['rolling_mean_avg_speed_w9'] = df_train.Average_Breeze_Speed.rolling(window=9).mean()
    df_train['rolling_mean_per_chng_avg_speed'] = df_train.per_chng_avg_speed.rolling(window=5).mean()
    df_train['rolling_mean_avg_pressure'] = df_train.Average_Atmospheric_Pressure.rolling(window=3).mean()
    df_train['rolling_mean_avg_pressure_w7'] = df_train.Average_Atmospheric_Pressure.rolling(window=7).mean()
    df_train['rolling_mean_avg_pressure_w9'] = df_train.Average_Atmospheric_Pressure.rolling(window=9).mean()
    df_train['rolling_mean_min_pressure'] = df_train.Min_Atmospheric_Pressure.rolling(window=3).mean()
    df_train['rolling_mean_min_moisture'] = df_train.Min_Moisture_In_Park.rolling(window=3).mean()
    df_train['rolling_mean_min_moisture_w7'] = df_train.Min_Moisture_In_Park.rolling(window=7).mean()
    df_train['rolling_mean_min_moisture_w9'] = df_train.Min_Moisture_In_Park.rolling(window=9).mean()
    df_train['rolling_mean_avg_pollution'] = df_train.Avg_Ambient_Pollution.rolling(window=3).mean()
    df_train['rolling_mean_avg_pollution_w7'] = df_train.Avg_Ambient_Pollution.rolling(window=7).mean()


# In[169]:

add_all_features(df_train)


# In[170]:

#adding another feature.
park_month_mois_mean = df_train.groupby(['Park_ID', 'month'], as_index=False)['Min_Moisture_In_Park'].mean()
df_train = df_train.merge(park_month_mois_mean, on=['Park_ID', 'month'], how='left', suffixes=('', '_park_month_mois'))


# In[171]:

df_train.fillna(-1, inplace=True)

# In[172]:

### best estimator
gbr = XGBRegressor( nthread=-1,  missing= -1, n_estimators=300, learning_rate=0.02, max_depth=17, subsample=0.9
                   , min_child_weight=3, colsample_bytree=0.7, reg_alpha=100, reg_lambda=100, silent=False)


# In[173]:

predictors = df_train.columns.drop(['Footfall', 'Date','year', 'ID', 'Location_Type'
                                    ,'weekday', 'Avg_Ambient_Pollution'])


# In[174]:

gbr.fit(df_train[predictors], df_train['Footfall'])


# In[176]:

df_test['Footfall'] = None


# In[177]:

#Adding features to the test set
add_all_features(df_test)


# In[178]:

df_test = df_test.merge(park_month_mois_mean, on=['Park_ID', 'month'], how='left', suffixes=('', '_park_month_mois'))


# In[179]:

df_test.fillna(-1, inplace=True)


# ### predicting

# In[180]:

#Predicting and generating submission.
submission = pd.DataFrame({'ID':df_test.ID, 'Footfall':gbr.predict(df_test[predictors])})


# In[181]:

submission = submission[['ID', 'Footfall']]


# In[182]:

submission['Footfall'] = submission.Footfall.apply(lambda x: round(x))


# In[184]:

submission.to_csv('final_solution_.csv', index=False)