train_file='Train.csv'
test_file='Test.csv'

import pandas as pd
import numpy as np
import datetime
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pressure_fea=['Average_Atmospheric_Pressure','Max_Atmospheric_Pressure','Min_Atmospheric_Pressure']
drop_fea=['ID','Date','Location_Type']
drop_fea.extend(pressure_fea)
drop_fea=list(set(drop_fea))

def get_wind_dir(x):
	if 315<x<=45: return 0
	if 45<x<=135: return 1
	if 135<x<=225: return 2
	if 225<x<=315: return 3
	else: return -1

def get_normal_date(x):
	if 0<x<=10: return x
	if 10<x<=20: return 15
	if 23<x<=25: return 24
	if 29<x<=31: return 30
	else: return x 

dfmean=0
def preprocess(file,istrian):
	df=pd.read_csv(file,parse_dates=['Date'],dayfirst=True)
	end_missing=['Average_Atmospheric_Pressure','Max_Atmospheric_Pressure',
	'Min_Atmospheric_Pressure','Min_Ambient_Pollution','Max_Ambient_Pollution']
	# df=df.fillna(-1)
	df=df.fillna(method='pad')
	if istrian:
		global dfmean
		dfmean=df.mean()
		df=df.fillna(dfmean)
		df=df[df.Park_ID!=19]
		outcome=df.Footfall
		df=df.drop(['Footfall'],axis=1)
	else:
		df=df.fillna(dfmean)
		outcome=np.nan

	df['month']=df['Date'].apply(lambda x: x.month)
	df['date']=df['Date'].apply(lambda x: x.day)
	df['sardiya']=df['month'].apply(lambda x: 1 if x in [1,2,11,12,3] else 0)
	df.date=df.date.apply(get_normal_date)
	df['Direction_Of_Wind2']=df.Direction_Of_Wind.apply(get_wind_dir)

	return df,outcome

#keras model 
def larger_model():
	model = Sequential()
	model.add(Dense(100, input_dim=16, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

#load training dataset
train,outcome=preprocess(train_file,True)
parkids=train.Park_ID
tardates=train.Date
ids_train=train.ID

#dimentionality reduction on pressure features
pca2=PCA(1)
train['pressure']=pca2.fit_transform(train[pressure_fea])

train.drop(drop_fea,axis=1,inplace=True)

# fix random seed for reproducibility
seed = 7
print 'classification start'
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, nb_epoch=40, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
clf=pipeline
clf.fit(train,outcome)

#load test datast
test,faaltu=preprocess(test_file,False)
ids=test.ID
parkids_test=test.Park_ID
test['pressure']=pca2.transform(test[pressure_fea])

test.drop(drop_fea,axis=1,inplace=True)
pred=clf.predict(test)
out_df=pd.DataFrame({'ID':ids,'Footfall':pred})

#save submission
out_df.to_csv('intermediate_keras.csv',index=False)
