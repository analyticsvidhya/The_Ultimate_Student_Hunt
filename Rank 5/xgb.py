train_file='Train.csv'
test_file='Test.csv'

import pandas as pd
import numpy as np
import datetime

from sklearn import preprocessing
import xgboost as xgb
from matplotlib import pylab as plt
import operator


drop_fea=['ID','Date','Park_ID','Location_Type']
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

def preprocess(file,istrian):
	df=pd.read_csv(file,parse_dates=['Date'],dayfirst=True)
	end_missing=['Average_Atmospheric_Pressure','Max_Atmospheric_Pressure',
	'Min_Atmospheric_Pressure','Min_Ambient_Pollution','Max_Ambient_Pollution']
	df=df.fillna(-1)
	if istrian:
		outcome=df.Footfall
		df=df.drop(['Footfall'],axis=1)
	else:
		outcome=np.nan

	df['month']=df['Date'].apply(lambda x: x.month)
	df['date']=df['Date'].apply(lambda x: x.day)
	df['weekday']=df['Date'].apply(lambda x: x.weekday())
	df['sardiya']=df['month'].apply(lambda x: 1 if x in [1,2,11,12,3] else 0)
	df.date=df.date.apply(get_normal_date)
	dummies=pd.get_dummies(df.Park_ID,prefix='park')
	dummies=pd.get_dummies(df.Location_Type,prefix='location')
	df['Direction_Of_Wind2']=df.Direction_Of_Wind.apply(get_wind_dir)

	return df,outcome

#load training set
train,outcome=preprocess(train_file,True)
parkids=train.Park_ID
tardates=train.Date
train.drop(drop_fea,axis=1,inplace=True)

#specify model parameter
param = {'max_depth':4, 'silent':0, 'objective':'reg:linear',
'seed':7, 'subsample':1,'min_child_weight':50, 'eta':0.05}
param['eval_metric'] = 'rmse'
num_round = 1100
param['silent']=1
print 'classification start'

test,faaltu=preprocess(test_file,False)
ids=test.ID
parkids_test=test.Park_ID
test.drop(drop_fea,axis=1,inplace=True)

#start training on park 12 and then predict on park 12
fdata=train[parkids==12]
label=outcome[parkids==12]
# clf.fit(fdata,label)
dtrain=xgb.DMatrix(fdata.as_matrix(),label=label)
bst = xgb.train(param, dtrain, num_round)
odata=test[parkids_test==12]
id_id1=ids[parkids_test==12]
odata=xgb.DMatrix(odata.as_matrix())
pred=bst.predict(odata)
df1=pd.DataFrame({'ID':id_id1,'Footfall':pred})

#train on other parks and on predict 
for id1 in range(13,19)+range(20,40):
	print id1
	fdata=train[parkids==id1]
	label=outcome[parkids==id1]
	dtrain=xgb.DMatrix(fdata.as_matrix(),label=label)
	bst = xgb.train(param, dtrain, num_round)
	odata=test[parkids_test==id1]
	id_id1=ids[parkids_test==id1]
	odata=xgb.DMatrix(odata.as_matrix())

	pred=bst.predict(odata)
	df2=pd.DataFrame({'ID':id_id1,'Footfall':pred})
	df1=pd.concat([df1,df2],axis=0)

#save submission xgb
df1.to_csv('intermediate_xgb.csv',index=False)
