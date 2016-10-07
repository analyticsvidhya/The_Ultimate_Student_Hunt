train_file='Train.csv'
test_file='Test.csv'

import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA

pressure_fea=['Average_Atmospheric_Pressure','Max_Atmospheric_Pressure','Min_Atmospheric_Pressure']

drop_fea=['ID','Date','Park_ID','Location_Type']
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
	df=df.fillna(method='pad')
	if istrian:
		global dfmean
		dfmean=df.mean()
		df=df.fillna(dfmean)
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

#load training dataset
train,outcome=preprocess(train_file,True)
parkids=train.Park_ID
tardates=train.Date
ids_train=train.ID

#dimentionality reduction on pressure features
pca2=PCA(1)
train['pressure']=pca2.fit_transform(train[pressure_fea])
train.drop(drop_fea,axis=1,inplace=True)

print 'classification start'
clf=GradientBoostingRegressor(max_depth=5,random_state=7,loss='huber',min_samples_leaf=100,
	learning_rate=0.05,n_estimators=500)

#load test datast
test,faaltu=preprocess(test_file,False)
ids=test.ID
parkids_test=test.Park_ID
test['pressure']=pca2.transform(test[pressure_fea])
test.drop(drop_fea,axis=1,inplace=True)

#train and predict for park 12
fdata=train[parkids==12]
label=outcome[parkids==12]
clf.fit(fdata,label)
odata=test[parkids_test==12]
id_id1=ids[parkids_test==12]
pred=clf.predict(odata)
df1=pd.DataFrame({'ID':id_id1,'Footfall':pred})

#train and predict for other parks
for id1 in range(13,19)+range(20,40):
	print id1
	fdata=train[parkids==id1]
	label=outcome[parkids==id1]
	clf.fit(fdata,label)
	odata=test[parkids_test==id1]
	id_id1=ids[parkids_test==id1]
	pred=clf.predict(odata)
	df2=pd.DataFrame({'ID':id_id1,'Footfall':pred})
	df1=pd.concat([df1,df2],axis=0)

df1.to_csv('intermediate_gbm.csv',index=False)
