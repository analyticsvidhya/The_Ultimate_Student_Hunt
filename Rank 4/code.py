import pandas as pd
import numpy as np
import scipy.stats as st
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

def missing_value_imputation(continuous_var,dataframes):
	loc_1 = [4,3,12,22]
	loc_2 = [23,20,17,16,14,11,9,5]
	loc_3 = [27,26,25,21,18,15,13,8,2,0]
	loc_4 = [10,24,6,7,19,1]
	locations = [loc_1,loc_2,loc_3,loc_4]
	for var in continuous_var:
		#print var
		for dataframe_name in dataframes:
			loc_value = 1
			for loc in locations:
				only_loc = dataframe_name[dataframe_name['Location_Type']==loc_value]
				grouped = only_loc[['Date',var]].groupby(['Date']).mean()
				dic = {}
				for index,row in grouped.iterrows():
				    dic[index] = row[var]
				for park in loc:
				    c = 0
				    for index,row in dataframe_name.iterrows():
				        if(row['Park_ID']==park and math.isnan(row[var])):
				        	dataframe_name.set_value(index,var,dic[row['Date']])
				loc_value+=1
	#print "Missing Value Imputation Done."

def noise_removal(dataframes):
	for dataframe_name in dataframes:
		for index,row in dataframe_name.iterrows():
			if(row['Var1']>400):
				a1 = 200.0 + (row['Var1']-400)/4.0
				dataframe_name.set_value(index,'Var1',a1)
			if(row['Max_Atmospheric_Pressure']>8600):
				a2 = row['Max_Atmospheric_Pressure']-100.0
				dataframe_name.set_value(index,'Max_Atmospheric_Pressure',a2)
			if(row['Min_Atmospheric_Pressure']<7900):
				a3 = 7900+(7900-row['Min_Atmospheric_Pressure'])/50.0
				dataframe_name.set_value(index,'Min_Atmospheric_Pressure',a3)
			if(row['Max_Ambient_Pollution']<100):
				a4 = row['Max_Ambient_Pollution']+50.0
				dataframe_name.set_value(index,'Max_Ambient_Pollution',a4)
			if(row['Average_Moisture_In_Park']<100):
				a5 = (100-row['Average_Moisture_In_Park'])+100.0
				dataframe_name.set_value(index,'Average_Moisture_In_Park',a5)
			if(row['Min_Moisture_In_Park']<50):
				a6 = (50-row['Min_Moisture_In_Park'])+50.0
				dataframe_name.set_value(index,'Min_Moisture_In_Park',a6)
			if(row['Max_Moisture_In_Park']<150):
				a7 = (150-row['Max_Moisture_In_Park'])+150.0
				dataframe_name.set_value(index,'Max_Moisture_In_Park',a7)
	
	for dataframe_name in dataframes:
		for index,row in dataframe_name.iterrows():
			if(row['Min_Atmospheric_Pressure']<7950):
				a1 = row['Min_Atmospheric_Pressure']+50.0
				dataframe_name.set_value(index,'Min_Atmospheric_Pressure',a1)

	#print "Noise Removal Done."

def run_model(model,dtrain,predictor_var,target,scoring_method='mean_squared_error'):
    cv_method = KFold(len(dtrain),5)
    cv_scores = cross_val_score(model,dtrain[predictor_var],dtrain[target],cv=cv_method,scoring=scoring_method)
    #print cv_scores, np.mean(cv_scores), np.sqrt((-1)*np.mean(cv_scores))
    
    dtrain_for_val = dtrain[dtrain['Year']<2000]
    dtest_for_val = dtrain[dtrain['Year']>1999]
    #cv_method = KFold(len(dtrain_for_val),5)
    #cv_scores_2 = cross_val_score(model,dtrain_for_val[predictor_var],dtrain_for_val[target],cv=cv_method,scoring=scoring_method)
    #print cv_scores_2, np.mean(cv_scores_2)
    
    dtrain_for_val_ini = dtrain_for_val[predictor_var]
    dtest_for_val_ini = dtest_for_val[predictor_var]
    model.fit(dtrain_for_val_ini,dtrain_for_val[target])
    pred_for_val = model.predict(dtest_for_val_ini)
        
    #print math.sqrt(mean_squared_error(dtest_for_val['Footfall'],pred_for_val))

def generate_csv(model,dtrain,dtest,predictor_var,target,filename):
    dtrain_ini = dtrain[predictor_var]
    model.fit(dtrain_ini,dtrain[target])
    dtest_ini = dtest[predictor_var]
    pred = model.predict(dtest_ini)
    test_for_sub = pd.read_csv('test.csv')
    test_for_sub[target] = 0
    i = 0
    for index,row in test_for_sub.iterrows():
        test_for_sub.set_value(index,target,pred[i])
        i+=1
    test_for_sub.to_csv(filename,columns=('ID',target),index=False)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

categorical_var = ['Park_ID','Dat','Month','Year','Location_Type']
continuous_var = ['Direction_Of_Wind','Average_Breeze_Speed','Max_Breeze_Speed','Min_Breeze_Speed','Var1','Average_Atmospheric_Pressure','Max_Atmospheric_Pressure','Min_Atmospheric_Pressure','Min_Ambient_Pollution','Max_Ambient_Pollution','Average_Moisture_In_Park','Max_Moisture_In_Park','Min_Moisture_In_Park']

dataframes = [train,test]

train['Park_ID'] = train['Park_ID'] - 12
test['Park_ID'] = test['Park_ID'] - 12

#Year
train['Year'] = 0
for index,row in train.iterrows():
    r = row['Date']
    train.set_value(index,'Year',int(r[6])*1000+int(r[7])*100+int(r[8])*10+int(r[9]))
test['Year'] = 0
for index,row in test.iterrows():
    r = row['Date']
    test.set_value(index,'Year',int(r[6])*1000+int(r[7])*100+int(r[8])*10+int(r[9]))
#Month
train['Month'] = 0
for index,row in train.iterrows():
    r = row['Date']
    train.set_value(index,'Month',int(r[3])*10+int(r[4]))
test['Month'] = 0
for index,row in test.iterrows():
    r = row['Date']
    test.set_value(index,'Month',int(r[3])*10+int(r[4]))
#Date
train['Dat'] = 0
for index,row in train.iterrows():
    r = row['Date']
    train.set_value(index,'Dat',int(r[0])*10+int(r[1]))
test['Dat'] = 0
for index,row in test.iterrows():
    r = row['Date']
    test.set_value(index,'Dat',int(r[0])*10+int(r[1]))

missing_value_imputation(continuous_var,dataframes) #Treating Missing Values
noise_removal(dataframes) #Removing Noise from the Data

train['DOW_Bin'] = train['Direction_Of_Wind']/60
test['DOW_Bin'] = test['Direction_Of_Wind']/60

train['Average_Mois_Bin'] = ((train['Average_Moisture_In_Park'])-100)/20
test['Average_Mois_Bin'] = ((test['Average_Moisture_In_Park'])-100)/20
train['Min_Mois_Bin'] = ((train['Min_Moisture_In_Park'])-50)/25
test['Min_Mois_Bin'] = ((test['Min_Moisture_In_Park'])-50)/25
train['Max_Mois_Bin'] = ((train['Max_Moisture_In_Park'])-150)/15
test['Max_Mois_Bin'] = ((test['Max_Moisture_In_Park'])-150)/15

train['Aver_Brez_Speed'] = train['Average_Breeze_Speed']/30
test['Aver_Brez_Speed'] = test['Average_Breeze_Speed']/30
train['Min_Brez_Speed'] = train['Min_Breeze_Speed']/30
test['Min_Brez_Speed'] = test['Min_Breeze_Speed']/30
train['Max_Brez_Speed'] = train['Max_Breeze_Speed']/30
test['Max_Brez_Speed'] = test['Max_Breeze_Speed']/30

train['Min_Ambi_Poll'] = train['Min_Ambient_Pollution']/40
test['Min_Ambi_Poll'] = test['Min_Ambient_Pollution']/40
train['Max_Ambi_Poll'] = train['Max_Ambient_Pollution']/40
test['Max_Ambi_Poll'] = test['Max_Ambient_Pollution']/40

train['Avg_Atm_Pres'] = (train['Average_Atmospheric_Pressure']-7890)/40
test['Avg_Atm_Pres'] = (test['Average_Atmospheric_Pressure']-7890)/40
train['Min_Atm_Pres'] = (train['Min_Atmospheric_Pressure']-7890)/40
test['Min_Atm_Pres'] = (test['Min_Atmospheric_Pressure']-7890)/40
train['Max_Atm_Pres'] = (train['Max_Atmospheric_Pressure']-7890)/40
test['Max_Atm_Pres'] = (test['Max_Atmospheric_Pressure']-7890)/40

for dataframe_name in dataframes: #Date Binning
	dataframe_name['Dat_Bin'] = 0
	for index,row in dataframe_name.iterrows():
	    r = row['Dat']
	    s = 0
	    if(r>=1 and r<=3):
	        s = 1
	    if(r>=4 and r<=6):
	        s = 2
	    if(r>=7 and r<=10):
	        s = 3
	    if(r>=11 and r<=14):
	        s = 4
	    if(r>=15 and r<=17):
	        s = 5
	    if(r>=18 and r<=20):
	        s = 6
	    if(r>=21 and r<=23):
	        s = 7
	    if(r>=24 and r<=26):
	        s = 8
	    if(r>=27 and r<=28):
	        s = 9
	    if(r>=29 and r<=31):
	        s = 10
	    dataframe_name.set_value(index,'Dat_Bin',s)

for dataframe_name in dataframes: #Month Binning
	dataframe_name['Month_Bin'] = 0
	for index,row in dataframe_name.iterrows():
	    r = row['Month']
	    s = 0
	    if(r==1 or r==2 or r==12):
	        s = 1
	    if(r==3):
	        s = 2
	    if(r==4):
	        s = 3
	    if(r==5):
	        s = 4
	    if(r==6):
	        s = 5
	    if(r==7 or r==8):
	        s = 6
	    if(r==10):
	        s = 7
	    if(r==11):
	        s = 8
	    if(r==9):
	        s = 9
	    dataframe_name.set_value(index,'Month_Bin',s)

for dataframe_name in dataframes: #Park Binning
	dataframe_name['Park_Bin'] = 0
	for index,row in dataframe_name.iterrows():
	    r = row['Park_ID']
	    s = 0
	    if(r==4):
	        s = 1
	    if(r==3 or 12):
	        s = 2
	    if(r==22):
	        s = 3
	    if(r==6 or r==7):
	        s = 4
	    if(r==1 or r==24):
	        s = 5
	    if(r==10):
	        s = 6
	    if(r==19):
	        s = 7
	    if(r==20):
	        s = 8
	    if(r==5):
	        s = 9
	    if(r==23):
	        s = 10
	    if(r==16):
	        s = 11
	    if(r==11 or r==14):
	        s = 12
	    if(r==9):
	        s = 13
	    if(r==21 or r==26 or r==27):
	        s = 14
	    if(r==18):
	        s = 15
	    if(r==15):
	        s = 16
	    if(r==0 or r==2 or r==25):
	        s = 17
	    if(r==8 or r==13):
	        s = 18
	    if(r==17):
	        s = 19
	    dataframe_name.set_value(index,'Park_Bin',s)

train['Var1_Bin'] = train['Var1']/20
test['Var1_Bin'] = test['Var1']/20

train2 = train[train['Park_ID']!=7]
gbr = GradientBoostingRegressor(n_estimators=900)
predictor_var = ['Var1_Bin','Park_Bin','Dat_Bin','Month_Bin','DOW_Bin','Aver_Brez_Speed','Max_Brez_Speed','Min_Brez_Speed','Average_Mois_Bin','Max_Mois_Bin','Min_Mois_Bin','Min_Ambi_Poll','Max_Ambi_Poll','Location_Type']
target = 'Footfall'
#print "run_model started."
#run_model(gbr,train2,predictor_var,target)
#print "generate_csv started."
generate_csv(gbr,train2,test,predictor_var,target,'final prediction.csv')


