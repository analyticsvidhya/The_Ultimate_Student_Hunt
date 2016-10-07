import pandas as pd 
import numpy as np
sub1=pd.read_csv('intermediate_gbm.csv')
sub2=pd.read_csv('intermediate_xgb.csv')
sub3=pd.read_csv('intermediate_keras.csv')

#take average and save
pred2=(sub1.Footfall+sub2.Footfall+sub3.Footfall)/3.0
sub1.Footfall=pred2
sub1.to_csv('final prediction.csv',index=False)
