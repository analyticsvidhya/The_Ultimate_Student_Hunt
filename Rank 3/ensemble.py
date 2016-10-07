import pandas as pd
import numpy as np

a = pd.read_csv('simple_xgboost_submission.csv')
b = pd.read_csv('keras_epoch_blend_submission_x2.csv')

sub = pd.DataFrame()
sub['ID'] = a['ID']
sub['Footfall'] = np.mean(a.Footfall, b.Footfall)
sub.to_csv('final prediction', index=False)
