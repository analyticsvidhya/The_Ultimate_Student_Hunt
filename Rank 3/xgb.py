import pandas as pd
import numpy as np
import xgboost as xgb
from operator import itemgetter
import datetime
import pickle

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance

compute_features = True

if compute_features:
    print('Reading input data ...')
    df_train = pd.read_csv('./train.csv')
    df_test = pd.read_csv('./test.csv')

    ########## FEATURE ENGINEERING ##########

    for df in [df_train, df_test]:
        df['year'] = df.Date.apply(lambda x: int(x.split('-')[2]))
        df['month'] = df['Date'].apply(lambda x: int(x.split('-')[1]))
        #df['monthday'] = df['Date'].apply(lambda x: int(x.split('-')[1]) * 30 + int(x.split('-')[0]))
        df['dateidix'] = df[['Date', 'Park_ID']].apply(lambda x: x[0] + str(x[1]), axis=1)
        df['date_dm'] = df['Date'].apply(lambda x: '-'.join(x.split('-')[:2]))
        df['Direction_Of_Wind'] = df['Direction_Of_Wind'].apply(lambda x: x + np.random.uniform(-10, 10))
        # df['pressure_range'] = df['Max_Atmospheric_Pressure'] - df['Min_Atmospheric_Pressure']
        # df['pollution_range'] = df['Max_Ambient_Pollution'] - df['Min_Ambient_Pollution']
        # df['breeze_range'] = df['Max_Breeze_Speed'] - df['Min_Breeze_Speed']

    for i in df_train['Park_ID'].unique().tolist():
        df_train['Park' + str(i)] = (df_train['Park_ID'].values == i)
        df_test['Park' + str(i)] = (df_test['Park_ID'] == i)

    date_encode = df_train.groupby('date_dm')['Footfall'].mean()
    df_train['date_encode'] = df_train['date_dm'].apply(lambda x: date_encode[x] + np.random.uniform(0, 25) if x in date_encode else 0)
    df_test['date_encode'] = df_test['date_dm'].apply(lambda x: date_encode[x] + np.random.uniform(0, 25) if x in date_encode else 0)

    ########## LAG/LEAD ##########

    table = pd.concat([df_train, df_test]).set_index('dateidix')
    vals = {}

    for c in ['Direction_Of_Wind', 'Average_Breeze_Speed', 'Var1', 'Average_Atmospheric_Pressure', 'Average_Moisture_In_Park']:
        print('Processing column', c)
        vals[c + '_lag1'] = []
        vals[c + '_lag2'] = []
        vals[c + '_lead1'] = []
        vals[c + '_lead2'] = []
        for row in table[['Date', 'Park_ID']].values:
            day, month, year = [int(x) for x in row[0].split('-')]
            lag1_ix = '-'.join([str(x).rjust(2, '0') for x in [day - 1, month, year]]) + str(row[1])
            lag2_ix = '-'.join([str(x).rjust(2, '0') for x in [day - 2, month, year]]) + str(row[1])
            lead1_ix = '-'.join([str(x).rjust(2, '0') for x in [day + 1, month, year]]) + str(row[1])
            lead2_ix = '-'.join([str(x).rjust(2, '0') for x in [day + 2, month, year]]) + str(row[1])

            row_vals = []
            for ix in [lag1_ix, lag2_ix, lead1_ix, lead2_ix]:
                try:
                    val = table.ix[lag1_ix][c]
                except KeyError:
                    val = np.nan
                row_vals.append(val)

            vals[c + '_lag1'].append(row_vals[0])
            vals[c + '_lag2'].append(row_vals[1])
            vals[c + '_lead1'].append(row_vals[2])
            vals[c + '_lead2'].append(row_vals[3])

    for a, b in vals.items():
        df_train[a] = b[:len(df_train)]
        df_test[a] = b[len(df_train):]

    pickle.dump([df_train, df_test], open('data.bin', 'wb'), protocol=4)

else:

    print('Reading cached data ...')
    df_train, df_test = pickle.load(open('data.bin', 'rb'))

###############################

print(df_train.head())
#print(df_train.groupby('Direction_Of_Wind')['Footfall'].mean())

drop_cols = ['Date', 'year', 'date_dm', 'dateidix']
#########################################

print('Train size', df_train.shape, 'Test size', df_test.shape)

validation = False
if validation:
    y_train = df_train.loc[df_train.year <= 1998]['Footfall']
    x_train = df_train.loc[df_train.year <= 1998].drop(['ID', 'Footfall'] + drop_cols, 1)
else:
    print('VALIDATION OFF')
    y_train = df_train['Footfall']
    x_train = df_train.drop(['ID', 'Footfall'] + drop_cols, 1)

y_valid = df_train.loc[df_train.year > 1998]['Footfall']
x_valid = df_train.loc[df_train.year > 1998].drop(['ID', 'Footfall'] + drop_cols, 1)

id_test = df_test['ID']
x_test = df_test.drop(['ID'] + drop_cols, 1)


print('Train size', x_train.shape, 'Valid size', x_test.shape)
print('Columns:', x_train.columns, '\n')

# Set parameters
params = {}
params['booster'] = 'gbtree'
params['eta'] = 0.02 # Learning rate
params['eval_metric'] = 'rmse'
params['max_depth'] = 3
params['colsample_bytree'] = 0.9
params['subsample'] = 0.9
params['silent'] = 1

# Convert data to xgboost format
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

# List of datasets to evaluate on, last one is used for early stopping
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

# Train!
# Third value is number of rounds (n_estimators), early_stopping_rounds stops training when it hasn't improved for that number of rounds
clf = xgb.train(params, d_train, 7200, watchlist, early_stopping_rounds=500, verbose_eval=25)

# Predict
d_test = xgb.DMatrix(x_test)
p_test = clf.predict(d_test) # Returns array with *single column*, probability of 1

print(get_importance(clf, list(x_train.columns.values)))

sub = pd.DataFrame()
sub['ID'] = id_test
sub['Footfall'] = p_test
sub.to_csv('simple_xgb_submission_full.csv', index=False)
