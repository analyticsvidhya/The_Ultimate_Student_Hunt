import pandas as pd
import numpy as np
import xgboost as xgb
from operator import itemgetter
import datetime
import keras as k
import keras.layers as l
import pickle
import glob
from keras.layers.advanced_activations import PReLU

class LossHistory(k.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

print('Reading input data ...')
df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

########## FEATURE ENGINEERING ##########

for df in [df_train, df_test]:
    df['year'] = df.Date.apply(lambda x: int(x.split('-')[2]))
    df['month'] = df['Date'].apply(lambda x: int(x.split('-')[1]))
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

print(df_train)
print(df_train.groupby('Direction_Of_Wind')['Footfall'].mean())


drop_cols = ['Date', 'year', 'date_dm', 'dateidix']
#########################################

df_train, df_test = pickle.load(open('data.bin', 'rb'))

print('Train size', df_train.shape, 'Test size', df_test.shape)

validation = True

if validation:
    y_train = df_train.loc[df_train.year <= 1998]['Footfall']
    x_train = df_train.loc[df_train.year <= 1998].drop(['ID', 'Footfall'] + drop_cols, 1)

    y_valid = df_train.loc[df_train.year > 1998]['Footfall']
    x_valid = df_train.loc[df_train.year > 1998].drop(['ID', 'Footfall'] + drop_cols, 1)
else:
    print('VALIDATION OFF')
    y_train = df_train.loc[df_train.year < 2001]['Footfall']
    x_train = df_train.loc[df_train.year < 2001].drop(['ID', 'Footfall'] + drop_cols, 1)

    y_valid = df_train.loc[df_train.year > 2000]['Footfall']
    x_valid = df_train.loc[df_train.year > 2000].drop(['ID', 'Footfall'] + drop_cols, 1)

id_test = df_test['ID']
x_test = df_test.drop(['ID'] + drop_cols, 1)

print('Train size', x_train.shape, 'Valid size', x_test.shape)
print('Columns:', x_train.columns, '\n')

m = k.models.Sequential()

m.add(l.Dense(512, input_dim=len(x_train.columns)))
m.add(l.Activation('relu'))
#m.add(PReLU())

# m.add(l.Dropout(0.3))

m.add(l.Dense(512))
m.add(l.Activation('relu'))
#m.add(PReLU())

#m.add(l.Dropout(0.1))
# m.add(l.Dense(512))
# m.add(l.Activation('relu'))

m.add(l.Dense(1))

opt = k.optimizers.Adam(lr=0.0001)
m.compile(loss='mse', optimizer=opt)

#history = LossHistory()
#chkpnt = k.callbacks.ModelCheckpoint(filepath='../cache/weights.{epoch:02d}-{val_loss:.1f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto')
#m.fit(x_train.fillna(0).values, y_train.values, validation_data=[x_valid.fillna(0).values, y_valid.values], nb_epoch=350, batch_size=64, verbose=2, callbacks=[history])

files = glob.glob('./cache/new*.hdf5')
weights = []
for f in files:
    score = int(f.split('-')[1].split('.')[0])
    print(f, score)
    if score < 9500 and 'full' not in f:
        weights.append(f)

ps = []
for w in weights:
    print('Loading', w)

    m.load_weights(w)
    p_test = m.predict(x_test.fillna(0).values, batch_size=4096)
    print(p_test)
    ps.append([i[0] for i in p_test.tolist()])

print(pd.DataFrame(ps).T)

p_test = pd.DataFrame(ps).T.mean(axis=1)

sub = pd.DataFrame()
sub['ID'] = id_test
sub['Footfall'] = p_test
sub.to_csv('keras_epoch_blend_submission_x2.csv', index=False)
