import pandas as pd
import os
import keras_tuner as kt
from datetime import datetime
## https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f

## load in data

df = pd.read_csv('sp500.csv')

df['Date'] = pd.to_datetime(df['Date'])
df['Close'] = df['Close'].astype(str)
df['Close'].replace(",","",regex=True, inplace=True)
df['Close'] = df['Close'].astype(float)

training = df.loc[:800, 'Close']
test = df.loc[800:,'Close']

training = training/max(df['Close'])
test = test/max(df['Close'])
all = df['Close'] / max(df['Close'])

# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []
X_test = []
y_test = []
for i in range(60, 800):
    X_train.append(training[i-60:i])
    y_train.append(training[i])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

for i in range(800,df.shape[0]-1):
    X_test.append(all[i-60:i])
    y_test.append(all[i])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras import models, layers
from keras import optimizers
from keras import regularizers
from keras.layers import SimpleRNN
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))

model.summary()

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 10, batch_size = 800)

predicted = model.predict(X_test)

df1 = pd.DataFrame

y_test  = y_test.reshape(y_test.shape[0],1)
df1 = pd.DataFrame(y_test, columns = ['actual'])
df2 = pd.DataFrame(predicted, columns = ['pred'])
df1['pred'] = df2['pred']

