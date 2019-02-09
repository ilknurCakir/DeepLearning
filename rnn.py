## importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

train_fname = 'Google_Stock_Price_Train.csv'
test_fname = 'Google_Stock_Price_Test.csv'

traindata  = pd.read_csv(train_fname)
train_data = traindata.iloc[:, 1:2]

testdata = pd.read_csv(test_fname)
test_data = testdata.iloc[:, 1:2]

# scaling prices
minmaxscaler = MinMaxScaler(feature_range = (0, 1))
train_scaled = minmaxscaler.fit_transform(train_data)

# RNN looks at last 3-month period to estimate today's price

X_train = []
y_train = []

for i in range(60, train_data.shape[0]):
    X_train.append(train_scaled[i-60:i, 0])
    y_train.append(train_scaled[i, 0])
    
# converting list to numpy array
X_train  = np.array(X_train)
y_train = np.array(y_train)

## 3D tensor with shape (batch_size, timesteps, input_dim)

## reshaping
## adding 3rd dimension
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras import optimizers

## Building the model
# 4 stacked LSTM

model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = X_train.shape[1:]))
model.add(Dropout(rate = 0.2))
model.add(LSTM(40, return_sequences = True))
model.add(Dropout(rate = 0.2))
model.add(LSTM(40, return_sequences = True))
model.add(Dropout(rate = 0.2))
model.add(LSTM(40, return_sequences = False))
model.add(Dropout(rate = 0.2))
model.add(Dense(1, activation = 'relu'))

optimizer = optimizers.Adam(lr = 0.01)
model.compile(optimizer = optimizer, 
              loss = "mean_squared_error")

model.fit(X_train, y_train, 
          epochs = 20,
          batch_size = 32)

## Predicting Test data

# cancatenate train dataand test data to get 3-month stock prices prior to
# some values in test data
data_total = pd.concat([train_data['Open'], test_data['Open']], axis = 0) #rowbase
inputs = data_total[len(data_total) - len(test_data) - 60:].values
inputs = inputs.reshape(-1, 1)

#scaling test data
inputs_scaled = minmaxscaler.transform(inputs)

## getting previous 3-month values for test data
X_test= []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs_scaled[i-60:i, 0])
    
# converting list to numpy array
X_test  = np.array(X_test)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

predicted_price = model.predict(X_test)
predicted_price_nominal = minmaxscaler.inverse_transform(predicted_price)

# visualizing the results

plt.plot(predicted_price_nominal, color = 'blue', label = 'predicted price')
plt.plot(test_data, color = 'red', label = 'actual price')
plt.legend()
plt.show()

























