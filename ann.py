# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sns


# reading in data
df = pd.read_csv('Churn_Modelling.csv')
df.head()
df.info()

## assigning X and y
X = df.iloc[:, 3:-1]
y = df.iloc[:, -1:]


## handling categorical data
## LabelEncoder and OneHotEncoder
labelencoder1 = LabelEncoder()
X.iloc[:, 1] = labelencoder1.fit_transform(X.iloc[:, 1])

labelencoder2 = LabelEncoder()
X.iloc[:, 2] = labelencoder2.fit_transform(X.iloc[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

# spitting train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 42)

# scaling input data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# building neural network

import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim = 11))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'Adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy']
              )

model.fit(X_train, y_train, batch_size = 10,
          epochs = 20)

y_pred = model.predict(X_test)

threshold = 0.5
y_pred= (y_pred >= threshold).astype('int')


cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True)

## K-fold Cross Validation with Keras

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_model(optimizer = 'Adam'):
    
    model = Sequential()
    model.add(Dense(100, activation = 'relu', init = 'uniform', input_dim = 11))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

classifier= KerasClassifier(build_model, batch_size = 10, epochs = 20)
accuracies = cross_val_score(classifier, X_train, y_train,  cv = 5, n_jobs = -1)

mean_acc = accuracies.mean()
var_acc = accuracies.std()
print('Acc: {}   Std of Acc: {}'.format(mean_acc, var_acc))

# Droput Regularization to fight against overfitting
## model.add(Dropout(p = 0.1)) after each Dense layer

## grid search with Keras

from sklearn.model_selection import GridSearchCV

param_grid = {'batch_size' :[5, 10],
              'epochs' : [5, 10],
              'optimizer' : ['adam', 'rmsprop']
              }

gridsearch = GridSearchCV(classifier, param_grid, cv = 3, scoring = 'accuracy')
gridsearch.fit(X_train, y_train)

best_model = gridsearch.best_estimator_
best_params = gridsearch.best_params_
best_acc = gridsearch.best_score_




























                          
                          
   