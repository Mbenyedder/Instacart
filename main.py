import pandas as pd
import numpy as np
from data import *
from Preprocessing import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix
import keras
from keras.layers import Dropout
from math import sqrt
from model import LSTM_model


T=4 #(timestep) On commence avec 4 achats car Il y a un achat à chaque fois et donc pas de 0

for i in range(len(L)):
  panier =pad_sequences(L[i], maxlen=T, dtype="int32", padding="post", truncating='pre', value=0.0)
  print(panier)



X1=np.zeros((50000,3,5)) #50000 = le nombre de client ; 3 = le nombre de panier ; 5 = le nombre d'achat /catégories

for i in range(len(L)):
  panier =pad_sequences(L[i], maxlen=5, dtype="int32", padding="post", truncating='pre', value=0.0)
  panier = panier [:3]
  X1[i]=panier


print(panier)
print(X1)


X = X1[:,:-1,:]
Y = X1[:,-1,:]

#print(X.shape)
#print(Y.shape)


#Train test split to train and test model
x_train, x_test, y_train, y_test = train_test_split(X, Y)


#model
model_instacart=LSTM_model(X.shape[1], X.shape[2])

model_instacart.summary()

callback_1 = keras.callbacks.TensorBoard(log_dir='trainings/t2')
callback_2 = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=5)
callback_3 = keras.callbacks.ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)

model_instacart.fit(x_train, y_train, epochs=50,batch_size=256,verbose=1, callbacks=[callback_1, callback_2, callback_3])


# Accuracy
accuracy = model_instacart.evaluate(x_test,y_test)
print(' Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accuracy[0],accuracy[1]))


