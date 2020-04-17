
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout




def LSTM_model(samples,num_feature):
  model = Sequential()
  model.add(LSTM(256,activation='relu', return_sequences=True, input_shape=(samples,num_feature)))
  model.add(LSTM(64, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(Y.shape[1], activation='softmax'))
  model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

  return model_instacart