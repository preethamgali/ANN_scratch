from data_prep import data, output
import numpy as np
import time

data = np.asarray(data)

output = np.asarray(output)
print(output)
input_size = 8
n_h1 = 10
n_h2 = 10
n_o = 2
epoch = 10000
learning_rate = 0.01

from keras.models import Sequential
from keras.layers.core import Dense, Activation
import tensorflow as tf
from keras.utils import to_categorical


full = Sequential()
full.add(Dense(n_h1,input_dim = input_size))
full.add(Activation("relu"))
full.add(Dense(n_h2))
full.add(Activation("relu"))
full.add(Dense(n_o))

full.compile(loss='mean_squared_error', optimizer='sgd')
# full.summary()
full.fit(data,output,epochs = 1000,batch_size=100)

print(full.evaluate(data,output))
# dont know if it works

