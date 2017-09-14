
import pandas as pd
df = pd.read_csv('')
data = df[df['isfraud'] != -1]
idx = [0,2,3,4,6,7,8,11,12,21,22,23,24,25]
train = data.drop(data.columns[idx], axis=1)
info=data.iloc[:,idx]

###
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.utils.layer_utils import print_layer_shapes


#sine and cos wave
import numpy as np


X = np.linspace(0,1000,10000)
Y = np.asarray([np.sin(X),np.cos(X)]).T


# data prep
# use 500 data points of historical data to predict 500 data points in the future
data = Y
examples = 500
y_examples = 500

nb_samples = len(data) - examples - y_examples


# input - 2 features
input_list = [np.expand_dims(np.atleast_2d(data[i:examples+i,:]), axis=0) for i in xrange(nb_samples)]
input_mat = np.concatenate(input_list, axis=0)


# target - the first column in merged dataframe
target_list = [np.atleast_2d(data[i+examples:examples+i+y_examples,0]) for i in xrange(nb_samples)]
target_mat = np.concatenate(target_list, axis=0)


# set up model
trials = input_mat.shape[0]
features = input_mat.shape[2]
print trials
print features
hidden = 64
model = Sequential()

model.add(LSTM(input_dim=features, output_dim=hidden))
model.add(Dropout(.2))
model.add(Dense(input_dim=hidden, output_dim=y_examples))

model.add(Activation('linear'))
model.compile(loss='mse', optimizer='rmsprop')


# Train

model.fit(input_mat, target_mat, nb_epoch=2)
print_layer_shapes(model, input_shapes =(input_mat.shape))

############3

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Input sequence
wholeSequence = [[0,0,0,0,0,0,0,0,0,2,1],
                 [0,0,0,0,0,0,0,0,2,1,0],
                 [0,0,0,0,0,0,0,2,1,0,0],
                 [0,0,0,0,0,0,2,1,0,0,0],
                 [0,0,0,0,0,2,1,0,0,0,0],
                 [0,0,0,0,2,1,0,0,0,0,0],
                 [0,0,0,2,1,0,0,0,0,0,0],
                 [0,0,2,1,0,0,0,0,0,0,0],
                 [0,2,1,0,0,0,0,0,0,0,0],
                 [2,1,0,0,0,0,0,0,0,0,0]]

# Preprocess Data:
wholeSequence = np.array(wholeSequence, dtype=float) # Convert to NP array.
data = wholeSequence[:-1] # all but last
target = wholeSequence[1:] # all but first

# Reshape training data for Keras LSTM model
# The training data needs to be (batchIndex, timeStepIndex, dimentionIndex)
# Single batch, 9 time steps, 11 dimentions
data = data.reshape((1, 9, 11))
target = target.reshape((1, 9, 11))

# Build Model
model = Sequential()  
model.add(LSTM(11, input_shape=(9, 11), unroll=True, return_sequences=True))
model.add(Dense(11))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(data, target, nb_epoch=2000, batch_size=1, verbose=2)

