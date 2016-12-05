import pickle
import numpy as np
from keras.layers import Input, LSTM, Dense, merge
from keras.models import Model
from keras.layers.core import Activation
from keras.models import load_model

maxlen = 20

xl = Input(shape=(20, 200))
shared_lstm = LSTM(1)
scorel = shared_lstm(xl)
model = Model(input=xl, output=scorel)
model.load_weights('bow_lstm.h5')

# model = load_model('bow_lstm.h5')
test_data = pickle.load(open("test.pkl", 'rb'))

X = []
result = []
for setid in range(1, 9, 1):
    cur_set = test_data[setid]
    for sample in cur_set:
        if sample[0].shape[0] < maxlen:
            X.append(np.vstack([sample[0], np.zeros((maxlen - sample[0].shape[0], sample[0].shape[1]))]))
        else:
            X.append(sample[0][:maxlen])
        result.append([setid, sample[1], 0])

X = np.reshape(X, (len(X), X[0].shape[0], X[0].shape[1]))
y = model.predict(X)
for i in range(len(result)):
    result[i][2] = y[i][0]
for r in result: print r