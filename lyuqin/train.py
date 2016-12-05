import pickle
import numpy as np
from keras.layers import Input, LSTM, Dense, merge
from keras.models import Model
from keras.layers.core import Activation

wordemb = 32
sentemb = 64
docemb = 128
train_data = pickle.load(open("train.pkl", 'rb'))

def make_pair(X, maxlen=20, maxsample = 10000):
    X = sorted(X, key = lambda x: x[1], reverse=True)
    Xleft, Xright = [], []
    cnt = 0
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if X[i][1] != X[j][1]:
                if X[i][0].shape[0] < maxlen:
                    Xleft.append(np.vstack([X[i][0], np.zeros((maxlen - X[i][0].shape[0], X[i][0].shape[1]))]))
                else:
                    Xleft.append(X[i][0][:maxlen])

                if X[j][0].shape[0] < maxlen:
                    Xright.append(np.vstack([X[j][0], np.zeros((maxlen - X[j][0].shape[0], X[j][0].shape[1]))]))
                else:
                    Xright.append(X[j][0][:maxlen])
                cnt += 1
            if maxsample != -1 and cnt >= maxsample: break
        if maxsample != -1 and cnt >= maxsample: break
    return Xleft, Xright


Xleft, Xright = [], []
Xvalleft, Xvalright = [], []

for setid in range(1, 9, 1):
    l, r = make_pair(train_data[setid])
    Xleft += l
    Xright += r

    # l, r = make_pair(val_data[setid])
    # Xvalleft += l
    # Xright += r

nsample = len(Xleft)
seqlen = Xleft[0].shape[0]
embsize = Xleft[0].shape[1]
Xleft = np.reshape(Xleft, (nsample, seqlen, embsize))
Xright = np.reshape(Xright, (nsample, seqlen, embsize))
y = np.zeros((nsample, 2))
y[:, 0] = 1
# yval = np.zeros((len(Xvalleft), 2))
# yval[:, 0] = 1

xl = Input(shape=(seqlen, embsize))
xr = Input(shape=(seqlen, embsize))
shared_lstm = LSTM(1)
scorel = shared_lstm(xl)
scorer = shared_lstm(xr)

merged_vector = merge([scorel, scorer], mode='concat', concat_axis=-1)
predictions = Activation('softmax')(merged_vector)
# and add a logistic regression on top
# predictions = Dense(1, activation='sigmoid')(merged_vector)

# we define a trainable model linking the
# tweet inputs to the predictions
model = Model(input=[xl, xr], output=predictions)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([Xleft, Xright], y, nb_epoch=2)
# model.fit([Xleft, Xright], y, nb_epoch=10, validation_data=([Xvalleft,Xvalright], yval))

model.save('bow_lstm.h5')
model.save_weights("bow-lstm.w")