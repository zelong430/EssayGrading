import cPickle
import numpy as np
from keras.layers import Input, LSTM, Dense, merge, GRU
from keras.models import Model, Sequential
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.core import Flatten
from keras.layers import Merge

max_len = 30
num_epoch = 5
train_sample_num = 10000
val_sample_num = 2000
model_type = 'LSTM'  # 'CNN' or 'LSTM'

val_data = cPickle.load(open("val.pkl", 'rb'))
train_data = cPickle.load(open("train.pkl", 'rb'))


def make_pair(X, maxlen=30, maxsample=3000):
    X = sorted(X, key=lambda x: x[1], reverse=True)
    Xleft, Xright = [], []
    cnt = 0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if X[i][1] > X[j][1]:
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
    l, r = make_pair(train_data[setid], maxlen=max_len, maxsample=train_sample_num)
    Xleft += l
    Xright += r

    l, r = make_pair(val_data[setid], maxlen=max_len, maxsample=val_sample_num)
    Xvalleft += l
    Xvalright += r

nsample = len(Xleft)
seqlen = Xleft[0].shape[0]
embsize = Xleft[0].shape[1]
Xleft = np.reshape(Xleft, (nsample, seqlen, embsize))
Xright = np.reshape(Xright, (nsample, seqlen, embsize))
Xvalleft = np.reshape(Xvalleft, (len(Xvalleft), Xvalleft[0].shape[0], Xvalleft[0].shape[1]))
Xvalright = np.reshape(Xvalright, (len(Xvalleft), Xvalleft[0].shape[0], Xvalleft[0].shape[1]))
y = np.ones((nsample, 1))
yval = np.ones((len(Xvalleft), 1))

score_model = Sequential()
if model_type == 'CNN':
    score_model.add(Convolution1D(64, 5, border_mode='same', input_shape=(max_len, embsize)))
    score_model.add(AveragePooling1D(pool_length=30, stride=None, border_mode='valid'))
    score_model.add(Flatten())
elif model_type == 'LSTM':
    score_model.add(LSTM(output_dim=64, return_sequences=True, input_shape=(max_len, embsize)))
    score_model.add(LSTM(output_dim=32, go_backwards=True, return_sequences=False, input_shape=(max_len, embsize)))
else:
    raise ValueError('model type is not defined')
score_model.add(Dense(1))

xl = Input(shape=(max_len, embsize))
xr = Input(shape=(max_len, embsize))
scorel = score_model(xl)
scorer = score_model(xr)

# merged = merge([scorel, scorer], mode=lambda x: x[0] - x[1], concat_axis=-1)
merged = merge([scorel, scorer], mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0])

predictions = Dense(1, activation='sigmoid')(merged)

model = Model(input=[xl, xr], output=predictions)
model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([Xleft, Xright], y, nb_epoch=num_epoch, validation_data=([Xvalleft, Xvalright], yval))

score_model.save('score_model.h5')
model.save('bow_lstm.h5')
