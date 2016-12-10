import pickle
import numpy as np
from keras.layers import Input, LSTM, Dense, merge
from keras.models import Model, Sequential
from keras.layers.core import Activation
from keras.models import load_model

maxlen = 30

# xl = Input(shape=(20, 200))
# shared_lstm = LSTM(output_dim=64, return_sequences=False)
# scorefun = Dense(output_dim=1, activation='sigmoid')
# hidl = shared_lstm(xl)
# scorel = scorefun(hidl)
# model = Model(input=xl, output=scorel)
# model.load_weights('bow_lstm.h5')

score_model = load_model('score_model.h5')

model = load_model('bow_lstm.h5')
test_data = pickle.load(open("test.pkl", 'rb'))

# X = []
result = []

Xleft, Xright = [], []

for setid in range(1, 9, 1):
    cur_set = test_data[setid]
    X = []
    for sample in cur_set:
        if sample[0].shape[0] < maxlen:
            X.append(np.vstack([sample[0], np.zeros((maxlen - sample[0].shape[0], sample[0].shape[1]))]))
        else:
            X.append(sample[0][:maxlen])
    for i in range(len(cur_set)):
        for j in range(i+1, len(cur_set)):
            if cur_set[i][1] < cur_set[j][1]:
                Xleft.append(X[j])
                Xright.append(X[i])
            elif cur_set[i][1] > cur_set[j][1]:
                Xleft.append(X[i])
                Xright.append(X[j])
        # result.append([setid, sample[2], 0])


# for i in range(len(X)):
#     for j in range(i+1, len(X)):
#         if result[i][0] == result[j][0]:



# X = np.reshape(X, (len(X), X[0].shape[0], X[0].shape[1]))
# print len(Xleft), Xleft[1].shape
Xleft = np.reshape(Xleft, (len(Xleft), Xleft[0].shape[0], Xleft[0].shape[1]))
Xright = np.reshape(Xright, (len(Xright), Xright[0].shape[0], Xright[0].shape[1]))



# y = score_model.predict(Xleft)
# for tmp in y: print tmp[0]

y = model.predict([Xleft, Xright])
cnt = 0
right = 0
for tmp in y:
    print tmp[0]
    if tmp[0] > 0.5:
        right += 1
    cnt += 1
print right, cnt, right*1.0/cnt