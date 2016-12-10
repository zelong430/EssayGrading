import cPickle
import numpy as np
import re
from keras.layers import Input, LSTM, Dense, merge
from keras.models import Model, Sequential
from keras.layers.core import Activation
from keras.models import load_model

def clean_data(text, keep_period = False):
    text = text.lower()

    NER_pat = re.compile("(@[a-z]+)[1-9]+")
    text = NER_pat.sub('\\1', text)

    NUM_pat = re.compile("(\d+)\S*")
    text = NUM_pat.sub('@number', text)

    if keep_period:
        text = re.sub('[^a-zA-Z0-9.@]', ' ', text)
    else:
        text = re.sub('[^a-zA-Z0-9@]', ' ', text)
    return text


maxlen = 30
score_model = load_model('score_model.h5')
# model = load_model('bow_lstm.h5')
# test_data = pickle.load(open("test.pkl", 'rb'))
# result = []
# Xleft, Xright = [], []
# for setid in range(1, 9, 1):
#     cur_set = test_data[setid]
#     X = []
#     for sample in cur_set:
#         if sample[0].shape[0] < maxlen:
#             X.append(np.vstack([sample[0], np.zeros((maxlen - sample[0].shape[0], sample[0].shape[1]))]))
#         else:
#             X.append(sample[0][:maxlen])
#     for i in range(len(cur_set)):
#         for j in range(i+1, len(cur_set)):
#             if cur_set[i][1] < cur_set[j][1]:
#                 Xleft.append(X[j])
#                 Xright.append(X[i])
#             elif cur_set[i][1] > cur_set[j][1]:
#                 Xleft.append(X[i])
#                 Xright.append(X[j])
#         # result.append([setid, sample[2], 0])
# # for i in range(len(X)):
# #     for j in range(i+1, len(X)):
# #         if result[i][0] == result[j][0]:
#
#
#
# # X = np.reshape(X, (len(X), X[0].shape[0], X[0].shape[1]))
# # print len(Xleft), Xleft[1].shape
# Xleft = np.reshape(Xleft, (len(Xleft), Xleft[0].shape[0], Xleft[0].shape[1]))
# Xright = np.reshape(Xright, (len(Xright), Xright[0].shape[0], Xright[0].shape[1]))



# y = score_model.predict(Xleft)
# for tmp in y: print tmp[0]
#
# y = model.predict([Xleft, Xright])
# cnt = 0
# right = 0
# for tmp in y:
#     print tmp[0]
#     if tmp[0] > 0.5:
#         right += 1
#     cnt += 1
# print right, cnt, right*1.0/cnt


with open('./word2emb.pkl', 'rb') as f:
    word2emb = cPickle.load(f)
embsize = 300

with open("../processed_data/test.tsv") as f:
    '''
    Schema:
    essay_id essay_set essay rater1_domain1 rater2_domain1 domain1_score
    '''
    with open("result.csv", 'w') as g:
        skip = 1
        for line in f.readlines():
            if skip == 1:
                skip = 0
                continue
            fields = line.rstrip().split('\t')
            sentence_list = clean_data(fields[2], True).split('.')
            cur_doc = []
            for sentence in sentence_list:
                sentence_emb = np.zeros(embsize)
                store = 0
                for word in sentence.split():
                    try:
                        sentence_emb += word2emb[word]
                        store = 1
                    except:
                        pass
                if store == 1:
                    cur_doc.append(sentence_emb)


            X = np.reshape(cur_doc, (len(cur_doc), embsize))
            if X.shape[0] < maxlen:
                X = np.vstack([X, np.zeros((maxlen - X.shape[0], X.shape[1]))])
            else:
                X = X[:maxlen, :]
            y = score_model.predict(np.reshape(X, (1, maxlen, embsize)))

            '''
                1. Set ID of essay
                2. Essay ID
                3. Predicted score
                4. Actual score
                delimited by ',' comma.
            '''
            g.write(fields[1]+','+fields[0]+','+str(y[0][0])+','+fields[-1]+'\n')
