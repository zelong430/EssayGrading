# import necessary pac
import pandas as pd
import numpy as np
import re
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Input,Activation,Conv1D,MaxPooling1D,Flatten,Dense,Embedding,LSTM,Merge, Dropout, TimeDistributedDense
from keras.models import Model, Sequential
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD
import evaluate
from keras.regularizers import l2, activity_l2
from w2v import w2v_Model 

def generate_output(data_df, prediction, name):
	data_df["prediction"] = prediction
	data_df = data_df[["essay_set", 'essay_id', 'prediction', 'domain1_score']]
	data_df.to_csv(name, index = False, header = False)

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

# data loading and pre-processing
w2v = w2v_Model('glove/glove.6B.300d.txt', 300)

train = pd.read_csv('processed_data/train.tsv',delimiter='\t')
val = pd.read_csv('processed_data/val.tsv',delimiter='\t')
test = pd.read_csv('processed_data/test.tsv',delimiter='\t')

texts_train = [clean_data(i).replace('@', '') for i in train["essay"].values]
texts_val = [clean_data(i).replace('@', '') for i in val["essay"].values]
texts_test = [clean_data(i).replace('@', '') for i in test["essay"].values]

train_x = np.array(map(lambda essay: w2v[essay.split()].mean(axis = 0), texts_train))
val_x = np.array(map(lambda essay: w2v[essay.split()].mean(axis = 0), texts_val))
test_x = np.array(map(lambda essay: w2v[essay.split()].mean(axis = 0), texts_test))

labels_train = train["domain1_score"].values
labels_val = val["domain1_score"].values
labels_test = test["domain1_score"].values

train_y = labels_train.astype('float32')
val_y = labels_val.astype('float32')
test_y = labels_test.astype('float32')

# model constuction
model = Sequential()
model.add(Dense(300, input_dim = 300, init='normal', activation='relu',\
          W_regularizer=l2(0.0001), bias = True, activity_regularizer=activity_l2(0.0001)))
model.add(Dropout(0.2))
model.add(Dense(300, init='normal', activation='relu',\
          W_regularizer=l2(0.0001), bias = True, activity_regularizer=activity_l2(0.0001)))
model.add(Dense(1, init='normal'))
model.compile(loss='mean_squared_error', metrics = ['mean_absolute_error'], optimizer='adam')

# fitting and prediction
model.fit(train_x, train_y, shuffle = True, validation_data = (val_x, val_y) ,batch_size = 128, nb_epoch = 1000 )
train_result = np.round(model.predict(train_x))
val_result = np.round(model.predict(val_x))
test_result = np.round(model.predict(test_x))


generate_output(train, train_result, 'mlp_train.csv')
generate_output(val, val_result, 'mlp_val.csv')
generate_output(test, test_result, 'mlp_test.csv')







