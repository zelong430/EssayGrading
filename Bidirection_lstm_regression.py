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
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD
import evaluate
from keras.regularizers import l2, activity_l2
from w2v import w2v_Model 
from matplotlib import pyplot as plt
from nltk.corpus import stopwords

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

def generate_output(data_df, prediction, name):
	data_df["prediction"] = prediction
	data_df = data_df[["essay_set", 'essay_id', 'prediction', 'domain1_score']]
	data_df.to_csv(name, index = False, header = False)

def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks

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

MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 400
EMBEDDING_DIM = 300
w2v = w2v_Model('glove/glove.6B.300d.txt', EMBEDDING_DIM)

train = pd.read_csv('processed_data/train.tsv',delimiter='\t')
val = pd.read_csv('processed_data/val.tsv',delimiter='\t')
test = pd.read_csv('processed_data/test.tsv',delimiter='\t')

texts_train = [clean_data(i).replace('@', '') for i in train["essay"].values]
texts_val = [clean_data(i).replace('@', '') for i in val["essay"].values]
texts_test = [clean_data(i).replace('@', '') for i in test["essay"].values]

labels_train = train["domain1_score"].values
labels_val = val["domain1_score"].values
labels_test = test["domain1_score"].values

# tokenize text string
tokenizer = keras.preprocessing.text.Tokenizer(nb_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_train + texts_test)
word_index = tokenizer.word_index
inv_map = {v: k for k, v in word_index.iteritems()}
print('Found %s unique tokens.' % len(word_index))

train_x = tokenizer.texts_to_sequences(texts_train)
test_x = tokenizer.texts_to_sequences(texts_test)
val_x = tokenizer.texts_to_sequences(texts_val)

# pad text string to same length of MAX_SEQUENCE_LENGTH
train_x = pad_sequences(train_x, padding = 'post',maxlen=MAX_SEQUENCE_LENGTH)
test_x = pad_sequences(test_x, padding = 'post', maxlen=MAX_SEQUENCE_LENGTH)
val_x = pad_sequences(val_x, padding = 'post', maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', train_x.shape)

# create embedding matrix
# embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     embedding_vector = w2v[word]
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector


embedding_matrix = np.zeros((MAX_NB_WORDS + 1, EMBEDDING_DIM))
for i in range(1,MAX_NB_WORDS + 1):
    word = inv_map[i]
    embedding_vector = w2v[word]
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print "embedding shape", embedding_matrix.shape

train_y = labels_train.astype('float32')
val_y = labels_val.astype('float32')
test_y = labels_test.astype('float32')

# train_x = np.hstack((train_x, to_categorical(train["essay_set"].values)))
# test_x = np.hstack((test_x, to_categorical(test["essay_set"].values)))
# val_x = np.hstack((test_x, to_categorical(test["essay_set"].values)))
# model constuction
model = Sequential()

model_left = Sequential()

model_left.add(Embedding(MAX_NB_WORDS + 1, EMBEDDING_DIM, \
                    weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable = True))

model_left.add(LSTM(64, return_sequences=True))
#, return_sequences=True
model_right = Sequential()
model_right.add(Embedding(MAX_NB_WORDS + 1, EMBEDDING_DIM, \
                    weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable = True))

model_right.add(LSTM(64, go_backwards=True, return_sequences=True))
model.add( Merge([model_left, model_right], mode='concat'))
model.add(Dropout(0.3))

left, right = fork(model)

left.add(LSTM(64))
right.add(LSTM(64,  go_backwards=True))

model = Sequential()
model.add(Merge([left, right], mode='concat'))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mean_absolute_error'])



# fitting and prediction
# call back functions
#reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience=2, min_lr=0.0001)
history = LossHistory()
csvlogger = keras.callbacks.CSVLogger('bi_lstm_rgs_log.csv')
early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10)


model.fit([train_x, train_x], train_y,callbacks=[history, csvlogger, early_stopper],\
            validation_data = ([val_x, val_x], val_y), batch_size = 128, nb_epoch= 100)

train_result =model.predict([train_x, train_x])
val_result = model.predict([val_x, val_x])
test_result = model.predict([test_x, test_x])


generate_output(train, train_result, 'Bidirectional_lstm_rgs_train.csv')
generate_output(val, val_result, 'Bidirectional_lstm_rgs_val.csv')
generate_output(test, test_result, 'Bidirectional_lstm_rgs_test.csv')

# plot the loss of train&val history
plt.plot(history.val_losses,'--', label = 'val_loss')
plt.plot(history.losses, label = 'train_loss')
plt.xlabel('num of epoch')
plt.ylabel('crossentropy loss')
plt.title('Crossentropy Loss of train&val using Bidirectional LSTM Regression Model')
plt.savefig('bi_lstm_rgs_plot.png')







