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
from matplotlib import pyplot as plt
import pydot
import graphviz

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



train_y = to_categorical(np.hstack((labels_train, labels_test, labels_val)))[0:len(labels_train),:]
test_y = to_categorical(np.hstack((labels_train, labels_test, labels_val)))[len(labels_train):len(labels_train)+len(labels_test),:]
val_y = to_categorical(np.hstack((labels_train, labels_test, labels_val)))[len(labels_train)+len(labels_test):,:]

# train_x = np.hstack((train_x, to_categorical(train["essay_set"].values)))
# test_x = np.hstack((test_x, to_categorical(test["essay_set"].values)))
# val_x = np.hstack((val_x, to_categorical(val["essay_set"].values)))
# model constuction
model = Sequential()
model.add(Dense(300, input_dim = 300, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(300, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(300, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(61, init='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# fitting and prediction
# call back functions
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience=5, min_lr=0.0001)
history = LossHistory()
csvlogger = keras.callbacks.CSVLogger('ml_cls_log.csv')
early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20)

model.fit(train_x, train_y, callbacks=[reduce_lr, history, csvlogger, early_stopper], \
        shuffle = True, validation_data = (val_x, val_y) ,batch_size = 64, nb_epoch = 200 )


train_result =model.predict_classes(train_x)
val_result = model.predict_classes(val_x)
test_result = model.predict_classes(test_x)

generate_output(train, train_result, 'ml_cls_train.csv')
generate_output(val, val_result, 'ml_cls_val.csv')
generate_output(test, test_result, 'ml_cls_test.csv')

# plot the loss of train&val history
plt.plot(history.val_losses, label = 'val_loss')
plt.plot(history.losses, label = 'train_loss')
plt.xlabel('num of epoch')
plt.ylabel('crossentropy loss')
plt.title('Crossentropy Loss of train&val using Multilayer Classification')
plt.legend()
plt.savefig('ml_cls_plot.png')
# print the model layout
#from keras.utils.visualize_util import plot
#plot(model, to_file='ml_cls_model.png')





