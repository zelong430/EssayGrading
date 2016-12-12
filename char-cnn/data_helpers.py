import string
import numpy as np
import pandas as pd
import csv
import cPickle
from keras.utils.np_utils import to_categorical

def prcnt_to_score(prcnt, sidx):
    if sidx == 1:
        return int(round(prcnt * 10 + 2))
    elif sidx == 2:
        return int(round(prcnt * 5 + 1))
    elif sidx in [3,4]:
        return int(round(prcnt * 3))
    elif sidx in [5,6]:
        return int(round(prcnt * 4))
    elif sidx == 7:
        return int(round(prcnt * 22 + 2))
    else:
        return int(round(prcnt * 50 + 10))

def load_data_6():
    train_reader = csv.reader(open('../processed_data/train_char6.tsv', 'r'), delimiter = ' ')

    x_train = []
    y_train = []
    i = 0
    for line in train_reader:
        y_train.append(int(line[0]))
        x_train.append(string.join(line[1:]))

    x_train = np.array(x_train)
    y_train = to_categorical(y_train)

    test_reader = csv.reader(open('../processed_data/test_char6.tsv', 'r'), delimiter = ' ')

    x_test = []
    y_test = []
    for line in test_reader:
        y_test.append(int(line[0]))
        x_test.append(string.join(line[1:]))

    x_test = np.array(x_test)
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)

def load_data_1():
    train_reader = csv.reader(open('../processed_data/train_char1.tsv', 'r'), delimiter = ' ')

    x_train = []
    y_train = []
    i = 0
    for line in train_reader:
        y_train.append(int(line[0]) - 2)
        x_train.append(string.join(line[1:]))

    x_train = np.array(x_train)
    y_train = to_categorical(y_train)

    test_reader = csv.reader(open('../processed_data/test_char1.tsv', 'r'), delimiter = ' ')

    x_test = []
    y_test = []
    for line in test_reader:
        y_test.append(int(line[0]) - 2)
        x_test.append(string.join(line[1:]))

    x_test = np.array(x_test)
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)

def load_data_reg():
    train = cPickle.load(open("../processed_data/train_char_reg.p","r"))
    x_train = np.array(train["essays"])
    y_train = np.reshape(np.array(train["scores"], dtype = float), (-1, 1))
    set_train = np.reshape(np.array(train["sets"], dtype = int), (-1, 1))

    test = cPickle.load(open("../processed_data/test_char_reg.p","r"))
    x_test = np.array(test["essays"])
    y_test = np.reshape(np.array(test["scores"], dtype = float), (-1, 1))
    set_test = np.reshape(np.array(test["sets"], dtype = int), (-1, 1))

    return (x_train, y_train, set_train), (x_test, y_test, set_test)



def mini_batch_generator(x, y, vocab, vocab_size, vocab_check, maxlen,
                         batch_size=128):

    for i in xrange(0, len(x), batch_size):
        x_sample = x[i:i + batch_size]
        y_sample = y[i:i + batch_size]

        input_data = encode_data(x_sample, maxlen, vocab, vocab_size,
                                 vocab_check)

        yield (input_data, y_sample)

def mini_batch_generator_reg(x, y, s, vocab, vocab_size, vocab_check, maxlen,
                         batch_size=128):

    for i in xrange(0, len(x), batch_size):
        x_sample = x[i:i + batch_size]
        y_sample = y[i:i + batch_size]
        s_sample = s[i:i + batch_size]

        input_data = encode_data(x_sample, maxlen, vocab, vocab_size,
                                 vocab_check)

        yield (input_data, y_sample, s_sample)

def mini_batch_generator_cnn(x, y, s, vocab, vocab_size, vocab_check, maxlen,
                         batch_size=128):

    for i in xrange(0, len(x), batch_size):
        x_sample = x[i:i + batch_size]
        y_sample = y[i:i + batch_size]
        s_sample = s[i:i + batch_size]

        input_data = encode_data_idx(x_sample, maxlen, vocab, vocab_size, vocab_check)

        yield (input_data, y_sample, s_sample)

def encode_data(x, maxlen, vocab, vocab_size, check):
    #Iterate over the loaded data and create a matrix of size maxlen x vocabsize
    #In this case that will be 1014x69. This is then placed in a 3D matrix of size
    #data_samples x maxlen x vocab_size. Each character is encoded into a one-hot
    #array. Chars not in the vocab are encoded into an all zero vector.

    input_data = np.zeros((len(x), maxlen, vocab_size))
    for dix, sent in enumerate(x):
        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))
        chars = list(sent.lower().replace(' ', ''))
        for c in chars:
            if counter >= maxlen:
                pass
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        input_data[dix, :, :] = sent_array

    return input_data

def encode_data_idx(x, maxlen, vocab, vocab_size, check):
    #Iterate over the loaded data and create a matrix of size maxlen x vocabsize
    #In this case that will be 1014x69. This is then placed in a 3D matrix of size
    #data_samples x maxlen x vocab_size. Each character is encoded into a one-hot
    #array. Chars not in the vocab are encoded into an all zero vector.

    input_data = np.zeros((len(x), maxlen))
    for dix, sent in enumerate(x):
        counter = 0
        char_idxs = np.zeros(maxlen)
        for c in list(sent.lower().replace(' ', '')):
            if counter >= maxlen:
                pass
            else:
                char_idxs[counter] = vocab[c]
        input_data[dix, :] = char_idxs

    return input_data


def shuffle_matrix(x, y):
    stacked = np.hstack((np.matrix(x).T, y))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    yi = np.array(stacked[:, 1:])

    return xi, yi

def shuffle_matrix_reg(x, y, s):
    stacked = np.hstack((np.matrix(x).T, s, y))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    seti = np.array(stacked[:, 1])
    yi = np.array(stacked[:, 2:])

    return xi, yi, seti


def create_vocab_set():
    #This alphabet is 69 chars vs. 70 reported in the paper since they include two
    # '-' characters. See https://github.com/zhangxiangxiao/Crepe#issues.

    alphabet = (list(string.ascii_lowercase) + list(string.digits) +
                list(string.punctuation) + ['\n'])
    vocab_size = len(alphabet)
    check = set(alphabet)

    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, check
