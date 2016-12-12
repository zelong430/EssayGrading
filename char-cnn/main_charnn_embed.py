'''
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py
'''

from __future__ import print_function
from __future__ import division
import json
import py_crepe_charnn_embed
import datetime
import numpy as np
import data_helpers
import string
import sys
import os
import csv
np.random.seed(0123)  # for reproducibility

# set parameters:
report_every = 10
subset = None

#Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = 2048

#Model params
embed_dim = 30
nb_filter = 64 # Number of filters for conv layers
dense_outputs = 128 #300 # Number of units in the dense layer
#filter_kernels = [10, 7, 3, 3, 3, 3]
filter_kernels = [64,64,32,16,4,4] #Conv layer kernel size
pooling = 6 # pooling over

#Whether to save model parameters
save = True
load = True
model_name_path = 'save/model' + string.join([str(f) for f in filter_kernels], '-') + '-emb.json'
model_weights_path = 'save/model_weights' + string.join([str(f) for f in filter_kernels], '-') + '-emb.h5'

#Compile/fit params
batch_size = 80
nb_epoch = 20

print('Loading data...')
#Expect x to be a list of sentences. Y to be a one-hot encoding of the
#categories.
(xt, yt, sett), (x_test, y_test, set_test) = data_helpers.load_data_reg()
print('Data length: {}'.format(len(xt)))
print('Batches: {}'.format(len(xt) / batch_size))

print('Creating vocab...')
vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()

test_data = data_helpers.encode_data_idx(x_test, maxlen, vocab, vocab_size, check)

print('Build model...')

model = py_crepe_charnn_embed.model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter, pooling, embed_dim)
if load and os.path.isfile(model_weights_path):
    model.load_weights(model_weights_path)

print('Fit model...')
initial = datetime.datetime.now()
e = 0
STOP = False
while not STOP:
    e += 1
    xi, yi, seti = data_helpers.shuffle_matrix_reg(xt, yt, sett)
    xi_test, yi_test, seti_test = data_helpers.shuffle_matrix_reg(x_test, y_test, set_test)
    if subset:
        batches = data_helpers.mini_batch_generator_cnn(xi[:subset], yi[:subset], seti[:subset],
                                                        vocab, vocab_size, check,
                                                        maxlen,
                                                        batch_size=batch_size)
    else:
        batches = data_helpers.mini_batch_generator_cnn(xi, yi, seti, vocab, vocab_size,
                                                        check, maxlen,
                                                        batch_size=batch_size)

    test_batches = data_helpers.mini_batch_generator_cnn(xi_test, yi_test, seti_test, vocab,
                                                        vocab_size, check, maxlen,
                                                        batch_size=batch_size)

    accuracy = 0.0
    loss = 0.0
    step = 1
    start = datetime.datetime.now()
    print('Epoch: {}'.format(e))
    for x_train, y_train, s_train in batches:
        print(step)
        f = model.train_on_batch(x_train, y_train)
        loss += f[0]
        loss_avg = loss / step
        for i, pred in enumerate(model.predict_on_batch(x_train).flatten()):
            #print(data_helpers.prcnt_to_score(pred, int(s_train[i,0])), data_helpers.prcnt_to_score(float(y_train[i,0]), int(s_train[i,0])))
            if data_helpers.prcnt_to_score(pred, int(s_train[i,0])) == data_helpers.prcnt_to_score(float(y_train[i,0]), int(s_train[i,0])):
                accuracy += 1
        accuracy_avg = accuracy / (batch_size * step)
        if step % report_every == 0:
            print('  Step: {}'.format(step))
            print('\tLoss: {}. Accuracy: {}'.format(loss_avg, accuracy_avg))
        step += 1

    test_accuracy = 0.0
    test_loss = 0.0
    test_step = 1
    
    for x_test_batch, y_test_batch, s_test_batch in test_batches:
        f_ev = model.test_on_batch(x_test_batch, y_test_batch)
        test_loss += f_ev[0]
        test_loss_avg = test_loss / test_step
        for i, pred in enumerate(model.predict_on_batch(x_test_batch).flatten()):
            #print(data_helpers.prcnt_to_score(pred, int(s_train[i,0])), data_helpers.prcnt_to_score(float(y_train[i,0]), int(s_train[i,0])))
            if data_helpers.prcnt_to_score(pred, int(s_test_batch[i,0])) == data_helpers.prcnt_to_score(float(y_test_batch[i,0]), int(s_test_batch[i,0])):
                test_accuracy += 1
        test_accuracy_avg = test_accuracy / (batch_size * test_step)
        test_step += 1
    stop = datetime.datetime.now()
    e_elap = stop - start
    t_elap = stop - initial
    print('Epoch {}. Loss: {}. Accuracy: {}\nEpoch time: {}. Total time: {}\n'.format(e, test_loss_avg, test_accuracy_avg, e_elap, t_elap))

    if e >= nb_epoch:# or abs(test_accuracy_avg - curr_accuracy) < 0.01:
        STOP = True

    if save and STOP:
        # Predictions
        preds = model.predict_on_batch(test_data).flatten()
        with open("pred-" + string.join([str(f) for f in filter_kernels], '-') + "-cnn-emb.csv", "w") as f:
            writer = csv.writer(f, delimiter = ",")
            curr_set_id = 0
            for p, y, s in zip(preds, y_test, set_test):
                if s[0] != curr_set_id :
                    es_idx = 1
                    curr_set_id = s[0]
                p = data_helpers.prcnt_to_score(p, s[0])
                y = data_helpers.prcnt_to_score(y[0], s[0])
                writer.writerow([s[0], es_idx, p, y])
                es_idx += 1

        # Accuracy
        with open("test-acc-" + string.join([str(f) for f in filter_kernels], '-') + "-cnn-emb.txt", "w") as f:
            f.write(str(accuracy_avg) + "\n")
            f.write(str(test_accuracy_avg))

if save:
    print('Saving model params...')
    json_string = model.to_json()
    with open(model_name_path, 'w') as f:
        json.dump(json_string, f)

    model.save_weights(model_weights_path)
