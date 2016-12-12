from keras.models import Model, Sequential
#from keras.layers.containers import Graph
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Embedding, Reshape, Activation, Merge
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D


def model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter, pooling, embed_dim):
    #Define what the input shape looks like
    inputs = Input(shape=([maxlen]), name='input', dtype='float32')

    # Embedding layer
    embed = Embedding(vocab_size, embed_dim)(inputs)

    #All the convolutional layers...
    conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size))(embed)
    conv = MaxPooling1D(pool_length=pooling)(conv)

    conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                          border_mode='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_length=3)(conv1)

    conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                          border_mode='valid', activation='relu')(conv1)

    conv3 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
                          border_mode='valid', activation='relu')(conv2)

    conv4 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],
                          border_mode='valid', activation='relu')(conv3)

    conv5 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5],
                          border_mode='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_length=3)(conv5)
    conv5 = Flatten()(conv5)

    #Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    #Output dense layer with softmax activation
    #pred = Dense(1, activation='linear', name='output')(z)
    pred = Dense(1, activation = 'sigmoid', name = 'output')(z)

    model = Model(input=inputs, output=pred)

    sgd = SGD(lr=0.01, momentum=0.9)
    #sgd = Adam(lr=0.01)
    #model.compile(loss='mean_squared_error', optimizer=sgd,
    #              metrics=['accuracy'])
    model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

    return model

    return model